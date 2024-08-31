import numpy as np
from collections import defaultdict
from .monitor import Monitor

def get_pattern_repetition_number_and_spike_statistics(monitor: Monitor, position_copypaste: np.ndarray, inference_starting_time: float, inference_ending_time: float, pattern_duration: float, P: int):
    """
    Calculates pattern repetition numbers and spike statistics during the inference period.

    Args:
        monitor (Monitor): The monitor object containing spike data.
        position_copypaste (np.ndarray): Array indicating pattern positions.
        inference_starting_time (float): Start time of the inference period.
        inference_ending_time (float): End time of the inference period.
        pattern_duration (float): Duration of each pattern.
        P (int): Number of patterns (excluding no pattern).

    Returns:
        tuple: A tuple containing two dictionaries:
            - pattern_repetition_number: Count of repetitions for each pattern.
            - criterion_info: Spike counts for each neuron during each pattern.

    This function analyzes the spike data within the specified inference period to determine:
    1. How many times each pattern (including no pattern) is repeated.
    2. How many times each neuron spikes during each pattern presentation.

    The function first determines the relevant indices in the position_copypaste array
    based on the inference time period. It then counts pattern repetitions and records
    spike statistics for each neuron during different pattern presentations.
    """
    inference_starting_index = int(inference_starting_time / pattern_duration)
    inference_ending_index = int(inference_ending_time / pattern_duration)

    pattern_repetition_number = {}
    criterion_info = defaultdict(lambda: defaultdict(int))
    pattern_sequence = position_copypaste[inference_starting_index:inference_ending_index]
    start_index = np.where(monitor.spikes_t > inference_starting_time)[0][0]
    spikes_t_i = list(zip(monitor.spikes_t[start_index:], monitor.spikes_i[start_index:]))

    for i in range(P+1):
        num_pattern_repetition = len(np.where(pattern_sequence == i)[0])
        if i == 0:
            pattern_repetition_number[f'no_pattern'] = num_pattern_repetition
        else:
            pattern_repetition_number[f'pattern_{i}'] = num_pattern_repetition

    for t, i in spikes_t_i:
        idx = int(t / pattern_duration)
        pattern_number = position_copypaste[idx]
        if pattern_number != 0:
            criterion_info[f'neuron_{int(i+1)}'][f'spikes_during_pattern_{pattern_number}'] += 1
        else:
            criterion_info[f'neuron_{int(i+1)}'][f'spikes_during_no_pattern'] += 1

    return pattern_repetition_number, criterion_info


def check_neuron_status(N, criterion_info, pattern_repetition_number, hit_rate_threshold, false_alarm_rate_threshold, inference_starting_time, inference_ending_time):
    neurons_stats = {}
    for neuron_number in range(1, N+1):
        neuron_info = criterion_info[f'neuron_{neuron_number}']
        sorted_patterns = sorted(neuron_info.items(), key=lambda x: x[1], reverse=True)
        learned_pattern, number_of_spikes = sorted_patterns[0] if sorted_patterns else (None, 0)
        if learned_pattern is None or learned_pattern == "spikes_during_no_pattern":
            neurons_stats[f'neuron_{neuron_number}'] = (None, None, 'dead')
        else:
            hit_rate = number_of_spikes / pattern_repetition_number[f'pattern_{learned_pattern[-1]}']
            # sum the number of spikes for during patterns other than the learned pattern
            false_alarm_number = sum([number_of_spikes for pattern, number_of_spikes in neuron_info.items() if pattern != learned_pattern])
            false_alarm_rate = false_alarm_number / (inference_ending_time - inference_starting_time)
            if hit_rate < hit_rate_threshold or false_alarm_rate > false_alarm_rate_threshold:
                status = "unsuccessful"
            else:
                status = "successful"
            neurons_stats[f'neuron_{neuron_number}'] = (hit_rate, false_alarm_rate, status)

    return neurons_stats