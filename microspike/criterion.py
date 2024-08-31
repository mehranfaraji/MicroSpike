import numpy as np
from collections import defaultdict
from collections import namedtuple
from typing import Dict, Tuple
from .monitor import Monitor

def get_pattern_repetition_number_and_spike_statistics(monitor: Monitor, position_copypaste: np.ndarray, inference_starting_time: float, inference_ending_time: float, pattern_duration: float, P: int) -> Tuple[Dict[str, int], Dict[str, Dict[int, int]]]:
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

    pattern_repetition_number: Dict[str, int] = {}
    criterion_info: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    pattern_sequence = position_copypaste[inference_starting_index:inference_ending_index]
    start_index = np.where(monitor.spikes_t > inference_starting_time)[0][0]
    spikes_t_i = list(zip(monitor.spikes_t[start_index:], monitor.spikes_i[start_index:]))

    for i in range(P+1):
        num_pattern_repetition = len(np.where(pattern_sequence == i)[0])
        if i == 0:
            pattern_repetition_number['no_pattern'] = num_pattern_repetition
        else:
            pattern_repetition_number[f'pattern_{i}'] = num_pattern_repetition

    for t, i in spikes_t_i:
        idx = int(t / pattern_duration)
        pattern_number = int(position_copypaste[idx])
        if pattern_number != 0:
            criterion_info[f'neuron_{int(i+1)}'][pattern_number] += 1
        else:
            criterion_info[f'neuron_{int(i+1)}']['no_pattern'] += 1

    return pattern_repetition_number, criterion_info


def check_neuron_status(N: int, criterion_info: Dict[str, Dict[int, int]], pattern_repetition_number: Dict[str, int], hit_rate_threshold: float, false_alarm_rate_threshold: float, inference_starting_time: float, inference_ending_time: float) -> Tuple[Dict[str, namedtuple], Dict[str, int]]:
    """
    Evaluate the status of neurons based on their spiking behavior during pattern presentations.

    This function analyzes the spiking activity of neurons during the inference period and determines
    their learned patterns, hit rates, false alarm rates, and overall status. It also keeps track of
    how many neurons have successfully learned each pattern.

    Args:
        N (int): Total number of neurons.
        criterion_info (Dict[str, Dict[int, int]]): Dictionary containing spike counts for each neuron during each pattern presentation.
        pattern_repetition_number (Dict[str, int]): Dictionary containing the number of repetitions for each pattern.
        hit_rate_threshold (float): Minimum acceptable hit rate for a neuron to be considered successful.
        false_alarm_rate_threshold (float): Maximum acceptable false alarm rate for a neuron to be considered successful.
        inference_starting_time (float): Start time of the inference period.
        inference_ending_time (float): End time of the inference period.

    Returns:
        tuple: A tuple containing two dictionaries:
            - neurons_stats (Dict[str, namedtuple]): Maps each neuron to its learned pattern, hit rate, false alarm rate, and status.
            - pattern_stats (Dict[str, int]): Maps each pattern to the number of neurons that successfully learned it.

    The function determines the status of each neuron as follows:
    1. If a neuron doesn't spike or only spikes during 'no_pattern', it's marked as 'dead'.
    2. Otherwise, it calculates the hit rate and false alarm rate for the neuron's most active pattern.
    3. If the hit rate is below the threshold or the false alarm rate is above the threshold, 
       the neuron is marked as 'unsuccessful'.
    4. If both criteria are met, the neuron is marked as 'successful', and the count for that pattern is incremented.

    The NeuronStats namedtuple is used to store information about each neuron's performance.
    """
    
    NeuronStats = namedtuple('NeuronStats', ['learned_pattern', 'hit_rate', 'false_alarm_rate', 'status'])
    neurons_stats: Dict[str, namedtuple] = {}
    pattern_stats: Dict[str, int] = {}
    for neuron_number in range(1, N+1):
        neuron_info = criterion_info[f'neuron_{neuron_number}']
        sorted_patterns = sorted(neuron_info.items(), key=lambda x: x[1], reverse=True)
        learned_pattern, number_of_spikes = sorted_patterns[0] if sorted_patterns else (None, 0)
        if learned_pattern is None or learned_pattern == "no_pattern":
            neurons_stats[f'neuron_{neuron_number}'] = NeuronStats(None, None, None, 'dead')
        else:
            hit_rate = number_of_spikes / pattern_repetition_number[f'pattern_{learned_pattern}']
            false_alarm_number = sum([number_of_spikes for pattern, number_of_spikes in neuron_info.items() if pattern != learned_pattern])
            false_alarm_rate = false_alarm_number / (inference_ending_time - inference_starting_time)
            if hit_rate < hit_rate_threshold or false_alarm_rate > false_alarm_rate_threshold:
                status = "unsuccessful"
            else:
                status = "successful"
                if f'pattern_{learned_pattern}' not in pattern_stats:
                    pattern_stats[f"pattern_{learned_pattern}"] = 1
                else:
                    pattern_stats[f"pattern_{learned_pattern}"] += 1
            neurons_stats[f'neuron_{neuron_number}'] = NeuronStats(learned_pattern, hit_rate, false_alarm_rate, status)
            
    return neurons_stats, pattern_stats

def get_successful_neurons_successful_patterns(pattern_stats: Dict[str, int], neurons_stats: Dict[str, namedtuple], P: int, N: int) -> Tuple[int, int]:
    """
    Calculate the number of successful patterns and successful neurons.

    This function analyzes the pattern statistics and neuron statistics to determine
    the number of patterns that were successfully learned by at least one neuron,
    and the total number of successful neurons.

    Args:
        pattern_stats (Dict[str, int]): A dictionary containing statistics for each pattern.
            The keys are in the format 'pattern_{number}', and the values represent
            the number of neurons that successfully learned that pattern.
        neurons_stats (Dict[str, namedtuple]): A dictionary containing statistics for each neuron.
            The keys are in the format 'neuron_{number}', and the values are
            NeuronStats objects containing information about each neuron's performance.
        P (int): Total number of patterns.
        N (int): Total number of neurons.

    Returns:
        tuple: A tuple containing two integers:
            - num_patterns_learned_by_at_least_one_neuron (int): The number of patterns
              that were successfully learned by at least one neuron.
            - num_successful_neurons (int): The total number of neurons that
              successfully learned any pattern.
    """
    num_patterns_learned_by_at_least_one_neuron = 0
    num_successful_neurons = 0
    for pattern_number in range(1, P+1):
        if f'pattern_{pattern_number}' in pattern_stats:
            num_patterns_learned_by_at_least_one_neuron += 1
    for neuron_number in range(1, N+1):
        if neurons_stats[f'neuron_{neuron_number}'].status == "successful":
            num_successful_neurons += 1

    return num_patterns_learned_by_at_least_one_neuron, num_successful_neurons