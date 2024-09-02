from microspike.criterion import get_pattern_repetition_number_and_spike_statistics, check_neuron_status
from microspike.criterion import get_successful_neurons_successful_patterns
from microspike.utils import generate_data_train_model, save_all_data, load_all_data
import os
import numpy as np
import json
from datetime import datetime

## Hyperparameters
# SRM Hyperparamters
inhibitory_connection=True
threshold = 500
reset = 0
refractory = 1
alpha = 0.25
k = 1
tau_m = 0.010
tau_s = 0.0025
K1 = 2
K2 = 4
window_time= 7 * 0.010 # Maximum time [second] to ignore inputs before it by kernels 
train_mode= True

# Synapse Hyperparameters
approximate = True
w_max = 1
w_min = 0
A_pre = 0.03125
B = 0.85
A_post = - A_pre * B
tau_pre=0.0168
tau_post=0.0337

# Model Examination Hyperparameters
hit_rate_threshold = 0.90
false_alarm_rate_threshold = 1 # Hz
pattern_duration = 0.050 # second

def save_examination_results(folder_name, filename, input_train, model, synapse, monitor, times, indices, position_copypaste, patterns_info, timing_pattern):
    # Create the specified folder if it doesn't exist
    os.makedirs(folder_name, exist_ok=True)

    # Prepare the data to be saved
    data = {
        'SRM_Hyperparameters': {
            'inhibitory_connection': inhibitory_connection,
            'threshold': threshold,
            'reset': reset,
            'refractory': refractory,
            'alpha': alpha,
            'k': k,
            'tau_m': tau_m,
            'tau_s': tau_s,
            'K1': K1,
            'K2': K2,
            'window_time': window_time,
            'train_mode': train_mode
        },
        'Synapse_Hyperparameters': {
            'approximate': approximate,
            'w_max': w_max,
            'w_min': w_min,
            'A_pre': A_pre,
            'B': B,
            'A_post': A_post,
            'tau_pre': tau_pre,
            'tau_post': tau_post
        },
        'Network_Parameters': {
            'P': int(P),
            'N': int(N),
            'M': int(M),
            'time': float(time)
        },
        'Model_Examination_Parameters': {
            'hit_rate_threshold': hit_rate_threshold,
            'false_alarm_rate_threshold': false_alarm_rate_threshold,
            'pattern_duration': pattern_duration,
            'inference_starting_time': inference_starting_time,
            'inference_ending_time': inference_ending_time
        },
        'Results': {
            'pattern_repetition_number': pattern_repetition_number,
            'neurons_stats': neurons_stats,
            'pattern_stats': pattern_stats
        },
        'Timestamp': datetime.now().isoformat()
    }

    # Save the data to a JSON file
    full_path = os.path.join(folder_name, f'{filename}.json')
    with open(full_path, 'w') as f:
        json.dump(data, f, indent=4)

    # Save additional data using save_all_data function
    save_all_data(input_train, model, synapse, monitor, times, indices, position_copypaste, patterns_info, timing_pattern, os.path.join(folder_name, f'{filename}_additional_data.pkl'))

    print(f"Examination results saved to {full_path}")

def read_examination_results(folder_name, filename):
    full_path = os.path.join(folder_name, f'{filename}.json')
    with open(full_path, 'r') as f:
        data = json.load(f)
    
    # Extract variables from the loaded data
    srm_params = data['SRM_Hyperparameters']
    synapse_params = data['Synapse_Hyperparameters']
    network_params = data['Network_Parameters']
    examination_params = data['Model_Examination_Parameters']
    results = data['Results']
    timestamp = data['Timestamp']

    # Return data as a dictionary
    return {
        'srm_params': srm_params,
        'synapse_params': synapse_params,
        'network_params': network_params,
        'examination_params': examination_params,
        'results': results,
        'timestamp': timestamp
    }

def read_examination_results_additional_data(folder_name, filename):
    # Load additional data
    input_train, model, synapse, monitor, times, indices, position_copypaste, patterns_info, timing_pattern = load_all_data(os.path.join(folder_name, f'{filename}_additional_data.pkl'))

    # Return additional data as a dictionary
    return {
        'input_train': input_train,
        'model': model,
        'synapse': synapse,
        'monitor': monitor,
        'times': times,
        'indices': indices,
        'position_copypaste': position_copypaste,
        'patterns_info': patterns_info,
        'timing_pattern': timing_pattern
    }

def update_summary_file(folder_name, filename, simulation_number, num_patterns_learned, num_successful_neurons):
    summary_file_path = os.path.join(folder_name, f'{filename}.txt')
    
    # Check if the file exists, if not create it with the header
    if not os.path.exists(summary_file_path):
        with open(summary_file_path, 'w') as f:
            f.write("simulation number, number of patterns learned by at least one neuron, number of successful neurons\n")
    
    # Append the new data
    with open(summary_file_path, 'a') as f:
        f.write(f"{simulation_number}, {num_patterns_learned}, {num_successful_neurons}\n")

if __name__ == "__main__":
    import argparse
    import shutil
    import os

    parser = argparse.ArgumentParser(description="Run model examination")
    parser.add_argument("--folder_name", type=str, required=True, help="Folder name for saving results")
    parser.add_argument("--filename", type=str, required=True, help="Filename for saving results")
    parser.add_argument("--num_simulation", type=int, required=True, help="Number of simulation repeats")
    parser.add_argument("--P", type=int, required=True, help="Number of patterns")
    parser.add_argument("--N", type=int, required=True, help="Number of neurons")
    parser.add_argument("--M", type=int, default=2000, help="Number of input neurons (default: 2000)")
    parser.add_argument("--time", type=float, required=True, help="Simulation time in milliseconds")
    parser.add_argument("--inference_starting_time", type=float, required=True, help="Start time of inference in seconds")
    parser.add_argument("--inference_ending_time", type=float, required=True, help="End time of inference in seconds")
    args = parser.parse_args()
    folder_name = args.folder_name
    filename = args.filename
    num_simulation = args.num_simulation
    # Network Hyperparameters
    P = args.P
    N = args.N
    M = args.M
    time = args.time
    inference_starting_time = args.inference_starting_time
    inference_ending_time = args.inference_ending_time

    folder_name = folder_name + f"-P{P}-N{N}"
    filename = filename + f"-P{P}-N{N}"
    
    for i in range(1, num_simulation+1):

        current_filename = filename + f"-simulation{i}"
        full_path = os.path.join(folder_name, f'{current_filename}.json')
        
        if os.path.exists(full_path):
            print(f"Simulation {i} already exists. Skipping...")
            continue

        ## Model Examination
        input_train, model, synapse, monitor, times, indices, position_copypaste, patterns_info, timing_pattern = generate_data_train_model(P, N, M, time, w_max, w_min, A_pre, A_post, tau_pre, tau_post, approximate,
                                    threshold, reset, refractory, alpha, k, tau_m, tau_s, K1, K2, window_time,
                                    inhibitory_connection)

        pattern_repetition_number, criterion_info = get_pattern_repetition_number_and_spike_statistics(monitor, position_copypaste, inference_starting_time, inference_ending_time, pattern_duration, P)
        neurons_stats, pattern_stats = check_neuron_status(N, criterion_info, pattern_repetition_number, hit_rate_threshold, false_alarm_rate_threshold, inference_starting_time, inference_ending_time)
        num_patterns_learned_by_at_least_one_neuron, num_successful_neurons = get_successful_neurons_successful_patterns(pattern_stats, neurons_stats, P, N)

        # Save the examination results
        save_examination_results(folder_name, current_filename, input_train, model, synapse, monitor, times, indices, position_copypaste, patterns_info, timing_pattern)

        # Update the summary file
        update_summary_file(folder_name, filename, i, num_patterns_learned_by_at_least_one_neuron, num_successful_neurons)
        
        print(f"-- Simulation {i} completed --")

    # Zip the folder and delete the original
    shutil.make_archive(folder_name, 'zip', folder_name)
    shutil.rmtree(folder_name)
    print(f"Folder {folder_name} has been zipped and deleted.")