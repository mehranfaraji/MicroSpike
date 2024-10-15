from microspike import PatternGenerator, SRMInhibitory, InputTrain, Synapse, Monitor, Network
from microspike.utils import model_config, synapse_config
from collections import defaultdict
import numpy as np
import os
import argparse
import json

from microspike.utils import w_uniform
import matplotlib.pyplot as plt

dt = 0.001


def get_latency_results(N, monitor, position_copypaste, inference_time, patternlength=0.050):
    result = defaultdict(list)
    idx = np.where(monitor.spikes_t > inference_time)[0]
    t = monitor.spikes_t[idx]
    i = monitor.spikes_i[idx]

    for spike_t, spike_i in zip(t, i):
        position_idx = int(spike_t / patternlength)
        pattern_number = position_copypaste[position_idx]
        latency = np.round(spike_t - position_idx * patternlength, decimals=3) * 1000
        if spike_i < N:
            result[f"pattern_{pattern_number}_original_neuron"].append(latency.item())
        else:
            result[f"pattern_{pattern_number}_new_neuron"].append(latency.item())

    return result

def plot_new_pattern_hist(P, result, display:bool, result_dir, image_name):
    fig, axs = plt.subplots(P + 1, 1, figsize=(6, 6))
    for i in range(P + 1):
        axs[i].hist(result[f"pattern_{i+1}_original_neuron"], alpha=0.5, label="Original Neuron")
        axs[i].hist(result[f"pattern_{i+1}_new_neuron"], alpha=0.5, label="New Neuron")
        axs[i].set_title(f"Pattern {i+1}")
        axs[i].set_xlabel("Latency (ms)")
        axs[i].set_ylabel("Frequency")
        axs[i].legend()
        axs[i].grid(True)
        axs[i].set_xlim([0, 50])
        axs[i].set_xticks(np.arange(0, 51, 5))

    plt.tight_layout()
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    plt.savefig(os.path.join(result_dir, image_name))

    if display:
        plt.show()
    
    



def main(P, N, M, num_new_neurons, time, inference_time, display, result_dir, image_name):
    generator = PatternGenerator(number_pattern=P, number_neurons=M,
                                    total_pattern_freq = 1/3,
                                    )
    times, indices, position_copypaste, patterns_info, timing_pattern = generator.generate()

    weight = w_uniform(M=M, N=N)

    input_train = InputTrain(times, indices)

    model = SRMInhibitory(N=N,
                **model_config
                )
    synapse = Synapse(w=weight,
                    **synapse_config
                    )

    monitor = Monitor(model)
    net = Network(dt=0.001)

    net.add_input_train(input_train)
    net.add_layer(model)
    net.add_synapse(synapse)

    net.run(time= time)


    times_added, indices_added, position_copypaste_added, patterns_info_added = generator.add_new_pattern(times, indices, position_copypaste, patterns_info)


    input_train = InputTrain(times_added, indices_added)
    model.add_new_neurons(num_new_neurons=num_new_neurons)
    synapse.add_new_neurons(num_new_neurons=num_new_neurons)
    monitor = Monitor(model)

    net = Network(dt=0.001)

    net.add_input_train(input_train)
    net.add_layer(model)
    net.add_synapse(synapse)

    net.run(time= time)

    result = get_latency_results(N, monitor, position_copypaste_added, inference_time, patternlength=0.050)

    plot_new_pattern_hist(P, result, display, result_dir, image_name)

    json_name = image_name.split('.png')[0] + '.json'
    with open(os.path.join(result_dir, json_name), 'w') as f:
        json.dump(result, f)

    return monitor, position_copypaste_added, synapse, model, net, input_train, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--P', type=int, required=True, help='number of patterns')
    parser.add_argument('--N', type=int, required=True, help='number of neurons')
    parser.add_argument('--num_new_neurons', type=int, required=True, help='number of new neurons')
    parser.add_argument('--M', type=int, required=True, help='number of presynaptic neurons')
    parser.add_argument('--time', type=int, required=True, help='total time')
    parser.add_argument('--inference_time', type=int, required=True, help='final histogram considers spikes happening after this time')
    parser.add_argument('--display', type=bool, required=True, help='display the plot')
    parser.add_argument('--result_dir', type=str, required=True, help='directory for the result image')
    parser.add_argument('--image_name', type=str, required=True, help='name of the result image')
    args = parser.parse_args()
    P = args.P
    N = args.N
    num_new_neurons = args.num_new_neurons
    M = args.M
    time = args.time
    inference_time = args.inference_time
    display = args.display
    result_dir = args.result_dir
    image_name = args.image_name


    monitor, position_copypaste, synapse, model, net, input_train, result  = main(P, N, M, num_new_neurons, time, inference_time, display, result_dir, image_name)


# 'new-pattern-hist.png'