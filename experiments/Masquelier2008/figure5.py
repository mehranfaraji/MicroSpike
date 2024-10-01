from microspike import SRMInhibitory, Synapse, utils, Network, Monitor, InputTrain, PatternGenerator
import matplotlib.pyplot as plt
import numpy as np
import os

HYPERPARAMETERS_JSON_MODEL = "model_hyperparameters.json"
HYPERPARAMETERS_JSON_SYNAPSE = "synapse_hyperparameters.json"
model_hyperparams = utils.load_hyperparameters(HYPERPARAMETERS_JSON_MODEL)
synapse_hyperparams = utils.load_hyperparameters(HYPERPARAMETERS_JSON_SYNAPSE)

def plot_fig5(latencies):
    picture_dir = 'picture'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    picture_dir = os.path.join(current_dir, picture_dir)

    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(latencies)), latencies, color='b', s=1)
    plt.xlim(0, len(latencies))
    plt.ylim(0, 50)
    plt.xlabel('# discharges')
    plt.ylabel('Postsynaptic spike latency (ms)')

    if not os.path.exists(picture_dir):
        os.makedirs(picture_dir)
    plt.savefig(os.path.join(picture_dir, 'figure5.png'))


def fig5():
    P = 1
    N = 1
    M = 2000
    time = 450 * 1000 #ms

    generator = PatternGenerator(number_pattern=P, number_neurons=M,
                                    total_pattern_freq = 1/3,
                                    )
    times, indices, position_copypaste, patterns_info, timing_pattern = generator.generate()


    weight = utils.w_uniform(M=M, N=N)
    input_train = InputTrain(times, indices)
    model = SRMInhibitory(N=N,
                **model_hyperparams)
    synapse = Synapse(w=weight,
                    **synapse_hyperparams)
    monitor = Monitor(model)
    net = Network(dt=0.001)
    net.add_input_train(input_train)
    net.add_layer(model)
    net.add_synapse(synapse)

    net.run(time=time)

    latencies = monitor.get_latencies(position_copypaste, patternlength=0.050)
    plot_fig5(latencies[0])


if __name__ == "__main__":

    fig5()
    plt.show()