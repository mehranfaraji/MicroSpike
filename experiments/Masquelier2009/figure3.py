from microspike import SRMInhibitory, Synapse, utils, Network, Monitor, InputTrain, PatternGenerator
import matplotlib.pyplot as plt
import numpy as np
import os

HYPERPARAMETERS_JSON_MODEL = "model_hyperparameters.json"
HYPERPARAMETERS_JSON_SYNAPSE = "synapse_hyperparameters.json"
model_hyperparams = utils.load_hyperparameters(HYPERPARAMETERS_JSON_MODEL)
synapse_hyperparams = utils.load_hyperparameters(HYPERPARAMETERS_JSON_SYNAPSE)


def plot_potential(ax, start_time, end_time, monitor, threshold, position_copypaste, patternlength):
    pattern_idx = np.where(position_copypaste > 0)[0]
    pattern_time = pattern_idx * patternlength
    ind = np.where((pattern_time >= start_time) & (pattern_time <= end_time))
    for time in pattern_time[ind]:
        ax.axvspan(time, time + patternlength, alpha=0.3, color='gray')
    linestyles = ['-', '--', ':' ]
    for i in range(monitor.N):
        neuron_number = i + 1
        potential = monitor.potential_rec[i]
        time_ms = np.arange(start_time, end_time, 0.001)
        ax.plot(time_ms, potential[int(start_time*1000):int(end_time*1000)], linestyle=linestyles[i], label=f'Neuron {neuron_number}', color='black')

    ax.hlines(threshold, xmin=start_time, xmax=end_time, color='black', linestyles='--', label='threshold')

    ax.set_xlabel('t (s)')
    ax.set_ylabel('Potential (a. u.)')
    ax.legend(loc='upper right')

    
def plot_latencies(ax, latencies):
    ax.scatter(range(len(latencies)), latencies, color='k', s=1)
    ax.set_xlim(0, len(latencies))
    ax.set_ylim(0, 50)
    ax.set_xlabel('# discharges')
    ax.set_ylabel('Postsynaptic spike latency (ms)')


def plot_fig3(monitor, threshold, position_copypaste, patternlength):
    picture_dir = 'picture'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    picture_dir = os.path.join(current_dir, picture_dir)
    
    latencies = monitor.get_latencies(position_copypaste, patternlength)

    plt.figure(figsize=(8, 7))
    # plot 1
    plt.subplot(3, 1, 1)
    start_time = 0
    end_time = 0.16
    plot_potential(plt.gca(), start_time, end_time, monitor, threshold, position_copypaste, patternlength)

    # plot 2
    plt.subplot(3, 1, 2)
    start_time = 224.65
    end_time = 225
    plot_potential(plt.gca(), start_time, end_time, monitor, threshold, position_copypaste, patternlength)
    plt.gca().legend()

    # plot 3 with 3 columns
    plt.subplot(3, 3, 7)
    plot_latencies(plt.gca(), latencies[0])

    plt.subplot(3, 3, 8)
    plot_latencies(plt.gca(), latencies[1])
    plt.gca().set_ylabel('')

    plt.subplot(3, 3, 9)
    plot_latencies(plt.gca(), latencies[2])
    plt.gca().set_ylabel('')

    plt.tight_layout()

    if not os.path.exists(picture_dir):
        os.makedirs(picture_dir)
    plt.savefig(os.path.join(picture_dir, 'figure3.png'))

def fig3():
    P = 1
    N = 3
    M = 2000
    time = 450 * 1000 #ms
    patternlength = 0.050
    threshold = 500

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

    plot_fig3(monitor, threshold, position_copypaste, patternlength)


if __name__ == "__main__":
    fig3()
    plt.show()
