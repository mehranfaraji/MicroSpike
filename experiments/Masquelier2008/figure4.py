from microspike import SRMInhibitory, Synapse, utils, Network, Monitor, InputTrain, PatternGenerator
import matplotlib.pyplot as plt
import numpy as np
import os

HYPERPARAMETERS_JSON_MODEL = "model_hyperparameters.json"
HYPERPARAMETERS_JSON_SYNAPSE = "synapse_hyperparameters.json"
model_hyperparams = utils.load_hyperparameters(HYPERPARAMETERS_JSON_MODEL)
synapse_hyperparams = utils.load_hyperparameters(HYPERPARAMETERS_JSON_SYNAPSE)


def plot_potential(ax, start_time, end_time, potential, threshold, resting_pot, position_copypaste, patternlength):
    pattern_idx = np.where(position_copypaste > 0)[0]
    pattern_time = pattern_idx * patternlength
    ind = np.where((pattern_time >= start_time) & (pattern_time <= end_time))
    
    for time in pattern_time[ind]:
        ax.axvspan(time, time + patternlength, alpha=0.3, color='gray')

    time_ms = np.arange(start_time, end_time, 0.001)
    ax.plot(time_ms, potential[int(start_time*1000):int(end_time*1000)], label='potential', color='blue')
    ax.hlines(threshold, xmin=start_time, xmax=end_time, color='red', linestyles='--', label='threshold')
    ax.hlines(resting_pot, xmin=start_time, xmax=end_time, color='black', linestyles=':', label='resting pot.')
    ax.set_xlabel('t (s)')
    ax.set_ylabel('Potential (a. u.)')
    ax.legend(loc='upper right')


def plot_fig4(monitor, position_copypaste):
    potential = monitor.potential_rec[0]
    patternlength = 0.050
    threshold = 500
    resting_pot = 0
    
    picture_dir = 'picture'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    picture_dir = os.path.join(current_dir, picture_dir)

    plt.figure(figsize=(10, 6))

    plt.subplot(3, 1, 1)
    plot_potential(plt.gca(), 0, 1, potential, threshold, resting_pot, position_copypaste , patternlength)

    plt.subplot(3, 1, 2)
    plot_potential(plt.gca(), 13.25, 14.25, potential, threshold, resting_pot, position_copypaste , patternlength)

    plt.subplot(3, 1, 3)
    plot_potential(plt.gca(), 449, 450, potential, threshold, resting_pot, position_copypaste , patternlength)

    plt.tight_layout()
    if not os.path.exists(picture_dir):
        os.makedirs(picture_dir)
    plt.savefig(os.path.join(picture_dir, 'figure4.png'))


def fig4():
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


    plot_fig4(monitor, position_copypaste)


if __name__ == "__main__":

    fig4()
    plt.show()