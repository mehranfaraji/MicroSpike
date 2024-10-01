from microspike import SRMInhibitory, Synapse, utils, Network, Monitor, InputTrain, PatternGenerator
import matplotlib.pyplot as plt
import numpy as np
import os

HYPERPARAMETERS_JSON_MODEL = "model_hyperparameters.json"
HYPERPARAMETERS_JSON_SYNAPSE = "synapse_hyperparameters.json"


def plot_fig3(time, potential, sample_times_ms, sample_model_hyperparams):
    picture_dir = 'picture'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    picture_dir = os.path.join(current_dir, picture_dir)
    
    time_ms = np.arange(0, time+1, 1)
    plt.figure(figsize=(8, 6))
    plt.plot(time_ms, potential, label='potential', color='blue')
    plt.vlines(sample_times_ms, ymin=-3, ymax=6, color='gray', linestyles='-.', label='input spike times')

    plt.hlines(sample_model_hyperparams['threshold'], xmin=0, xmax=time+1, color='red', linestyles='--', label='threshold')
    plt.hlines(0, xmin=0, xmax=time+1, color='black', linestyles=':', label='resting pot.')

    plt.xlabel('t (ms)')
    plt.ylabel('Potential (arbitrary units)')
    plt.legend(loc='upper right')

    plt.ylim(-3, 6)
    plt.xlim(0, 80)

    if not os.path.exists(picture_dir):
        os.makedirs(picture_dir)
    plt.savefig(os.path.join(picture_dir, 'figure3.png'))


def fig3():
    N = 1
    time = 75

    sample_model_hyperparams = utils.load_hyperparameters(HYPERPARAMETERS_JSON_MODEL)
    sample_synapse_hyperparams = utils.load_hyperparameters(HYPERPARAMETERS_JSON_SYNAPSE)
    sample_model_hyperparams['threshold'] = 2.9
    sample_synapse_hyperparams['A_post'] = 0
    sample_synapse_hyperparams['A_pre'] = 0

    sample_times = np.array([0.002, 0.023, 0.045, 0.046, 0.049, 0.061])
    sample_times_ms = sample_times * 1000
    sample_indices = np.array([0, 0, 0, 0, 0, 0])
    sample_weight = np.array([[1]])
    sample_input_train = InputTrain(sample_times, sample_indices)
    sample_model = SRMInhibitory(N=N,
                                **sample_model_hyperparams)
    sample_monitor = Monitor(sample_model)
    sample_synapse = Synapse(w=sample_weight,
                    **sample_synapse_hyperparams)

    sample_net = Network(dt=0.001)
    sample_net.add_input_train(sample_input_train)
    sample_net.add_layer(sample_model)
    sample_net.add_synapse(sample_synapse)

    sample_net.run(time=75)
    potential = sample_monitor.potential_rec[0]

    plot_fig3(time, potential, sample_times_ms, sample_model_hyperparams)
    


if __name__ == "__main__":

    fig3()
    plt.show()