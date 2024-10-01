# MicroSpike

MicroSpike is a Python library designed to simulate Spiking Neural Networks (SNNs) and train them using the Spike-Timing-Dependent Plasticity (STDP) learning rule. The library's structure is inspired by the Brian2 simulator and focuses on simplicity, allowing users to implement and simulate basic spiking neuron models. Currently, MicroSpike supports a single-layer, feed-forward architecture with neurons modeled using the Spike Response Model (SRM).

## Features
- **SRM Neuron Model:** Use kernels to compute the postsynaptic potential from incoming spikes.
- **STDP Synapses:** Simulate synapses that support spike-timing-dependent plasticity.
- **Input Spike Trains:** Generate input spike trains with random bursts of noise and embedded patterns.
- **Network Components:** Build networks using `InputTrain`, `Synapse`, `SRM`, `Monitor`, and `Network` components.
- **Pattern Learning:** Train neurons with specific patterns and subsequently add new neurons to learn new patterns.

## Installation

MicroSpike requires Python 3.10 and the following libraries:

```bash
numpy==1.26.4
numba==0.59.1
matplotlib==0.1.7
```

To install the package, first clone the repository, then install using `pip`:

```bash
git clone https://github.com/your-repo/microspike.git
cd microspike
pip install .
```

We recommend using a Python virtual environment for installation.

## Usage

Here is a simple example of using MicroSpike to set up a basic network:

```python
from microspike import InputTrain, Synapse, SRM, Monitor, Network, PatternGenerator

# Define hyperparameters
threshold = 500
reset = 0
tau_m = 0.010
tau_s = 0.0025
K1 = 2
K2 = 4
window_time = 7 * 0.010

# Synapse parameters
w_max = 1
A_pre = 0.03125
B = 0.85
A_post = -A_pre * B
tau_pre = 0.0168
tau_post = 0.0337

# Generate input spike train
generator = PatternGenerator(number_pattern=2, number_neurons=100, total_pattern_freq=1/3)
times, indices, *_ = generator.generate()

# Initialize components
input_train = InputTrain(times, indices)
model = SRM(N=100, threshold=threshold, reset=reset, tau_m=tau_m, tau_s=tau_s, K1=K1, K2=K2, window_time=window_time)
synapse = Synapse(w_max=w_max, A_pre=A_pre, A_post=A_post, tau_pre=tau_pre, tau_post=tau_post)
monitor = Monitor(model)
net = Network()

# Build and run the network
net.add_input_train(input_train)
net.add_layer(model)
net.add_synapse(synapse)
net.run(time=1.0)  # Run for 1 second
```

## Experiments

MicroSpike provides re-implementations of key Spiking Neural Network experiments. See the [experiments](./experiments) directory for more details.

- **Masquelier et al. (2008):** STDP finds repeating patterns in spike trains.
- **Masquelier et al. (2009):** Competitive STDP-based spike pattern learning.

## License
This project is licensed under the terms of the MIT License.