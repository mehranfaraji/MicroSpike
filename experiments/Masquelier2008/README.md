# Masquelier et al. (2008) Experiment

This directory contains the re-implementation of the experiment described in:

- **Masquelier T, Guyonneau R, Thorpe SJ (2008):** Spike Timing Dependent Plasticity Finds the Start of Repeating Patterns in Continuous Spike Trains. [DOI](https://doi.org/10.1371/journal.pone.0001377)

## How to Run

To generate all the important figures from the paper, simply run the main script:

```bash
python Masquelier2008.py
```

If you only want to generate specific figures, you can run the individual figure scripts:

```bash
python figure3.py  # Generates and saves Figure 3
python figure4.py  # Generates and saves Figure 4
python figure5.py  # Generates and saves Figure 5
```

The generated figures will be saved as `.png` files in the `picture` directory.

## Key Figures

- **Figure 3:** Illustrative example of the behaviour of the Leaky Integrate-and-Fire neuron with only 6 input spikes.
- **Figure 4:** Overview of the 450 seocnds simulation.
- **Figure 5:** Latenct Reduction

For further details on how the simulation replicates the paper's findings, refer to the paper or explore the provided scripts.
