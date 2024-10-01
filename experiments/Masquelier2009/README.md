# Masquelier et al. (2009) Experiment

This directory contains the re-implementation of the experiment described in:

- **Masquelier T, Guyonneau R, Thorpe SJ (2009):** Competitive STDP-based spike pattern learning. [DOI](https://doi.org/10.1162/neco.2008.06-08-804)

## How to Run

To generate all the important figures from the paper, simply run the main script:

```bash
python Masquelier2009.py
```

If you only want to generate specific figures, you can run the individual figure scripts:

```bash
python figure3.py  # Generates and saves Figure 3
```

The generated figures will be saved as `.png` files in the `picture` directory.

## Key Figures

- **Figure 3:** Spike pattern learning with competitive inhibition during different simulation times and latency reduction of each neuron.