# Experiments

This directory contains re-implementations of key experiments that use the MicroSpike library to simulate results from the following papers:

- **Masquelier et al. (2008):** Spike Timing Dependent Plasticity Finds the Start of Repeating Patterns in Continuous Spike Trains. [DOI](https://doi.org/10.1371/journal.pone.0001377)
- **Masquelier et al. (2009):** Competitive STDP-based spike pattern learning. [DOI](https://doi.org/10.1162/neco.2008.06-08-804)

Each subdirectory corresponds to a specific paper and contains Python scripts to generate the key figures from the respective publication.

## Usage

After installing MicroSpike, navigate to the desired experiment's directory and run the respective script to generate all figures, or individual figure scripts to create specific results.

Example for the Masquelier2008 experiment:

```bash
cd Masquelier2008
python Masquelier2008.py  # Generates all important figures
```

To generate a specific figure:

```bash
python figure4.py  # Only creates Figure 4 from the paper
```

The results will be saved as `.png` images in the `picture` directory.

For more detailed usage and explanations, refer to the specific experiment's README.