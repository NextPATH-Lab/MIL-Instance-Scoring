# Instance Attention is Misleading: Bag Composition Affects Interpretability in Digital Pathology Multiple Instance Learning
Previously *Instance Scoring via Distillation of Multiple Instance Classifiers for Interpretable Digital Pathology*  

This repository houses code for the following related publications:
* **SPIE Journal of Medical Imaging (*under review*)** Instance Attention is Misleading: Bag Composition Affects Interpretability in Digital Pathology Multiple Instance Learning
* Fumiya Inaba, Mira Keyes, Calum MacAulay, et al. "Instance scoring via distillation of multiple instance classifiers for interpretable digital pathology", Proc. SPIE 13932, Medical Imaging 2026: Digital and Computational Pathology, 1393205 (2 Apr 2026); https://doi.org/10.1117/12.3087751

## Experimentation
### Setting up environment
The environment can be reproduced using the `pyproject.toml` file provided. The file should support Linux, Windows, and MacOS systems.
For a brief guide on using `uv` to recreate the environment, see [Using UV to reproduce environments](./uv-environments.md).

**NOTE:** Python 3.12 is the recommended version of Python.

### Table of Contents for Experiments
1. [Proportion MNIST Bags Experiment](experiments/experiment-promnist-bags.py)
2. [Proportion MNIST Bags - Instance Logit \& Attention](experiments/experiment-promnist-za-dists.py)

### Proportion MNIST Bags
#### 1. Proportion MNIST Bags Experiment
This script (`experiment-promnist-bags.py`) runs the full experiment, from dataset creation to training ABMIL classifiers, computing correlation between instance logit and its attention weight.
Saves relevant results data to `promnist.csv` file.
This script should be run first.

#### 2. Proportion MNIST Bags - Instance Logit & Attention
This script (`experiment-promnist-za-dists.py`) creates another `.csv` file to be used for plotting instance attention against instance logit for specific datasets with different $\tau$ values.
This relies on the datasets created in the part above, so should be run after (`experiment-promnist-bags.py`).