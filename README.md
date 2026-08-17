# Instance Attention is Misleading: Bag Composition Affects Interpretability in Digital Pathology Multiple Instance Learning
Previously *Instance Scoring via Distillation of Multiple Instance Classifiers for Interpretable Digital Pathology*  
This branch specifically contains code necessary to replicate the MNIST Bags experiment of the conference proceedings paper.

This repository houses code for the following related publications:
* Fumiya Inaba, Mira Keyes, Calum MacAulay, et al. "Instance scoring via distillation of multiple instance classifiers for interpretable digital pathology", Proc. SPIE 13932, Medical Imaging 2026: Digital and Computational Pathology, 1393205 (2 Apr 2026); https://doi.org/10.1117/12.3087751

## Experimentation
### Setting up environment
The environment can be reproduced using the `pyproject.toml` file provided. The file should support Linux, Windows, and MacOS systems.
For a brief guide on using `uv` to recreate the environment, see [Using UV to reproduce environments](./uv-environments.md).

**NOTE:** Python 3.12 is the recommended version of Python.

### Downloading Data
There are three datasets used in this repository. Below is a brief guide to downloading these datasets locally.

#### 1. Proportion MNIST Bags Data
The proportion MNIST bags dataset is created using the MNIST dataset, available from the `torchvision` package (`torchvision.datasets.MNIST`)  
The proportion MNSIT bags dataset is created in the `experiment-promnist-bags.py` script.

#### 2. Prostate Cancer LDO Dataset
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21983424-blue)](https://doi.org/10.5281/zenodo.21983424)  
LDO features are computed from segmented nuclei of Feulgen-thionin stained prostate core needle biopsies.
The collection of features for each patient can be accessed above.
After download, the data should be formatted in the following manner:
```
data_folder/
    |-- Tensors/
    |-- |-- 115.csv
    |-- |-- ...
    |-- clinical_deidentified_filtered.csv
```
The location of `data_folder` should be saved in the `.env` file.

### Table of Contents for Experiments
1. [Proportion MNIST Bags Experiment](experiments/experiment-promnist-bags.py)
2. [Proportion MNIST Bags - Instance Logit \& Attention](experiments/experiment-promnist-za-dists.py)
3. [Large-scale DNA Organization for Prostate Cancer - Feature Selection](#1-ldo-feature-selection)
4. [Large-scale DNA Organization for Prostate Cancer - ABMILite](#2-prognosis-via-attention-based-ldo-analysis)

### **Proportion MNIST Bags**
#### 1. Proportion MNIST Bags Experiment
This script (`experiment-promnist-bags.py`) runs the full experiment, from dataset creation to training ABMIL classifiers, computing correlation between instance logit and its attention weight.
Saves relevant results data to `promnist.csv` file.
This script should be run first.

#### 2. Proportion MNIST Bags - Instance Logit & Attention
This script (`experiment-promnist-za-dists.py`) creates another `.csv` file to be used for plotting instance attention against instance logit for specific datasets with different $\tau$ values.
This relies on the datasets created in the part above, so should be run after (`experiment-promnist-bags.py`).

### **Large-scale DNA Organization (LDO) for Prostate Cancer**
#### 1. LDO Feature Selection
Use the `utils-ldo-feature-selection.py` script to remove features with redundancies (high Pearson correlation `>0.9`). This will create a new folder within the data directory called `filtered_copy`. The newly created files will be used in model training.

#### 2. Prognosis via Attention-based LDO Analysis
Use the `experiment-ldo-abmilite-training.py` script to train ABMILite model, save its weights along with creating figures for manuscript.
