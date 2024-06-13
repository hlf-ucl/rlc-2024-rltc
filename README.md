# Trust-based Consensus in Multi-Agent Reinforcement Learning Systems

This repository contains the experimental code of
_Trust-based Consensus in Multi-Agent Reinforcement Learning Systems_.

## Prerequisites

The experimental code in this repository was tested and run on Linux (Centos 7.9.2009) using Python 3.10.3.
Dependencies are specified in `requirements.txt`. 
The containerized setup uses Singularity 3.8.7-1.el7 (see [Singularity container](#singularity-container)).

## Instructions

Below provides instructions for setting up and running the experiment code
either within a Python 3 virtual environment or as a Singularity container.
Both methods are more-or-less equivalent as long as the Python version is correct (3.10.3),
as they both use dependencies specified by the same `requirements.txt`.

### Virtual environment

Create a Python 3 virtual environment in the source root
directory and install dependencies from `requirements.txt`:

```
python3 -m venv ./venv
source venv/bin/activate
pip3 install -r requirements.txt
```

The experiments presented in the paper were run using the following command:

```
python3 -m game_ext.main ./configs/config.json ./seeds/seeds.txt --results . --num-workers 16
```

`configs/config.json` and `seeds/seeds.txt` contain the parameter combinations and random seeds respectively.
A new folder named `results` will be created at the location specified by `--results`
(the current working directory in the above example).
`--num-workers` specifies how many separate processes to run experiments in parallel. `N=16` was used, but another value can be specified (default=1).

### Singularity container

Alternatively, one can create a Singularity container
from the definition file `consensus.def` provided in the source root, then run it as a command:

```
sudo singularity build consensus.sif consensus.def
singularity run consensus.sif /experiments/configs/config.json /experiments/seeds/seeds.txt --results . --num-workers 16
```

## Repository Structure

Refer to the READMEs within each directory for more information.

- `game_ext`: package containing code for the environment, RL agent and training/evaluation.
- `notebooks`: Jupyter notebooks and scripts used to generate plots and visualizations for the paper.
- `configs`: JSON files containing parameter combinations for running experiments.
- `seeds` : random seeds used in experiments.