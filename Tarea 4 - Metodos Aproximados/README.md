# IIC3675 Aprendizaje Reforzado: Tarea 4 - Métodos Aproximados

## Overview

This repository contains the code and experiments for Assignment 4 of IIC3675, focusing on approximate reinforcement learning methods. The main tasks include implementing and evaluating Sarsa, Q-Learning, DDPG, and DQN agents on the MountainCar environments, as well as performing hyperparameter searches and result analysis.

## Repository Structure
```bash
.
├── Code
│   ├── actor_critic.py
│   ├── ddpg_param_trial.py
│   ├── ddpg.py
│   ├── dqn_parameter_trial.py
│   ├── dqn.py
│   ├── FeatureExtractor.py
│   ├── find_best_models.py
│   ├── LinearActorCritic.py
│   ├── Plotting/ 
│   │   ├── taskA.py
│   │   └── taskB_D.py
│   ├── QLearning.py
│   ├── RandomAgent.py
│   ├── Sarsa.py
│   ├── taskA.py
│   ├── tiles3.py
│   └── tqdmCallback.py
├── code_t4.zip
├── Data
│   ├── DDPG/
│   ├── DDPG_Trials/
│   ├── DQN/
│   ├── DQN_Trials/
│   ├── taskA_q_learning_results.csv
│   └── taskA_sarsa_results.csv
├── Enunciado_T4.pdf
├── env.yaml 
├── IIC3675_T4_Tabular_Methods.pdf
├── Models
│   ├── DDPG/
│   ├── DDPG_Trials/
│   ├── DQN/
│   └── DQN_Trials/
├── Plots
│   ├── ddpg_mountain_car_cont.png
│   ├── dqn_mean.png
│   ├── Sarsa_vs_q_learning.png
│   └── sarsa_vs_ql_small.png
└── README.md

14 directories, 24 files

```

- `Code/`: Contains the implementation of various reinforcement learning algorithms and utilities.
  - `Plotting/`: Scripts for plotting results.
- `Data/`: Output data and results from experiments.
- `Models/`: Saved models from training runs.
- `Plots/`: Generated plots and figures.
- `Enunciado_T4.pdf`: The assignment description and requirements.
- `code_t4.zip`: Skeleton code for the assignment.
- `env.yaml`: Conda environment file for setting up the required dependencies.
- `IIC3675_T4_Tabular_Methods.pdf`: Document containing results and analysis of the tabular methods.
- `README.md`: This file, providing an overview of the repository.

## Dependencies
Install the required packages using:
```bash
conda env create -f env.yaml
conda activate <your_env_name>
```

## Running the Experiments

1. For Sarsa and Q-Learning with MountainCar-v0 run:

```bash
python Code/taskA.py
```

2. For DQN hyperparameter trials, run:

```bash
python Code/dqn_parameter_trial.py
```

3. For DQN with MountainCar-v0, run:

```bash
python Code/dqn.py
```

4. For Actor-Critic with DDPG, run:

```bash
python Code/actor_critic.py
```
- Note that this script plots automatically.
  
1. For DDPG hyperparameter trials, run:

```bash
python Code/ddpg_param_trial.py
```

6. For DDPG with MountainCarContinuous-v0, run:

```bash
python Code/ddpg.py
```

7. For plotting results, use the scripts in the `Code/Plotting/` directory. For example:

```bash
python taskA.py
```
- **Note**: The plotting scripts has to be run from the `Code/Plotting/` directory. If not, please adjust the paths accordingly.
  
## Notes
- Results may very due to the stochastic nature of the environments and algorithms.

## Authors

- William Aarland - waarland0@uc.cl
- Pascal Lopez Wilkendorf - pascal.lopez@uc.cl
