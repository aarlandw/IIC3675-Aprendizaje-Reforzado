# IIC3675 - Entrega 2: Reinforcement Learning — MDPs and Monte Carlo

- [IIC3675 - Entrega 2: Reinforcement Learning — MDPs and Monte Carlo](#iic3675---entrega-2-reinforcement-learning--mdps-and-monte-carlo)
  - [Folder Structure](#folder-structure)
  - [Running the code](#running-the-code)
    - [Installation](#installation)
    - [Running the MDPs code](#running-the-mdps-code)
    - [Running the Monte Carlo code](#running-the-monte-carlo-code)
  - [Report](#report)
  - [Authors](#authors)

This project contains the implementation and analysis for the second assignment of IIC3675, covering:

- Dynamic Programming methods: Iterative Policy Evaluation, Greedy Policy Evaluation, and Value Iteration.

- Monte Carlo Control methods using every-visit updates and $\epsilon$-soft policies.


## Folder Structure

```bash
.
├── Code_T2
│   ├── MDPs
│   │   ├── main_g.py
│   │   ├── main_h.py
│   │   ├── Main.py
│   │   └── Problems
│   │       ├── AbstractProblem.py
│   │       ├── CookieProblem.py
│   │       ├── GamblerProblem.py
│   │       ├── GridLocations.py
│   │       └── GridProblem.py
│   └── MonteCarlo
│       ├── blackjack_mc_rewards.csv
│       ├── cliff_mc_rewards.csv
│       ├── Environments
│       │   ├── AbstractEnv.py
│       │   ├── BlackjackEnv.py
│       │   ├── CliffEnv.py
│       │   └── CliffVisualizer.py
│       ├── graficos.py
│       ├── Main_BlackJack.py
│       └── Main_Cliff.py
├── Code_T2.zip
├── Enunciado_T2.pdf
├── env.yaml
├── IIC3675-Tarea2-MDPs-Monte_Carlo.pdf
└── README.md

5 directories, 22 files
```

- `Code_T2/MDPs`: Contains the code for the MDPs part of the assignment.
  - `Problems/`: Contains starter code of problems for the MDPs part of the assignment.
    - `AbstractProblem.py`: Abstract class for the problems.
    - `CookieProblem.py`: Cookie problem.
    - `GamblerProblem.py`: Gambler problem.
    - `GridLocations.py`: Grid locations problem.
    - `GridProblem.py`: Grid problem.
  - `main_g.py`: Main file for task g in the assignment.
  - `main_h.py`: Main file for task h in the assignment.
- `Code_T2/MonteCarlo`: Contains the code for the Monte Carlo part of the assignment.
  - `Environments/`: Contains the starter code for the environments used in the Monte Carlo part of the assignment.
    - `AbstractEnv.py`: Abstract class for the environments.
    - `BlackjackEnv.py`: Blackjack environment.
    - `CliffEnv.py`: Cliff environment.
    - `CliffVisualizer.py`: Cliff visualizer.
- `Code_T2.zip`: Contains the starter code handed out by the professor for the project.
- `Enunciado_T2.pdf`: The assignment.
- `env.yaml`: The environment file for the project.
- `IIC3675-Tarea2-MDPs-Monte_Carlo.pdf`: The report for the project.
- `README.md`: This file.

## Running the code

### Installation

The packages used in this project are listed in the `env.yaml` file. To create the environment, run:

```bash
conda env create -f env.yaml
```

or if you want to create the environment with a specific name, run:

```bash
conda env create -f env.yml -n {your_env_name}
```

Then, activate the environment with:

```bash
conda activate {your_env_name}
```

### Running the MDPs code

To run the MDPs code, navigate to the `Code_T2/MDPs` folder and run:

```python
python Main.py
```

for task d, or

```python
python main_g.py
```

for task g, or

```python
python main_h.py
```

for task h.

### Running the Monte Carlo code

To run the Monte Carlo code, navigate to the `Code_T2/MonteCarlo` folder and run:

```python
python Main_BlackJack.py
```

for the Blackjack environment, or

```python
python Main_Cliff.py
```

for the Cliff environment.

## Report

The report for the project is in the `IIC3675-Tarea2-MDPs-Monte_Carlo.pdf` file. It contains the analysis and results of the assignment.

## Authors

William Aarland - waarland0@uc.cl \
Pascal Lopez Wilkendorf - pascal.lopez@uc.cl
