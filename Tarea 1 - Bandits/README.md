# IIC3675 Reinforcement Learning: Assignment 1 - Bandits

The purpose of this assignment is to familiarize the student with the $\epsilon$-greedy algorithm widely used in reinforcement learning.
The assignemnt description, and tasks can be read in this [PDF](/Tarea%201%20-%20Bandits/Enunciado_T1.pdf).

## Replicating and executing code

By utilizing the `env.yaml` file in the repository, the exact same dependencies, and versions can be used to create the exact same environment for code executing

### Steps to replicate environment

For this step, I have been using `conda` (read more [here](https://anaconda.org/anaconda/conda)). To replicate the environment execute:
`conda env create -f env.yaml`
in your terminal
This should create the exact same environment.

### Executing the code
Lipsum lorum whatever

## Tasks

a) Super nice answer

b) Figure 2.2 in *Reinforcement Learning: An Introduction* shows that even with an $\epsilon = 0.1$ the optimal action is only chosen about $80\%$ of the time. The reason being that the agent is unaware of the actual true value, or expected reward, or action $a$ at time $t$, denoted $q_{*}(a)$. However, the agent only has en estimate of $q_{*}(a)$, denoted $Q_{t}(a)$, which is not the true expected reward, but $Q_{t}(a)$ will asymptotically converge towards the true expected value. Written mathematically as follows:
$$
Q_{t}(a) \rightarrow q_{*}(a) \text{ when } t \rightarrow \infty
$$

c)
