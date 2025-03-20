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

b) Figure 2.2 in *Reinforcement Learning: An Introduction* shows that even with an $\epsilon = 0.1$ the optimal action is only chosen about $80\%$ of the time. The reason being that the agent is unaware of the actual true value, or expected reward, or action $a$ at time $t$, denoted $q_{*}(a)$. However, the agent only has en estimate of $q_{*} (a)$, denoted $Q_{t}(a)$, which is not the true expected reward, but $Q_{t}(a)$ will asymptotically converge towards the true expected value. Written mathematically as follows:

$$
Q_{t}(a) \rightarrow q_{*} (a) \text{ when } t \rightarrow \infty
$$

c) This task entails us to make a variation of the *simple bandit algorithm* with a *fixed step size*, $\alpha \in ( 0, 1 ] $ to alter how the agent views, and emphasizes the recent versus older observations. This is shown mathematically by Equation 2.6 in *Reinforcement Learning: An Introduction*. Furthermore, by implementing *optimistic initial values* to the estimatees of the actions, the agent will be motivated to explore more in the beginning.
This is shown in the figure below, where we see that the agent in the beginning chooses actions that are not necessarily the optimal, but chooses them because of the inital values. So the first steps deviate from the optimal, but the agent quickly learns and surpasses the realistic agent as expected.

![](/Tarea%201%20-%20Bandits/Bandits/Plots/optimist_and_realist_bandint.png)

d) As seen in the graph for the optimistic agent with $\alpha = 0.1$ and $\epsilon = 0$, there is a sudden performance increase followed by a sharp drop. The reason being is that in the beginning the agent encourages self-exploration without the need of randomness (without the need for randomness (since ε = 0) , leading to the agent choosing the optimal actions. However this process causes some temporary misjudgment, where suboptimal actions may still appear slightly better than the true optimal action due to the initial overestimation, leading to the sharp drop right after the early spike in the graph.

e) As displayed in the graph, the optimistic agent seems to converge to a value of $85\%$ of optimal actions taken. This can be due to several factors. One possibility is that the $\epsilon = 0$, meaning that the agent might choose an suboptimal action due to the stochastic nature of the rewards received, and does not further explore. In other terms, the agent overestimates, and favours, leading to a wrong action being favoured.

f)


g) Using **μ + 4** impacts the values in experiments without a baseline. When a baseline is used, the values are adjusted to the mean, ensuring the results remain unaffected. However, **μ + 4** influences performance because preference updates,  **H(a)** , do not adjust to the mean of the rewards. In contrast, setting **μ = 0** ensures performance remains unchanged, whether a baseline is used or not, since the rewards are naturally centered around  **0** , which serves as the baseline value.
