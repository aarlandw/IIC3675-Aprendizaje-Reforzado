import gymnasium as gym
from Sarsa import Sarsa
from QLearning import QLearning
from tqdm import trange
import pandas as pd
import numpy as np


def run_agent(agent_class, num_runs: int, num_episodes: int, env_name: str, *agent_args):
    results = []
    for _ in trange(num_runs, desc=f"{agent_class.__name__} runs"):
        with gym.make(env_name) as env:
            num_actions = env.action_space.n
            agent = agent_class(num_actions, *agent_args)
            rewards = agent.run(env, num_episodes=num_episodes)
            results.append(rewards)
    return results

def taskA():
    """
    Task A of the assignment: Implement and compare Sarsa and Q-learning agents with linear approximation on the MountainCar-v0 environment.
    """
    EPSILON = 0
    ALPHA = 0.5 / 8
    GAMMA = 1
    NUM_RUNS = 30
    NUM_EPISODES = 1000

    sarsa_results = run_agent(Sarsa, NUM_RUNS, NUM_EPISODES, "MountainCar-v0", EPSILON, ALPHA, GAMMA)
    pd.DataFrame(sarsa_results).to_csv("taskA_sarsa_results.csv", index=False)

    q_learning_results = run_agent(QLearning, NUM_RUNS, NUM_EPISODES, "MountainCar-v0", EPSILON, ALPHA, GAMMA)
    pd.DataFrame(q_learning_results).to_csv("taskA_q_learning_results.csv", index=False)
    

if __name__ == "__main__":
    taskA()