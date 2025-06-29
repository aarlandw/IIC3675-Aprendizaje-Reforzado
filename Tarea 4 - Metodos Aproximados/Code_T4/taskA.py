import gymnasium as gym
from Sarsa import Sarsa
from QLearning import QLearning
from tqdm import trange
import pandas as pd
import numpy as np

def taskA():
    """
    Task A of the assignment: Implement and compare Sarsa and Q-learning agents with linear approximation on the MountainCar-v0 environment.
    """
    EPSILON = 0
    ALPHA = 0.5 / 8
    GAMMA = 1
    NUM_RUNS = 30
    NUM_EPISODES = 1000

    sarsa_results = []
    q_learning_results = []

    for _ in trange(NUM_RUNS, desc="Sarsa runs"):
        env = gym.make("MountainCar-v0")
        num_actions = env.action_space.n
        sarsa_agent = Sarsa(num_actions, EPSILON, ALPHA, GAMMA)
        rewards = sarsa_agent.run(env, num_episodes=NUM_EPISODES)
        sarsa_results.append(rewards)
        env.close()

    # Save Sarsa results
    df_sarsa = pd.DataFrame(sarsa_results)
    df_sarsa.to_csv("taskA_sarsa_results.csv", index=False)

    # Optionally do the same for Q-learning
    for _ in trange(NUM_RUNS, desc="Q-learning runs"):
        env = gym.make("MountainCar-v0")
        num_actions = env.action_space.n
        q_agent = QLearning(num_actions, EPSILON, ALPHA, GAMMA)
        rewards = q_agent.run(env, num_episodes=NUM_EPISODES)
        q_learning_results.append(rewards)
        env.close()

    # Save Q-learning results
    df_q_learning = pd.DataFrame(q_learning_results)
    df_q_learning.to_csv("taskA_q_learning_results.csv", index=False)
    

if __name__ == "__main__":
    taskA()