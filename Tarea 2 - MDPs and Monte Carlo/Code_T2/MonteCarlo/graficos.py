import numpy as np
import matplotlib.pyplot as plt
import csv

file_path = 'cliff_mc_rewards.csv' 
all_rewards = []
with open(file_path, 'r') as f:
    reader = csv.reader(f)
    for row in reader:

        rewards = [float(x) for x in row]
        all_rewards.append(rewards)


num_runs = len(all_rewards)
num_episodes = (len(all_rewards[0]) - 1) * 1000 
episode_points = list(range(0, num_episodes + 1, 1000))

# Graficar
plt.figure(figsize=(12, 6))
for run in range(num_runs):
    plt.plot(episode_points, all_rewards[run], label=f'run {run+1}')
plt.xlabel('Training episodes')
plt.ylabel('Evaluation reward (greedy)')
plt.title('CliffWalking - Every-Visit MC Control (5 runs)')
plt.legend()
plt.grid(True)
plt.show()