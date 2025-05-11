import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import csv

from Environments.MultiAgentEnvs.HunterEnv import HunterEnv
from Environments.MultiAgentEnvs.CentralizedHunterEnv import CentralizedHunterEnv
from Environments.MultiAgentEnvs.HunterAndPreyEnv import HunterAndPreyEnv

# --- Descentralizado Cooperativo ---
def decentralized_qlearning(env, num_episodes=50000, alpha=0.1, gamma=0.95, epsilon=0.1, num_runs=30):
    episode_lengths_all_runs = np.zeros((num_runs, num_episodes // 100))
    n_actions = len(env.single_agent_action_space)
    action_map = {i: a for i, a in enumerate(env.single_agent_action_space)}

    for run in tqdm(range(num_runs), desc="Runs descentralizado"):
        Q1 = defaultdict(lambda: np.ones(n_actions))  # Q-table para cazador 1
        Q2 = defaultdict(lambda: np.ones(n_actions))  # Q-table para cazador 2
        episode_lengths = np.zeros(num_episodes // 100)
        
        for episode in tqdm(range(num_episodes), desc="Runs episodes descentralizado"):
            state = env.reset()
            done = False
            steps = 0

            while not done:
                # Elegir acción para cada agente (ε-greedy)
                a1 = np.random.randint(n_actions) if np.random.rand() < epsilon else np.argmax(Q1[state])
                a2 = np.random.randint(n_actions) if np.random.rand() < epsilon else np.argmax(Q2[state])

                # Ejecutar acción conjunta
                action1 = action_map[a1]
                action2 = action_map[a2]

                next_state, rewards, done = env.step((action1, action2))
                r1, r2 = rewards  # recompensas individuales

                # Actualización Q-learning
                if not done:
                    Q1[state][a1] += alpha * (r1 + gamma * np.max(Q1[next_state]) - Q1[state][a1])
                    Q2[state][a2] += alpha * (r2 + gamma * np.max(Q2[next_state]) - Q2[state][a2])
                else:
                    Q1[state][a1] += alpha * (r1 - Q1[state][a1])
                    Q2[state][a2] += alpha * (r2 - Q2[state][a2])

                state = next_state
                steps += 1

            # Suavizado exponencial cada 100 episodios
            if episode % 100 == 0:
                idx = episode // 100
                episode_lengths[idx] = steps if idx == 0 else 0.99 * episode_lengths[idx - 1] + 0.01 * steps

        episode_lengths_all_runs[run] = episode_lengths

    return np.mean(episode_lengths_all_runs, axis=0)

# --- Centralizado ---
def centralized_qlearning(env, num_episodes, alpha, gamma, epsilon, num_runs):
    all_lengths = np.zeros((num_runs, num_episodes // 100))
    action_map = {i: a for i, a in enumerate(env.action_space)}

    for run in tqdm(range(num_runs), desc="Runs centralizado"):
        Q = defaultdict(lambda: np.ones(len(env.action_space)))
        episode_lengths = np.zeros(num_episodes // 100)

        for episode in tqdm(range(num_episodes), desc="Runs episodes centralizado"):
            state = env.reset()
            done = False
            steps = 0

            while not done:
                a_idx = np.random.randint(len(env.action_space)) if np.random.rand() < epsilon else np.argmax(Q[state])
                action = action_map[a_idx]
                next_state, reward, done = env.step(action)

                best_next = np.max(Q[next_state]) if not done else 0
                Q[state][a_idx] += alpha * (reward + gamma * best_next - Q[state][a_idx])
                state = next_state
                steps += 1

            if episode % 100 == 0:
                idx = episode // 100
                episode_lengths[idx] = steps if idx == 0 else 0.99 * episode_lengths[idx - 1] + 0.01 * steps

        all_lengths[run] = episode_lengths

    return np.mean(all_lengths, axis=0)

# --- Descentralizado Competitivo ---
def competitive_qlearning(env, num_episodes=50000, alpha=0.1, gamma=0.95, epsilon=0.1, num_runs=30):
    episode_lengths_all_runs = np.zeros((num_runs, num_episodes // 100))
    n_actions = len(env.single_agent_action_space)
    action_map = {i: a for i, a in enumerate(env.single_agent_action_space)}

    for run in tqdm(range(num_runs), desc="Runs competitivo"):
        Q1 = defaultdict(lambda: np.ones(n_actions))  # Q-table cazador 1
        Q2 = defaultdict(lambda: np.ones(n_actions))  # Q-table cazador 2
        Q3 = defaultdict(lambda: np.ones(n_actions))  # Q-table presa
        episode_lengths = np.zeros(num_episodes // 100)

        for episode in tqdm(range(num_episodes), desc="Runs episodes competitivo"):
            state = env.reset()
            done = False
            steps = 0

            while not done:
                # Política ε-greedy para cada agente
                a1 = np.random.randint(n_actions) if np.random.rand() < epsilon else np.argmax(Q1[state])
                a2 = np.random.randint(n_actions) if np.random.rand() < epsilon else np.argmax(Q2[state])
                a3 = np.random.randint(n_actions) if np.random.rand() < epsilon else np.argmax(Q3[state])
                
                action1 = action_map[a1]
                action2 = action_map[a2]
                action3 = action_map[a3]
                next_state, rewards, done = env.step((action1, action2, action3))
                r1, r2, r3 = rewards

                # Actualización Q para cada agente
                if not done:
                    Q1[state][a1] += alpha * (r1 + gamma * np.max(Q1[next_state]) - Q1[state][a1])
                    Q2[state][a2] += alpha * (r2 + gamma * np.max(Q2[next_state]) - Q2[state][a2])
                    Q3[state][a3] += alpha * (r3 + gamma * np.max(Q3[next_state]) - Q3[state][a3])
                else:
                    Q1[state][a1] += alpha * (r1 - Q1[state][a1])
                    Q2[state][a2] += alpha * (r2 - Q2[state][a2])
                    Q3[state][a3] += alpha * (r3 - Q3[state][a3])

                state = next_state
                steps += 1

            # Suavizado exponencial cada 100 episodios
            if episode % 100 == 0:
                idx = episode // 100
                episode_lengths[idx] = steps if idx == 0 else 0.99 * episode_lengths[idx - 1] + 0.01 * steps

        episode_lengths_all_runs[run] = episode_lengths

    return np.mean(episode_lengths_all_runs, axis=0)

# === MAIN ===
if __name__ == "__main__":
    num_episodes = 50000
    alpha = 0.1
    gamma = 0.95
    epsilon = 0.1
    num_runs = 30

    decentralized_env = HunterEnv()
    centralized_env = CentralizedHunterEnv()
    competitive_env = HunterAndPreyEnv()

    avg_centralized = centralized_qlearning(centralized_env, num_episodes, alpha, gamma, epsilon, num_runs)
    avg_decentralized = decentralized_qlearning(decentralized_env, num_episodes, alpha, gamma, epsilon, num_runs)
    avg_competitive = competitive_qlearning(competitive_env, num_episodes, alpha, gamma, epsilon, num_runs)
    


    # Guardar resultados
    with open("largo_episodio_f.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Episode (x100)", "Decentralized", "Centralized", "Competitive"])
        for i, (d, c, comp) in enumerate(zip(avg_decentralized, avg_centralized, avg_competitive)):
            writer.writerow([i, d, c, comp])

    # Graficar
    plt.figure(figsize=(10, 6))
    plt.plot(avg_decentralized, label=" Decentralized Q-learning (HunterEnv)")
    plt.plot(avg_centralized, label="Centralized Q-learning (CentralizedHunterEnv)")
    plt.plot(avg_competitive, label="Competitive Q-learning (HunterAndPreyEnv)")
    plt.xlabel("Episodes (x100)")
    plt.ylabel("Average Episode Length")
    plt.title("Comparison of episode length in three multi-agent settings")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()