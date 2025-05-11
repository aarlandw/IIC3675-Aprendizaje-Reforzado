import numpy as np
from collections import defaultdict
from Environments.MultiAgentEnvs.CentralizedHunterEnv import CentralizedHunterEnv
import matplotlib.pyplot as plt
from tqdm import tqdm

def centralized_qlearning(env, num_episodes, alpha, gamma, epsilon):

    Q = defaultdict(lambda: np.ones(len(env.action_space)))  # Inicialización en 1.0
    action_map = {i: a for i, a in enumerate(env.action_space)}
    episode_lengths = np.zeros(num_episodes // 100)
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        steps = 0
        
        while not done:
            # Selección ε-greedy
            if np.random.rand() < epsilon:
                a_idx = np.random.randint(len(env.action_space))
            else:
                a_idx = np.argmax(Q[state])
            
            action = action_map[a_idx]
            next_state, reward, done = env.step(action)
            steps += 1
            
            # Actualización Q-learning
            best_next = np.max(Q[next_state]) if not done else 0
            Q[state][a_idx] += alpha * (reward + gamma * best_next - Q[state][a_idx])
            
            state = next_state
        
        # Almacenar cada 100 episodios (suavizado exponencial)
        if episode % 100 == 0:
            idx = episode // 100
            if idx == 0:
                episode_lengths[idx] = steps
            else:
                episode_lengths[idx] = 0.99 * episode_lengths[idx-1] + 0.01 * steps
    
    return episode_lengths


if __name__ == "__main__":
    gamma = 0.95
    epsilon = 0.1
    alpha = 0.1
    num_episodes = 50000
    num_runs = 30

    all_lengths = np.zeros((num_runs, num_episodes // 100))


    for run in tqdm(range(num_runs), desc="Runs"):
        env = CentralizedHunterEnv()  # Nuevo entorno por corrida
        lengths = centralized_qlearning(env, num_episodes, alpha, gamma, epsilon)
        all_lengths[run] = lengths

    mean_lengths = np.mean(all_lengths, axis=0)
    std_lengths = np.std(all_lengths, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(mean_lengths, label='Promedio')
    plt.fill_between(range(len(mean_lengths)),
                     mean_lengths - std_lengths,
                     mean_lengths + std_lengths,
                     alpha=0.3, label='±1 std')
    plt.xlabel("Episodios (x100)")
    plt.ylabel("Longitud promedio del episodio")
    plt.title("Q-learning centralizado en CentralizedHunterEnv")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

