import numpy as np
from collections import defaultdict
from Environments.MultiAgentEnvs.CentralizedHunterEnv import CentralizedHunterEnv
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

def decentralized_qlearning(env, num_episodes=50000, alpha=0.1, gamma=0.95, epsilon=0.1, num_runs=30):
    episode_lengths_all_runs = np.zeros((num_runs, num_episodes // 100))
    n_actions = len(env.action_space_single)  # Acciones individuales por cazador

    for run in range(num_runs):
        Q1 = defaultdict(lambda: np.ones(n_actions))  # Q para cazador 1
        Q2 = defaultdict(lambda: np.ones(n_actions))  # Q para cazador 2
        episode_lengths = np.zeros(num_episodes // 100)

        for episode in range(num_episodes):
            state = env.reset()
            done = False
            steps = 0

            while not done:
                state1, state2 = env.get_individual_states(state)

                # ε-greedy para cada agente
                if np.random.rand() < epsilon:
                    a1 = np.random.randint(n_actions)
                else:
                    a1 = np.argmax(Q1[state1])

                if np.random.rand() < epsilon:
                    a2 = np.random.randint(n_actions)
                else:
                    a2 = np.argmax(Q2[state2])

                # Ejecutar acción conjunta
                next_state, reward, done = env.step((a1, a2))
                next_state1, next_state2 = env.get_individual_states(next_state)

                # Q-learning independiente (misma recompensa)
                if not done:
                    Q1[state1][a1] += alpha * (reward + gamma * np.max(Q1[next_state1]) - Q1[state1][a1])
                    Q2[state2][a2] += alpha * (reward + gamma * np.max(Q2[next_state2]) - Q2[state2][a2])
                else:
                    Q1[state1][a1] += alpha * (reward - Q1[state1][a1])
                    Q2[state2][a2] += alpha * (reward - Q2[state2][a2])

                state = next_state
                steps += 1

            # Guardar promedio suavizado cada 100 episodios
            if episode % 100 == 0:
                idx = episode // 100
                if idx == 0:
                    episode_lengths[idx] = steps
                else:
                    episode_lengths[idx] = 0.99 * episode_lengths[idx - 1] + 0.01 * steps

        episode_lengths_all_runs[run] = episode_lengths

    return np.mean(episode_lengths_all_runs, axis=0)


# === MAIN ===
if __name__ == "__main__":
    from hunter_env import HunterEnv  # Asegúrate de importar correctamente tu entorno

    env = HunterEnv()
    avg_lengths = decentralized_qlearning(env)

    # Graficar resultados
    plt.plot(avg_lengths)
    plt.xlabel("x100 episodios")
    plt.ylabel("Longitud promedio del episodio")
    plt.title("Descentralized Q-learning - HunterEnv")
    plt.grid(True)
    plt.show()



if __name__ == "__main__":

    num_runs = 30
    num_episodes = 50000
    episode_lengths_all = np.zeros((num_runs, num_episodes // 100))

    for run in tqdm(range(num_runs), desc="Ejecutando runs"):
        env = HunterEnv()  # Nuevo entorno por cada run
        episode_lengths = run_single_decentralized_experiment(env, num_episodes)
        episode_lengths_all[run] = episode_lengths

    # Promedio sobre las corridas
    avg_lengths = np.mean(episode_lengths_all, axis=0)

    # Graficar resultado
    plt.plot(avg_lengths)
    plt.xlabel("x100 episodios")
    plt.ylabel("Longitud promedio del episodio")
    plt.title("Q-learning descentralizado - HunterEnv")
    plt.grid(True)
    plt.show()