from Environments.PartiallyObservableEnvs.InvisibleDoorEnv import InvisibleDoorEnv
from MemoryWrappers.KOrderMemory import KOrderMemory
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv


def q_learning_with_memory(env, num_episodes, alpha, gamma, epsilon):
    Q = defaultdict(lambda: np.ones(len(env.action_space)))
    episode_lengths = []

    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        steps = 0

        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space[np.random.randint(len(env.action_space))]
            else:
                action = env.action_space[np.argmax(Q[obs])]

            next_obs, reward, done = env.step(action)
            steps += 1

            best_next = np.max(Q[next_obs]) if not done else 0
            Q[obs][env.action_space.index(action)] += alpha * (
                reward + gamma * best_next - Q[obs][env.action_space.index(action)])

            obs = next_obs

        episode_lengths.append(steps)

    return episode_lengths


def sarsa_with_memory(env, num_episodes, alpha, gamma, epsilon):
    Q = defaultdict(lambda: np.ones(len(env.action_space)))
    episode_lengths = []

    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        steps = 0

        if np.random.rand() < epsilon:
            action_idx = np.random.randint(len(env.action_space))
        else:
            action_idx = np.argmax(Q[obs])

        while not done:
            action = env.action_space[action_idx]
            next_obs, reward, done = env.step(action)
            steps += 1

            if not done:
                if np.random.rand() < epsilon:
                    next_action_idx = np.random.randint(len(env.action_space))
                else:
                    next_action_idx = np.argmax(Q[next_obs])
            else:
                next_action_idx = None

            target = 0 if done else Q[next_obs][next_action_idx]
            Q[obs][action_idx] += alpha * (reward + gamma * target - Q[obs][action_idx])

            obs = next_obs
            action_idx = next_action_idx

        episode_lengths.append(steps)

    return episode_lengths


def n_step_sarsa_with_memory(env, num_episodes, alpha, gamma, epsilon, n=16):
    Q = defaultdict(lambda: np.ones(len(env.action_space)))
    episode_lengths = []

    for _ in range(num_episodes):
        O = [None] * (n + 1)
        A = [None] * (n + 1)
        R = [None] * (n + 1)

        obs = env.reset()
        O[0] = obs
        if np.random.rand() < epsilon:
            A[0] = np.random.randint(len(env.action_space))
        else:
            A[0] = np.argmax(Q[obs])

        T = float('inf')
        t = 0

        while True:
            if t < T:
                action = env.action_space[A[t % (n + 1)]]
                next_obs, r, done = env.step(action)

                O[(t + 1) % (n + 1)] = next_obs
                R[(t + 1) % (n + 1)] = r

                if done:
                    T = t + 1
                else:
                    if np.random.rand() < epsilon:
                        A[(t + 1) % (n + 1)] = np.random.randint(len(env.action_space))
                    else:
                        A[(t + 1) % (n + 1)] = np.argmax(Q[next_obs])

            tau = t - n + 1
            if tau >= 0:
                G = sum(gamma**(i - tau - 1) * R[i % (n + 1)]
                        for i in range(tau + 1, min(tau + n + 1, T + 1)))

                if tau + n < T:
                    o_n = O[(tau + n) % (n + 1)]
                    a_n = A[(tau + n) % (n + 1)]
                    G += gamma**n * Q[o_n][a_n]

                o_tau = O[tau % (n + 1)]
                a_tau = A[tau % (n + 1)]
                Q[o_tau][a_tau] += alpha * (G - Q[o_tau][a_tau])

            if tau == T - 1:
                break

            t += 1

        episode_lengths.append(t + 1)

    return episode_lengths


if __name__ == "__main__":
    memory_size = 2
    env = InvisibleDoorEnv()
    env = KOrderMemory(env, memory_size)

    params = {
        'num_runs': 30,
        'num_episodes': 1000,
        'alpha': 0.1,
        'gamma': 0.99,
        'epsilon': 0.01
    }

    results = {
        'Q-learning': [],
        'SARSA': [],
        '16-step SARSA': []
    }

    for run in tqdm(range(params['num_runs']), desc="Ejecutando experimentos con memoria"):
        results['Q-learning'].append(q_learning_with_memory(
            env, params['num_episodes'], params['alpha'], params['gamma'], params['epsilon']))

        results['SARSA'].append(sarsa_with_memory(
            env, params['num_episodes'], params['alpha'], params['gamma'], params['epsilon']))

        results['16-step SARSA'].append(n_step_sarsa_with_memory(
            env, params['num_episodes'], params['alpha'], params['gamma'], params['epsilon'], n=16))

    q_avg = np.mean(results['Q-learning'], axis=0)
    s_avg = np.mean(results['SARSA'], axis=0)
    s16_avg = np.mean(results['16-step SARSA'], axis=0)

    with open(f'resultados_memoria_k{memory_size}_g1.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Episodio', 'Q_learning', 'SARSA', '16_step_SARSA'])
        for i in range(params['num_episodes']):
            writer.writerow([i + 1, q_avg[i], s_avg[i], s16_avg[i]])

    plt.figure(figsize=(12, 6))
    plt.plot(q_avg, label='Q-learning')
    plt.plot(s_avg, label='SARSA')
    plt.plot(s16_avg, label='16-step SARSA')
    plt.xlabel('Episodios')
    plt.ylabel('Largo promedio del episodio')
    plt.title(f'Comparación con {memory_size}-order memory')
    plt.legend()
    plt.grid()
    plt.show()
