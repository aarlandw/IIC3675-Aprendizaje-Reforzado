from Environments.SimpleEnvs.CliffEnv import CliffEnv
from Environments.SimpleEnvs.EscapeRoomEnv import EscapeRoomEnv
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def show(env, current_state, reward=None):
    env.show()
    print(f"Raw state: {current_state}")
    if reward:
        print(f"Reward: {reward}")


def get_action_from_user(valid_actions):
    key = input()
    while key not in valid_actions:
        key = input()
    return valid_actions[key]


def play_simple_env(simple_env):
    key2action = {'a': "left", 'd': "right", 'w': "up", 's': "down"}
    s = simple_env.reset()
    show(simple_env, s)
    done = False
    while not done:
        print("Action: ", end="")
        action = get_action_from_user(key2action)
        s, r, done = simple_env.step(action)
        show(simple_env, s, r)



def sarsa(env, num_episodes, alpha, gamma, epsilon):

    Q = {}
    returns = []
    n_actions = len(env.action_space)
    action_map = {i: a for i, a in enumerate(env.action_space)}

    for ep in range(num_episodes):
        s = env.reset()
        if s not in Q:
            Q[s] = np.zeros(len(env.action_space))
        if np.random.rand() < epsilon:
            a_idx = np.random.randint(n_actions) 
        else:
            a_idx = np.argmax(Q[s])

        total_reward = 0
        done = False

        while not done:
            action = action_map[a_idx]
            s_next, r, done = env.step(action)
            total_reward += r

            if s_next not in Q:
                Q[s_next] = np.zeros(len(env.action_space))
            if np.random.rand() < epsilon:
                a_idx_next = np.random.randint(n_actions)
            else:
                a_idx_next = np.argmax(Q[s_next])

            Q[s][a_idx] += alpha * (r + gamma * Q[s_next][a_idx_next] - Q[s][a_idx])
            s, a_idx = s_next, a_idx_next

        returns.append(total_reward)

    return returns

def n_step_sarsa(env, num_episodes, alpha, gamma, epsilon, n=4):

    returns = []
    Q = {}
    n_actions = len(env.action_space)
    action_map = {i: a for i, a in enumerate(env.action_space)}

    for ep in range(num_episodes):

        s = env.reset()
        done = False
        total_reward = 0

        S, A, R = [s], [], [0]  # R[0] = 0 por convención
        T = float('inf')
        t = 0

        if s not in Q:
            Q[s] = np.zeros(len(env.action_space))
        if np.random.rand() < epsilon:
            a_idx = np.random.randint(n_actions)
        else:
            a_idx = np.argmax(Q[s])
        A.append(a_idx)

        while not done:
            if t < T:
                action = action_map[A[t]]
                s_next, r, done = env.step(action)
                S.append(s_next)
                R.append(r)
                total_reward += r
                if s_next not in Q:
                    Q[s_next] = np.zeros(len(env.action_space))
                if done:
                    T = t + 1
                else:
                    if np.random.rand() < epsilon:
                        a_idx_next = np.random.randint(n_actions)  
                    else:
                        a_idx_next = np.argmax(Q[s_next])
                    A.append(a_idx_next)

            tau = t - n + 1
            if tau >= 0:
                G = sum([gamma**(i - tau - 1) * R[i] for i in range(tau + 1, min(tau + n , T))])
                if tau + n < T:
                    G += gamma**n * Q[S[tau + n]][A[tau + n]]
                Q[S[tau]][A[tau]] += alpha * (G - Q[S[tau]][A[tau]])

            if tau == T - 1:
                break
            t += 1

        returns.append(total_reward)

    return returns

def q_learning(env, num_episodes, alpha, gamma, epsilon):

    Q = {}
    returns = []
    n_actions = len(env.action_space)
    action_map = {i: a for i, a in enumerate(env.action_space)}

    for ep in range(num_episodes):
        s = env.reset()
        total_reward = 0
        done = False

        while not done:
            if np.random.rand() < epsilon or s not in Q:
                a_idx = np.random.randint(n_actions)
            else:
                a_idx = np.argmax(Q[s])

            action = action_map[a_idx]
            s_next, r, done = env.step(action)
            total_reward += r

            if s not in Q:
                Q[s] = np.zeros(len(env.action_space))
            if s_next not in Q:
                Q[s_next] = np.zeros(len(env.action_space))

            best_next = np.max(Q[s_next])
            Q[s][a_idx] += alpha * (r + gamma * np.max(Q[s_next]) - Q[s][a_idx])

            s = s_next

        returns.append(total_reward)

    return returns

if __name__ == "__main__":
    env = CliffEnv()
    episodes = 500
    runs = 100
    alpha = 0.1
    gamma = 1.0
    epsilon = 0.1

    all_q = []
    all_s = []
    all_s4 = []

    for i in tqdm(range(runs)):
        all_q.append(q_learning(env, episodes, alpha, gamma, epsilon))
        all_s.append(sarsa(env, episodes, alpha, gamma, epsilon))
        all_s4.append(n_step_sarsa(env, episodes, alpha, gamma, epsilon, n=4))

    q_avg = np.mean(all_q, axis=0)
    s_avg = np.mean(all_s, axis=0)
    s4_avg = np.mean(all_s4, axis=0)

    plt.figure(figsize=(10,6))
    plt.plot(q_avg, label="Q-Learning")
    plt.plot(s_avg, label="SARSA")
    plt.plot(s4_avg, label="4-step SARSA")
    plt.ylim(-200, 0)
    plt.xlabel("Episodio")
    plt.ylabel("Retorno promedio")
    plt.title("Comparación en CliffEnv")
    plt.legend()
    plt.grid()
    plt.show()
    # env = EscapeRoomEnv()
    #play_simple_env(env)

