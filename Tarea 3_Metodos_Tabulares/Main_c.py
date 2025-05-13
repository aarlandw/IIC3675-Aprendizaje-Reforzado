from Environments.SimpleEnvs.CliffEnv import CliffEnv
from Environments.SimpleEnvs.EscapeRoomEnv import EscapeRoomEnv
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

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
    Q = defaultdict(lambda: np.zeros(len(env.action_space)))
    returns = []
    
    for ep in range(num_episodes):
        # Buffers circulares (tamaño n+1)
        S = [None] * (n + 1)
        A = [None] * (n + 1)
        R = [None] * (n + 1)
        
        # Inicialización del episodio
        s = env.reset()
        S[0] = s
        if np.random.rand() < epsilon:
            A[0] = np.random.randint(len(env.action_space))
        else:
            A[0] = np.argmax(Q[s])
        
        T = float('inf')
        t = 0
        total_reward = 0
        
        while True:
            if t < T:
                # Ejecutar acción A[t % (n+1)]
                action = env.action_space[A[t % (n + 1)]]
                s_next, r, done = env.step(action)
                total_reward += r
                
             
                S[(t + 1) % (n + 1)] = s_next
                R[(t + 1) % (n + 1)] = r
                
                if done:
                    T = t + 1
                else:
    
                    if np.random.rand() < epsilon:
                        A[(t + 1) % (n + 1)] = np.random.randint(len(env.action_space))
                    else:
                        A[(t + 1) % (n + 1)] = np.argmax(Q[s_next])
            
            tau = t - n + 1
            if tau >= 0:
                # Calcular retorno n-step
                G = 0
                for i in range(tau + 1, min(tau + n + 1, T + 1)):
                    G += gamma**(i - tau - 1) * R[i % (n + 1)]
                
                if tau + n < T:
                    s_tau_n = S[(tau + n) % (n + 1)]
                    a_tau_n = A[(tau + n) % (n + 1)]
                    G += gamma**n * Q[s_tau_n][a_tau_n]
                
                # Actualizar Q-value
                s_tau = S[tau % (n + 1)]
                a_tau = A[tau % (n + 1)]
                Q[s_tau][a_tau] += alpha * (G - Q[s_tau][a_tau])
            
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

