from Environments.SimpleEnvs.CliffEnv import CliffEnv
from Environments.SimpleEnvs.EscapeRoomEnv import EscapeRoomEnv
import numpy as np
from tqdm import tqdm, trange
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

def run_value_iteration(Np, Nt, states, actions, Rmax, k, gamma, theta=1e-5):
    V = defaultdict(float)
    while True:
        delta = 0
        for s in states:
            v = V[s]
            max_q = float('-inf')
            for a in actions:
                if Nt[(s, a)] < k:
                    q = Rmax + gamma * V["terminal"]
                else:
                    q = 0
                    for (s_next, r), count in Np[(s, a)].items():
                        p = count / Nt[(s, a)]
                        q += p * (r + gamma * V[s_next])
                if q > max_q:
                    max_q = q
            V[s] = max_q
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V

# Algoritmo Rmax principal
def rmax(env, Rmax=1.0, k=2, gamma=1.0, episodes=20, runs=5):
    actions = env.action_space
    all_returns = np.zeros((runs, episodes))

    for run in range(runs):
        Nt = defaultdict(int)
        Np = defaultdict(lambda: defaultdict(int))
        states = set()

        for ep in range(episodes):
            s = tuple(env.reset())
            done = False
            total_reward = 0

            while not done:
                states.add(s)
                V = run_value_iteration(Np, Nt, states, actions, Rmax, k, gamma)
                best_q = float('-inf')
                for a in actions:
                    if Nt[(s, a)] < k:
                        q = Rmax
                    else:
                        q = 0
                        for (s_next, r), count in Np[(s, a)].items():
                            p = count / Nt[(s, a)]
                            q += p * (r + gamma * V[s_next])
                    if q > best_q:
                        best_q = q
                        best_a = a
                a = best_a
                s_next, r, done = env.step(a)
                s_next = tuple(s_next)
                total_reward += r
                Nt[(s, a)] += 1
                Np[(s, a)][(s_next, r)] += 1
                s = s_next

            all_returns[run, ep] = total_reward

    return all_returns

# Algoritmo Dyna-Q
def dyna_q(env, alpha=0.5, gamma=1.0, epsilon=0.1, planning_steps=0, episodes=20, runs=5):
    actions = env.action_space
    all_returns = np.zeros((runs, episodes))

    for run in range(runs):
        Q = defaultdict(lambda: np.zeros(len(actions)))
        model = dict()

        n_actions = len(env.action_space)
        action_map = {i: a for i, a in enumerate(env.action_space)}

        for ep in range(episodes):
            s = tuple(env.reset())
            done = False
            total_reward = 0
            avance = 0
            while not done:
                avance += 1
                if planning_steps == 10000 and avance%100==0:
                    print(f"transcurieron {avance}")
                
                if random.random() < epsilon:
                    a_idx = np.random.randint(n_actions)
                else:
                    a_idx = np.argmax(Q[s])
                    
                a = action_map[a_idx]
                s_next, r, done = env.step(a)
                s_next = tuple(s_next)
                total_reward += r

                Q[s][a_idx] += alpha * (r + gamma * np.max(Q[s_next]) - Q[s][a_idx])
                model[(s, a_idx)] = (s_next, r)

                for i in range(planning_steps):
                    s_p, a_p = random.choice(list(model.keys()))
                    s_next_p, r_p = model[(s_p, a_p)]
                    Q[s_p][a_p] += alpha * (r_p + gamma * np.max(Q[s_next_p]) - Q[s_p][a_p])

                s = s_next

            all_returns[run, ep] = total_reward

    return Q, all_returns
  

def taskC():
    env = CliffEnv()
    episodes = 500
    runs = 100
    alpha = 0.1
    gamma = 1.0
    epsilon = 0.1

    all_q = []
    all_s = []
    all_s4 = []

    for i in trange(runs):
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

def taskD():
    env = EscapeRoomEnv()
    episodes = 20
    runs = 5
    planning_steps_list = [0, 1, 10, 100, 1000, 10000]

    dyna_returns = {n: [] for n in planning_steps_list}
    rmax_returns = []

    for i in tqdm(range(runs)):
        rmax_ret = rmax(env, episodes=episodes, Rmax=1.0, k=1, gamma=1.0)
        rmax_returns.append(np.mean(rmax_ret))
        for n in planning_steps_list:
            Q, returns = dyna_q(env, alpha=0.5, gamma=1.0, epsilon=0.1, planning_steps=n, episodes=episodes, runs=1)
            dyna_returns[n].append(np.mean(returns))
            print(f"dyna planning_steps = {n} : listo")
        print("finalizado dyna")
        

    print("\nRetorno medio por episodio:")
    print("Método\t\tSteps de planeamiento\tRetorno promedio")
    for n in planning_steps_list:
        print(f"Dyna-Q\t\t{n}\t\t\t{np.mean(dyna_returns[n]):.2f}")
    print(f"RMax\t\t-\t\t\t{np.mean(rmax_returns):.2f}")

if __name__ == "__main__":
    # taskC()
    taskD()