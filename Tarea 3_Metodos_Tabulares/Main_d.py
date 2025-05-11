from Environments.SimpleEnvs.CliffEnv import CliffEnv
from Environments.SimpleEnvs.EscapeRoomEnv import EscapeRoomEnv
import numpy as np
import random
from tqdm import tqdm
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
            loop = 0
            while not done:
                loop += 1
                if planning_steps == 10000 and loop%100==0:
                    print(f"loop numbre {loop}")
                
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


if __name__ == "__main__":
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