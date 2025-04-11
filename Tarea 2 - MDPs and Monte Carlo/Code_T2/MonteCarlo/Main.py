from Environments.BlackjackEnv import BlackjackEnv
from Environments.CliffEnv import CliffEnv
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import csv


def get_action_from_user(actions):
    print("Valid actions:")
    for i in range(len(actions)):
        print(f"{i}. {actions[i]}")
    print("Please select an action:")
    selected_id = -1
    while not (0 <= selected_id < len(actions)):
        selected_id = int(input())
    return actions[selected_id]


def play(env):
    actions = env.action_space
    state = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        env.show()
        action = get_action_from_user(actions)
        state, reward, done = env.step(action)
        total_reward += reward
    env.show()
    print("Done.")
    print(f"Total reward: {total_reward}")


def play_blackjack():
    env = BlackjackEnv()
    play(env)


def play_cliff():
    cliff_width = 6
    env = CliffEnv(cliff_width)
    play(env)


import numpy as np

def mc_control(env, num_episodes=1e6, gamma=1.0, epsilon=0.1):
    # Inicialización sin defaultdict
    acciones_posibles = env.action_space
    n_actions = len(acciones_posibles)
    Q = {}  # Diccionario para Q-values
    N = {}  # Contador de visitas: N[state][action]
    Returns = {}  # Diccionario para retornos acumulados
    policy = {}  # Diccionario para la política ε-greedy
    eval_rewards = []

    for episode in range(1, int(num_episodes) + 1):
        if episode % 500000 == 0:
            print(f"Episodio {episode}/{num_episodes}")

        # Generar episodio bajo la política actual
        state = env.reset()
        episode_history = []
        done = False

        while not done:
            # Inicializar política para el estado si no existe
            if state not in policy:
                policy[state] = np.ones(n_actions) / n_actions  # Política uniforme inicial
            
            action = acciones_posibles[np.random.choice(n_actions, p=policy[state])]
            next_state, reward, done = env.step(action)
            episode_history.append((state, action, reward))
            state = next_state

      
        G = 0
        for t in reversed(range(len(episode_history))):
            state, action, reward = episode_history[t]
            G = gamma * G + reward
            
            # Inicializar Q y N para (state, action) si no existen
            if state not in Q:
                Q[state] = np.zeros(n_actions)
            if state not in N:
                N[state] = np.zeros(n_actions, dtype=int)
            
            # Actualización incremental
            index_action = env.action_space.index(action)
            N[state][index_action] += 1
            Q[state][index_action] += (G - Q[state][index_action]) / N[state][index_action]
            
            # Mejorar la política ε-greedy
            best_action = np.argmax(Q[state])
            policy[state] = np.ones(n_actions) * epsilon / n_actions
            policy[state][best_action] = 1 - epsilon + epsilon / n_actions
    
        # Evaluar cada 1000 episodios
        if episode % 1000 == 0 or episode == 1:
            total_reward = 0
            eval_state = env.reset()
            eval_done = False
            while not eval_done:
                if eval_state in Q:
                    action_idx = np.argmax(Q[eval_state])
                    action = env.action_space[action_idx]
                else:
                    action = env.action_space.sample()
                eval_state, reward, eval_done = env.step(action)
                total_reward += reward
            eval_rewards.append(total_reward)
    
    return eval_rewards

if __name__ == '__main__':

    # Configuración del experimento CliffWalking
    num_runs = 5
    num_episodes = 200000
    cliff_length = 6
    epsilon = 0.1
    gamma = 1.0


    filename = "cliff_mc_rewards.csv"
    all_rewards = []
    for run in range(num_runs):
        print(f"Run {run+1}/{num_runs}")
        env = CliffEnv()
        rewards = mc_control(env, num_episodes, gamma, epsilon)  # lista de retornos por cada 1000 episodios
        all_rewards.append(rewards)
        # Agrega una fila al archivo CSV con los rewards de esta corrida
        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(rewards)



    # Graficar
    episode_points = list(range(0, num_episodes + 1, 1000))
    plt.figure(figsize=(12, 6))
    for run in range(num_runs):
        plt.plot(episode_points, all_rewards[run], label=f'run {run+1}')
    plt.xlabel('Training episodes')
    plt.ylabel('Evaluation reward (greedy)')
    plt.title('CliffWalking - Every-Visit MC Control (5 runs)')
    plt.legend()
    plt.grid(True)
    plt.show()