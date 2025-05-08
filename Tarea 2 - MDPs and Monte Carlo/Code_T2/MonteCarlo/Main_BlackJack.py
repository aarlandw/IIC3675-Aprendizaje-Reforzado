from Environments.BlackjackEnv import BlackjackEnv
from Environments.CliffEnv import CliffEnv
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import csv
import random
from tqdm import tqdm


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



def mc_control(run, env, num_episodes=1e6, gamma=1.0, epsilon=0.1):
    # Inicialización 
    acciones_posibles = env.action_space
    n_actions = len(acciones_posibles)
    Q = {}  
    N = {}  
    Returns = {}  
    policy = {}  
    eval_rewards = []
    eval_points = list(range(0, int(num_episodes) + 1, 500000))

    for episode in tqdm(range(1, int(num_episodes) + 1), desc="Episodios", unit="episodio"):
        # if episode % 500000 == 0:
            # print(f"Episodio {episode}/{num_episodes}")

        # Generar episodio bajo la política actual
        state = env.reset()
        episode_history = []
        done = False

        while not done:
            # Inicializar política para el estado si no existe
            if state not in policy:
                policy[state] = np.ones(n_actions) / n_actions 
            
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
    
        # Evaluar cada 500k episodios
        if episode in eval_points[1:] or episode == 1:
            test_rewards = []
            for i in range(100000):  
                test_state = env.reset()
                test_done = False
                total_reward = 0
                while not test_done:
                    if test_state in Q:
                        action_idx = np.argmax(Q[test_state])
                        action = env.action_space[action_idx]
                    else:
                        action = random.choice(env.action_space)
                    test_state, reward, test_done = env.step(action)
                    total_reward += reward
                    
                test_rewards.append(total_reward)
            mean_reward = np.mean(test_rewards)
            eval_rewards.append(mean_reward)
            print(f"Run {run} | Episodio {episode}/{num_episodes} | Recompensa media: {mean_reward:.4f}")
    
    return eval_rewards , Q

def show_policy(Q, env):
    actions = env.action_space
    for usable_ace in [True, False]:
        print(f"\n Usable Ace = {usable_ace}")
        for dealer_card in range(1, 11):
            linea = f"Dealer: {dealer_card} | "
            for player_sum in range(12, 22):  
                linea += f"Player sum: {player_sum} |"
                state = (player_sum, usable_ace, dealer_card)
                if state in Q:
                    action = actions[np.argmax(Q[state])]
                else:
                    action = "?"
                linea += f"{action[0]} \n"  
            print(linea)



if __name__ == '__main__':

    # Configuración del experimento CliffWalking
    num_runs = 5
    num_episodes = 10_000_000
    epsilon = 0.01
    gamma = 1.0


    filename = "blackjack_mc_rewards.csv"
    all_rewards = []
    all_run_history_Q = []
    for run in range(num_runs):
        print(f"Run {run+1}/{num_runs}")
        env = BlackjackEnv()
        rewards,Q = mc_control(run+1, env, num_episodes, gamma, epsilon) 
        all_rewards.append(rewards)
        all_run_history_Q.append(Q)

        show_policy(Q, env)
        # Agrega una fila al archivo CSV con los rewards de esta corrida
        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(rewards)
        


    episode_points = list(range(0, num_episodes + 1, 500000))
    plt.figure(figsize=(12, 6))
    for run in range(num_runs):
        plt.plot(episode_points, all_rewards[run], label=f'run {run+1}')
    plt.xlabel('Training episodes')
    plt.ylabel('Evaluation reward (greedy)')
    plt.title('Blackjack - Every-Visit MC Control (5 runs)')
    plt.legend()
    plt.grid(True)
    plt.show()