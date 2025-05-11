import numpy as np
from Environments.MultiGoalEnvs.RoomEnv import RoomEnv
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv

def initialize_q_values(env):
    """Inicializa Q-values en 1.0 como especifica el enunciado"""
    return defaultdict(lambda: np.ones(len(env.action_space)))

def epsilon_greedy_action(Q, state, goal, action_space, epsilon):
    """Selección de acción ε-greedy"""
    n_actions = len(env.action_space)
    if np.random.random() < epsilon:
        return np.random.randint(n_actions)
    else:
        return np.argmax(Q[(state, goal)])

## Versiones estándar (sin multi-goal)

def q_learning(env, alpha=0.1, gamma=0.99, epsilon=0.1, num_episodes=500):
    """Q-learning estándar para multi-objetivo (un solo run)"""
    episode_lengths = np.zeros(num_episodes)
    n_actions = len(env.action_space)
    action_map = {i: a for i, a in enumerate(env.action_space)}
    
    Q = initialize_q_values(env)
    
    for episode in range(num_episodes):
        state = env.reset()
        current_goal = state[1]  # El objetivo está en el estado
        done = False
        steps = 0
        
        while not done:
            a = epsilon_greedy_action(Q, state, current_goal, env.action_space, epsilon)
            action = action_map[a]
            next_state, reward, done = env.step(action)
            next_goal = next_state[1]
            steps += 1
            
            # Actualización Q-learning estándar
            current_q = Q[(state, current_goal)][a]
            max_next_q = max(Q[(next_state, current_goal)]) if not done else 0
            Q[(state, current_goal)][a] = current_q + alpha * (reward + gamma * max_next_q - current_q)
            
            state, current_goal = next_state, next_goal
        
        episode_lengths[episode] = steps

    return episode_lengths

def sarsa(env, alpha=0.1, gamma=0.99, epsilon=0.1, num_episodes=500, n_steps=1):
    """Sarsa estándar (con n-step) para multi-objetivo (un solo run)"""
    episode_lengths = np.zeros(num_episodes)
    n_actions = len(env.action_space)
    action_map = {i: a for i, a in enumerate(env.action_space)}

    Q = initialize_q_values(env)
    
    for episode in range(num_episodes):
        state = env.reset()
        current_goal = state[1]
        done = False
        steps = 0
        history = []
        
        # Selección acción inicial
        a = epsilon_greedy_action(Q, state, current_goal, env.action_space, epsilon)

        while not done:
            action = action_map[a]
            next_state, reward, done = env.step(action)
            next_goal = next_state[1]
            steps += 1
            
            # Guardar experiencia
            history.append(((state, current_goal), a, reward))
            
            # Selección siguiente acción
            next_a = epsilon_greedy_action(Q, next_state, next_goal, env.action_space, epsilon) if not done else None
            
            # Actualización n-step Sarsa
            if len(history) >= n_steps or done:
                tau = len(history) - n_steps
                if tau < 0: tau = 0
                
                # Calcular retorno n-step
                G = sum(gamma**(i - tau) * history[i][2] for i in range(tau, min(tau + n_steps, len(history))))
                
                if tau + n_steps < len(history):
                    s_n, a_n = history[tau + n_steps][0], history[tau + n_steps][1]
                    G += gamma**n_steps * Q[s_n][a_n]
                
                # Actualizar Q-value
                s_tau, a_tau = history[tau][0], history[tau][1]
                Q[s_tau][a_tau] += alpha * (G - Q[s_tau][a_tau])
                
                # Eliminar experiencia antigua
                if len(history) > n_steps:
                    history.pop(0)
            
            state, current_goal, a = next_state, next_goal, next_a
        
        episode_lengths[episode] = steps

    return episode_lengths

## Versiones multi-goal

def multi_goal_q_learning(env, alpha=0.1, gamma=0.99, epsilon=0.1, num_episodes=500):
    """Q-learning con actualización multi-goal (solo un run)"""
    episode_lengths = np.zeros(num_episodes)
    n_actions = len(env.action_space)
    action_map = {i: a for i, a in enumerate(env.action_space)}  

    Q = initialize_q_values(env)
    
    for episode in range(num_episodes):
        state = env.reset()
        current_goal = state[1]
        done = False
        steps = 0
        
        while not done:
            a = epsilon_greedy_action(Q, state, current_goal, env.action_space, epsilon)
            action = action_map[a]
            next_state, reward, done = env.step(action)
            next_goal = next_state[1]
            steps += 1
            
            # Actualización multi-goal: para todos los objetivos posibles
            for goal in env.goals:
                current_q = Q[(state, goal)][a]
                max_next_q = max(Q[(next_state, goal)]) if not done else 0
                Q[(state, goal)][a] = current_q + alpha * (reward + gamma * max_next_q - current_q)
            
            state, current_goal = next_state, next_goal
        
        episode_lengths[episode] = steps

    return episode_lengths

def multi_goal_sarsa(env, alpha=0.1, gamma=0.99, epsilon=0.1, num_episodes=500):
    """Sarsa con actualización multi-goal para un solo run"""
    episode_lengths = np.zeros(num_episodes)
    action_list = list(env.action_space)
    action_size = len(action_list)

    Q = defaultdict(lambda: np.ones(action_size))
    
    for episode in range(num_episodes):
        state = env.reset()
        current_goal = state[1]
        done = False
        steps = 0

        # Selección acción inicial
        if np.random.random() < epsilon:
            current_action_idx = np.random.randint(action_size)
        else:
            current_action_idx = np.argmax(Q[(state, current_goal)])

        while not done:
            action = action_list[current_action_idx]
            next_state, reward, done = env.step(action)
            next_goal = next_state[1]
            steps += 1

            if not done:
                if np.random.random() < epsilon:
                    next_action_idx = np.random.randint(action_size)
                else:
                    next_action_idx = np.argmax(Q[(next_state, next_goal)])
            else:
                next_action_idx = None

            for goal in env.goals:
                if goal == current_goal:
                    target_action_idx = next_action_idx
                else:
                    if np.random.random() < epsilon:
                        target_action_idx = np.random.randint(action_size)
                    else:
                        target_action_idx = np.argmax(Q[(next_state, goal)])

                target_q = 0 if done else Q[(next_state, goal)][target_action_idx]

                Q[(state, goal)][current_action_idx] += alpha * (
                    reward + gamma * target_q - Q[(state, goal)][current_action_idx]
                )

            state, current_goal, current_action_idx = next_state, next_goal, next_action_idx

        episode_lengths[episode] = steps

    return episode_lengths

if __name__ == "__main__":
    env = RoomEnv()
    params = {
        'alpha': 0.1,
        'gamma': 0.99,
        'epsilon': 0.1,
        'num_episodes': 500
    }
     
    
    all_q = []
    all_s = []
    all_s8 = []
    all_qmult = []
    all_smult = []

    print("Ejecutando experimentos...")
    runs = 100
    for i in tqdm(range(runs)):
        all_q.append( q_learning(env, **params))
        all_s.append( sarsa(env, n_steps=1, **params))
        all_s8.append(sarsa(env, n_steps=8, **params))
        all_qmult.append( multi_goal_q_learning(env, **params))
        all_smult.append( multi_goal_sarsa(env, **params))

    q_learning_result = np.mean(all_q, axis=0)
    sarsa_1_result = np.mean(all_s, axis=0)
    sarsa_8_result = np.mean(all_s8, axis=0)
    multi_q_result = np.mean(all_qmult, axis=0)
    multi_sarsa_result = np.mean(all_smult, axis=0)



    # Guardar resultados en CSV
    with open("largo_episodio_e.csv", "w", newline="") as f:
        writer = csv.writer(f)
        # Escribir encabezado
        writer.writerow(["Episodio", "Q-learning", "Sarsa (1-step)", "Sarsa (8-step)", "Multi-goal Q-learning", "Multi-goal Sarsa"])
        
        # Escribir los valores por episodio
        for i in range(params['num_episodes']):
            writer.writerow([
                i + 1,
                q_learning_result[i],
                sarsa_1_result[i],
                sarsa_8_result[i],
                multi_q_result[i],
                multi_sarsa_result[i]
            ])

    # Graficar resultados
    plt.figure(figsize=(12, 6))
    plt.plot(q_learning_result, label='Q-learning')
    plt.plot(sarsa_1_result, label='Sarsa (1-step)')
    plt.plot(sarsa_8_result, label='Sarsa (8-step)')
    plt.plot(multi_q_result, label='Multi-goal Q-learning')
    plt.plot(multi_sarsa_result, label='Multi-goal Sarsa')

    plt.xlabel('Episodes')
    plt.ylabel('Average episode length')
    plt.title('RoomEnv episode length comparison (Multi-goal)')
    plt.legend()
    plt.grid()
    plt.show()

