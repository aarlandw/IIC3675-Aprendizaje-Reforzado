import numpy as np
from Environments.MultiGoalEnvs.RoomEnv import RoomEnv
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm, trangera
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
        current_goal = state[1] 
        done = False
        steps = 0
        
        while not done:
            a = epsilon_greedy_action(Q, state, current_goal, env.action_space, epsilon)
            action = action_map[a]
            next_state, reward, done = env.step(action)
            next_goal = next_state[1]
            steps += 1
            
            
            current_q = Q[(state, current_goal)][a]
            max_next_q = max(Q[(next_state, current_goal)]) if not done else 0
            Q[(state, current_goal)][a] = current_q + alpha * (reward + gamma * max_next_q - current_q)
            
            state, current_goal = next_state, next_goal
        
        episode_lengths[episode] = steps

    return episode_lengths
def sarsa(env, num_episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    """
    SARSA estándar que solo actualiza los Q-values para el goal actual del episodio.
    No realiza actualizaciones multi-goal.
    """
    Q = defaultdict(lambda: np.ones(len(env.action_space)))  # Inicialización con 1s
    episode_lengths = []
    action_list = list(env.action_space)
    
    for episode in range(num_episodes):
        # Inicialización del episodio
        state = env.reset()
        current_goal = state[1]  
        current_state = (state, current_goal)
        
 
        if np.random.rand() < epsilon:
            action_idx = np.random.randint(len(action_list))
        else:
            action_idx = np.argmax(Q[current_state])
        
        done = False
        steps = 0
        
        while not done:
            # Ejecutar acción
            action = action_list[action_idx]
            next_state, reward, done = env.step(action)
            next_goal = next_state[1]
            next_state_tuple = (next_state, next_goal)
            steps += 1
            
    
            if not done:
                if np.random.rand() < epsilon:
                    next_action_idx = np.random.randint(len(action_list))
                else:
                    next_action_idx = np.argmax(Q[next_state_tuple])
            else:
                next_action_idx = None
            
            # Actualización SARSA solo para el goal actual
            if not done:
                
                Q[current_state][action_idx] += alpha * (
                    reward + gamma * Q[next_state_tuple][next_action_idx] - Q[current_state][action_idx]
                )
            else:
             
                Q[current_state][action_idx] += alpha * (reward - Q[current_state][action_idx])
            
            # Transición al siguiente estado
            current_state = next_state_tuple
            action_idx = next_action_idx
        
        episode_lengths.append(steps)
    
    return np.array(episode_lengths)

def n_step_sarsa(env, num_episodes, alpha, gamma, epsilon, n=8):
    Q = defaultdict(lambda: np.ones(len(env.action_space)))  
    episode_lengths = []
    
    for episode in range(num_episodes):
        # Buffers circulares para n-step
        S = [None] * (n + 1)  
        A = [None] * (n + 1) 
        R = [None] * (n + 1)  
        # Inicialización del episodio
        state = env.reset()
        current_goal = state[1]  
        S[0] = (state, current_goal)
        

        if np.random.rand() < epsilon:
            A[0] = np.random.randint(len(env.action_space))
        else:
            A[0] = np.argmax(Q[S[0]])
        
        T = float('inf')  
        t = 0
        steps = 0
        
        while True:
            if t < T:
                # Ejecutar acción
                action = env.action_space[A[t % (n + 1)]]
                next_state, reward, done = env.step(action)
                next_goal = next_state[1]
                
            
                S[(t + 1) % (n + 1)] = (next_state, next_goal)
                R[(t + 1) % (n + 1)] = reward
                steps += 1
                
                if done:
                    T = t + 1
                else:
                   
                    if np.random.rand() < epsilon:
                        A[(t + 1) % (n + 1)] = np.random.randint(len(env.action_space))
                    else:
                        A[(t + 1) % (n + 1)] = np.argmax(Q[S[(t + 1) % (n + 1)]])
            
            tau = t - n + 1 
            
            if tau >= 0:
                # Calcular retorno n-step
                G = sum(
                    gamma**(i - tau - 1) * R[i % (n + 1)] 
                    for i in range(tau + 1, min(tau + n + 1, T + 1))
                )
                
        
                if tau + n < T:
                    s_n = S[(tau + n) % (n + 1)]
                    a_n = A[(tau + n) % (n + 1)]
                    G += gamma**n * Q[s_n][a_n]
                
                # Actualizar Q-value solo para el goal actual
                s_tau = S[tau % (n + 1)]
                a_tau = A[tau % (n + 1)]
                Q[s_tau][a_tau] += alpha * (G - Q[s_tau][a_tau])
            
            if tau == T - 1:
                break
                
            t += 1
        
        episode_lengths.append(steps)
    
    return np.array(episode_lengths)
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
    """SARSA que maneja múltiples goals, actualizando Q-values para todos los goals en cada paso"""
    Q = defaultdict(lambda: np.ones(len(env.action_space)))  # Inicialización con 1s
    episode_lengths = []
    n_actions = len(env.action_space)
    action_map = {i: a for i, a in enumerate(env.action_space)}
    
    for episode in range(num_episodes):
        state = env.reset()
        current_goal = state[1]  # Asumimos que el goal está en state[1]
        done = False
        steps = 0
        
        # Selección acción inicial ε-greedy para el goal actual
        if np.random.rand() < epsilon:
            a_idx = np.random.randint(n_actions)
        else:
            a_idx = np.argmax(Q[(state, current_goal)])
        
        while not done:
            action = action_map[a_idx]
            next_state, reward, done = env.step(action)
            next_goal = next_state[1]
            steps += 1
            
            # Selección siguiente acción ε-greedy para el nuevo goal
            if not done:
                if np.random.rand() < epsilon:
                    a_idx_next = np.random.randint(n_actions)
                else:
                    a_idx_next = np.argmax(Q[(next_state, next_goal)])
            else:
                a_idx_next = None
            
            # Actualización para TODOS los goals (multi-goal)
            for goal in env.goals:
                if goal == current_goal:
                    target_action = a_idx_next
                else:
                    # Para otros goals, seleccionamos acción ε-greedy
                    if np.random.rand() < epsilon:
                        target_action = np.random.randint(n_actions)
                    else:
                        target_action = np.argmax(Q[(next_state, goal)])
                
                # Calcular target Q-value
                if done:
                    target_q = 0 
                else:
                    target_q = Q[(next_state, goal)][target_action]
                
                # Actualizar Q-value para este goal
                Q[(state, goal)][a_idx] += alpha * (
                    reward + gamma * target_q - Q[(state, goal)][a_idx]
                )
            
            # Avanzar al siguiente estado
            state, current_goal, a_idx = next_state, next_goal, a_idx_next
        
        episode_lengths.append(steps)
    
    return np.array(episode_lengths)



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
    for i in trange(runs):
        all_s.append(sarsa(env, **params))
        all_q.append( q_learning(env, **params))
        all_s8.append(n_step_sarsa(env, n=8, **params))
        all_qmult.append( multi_goal_q_learning(env, **params))
        all_smult.append( multi_goal_sarsa(env, **params))

    q_learning_result = np.mean(all_q, axis=0)
    sarsa_1_result = np.mean(all_s, axis=0)
    sarsa_8_result = np.mean(all_s8, axis=0)
    multi_q_result = np.mean(all_qmult, axis=0)
    multi_sarsa_result = np.mean(all_smult, axis=0)



    # Guardar resultados en CSV
    with open("largo_episodio_e2.csv", "w", newline="") as f:
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