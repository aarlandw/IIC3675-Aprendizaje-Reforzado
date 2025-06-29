from Environments.PartiallyObservableEnvs.InvisibleDoorEnv import InvisibleDoorEnv
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv

def q_learning_partial_obs(env, num_episodes, alpha=0.1, gamma=0.99, epsilon=0.01):
    Q = defaultdict(lambda: np.ones(len(env.action_space)))
    episode_lengths = []
    
    for _ in tqdm(range(num_episodes), desc="q_leurning"):
        s = env.reset()
        done = False
        steps = 0
        
        while not done:
            # Selección de acción ε-greedy
            if np.random.rand() < epsilon:
                action = env.action_space[np.random.randint(len(env.action_space))]
            else:
                action = env.action_space[np.argmax(Q[s])]
            
            # Ejecutar acción
            next_s, reward, done = env.step(action)
            steps += 1
            
            # Actualización Q-learning
            best_next = np.max(Q[next_s]) if not done else 0
            Q[s][env.action_space.index(action)] += alpha * (
                reward + gamma * best_next - Q[s][env.action_space.index(action)]
            )
            
            s = next_s
        
        episode_lengths.append(steps)
    
    return episode_lengths

def sarsa_partial_obs(env, num_episodes, alpha=0.1, gamma=0.99, epsilon=0.01):
    """SARSA estándar para observación parcial"""
    Q = defaultdict(lambda: np.ones(len(env.action_space)))
    episode_lengths = []
    
    for _ in tqdm(range(num_episodes), desc="sarsa"):
        obs = env.reset()
        done = False
        steps = 0
        
        # Selección acción inicial
        if np.random.rand() < epsilon:
            action_idx = np.random.randint(len(env.action_space))
        else:
            action_idx = np.argmax(Q[obs])
        
        while not done:
            action = env.action_space[action_idx]
            next_obs, reward, done = env.step(action)
            steps += 1
            
            # Selección siguiente acción
            if not done:
                if np.random.rand() < epsilon:
                    next_action_idx = np.random.randint(len(env.action_space))
                else:
                    next_action_idx = np.argmax(Q[next_obs])
            else:
                next_action_idx = None
            
            # Actualización SARSA
            target = 0 if done else Q[next_obs][next_action_idx]
            Q[obs][action_idx] += alpha * (
                reward + gamma * target - Q[obs][action_idx]
            )
            
            obs = next_obs
            action_idx = next_action_idx
        
        episode_lengths.append(steps)
    
    return episode_lengths

def n_step_sarsa_partial_obs(env, num_episodes, alpha=0.1, gamma=0.99, epsilon=0.01, n=16):
    Q = defaultdict(lambda: np.ones(len(env.action_space)))
    episode_lengths = []
    for _ in tqdm(range(num_episodes), desc="n_step_sarsa"):
        O = [None] * (n + 1) 
        A = [None] * (n + 1)  
        R = [None] * (n + 1) 
        
        # Inicialización
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
                # Ejecutar acción
                action = env.action_space[A[t % (n + 1)]]
                next_obs, r, done = env.step(action)
                O[(t + 1) % (n + 1)] = next_obs
                R[(t + 1) % (n + 1)] = r
                
                if done:
                    T = t + 1
                else:
                    # Selección siguiente acción
                    if np.random.rand() < epsilon:
                        A[(t + 1) % (n + 1)] = np.random.randint(len(env.action_space))
                    else:
                        A[(t + 1) % (n + 1)] = np.argmax(Q[next_obs])
            
            tau = t - n + 1
            if tau >= 0:
                # Calcular retorno n-step
                G = sum(
                    gamma**(i - tau - 1) * R[i % (n + 1)] 
                    for i in range(tau + 1, min(tau + n + 1, T + 1))
                )
                
                if tau + n < T:
                    o_n = O[(tau + n) % (n + 1)]
                    a_n = A[(tau + n) % (n + 1)]
                    G += gamma**n * Q[o_n][a_n]
                
                # Actualizar Q-value
                o_tau = O[tau % (n + 1)]
                a_tau = A[tau % (n + 1)]
                Q[o_tau][a_tau] += alpha * (G - Q[o_tau][a_tau])
            
            if tau == T - 1:
                break
                
            t += 1
        
        episode_lengths.append(t + 1)
    
    return episode_lengths

if __name__ == "__main__":
    # Configuracion del experimento
    env = InvisibleDoorEnv()
    num_runs = 30
    num_episodes = 1000
    alpha = 0.1
    gamma = 0.99
    epsilon = 0.01
    
    # Almacenamiento de resultados
    results = {
        'Q-learning': [],
        'SARSA': [],
        '16-step SARSA': []
    }
    
    # Ejecución de los algoritmos
    for run in tqdm(range(num_runs), desc="Ejecutando experimentos"):
        # Q-learning
        lengths = q_learning_partial_obs(env, num_episodes, alpha, gamma, epsilon)
        results['Q-learning'].append(lengths)
        
        # SARSA
        lengths = sarsa_partial_obs(env, num_episodes, alpha, gamma, epsilon)
        results['SARSA'].append(lengths)
        
        # 16-step SARSA
        lengths = n_step_sarsa_partial_obs(env, num_episodes, alpha, gamma, epsilon, n=16)
        results['16-step SARSA'].append(lengths)
    
    # Procesamiento de resultados
    q_avg = np.mean(results['Q-learning'], axis=0)
    s_avg = np.mean(results['SARSA'], axis=0)
    s16_avg = np.mean(results['16-step SARSA'], axis=0)
    
    # Procesamiento de resultados
    q_avg = np.mean(results['Q-learning'], axis=0)
    s_avg = np.mean(results['SARSA'], axis=0)
    s16_avg = np.mean(results['16-step SARSA'], axis=0)
    
    # Guardar resultados en CSV con el nombre especificado
    filename = "largo_episodios_g1.csv"
    
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['Episodio', 'Q_learning', 'SARSA', '16_step_SARSA']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for i in range(num_episodes):
            writer.writerow({
                'Episodio': i+1,
                'Q_learning': q_avg[i],
                'SARSA': s_avg[i],
                '16_step_SARSA': s16_avg[i]
            })
    
    print(f"\nResultados guardados en: {filename}")
    
    # Visualización (opcional)
    plt.figure(figsize=(12, 6))
    plt.plot(q_avg, label="Q-learning")
    plt.plot(s_avg, label="SARSA")
    plt.plot(s16_avg, label="16-step SARSA")
    plt.xlabel("Episodios")
    plt.ylabel("Largo promedio del episodio")
    plt.title("Comparación de algoritmos en InvisibleDoorEnv (observación parcial)")
    plt.legend()
    plt.grid()
    plt.show()
    # Visualización
    plt.figure(figsize=(12, 6))
    plt.plot(q_avg, label="Q-learning")
    plt.plot(s_avg, label="SARSA")
    plt.plot(s16_avg, label="16-step SARSA")
    plt.xlabel("Episodios")
    plt.ylabel("Largo promedio del episodio")
    plt.title("Comparación de algoritmos en InvisibleDoorEnv (observación parcial)")
    plt.legend()
    plt.grid()
    plt.show()
