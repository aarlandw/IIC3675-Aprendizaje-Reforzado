import matplotlib.pyplot as plt
from env import QuadcopterEnv  
import numpy as np

# Crear el entorno
env = QuadcopterEnv()

# Variables para registrar recompensas
episode_rewards = []

# Ejecutar 500 episodios
for episode in range(500):
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = env.action_space.sample()  # Acción aleatoria
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

    episode_rewards.append(total_reward)


# Graficar
plt.figure(figsize=(12, 6))
plt.plot(episode_rewards, label="Recompensa por episodio", alpha=0.5)
plt.xlabel("Episodio")
plt.ylabel("Recompensa total")
plt.title("Recompensa por episodio (política aleatoria con objetivo fijo)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

env.close()