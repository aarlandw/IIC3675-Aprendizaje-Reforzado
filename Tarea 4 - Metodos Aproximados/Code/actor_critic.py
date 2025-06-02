import gymnasium as gym
import numpy as np
from FeatureExtractor import FeatureExtractor  
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

class LinearValueFunction:
    def __init__(self, num_features, alpha=0.001):
        self.weights = np.zeros(num_features)
        self.alpha = alpha

    def value(self, features):
        return np.dot(self.weights, features)

    def update(self, features, target):
        error = target - self.value(features)
        self.weights += self.alpha * error * features

class GaussianPolicy:
    def __init__(self, num_features, alpha=0.0001):
        self.mean_weights = np.zeros(num_features)
        self.log_std = np.zeros(1)
        self.alpha = alpha

    def get_action(self, features):
        mean = np.dot(self.mean_weights, features)
        std = np.exp(self.log_std)
        action = np.random.normal(mean, std)
        return np.clip(action, -1, 1), mean, std

    def update(self, features, action, mean, std, advantage):
        grad_log = (action - mean) / (std**2) * features
        self.mean_weights += self.alpha * advantage * grad_log

def run_episode(env, actor, critic, feature_extractor, gamma):
    observation, _ = env.reset()
    done = False
    total_reward = 0
    step = 0

    while not done:
        features = feature_extractor.get_features(observation)
        action, mean, std = actor.get_action(features)
        new_observation, reward, terminated, truncated, _ = env.step([action])
        done = terminated or truncated
        new_features = feature_extractor.get_features(new_observation)
        target = reward + gamma * critic.value(new_features) * (not done)
        advantage = target - critic.value(features)

        critic.update(features, target)
        actor.update(features, action, mean, std, advantage)

        observation = new_observation
        total_reward += reward
        step += 1

    return step

if __name__ == "__main__":
    NUM_RUNS = 30
    NUM_EPISODES = 1000
    REPORT_INTERVAL = 10

    avg_lengths = np.zeros((NUM_RUNS, NUM_EPISODES // REPORT_INTERVAL))

    for run in trange(NUM_RUNS, desc="Runs", leave=True):
        env = gym.make("MountainCarContinuous-v0")
        fe = FeatureExtractor()
        critic = LinearValueFunction(fe.num_of_features, alpha=0.001)
        actor = GaussianPolicy(fe.num_of_features, alpha=0.0001)
        
        lengths = []
        for ep in trange(NUM_EPISODES, leave=False):
            ep_len = run_episode(env, actor, critic, fe, gamma=1.0)
            lengths.append(ep_len)
            if (ep + 1) % REPORT_INTERVAL == 0:
                avg_lengths[run, ep // REPORT_INTERVAL] = np.mean(lengths[-REPORT_INTERVAL:])

        env.close()
    mean_lengths = np.mean(avg_lengths, axis=0)
    std_lengths = np.std(avg_lengths, axis=0)
    x = np.arange(0, NUM_EPISODES, REPORT_INTERVAL)

    # Graficar
    plt.figure(figsize=(10, 6))
    plt.plot(x, mean_lengths, label="Longitud promedio", color='blue')
    plt.fill_between(x, mean_lengths - std_lengths, mean_lengths + std_lengths, color='blue', alpha=0.2, label="±1 desvío estándar")

    plt.title("Longitud promedio de episodios (Actor-Critic)")
    plt.xlabel("Episodio")
    plt.ylabel("Longitud del episodio")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

  # guardar resultados 
    results = np.vstack((x, mean_lengths, std_lengths)).T
    np.savetxt("../Data/resultados_actor_critic.csv", results, delimiter=",", header="Episodio,Longitud_Promedio,Desviacion_Estandar", comments='')
    