import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from env import QuadcopterEnv  # Tu entorno personalizado
from FeatureExtractor import FeatureExtractor  # Asegúrate que este existe y funciona

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
    def __init__(self, num_features, action_dim=2, alpha=0.0001):
        self.mean_weights = np.zeros((action_dim, num_features))
        self.log_std = np.zeros(action_dim)
        self.alpha = alpha
        self.action_dim = action_dim

    def get_action(self, features):
        means = np.dot(self.mean_weights, features)
        stds = np.exp(self.log_std)
        actions = np.random.normal(means, stds)
        return np.clip(actions, -1, 1), means, stds

    def update(self, features, actions, means, stds, advantage):
        for i in range(self.action_dim):
            grad_log = (actions[i] - means[i]) / (stds[i]**2) * features
            self.mean_weights[i] += self.alpha * advantage * grad_log

def run_episode(env, actor, critic, feature_extractor, gamma):
    observation, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        features = feature_extractor.get_features(observation)
        actions, means, stds = actor.get_action(features)
        new_obs, reward, terminated, truncated, _ = env.step(actions)
        done = terminated or truncated

        new_features = feature_extractor.get_features(new_obs)
        target = reward + gamma * critic.value(new_features) * (not done)
        advantage = target - critic.value(features)

        critic.update(features, target)
        actor.update(features, actions, means, stds, advantage)

        observation = new_obs
        total_reward += reward

    return total_reward

if __name__ == "__main__":
    NUM_RUNS = 1
    NUM_EPISODES = 100000
    REPORT_INTERVAL = 10

    avg_rewards = np.zeros((NUM_RUNS, NUM_EPISODES // REPORT_INTERVAL))

    for run in range(NUM_RUNS):
        env = QuadcopterEnv()
        env.max_collected = 1
        fe = FeatureExtractor()
        critic = LinearValueFunction(fe.num_of_features, alpha=0.001)
        actor = GaussianPolicy(fe.num_of_features, action_dim=2, alpha=0.0001)

        rewards = []
        for ep in tqdm(range(NUM_EPISODES), desc=f"Run {run + 1}/{NUM_RUNS}"):
            ep_reward = run_episode(env, actor, critic, fe, gamma=0.99)
            rewards.append(ep_reward)

            if (ep + 1) % REPORT_INTERVAL == 0:
                avg_rewards[run, ep // REPORT_INTERVAL] = np.mean(rewards[-REPORT_INTERVAL:])

        env.close()

    mean_rewards = np.mean(avg_rewards, axis=0)
    std_rewards = np.std(avg_rewards, axis=0)
    x = np.arange(0, NUM_EPISODES, REPORT_INTERVAL)

    # Gráfico de recompensas
    plt.figure(figsize=(10, 6))
    plt.plot(x, mean_rewards, label="Recompensa promedio", color='green')
    plt.fill_between(x, mean_rewards - std_rewards, mean_rewards + std_rewards,
                     color='green', alpha=0.2, label="±1 desvío estándar")

    plt.title("Recompensa promedio por episodio (Actor-Critic con objetivo fijo)")
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa promedio")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Guardar resultados
    results = np.vstack((x, mean_rewards, std_rewards)).T
    np.savetxt("recompensas_actor_critic.csv", results, delimiter=",",
               header="Episodio,Recompensa_Promedio,Desviacion_Estandar", comments='')