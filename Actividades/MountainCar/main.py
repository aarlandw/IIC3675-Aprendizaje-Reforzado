import gymnasium as gym
from tqdm import tqdm, trange
from Sarsa import Sarsa


def run_sarsa_with_linear_approx():
    num_episodes = 1000
    epsilon = 0.0
    gamma = 1.0
    alpha = 0.5 / 8

    env = gym.make("MountainCar-v0")
    n_actions = env.action_space.n

    sarsa = Sarsa(n_actions, epsilon, alpha, gamma)
    for episode in trange(num_episodes, desc="Training", unit="episode"):
        observation, info = env.reset()
        action = sarsa.sample_action(observation)
        terminated = truncated = False
        ep_reward = 0

        while not terminated and not truncated:
            next_observation, reward, terminated, truncated, info = env.step(action)
            next_action = sarsa.sample_action(next_observation)

            ep_reward += reward

            sarsa.learn(observation, action, reward, next_observation, next_action, terminated)

            observation, action = next_observation, next_action

        if (episode + 1) % 10 == 0:
            tqdm.write(f"Episode: {episode + 1}, Reward: {ep_reward}")
    env.close()

    show(sarsa)


def show(sarsa):
    env = gym.make("MountainCar-v0", render_mode="human")
    observation, info = env.reset()
    for _ in range(1000):
        action = sarsa.argmax(observation)
        observation, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()


if __name__ == "__main__":
    run_sarsa_with_linear_approx()
