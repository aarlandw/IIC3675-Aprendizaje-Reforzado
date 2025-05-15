import gymnasium as gym

from QLearning import QLearning


def run_qlearning_with_linear_approx():
    num_episodes = 1000
    epsilon = 0.0
    gamma = 1.0
    alpha = 0.5 / 8

    env = gym.make("MountainCar-v0")
    n_actions = env.action_space.n

    qlearning = QLearning(n_actions, epsilon, alpha, gamma)
    for episode in range(num_episodes):
        observation, info = env.reset()
        terminated = truncated = False
        ep_reward = 0

        while not terminated and not truncated:
            action = qlearning.sample_action(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            qlearning.learn(observation, action, reward, next_observation, terminated)
            ep_reward += reward
            observation = next_observation

        if (episode + 1) % 10 == 0:
            print(f"Episode: {episode + 1}, Reward: {ep_reward}")
    env.close()

    show(qlearning)


def show(qlearning):
    env = gym.make("MountainCar-v0", render_mode="human")
    observation, info = env.reset()
    for _ in range(1000):
        action = qlearning.argmax(observation)
        observation, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()


if __name__ == "__main__":
    run_qlearning_with_linear_approx()
