import gymnasium as gym

from ActorCritic import ActorCritic


def run_actor_critic_with_linear_approx():
    num_episodes = 1000
    gamma = 1.0
    alpha_v = 0.001
    alpha_pi = 0.0001

    env = gym.make("MountainCarContinuous-v0")

    actor_critic = ActorCritic(gamma, alpha_v, alpha_pi)
    for episode in range(num_episodes):
        observation, info = env.reset()
        actor_critic.reset_episode_values()
        
        terminated = truncated = False
        ep_reward = 0

        while not terminated and not truncated:
            action = actor_critic.sample_action(observation)
            next_observation, reward, terminated, truncated, info = env.step([action])

            ep_reward += reward

            actor_critic.learn(observation, action, reward, next_observation, terminated)

            observation = next_observation

        if (episode + 1) % 10 == 0:
            print(f"Episode: {episode + 1}, Reward: {ep_reward}")
    env.close()

    show(actor_critic)


def show(actor_critic):
    env = gym.make("MountainCarContinuous-v0", render_mode="human")
    observation, info = env.reset()
    for _ in range(1000):
        action = actor_critic.sample_action(observation)
        observation, _, terminated, truncated, _ = env.step([action])

        if terminated or truncated:
            observation, info = env.reset()

    env.close()


if __name__ == "__main__":
    run_actor_critic_with_linear_approx()
