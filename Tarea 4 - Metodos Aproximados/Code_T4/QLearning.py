import random

import numpy as np
from tqdm import tqdm, trange

from FeatureExtractor import FeatureExtractor


class QLearning:

    def __init__(self, num_actions: int, epsilon: float, alpha: float, gamma: float):
        self.__num_actions = num_actions
        self.__epsilon = epsilon
        self.__alpha = alpha
        self.__gamma = gamma
        self.__feature_extractor = FeatureExtractor()
        self.__num_features = self.__feature_extractor.num_of_features
        self.__weights = np.zeros(self.__num_features)

    @property
    def weights(self):
        return self.__weights

    @property
    def alpha(self):
        return self.__alpha

    @property
    def gamma(self):
        return self.__gamma

    @property
    def epsilon(self):
        # Possibility: Epsilon Decay
        assert (
            self.__epsilon >= 0 and self.__epsilon <= 1
        ), "Epsilon must be between 0 and 1"
        return self.__epsilon

    @property
    def num_actions(self):
        return self.__num_actions

    @property
    def num_features(self):
        return self.__num_features

    @weights.setter
    def weights(self, weights):
        assert (
            len(weights) == self.num_features
        ), "Weights must have the same size as the number of features"
        self.__weights = weights

    def sample_action(self, observation):
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        return self.argmax(observation)

    def argmax(self, observation):
        a_max = None  # ? Should this be an empty list?
        q_max = float("-inf")
        for action in range(self.num_actions):
            q_value = self.get_q_estimate(observation, action)
            if q_value > q_max:
                q_max = q_value
                a_max = [action]
            elif q_value == q_max:
                a_max.append(action)
        return random.choice(a_max)

    def get_feature_vector(self, observation, action):
        return self.__feature_extractor.get_features(observation, action)

    def get_q_estimate(self, observation, action):
        return np.dot(self.weights, self.get_feature_vector(observation, action))

    def learn(self, observation, action, reward, next_observation, done):
        """Update the weights using the Q-learning rule with linear function approximation."""

        # Get the feature vector for (s, a)
        x = self.get_feature_vector(observation, action)

        # Estimate current Q(s, a)
        q = np.dot(self.weights, x)

        # Estimate max_a' Q(s', a') — unless done
        if done:
            q_next = 0.0
        else:
            best_next_action = self.argmax(next_observation)
            q_next = self.get_q_estimate(next_observation, best_next_action)

        # Compute TD error
        td_error = reward + self.gamma * q_next - q

        # Gradient descent step: w ← w + α * δ * φ(s,a)
        self.weights += self.alpha * td_error * x

    def run(self, env, num_episodes: int):
        """
        Run the agent in the environment for a given number of episodes.
        """
        rewards = []
        for episode in trange(num_episodes, desc="Episodes: ", leave=False):
            observation, info = env.reset()
            terminated = truncated = False
            ep_reward = 0

            while not terminated and not truncated:
                action = self.sample_action(observation)
                next_observation, reward, terminated, truncated, info = env.step(action)
                self.learn(observation, action, reward, next_observation, terminated)
                ep_reward += reward
                observation = next_observation

            rewards.append(ep_reward)
            if (episode + 1) % 10 == 0:
                tqdm.write(f"Episode: {episode + 1}, Reward: {ep_reward}")
        env.close()
        return rewards
