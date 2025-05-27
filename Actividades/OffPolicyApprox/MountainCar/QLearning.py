import random

import numpy as np

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
    def alpha(self):
        return self.__alpha
    
    @property
    def gamma(self):
        return self.__gamma
    
    def sample_action(self, observation):
        if random.random() < self.__epsilon:
            return random.randrange(self.__num_actions)
        return self.argmax(observation)

    def argmax(self, observation):
        a_max = None
        q_max = float('-inf')
        for action in range(self.__num_actions):
            q_value = self.__get_q_estimate(observation, action)
            if q_value > q_max:
                q_max = q_value
                a_max = [action]
            elif q_value == q_max:
                a_max.append(action)
        return random.choice(a_max)

    def __get_q_estimate(self, observation, action):
        x = self.__feature_extractor.get_features(observation, action)
        return np.dot(self.__weights, x)

    @__weights.setter
    def __weights(self, weights):
        assert len(weights) == self.__num_features, "Weights must have the same size as the number of features"
        self.__weights = weights
        
    
    
    
    # def learn(self, observation, action, reward, next_observation, done):
    #     # TODO: Agrega acá la regla de aprendizaje de Q-Learning con una aproximación lineal
    #     weights = self.__weights
    #     q = self.__get_q_estimate(observation, action)
    #     q_next = 0.0 if done else self.__get_q_estimate(next_observation, self.argmax(next_observation))
    #     weights_next = weights + self.__alpha * (reward + self.__gamma * q_next - q) 
    #     self.__weights = weights_next 
        
    def learn(self, observation, action, reward, next_observation, done):
        """Update the weights using the Q-learning rule with linear function approximation."""

        # Get the feature vector for (s, a)
        phi = self.get_feature_vector(observation, action)

        # Estimate current Q(s, a)
        q = np.dot(self.weights, phi)

        # Estimate max_a' Q(s', a') — unless done
        if done:
            q_next = 0.0
        else:
            best_next_action = self.argmax(next_observation)
            q_next = self.get_q_estimate(next_observation, best_next_action)

        # Compute TD error
        td_error = reward + self.gamma * q_next - q

        

        # Gradient descent step: w ← w + α * δ * φ(s,a)
        self.weights += self.alpha * td_error * phi
        