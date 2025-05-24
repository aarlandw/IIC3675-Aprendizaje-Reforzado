import random

import numpy as np

from FeatureExtractor import FeatureExtractor


class Sarsa:

    def __init__(self, num_actions: int, epsilon: float, alpha: float, gamma: float):
        self.__num_actions = num_actions
        self.__epsilon = epsilon
        self.__alpha = alpha
        self.__gamma = gamma
        self.__feature_extractor = FeatureExtractor() #? Add the number of actions to the feature extractor
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
        assert self.__epsilon >= 0 and self.__epsilon <= 1, "Epsilon must be between 0 and 1"
        return self.__epsilon

    @property
    def num_actions(self):
        return self.__num_actions

    @property
    def num_features(self):
        return self.__num_features
    
    @weights.setter
    def __weights(self, weights):
        assert len(weights) == self.num_features, "Weights must have the same size as the number of features"
        self.__weights = weights

    def sample_action(self, observation):
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        return self.argmax(observation)

    def argmax(self, observation):
        a_max = None
        q_max = float('-inf')
        for action in range(self.num_actions):
            q_value = self.__get_q_estimate(observation, action)
            if q_value > q_max:
                q_max = q_value
                a_max = [action]
            elif q_value == q_max:
                a_max.append(action)
        return random.choice(a_max)

    def __get_q_estimate(self, observation, action):
        x = self.__feature_extractor.get_features(observation, action)
        return np.dot(self.weights, x)

    def learn(self, observation, action, reward, next_observation, next_action, done):
        #Todo: Complete
        pass
        
