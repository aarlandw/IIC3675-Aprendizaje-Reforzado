import numpy as np

from FeatureExtractor import FeatureExtractor


class ActorCritic:

    def __init__(self, gamma: float, alpha_v: float, alpha_pi: float):
        self.__alpha_v = alpha_v
        self.__alpha_pi = alpha_pi
        self.__gamma = gamma
        self.__feature_extractor = FeatureExtractor()
        self.__num_features = self.__feature_extractor.num_of_features
        self.__weights_v = np.zeros(self.__num_features)
        self.__theta_mu = np.zeros(self.__num_features)
        self.__theta_sigma = np.zeros(self.__num_features)
        self.__I = None

    def reset_episode_values(self):
        self.__I = 1.0

    def sample_action(self, observation):
        x = self.__feature_extractor.get_features(observation)
        mu = np.dot(self.__theta_mu, x)
        sigma = np.exp(np.dot(self.__theta_sigma, x))
        return np.random.normal(loc=mu, scale=sigma, size=None)

    def learn(self, observation, action, reward, next_observation, done):
        # TODO: Implementar la regla de aprendizaje de actor-critic 
        
        pass