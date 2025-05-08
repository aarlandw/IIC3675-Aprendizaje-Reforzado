import numpy as np

from tiles3 import IHT, tiles


class FeatureExtractor:

    def __init__(self, num_actions: int):
        self.__num_obs_features = 4096
        self.__num_action_features = num_actions
        self.__iht = IHT(self.__num_obs_features)
        self.__num_of_tiles = 8

    @property
    def num_of_features(self):
        return self.__num_obs_features + self.__num_action_features

    def get_features(self, observation, action):
        obs_features = self.__get_observation_features(observation)
        action_features = self.__get_action_features(action)
        return np.concatenate((obs_features, action_features))

    def __get_observation_features(self, observation):
        x = observation[0]
        xdot = observation[1]
        scaled_obs = [8 * x / (0.5 + 1.2), 8 * xdot / (0.07 + 0.07)]
        tile_result = tiles(self.__iht, self.__num_of_tiles, scaled_obs)
        obs_features = np.zeros(self.__num_obs_features)
        for tile_id, tile_pos in enumerate(tile_result):
            obs_features[tile_pos] = 1
        return obs_features

    def __get_action_features(self, action):
        action_features = np.zeros(self.__num_action_features)
        action_features[action] = 1
        return action_features
