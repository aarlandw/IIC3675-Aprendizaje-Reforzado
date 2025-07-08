import numpy as np

from tiles3 import IHT, tiles


class FeatureExtractor:

    def __init__(self):
        self.__num_obs_features = 4096
        self.__iht = IHT(self.__num_obs_features)
        self.__num_of_tiles = 8

    @property
    def num_of_features(self):
        return self.__num_obs_features

    def get_features(self, observation, action=None):
        x = observation[0]
        xdot = observation[1]
        scaled_obs = [8 * x / (0.5 + 1.2), 8 * xdot / (0.07 + 0.07)]
        if action is None:
            tile_result = tiles(self.__iht, self.__num_of_tiles, scaled_obs)
        else:
            tile_result = tiles(self.__iht, self.__num_of_tiles, scaled_obs, [action])
        features = np.zeros(self.__num_obs_features)
        for tile_id, tile_pos in enumerate(tile_result):
            features[tile_pos] = 1
        return features
