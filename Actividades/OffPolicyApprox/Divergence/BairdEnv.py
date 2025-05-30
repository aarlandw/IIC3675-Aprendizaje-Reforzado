import random

import numpy as np

from AbstractEnv import AbstractEnv


class BairdEnv(AbstractEnv):

    def __init__(self):
        self.__state2features = np.zeros((7, 8))
        for i in range(6):
            self.__state2features[i, i] = 2.0
            self.__state2features[i, 7] = 1.0
        self.__state2features[6, 6] = 1.0
        self.__state2features[6, 7] = 2.0
        self.__current_state = None

    @property
    def action_space(self):
        return ["solid", "dashed"]

    def reset(self):
        self.__current_state = random.randrange(6)
        return self.__state2features[self.__current_state, :]

    def step(self, action):
        if action == "solid":
            self.__current_state = 6
        else:
            self.__current_state = random.randrange(6)
        reward = 0.0
        done = False
        return self.__state2features[self.__current_state, :], reward, done
