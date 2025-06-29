from abc import ABC, abstractmethod


class Policy(ABC):

    @abstractmethod
    def sample(self, observation):
        """
        :return: a sampled action.
        """
        pass

    @abstractmethod
    def get_probability(self, observation, action):
        """
        :return: the probability of selecting action given observation.
        """
        pass
