import random

from Policies.Policy import Policy


class BehaviorPolicy(Policy):

    def sample(self, observation):
        if random.random() < self.get_probability(observation, "solid"):
            return "solid"
        return "dashed"

    def get_probability(self, observation, action):
        if action == "solid":
            return 1.0 / 7.0
        return 6.0 / 7.0
