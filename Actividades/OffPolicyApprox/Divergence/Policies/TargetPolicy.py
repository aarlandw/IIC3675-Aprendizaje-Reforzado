from Policies.Policy import Policy


class TargetPolicy(Policy):

    def sample(self, observation):
        return "solid"

    def get_probability(self, observation, action):
        if action == "solid":
            return 1.0
        return 0.0
