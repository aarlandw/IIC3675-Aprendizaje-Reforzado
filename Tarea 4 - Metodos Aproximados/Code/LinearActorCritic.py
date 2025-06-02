import numpy as np


class LinearActorCritic:
    def __init__(self, state_dim, action_dim, learning_rate=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        
        # Initialize weights for actor and critic
        self.actor_weights = np.random.rand(state_dim, action_dim)
        self.critic_weights = np.random.rand(state_dim, 1)

    def predict_action(self, state):
        # Predict action probabilities using the actor
        return np.dot(state, self.actor_weights)

    def predict_value(self, state):
        # Predict value using the critic
        return np.dot(state, self.critic_weights)