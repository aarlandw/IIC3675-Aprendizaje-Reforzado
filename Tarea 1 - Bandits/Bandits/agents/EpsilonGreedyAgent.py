from BaseAgent import BaseAgent
import numpy as np
import random

class EpsilonGreedyAgent(BaseAgent):
    def __init__(self, num_of_actions: int, epsilon: float):
        self.num_of_actions = num_of_actions
        self.epsilon = epsilon
        
        for i in range(num_of_actions):
            self.q_values[i] = 0
            self.action_counts[i] = 0
         
    def get_action(self) -> int:
        if np.random.rand() < self.epsilon:
            return random.randrange(self.num_of_actions)
        else:
            return self.q_values.index(max(self.q_values))
    
    
    def learn(self, action, reward) -> None:
        self.action_counts[action] += 1
        self.q_values[action] += 1/self.action_counts[action] * (reward - self.q_values[action])
        