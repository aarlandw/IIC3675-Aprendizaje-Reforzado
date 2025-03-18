from agents.BaseAgent import BaseAgent
import numpy as np
import random

class EpsilonGreedyAgent(BaseAgent):
    def __init__(self, num_of_actions: int, epsilon: float = 0.1):
        self.num_of_actions = num_of_actions
        self.epsilon = epsilon
        self.q_values = np.zeros(num_of_actions) 
        self.action_counts = np.zeros(num_of_actions) 

         
    def get_action(self) -> int:
        if np.random.rand() < self.epsilon:
            return random.randrange(self.num_of_actions)
        else:
            return np.argmax(self.q_values)
    
    
    def learn(self, action, reward) -> None:
        self.action_counts[action] += 1
        self.q_values[action] += 1/self.action_counts[action] * (reward - self.q_values[action])
        