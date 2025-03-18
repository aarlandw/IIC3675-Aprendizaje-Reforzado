from agents.BaseAgent import BaseAgent
import numpy as np
import random

class FixedStepSizeAgent(BaseAgent):
    def __init__(self, num_of_actions: int, epsilon: float = 0.1, alpha: float = 0.1):
        self.num_of_actions = num_of_actions
        self.epsilon = epsilon
        self.alpha = alpha
        self.action_counts = np.zeros(num_of_actions) 

         
    def get_action(self) -> int:
        if np.random.rand() < self.epsilon:
            return random.randrange(self.num_of_actions)
        else:
            return np.argmax(self.q_values)
    
    
    def learn(self, action, reward) -> None:
        self.q_values[action] += self.alpha * (reward - self.q_values[action])
        