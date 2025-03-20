from agents.BaseAgent import BaseAgent
import numpy as np
import random

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Restar el máximo para estabilidad 
    return exp_x / np.sum(exp_x)

class GradientBanditAgent(BaseAgent):
    def __init__(self, num_of_actions: int, alpha: float = 0.1):
        self.num_of_actions = num_of_actions
        self.alpha = alpha  
        self.H = np.zeros(num_of_actions)  
        self.average_reward = 0.0 
        self.step_count = 0 
        self.baseline = True

    def get_action(self) -> int:
        probabilities_pi = softmax(self.H)
        return np.random.choice(self.num_of_actions, p=probabilities_pi)

    def learn(self, action, reward) -> None:
        probabilities_pi = softmax(self.H)

        #actualizamos H
        for a in range(self.num_of_actions):
            if a == action:
                self.H[a] += self.alpha * (reward - self.average_reward) * (1 - probabilities_pi[a])
            else:
                self.H[a] += self.alpha * (reward - self.average_reward) * (0 - probabilities_pi[a])
        self.step_count += 1
        if self.baseline:
          self.average_reward += (reward - self.average_reward) / self.step_count
        else:
          self.average_reward = 0.0

