"""
Quadcopter RL Training Project

A comprehensive reinforcement learning project for training quadcopter agents
using SAC, SAC+HER, and Actor-Critic algorithms.

Main components:
- environments: Quadcopter simulation environments
- agents: RL algorithm implementations
- training: Training scripts and configurations
- evaluation: Model evaluation and visualization tools
- utils: Utility functions and helpers
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Import main components for easy access
from .environments import QuadcopterEnv, GoalConditionedDroneEnvironment
from .agents import SACAgent, HERAgent, ActorCriticAgent
from .training import train_sac, train_her
from .evaluation import evaluate_model

__all__ = [
    "QuadcopterEnv",
    "GoalConditionedDroneEnvironment",
    "SACAgent",
    "HERAgent",
    "ActorCriticAgent",
    "train_sac",
    "train_her",
    "evaluate_model",
]
