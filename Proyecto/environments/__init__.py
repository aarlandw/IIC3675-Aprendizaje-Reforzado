"""
Environments module for quadcopter RL training.

Contains all environment implementations:
- base_env: Core quadcopter physics and rendering
- goal_conditioned_env: HER-compatible goal-conditioned environment
- variants: Environment variations for different training scenarios
"""

from .base_env import QuadcopterEnv
from .goal_conditioned_env import GoalConditionedDroneEnvironment

__all__ = ["QuadcopterEnv", "GoalConditionedDroneEnvironment"]
