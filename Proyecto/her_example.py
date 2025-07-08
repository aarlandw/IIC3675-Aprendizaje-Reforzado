"""
Example of how to implement HER with your drone environment.
This is a template - not ready to run yet.
"""

import gymnasium as gym
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

# For HER, you'd need to modify your environment to:
# 1. Include goal in observation space
# 2. Use goal-conditioned rewards
# 3. Return "achieved_goal" and "desired_goal" in info


class GoalConditionedDroneEnv(gym.Env):
    """
    Modified drone environment for HER.
    Key changes:
    - Observation includes current position + goal position
    - Reward is sparse: +1 if close to goal, 0 otherwise
    - Info contains achieved_goal and desired_goal
    """

    def __init__(self):
        super().__init__()

        # Action space remains the same
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))

        # Observation space now includes goal
        # observation = {
        #     'observation': [drone_state],  # 12 values as before
        #     'achieved_goal': [x, y],       # current position
        #     'desired_goal': [x, y]         # target position
        # }
        self.observation_space = gym.spaces.Dict(
            {
                "observation": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(12,)),
                "achieved_goal": gym.spaces.Box(low=0, high=800, shape=(2,)),
                "desired_goal": gym.spaces.Box(low=0, high=800, shape=(2,)),
            }
        )

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Goal-conditioned reward function"""
        distance = np.linalg.norm(achieved_goal - desired_goal)
        return 1.0 if distance < 50 else 0.0

    def reset(self, seed=None, options=None):
        # Reset drone to center
        self.x, self.y = 400, 400
        # Set random goal
        self.goal = np.random.uniform([200, 200], [600, 600])

        obs = {
            "observation": self._get_drone_state(),
            "achieved_goal": np.array([self.x, self.y]),
            "desired_goal": self.goal.copy(),
        }
        return obs, {}

    def step(self, action):
        # ... physics simulation ...

        achieved_goal = np.array([self.x, self.y])
        reward = self.compute_reward(achieved_goal, self.goal, {})

        obs = {
            "observation": self._get_drone_state(),
            "achieved_goal": achieved_goal,
            "desired_goal": self.goal.copy(),
        }

        info = {"achieved_goal": achieved_goal, "desired_goal": self.goal.copy()}

        return obs, reward, done, False, info


# How to use HER:
def create_her_model():
    env = GoalConditionedDroneEnv()
    env = DummyVecEnv([lambda: env])

    model = SAC(
        "MultiInputPolicy",  # Important: MultiInputPolicy for Dict spaces
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,  # Number of HER goals to sample
            goal_selection_strategy=GoalSelectionStrategy.FUTURE,
        ),
        verbose=1,
    )

    return model, env


# To train:
# model, env = create_her_model()
# model.learn(total_timesteps=1_000_000)
