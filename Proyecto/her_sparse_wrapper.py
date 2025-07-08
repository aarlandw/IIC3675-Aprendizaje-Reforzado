#!/usr/bin/env python3
"""
Sparse Reward HER Wrapper - Optimal for HER Learning
Only meaningful rewards: +1 balloon, -1 death, 0 otherwise
"""

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Dict, Box
from env_SAC import DroneEnvironment


class SparseRewardHERWrapper(gym.Wrapper):
    """
    Ultra-sparse reward wrapper for HER
    Perfect for learning goal-conditioned behavior
    """
    
    def __init__(self, env, goal_threshold=25.0):
        super().__init__(env)
        self.goal_threshold = goal_threshold
        
        # HER Dict observation space
        self.observation_space = Dict({
            "observation": Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32),
            "achieved_goal": Box(low=0, high=800, shape=(2,), dtype=np.float32),
            "desired_goal": Box(low=0, high=800, shape=(2,), dtype=np.float32)
        })
        
        self.current_goal = None
        self.balloons_collected = 0
        self.previous_balloon_pos = None
        
    def _get_observation(self):
        """Minimal observation - just what's needed"""
        return np.array([
            self.env.x,           # drone x
            self.env.y,           # drone y
            self.env.xd,          # drone x velocity  
            self.env.yd,          # drone y velocity
            self.env.a,           # drone angle
            self.env.ad,          # drone angular velocity
            self.env.x_pajaro,    # bird 1 x (for avoidance)
            self.env.y_pajaro,    # bird 1 y (for avoidance)
        ], dtype=np.float32)
    
    def _get_achieved_goal(self):
        """Current drone position"""
        return np.array([self.env.x, self.env.y], dtype=np.float32)
    
    def _get_desired_goal(self):
        """Current balloon position"""
        return np.array([self.env.xt, self.env.yt], dtype=np.float32)
    
    def reset(self, **kwargs):
        """Reset with sparse reward mindset"""
        obs, info = self.env.reset(**kwargs)
        
        self.current_goal = self._get_desired_goal()
        self.balloons_collected = 0
        self.previous_balloon_pos = self.current_goal.copy()
        
        obs_dict = {
            "observation": self._get_observation(),
            "achieved_goal": self._get_achieved_goal(),
            "desired_goal": self.current_goal.copy()
        }
        
        info["is_success"] = False
        return obs_dict, info
    
    def step(self, action):
        """Step with ULTRA-SPARSE rewards"""
        obs, original_reward, terminated, truncated, info = self.env.step(action)
        
        achieved_goal = self._get_achieved_goal()
        current_balloon_pos = self._get_desired_goal()
        
        # Check if balloon was collected (position changed)
        balloon_collected = not np.allclose(current_balloon_pos, self.previous_balloon_pos, atol=10.0)
        
        if balloon_collected:
            self.balloons_collected += 1
            self.current_goal = current_balloon_pos.copy()
            self.previous_balloon_pos = current_balloon_pos.copy()
        
        # SPARSE REWARD FUNCTION - The key to HER success!
        sparse_reward = 0.0  # Default: no reward
        
        # Only 3 possible rewards:
        if balloon_collected:
            sparse_reward = +1.0  # BIG reward for success
        elif terminated and original_reward <= -1:
            sparse_reward = -7.0  # Penalty for dying
        # else: sparse_reward = 0.0 (no reward for anything else)
        
        obs_dic = {
            "observation": self._get_observation(),
            "achieved_goal": achieved_goal,
            "desired_goal": self.current_goal.copy()
        }
        
        # Success = close to balloon
        distance_to_goal = np.linalg.norm(achieved_goal - self.current_goal)
        is_success = distance_to_goal < self.goal_threshold
        
        info["is_success"] = is_success
        info["balloons_collected"] = self.balloons_collected
        info["original_reward"] = original_reward
        info["sparse_reward"] = sparse_reward
        
        return obs_dict, sparse_reward, terminated, truncated, info
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        HER reward function - FIXED to return numpy array
        This is what makes HER work!
        """
        # Handle both single goals and batches
        if achieved_goal.ndim == 1:
            # Single goal case
            distance = np.linalg.norm(achieved_goal - desired_goal)
            reward = 1.0 if distance < self.goal_threshold else 0.0
            return np.array([reward], dtype=np.float32)
        else:
            # Batch case
            distances = np.linalg.norm(achieved_goal - desired_goal, axis=1)
            rewards = (distances < self.goal_threshold).astype(np.float32)
            return rewards
    
    def render(self, mode="human"):
        """Render with goal visualization"""
        result = self.env.render(mode)
        
        if (hasattr(self.env, 'screen') and 
            self.env.screen is not None and 
            self.current_goal is not None):
            
            import pygame
            # Draw goal area
            pygame.draw.circle(
                self.env.screen,
                (0, 255, 0),  # Green
                (int(self.current_goal[0]), int(self.current_goal[1])),
                int(self.goal_threshold),
                3
            )
            
            # Draw success/failure indicator
            distance = np.linalg.norm(self._get_achieved_goal() - self.current_goal)
            color = (0, 255, 0) if distance < self.goal_threshold else (255, 0, 0)
            
            pygame.draw.line(
                self.env.screen,
                color,
                (int(self.env.x), int(self.env.y)),
                (int(self.current_goal[0]), int(self.current_goal[1])),
                2
            )
            
            # Display sparse reward info
            font = pygame.font.Font(None, 24)
            text = font.render(f"Balloons: {self.balloons_collected}", True, (255, 255, 255))
            self.env.screen.blit(text, (20, 120))
            
            pygame.display.update()
        
        return result