#!/usr/bin/env python3
"""
Train HER with SPARSE rewards - optimal for HER learning
"""

from stable_baselines3 import SAC
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from env_SAC import DroneEnvironment
from her_sparse_wrapper import SparseRewardHERWrapper
import numpy as np
import matplotlib.pyplot as plt

class SparseHERCallback(BaseCallback):
    """Track sparse reward HER training"""
    
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.balloons_collected = []
        self.success_rates = []
        
    def _on_step(self):
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]
            if 'episode' in info:
                episode_reward = info['episode']['r']
                self.episode_rewards.append(episode_reward)
                
                balloons = info.get('balloons_collected', 0)
                is_success = info.get('is_success', False)
                
                self.balloons_collected.append(balloons)
                self.success_rates.append(float(is_success))
                
                if len(self.episode_rewards) % 100 == 0:
                    print(f"Episode {len(self.episode_rewards)}: "
                          f"Balloons: {np.mean(self.balloons_collected[-100:]):.1f}, "
                          f"Success: {np.mean(self.success_rates[-100:]):.2%}")
        
        return True

def train_sparse_her(timesteps=500_000):
    """Train HER with sparse rewards"""
    
    print("🎯 Training HER with SPARSE REWARDS")
    print("   This is the optimal setup for HER!")
    print("=" * 50)
    
    # Create environment
    base_env = DroneEnvironment(render_mode=None)
    her_env = SparseRewardHERWrapper(base_env, goal_threshold=25.0)
    vec_env = DummyVecEnv([lambda: her_env])
    
    # SAC with HER - CONSERVATIVE SETTINGS
    model = SAC(
        "MultiInputPolicy",
        vec_env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={
            "n_sampled_goal": 4,  # Conservative goal sampling
            "goal_selection_strategy": "future",
        },
        learning_rate=3e-4,
        buffer_size=50_000,   # Smaller buffer size
        batch_size=256,       # Standard batch size
        tau=0.005,
        gamma=0.98,           # Higher gamma for sparse rewards
        train_freq=1,         # Train every step
        gradient_steps=1,     # One gradient step per update
        verbose=1,
        device="auto"
    )
    
    callback = SparseHERCallback()
    
    print(f"🚀 Training for {timesteps:,} timesteps...")
    model.learn(total_timesteps=timesteps, callback=callback)
    
    model.save("sparse_her_balloon")
    
    print("✅ Sparse HER training complete!")
    if len(callback.balloons_collected) > 0:
        print(f"   Final performance: {np.mean(callback.balloons_collected[-100:]):.1f} balloons")
    
    return model

if __name__ == "__main__":
    train_sparse_her(timesteps=10_000)  # Start with shorter training