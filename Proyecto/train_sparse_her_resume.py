#!/usr/bin/env python3
"""
Train HER with SPARSE rewards - with save/resume functionality
"""

import os
import time
from stable_baselines3 import SAC
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from env_SAC import DroneEnvironment
from her_sparse_wrapper import SparseRewardHERWrapper
import numpy as np
import matplotlib.pyplot as plt
import pickle

class SparseHERCallback(BaseCallback):
    """Enhanced callback with automatic saving"""
    
    def __init__(self, save_freq=10_000, model_name="sparse_her_balloon"):
        super().__init__()
        self.save_freq = save_freq
        self.model_name = model_name
        self.episode_rewards = []
        self.balloons_collected = []
        self.success_rates = []
        self.total_timesteps = 0
        self.best_performance = 0
        
        # Load previous stats if they exist
        self.load_training_stats()
        
    def _on_step(self):
        self.total_timesteps += 1
        
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]
            if 'episode' in info:
                episode_reward = info['episode']['r']
                self.episode_rewards.append(episode_reward)
                
                balloons = info.get('balloons_collected', 0)
                is_success = info.get('is_success', False)
                
                self.balloons_collected.append(balloons)
                self.success_rates.append(float(is_success))
                
                # Check for best performance
                if len(self.balloons_collected) >= 10:
                    recent_performance = np.mean(self.balloons_collected[-10:])
                    if recent_performance > self.best_performance:
                        self.best_performance = recent_performance
                        print(f"🏆 New best performance: {recent_performance:.2f} balloons!")
                        # Save best model
                        self.model.save(f"{self.model_name}_best")
                
                # Print progress
                if len(self.episode_rewards) % 50 == 0:
                    avg_balloons = np.mean(self.balloons_collected[-50:])
                    avg_success = np.mean(self.success_rates[-50:])
                    print(f"Episode {len(self.episode_rewards)}: "
                          f"Balloons: {avg_balloons:.1f}, "
                          f"Success: {avg_success:.2%}, "
                          f"Total Steps: {self.total_timesteps:,}")
        
        # Auto-save checkpoint
        if self.total_timesteps % self.save_freq == 0:
            self.save_checkpoint()
            
        return True
    
    def save_checkpoint(self):
        """Save training checkpoint"""
        print(f"💾 Saving checkpoint at {self.total_timesteps:,} timesteps...")
        
        # Save model
        self.model.save(f"{self.model_name}_checkpoint")
        
        # Save training statistics
        stats = {
            'episode_rewards': self.episode_rewards,
            'balloons_collected': self.balloons_collected,
            'success_rates': self.success_rates,
            'total_timesteps': self.total_timesteps,
            'best_performance': self.best_performance
        }
        
        with open(f"{self.model_name}_training_stats.pkl", 'wb') as f:
            pickle.dump(stats, f)
        
        print(f"✅ Checkpoint saved!")
    
    def load_training_stats(self):
        """Load previous training statistics"""
        stats_file = f"{self.model_name}_training_stats.pkl"
        if os.path.exists(stats_file):
            try:
                with open(stats_file, 'rb') as f:
                    stats = pickle.load(f)
                
                self.episode_rewards = stats.get('episode_rewards', [])
                self.balloons_collected = stats.get('balloons_collected', [])
                self.success_rates = stats.get('success_rates', [])
                self.total_timesteps = stats.get('total_timesteps', 0)
                self.best_performance = stats.get('best_performance', 0)
                
                print(f"📊 Loaded previous training stats:")
                print(f"   Episodes: {len(self.episode_rewards)}")
                print(f"   Total timesteps: {self.total_timesteps:,}")
                print(f"   Best performance: {self.best_performance:.2f} balloons")
                
            except Exception as e:
                print(f"⚠️ Could not load training stats: {e}")

def train_sparse_her_with_resume(
    timesteps=500_000, 
    model_name="sparse_her_balloon",
    resume_from_checkpoint=True
):
    """Train HER with save/resume functionality"""
    
    print("🎯 Training HER with SPARSE REWARDS + RESUME")
    print("=" * 60)
    
    # Create environment
    base_env = DroneEnvironment(render_mode=None)
    her_env = SparseRewardHERWrapper(base_env, goal_threshold=25.0)
    vec_env = DummyVecEnv([lambda: her_env])
    
    # Check for existing model
    checkpoint_path = f"{model_name}_checkpoint.zip"
    model_path = f"{model_name}.zip"
    
    model = None
    starting_timesteps = 0
    
    if resume_from_checkpoint and os.path.exists(checkpoint_path):
        print(f"🔄 Resuming from checkpoint: {checkpoint_path}")
        try:
            model = SAC.load(checkpoint_path, env=vec_env)
            print("✅ Checkpoint loaded successfully!")
            
            # Load training stats to get starting timesteps
            stats_file = f"{model_name}_training_stats.pkl"
            if os.path.exists(stats_file):
                with open(stats_file, 'rb') as f:
                    stats = pickle.load(f)
                starting_timesteps = stats.get('total_timesteps', 0)
                print(f"📊 Resuming from {starting_timesteps:,} timesteps")
        except Exception as e:
            print(f"⚠️ Could not load checkpoint: {e}")
            print("🔄 Starting fresh training...")
            model = None
    
    elif resume_from_checkpoint and os.path.exists(model_path):
        print(f"🔄 Resuming from final model: {model_path}")
        try:
            model = SAC.load(model_path, env=vec_env)
            print("✅ Previous model loaded successfully!")
        except Exception as e:
            print(f"⚠️ Could not load model: {e}")
            print("🔄 Starting fresh training...")
            model = None
    
    # Create new model if no existing model found
    if model is None:
        print("🧠 Creating new SAC+HER model...")
        model = SAC(
            "MultiInputPolicy",
            vec_env,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs={
                "n_sampled_goal": 4,
                "goal_selection_strategy": "future",
            },
            learning_rate=3e-4,
            buffer_size=50_000,
            batch_size=256,
            tau=0.005,
            gamma=0.98,
            train_freq=1,
            gradient_steps=1,
            verbose=1,
            device="auto"
        )
    
    # Setup callback
    callback = SparseHERCallback(save_freq=10_000, model_name=model_name)
    
    # Calculate remaining timesteps
    remaining_timesteps = max(0, timesteps - starting_timesteps)
    
    if remaining_timesteps > 0:
        print(f"🚀 Training for {remaining_timesteps:,} more timesteps...")
        print(f"   (Total target: {timesteps:,}, Already completed: {starting_timesteps:,})")
        
        start_time = time.time()
        model.learn(
            total_timesteps=remaining_timesteps, 
            callback=callback,
            reset_num_timesteps=False  # Don't reset timestep counter
        )
        training_time = time.time() - start_time
        
        # Final save
        print(f"💾 Saving final model as {model_name}.zip...")
        model.save(model_name)
        
        # Save final stats
        callback.save_checkpoint()
        
        print(f"✅ Training complete!")
        print(f"   Training time: {training_time/3600:.1f} hours")
        print(f"   Total timesteps: {timesteps:,}")
        
        if len(callback.balloons_collected) > 0:
            final_performance = np.mean(callback.balloons_collected[-100:])
            print(f"   Final performance: {final_performance:.1f} balloons")
            print(f"   Best performance: {callback.best_performance:.1f} balloons")
    else:
        print("✅ Training already complete!")
    
    return model

def continue_training(model_name="sparse_her_balloon", additional_timesteps=100_000):
    """Continue training an existing model"""
    
    print(f"🔄 Continuing training for {model_name}")
    print(f"   Adding {additional_timesteps:,} more timesteps")
    
    # Load current stats to get total timesteps
    stats_file = f"{model_name}_training_stats.pkl"
    current_timesteps = 0
    
    if os.path.exists(stats_file):
        try:
            with open(stats_file, 'rb') as f:
                stats = pickle.load(f)
            current_timesteps = stats.get('total_timesteps', 0)
        except:
            pass
    
    new_target = current_timesteps + additional_timesteps
    
    return train_sparse_her_with_resume(
        timesteps=new_target,
        model_name=model_name,
        resume_from_checkpoint=True
    )

if __name__ == "__main__":
    # Train with resume functionality
    train_sparse_her_with_resume(timesteps=20_000)