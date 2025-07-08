import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
from env import QuadcopterEnv  
from env_SAC import droneEnv
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime
from gymnasium.wrappers import RecordVideo

class EnhancedRewardCallback(BaseCallback):
    def __init__(self, total_timesteps, save_freq=10000):
        super().__init__()
        self.episode_rewards = []
        self.episode_lengths = []
        self.timesteps_list = []
        self.total_timesteps = total_timesteps
        self.save_freq = save_freq
        self.pbar = None
        self.current_episode = 0
        self.start_time = None
        
        # Create results directory
        self.results_dir = f"sac_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_dir, exist_ok=True)
        print(f"📁 Results will be saved to: {self.results_dir}")

    def _on_training_start(self):
        """Initialize progress bar and start time"""
        self.pbar = tqdm(total=self.total_timesteps, desc="🚁 Training SAC")
        self.start_time = datetime.now()
        print(f"🚀 Training started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    def _on_step(self) -> bool:
        """Called at each training step"""
        infos = self.locals.get("infos", [])
        
        # Check if episode finished
        if infos and "episode" in infos[0]:
            episode_reward = infos[0]["episode"]["r"]
            episode_length = infos[0]["episode"]["l"]
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.timesteps_list.append(self.num_timesteps)
            self.current_episode += 1
            
            # Print episode info
            if self.current_episode % 50 == 0:  # Every 50 episodes
                avg_reward = np.mean(self.episode_rewards[-50:])
                max_reward = np.max(self.episode_rewards[-50:])
                print(f"\n📊 Episode {self.current_episode}: "
                      f"Avg Reward (last 50): {avg_reward:.2f}, "
                      f"Max Reward: {max_reward:.2f}")

        # Save data periodically
        if self.num_timesteps % self.save_freq == 0 and len(self.episode_rewards) > 0:
            self._save_data(intermediate=True)

        # Update progress bar
        if self.pbar:
            self.pbar.update(1)
            
        return True

    def _on_training_end(self):
        """Save final results when training ends"""
        if self.pbar:
            self.pbar.close()
            
        print(f"\n✅ Training completed!")
        print(f"📊 Total episodes: {len(self.episode_rewards)}")
        print(f"⏱️ Training time: {datetime.now() - self.start_time}")
        
        # Save final data
        self._save_data(intermediate=False)

    def _save_data(self, intermediate=False):
        """Save training data in multiple formats"""
        if len(self.episode_rewards) == 0:
            return
            
        # Create DataFrame
        data = {
            'episode': list(range(1, len(self.episode_rewards) + 1)),  # ← FIXED: Convert range to list
            'reward': self.episode_rewards,
            'episode_length': self.episode_lengths,
            'timestep': self.timesteps_list
        }
        df = pd.DataFrame(data)
        
        # Add moving averages
        window = min(50, len(df))
        if window > 1:
            df['reward_ma_50'] = df['reward'].rolling(window=window, min_periods=1).mean()
            df['reward_ma_100'] = df['reward'].rolling(window=min(100, len(df)), min_periods=1).mean()
        
        prefix = "intermediate_" if intermediate else "final_"
        
        # 1. Save as CSV (most common format)
        csv_file = f"{self.results_dir}/{prefix}training_results.csv"
        df.to_csv(csv_file, index=False)
        
        # 2. Save as JSON (human readable) - FIXED
        json_file = f"{self.results_dir}/{prefix}training_results.json"
        with open(json_file, 'w') as f:
            json.dump({
                'metadata': {
                    'total_episodes': len(self.episode_rewards),
                    'total_timesteps': self.num_timesteps,
                    'training_time': str(datetime.now() - self.start_time) if self.start_time else "Unknown",
                    'best_reward': float(np.max(self.episode_rewards)),
                    'average_reward': float(np.mean(self.episode_rewards)),
                    'final_100_avg': float(np.mean(self.episode_rewards[-100:])) if len(self.episode_rewards) >= 100 else float(np.mean(self.episode_rewards))
                },
                'data': {
                    'episode': list(range(1, len(self.episode_rewards) + 1)),  # ← FIXED: Explicit list conversion
                    'reward': self.episode_rewards,
                    'episode_length': self.episode_lengths,
                    'timestep': self.timesteps_list
                }
            }, f, indent=2)
        
        # 3. Save as pickle (preserves exact data types) - This one was fine
        pickle_file = f"{self.results_dir}/{prefix}training_results.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump({
                'episode_rewards': self.episode_rewards,
                'episode_lengths': self.episode_lengths,
                'timesteps': self.timesteps_list,
                'metadata': {
                    'total_episodes': len(self.episode_rewards),
                    'total_timesteps': self.num_timesteps,
                    'start_time': self.start_time,
                    'end_time': datetime.now()
                }
            }, f)
        
        if not intermediate:
            print(f"💾 Data saved:")
            print(f"   📊 CSV: {csv_file}")
            print(f"   📋 JSON: {json_file}")
            print(f"   🗃️ Pickle: {pickle_file}")

# Create environment
env = droneEnv(render_mode=None, render_every_frame=False, mouse_target=False)

# Envolver con RecordVideo para grabar cada 200 episodios
video_folder = "video2"
os.makedirs(video_folder, exist_ok=True)

# env = RecordVideo(
#     env,
#     video_folder=video_folder,
#     episode_trigger=lambda ep: ep % 200 == 0,
#     name_prefix="drone_episode",
#     disable_logger=True
# )

# Initialize enhanced callback
callback = EnhancedRewardCallback(1_000_000, save_freq=10000)

# Create SAC model
model = SAC("MlpPolicy", env, verbose=0)

try:
    # Train the model
    model.learn(total_timesteps=1_000_000, callback=callback)

except KeyboardInterrupt:
    print("\n⏹️ Training interrupted by user. Saving progress...")

finally:
    # Save the model
    model_file = f"{callback.results_dir}/sac_drone_model.zip"
    model.save(model_file)
    print(f"🤖 Model saved as: {model_file}")

    # Create and save plots
    if callback.episode_rewards:
        # Plot rewards
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: Episode rewards
        plt.subplot(2, 2, 1)
        plt.plot(callback.episode_rewards, alpha=0.6, label="Episode Reward")
        if len(callback.episode_rewards) > 50:
            plt.plot(pd.Series(callback.episode_rewards).rolling(50).mean(), 
                    color='red', linewidth=2, label="50-episode MA")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Reward per Episode")
        plt.legend()
        plt.grid(True)
        
        # Subplot 2: Episode lengths
        plt.subplot(2, 2, 2)
        plt.plot(callback.episode_lengths, alpha=0.6, label="Episode Length")
        if len(callback.episode_lengths) > 50:
            plt.plot(pd.Series(callback.episode_lengths).rolling(50).mean(), 
                    color='orange', linewidth=2, label="50-episode MA")
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.title("Episode Length")
        plt.legend()
        plt.grid(True)
        
        # Subplot 3: Reward distribution
        plt.subplot(2, 2, 3)
        plt.hist(callback.episode_rewards, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel("Reward")
        plt.ylabel("Frequency")
        plt.title("Reward Distribution")
        plt.grid(True)
        
        # Subplot 4: Performance over time
        plt.subplot(2, 2, 4)
        episodes_per_window = 100
        if len(callback.episode_rewards) >= episodes_per_window:
            windowed_avg = [np.mean(callback.episode_rewards[i:i+episodes_per_window]) 
                           for i in range(0, len(callback.episode_rewards) - episodes_per_window + 1, episodes_per_window//2)]
            plt.plot(windowed_avg, marker='o', linewidth=2, label=f"Avg per {episodes_per_window} episodes")
            plt.xlabel(f"Window (each = {episodes_per_window//2} episodes)")
            plt.ylabel("Average Reward")
            plt.title("Learning Progress")
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plot_file = f"{callback.results_dir}/training_plots.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        # plt.show()
        print(f"📈 Plots saved as: {plot_file}")
    else:
        print("⚠️ No rewards recorded for plotting.")
        
    # Print final statistics
    if callback.episode_rewards:
        print(f"\n📊 FINAL STATISTICS:")
        print(f"   Episodes completed: {len(callback.episode_rewards)}")
        print(f"   Best reward: {np.max(callback.episode_rewards):.2f}")
        print(f"   Average reward: {np.mean(callback.episode_rewards):.2f}")
        print(f"   Final 100 episodes avg: {np.mean(callback.episode_rewards[-100:]):.2f}")
        print(f"   Total timesteps: {callback.num_timesteps:,}")