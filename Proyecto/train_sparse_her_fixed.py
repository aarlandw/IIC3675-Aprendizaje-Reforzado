#!/usr/bin/env python3
"""
Fixed HER training with proper buffer handling
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
import pickle


class SparseHERCallback(BaseCallback):
    """Enhanced callback with proper buffer handling"""

    def __init__(self, save_freq=10_000, model_name="sparse_her_balloon"):
        super().__init__()
        self.save_freq = save_freq
        self.model_name = model_name
        self.episode_rewards = []
        self.balloons_collected = []
        self.success_rates = []
        self.total_timesteps = 0
        self.best_performance = 0
        self.warmup_complete = False

        # Load previous stats if they exist
        self.load_training_stats()

    def _on_step(self):
        self.total_timesteps += 1

        # Check if we're past warmup period
        if not self.warmup_complete and self.num_timesteps > 2000:
            print("🔥 Warmup period complete - HER buffer ready!")
            self.warmup_complete = True

        if len(self.locals.get("infos", [])) > 0:
            info = self.locals["infos"][0]
            if "episode" in info:
                episode_reward = info["episode"]["r"]
                self.episode_rewards.append(episode_reward)

                balloons = info.get("balloons_collected", 0)
                is_success = info.get("is_success", False)

                self.balloons_collected.append(balloons)
                self.success_rates.append(float(is_success))

                # Check for best performance (only after warmup)
                if self.warmup_complete and len(self.balloons_collected) >= 10:
                    recent_performance = np.mean(self.balloons_collected[-10:])
                    if recent_performance > self.best_performance:
                        self.best_performance = recent_performance
                        print(
                            f"🏆 New best performance: {recent_performance:.2f} balloons!"
                        )
                        self.model.save(f"{self.model_name}_best")

                # Print progress
                if len(self.episode_rewards) % 25 == 0:
                    avg_balloons = (
                        np.mean(self.balloons_collected[-25:])
                        if len(self.balloons_collected) >= 25
                        else np.mean(self.balloons_collected)
                    )
                    avg_success = (
                        np.mean(self.success_rates[-25:])
                        if len(self.success_rates) >= 25
                        else np.mean(self.success_rates)
                    )

                    warmup_status = (
                        "🔥 LEARNING" if self.warmup_complete else "🌡️ WARMUP"
                    )
                    print(
                        f"{warmup_status} Episode {len(self.episode_rewards)}: "
                        f"Balloons: {avg_balloons:.1f}, "
                        f"Success: {avg_success:.2%}, "
                        f"Steps: {self.num_timesteps:,}"
                    )

        # Auto-save checkpoint (only after warmup)
        if self.warmup_complete and self.total_timesteps % self.save_freq == 0:
            self.save_checkpoint()

        return True

    def save_checkpoint(self):
        """Save training checkpoint"""
        print(f"💾 Saving checkpoint at {self.num_timesteps:,} timesteps...")

        # Save model
        self.model.save(f"{self.model_name}_checkpoint")

        # Save training statistics
        stats = {
            "episode_rewards": self.episode_rewards,
            "balloons_collected": self.balloons_collected,
            "success_rates": self.success_rates,
            "total_timesteps": self.num_timesteps,  # Use actual timesteps
            "best_performance": self.best_performance,
        }

        with open(f"{self.model_name}_training_stats.pkl", "wb") as f:
            pickle.dump(stats, f)

        print(f"✅ Checkpoint saved!")

    def load_training_stats(self):
        """Load previous training statistics"""
        stats_file = f"{self.model_name}_training_stats.pkl"
        if os.path.exists(stats_file):
            try:
                with open(stats_file, "rb") as f:
                    stats = pickle.load(f)

                self.episode_rewards = stats.get("episode_rewards", [])
                self.balloons_collected = stats.get("balloons_collected", [])
                self.success_rates = stats.get("success_rates", [])
                self.total_timesteps = stats.get("total_timesteps", 0)
                self.best_performance = stats.get("best_performance", 0)

                print(f"📊 Loaded previous training stats:")
                print(f"   Episodes: {len(self.episode_rewards)}")
                print(f"   Previous timesteps: {self.total_timesteps:,}")
                print(f"   Best performance: {self.best_performance:.2f} balloons")

            except Exception as e:
                print(f"⚠️ Could not load training stats: {e}")


def train_her_with_proper_resume(timesteps=100_000, model_name="sparse_her_balloon"):
    """Train HER with proper buffer handling"""

    print("🎯 Training HER with PROPER BUFFER HANDLING")
    print("=" * 60)

    # Create environment
    base_env = DroneEnvironment(render_mode=None)
    
    if hasattr(base_env, 'max_episode_steps'):
        base_env.max_episode_steps = 10_000
    
    her_env = SparseRewardHERWrapper(base_env, goal_threshold=25.0)
    vec_env = DummyVecEnv([lambda: her_env])

    # Check for existing model
    checkpoint_path = f"{model_name}_checkpoint.zip"

    if os.path.exists(checkpoint_path):
        print(f"🔄 Found existing checkpoint: {checkpoint_path}")
        print("⚠️  IMPORTANT: Replay buffer will be empty and needs warmup!")

        try:
            # Load model with environment
            model = SAC.load(checkpoint_path, env=vec_env)
            print("✅ Model loaded successfully!")

            # CRITICAL: Set learning_starts for buffer warmup
            original_learning_starts = model.learning_starts
            model.learning_starts = 2000  # Force warmup period

            print(f"🌡️  Setting warmup period: {model.learning_starts} steps")
            print("   During warmup: collecting experience, no learning")
            print("   After warmup: normal HER learning resumes")

        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            print("🆕 Creating new model instead...")
            model = None
    else:
        print("🆕 No checkpoint found, creating new model...")
        model = None

    # Create new model if needed
    if model is None:
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
            learning_starts=2000,  # Warmup period
            verbose=1,
            device="auto",
        )
        print("🧠 New SAC+HER model created!")

    # Setup callback
    callback = SparseHERCallback(save_freq=15_000, model_name=model_name)

    print(f"🚀 Training for {timesteps:,} timesteps...")
    print("📊 Progress Legend:")
    print("   🌡️ WARMUP = Building replay buffer")
    print("   🔥 LEARNING = HER actively learning")

    start_time = time.time()

    try:
        model.learn(
            total_timesteps=timesteps,
            callback=callback,
            progress_bar=True,
            reset_num_timesteps=True,  # Start fresh timestep counting
        )

        training_time = time.time() - start_time

        # Final save
        print(f"💾 Saving final model...")
        model.save(model_name)
        callback.save_checkpoint()

        print(f"✅ Training complete!")
        print(f"   Training time: {training_time/60:.1f} minutes")
        print(f"   Total timesteps: {timesteps:,}")

        if len(callback.balloons_collected) > 0:
            final_performance = (
                np.mean(callback.balloons_collected[-50:])
                if len(callback.balloons_collected) >= 50
                else np.mean(callback.balloons_collected)
            )
            print(f"   Final performance: {final_performance:.1f} balloons")
            print(f"   Best performance: {callback.best_performance:.1f} balloons")

    except Exception as e:
        print(f"❌ Training error: {e}")
        print("💾 Saving emergency checkpoint...")
        model.save(f"{model_name}_emergency")

    return model


if __name__ == "__main__":
    # Train with proper buffer handling
    train_her_with_proper_resume(timesteps=500_000)
