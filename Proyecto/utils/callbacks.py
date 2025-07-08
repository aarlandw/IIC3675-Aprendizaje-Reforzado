"""
Training callbacks for monitoring and saving models during training.
"""

import os
import csv
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm


class TrainingCallback(BaseCallback):
    """
    Enhanced callback for training with progress tracking and model saving.
    """

    def __init__(
        self, total_timesteps, save_freq=50_000, model_name="model", csv_file=None
    ):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.save_freq = save_freq
        self.model_name = model_name
        self.episode_rewards = []
        self.episode_lengths = []
        self.pbar = None

        # CSV logging
        if csv_file is None:
            csv_file = f"logs/{model_name}_training.csv"
        self.csv_file = csv_file
        self.episode_count = 0

        # Initialize CSV file
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)
        with open(self.csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "reward", "length", "timestep"])

    def _on_training_start(self):
        """Initialize progress bar when training starts."""
        self.pbar = tqdm(
            total=self.total_timesteps, desc=f"🎯 Training {self.model_name}"
        )

    def _on_step(self) -> bool:
        """Called at each step to update progress and collect metrics."""
        # Check for episode completion
        infos = self.locals.get("infos", [])
        if infos and "episode" in infos[0]:
            episode_reward = infos[0]["episode"]["r"]
            episode_length = infos[0]["episode"]["l"]

            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.episode_count += 1

            # Log to CSV
            with open(self.csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        self.episode_count,
                        episode_reward,
                        episode_length,
                        self.num_timesteps,
                    ]
                )

            # Update progress bar
            if self.pbar:
                avg_reward = sum(self.episode_rewards[-10:]) / min(
                    10, len(self.episode_rewards)
                )
                self.pbar.set_postfix(
                    {
                        "Avg Reward (10)": f"{avg_reward:.2f}",
                        "Episodes": len(self.episode_rewards),
                        "Last Reward": f"{episode_reward:.2f}",
                    }
                )

        # Update progress bar
        if self.pbar:
            self.pbar.update(1)

        # Save model periodically
        if self.num_timesteps % self.save_freq == 0:
            checkpoint_path = (
                f"models/{self.model_name}_checkpoint_{self.num_timesteps}.zip"
            )
            self.model.save(checkpoint_path)
            if self.pbar:
                self.pbar.write(f"💾 Checkpoint saved: {checkpoint_path}")

        return True

    def _on_training_end(self):
        """Clean up progress bar when training ends."""
        if self.pbar:
            self.pbar.close()


class HERCallback(BaseCallback):
    """
    Specialized callback for HER training with success rate tracking.
    """

    def __init__(self, total_timesteps, csv_file="logs/her_training.csv"):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.episode_rewards = []
        self.episode_successes = []
        self.pbar = None
        self.csv_file = csv_file
        self.episode_count = 0

        # Initialize CSV file
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)
        with open(self.csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "episode",
                    "reward",
                    "length",
                    "success",
                    "success_rate_100",
                    "timestep",
                ]
            )

    def _on_training_start(self):
        """Initialize progress bar when training starts."""
        self.pbar = tqdm(total=self.total_timesteps, desc="🎯 Training SAC+HER")

    def _on_step(self) -> bool:
        """Called at each step to update progress and collect metrics."""
        infos = self.locals.get("infos", [])
        if infos and "episode" in infos[0]:
            episode_reward = infos[0]["episode"]["r"]
            episode_length = infos[0]["episode"]["l"]

            # Track success (any positive reward means we reached at least one target)
            is_success = episode_reward > 0

            self.episode_rewards.append(episode_reward)
            self.episode_successes.append(is_success)
            self.episode_count += 1

            # Calculate recent success rate (last 100 episodes)
            recent_successes = self.episode_successes[-100:]
            success_rate = (
                sum(recent_successes) / len(recent_successes) if recent_successes else 0
            )

            # Log to CSV
            with open(self.csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        self.episode_count,
                        episode_reward,
                        episode_length,
                        is_success,
                        success_rate,
                        self.num_timesteps,
                    ]
                )

            # Update progress bar with HER-specific metrics
            if self.pbar:
                self.pbar.set_postfix(
                    {
                        "Episode Reward": f"{episode_reward:.2f}",
                        "Targets Hit": episode_length,
                        "Success Rate": f"{success_rate:.2%}",
                        "Episodes": len(self.episode_rewards),
                    }
                )

        if self.pbar:
            self.pbar.update(1)

        return True

    def _on_training_end(self):
        """Clean up progress bar when training ends."""
        if self.pbar:
            self.pbar.close()
