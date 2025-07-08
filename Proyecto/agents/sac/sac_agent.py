import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
from env_SAC import DroneEnvironment
import matplotlib.pyplot as plt
import os
import argparse
from gymnasium.wrappers import RecordVideo


class TrainingConfig:
    """Configuration class for SAC training parameters"""

    def __init__(self, render_mode="none", debug_mode=False, fast_mode=False):
        # Training parameters
        self.TOTAL_TIMESTEPS = 100_000 if fast_mode else 1_000_000
        self.MODEL_NAME = "sac_drone_model.zip"
        self.VIDEO_FOLDER = "video2"
        self.VIDEO_RECORD_FREQUENCY = (
            50 if debug_mode else 200
        )  # Record more frequently in debug

        # SAC hyperparameters optimized for drone environment
        self.SAC_PARAMS = {
            "learning_rate": 3e-4,
            "buffer_size": (
                50000 if fast_mode else 300000
            ),  # Smaller buffer for fast mode
            "learning_starts": 1000 if fast_mode else 10000,
            "batch_size": 256,
            "tau": 0.02,
            "gamma": 0.98,
            "train_freq": 4,
            "gradient_steps": 1,
            "ent_coef": "auto",
            "target_update_interval": 1,
            "target_entropy": "auto",
        }

        # Environment parameters - dynamically set based on mode
        self.ENV_PARAMS = self._get_env_params(render_mode, debug_mode)

    def _get_env_params(self, render_mode, debug_mode):
        """Configure environment parameters based on training mode"""
        if render_mode == "human":
            return {
                "render_mode": "human",
                "render_every_frame": True,
                "mouse_target": debug_mode,  # Allow mouse control in debug mode
            }
        elif render_mode == "rgb_array":
            return {
                "render_mode": "rgb_array",
                "render_every_frame": debug_mode,  # Only render every frame in debug
                "mouse_target": False,
            }
        else:  # render_mode == "none"
            return {
                "render_mode": None,
                "render_every_frame": False,
                "mouse_target": False,
            }


class RewardCallback(BaseCallback):
    """Callback to track episode rewards and show progress"""

    def __init__(self, total_timesteps):
        super().__init__()
        self.episode_rewards = []
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self):
        """Initialize progress bar when training starts"""
        self.pbar = tqdm(total=self.total_timesteps, desc="🚁 Training Drone")

    def _on_step(self) -> bool:
        """Called at each step to update progress and collect rewards"""
        infos = self.locals.get("infos", [])
        if infos and "episode" in infos[0]:
            episode_reward = infos[0]["episode"]["r"]
            episode_length = infos[0]["episode"]["l"]
            self.episode_rewards.append(episode_reward)

            # Update progress bar with latest episode info
            self.pbar.set_postfix(
                {
                    "Episode Reward": f"{episode_reward:.2f}",
                    "Targets Collected": episode_length,
                    "Episodes": len(self.episode_rewards),
                }
            )

        self.pbar.update(1)
        return True

    def _on_training_end(self):
        """Clean up progress bar when training ends"""
        if self.pbar:
            self.pbar.close()


class DroneTrainer:
    """Main trainer class for the drone SAC agent"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.env = None
        self.model = None
        self.callback = None

    def setup_environment(self):
        """Create and wrap the drone environment"""
        render_mode = self.config.ENV_PARAMS.get("render_mode")
        print(f"🏗️  Setting up environment (render_mode: {render_mode})...")

        # Create base environment
        self.env = DroneEnvironment(**self.config.ENV_PARAMS)

        # Only add video recording if we're using rgb_array mode
        if render_mode == "rgb_array":
            os.makedirs(self.config.VIDEO_FOLDER, exist_ok=True)
            self.env = RecordVideo(
                self.env,
                video_folder=self.config.VIDEO_FOLDER,
                episode_trigger=lambda ep: ep % self.config.VIDEO_RECORD_FREQUENCY == 0,
                name_prefix="drone_episode",
                disable_logger=True,
            )
            print(
                f"📹 Video recording enabled (every {self.config.VIDEO_RECORD_FREQUENCY} episodes)"
            )
        elif render_mode == "human":
            print("👁️  Human rendering enabled - you'll see the game window")
        else:
            print("🚀 Headless training mode - maximum speed")

        print("✅ Environment setup complete")

    def setup_model(self):
        """Create or load SAC model"""
        model_path = self.config.MODEL_NAME

        if os.path.exists(model_path):
            print(f"🔄 Loading existing model from {model_path}")
            self.model = SAC.load(model_path, env=self.env)
            print("✅ Model loaded successfully. Continuing training...")
        else:
            print("🆕 Creating new SAC model")
            self.model = SAC("MlpPolicy", self.env, verbose=1, **self.config.SAC_PARAMS)

    def setup_callback(self):
        """Initialize training callback"""
        self.callback = RewardCallback(self.config.TOTAL_TIMESTEPS)

    def train(self):
        """Main training loop with error handling"""
        print(f"🚀 Starting training for {self.config.TOTAL_TIMESTEPS:,} timesteps...")

        try:
            self.model.learn(
                total_timesteps=self.config.TOTAL_TIMESTEPS,
                callback=self.callback,
                reset_num_timesteps=False,
            )
            print("🎉 Training completed successfully!")

        except KeyboardInterrupt:
            print("\n⏹️  Training interrupted by user. Saving progress...")

    def save_model(self):
        """Save the trained model"""
        self.model.save(self.config.MODEL_NAME)
        print(f"💾 Model saved as '{self.config.MODEL_NAME}'")

    def plot_results(self):
        """Generate and display training results"""
        if not self.callback.episode_rewards:
            print("⚠️ No episode rewards recorded for plotting.")
            return

        print("📊 Generating training plots...")

        plt.figure(figsize=(12, 6))

        # Plot episode rewards
        plt.subplot(1, 2, 1)
        rewards = self.callback.episode_rewards
        episodes = range(1, len(rewards) + 1)

        plt.plot(episodes, rewards, alpha=0.6, color="blue", linewidth=1)

        # Add moving average for trend
        if len(rewards) > 10:
            window_size = min(50, len(rewards) // 10)
            moving_avg = []
            for i in range(len(rewards)):
                start_idx = max(0, i - window_size + 1)
                moving_avg.append(sum(rewards[start_idx : i + 1]) / (i - start_idx + 1))
            plt.plot(
                episodes,
                moving_avg,
                color="red",
                linewidth=2,
                label=f"Moving Avg ({window_size})",
            )
            plt.legend()

        plt.xlabel("Episode")
        plt.ylabel("Episode Reward")
        plt.title("Training Progress - Episode Rewards")
        plt.grid(True, alpha=0.3)

        # Plot reward distribution
        plt.subplot(1, 2, 2)
        plt.hist(rewards, bins=30, alpha=0.7, color="green", edgecolor="black")
        plt.xlabel("Episode Reward")
        plt.ylabel("Frequency")
        plt.title("Reward Distribution")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print statistics
        print(f"\n📈 Training Statistics:")
        print(f"   Total Episodes: {len(rewards)}")
        print(f"   Best Episode Reward: {max(rewards):.2f}")
        print(f"   Average Reward: {sum(rewards)/len(rewards):.2f}")
        print(
            f"   Final 10 Episodes Avg: {sum(rewards[-10:])/min(10, len(rewards)):.2f}"
        )

    def run_full_training(self):
        """Complete training pipeline"""
        self.setup_environment()
        self.setup_model()
        self.setup_callback()

        try:
            self.train()
        finally:
            self.save_model()
            self.plot_results()


def parse_arguments():
    """Parse command line arguments for training configuration"""
    parser = argparse.ArgumentParser(
        description="Train SAC agent for drone environment"
    )

    parser.add_argument(
        "--render",
        choices=["none", "human", "rgb_array"],
        default="none",
        help="Rendering mode: 'none' (fastest), 'human' (visual), 'rgb_array' (video recording)",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (more frequent video recording, mouse control available)",
    )

    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast training mode (fewer timesteps, smaller buffer - good for testing)",
    )

    parser.add_argument(
        "--timesteps", type=int, help="Override total training timesteps"
    )

    return parser.parse_args()


def main():
    """Main entry point"""
    # Parse command line arguments
    args = parse_arguments()

    # Create configuration based on arguments
    config = TrainingConfig(
        render_mode=args.render, debug_mode=args.debug, fast_mode=args.fast
    )

    # Override timesteps if specified
    if args.timesteps:
        config.TOTAL_TIMESTEPS = args.timesteps

    # Print configuration summary
    print("🎮 Training Configuration:")
    print(f"   Render Mode: {args.render}")
    print(f"   Debug Mode: {args.debug}")
    print(f"   Fast Mode: {args.fast}")
    print(f"   Total Timesteps: {config.TOTAL_TIMESTEPS:,}")
    print(f"   Video Recording: {'Yes' if args.render == 'rgb_array' else 'No'}")
    print()

    # Create and run trainer
    trainer = DroneTrainer(config)
    trainer.run_full_training()


if __name__ == "__main__":
    main()
