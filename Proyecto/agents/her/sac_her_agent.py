import gymnasium as gym
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm
from env_HER import GoalConditionedDroneEnvironment
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from gymnasium.wrappers import RecordVideo


class SafeSAC(SAC):
    """SAC with additional safety checks for batch size vs buffer size and HER compatibility"""

    def train(self, gradient_steps: int, batch_size: int = 64):
        """Override train to handle buffer size issues and HER tensor mismatches gracefully"""
        if self.replay_buffer.size() < batch_size:
            return  # Skip training if not enough samples

        # For HER, we need to be extremely conservative due to goal relabeling tensor issues
        # Try only the most stable batch sizes
        max_buffer_batch = max(
            1, self.replay_buffer.size() // 8
        )  # Even more conservative divisor

        # Only try batch sizes that are known to work well with HER
        safe_batch_sizes = [
            min(2, max_buffer_batch),  # Ultra minimal - most stable for HER
            1,  # Absolute last resort
        ]

        # Remove invalid sizes
        safe_batch_sizes = [
            size
            for size in safe_batch_sizes
            if size > 0 and size <= self.replay_buffer.size()
        ]

        if not safe_batch_sizes:
            print("No valid batch sizes available, skipping training step")
            return

        for attempt, safe_batch_size in enumerate(safe_batch_sizes):
            try:
                # For HER, reduce gradient steps to avoid accumulating tensor issues
                reduced_gradient_steps = min(gradient_steps, 1)
                return super().train(reduced_gradient_steps, safe_batch_size)
            except Exception as e:
                error_msg = str(e).lower()
                if any(
                    keyword in error_msg
                    for keyword in [
                        "batch_size",
                        "tensor",
                        "dimension",
                        "size",
                        "shape",
                        "broadcast",
                        "mismatch",
                    ]
                ):
                    if attempt == 0:  # Only print warning on first attempt
                        print(
                            f"Warning: HER tensor issue with batch_size={safe_batch_size}"
                        )

                    if attempt < len(safe_batch_sizes) - 1:
                        continue  # Try next batch size
                    else:
                        print(
                            "All batch sizes failed. Skipping this training step to avoid infinite loop."
                        )
                        return  # Give up gracefully
                else:
                    # Different error type, re-raise
                    raise e

        return  # Should not reach here


class HERTrainingConfig:
    """Configuration class for SAC+HER training parameters"""

    def __init__(
        self,
        render_mode="none",
        debug_mode=False,
        fast_mode=False,
        penalty_system="minimal",
    ):
        # Training parameters
        self.TOTAL_TIMESTEPS = 200_000 if fast_mode else 1_000_000
        self.MODEL_NAME = "sac_her_drone_model.zip"
        self.VIDEO_FOLDER = "video_her"
        self.VIDEO_RECORD_FREQUENCY = 50 if debug_mode else 200

        # Penalty system
        self.PENALTY_SYSTEM = penalty_system

        # HER specific parameters
        self.HER_PARAMS = {
            "n_sampled_goal": 4,  # Number of HER goals to sample per transition
            "goal_selection_strategy": GoalSelectionStrategy.FUTURE,  # Sample from future states
        }

        # SAC hyperparameters optimized for HER
        # Use ultra-conservative batch sizes to prevent tensor dimension mismatches
        # HER's goal relabeling is very sensitive to batch dimensions
        # Always use batch_size = 2 for maximum stability with HER
        batch_size = 2  # Fixed ultra-minimal size for HER stability

        # Override any larger values - HER cannot handle larger batches reliably
        if batch_size > 2:
            batch_size = 2
            print(f"🔧 Forcing batch_size=2 for HER stability")

        self.SAC_PARAMS = {
            "learning_rate": 1e-3,  # Slightly higher for HER
            "buffer_size": 100_000 if fast_mode else 1_000_000,  # Larger buffer for HER
            "learning_starts": self._get_learning_starts(
                render_mode, fast_mode, batch_size
            ),
            "batch_size": batch_size,
            "tau": 0.05,  # Faster target updates for HER
            "gamma": 0.98,
            "train_freq": 4,
            "gradient_steps": 1,
            "ent_coef": "auto",
            "target_update_interval": 1,
            "target_entropy": "auto",
            "replay_buffer_class": HerReplayBuffer,
            "replay_buffer_kwargs": self.HER_PARAMS,
        }

        # Environment parameters
        self.ENV_PARAMS = self._get_env_params(render_mode, debug_mode)

    def _get_learning_starts(self, render_mode, fast_mode, batch_size):
        """Determine learning_starts based on render mode and training speed"""
        # CRITICAL: HER requires learning_starts > max episode length
        # The environment has time_limit = 20 steps (1000 for mouse mode)
        # Each step = 5 physics frames, so max episode length ≈ 100 timesteps
        # We need learning_starts to be significantly larger than this

        # Very conservative base for HER stability
        base_learning_starts = 1000  # Much higher for ultra-small batch sizes

        # For batch_size=2, we need a lot more samples before training
        # HER needs extensive buffer diversity to work with such small batches
        minimum_starts = max(
            base_learning_starts, batch_size * 100
        )  # Much higher multiplier

        if render_mode == "human":
            return minimum_starts + 500  # Extra buffer for visual mode
        elif fast_mode:
            return minimum_starts  # Minimal but safe
        else:
            return minimum_starts + 500  # Conservative for normal mode

    def _get_env_params(self, render_mode, debug_mode):
        """Configure environment parameters based on training mode"""
        if render_mode == "human":
            return {
                "render_mode": "human",
                "render_every_frame": True,
                "mouse_target": debug_mode,
                "penalty_system": self.PENALTY_SYSTEM,
            }
        elif render_mode == "rgb_array":
            return {
                "render_mode": "rgb_array",
                "render_every_frame": debug_mode,
                "mouse_target": False,
                "penalty_system": self.PENALTY_SYSTEM,
            }
        else:
            return {
                "render_mode": None,
                "render_every_frame": False,
                "mouse_target": False,
                "penalty_system": self.PENALTY_SYSTEM,
            }


class HERRewardCallback(BaseCallback):
    """Enhanced callback for HER training with success rate tracking"""

    def __init__(self, total_timesteps, csv_file="her_training_log.csv"):
        super().__init__()
        self.episode_rewards = []
        self.episode_successes = []
        self.success_rate_window = []
        self.total_timesteps = total_timesteps
        self.pbar = None
        self.csv_file = csv_file
        self.episode_count = 0

        # Initialize CSV file with headers
        import csv

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
        """Initialize progress bar when training starts"""
        self.pbar = tqdm(total=self.total_timesteps, desc="🎯 Training SAC+HER")

    def _on_step(self) -> bool:
        """Called at each step to update progress and collect metrics"""
        infos = self.locals.get("infos", [])
        if infos and "episode" in infos[0]:
            episode_reward = infos[0]["episode"]["r"]
            episode_length = infos[0]["episode"]["l"]

            # Track if episode was successful (any reward > 0 means we reached at least one target)
            is_success = episode_reward > 0

            self.episode_rewards.append(episode_reward)
            self.episode_successes.append(is_success)
            self.episode_count += 1

            # Calculate recent success rate (last 100 episodes)
            recent_successes = self.episode_successes[-100:]
            success_rate = (
                sum(recent_successes) / len(recent_successes) if recent_successes else 0
            )

            # Write episode data to CSV
            import csv

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
        """Clean up progress bar when training ends"""
        if self.pbar:
            self.pbar.close()


class HERDroneTrainer:
    """Main trainer class for SAC+HER drone agent"""

    def __init__(self, config: HERTrainingConfig):
        self.config = config
        self.env = None
        self.model = None
        self.callback = None

    def setup_environment(self):
        """Create and wrap the goal-conditioned drone environment"""
        render_mode = self.config.ENV_PARAMS.get("render_mode")
        print(f"🏗️  Setting up HER environment (render_mode: {render_mode})...")

        # Create base environment
        base_env = GoalConditionedDroneEnvironment(**self.config.ENV_PARAMS)

        # Add video recording if needed
        if render_mode == "rgb_array":
            os.makedirs(self.config.VIDEO_FOLDER, exist_ok=True)
            base_env = RecordVideo(
                base_env,
                video_folder=self.config.VIDEO_FOLDER,
                episode_trigger=lambda ep: ep % self.config.VIDEO_RECORD_FREQUENCY == 0,
                name_prefix="her_drone_episode",
                disable_logger=True,
            )
            print(
                f"📹 Video recording enabled (every {self.config.VIDEO_RECORD_FREQUENCY} episodes)"
            )

        # Wrap in VecEnv (required for HER)
        self.env = DummyVecEnv([lambda: base_env])

        if render_mode == "human":
            print("👁️  Human rendering enabled - you'll see the game window")
        elif render_mode is None:
            print("🚀 Headless HER training mode - maximum speed")

        print("✅ HER environment setup complete")

    def setup_model(self, fresh_start=False):
        """Create or load SAC+HER model"""
        model_path = self.config.MODEL_NAME

        # Try to load existing model if not fresh start and model exists
        if not fresh_start and os.path.exists(model_path):
            try:
                print(f"🔄 Loading existing HER model from {model_path}")
                loaded_model = SAC.load(model_path, env=self.env)

                # Check if the loaded model has compatible parameters
                if hasattr(loaded_model, "policy"):
                    print("✅ HER model loaded successfully. Continuing training...")
                    self.model = loaded_model

                    # Reset learning_starts to continue training smoothly
                    if hasattr(self.model, "learning_starts"):
                        original_learning_starts = self.model.learning_starts
                        self.model.learning_starts = min(
                            original_learning_starts, self.model.num_timesteps
                        )
                        print(
                            f"🔧 Adjusted learning_starts from {original_learning_starts} to {self.model.learning_starts}"
                        )

                    print(f"📊 Resumed from timestep: {self.model.num_timesteps}")
                else:
                    print("⚠️ Loaded model appears incompatible, creating new model")
                    self._create_new_model()

            except Exception as e:
                print(f"❌ Failed to load existing model: {e}")
                print("🆕 Creating new model...")
                self._create_new_model()
        else:
            if fresh_start:
                print("🆕 Fresh start requested - creating new model")
            else:
                print("🆕 No existing model found - creating new model")
            self._create_new_model()

    def _create_new_model(self):
        """Create a new SAC+HER model"""
        print("🆕 Creating new SAC+HER model")
        print(
            f"   🎯 HER Strategy: {self.config.HER_PARAMS['goal_selection_strategy']}"
        )
        print(f"   🔄 Goals per transition: {self.config.HER_PARAMS['n_sampled_goal']}")

        self.model = SafeSAC(
            "MultiInputPolicy",  # Required for Dict observation spaces
            self.env,
            verbose=1,
            **self.config.SAC_PARAMS,
        )

    def setup_callback(self):
        """Initialize HER training callback"""
        self.callback = HERRewardCallback(self.config.TOTAL_TIMESTEPS)

    def train(self):
        """Main training loop with error handling"""
        print(
            f"🚀 Starting HER training for {self.config.TOTAL_TIMESTEPS:,} timesteps..."
        )
        print("💡 HER will learn from 'failures' by treating final positions as goals!")

        try:
            if self.model is None:
                raise RuntimeError("Model not initialized. Call setup_model() first.")

            self.model.learn(
                total_timesteps=self.config.TOTAL_TIMESTEPS,
                callback=self.callback,
                reset_num_timesteps=False,
            )
            print("🎉 HER training completed successfully!")

        except KeyboardInterrupt:
            print("\n⏹️  HER training interrupted by user. Saving progress...")

    def save_model(self):
        """Save the trained HER model"""
        if self.model is None:
            print("⚠️ No model to save.")
            return

        self.model.save(self.config.MODEL_NAME)
        print(f"💾 HER model saved as '{self.config.MODEL_NAME}'")

    def plot_results(self):
        """Generate and display HER training results"""
        if (
            self.callback is None
            or not hasattr(self.callback, "episode_rewards")
            or not self.callback.episode_rewards
        ):
            print("⚠️ No episode rewards recorded for plotting.")
            return

        print("📊 Generating HER training plots...")

        plt.figure(figsize=(15, 8))

        # Plot episode rewards
        plt.subplot(2, 2, 1)
        rewards = self.callback.episode_rewards
        episodes = range(1, len(rewards) + 1)
        plt.plot(episodes, rewards, alpha=0.6, color="blue", linewidth=1)

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
        plt.title("HER Training - Episode Rewards")
        plt.grid(True, alpha=0.3)

        # Plot success rate
        plt.subplot(2, 2, 2)
        successes = self.callback.episode_successes
        if successes:
            # Calculate rolling success rate
            window = 50
            success_rates = []
            for i in range(len(successes)):
                start_idx = max(0, i - window + 1)
                rate = sum(successes[start_idx : i + 1]) / (i - start_idx + 1)
                success_rates.append(rate)

            plt.plot(episodes, success_rates, color="green", linewidth=2)
            plt.xlabel("Episode")
            plt.ylabel("Success Rate")
            plt.title(f"Success Rate (Rolling {window} episodes)")
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)

        # Plot reward distribution
        plt.subplot(2, 2, 3)
        plt.hist(rewards, bins=30, alpha=0.7, color="purple", edgecolor="black")
        plt.xlabel("Episode Reward")
        plt.ylabel("Frequency")
        plt.title("Reward Distribution")
        plt.grid(True, alpha=0.3)

        # Plot success count over time
        plt.subplot(2, 2, 4)
        if successes:
            cumulative_successes = np.cumsum(successes)
            plt.plot(episodes, cumulative_successes, color="orange", linewidth=2)
            plt.xlabel("Episode")
            plt.ylabel("Cumulative Successes")
            plt.title("Total Successful Episodes")
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print HER statistics
        if successes:
            total_successes = sum(successes)
            final_success_rate = sum(successes[-100:]) / min(100, len(successes))

            print(f"\n🎯 HER Training Statistics:")
            print(f"   Total Episodes: {len(rewards)}")
            print(f"   Successful Episodes: {total_successes}")
            print(f"   Overall Success Rate: {total_successes/len(successes):.2%}")
            print(f"   Final Success Rate (last 100): {final_success_rate:.2%}")
            print(f"   Best Episode Reward: {max(rewards):.2f}")
            print(f"   Average Episode Reward: {sum(rewards)/len(rewards):.2f}")

    def run_full_training(self, fresh_start=False):
        """Complete HER training pipeline"""
        self.setup_environment()
        self.setup_model(fresh_start=fresh_start)
        self.setup_callback()

        try:
            self.train()
        finally:
            self.save_model()
            self.plot_results()

    def test_model(self, num_episodes=5, render=True):
        """Test the trained model to evaluate performance"""
        if not self.model:
            print("⚠️ No model loaded. Loading from file...")
            if os.path.exists(self.config.MODEL_NAME):
                self.model = SAC.load(self.config.MODEL_NAME)
                print(f"✅ Model loaded from {self.config.MODEL_NAME}")
            else:
                print(f"❌ No trained model found at {self.config.MODEL_NAME}")
                return

        # Create test environment with rendering if requested
        test_env_params = self.config.ENV_PARAMS.copy()
        if render:
            test_env_params["render_mode"] = "human"
            test_env_params["render_every_frame"] = True
            print(
                f"🎮 Testing model with visual rendering for {num_episodes} episodes..."
            )
        else:
            test_env_params["render_mode"] = None
            print(f"🚀 Testing model headless for {num_episodes} episodes...")

        test_env = GoalConditionedDroneEnvironment(**test_env_params)
        test_env = DummyVecEnv([lambda: test_env])

        episode_rewards = []
        success_count = 0

        for episode in range(num_episodes):
            obs = test_env.reset()
            episode_reward = 0
            done = False
            step_count = 0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = test_env.step(action)
                episode_reward += reward[0]
                step_count += 1

                if done:
                    episode_rewards.append(episode_reward)
                    if episode_reward > 0:
                        success_count += 1

                    print(
                        f"   Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {step_count}"
                    )

        test_env.close()

        # Print test results
        avg_reward = sum(episode_rewards) / len(episode_rewards)
        success_rate = success_count / num_episodes

        print(f"\n🎯 Test Results:")
        print(f"   Episodes: {num_episodes}")
        print(f"   Average Reward: {avg_reward:.2f}")
        print(f"   Success Rate: {success_rate:.2%}")
        print(f"   Best Episode: {max(episode_rewards):.2f}")

        return {
            "avg_reward": avg_reward,
            "success_rate": success_rate,
            "episode_rewards": episode_rewards,
        }


def parse_arguments():
    """Parse command line arguments for HER training configuration"""
    parser = argparse.ArgumentParser(
        description="Train SAC+HER agent for goal-conditioned drone environment"
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
        help="Enable debug mode (more frequent video recording, mouse control for goal setting)",
    )

    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast training mode (fewer timesteps, smaller buffer - good for testing HER)",
    )

    parser.add_argument(
        "--timesteps", type=int, help="Override total training timesteps"
    )

    parser.add_argument(
        "--penalty",
        choices=["none", "minimal", "medium", "full"],
        default="minimal",
        help="Penalty system: 'none' (pure sparse), 'minimal' (-0.1), 'medium' (-0.3), 'full' (-1.0)",
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: load trained model and evaluate performance with rendering",
    )

    parser.add_argument(
        "--test-episodes",
        type=int,
        default=5,
        help="Number of episodes to run in test mode",
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Force training from scratch (ignore existing model)",
    )

    return parser.parse_args()


def main():
    """Main entry point for HER training"""
    # Parse command line arguments
    args = parse_arguments()

    # Create HER configuration
    config = HERTrainingConfig(
        render_mode=args.render,
        debug_mode=args.debug,
        fast_mode=args.fast,
        penalty_system=args.penalty,
    )

    # Override timesteps if specified
    if args.timesteps:
        config.TOTAL_TIMESTEPS = args.timesteps

    # Create HER trainer
    trainer = HERDroneTrainer(config)

    if args.test:
        # Test mode: evaluate trained model
        print("🧪 TEST MODE: Evaluating trained model performance")
        print(f"   Test Episodes: {args.test_episodes}")
        print(f"   Visual Rendering: Yes")
        print()
        trainer.test_model(num_episodes=args.test_episodes, render=True)
    else:
        # Training mode: print configuration and train
        print("🎯 HER Training Configuration:")
        print(f"   Render Mode: {args.render}")
        print(f"   Debug Mode: {args.debug}")
        print(f"   Fast Mode: {args.fast}")
        print(f"   Penalty System: {args.penalty}")
        print(f"   Total Timesteps: {config.TOTAL_TIMESTEPS:,}")
        print(f"   HER Strategy: {config.HER_PARAMS['goal_selection_strategy']}")
        print(f"   Goals per Transition: {config.HER_PARAMS['n_sampled_goal']}")
        print(f"   Learning Starts: {config.SAC_PARAMS['learning_starts']:,}")
        print(f"   Batch Size: {config.SAC_PARAMS['batch_size']}")
        print(f"   Video Recording: {'Yes' if args.render == 'rgb_array' else 'No'}")
        print()
        print("💡 HER Magic: Every 'failed' episode becomes a successful one!")
        print(
            "   The agent will learn from reaching ANY position, not just the target!"
        )
        if args.penalty != "none":
            penalty_values = {"minimal": "-0.1", "medium": "-0.3", "full": "-1.0"}
            print(
                f"   🚨 Crash Penalty: {penalty_values[args.penalty]} (encourages safety)"
            )
        print()

        # Run training
        trainer.run_full_training(fresh_start=args.fresh)


if __name__ == "__main__":
    main()
