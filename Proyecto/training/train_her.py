#!/usr/bin/env python3
"""
Clean HER training script for goal-conditioned quadcopter environment.

This script provides a simplified interface for training SAC+HER agents.
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.vec_env import DummyVecEnv
from environments.goal_conditioned_env import GoalConditionedDroneEnvironment
from utils.callbacks import HERCallback
from utils.plotting import plot_her_results


def create_her_model(env, config):
    """Create SAC+HER model with given configuration."""
    return SAC(
        "MultiInputPolicy",  # Required for goal-conditioned environments
        env,
        learning_rate=config.get("learning_rate", 1e-3),
        buffer_size=config.get("buffer_size", 1_000_000),
        learning_starts=config.get("learning_starts", 1000),
        batch_size=config.get("batch_size", 2),  # Ultra-conservative for HER
        tau=config.get("tau", 0.05),
        gamma=config.get("gamma", 0.98),
        train_freq=config.get("train_freq", 4),
        gradient_steps=config.get("gradient_steps", 1),
        ent_coef=config.get("ent_coef", "auto"),
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={
            "n_sampled_goal": config.get("n_sampled_goal", 4),
            "goal_selection_strategy": config.get(
                "goal_selection_strategy", GoalSelectionStrategy.FUTURE
            ),
        },
        verbose=1,
    )


def train_her(
    total_timesteps=1_000_000,
    model_name="sac_her_quadcopter_model",
    render_mode=None,
    penalty_system="minimal",
):
    """
    Train SAC+HER agent on goal-conditioned quadcopter environment.

    Args:
        total_timesteps: Total training timesteps
        model_name: Name for saved model
        render_mode: Rendering mode ('human', 'rgb_array', or None)
        penalty_system: Penalty system ('none', 'minimal', 'medium', 'full')
    """
    print("🎯 Starting SAC+HER Training")
    print(f"   Total timesteps: {total_timesteps:,}")
    print(f"   Render mode: {render_mode}")
    print(f"   Model name: {model_name}")
    print(f"   Penalty system: {penalty_system}")
    print("💡 HER will learn from 'failures' by treating final positions as goals!")

    # Create environment
    env_params = {
        "render_mode": render_mode,
        "penalty_system": penalty_system,
        "render_every_frame": render_mode == "human",
        "mouse_target": False,  # Set to True for interactive goal setting
    }

    base_env = GoalConditionedDroneEnvironment(**env_params)
    env = DummyVecEnv([lambda: base_env])  # HER requires vectorized environment
    print("✅ Goal-conditioned environment created")

    # HER configuration
    her_config = {
        "learning_rate": 1e-3,
        "buffer_size": 500_000,
        "learning_starts": 1000,  # Conservative for HER
        "batch_size": 2,  # Ultra-conservative to prevent tensor issues
        "tau": 0.05,
        "gamma": 0.98,
        "train_freq": 4,
        "gradient_steps": 1,
        "ent_coef": "auto",
        "n_sampled_goal": 4,
        "goal_selection_strategy": GoalSelectionStrategy.FUTURE,
    }

    # Create model
    model = create_her_model(env, her_config)
    print("✅ SAC+HER model created")
    print(f"   HER Strategy: {her_config['goal_selection_strategy']}")
    print(f"   Goals per transition: {her_config['n_sampled_goal']}")
    print(
        f"   Batch size: {her_config['batch_size']} (ultra-conservative for stability)"
    )

    # Create callback for tracking progress
    callback = HERCallback(
        total_timesteps=total_timesteps,
        csv_file=f"logs/{model_name}_training.csv",
    )

    try:
        # Train the model
        print("🚀 Starting HER training...")
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=False,  # We use our own progress bar
        )
        print("🎉 HER training completed!")

    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")

    finally:
        # Save final model
        model_path = f"models/{model_name}.zip"
        model.save(model_path)
        print(f"💾 Model saved: {model_path}")

        # Plot results if we have episode data
        if (
            hasattr(callback, "episode_rewards")
            and callback.episode_rewards
            and hasattr(callback, "episode_successes")
            and callback.episode_successes
        ):
            plot_her_results(
                callback.episode_rewards,
                callback.episode_successes,
                f"{model_name} HER Training",
            )

        env.close()

    return model


def test_her_model(model_path, num_episodes=5, render=True):
    """
    Test a trained HER model.

    Args:
        model_path: Path to the saved model
        num_episodes: Number of episodes to test
        render: Whether to render the environment
    """
    print(f"🧪 Testing HER model: {model_path}")

    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return

    # Load model
    model = SAC.load(model_path)
    print("✅ Model loaded successfully")

    # Create test environment
    env_params = {
        "render_mode": "human" if render else None,
        "penalty_system": "minimal",
        "render_every_frame": True,
        "mouse_target": False,
    }

    base_env = GoalConditionedDroneEnvironment(**env_params)
    env = DummyVecEnv([lambda: base_env])

    episode_rewards = []
    success_count = 0

    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        step_count = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            step_count += 1

            if done[0]:
                episode_rewards.append(episode_reward)
                if episode_reward > 0:
                    success_count += 1

                print(
                    f"   Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {step_count}"
                )
                break

    env.close()

    # Print test results
    avg_reward = sum(episode_rewards) / len(episode_rewards)
    success_rate = success_count / num_episodes

    print(f"\n🎯 HER Test Results:")
    print(f"   Episodes: {num_episodes}")
    print(f"   Average Reward: {avg_reward:.2f}")
    print(f"   Success Rate: {success_rate:.2%}")
    print(f"   Best Episode: {max(episode_rewards):.2f}")


def main():
    """Main entry point for HER training."""
    parser = argparse.ArgumentParser(
        description="Train SAC+HER agent for goal-conditioned quadcopter control"
    )

    parser.add_argument(
        "--timesteps",
        "-t",
        type=int,
        default=1_000_000,
        help="Total training timesteps (default: 1,000,000)",
    )

    parser.add_argument(
        "--model-name",
        "-m",
        type=str,
        default="sac_her_quadcopter_model",
        help="Name for the saved model (default: sac_her_quadcopter_model)",
    )

    parser.add_argument(
        "--render",
        "-r",
        choices=["none", "human", "rgb_array"],
        default="none",
        help="Rendering mode (default: none)",
    )

    parser.add_argument(
        "--penalty",
        choices=["none", "minimal", "medium", "full"],
        default="minimal",
        help="Penalty system for crashes (default: minimal)",
    )

    parser.add_argument(
        "--test", action="store_true", help="Test mode: load and evaluate trained model"
    )

    parser.add_argument(
        "--test-episodes",
        type=int,
        default=5,
        help="Number of episodes for testing (default: 5)",
    )

    args = parser.parse_args()

    # Ensure output directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    if args.test:
        # Test mode
        model_path = f"models/{args.model_name}.zip"
        test_her_model(model_path, args.test_episodes, render=True)
    else:
        # Training mode
        render_mode = None if args.render == "none" else args.render

        train_her(
            total_timesteps=args.timesteps,
            model_name=args.model_name,
            render_mode=render_mode,
            penalty_system=args.penalty,
        )


if __name__ == "__main__":
    main()
