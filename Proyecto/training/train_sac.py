#!/usr/bin/env python3
"""
Clean SAC training script for quadcopter environment.

This script provides a simplified interface for training SAC agents.
"""

import argparse
import os
import sys
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import SAC
from environments.base_env import QuadcopterEnv
from utils.callbacks import TrainingCallback
from utils.plotting import plot_training_results


def create_sac_model(env, config):
    """Create SAC model with given configuration."""
    return SAC(
        "MlpPolicy",
        env,
        learning_rate=config.get("learning_rate", 3e-4),
        buffer_size=config.get("buffer_size", 1_000_000),
        learning_starts=config.get("learning_starts", 1000),
        batch_size=config.get("batch_size", 64),
        tau=config.get("tau", 0.005),
        gamma=config.get("gamma", 0.99),
        train_freq=config.get("train_freq", 1),
        gradient_steps=config.get("gradient_steps", 1),
        ent_coef=config.get("ent_coef", "auto"),
        verbose=1,
    )


def train_sac(
    total_timesteps=500_000,
    model_name="sac_quadcopter_model",
    render_mode=None,
    save_freq=50_000,
):
    """
    Train SAC agent on quadcopter environment.

    Args:
        total_timesteps: Total training timesteps
        model_name: Name for saved model
        render_mode: Rendering mode ('human', 'rgb_array', or None)
        save_freq: Frequency to save intermediate models
    """
    print("🚀 Starting SAC Training")
    print(f"   Total timesteps: {total_timesteps:,}")
    print(f"   Render mode: {render_mode}")
    print(f"   Model name: {model_name}")

    # Create environment
    env = QuadcopterEnv(render_mode=render_mode)
    print("✅ Environment created")

    # SAC configuration
    sac_config = {
        "learning_rate": 3e-4,
        "buffer_size": 500_000,
        "learning_starts": 1000,
        "batch_size": 64,
        "tau": 0.005,
        "gamma": 0.99,
        "train_freq": 1,
        "gradient_steps": 1,
        "ent_coef": "auto",
    }

    # Create model
    model = create_sac_model(env, sac_config)
    print("✅ SAC model created")

    # Create callback for tracking progress
    callback = TrainingCallback(
        total_timesteps=total_timesteps,
        save_freq=save_freq,
        model_name=model_name,
    )

    try:
        # Train the model
        print("🎯 Starting training...")
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True,
        )
        print("🎉 Training completed!")

    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")

    finally:
        # Save final model
        model_path = f"models/{model_name}.zip"
        model.save(model_path)
        print(f"💾 Model saved: {model_path}")

        # Plot results if we have episode data
        if hasattr(callback, "episode_rewards") and callback.episode_rewards:
            plot_training_results(callback.episode_rewards, f"{model_name}_training")

        env.close()

    return model


def main():
    """Main entry point for SAC training."""
    parser = argparse.ArgumentParser(
        description="Train SAC agent for quadcopter control"
    )

    parser.add_argument(
        "--timesteps",
        "-t",
        type=int,
        default=500_000,
        help="Total training timesteps (default: 500,000)",
    )

    parser.add_argument(
        "--model-name",
        "-m",
        type=str,
        default="sac_quadcopter_model",
        help="Name for the saved model (default: sac_quadcopter_model)",
    )

    parser.add_argument(
        "--render",
        "-r",
        choices=["none", "human", "rgb_array"],
        default="none",
        help="Rendering mode (default: none)",
    )

    parser.add_argument(
        "--save-freq",
        "-s",
        type=int,
        default=50_000,
        help="Frequency to save intermediate models (default: 50,000)",
    )

    args = parser.parse_args()

    # Ensure output directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Convert render argument
    render_mode = None if args.render == "none" else args.render

    # Start training
    train_sac(
        total_timesteps=args.timesteps,
        model_name=args.model_name,
        render_mode=render_mode,
        save_freq=args.save_freq,
    )


if __name__ == "__main__":
    main()
