#!/usr/bin/env python3
"""
Model Deployment Script - Test Trained SAC Model

This script loads a pre-trained SAC model and runs it in the environment
to see how it performs without any training.

Usage:
    python deploy_model.py [model_file] [--episodes N] [--render]
"""

import json
import argparse
import numpy as np
from stable_baselines3 import SAC
from env import QuadcopterEnv
import matplotlib.pyplot as plt


def deploy_model(model_path, n_episodes=10, render=False, show_stats=True):
    """
    Load and test a trained SAC model

    Args:
        model_path: Path to the saved model (.zip file)
        n_episodes: Number of episodes to run
        render: Whether to show visual rendering
        show_stats: Whether to print statistics
    """

    print(f"🚀 Deploying model: {model_path}")

    # Load the trained model
    try:
        model = SAC.load(model_path)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

    # Create environment
    env = QuadcopterEnv(render_mode="human" if render else None)
    print(f"🎮 Environment created (render: {render})")

    # Test the model
    episode_rewards = []
    episode_lengths = []

    print(f"\n🧪 Testing model for {n_episodes} episodes...")

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0

        while not done:
            # Use the trained model to predict actions (deterministic=True for best performance)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        if show_stats:
            print(
                f"   Episode {episode + 1:2d}: Reward = {episode_reward:8.2f}, Length = {episode_length:3d}"
            )

    env.close()

    # Calculate statistics
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_length = np.mean(episode_lengths)

    if show_stats:
        print(f"\n📊 Performance Statistics:")
        print(f"   Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"   Best Episode:   {max(episode_rewards):.2f}")
        print(f"   Worst Episode:  {min(episode_rewards):.2f}")
        print(f"   Average Length: {avg_length:.1f} steps")
        print(
            f"   Success Rate:   {sum(1 for r in episode_rewards if r > 0) / len(episode_rewards):.2%}"
        )

    return {
        "rewards": episode_rewards,
        "lengths": episode_lengths,
        "avg_reward": avg_reward,
        "std_reward": std_reward,
        "avg_length": avg_length,
    }


def plot_deployment_results(results, model_name="Model"):
    """Plot the deployment test results"""
    if not results:
        return

    rewards = results["rewards"]

    plt.figure(figsize=(12, 4))

    # Plot episode rewards
    plt.subplot(1, 2, 1)
    episodes = range(1, len(rewards) + 1)
    plt.plot(episodes, rewards, "bo-", linewidth=2, markersize=6)
    plt.axhline(
        y=results["avg_reward"],
        color="r",
        linestyle="--",
        label=f'Average: {results["avg_reward"]:.2f}',
    )
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title(f"{model_name} - Episode Rewards")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot reward histogram
    plt.subplot(1, 2, 2)
    plt.hist(
        rewards, bins=min(10, len(rewards)), alpha=0.7, color="green", edgecolor="black"
    )
    plt.axvline(
        x=results["avg_reward"],
        color="r",
        linestyle="--",
        label=f'Average: {results["avg_reward"]:.2f}',
    )
    plt.xlabel("Episode Reward")
    plt.ylabel("Frequency")
    plt.title(f"{model_name} - Reward Distribution")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(
        f'{model_name.lower().replace(" ", "_")}_deployment_results.png',
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def compare_models(model_paths, n_episodes=10):
    """Compare multiple trained models"""
    results = {}

    for model_path in model_paths:
        model_name = model_path.replace(".zip", "").replace("_", " ").title()
        print(f"\n{'='*50}")
        print(f"Testing: {model_name}")
        print("=" * 50)

        result = deploy_model(model_path, n_episodes, render=False, show_stats=True)
        if result:
            results[model_name] = result

    # Plot comparison
    if len(results) > 1:
        plt.figure(figsize=(12, 6))

        model_names = list(results.keys())
        avg_rewards = [results[name]["avg_reward"] for name in model_names]
        std_rewards = [results[name]["std_reward"] for name in model_names]

        plt.bar(model_names, avg_rewards, yerr=std_rewards, capsize=5, alpha=0.7)
        plt.xlabel("Model")
        plt.ylabel("Average Reward")
        plt.title("Model Performance Comparison")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("model_comparison.png", dpi=300, bbox_inches="tight")
        plt.show()


def visual_inspection(model_path, n_episodes=3):
    """
    Special visual inspection mode with detailed info display
    """
    print(f"👁️  VISUAL INSPECTION MODE")
    print(f"📱 Model: {model_path}")
    print(f"🎮 Episodes: {n_episodes}")
    print(f"📋 Press SPACE to pause, ESC to quit during episodes")
    print("=" * 60)

    # Load model
    try:
        model = SAC.load(model_path)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Create environment with human rendering
    env = QuadcopterEnv(render_mode="human")

    for episode in range(n_episodes):
        print(f"\n🎬 Starting Episode {episode + 1}/{n_episodes}")
        print("   Watch the drone's behavior closely...")

        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        action_history = []

        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            action_history.append(action.copy())

            # Step environment
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1

            # Optional: Add small delay for better observation
            import time

            time.sleep(0.05)  # 50ms delay for smoother viewing

        print(f"   ✅ Episode {episode + 1} completed!")
        print(f"   📊 Reward: {episode_reward:.2f}, Length: {episode_length} steps")
        print(
            f"   🎯 Action range: [{min(min(a) for a in action_history):.2f}, {max(max(a) for a in action_history):.2f}]"
        )

        if episode < n_episodes - 1:
            input("   ⏸️  Press ENTER to continue to next episode...")

    env.close()
    print("\n🎉 Visual inspection complete!")


def main():
    parser = argparse.ArgumentParser(description="Deploy and test a trained SAC model")
    parser.add_argument(
        "model",
        nargs="?",
        default="best_model_optuna.zip",
        help="Path to the model file (default: best_model_optuna.zip)",
    )
    parser.add_argument(
        "--episodes",
        "-e",
        type=int,
        default=10,
        help="Number of episodes to test (default: 10)",
    )
    parser.add_argument(
        "--render",
        "-r",
        action="store_true",
        help="Show visual rendering during testing",
    )
    parser.add_argument(
        "--plot", "-p", action="store_true", help="Generate performance plots"
    )
    parser.add_argument(
        "--compare",
        "-c",
        nargs="+",
        help="Compare multiple models (provide multiple model paths)",
    )
    parser.add_argument(
        "--inspect",
        "-i",
        action="store_true",
        help="Special visual inspection mode with detailed monitoring",
    )

    args = parser.parse_args()

    if args.inspect:
        # Visual inspection mode
        visual_inspection(args.model, min(args.episodes, 5))
    elif args.compare:
        # Compare multiple models
        compare_models(args.compare, args.episodes)
    else:
        # Test single model
        results = deploy_model(args.model, args.episodes, args.render)

        if results and args.plot:
            model_name = args.model.replace(".zip", "").replace("_", " ").title()
            plot_deployment_results(results, model_name)


if __name__ == "__main__":
    main()
