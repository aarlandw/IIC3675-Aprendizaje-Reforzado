"""
Plotting utilities for training visualization.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path


def plot_training_results(episode_rewards, title="Training Results", save_path=None):
    """
    Plot training results with episode rewards and moving averages.

    Args:
        episode_rewards: List of episode rewards
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    if not episode_rewards:
        print("No episode rewards to plot")
        return

    plt.figure(figsize=(12, 8))

    # Plot raw episode rewards
    plt.subplot(2, 2, 1)
    episodes = range(1, len(episode_rewards) + 1)
    plt.plot(episodes, episode_rewards, alpha=0.6, color="blue", linewidth=1)

    # Add moving averages
    if len(episode_rewards) > 10:
        window_sizes = [10, 50, 100]
        colors = ["red", "green", "orange"]

        for window_size, color in zip(window_sizes, colors):
            if len(episode_rewards) >= window_size:
                moving_avg = (
                    pd.Series(episode_rewards).rolling(window=window_size).mean()
                )
                plt.plot(
                    episodes,
                    moving_avg,
                    color=color,
                    linewidth=2,
                    label=f"MA-{window_size}",
                )

        plt.legend()

    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title(f"{title} - Episode Rewards")
    plt.grid(True, alpha=0.3)

    # Plot reward distribution
    plt.subplot(2, 2, 2)
    plt.hist(episode_rewards, bins=30, alpha=0.7, color="purple", edgecolor="black")
    plt.axvline(
        np.mean(episode_rewards),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(episode_rewards):.2f}",
    )
    plt.xlabel("Episode Reward")
    plt.ylabel("Frequency")
    plt.title("Reward Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot learning curve (cumulative average)
    plt.subplot(2, 2, 3)
    cumulative_avg = np.cumsum(episode_rewards) / np.arange(1, len(episode_rewards) + 1)
    plt.plot(episodes, cumulative_avg, color="green", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Average Reward")
    plt.title("Learning Curve")
    plt.grid(True, alpha=0.3)

    # Plot recent performance (last 20% of episodes)
    plt.subplot(2, 2, 4)
    recent_start = max(1, int(0.8 * len(episode_rewards)))
    recent_episodes = episodes[recent_start - 1 :]
    recent_rewards = episode_rewards[recent_start - 1 :]

    plt.plot(recent_episodes, recent_rewards, "bo-", alpha=0.7)
    plt.axhline(
        np.mean(recent_rewards),
        color="red",
        linestyle="--",
        label=f"Recent Mean: {np.mean(recent_rewards):.2f}",
    )
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title("Recent Performance (Last 20%)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot if path provided
    if save_path:
        plt.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")
        print(f"📊 Plot saved: {save_path}.png")

    plt.show()


def plot_her_results(
    episode_rewards, episode_successes, title="HER Training Results", save_path=None
):
    """
    Plot HER training results with success rate tracking.

    Args:
        episode_rewards: List of episode rewards
        episode_successes: List of episode success flags
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    if not episode_rewards or not episode_successes:
        print("No episode data to plot")
        return

    plt.figure(figsize=(15, 10))
    episodes = range(1, len(episode_rewards) + 1)

    # Plot episode rewards
    plt.subplot(2, 3, 1)
    plt.plot(episodes, episode_rewards, alpha=0.6, color="blue", linewidth=1)

    if len(episode_rewards) > 50:
        moving_avg = pd.Series(episode_rewards).rolling(window=50).mean()
        plt.plot(episodes, moving_avg, color="red", linewidth=2, label="MA-50")
        plt.legend()

    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title(f"{title} - Episode Rewards")
    plt.grid(True, alpha=0.3)

    # Plot success rate
    plt.subplot(2, 3, 2)
    window = 50
    success_rates = []
    for i in range(len(episode_successes)):
        start_idx = max(0, i - window + 1)
        rate = sum(episode_successes[start_idx : i + 1]) / (i - start_idx + 1)
        success_rates.append(rate)

    plt.plot(episodes, success_rates, color="green", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    plt.title(f"Success Rate (Rolling {window} episodes)")
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)

    # Plot reward distribution
    plt.subplot(2, 3, 3)
    plt.hist(episode_rewards, bins=30, alpha=0.7, color="purple", edgecolor="black")
    plt.axvline(
        np.mean(episode_rewards),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(episode_rewards):.2f}",
    )
    plt.xlabel("Episode Reward")
    plt.ylabel("Frequency")
    plt.title("Reward Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot cumulative successes
    plt.subplot(2, 3, 4)
    cumulative_successes = np.cumsum(episode_successes)
    plt.plot(episodes, cumulative_successes, color="orange", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Successes")
    plt.title("Total Successful Episodes")
    plt.grid(True, alpha=0.3)

    # Plot success vs failure comparison
    plt.subplot(2, 3, 5)
    total_successes = sum(episode_successes)
    total_failures = len(episode_successes) - total_successes

    plt.pie(
        [total_successes, total_failures],
        labels=["Successes", "Failures"],
        colors=["green", "red"],
        autopct="%1.1f%%",
        startangle=90,
    )
    plt.title("Success/Failure Ratio")

    # Plot recent performance
    plt.subplot(2, 3, 6)
    recent_start = max(1, int(0.8 * len(episode_rewards)))
    recent_episodes = episodes[recent_start - 1 :]
    recent_rewards = episode_rewards[recent_start - 1 :]
    recent_successes = episode_successes[recent_start - 1 :]

    colors = ["green" if success else "red" for success in recent_successes]
    plt.scatter(recent_episodes, recent_rewards, c=colors, alpha=0.7)
    plt.axhline(
        np.mean(recent_rewards),
        color="blue",
        linestyle="--",
        label=f"Recent Mean: {np.mean(recent_rewards):.2f}",
    )
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title("Recent Performance (Green=Success, Red=Failure)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot if path provided
    if save_path:
        plt.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")
        print(f"📊 HER plot saved: {save_path}.png")

    plt.show()

    # Print statistics
    total_episodes = len(episode_rewards)
    final_success_rate = sum(episode_successes[-100:]) / min(
        100, len(episode_successes)
    )

    print(f"\n🎯 HER Training Statistics:")
    print(f"   Total Episodes: {total_episodes}")
    print(f"   Successful Episodes: {total_successes}")
    print(f"   Overall Success Rate: {total_successes/total_episodes:.2%}")
    print(f"   Final Success Rate (last 100): {final_success_rate:.2%}")
    print(f"   Best Episode Reward: {max(episode_rewards):.2f}")
    print(f"   Average Episode Reward: {np.mean(episode_rewards):.2f}")


def plot_comparison(results_dict, title="Model Comparison", save_path=None):
    """
    Compare multiple models' performance.

    Args:
        results_dict: Dict of {model_name: episode_rewards_list}
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    if not results_dict:
        print("No results to compare")
        return

    plt.figure(figsize=(15, 8))

    # Plot 1: Average performance comparison
    plt.subplot(2, 2, 1)
    model_names = list(results_dict.keys())
    avg_rewards = [np.mean(rewards) for rewards in results_dict.values()]
    std_rewards = [np.std(rewards) for rewards in results_dict.values()]

    bars = plt.bar(model_names, avg_rewards, yerr=std_rewards, capsize=5, alpha=0.7)
    plt.xlabel("Model")
    plt.ylabel("Average Reward")
    plt.title("Average Performance Comparison")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, avg in zip(bars, avg_rewards):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{avg:.2f}",
            ha="center",
            va="bottom",
        )

    # Plot 2: Learning curves
    plt.subplot(2, 2, 2)
    for model_name, rewards in results_dict.items():
        episodes = range(1, len(rewards) + 1)
        cumulative_avg = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
        plt.plot(episodes, cumulative_avg, linewidth=2, label=model_name)

    plt.xlabel("Episode")
    plt.ylabel("Cumulative Average Reward")
    plt.title("Learning Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3: Reward distributions
    plt.subplot(2, 2, 3)
    data_for_box = [rewards for rewards in results_dict.values()]
    plt.boxplot(data_for_box, labels=model_names)
    plt.xlabel("Model")
    plt.ylabel("Episode Reward")
    plt.title("Reward Distributions")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    # Plot 4: Performance stability (coefficient of variation)
    plt.subplot(2, 2, 4)
    cv_values = [
        np.std(rewards) / np.mean(rewards) for rewards in results_dict.values()
    ]
    bars = plt.bar(model_names, cv_values, alpha=0.7, color="orange")
    plt.xlabel("Model")
    plt.ylabel("Coefficient of Variation")
    plt.title("Performance Stability (Lower = More Stable)")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    # Add value labels
    for bar, cv in zip(bars, cv_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{cv:.3f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")
        print(f"📊 Comparison plot saved: {save_path}.png")

    plt.show()
