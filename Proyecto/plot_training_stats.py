#!/usr/bin/env python3
"""
Plot training statistics from the HER training
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_training_stats(stats_file="sparse_her_balloon_training_stats.pkl"):
    """Plot comprehensive training statistics"""
    
    if not os.path.exists(stats_file):
        print(f"❌ Stats file not found: {stats_file}")
        print("📁 Available stats files:")
        for file in os.listdir("."):
            if file.endswith("_training_stats.pkl"):
                print(f"   - {file}")
        return
    
    # Load stats
    try:
        with open(stats_file, 'rb') as f:
            stats = pickle.load(f)
        print(f"📊 Loaded stats from {stats_file}")
    except Exception as e:
        print(f"❌ Error loading stats: {e}")
        return
    
    # Extract data
    episode_rewards = stats.get('episode_rewards', [])
    balloons_collected = stats.get('balloons_collected', [])
    success_rates = stats.get('success_rates', [])
    total_timesteps = stats.get('total_timesteps', 0)
    best_performance = stats.get('best_performance', 0)
    
    if len(episode_rewards) == 0:
        print("❌ No training data found")
        return
    
    print(f"📈 Episodes: {len(episode_rewards)}")
    print(f"⏱️  Total timesteps: {total_timesteps:,}")
    print(f"🏆 Best performance: {best_performance:.2f} balloons")
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('🎈 HER Training Progress Analysis', fontsize=16, fontweight='bold')
    
    episodes = range(1, len(episode_rewards) + 1)
    
    # 1. Episode Rewards
    ax1.plot(episodes, episode_rewards, alpha=0.6, color='blue', linewidth=0.5)
    if len(episode_rewards) > 50:
        # Moving average
        window = min(50, len(episode_rewards) // 10)
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(episodes[window-1:], moving_avg, color='red', linewidth=2, label=f'Moving Avg ({window})')
        ax1.legend()
    ax1.set_title('🏆 Episode Rewards (HER Sparse)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.grid(True, alpha=0.3)
    
    # 2. Balloons Collected
    ax2.plot(episodes, balloons_collected, alpha=0.7, color='green', marker='o', markersize=2)
    if len(balloons_collected) > 50:
        window = min(50, len(balloons_collected) // 10)
        balloon_avg = np.convolve(balloons_collected, np.ones(window)/window, mode='valid')
        ax2.plot(episodes[window-1:], balloon_avg, color='darkgreen', linewidth=3, label=f'Moving Avg ({window})')
        ax2.legend()
    ax2.set_title('🎈 Balloons Collected per Episode')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Balloons')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    
    # 3. Success Rate
    if len(success_rates) > 50:
        window = min(50, len(success_rates) // 10)
        success_avg = np.convolve(success_rates, np.ones(window)/window, mode='valid')
        ax3.plot(episodes[window-1:], [x*100 for x in success_avg], color='purple', linewidth=2)
    ax3.scatter(episodes, [x*100 for x in success_rates], alpha=0.4, color='purple', s=10)
    ax3.set_title('🎯 Success Rate (%)')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Success Rate (%)')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 100)
    
    # 4. Performance Distribution
    balloon_counts = {}
    for count in balloons_collected:
        balloon_counts[count] = balloon_counts.get(count, 0) + 1
    
    counts = list(balloon_counts.keys())
    frequencies = list(balloon_counts.values())
    
    ax4.bar(counts, frequencies, alpha=0.7, color='orange')
    ax4.set_title('🎈 Balloon Collection Distribution')
    ax4.set_xlabel('Balloons per Episode')
    ax4.set_ylabel('Frequency')
    ax4.grid(True, alpha=0.3)
    
    # Add performance stats
    recent_episodes = 100
    if len(balloons_collected) >= recent_episodes:
        recent_avg = np.mean(balloons_collected[-recent_episodes:])
        recent_success = np.mean(success_rates[-recent_episodes:])
        ax4.text(0.02, 0.98, f'Recent {recent_episodes} episodes:\nAvg Balloons: {recent_avg:.2f}\nSuccess Rate: {recent_success:.1%}', 
                transform=ax4.transAxes, verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('her_training_progress.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print analysis
    print(f"\n📊 TRAINING ANALYSIS")
    print("=" * 40)
    
    if len(balloons_collected) >= 100:
        early_performance = np.mean(balloons_collected[:100])
        late_performance = np.mean(balloons_collected[-100:])
        improvement = late_performance - early_performance
        
        print(f"🌱 Early performance (first 100): {early_performance:.2f} balloons")
        print(f"🚀 Recent performance (last 100): {late_performance:.2f} balloons")
        print(f"📈 Improvement: {improvement:+.2f} balloons")
        
        if improvement > 0.5:
            print("✅ Good learning progress!")
        elif improvement > 0.1:
            print("⚠️ Slow but steady progress")
        else:
            print("❌ Little to no improvement - needs reward tuning")
    
    max_balloons = max(balloons_collected) if balloons_collected else 0
    zero_balloon_episodes = balloons_collected.count(0)
    zero_percentage = (zero_balloon_episodes / len(balloons_collected)) * 100
    
    print(f"🏆 Best episode: {max_balloons} balloons")
    print(f"❌ Episodes with 0 balloons: {zero_balloon_episodes} ({zero_percentage:.1f}%)")
    
    if zero_percentage > 80:
        print("⚠️ HIGH FAILURE RATE - Reward function needs adjustment!")

if __name__ == "__main__":
    plot_training_stats()