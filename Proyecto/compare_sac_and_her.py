#!/usr/bin/env python3
"""
Compare SAC-only vs HER+SAC performance with correct reward expectations
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

def load_training_data(sac_file, her_file):
    """Load training data from both approaches"""
    
    # Load SAC data
    try:
        with open(sac_file, 'rb') as f:
            sac_data = pickle.load(f)
        print(f"✅ Loaded SAC data from {sac_file}")
    except Exception as e:
        print(f"❌ Error loading SAC data: {e}")
        return None, None
    
    # Load HER data  
    try:
        with open(her_file, 'rb') as f:
            her_data = pickle.load(f)
        print(f"✅ Loaded HER data from {her_file}")
    except Exception as e:
        print(f"❌ Error loading HER data: {e}")
        return None, None
    
    return sac_data, her_data

def analyze_reward_structure(rewards):
    """Analyze the actual reward structure in the data"""
    rewards = np.array(rewards)
    unique_rewards = np.unique(rewards)
    
    print(f"📊 Unique reward values found: {unique_rewards}")
    
    # Categorize rewards
    positive_rewards = rewards[rewards > 0]
    negative_rewards = rewards[rewards < 0]
    zero_rewards = rewards[rewards == 0]
    
    analysis = {
        'unique_values': unique_rewards,
        'positive_count': len(positive_rewards),
        'negative_count': len(negative_rewards),
        'zero_count': len(zero_rewards),
        'max_positive': np.max(positive_rewards) if len(positive_rewards) > 0 else 0,
        'min_negative': np.min(negative_rewards) if len(negative_rewards) > 0 else 0,
        'total_episodes': len(rewards)
    }
    
    return analysis

def calculate_learning_metrics_corrected(rewards, reward_analysis):
    """Calculate learning metrics with correct reward understanding"""
    rewards = np.array(rewards)
    
    metrics = {}
    
    # Basic stats
    metrics['total_episodes'] = len(rewards)
    metrics['mean_reward'] = np.mean(rewards)
    metrics['std_reward'] = np.std(rewards)
    metrics['best_reward'] = np.max(rewards)
    metrics['worst_reward'] = np.min(rewards)
    
    # Success metrics (positive rewards = balloon collection)
    success_episodes = rewards > 0
    failure_episodes = rewards < 0  # Death/crash episodes
    neutral_episodes = rewards == 0  # Exploration episodes
    
    metrics['success_rate'] = np.mean(success_episodes)
    metrics['failure_rate'] = np.mean(failure_episodes)
    metrics['exploration_rate'] = np.mean(neutral_episodes)
    
    metrics['total_successes'] = np.sum(success_episodes)
    metrics['total_failures'] = np.sum(failure_episodes)
    metrics['total_neutral'] = np.sum(neutral_episodes)
    
    # Sample efficiency metrics
    first_success = np.where(success_episodes)[0]
    metrics['episodes_to_first_success'] = first_success[0] + 1 if len(first_success) > 0 else len(rewards)
    
    # Learning curve analysis
    window = min(100, len(rewards) // 4)
    if len(rewards) >= window * 2:
        early_rewards = rewards[:window]
        late_rewards = rewards[-window:]
        
        metrics['early_success_rate'] = np.mean(early_rewards > 0)
        metrics['late_success_rate'] = np.mean(late_rewards > 0)
        metrics['success_improvement'] = metrics['late_success_rate'] - metrics['early_success_rate']
        
        metrics['early_failure_rate'] = np.mean(early_rewards < 0)
        metrics['late_failure_rate'] = np.mean(late_rewards < 0)
        metrics['failure_reduction'] = metrics['early_failure_rate'] - metrics['late_failure_rate']
    
    # Safety metric (avoiding deaths)
    metrics['safety_score'] = 1 - metrics['failure_rate']
    
    return metrics

def plot_comparison_corrected(sac_data, her_data, save_path="sac_vs_her_comparison_corrected.png"):
    """Create comparison plots with correct reward understanding"""
    
    # Extract rewards
    sac_rewards = sac_data.get('episode_rewards', [])
    her_rewards = her_data.get('episode_rewards', [])
    
    if len(sac_rewards) == 0 or len(her_rewards) == 0:
        print("❌ No reward data found in files")
        return
    
    # Analyze reward structures
    print("🔍 Analyzing SAC reward structure:")
    sac_analysis = analyze_reward_structure(sac_rewards)
    
    print("🔍 Analyzing HER reward structure:")
    her_analysis = analyze_reward_structure(her_rewards)
    
    # Calculate metrics
    sac_metrics = calculate_learning_metrics_corrected(sac_rewards, sac_analysis)
    her_metrics = calculate_learning_metrics_corrected(her_rewards, her_analysis)
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('🚁 SAC vs HER+SAC Performance Comparison (Corrected)', fontsize=16, fontweight='bold')
    
    # 1. Normalized Rewards Over Time (Fair Comparison)
    ax1 = axes[0, 0]
    episodes_sac = range(len(sac_rewards))
    episodes_her = range(len(her_rewards))
    
    # Normalize rewards to [0, 1] scale for fair comparison
    def normalize_rewards(rewards):
        rewards_array = np.array(rewards)
        min_r, max_r = np.min(rewards_array), np.max(rewards_array)
        if max_r == min_r:
            return np.zeros_like(rewards_array)
        return (rewards_array - min_r) / (max_r - min_r)
    
    sac_normalized = normalize_rewards(sac_rewards)
    her_normalized = normalize_rewards(her_rewards)
    
    ax1.plot(episodes_sac, sac_normalized, alpha=0.6, label='SAC-only (normalized)', color='blue')
    ax1.plot(episodes_her, her_normalized, alpha=0.6, label='HER+SAC (normalized)', color='red')
    
    # Add moving averages for normalized data
    window = 100
    if len(sac_normalized) > window:
        sac_ma_norm = pd.Series(sac_normalized).rolling(window).mean()
        ax1.plot(episodes_sac, sac_ma_norm, color='darkblue', linewidth=2, label=f'SAC MA({window})')
    
    if len(her_normalized) > window:
        her_ma_norm = pd.Series(her_normalized).rolling(window).mean()
        ax1.plot(episodes_her, her_ma_norm, color='darkred', linewidth=2, label=f'HER MA({window})')
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Normalized Reward [0,1]')
    ax1.set_title('Normalized Rewards (Fair Scale Comparison)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Add text showing original scales
    ax1.text(0.02, 0.98, f'SAC range: {sac_analysis["min_negative"]:.0f} to {sac_analysis["max_positive"]:.0f}', 
             transform=ax1.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax1.text(0.02, 0.88, f'HER range: {her_analysis["min_negative"]:.1f} to {her_analysis["max_positive"]:.1f}', 
             transform=ax1.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    # 2. Raw Rewards Over Time (Different Scales)
    ax2 = axes[0, 1]
    
    # Create dual y-axis plot to show both scales
    ax2_twin = ax2.twinx()
    
    # Plot SAC on left axis
    line1 = ax2.plot(episodes_sac, sac_rewards, alpha=0.6, label='SAC-only (left axis)', color='blue')
    line2 = []
    if len(sac_rewards) > window:
        sac_ma = pd.Series(sac_rewards).rolling(window).mean()
        line2 = ax2.plot(episodes_sac, sac_ma, color='darkblue', linewidth=2, label=f'SAC MA({window})')
    
    # Plot HER on right axis
    line3 = ax2_twin.plot(episodes_her, her_rewards, alpha=0.6, label='HER+SAC (right axis)', color='red')
    line4 = []
    if len(her_rewards) > window:
        her_ma = pd.Series(her_rewards).rolling(window).mean()
        line4 = ax2_twin.plot(episodes_her, her_ma, color='darkred', linewidth=2, label=f'HER MA({window})')
    
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('SAC Raw Reward', color='blue')
    ax2_twin.set_ylabel('HER Raw Reward', color='red')
    ax2.set_title('Raw Rewards (Dual Scale)')
    
    # Combine legends
    lines = line1 + line3 + line2 + line4
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper left')
    
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2_twin.tick_params(axis='y', labelcolor='red')
    
    # 3. Success/Failure/Exploration Rates Over Time
    ax3 = axes[0, 2]
    window = 50
    
    # SAC success rate
    sac_success_rate = pd.Series(np.array(sac_rewards) > 0).rolling(window).mean()
    sac_failure_rate = pd.Series(np.array(sac_rewards) < 0).rolling(window).mean()
    
    # HER success rate  
    her_success_rate = pd.Series(np.array(her_rewards) > 0).rolling(window).mean()
    her_failure_rate = pd.Series(np.array(her_rewards) < 0).rolling(window).mean()
    
    ax3.plot(episodes_sac, sac_success_rate, color='blue', linewidth=2, label='SAC Success Rate')
    ax3.plot(episodes_sac, sac_failure_rate, color='blue', linewidth=2, linestyle='--', label='SAC Failure Rate')
    ax3.plot(episodes_her, her_success_rate, color='red', linewidth=2, label='HER Success Rate')
    ax3.plot(episodes_her, her_failure_rate, color='red', linewidth=2, linestyle='--', label='HER Failure Rate')
    
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Rate')
    ax3.set_title(f'Success/Failure Rates (Rolling {window} episodes)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    # 4. Cumulative Success Comparison
    ax4 = axes[1, 0]
    sac_cumulative_success = np.cumsum(np.array(sac_rewards) > 0)
    her_cumulative_success = np.cumsum(np.array(her_rewards) > 0)
    
    ax4.plot(episodes_sac, sac_cumulative_success, color='blue', linewidth=2, label='SAC Cumulative Successes')
    ax4.plot(episodes_her, her_cumulative_success, color='red', linewidth=2, label='HER Cumulative Successes')
    
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Cumulative Balloon Collections')
    ax4.set_title('Sample Efficiency Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Reward Distribution Comparison
    ax5 = axes[1, 1]
    
    # Create bins that capture the key reward values
    all_rewards = np.concatenate([sac_rewards, her_rewards])
    bins = np.linspace(min(all_rewards), max(all_rewards), 50)
    
    ax5.hist(sac_rewards, bins=bins, alpha=0.6, label='SAC-only', color='blue', density=True)
    ax5.hist(her_rewards, bins=bins, alpha=0.6, label='HER+SAC', color='red', density=True)
    
    # Mark key values
    if her_analysis['max_positive'] > 0:
        ax5.axvline(x=her_analysis['max_positive'], color='green', linestyle='--', alpha=0.8, label=f'HER Success ({her_analysis["max_positive"]})')
    if her_analysis['min_negative'] < 0:
        ax5.axvline(x=her_analysis['min_negative'], color='red', linestyle='--', alpha=0.8, label=f'HER Failure ({her_analysis["min_negative"]})')
    
    ax5.set_xlabel('Reward Value')
    ax5.set_ylabel('Density')
    ax5.set_title('Reward Distribution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Learning Progress (Success Rate Improvement)
    ax6 = axes[1, 2]
    
    # Plot learning curves
    chunk_size = max(1, len(sac_rewards) // 20)  # 20 data points
    sac_chunks = [sac_rewards[i:i+chunk_size] for i in range(0, len(sac_rewards), chunk_size)]
    sac_success_progression = [np.mean(np.array(chunk) > 0) for chunk in sac_chunks if len(chunk) > 0]
    
    chunk_size = max(1, len(her_rewards) // 20)
    her_chunks = [her_rewards[i:i+chunk_size] for i in range(0, len(her_rewards), chunk_size)]
    her_success_progression = [np.mean(np.array(chunk) > 0) for chunk in her_chunks if len(chunk) > 0]
    
    # Create Performance Metrics Table instead of learning curve
    ax6.axis('off')
    
    # Create comparison table with corrected metrics
    comparison_data = [
        ['Metric', 'SAC-only', 'HER+SAC', 'Winner'],
        ['Total Episodes', f"{sac_metrics['total_episodes']}", f"{her_metrics['total_episodes']}", '-'],
        ['Success Rate', f"{sac_metrics['success_rate']:.3f}", f"{her_metrics['success_rate']:.3f}", 
         'HER' if her_metrics['success_rate'] > sac_metrics['success_rate'] else 'SAC'],
        ['Failure Rate', f"{sac_metrics['failure_rate']:.3f}", f"{her_metrics['failure_rate']:.3f}", 
         'SAC' if her_metrics['failure_rate'] > sac_metrics['failure_rate'] else 'HER'],
        ['Safety Score', f"{sac_metrics['safety_score']:.3f}", f"{her_metrics['safety_score']:.3f}", 
         'HER' if her_metrics['safety_score'] > sac_metrics['safety_score'] else 'SAC'],
        ['Episodes to 1st Success', f"{sac_metrics['episodes_to_first_success']}", f"{her_metrics['episodes_to_first_success']}", 
         'HER' if her_metrics['episodes_to_first_success'] < sac_metrics['episodes_to_first_success'] else 'SAC'],
        ['Total Successes', f"{sac_metrics['total_successes']}", f"{her_metrics['total_successes']}", 
         'HER' if her_metrics['total_successes'] > sac_metrics['total_successes'] else 'SAC'],
    ]
    
    # Create table
    table = ax6.table(cellText=comparison_data, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Color code winners
    for i in range(1, len(comparison_data)):
        if comparison_data[i][3] == 'HER':
            table[(i, 2)].set_facecolor('#ffcccc')  # Light red for HER
        elif comparison_data[i][3] == 'SAC':
            table[(i, 1)].set_facecolor('#ccccff')  # Light blue for SAC
    
    ax6.set_title('Performance Metrics Comparison')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Corrected comparison plot saved as: {save_path}")
    plt.show()
    
    return sac_metrics, her_metrics, sac_analysis, her_analysis

# Usage example
if __name__ == "__main__":
    # Replace with your actual file names
    sac_pkl_file = "sac_results_20250708_164304/intermediate_training_results.pkl"
    her_pkl_file = "sparse_her_balloon_training_stats.pkl"
    
    print("🔍 Loading training data...")
    sac_data, her_data = load_training_data(sac_pkl_file, her_pkl_file)
    
    if sac_data is not None and her_data is not None:
        print("📊 Creating corrected comparison plots...")
        result = plot_comparison_corrected(sac_data, her_data)
        
        if result is not None:
            sac_metrics, her_metrics, sac_analysis, her_analysis = result
            
            print(f"\n📊 CORRECTED PERFORMANCE ANALYSIS")
            print("="*60)
            
            print(f"\n🔵 SAC-only:")
            print(f"   Reward range: {sac_analysis['min_negative']:.1f} to {sac_analysis['max_positive']:.1f}")
            print(f"   Success rate: {sac_metrics['success_rate']:.3f}")
            print(f"   Failure rate: {sac_metrics['failure_rate']:.3f}")
            print(f"   Safety score: {sac_metrics['safety_score']:.3f}")
            
            print(f"\n🔴 HER+SAC:")
            print(f"   Reward range: {her_analysis['min_negative']:.1f} to {her_analysis['max_positive']:.1f}")
            print(f"   Success rate: {her_metrics['success_rate']:.3f}")
            print(f"   Failure rate: {her_metrics['failure_rate']:.3f}")
            print(f"   Safety score: {her_metrics['safety_score']:.3f}")
            
            # Determine winner based on key metrics
            her_wins = 0
            sac_wins = 0
            
            if her_metrics['success_rate'] > sac_metrics['success_rate']:
                her_wins += 1
            else:
                sac_wins += 1
                
            if her_metrics['safety_score'] > sac_metrics['safety_score']:
                her_wins += 1
            else:
                sac_wins += 1
            
            if her_metrics['episodes_to_first_success'] < sac_metrics['episodes_to_first_success']:
                her_wins += 1
            else:
                sac_wins += 1
            
            print(f"\n🏆 OVERALL WINNER:")
            if her_wins > sac_wins:
                print(f"   🔴 HER+SAC wins! ({her_wins} out of 3 key metrics)")
            elif sac_wins > her_wins:
                print(f"   🔵 SAC-only wins! ({sac_wins} out of 3 key metrics)")
            else:
                print(f"   🤝 It's a tie! Both approaches perform similarly")
        else:
            print("❌ Could not generate comparison plots")
    else:
        print("❌ Could not load data files")