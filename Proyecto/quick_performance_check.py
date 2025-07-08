#!/usr/bin/env python3
"""
Quick performance check without visual rendering
"""

import os
import numpy as np
from stable_baselines3 import SAC
from env_SAC import DroneEnvironment
from her_sparse_wrapper import SparseRewardHERWrapper

def quick_check(model_path="sparse_her_balloon.zip"):
    """Quick performance check"""
    
    print(f"⚡ Quick Performance Check: {model_path}")
    print("=" * 40)
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    # Create environment FIRST (needed for HER models)
    base_env = DroneEnvironment(render_mode=None)
    her_env = SparseRewardHERWrapper(base_env, goal_threshold=25.0)
    
    # Load model WITH environment
    try:
        model = SAC.load(model_path, env=her_env)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        her_env.env._close()
        return
    
    # Test for 10 episodes
    balloons_collected = []
    episode_lengths = []
    
    for episode in range(10):
        obs, _ = her_env.reset()
        episode_balloons = 0
        steps = 0
        
        for _ in range(2000):  # Max 2000 steps
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = her_env.step(action)
            
            episode_balloons = info.get('balloons_collected', 0)
            steps += 1
            
            if terminated or truncated:
                break
        
        balloons_collected.append(episode_balloons)
        episode_lengths.append(steps)
        
        print(f"Episode {episode + 1}: {episode_balloons} balloons in {steps} steps")
    
    # Results
    avg_balloons = np.mean(balloons_collected)
    avg_length = np.mean(episode_lengths)
    
    print(f"\n📊 Results:")
    print(f"   Average balloons: {avg_balloons:.1f}")
    print(f"   Average steps: {avg_length:.1f}")
    print(f"   Best episode: {max(balloons_collected)} balloons")
    print(f"   Success rate: {np.mean(np.array(balloons_collected) > 0):.2%}")
    
    # Performance rating
    if avg_balloons >= 3:
        print("🌟🌟🌟 EXCELLENT performance!")
    elif avg_balloons >= 2:
        print("🌟🌟 GOOD performance!")
    elif avg_balloons >= 1:
        print("🌟 FAIR performance")
    else:
        print("❌ Needs more training")
    
    her_env.env._close()

if __name__ == "__main__":
    quick_check()