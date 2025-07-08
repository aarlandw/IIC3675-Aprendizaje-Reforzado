#!/usr/bin/env python3
"""
Quick test script for the refactored drone environment
"""

from env_SAC import DroneEnvironment
import numpy as np


def test_environment():
    """Test the refactored environment to make sure it works"""
    print("🧪 Testing refactored DroneEnvironment...")

    # Create environment
    env = DroneEnvironment(render_mode=None, render_every_frame=False)

    print(f"✅ Environment created successfully")
    print(f"   Action space: {env.action_space}")
    print(f"   Observation space: {env.observation_space}")

    # Test reset
    obs, info = env.reset()
    print(f"✅ Reset successful - Observation shape: {obs.shape}")

    # Test a few random steps
    print("🎯 Testing random actions...")
    total_reward = 0

    for step in range(10):
        action = env.action_space.sample()  # Random action
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        print(f"   Step {step+1}: reward={reward:.3f}, done={done}")

        if done:
            print("   Episode ended, resetting...")
            obs, info = env.reset()
            total_reward = 0

    print(f"✅ Environment test completed successfully!")
    env.close()


if __name__ == "__main__":
    test_environment()
