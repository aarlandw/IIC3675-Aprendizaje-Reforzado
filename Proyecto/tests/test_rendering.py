#!/usr/bin/env python3
"""
Test script to verify environment rendering works correctly
"""

import os
import sys
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env import QuadcopterEnv

def test_environment_rendering():
    """Test basic environment rendering"""
    print("Testing environment rendering...")
    
    # Create environment with human rendering
    env = QuadcopterEnv(render_mode="human")
    
    try:
        # Reset environment
        obs, info = env.reset()
        print(f"Initial observation: {obs}")
        
        # Run a few steps with rendering
        for step in range(50):
            # Take random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Render the environment
            env.render()
            
            print(f"Step {step}: Action={action}, Reward={reward:.2f}, Terminated={terminated}")
            
            if terminated or truncated:
                print("Episode finished, resetting...")
                obs, info = env.reset()
                
            # Add small delay to see the rendering
            time.sleep(0.1)
            
        print("Rendering test completed successfully!")
        
    except Exception as e:
        print(f"Error during rendering test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()

if __name__ == "__main__":
    test_environment_rendering()
