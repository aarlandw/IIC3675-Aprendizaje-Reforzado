#!/usr/bin/env python3
"""
Test the existing SAC environment to see if it works as intended
"""

from env_SAC import DroneEnvironment
import numpy as np
import pygame

def test_sac_environment():
    """Test the SAC environment functionality"""
    
    print("🚁 Testing SAC Drone Environment")
    print("=" * 50)
    
    # Create environment
    try:
        env = DroneEnvironment(render_mode="human", render_every_frame=True)
        print("✅ Environment created successfully!")
    except Exception as e:
        print(f"❌ Error creating environment: {e}")
        return
    
    # Check spaces
    print(f"\n📋 Environment Spaces:")
    print(f"   Action Space: {env.action_space}")
    print(f"   Observation Space: {env.observation_space}")
    
    # Reset and get initial observation
    try:
        obs, info = env.reset()
        print(f"\n🔄 Initial State:")
        print(f"   Observation shape: {obs.shape}")
        print(f"   Drone position: ({env.x:.1f}, {env.y:.1f})")
        print(f"   Target position: ({env.xt:.1f}, {env.yt:.1f})")
        print(f"   Bird 1 position: ({env.x_pajaro:.1f}, {env.y_pajaro:.1f})")
        print(f"   Bird 2 position: ({env.x_pajaro2:.1f}, {env.y_pajaro2:.1f})")
    except Exception as e:
        print(f"❌ Error during reset: {e}")
        return
    
    print(f"\n🎮 Testing with random actions...")
    print("   Watch the drone try to avoid the two moving birds!")
    print("   Red balloons are targets to collect")
    print("   Birds are obstacles to avoid")
    print("   Press ESC or close window to stop")
    
    # Test loop
    episode_count = 0
    step_count = 0
    total_reward = 0
    
    try:
        running = True
        while running and episode_count < 5:  # Test 5 episodes
            # Take random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            step_count += 1
            
            # Handle pygame events to avoid freezing
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            # Print progress
            if step_count % 100 == 0:
                print(f"   Step {step_count}: Reward = {reward:.2f}, Total = {total_reward:.2f}")
                print(f"     Targets collected: {env.target_counter}")
                print(f"     Drone at: ({env.x:.1f}, {env.y:.1f})")
            
            # Episode ended
            if terminated or truncated:
                episode_count += 1
                episode_reward = info.get('episode', {}).get('r', total_reward)
                targets_collected = info.get('episode', {}).get('l', env.target_counter)
                
                print(f"\n🏁 Episode {episode_count} Complete:")
                print(f"   Episode Reward: {episode_reward:.2f}")
                print(f"   Targets Collected: {targets_collected}")
                print(f"   Steps: {step_count}")
                
                if episode_count < 5:
                    obs, info = env.reset()
                    step_count = 0
                    total_reward = 0
                    print(f"🔄 Starting episode {episode_count + 1}...")
    
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
    
    finally:
        env._close()
        print(f"\n✅ Test completed!")

def test_environment_components():
    """Test individual components of the environment"""
    
    print("\n🔧 Testing Environment Components")
    print("=" * 50)
    
    env = DroneEnvironment(render_mode=None)  # No rendering for component tests
    
    # Test physics
    print("🎯 Testing Physics:")
    env.reset()
    initial_pos = (env.x, env.y)
    
    # Apply upward thrust
    action = np.array([1.0, 0.0])  # Max thrust, no differential
    env.step(action)
    
    print(f"   Initial position: {initial_pos}")
    print(f"   After thrust: ({env.x:.1f}, {env.y:.1f})")
    print(f"   Velocity: ({env.xd:.3f}, {env.yd:.3f})")
    
    # Test obstacle movement
    print("\n🐦 Testing Obstacle Movement:")
    initial_bird1 = (env.x_pajaro, env.y_pajaro)
    initial_bird2 = (env.x_pajaro2, env.y_pajaro2)
    
    for _ in range(10):
        env._update_obstacles()
    
    print(f"   Bird 1: {initial_bird1} -> ({env.x_pajaro:.1f}, {env.y_pajaro:.1f})")
    print(f"   Bird 2: {initial_bird2} -> ({env.x_pajaro2:.1f}, {env.y_pajaro2:.1f})")
    
    # Test reward system
    print("\n🎁 Testing Reward System:")
    env.reset()
    initial_targets = env.target_counter
    
    # Move drone close to target
    env.x, env.y = env.xt, env.yt
    env.step(np.array([0.0, 0.0]))
    
    print(f"   Initial targets: {initial_targets}")
    print(f"   After collecting: {env.target_counter}")
    print(f"   Reward: {env.reward}")
    
    env._close()

if __name__ == "__main__":
    # Run tests
    test_sac_environment()
    test_environment_components()