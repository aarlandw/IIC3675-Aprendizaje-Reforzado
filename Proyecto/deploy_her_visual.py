#!/usr/bin/env python3
"""
Deploy and visually test the HER model
Watch the trained agent collect balloons while avoiding birds!
"""

import argparse
import os
import time
import pygame
import numpy as np
from stable_baselines3 import SAC
from env_SAC import DroneEnvironment
from her_sparse_wrapper import SparseRewardHERWrapper


def deploy_her_model(
    model_path="sparse_her_balloon.zip", episodes=5, render_speed=0.03
):
    """Deploy and test the HER model visually"""

    print(f"🎈 Deploying HER Model: {model_path}")
    print("=" * 60)

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("📁 Available models:")
        for file in os.listdir("."):
            if file.endswith(".zip"):
                print(f"   - {file}")
        return

    # Create environment FIRST (needed for HER model loading)
    print("🎮 Creating environment...")
    base_env = DroneEnvironment(render_mode="human", render_every_frame=True)
    her_env = SparseRewardHERWrapper(base_env, goal_threshold=25.0)

    # Load the trained model WITH environment
    try:
        print("🧠 Loading HER model...")
        model = SAC.load(model_path, env=her_env)  # ← KEY FIX: Pass environment
        print("✅ Model loaded successfully!")

        # Get model info
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        print(f"📊 Model size: {file_size:.1f} MB")

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        her_env.env._close()
        return

    print(f"🎯 Testing for {episodes} episodes...")
    print("\n📋 What to watch for:")
    print("   🚁 Blue drone (your trained agent)")
    print("   🎈 Red balloons (targets to collect)")
    print("   🐦 Moving birds (obstacles to avoid)")
    print("   🟢 Green circle (goal area around current balloon)")
    print("   📏 Line from drone to balloon (green=close, red=far)")
    print("   📊 Balloon counter on screen")
    print("\n🎮 Controls:")
    print("   ESC or close window = stop testing")
    print("   Watch how the agent navigates!\n")

    episode_stats = []
    total_balloons = 0

    try:
        for episode in range(episodes):
            print(f"🎯 Episode {episode + 1}/{episodes}")

            obs, info = her_env.reset()
            episode_reward = 0
            episode_length = 0
            balloons_collected = 0
            max_balloons = 0
            positions_visited = []

            running = True
            while running:
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False

                if not running:
                    break

                # Get action from trained model
                # Ensure obs is in the correct format for the model
                if isinstance(obs, dict):
                    obs_dict = obs
                else:
                    # If obs is not a dict, convert it to the expected format
                    obs_dict = {
                        "observation": obs,
                        "achieved_goal": obs,
                        "desired_goal": obs,
                    }

                action, _ = model.predict(obs_dict, deterministic=True)
                obs, reward, terminated, truncated, info = her_env.step(action)

                episode_reward += reward
                episode_length += 1
                balloons_collected = info.get("balloons_collected", 0)
                max_balloons = max(max_balloons, balloons_collected)

                # Track drone position
                drone_pos = (her_env.env.x, her_env.env.y)
                positions_visited.append(drone_pos)

                # Render the environment
                her_env.render()

                # Control rendering speed
                time.sleep(render_speed)

                # Print progress every 200 steps
                if episode_length % 200 == 0:
                    success = info.get("is_success", False)
                    distance_to_goal = np.linalg.norm(
                        np.array([her_env.env.x, her_env.env.y]) - her_env.current_goal
                    )
                    print(
                        f"   Step {episode_length}: "
                        f"Balloons={balloons_collected}, "
                        f"Distance to goal={distance_to_goal:.1f}, "
                        f"Success={success}"
                    )

                # Episode ended
                if terminated or truncated:
                    break

            if not running:
                print("🛑 Testing stopped by user")
                break

            # Calculate episode statistics
            total_balloons += max_balloons
            original_reward = info.get("original_reward", 0)
            success = info.get("is_success", False)

            # Calculate path efficiency
            if len(positions_visited) > 1:
                total_distance = sum(
                    np.linalg.norm(
                        np.array(positions_visited[i + 1])
                        - np.array(positions_visited[i])
                    )
                    for i in range(len(positions_visited) - 1)
                )
                efficiency = (
                    max_balloons / (total_distance / 100) if total_distance > 0 else 0
                )
            else:
                efficiency = 0

            episode_stats.append(
                {
                    "episode": episode + 1,
                    "balloons": max_balloons,
                    "length": episode_length,
                    "reward": episode_reward,
                    "original_reward": original_reward,
                    "success": success,
                    "efficiency": efficiency,
                }
            )

            # Episode summary
            print(f"   ✅ Episode {episode + 1} Results:")
            print(f"      🎈 Balloons: {max_balloons}")
            print(f"      📏 Steps: {episode_length}")
            print(f"      🏆 HER Reward: {episode_reward:.1f}")
            print(f"      🎯 Success: {'✅' if success else '❌'}")
            print(f"      🛣️  Efficiency: {efficiency:.2f}")
            print()

    except KeyboardInterrupt:
        print("\n🛑 Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
    finally:
        her_env.env._close()

    # Overall performance analysis
    if episode_stats:
        print(f"📊 OVERALL PERFORMANCE ANALYSIS")
        print("=" * 50)

        avg_balloons = np.mean([s["balloons"] for s in episode_stats])
        avg_length = np.mean([s["length"] for s in episode_stats])
        avg_reward = np.mean([s["reward"] for s in episode_stats])
        success_rate = np.mean([s["success"] for s in episode_stats])
        avg_efficiency = np.mean([s["efficiency"] for s in episode_stats])

        print(f"🎈 Average Balloons per Episode: {avg_balloons:.1f}")
        print(f"📏 Average Episode Length: {avg_length:.1f} steps")
        print(f"🏆 Average HER Reward: {avg_reward:.1f}")
        print(f"🎯 Success Rate: {success_rate:.2%}")
        print(f"🛣️  Average Efficiency: {avg_efficiency:.2f}")
        print(f"📊 Total Balloons Collected: {total_balloons}")

        # Performance rating
        if avg_balloons >= 3:
            rating = "🌟🌟🌟 EXCELLENT"
        elif avg_balloons >= 2:
            rating = "🌟🌟 GOOD"
        elif avg_balloons >= 1:
            rating = "🌟 FAIR"
        else:
            rating = "❌ NEEDS IMPROVEMENT"

        print(f"📈 Performance Rating: {rating}")

        # Best episode
        best_episode = max(episode_stats, key=lambda x: x["balloons"])
        print(f"\n🏆 Best Episode: #{best_episode['episode']}")
        print(f"   Balloons: {best_episode['balloons']}")
        print(f"   Steps: {best_episode['length']}")
        print(f"   Efficiency: {best_episode['efficiency']:.2f}")

        # Detailed breakdown
        print(f"\n📋 Episode-by-Episode Breakdown:")
        for stat in episode_stats:
            status = "✅" if stat["success"] else "❌"
            print(
                f"   Episode {stat['episode']}: "
                f"{stat['balloons']} balloons, "
                f"{stat['length']} steps, "
                f"eff: {stat['efficiency']:.2f} {status}"
            )

    print(f"\n✅ Visual deployment complete!")


def compare_models(model_paths, episodes=3):
    """Compare multiple models side by side"""
    print("🔬 Model Comparison Mode")
    print("=" * 40)

    results = {}

    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f"\n🧪 Testing {model_path}...")

            # Create environment for each model
            base_env = DroneEnvironment(render_mode=None)
            her_env = SparseRewardHERWrapper(base_env, goal_threshold=25.0)

            # Load model with environment
            model = SAC.load(model_path, env=her_env)

            episode_balloons = []

            for episode in range(episodes):
                obs, _ = her_env.reset()
                balloons = 0

                for _ in range(1000):  # Max 1000 steps
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = her_env.step(action)
                    balloons = info.get("balloons_collected", 0)

                    if terminated or truncated:
                        break

                episode_balloons.append(balloons)

            avg_balloons = np.mean(episode_balloons)
            results[model_path] = avg_balloons

            print(f"   Average balloons: {avg_balloons:.1f}")
            her_env.env._close()

    # Show comparison
    print(f"\n📊 Model Comparison Results:")
    print("=" * 40)
    for model_path, performance in sorted(
        results.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"   {model_path}: {performance:.1f} balloons")


def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description="Deploy and test HER model")
    parser.add_argument(
        "--model", default="sparse_her_balloon.zip", help="Model to test"
    )
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    parser.add_argument(
        "--speed", type=float, default=0.03, help="Rendering speed (lower = slower)"
    )
    parser.add_argument("--compare", nargs="+", help="Compare multiple models")

    args = parser.parse_args()

    if args.compare:
        compare_models(args.compare, episodes=3)
    else:
        deploy_her_model(args.model, args.episodes, args.speed)


if __name__ == "__main__":
    main()
