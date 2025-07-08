#!/usr/bin/env python3
"""
Fixed clean drone recording for HER models
"""

import numpy as np
import pygame
import cv2
import os
import time
from stable_baselines3 import SAC
from env_SAC import DroneEnvironment
from her_sparse_wrapper import SparseRewardHERWrapper
from stable_baselines3.common.vec_env import DummyVecEnv


def record_drone_flight(
    model_path,
    output_file="drone_flight.mp4",
    episodes=3,
    fps=30,
    resolution=(800, 600),
):
    """
    Record clean drone flight without HER visualization - FIXED for HER models
    """

    print(f"🎥 Recording drone flight from: {model_path}")
    print(f"📁 Output: {output_file}")
    print(f"🎬 Episodes: {episodes}, FPS: {fps}")

    # Create environment for recording (clean, no HER wrapper)
    base_env = DroneEnvironment(render_mode="rgb_array")

    # Create HER environment for model loading (needed for HER buffer)
    her_env = SparseRewardHERWrapper(
        DroneEnvironment(render_mode=None), goal_threshold=25.0
    )
    vec_env = DummyVecEnv([lambda: her_env])

    # Load model WITH environment (required for HER)
    try:
        if model_path.endswith(".zip"):
            print("🔄 Loading HER model (requires environment)...")
            model = SAC.load(model_path, env=vec_env)  # ← FIX: Pass environment!
            print("✅ HER model loaded successfully!")
        else:
            print(f"❌ Model file should end with .zip: {model_path}")
            return
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("💡 Make sure the model was trained with HER!")
        return

    # Setup video recording
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, resolution)

    total_frames = 0

    try:
        for episode in range(episodes):
            print(f"\n🎬 Recording Episode {episode + 1}/{episodes}")

            # Reset environment
            obs, info = base_env.reset()
            episode_frames = 0
            balloons_collected = 0

            # Track balloon position for counting
            prev_balloon_pos = np.array([base_env.xt, base_env.yt])

            while True:
                # Convert observation to HER format for prediction
                obs_dict = {
                    "observation": np.array(
                        [
                            base_env.x,
                            base_env.y,
                            base_env.xd,
                            base_env.yd,
                            base_env.a,
                            base_env.ad,
                            base_env.xt,
                            base_env.yt,
                        ],
                        dtype=np.float32,
                    ),
                    "achieved_goal": np.array(
                        [base_env.x, base_env.y], dtype=np.float32
                    ),
                    "desired_goal": np.array(
                        [base_env.xt, base_env.yt], dtype=np.float32
                    ),
                }

                # Get action from HER model
                action, _ = model.predict(obs_dict, deterministic=True)

                # Step environment
                obs, reward, terminated, truncated, info = base_env.step(action)

                # Count balloons collected
                current_balloon_pos = np.array([base_env.xt, base_env.yt])
                if not np.allclose(current_balloon_pos, prev_balloon_pos, atol=10.0):
                    balloons_collected += 1
                    prev_balloon_pos = current_balloon_pos.copy()
                    print(f"🎈 Balloon #{balloons_collected} collected!")

                # Render frame (clean, no HER visualization)
                frame = base_env.render()
                if frame is not None:
                    # Convert RGB to BGR for OpenCV
                    if len(frame.shape) == 3:
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                        # Resize if needed
                        if frame_bgr.shape[:2] != (resolution[1], resolution[0]):
                            frame_bgr = cv2.resize(frame_bgr, resolution)

                        # Add clean info overlay
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        # cv2.putText(
                        #     frame_bgr,
                        #     f"Episode {episode + 1}/{episodes}",
                        #     (10, 30),
                        #     font,
                        #     1,
                        #     (255, 255, 255),
                        #     2,
                        # )
                        # cv2.putText(
                        #     frame_bgr,
                        #     f"Balloons: {balloons_collected}",
                        #     (10, 60),
                        #     font,
                        #     1,
                        #     (0, 255, 0),  # Green for balloons
                        #     2,
                        # )
                        # cv2.putText(
                        #     frame_bgr,
                        #     f"Steps: {episode_frames}",
                        #     (10, 90),
                        #     font,
                        #     0.7,
                        #     (255, 255, 255),
                        #     2,
                        # )

                        # Write frame
                        video_writer.write(frame_bgr)
                        total_frames += 1
                        episode_frames += 1

                # Check for episode end
                if terminated or truncated:
                    print(
                        f"   Episode {episode + 1} complete: {episode_frames} frames, {balloons_collected} balloons"
                    )
                    break

                # Safety limit
                if episode_frames > 5000:  # Longer limit for extended episodes
                    print(
                        f"   Episode {episode + 1} reached frame limit with {balloons_collected} balloons"
                    )
                    break

    except KeyboardInterrupt:
        print("\n⏹️ Recording interrupted by user")

    finally:
        # Clean up
        video_writer.release()
        base_env.close()
        vec_env.close()

        print(f"\n✅ Recording complete!")
        print(f"📁 Saved: {output_file}")
        print(f"🎬 Total frames: {total_frames}")
        print(f"⏱️ Duration: {total_frames/fps:.1f} seconds")


def record_multiple_models():
    """Record flights from multiple models for comparison"""

    models_to_test = [
        ("sparse_her_balloon_best.zip", "best_model_flight.mp4"),
        ("sparse_her_balloon_checkpoint.zip", "checkpoint_flight.mp4"),
        ("sparse_her_balloon.zip", "final_model_flight.mp4"),
    ]

    for model_path, output_file in models_to_test:
        if os.path.exists(model_path):
            print(f"\n{'='*50}")
            record_drone_flight(model_path, output_file, episodes=2, fps=30)
        else:
            print(f"⚠️ Model not found: {model_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Record clean drone flight videos")
    parser.add_argument(
        "--model",
        type=str,
        default="sparse_her_balloon_best.zip",
        help="Path to model file (.zip)",
    )
    parser.add_argument(
        "--output", type=str, default="drone_flight.mp4", help="Output video file"
    )
    parser.add_argument(
        "--episodes", type=int, default=3, help="Number of episodes to record"
    )
    parser.add_argument("--fps", type=int, default=30, help="Video FPS")
    parser.add_argument(
        "--all", action="store_true", help="Record all available models"
    )

    args = parser.parse_args()

    if args.all:
        record_multiple_models()
    else:
        record_drone_flight(args.model, args.output, args.episodes, args.fps)
