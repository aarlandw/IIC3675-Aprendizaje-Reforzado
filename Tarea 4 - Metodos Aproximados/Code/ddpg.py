import gymnasium as gym
from stable_baselines3 import DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from tqdmCallback import TQDMCallback
from tqdm import tqdm, trange
import os
import numpy as np


NUM_RUNS = 30
MODEL_PATH = "../Models/DDPG/"
RESULT_PATH = "../Data/DDPG/"
TIMESTEPS = 300_000
LOG_INTERVAL = 10

config = {
    "learning_rate": 0.0001,
    "noise_sigma": 0.1,
    "noise_mean": 0.0,
    "buffer_size": 1000000.0,
    "batch_size": 32,
    "tau": 0.01,
    "gamma": 1.0,
}


for i in trange(NUM_RUNS, desc="Running Configurations"):
    
    # Check if model already exists
    if os.path.exists(f"{MODEL_PATH}ddpg_model_{i}.zip"):
        continue

    # Set up environment and monitoring
    env = gym.make("MountainCarContinuous-v0")
    env = Monitor(env, filename=f"{RESULT_PATH}ddpg_results_{i}")

    # Configure action noise
    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(
        mean=config["noise_mean"], sigma=config["noise_sigma"] * np.ones(n_actions)
    )

    # Create model
    model = DDPG(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=config["learning_rate"],
        buffer_size=int(config["buffer_size"]),
        # learning_starts=1000,
        gamma=config["gamma"],
        # exploration_fraction=config["exploration_fraction"],
        train_freq=(1, "episode"),  # Standard
        batch_size=config["batch_size"],
        tau=config["tau"],
        action_noise=action_noise,
    )

    # Train
    callback = TQDMCallback(total_timesteps=TIMESTEPS, verbose=1)
    model.learn(total_timesteps=TIMESTEPS, log_interval=10, callback=callback)

    # Save model
    model.save(f"{MODEL_PATH}ddpg_model_{i}")


