from stable_baselines3 import DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from tqdmCallback import TQDMCallback
from tqdm import tqdm, trange
import json
import random
import itertools
import os
from dqn_parameter_trial import *
import numpy as np

JSON_PATH = "../Data/DDPG_Trials/ddpg_mountain_car_configs.json"
MODEL_PATH = "../Models/DDPG_Trials/"
RESULT_PATH = "../Data/DDPG_Trials/"
TIMESTEPS = 300_000

# param_grid = {
#     "learning_rate": [1e-3, 5e-4, 1e-4],
#     "noise_sigma": [0.1, 0.2, 0.3],
#     "noise_mean": [0.0, 0.1, 0.2],
#     # "exploration_fraction": [0.1, 0.2, 0.3],
#     "buffer_size": [1e6, 1e5, 1e4],
#     "batch_size": [32, 64, 128, 256],
#     "tau": [0.005, 0.001, 0.01],
#     "gamma": [0.95, 0.99, 1.0],
# }

param_grid = {
    "learning_rate": [1e-3, 1e-4],
    "noise_sigma": [0.2],
    "noise_mean": [0.0],
    # "exploration_fraction": [0.1, 0.2, 0.3],
    "buffer_size": [1e6],
    "batch_size": [64],
    "tau": [0.001],
    "gamma": [0.99],
}

if __name__ == "__main__":
    param_configs = get_N_random_sample_params(param_grid, 2)
    append_non_existing_configs_to_json(param_configs, JSON_PATH)
    configs = load_configs_from_json(JSON_PATH)

    for i, config in tqdm(
        enumerate(configs), total=len(configs), desc="Running Configurations"
    ):
        # Check if model already exists
        if os.path.exists(f"{MODEL_PATH}ddpg_model_{i}.zip"):
            continue
        print(f"Running configuration {i}: {config}")

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
