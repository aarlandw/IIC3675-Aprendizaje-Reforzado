import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from tqdmCallback import TQDMCallback
from tqdm import tqdm
import json
import random
import itertools
import os


JSON_PATH = "../Data/DQN_Trials/dqn_mountain_car_configs.json"
MODEL_PATH = "../Models/DQN_Trials/"
RESULT_PATH = "../Data/DQN_Trials/"

param_grid = {
    "learning_rate": [1e-3, 5e-4, 1e-4],
    "exploration_fraction": [0.1, 0.2, 0.3],
    "gamma": [0.99],
    "net_arch": [
        # [64, 64],
        [128, 128],
    ],
}

def get_N_random_sample_params(param_grid, N):
    keys, values = zip(*param_grid.items())
    param_configs = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return random.sample(param_configs, k=N)

def get_all_param_combinations(param_grid):
    keys, values = zip(*param_grid.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


def save_configs_to_json(param_configs, json_path):
    with open(json_path, "w") as f:
        json.dump(param_configs, f, indent=4)
    print(f"Saved {len(param_configs)} configurations to {json_path}")
    
def load_configs_from_json(json_path):
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} existing configurations from {json_path}")
        return data
    except FileNotFoundError:
        print(f"File {json_path} not found.")
        return []
    
def append_non_existing_configs_to_json(param_configs, json_path):
    existing_configs = load_configs_from_json(json_path)
    for config in param_configs:
        if config not in existing_configs:
            existing_configs.append(config)
    save_configs_to_json(existing_configs, json_path)
    

if __name__ == "__main__":
    
    configs = load_configs_from_json(JSON_PATH)
    for i, config in tqdm(
        enumerate(configs), total=len(configs), desc="Running Configurations"
    ):
        # Check if model already exists
        if os.path.exists(f"{MODEL_PATH}dqn_model_{i}.zip"):
            print(f"Skipping config {i+1}/{len(configs)}: {config}")
            continue
        print(f"Running config {i+1}/{len(configs)}: {config}")

        # Set up environment and monitoring
        env = gym.make("MountainCar-v0")
        env = Monitor(env, filename=f"{RESULT_PATH}dqn_results_{i}")

        # Configure network architecture
        policy_kwargs = dict(net_arch=config["net_arch"])

        # Create model
        model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=config["learning_rate"],
            buffer_size=50000,
            # learning_starts=1000,
            gamma=config["gamma"],
            exploration_initial_eps=1.0,
            exploration_final_eps=0.01,
            exploration_fraction=config["exploration_fraction"],
            train_freq=4,
            batch_size=64,
            target_update_interval=1000,
            policy_kwargs=policy_kwargs,
        )

        # Train
        callback = TQDMCallback(total_timesteps=300_000, verbose=1)
        model.learn(total_timesteps=300_000, log_interval=10, callback=callback)

        # Save model
        model.save(f"{MODEL_PATH}dqn_model_{i}")


