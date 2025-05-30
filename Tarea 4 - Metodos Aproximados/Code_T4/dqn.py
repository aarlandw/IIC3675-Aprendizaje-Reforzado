import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from tqdmCallback import TQDMCallback
from tqdm import tqdm, trange
import os

NUM_RUNS = 30
MODEL_PATH = "../Models/DQN/"
DATA_PATH = "../Data/DQN/"
TIMESTEPS = 300_000
LOG_INTERVAL = 10

dqn_param_config = {
    "learning_rate": 0.001,
    "exploration_fraction": 0.1,
    "gamma": 0.99,
    "net_arch": 
        [128, 128],
    
}

for i in trange(NUM_RUNS, desc="Running Configurations"):
    
    # Check if model already exists
    if os.path.exists(f"{MODEL_PATH}dqn_model_{i}.zip"):
        continue

    # Set up environment and monitoring
    env = gym.make("MountainCar-v0")
    env = Monitor(env, filename=f"{DATA_PATH}dqn_results_{i}")

    # Configure network architecture
    policy_kwargs = dict(net_arch=dqn_param_config["net_arch"])

    # Create model
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=dqn_param_config["learning_rate"],
        buffer_size=50000,
        learning_starts=1000,
        gamma=dqn_param_config["gamma"],
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        exploration_fraction=dqn_param_config["exploration_fraction"],
        train_freq=4,
        batch_size=64,
        target_update_interval=1000,
        policy_kwargs=policy_kwargs,
    )

    # Train
    callback = TQDMCallback(total_timesteps=TIMESTEPS, verbose=1)
    model.learn(total_timesteps=TIMESTEPS, log_interval=LOG_INTERVAL, callback=callback)

    # Save model
    model.save(f"{MODEL_PATH}dqn_model_{i}")

