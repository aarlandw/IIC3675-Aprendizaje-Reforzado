#!/usr/bin/env python3
"""
Hyperparameter tuning script for SAC+HER drone training.
Uses Optuna for efficient hyperparameter optimization.
"""

import optuna
import os
import sys
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from env_HER import GoalConditionedDroneEnvironment
import json
import pickle


class HyperparameterCallback(BaseCallback):
    """Callback to track success rate for hyperparameter optimization"""

    def __init__(self, eval_freq=1000):
        super().__init__()
        self.eval_freq = eval_freq
        self.episode_successes = []
        self.success_rates = []

    def _on_step(self) -> bool:
        # Track episode success
        infos = self.locals.get("infos", [])
        if infos and "episode" in infos[0]:
            # Check if any reward was obtained (success)
            episode_reward = infos[0]["episode"]["r"]
            is_success = episode_reward > 0
            self.episode_successes.append(is_success)

            # Calculate rolling success rate every eval_freq steps
            if (
                self.num_timesteps % self.eval_freq == 0
                and len(self.episode_successes) >= 10
            ):
                recent_successes = self.episode_successes[-50:]  # Last 50 episodes
                success_rate = sum(recent_successes) / len(recent_successes)
                self.success_rates.append(success_rate)

        return True

    def get_final_success_rate(self):
        """Get the final success rate for optimization"""
        if not self.episode_successes:
            return 0.0

        # Use last 100 episodes for final evaluation
        recent = (
            self.episode_successes[-100:]
            if len(self.episode_successes) >= 100
            else self.episode_successes
        )
        return sum(recent) / len(recent) if recent else 0.0


def create_env():
    """Create the HER drone environment"""
    base_env = GoalConditionedDroneEnvironment(
        render_mode=None, render_every_frame=False, mouse_target=False
    )
    return DummyVecEnv([lambda: base_env])


def objective(trial):
    """Optuna objective function for hyperparameter optimization"""

    # Sample hyperparameters
    n_sampled_goal = trial.suggest_int("n_sampled_goal", 2, 8)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
    tau = trial.suggest_float("tau", 0.01, 0.1)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    goal_threshold = trial.suggest_float("goal_threshold", 30.0, 100.0)

    print(f"\n🧪 Trial {trial.number}: Testing hyperparameters:")
    print(f"   n_sampled_goal: {n_sampled_goal}")
    print(f"   learning_rate: {learning_rate:.2e}")
    print(f"   tau: {tau:.3f}")
    print(f"   batch_size: {batch_size}")
    print(f"   goal_threshold: {goal_threshold:.1f}")

    try:
        # Create environment with custom goal threshold
        env = create_env()
        # Update goal threshold in the base environment
        env.envs[0].goal_threshold = goal_threshold

        # HER parameters
        her_params = {
            "n_sampled_goal": n_sampled_goal,
            "goal_selection_strategy": GoalSelectionStrategy.FUTURE,
        }

        # SAC parameters
        model = SAC(
            "MultiInputPolicy",
            env,
            learning_rate=learning_rate,
            buffer_size=50000,  # Smaller for faster tuning
            learning_starts=1000,
            batch_size=batch_size,
            tau=tau,
            gamma=0.98,
            train_freq=4,
            gradient_steps=1,
            ent_coef="auto",
            target_update_interval=1,
            target_entropy="auto",
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=her_params,
            verbose=0,  # Reduce output during tuning
        )

        # Training callback
        callback = HyperparameterCallback(eval_freq=1000)

        # Train for limited timesteps (faster tuning)
        timesteps = 20000
        model.learn(total_timesteps=timesteps, callback=callback)

        # Get final success rate
        success_rate = callback.get_final_success_rate()

        print(f"   ✅ Success rate: {success_rate:.2%}")

        # Clean up
        env.close()
        del model

        return success_rate

    except Exception as e:
        print(f"   ❌ Trial failed: {e}")
        return 0.0


def run_hyperparameter_tuning(n_trials=50):
    """Run hyperparameter optimization"""

    print("🔍 Starting HER Hyperparameter Tuning")
    print(f"   Trials: {n_trials}")
    print(f"   Training steps per trial: 20,000")
    print("   Objective: Maximize success rate")
    print()

    # Create study
    study = optuna.create_study(
        direction="maximize",
        study_name="her_drone_optimization",
        storage=None,  # In-memory storage
    )

    # Run optimization
    study.optimize(objective, n_trials=n_trials)

    # Results
    print("\n🎉 Hyperparameter Tuning Complete!")
    print(f"   Best success rate: {study.best_value:.2%}")
    print("   Best parameters:")
    for key, value in study.best_params.items():
        print(f"     {key}: {value}")

    # Save results
    results = {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "n_trials": n_trials,
    }

    with open("best_her_params.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to 'best_her_params.json'")

    return study.best_params, study.best_value


def create_optimized_config(best_params):
    """Create an updated config file with optimized parameters"""

    config_template = f"""
# Optimized HER Configuration
# Generated from hyperparameter tuning

HER_PARAMS = {{
    "n_sampled_goal": {best_params['n_sampled_goal']},
    "goal_selection_strategy": GoalSelectionStrategy.FUTURE,
}}

SAC_PARAMS = {{
    "learning_rate": {best_params['learning_rate']:.2e},
    "buffer_size": 1_000_000,  # Keep large for full training
    "learning_starts": 5000,
    "batch_size": {best_params['batch_size']},
    "tau": {best_params['tau']:.3f},
    "gamma": 0.98,
    "train_freq": 4,
    "gradient_steps": 1,
    "ent_coef": "auto",
    "target_update_interval": 1,
    "target_entropy": "auto",
    "replay_buffer_class": HerReplayBuffer,
    "replay_buffer_kwargs": HER_PARAMS,
}}

# Environment parameter
GOAL_THRESHOLD = {best_params['goal_threshold']:.1f}
"""

    with open("optimized_her_config.py", "w") as f:
        f.write(config_template)

    print(f"📝 Optimized config saved to 'optimized_her_config.py'")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tune HER hyperparameters")
    parser.add_argument(
        "--trials",
        type=int,
        default=30,
        help="Number of optimization trials (default: 30)",
    )

    args = parser.parse_args()

    # Install optuna if not available
    try:
        import optuna
    except ImportError:
        print("Installing optuna for hyperparameter tuning...")
        os.system("pip install optuna")
        import optuna

    # Run tuning
    best_params, best_value = run_hyperparameter_tuning(args.trials)

    # Create optimized config
    create_optimized_config(best_params)

    print(f"\n🚀 Next steps:")
    print(f"   1. Review 'best_her_params.json' for detailed results")
    print(f"   2. Update your Sac_HER.py with the optimized parameters")
    print(f"   3. Run full training with: python Sac_HER.py")
