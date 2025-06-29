import json
import os
import numpy as np
from tqdm import tqdm
import optuna
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from env import QuadcopterEnv  # Asegúrate de que este import apunte correctamente a tu archivo
import matplotlib.pyplot as plt
# Configuración
N_TRIALS = 50
TOTAL_TIMESTEPS = 100_000
EVAL_EPISODES = 10
BEST_PARAMS_FILE = "best_params_optuna.json"

from tqdm import tqdm
from stable_baselines3.common.callbacks import BaseCallback

class RewardCallback(BaseCallback):
    def __init__(self, total_timesteps=100_000):
        super().__init__()
        self.episode_rewards = []
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self) -> None:
        from tqdm import tqdm
        self.pbar = tqdm(total=self.total_timesteps, desc="Entrenando", unit="step")

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if infos and "episode" in infos[0]:
            self.episode_rewards.append(infos[0]["episode"]["r"])
        if self.pbar:
            self.pbar.update(1)
        return True

    def _on_training_end(self) -> None:
        if self.pbar:
            self.pbar.close()

# Función para evaluar el modelo entrenado
def evaluate_model(model, env, n_episodes=EVAL_EPISODES):
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        rewards.append(episode_reward)
    return np.mean(rewards)

# Función objetivo de Optuna
def objective(trial):
    params = {
        "gamma": trial.suggest_float("gamma", 0.95, 0.99),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-4),
        "buffer_size": trial.suggest_categorical("buffer_size", [10_000, 50_000, 100_000]),
        "tau": trial.suggest_categorical("tau", [0.005, 0.01]),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128]),
    }

    env = QuadcopterEnv()
    model = SAC(
        "MlpPolicy",
        env,
        verbose=0,
        tensorboard_log="./logs/optuna/",
        **params
    )

    callback = RewardCallback()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

    mean_reward = evaluate_model(model, env)
    env.close()

    def guardar_mejor_modelo():
        model.save("best_model_optuna.zip")
        print(f" Nuevo mejor modelo! Recompensa: {mean_reward:.2f}")
        with open(BEST_PARAMS_FILE, "w") as f:
            json.dump(params, f, indent=4)
        with open("best_episode_rewards.json", "w") as f:
            json.dump(callback.episode_rewards, f)
        with open("best_params_for_plot.json", "w") as f:
            json.dump(params, f)

    try:
        if mean_reward > trial.study.best_value:
            guardar_mejor_modelo()
    except ValueError:
        # Primer trial o aún no hay mejores → guardamos por defecto
        guardar_mejor_modelo()

    # Resultados parciales
    print(f" Trial #{trial.number} finalizado. Recompensa media: {mean_reward:.2f}")
    print(f" Parámetros usados: {json.dumps(params)}\n")

    return mean_reward
# Ejecutar la optimización
def optimize_hyperparameters():
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner()
    )

    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print("\nOptimización completada!")
    print(f"Mejor recompensa: {study.best_value:.2f}")
    print("Mejores hiperparámetros:")
    print(json.dumps(study.best_params, indent=4))

    return study.best_params

def plot_best_model_rewards():
    import matplotlib.pyplot as plt
    with open("best_episode_rewards.json", "r") as f:
        rewards = json.load(f)
    with open("best_params_for_plot.json", "r") as f:
        best_params = json.load(f)

    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa")
    plt.title("Recompensas por Episodio del Mejor Modelo")
    legend = ", ".join([f"{k}={v}" for k, v in best_params.items()])
    plt.legend([legend], loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.grid()
    plt.savefig("mejor_modelo_recompensas.png")
    plt.show()


if __name__ == "__main__":
    try:
        best_params = optimize_hyperparameters()
        plot_best_model_rewards()
    except KeyboardInterrupt:
        print("\n Entrenamiento interrumpido por el usuario.")

        # Si hay recompensas parciales, intentamos graficarlas
        if os.path.exists("best_episode_rewards.json") and os.path.exists("best_params_for_plot.json"):
            print(" Mostrando gráfica con resultados parciales...")
            plot_best_model_rewards()
        else:
            print("⚠️ No se encontraron datos de recompensas guardados aún.")