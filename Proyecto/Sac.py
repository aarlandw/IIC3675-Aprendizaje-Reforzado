import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
from env import QuadcopterEnv  # Asegúrate de que esté correctamente implementado
from env_SAC import droneEnv
import matplotlib.pyplot as plt
import os
from gymnasium.wrappers import RecordVideo

class RewardCallback(BaseCallback):
    def __init__(self, total_timesteps):
        super().__init__()
        self.episode_rewards = []
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self):
        # Crear barra de progreso al inicio del entrenamiento
        self.pbar = tqdm(total=self.total_timesteps, desc="Entrenando")

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if infos and "episode" in infos[0]:
            r = infos[0]["episode"]["r"]
            self.episode_rewards.append(r)

        # Avanzar la barra de progreso
        self.pbar.update(1)
        return True

    def _on_training_end(self):
        self.pbar.close()

# Crear entorno (si no lo registraste con gym, usa QuadcopterEnv directamente)
env = droneEnv(render_mode="rgb_array", render_every_frame=True, mouse_target=False)

# Envolver con RecordVideo para grabar cada 200 episodios
video_folder = "video2"
os.makedirs(video_folder, exist_ok=True)

env = RecordVideo(
    env,
    video_folder=video_folder,
    episode_trigger=lambda ep: ep % 200 == 0,  # Guardar cada 200 episodios
    name_prefix="drone_episode",
    disable_logger=True
)

# Inicializar callback
callback = RewardCallback(1_000_000)

# Crear el modelo SAC
model = SAC("MlpPolicy", env, verbose=0)

try:
    # Entrenamiento del modelo
    model.learn(total_timesteps=1_000_000, callback=callback)

except KeyboardInterrupt:
    print("\n Entrenamiento interrumpido por el usuario. Guardando progreso...")

finally:
    # Guardar el modelo
    model.save("sac_drone_model.zip")
    print(" Modelo guardado como 'sac_drone_model.zip'")

    # Graficar recompensas
    if callback.episode_rewards:
        plt.figure(figsize=(10, 5))
        plt.plot(callback.episode_rewards, label="Recompensa por episodio", alpha=0.8)
        plt.xlabel("Episodio")
        plt.ylabel("Recompensa")
        plt.title("Recompensa por Episodio - SAC objetivo aleatorio")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("⚠️ No se registraron recompensas para graficar.")