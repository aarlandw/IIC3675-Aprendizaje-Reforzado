import gymnasium as gym
from gymnasium import spaces
import numpy as np
from math import sin, cos, pi, sqrt
from random import randrange


class QuadcopterEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()

        # Parámetros físicos
        self.gravity = 0.08
        self.thruster_mean = 0.04
        self.thruster_amplitude = 0.04
        self.diff_amplitude = 0.003
        self.mass = 1
        self.arm = 25
        self.time_limit = 30  # segundos

        self.WIDTH = 800
        self.HEIGHT = 800

        # Acción: [up/down thrust, left/right torque] -> continuo
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32,
        )

        # Observación: posición (x, y), velocidad (x_dot, y_dot), ángulo y velocidad angular, posición objetivo
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -np.inf, -np.inf, -np.pi, -np.inf, 0, 0]),
            high=np.array([800, 800, np.inf, np.inf, np.pi, np.inf, 800, 800]),
            dtype=np.float32,
        )

        self.render_mode = render_mode
        self.window = None
        self.clock = None

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Estado inicial
        self.angle = 0.0
        self.angular_speed = 0.0
        self.x_position = 400.0
        self.y_position = 400.0
        self.x_speed = 0.0
        self.y_speed = 0.0

        self.x_target = randrange(200, 600)
        self.y_target = randrange(200, 600)

        self.elapsed_time = 0.0
        self.collected = 0

        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        return np.array([
            self.x_position,
            self.y_position,
            self.x_speed,
            self.y_speed,
            self.angle,
            self.angular_speed,
            self.x_target,
            self.y_target,
        ], dtype=np.float32)

    def step(self, action):
        up_thrust = np.clip(action[0], -1.0, 1.0) * self.thruster_amplitude
        diff = np.clip(action[1], -1.0, 1.0) * self.diff_amplitude

        thruster_left = self.thruster_mean + up_thrust - diff
        thruster_right = self.thruster_mean + up_thrust + diff

        # Aceleraciones
        x_acc = -(thruster_left + thruster_right) * sin(self.angle) / self.mass
        y_acc = self.gravity - (thruster_left + thruster_right) * cos(self.angle) / self.mass
        angular_acc = self.arm * (thruster_right - thruster_left) / self.mass

        # Integración
        self.x_speed += x_acc
        self.y_speed += y_acc
        self.angular_speed += angular_acc

        self.x_position += self.x_speed
        self.y_position += self.y_speed
        self.angle += self.angular_speed

        # Tiempo y estado
        self.elapsed_time += 1 / 60.0

        # Calcular distancia al objetivo
        dist = sqrt((self.x_position - self.x_target) ** 2 + (self.y_position - self.y_target) ** 2)
        reward = -dist / 800.0  # castigo proporcional a la distancia

        terminated = False
        truncated = self.elapsed_time >= self.time_limit

        if dist < 50:
            reward += 10.0  # recompensa por alcanzar el objetivo
            self.collected += 1
            self.x_target = randrange(200, 600)
            self.y_target = randrange(200, 600)

        if dist > 1000 or self.x_position < 0 or self.y_position < 0 or self.x_position > self.WIDTH or self.y_position > self.HEIGHT:
            reward -= 10.0
            terminated = True

        obs = self._get_obs()
        return obs, reward, terminated, truncated, {}

    def render(self):
        # Puedes integrar aquí pygame si quieres visualización
        pass

    def close(self):
        pass