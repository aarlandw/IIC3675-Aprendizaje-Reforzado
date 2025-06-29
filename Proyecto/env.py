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
        

        self.WIDTH = 800
        self.HEIGHT = 800
    
        self.max_collected = 5 # objetivos 
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

        # episodio recompensa
        self.episode_rewards = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
       
        # Estado inicial
        self.angle = 0.0
        self.angular_speed = 0.0
        self.x_position = 400.0
        self.y_position = 400.0
        self.x_speed = 0.0
        self.y_speed = 0.0

        self.collected = 0

        self.x_target = randrange(200, 600)
        self.y_target = randrange(200, 600)


        self.collected = 0

        obs = self._get_obs()

        self.episode_rewards = 0
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

        
        # Calcular distancia al objetivo
        dist = sqrt((self.x_position - self.x_target) ** 2 + (self.y_position - self.y_target) ** 2)

        #castigola distancia
        reward = - dist/500 #1/(1 + dist)  # reward = -dist / 800.0 
        #premio por no morir
        reward += 1 / 50

        terminated = False

        if dist < 50:
            reward += 100.0
            self.collected += 1
            self.x_target = randrange(200, 600)
            self.y_target = randrange(200, 600)

        if self.collected >= self.max_collected:
            terminated = True


        if dist > 1000 or self.x_position < 0 or self.y_position < 0 or self.x_position > self.WIDTH or self.y_position > self.HEIGHT:
            reward -= 1000.0
            terminated = True

        obs = self._get_obs()
        info = {}
        self.episode_rewards += reward
        if terminated:
            info["episode"] = {"r": self.episode_rewards, "l": self.collected}
            self.episode_rewards = 0
        return obs, reward, terminated, False, info

    def render(self):
        pass

    def close(self):
        pass
