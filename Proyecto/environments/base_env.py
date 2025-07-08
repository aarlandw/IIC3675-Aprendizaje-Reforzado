import gymnasium as gym
from gymnasium import spaces
import numpy as np
from math import sin, cos, pi, sqrt
from random import randrange
import pygame
import os


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
        
        # Pygame rendering setup
        self.screen = None
        self.pygame_initialized = False
        self.player_sprites = []
        self.target_sprites = []
        self.background_loaded = False

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
        if self.render_mode != "human":
            return
            
        if not self.pygame_initialized:
            self._init_pygame()
            
        # Clear screen with sky blue background
        self.screen.fill((135, 206, 235))
        
        # Draw background elements if available
        self._draw_background()
        
        # Draw target (red circle)
        pygame.draw.circle(self.screen, (255, 0, 0), 
                         (int(self.x_target), int(self.y_target)), 25)
        
        # Draw drone (simple representation)
        drone_x = int(self.x_position)
        drone_y = int(self.y_position)
        
        # Draw drone body
        pygame.draw.circle(self.screen, (50, 50, 50), (drone_x, drone_y), 15)
        
        # Draw drone arms based on angle
        arm_length = 25
        for i in range(4):
            arm_angle = self.angle + i * pi / 2
            end_x = drone_x + arm_length * cos(arm_angle)
            end_y = drone_y + arm_length * sin(arm_angle)
            pygame.draw.line(self.screen, (100, 100, 100), 
                           (drone_x, drone_y), (end_x, end_y), 3)
            # Draw propellers
            pygame.draw.circle(self.screen, (200, 200, 200), 
                             (int(end_x), int(end_y)), 8)
        
        # Draw info text
        collected_text = pygame.font.Font(None, 36).render(
            f"Collected: {self.collected}/{self.max_collected}", True, (255, 255, 255))
        self.screen.blit(collected_text, (10, 10))
        
        position_text = pygame.font.Font(None, 24).render(
            f"Pos: ({int(self.x_position)}, {int(self.y_position)})", True, (255, 255, 255))
        self.screen.blit(position_text, (10, 50))
        
        angle_text = pygame.font.Font(None, 24).render(
            f"Angle: {self.angle:.2f}", True, (255, 255, 255))
        self.screen.blit(angle_text, (10, 75))
        
        # Update display
        pygame.display.flip()
        self.clock.tick(60)
        
    def _init_pygame(self):
        """Initialize pygame components"""
        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Quadcopter Training Environment")
        self.clock = pygame.time.Clock()
        self.pygame_initialized = True
        
        # Try to load sprites if available
        self._load_sprites()
        
    def _load_sprites(self):
        """Load sprite assets if available"""
        try:
            # Try to load drone sprites
            for i in range(1, 5):
                sprite_path = os.path.join(
                    "assets/balloon-flat-asset-pack/png/objects/drone-sprites/",
                    f"drone-{i}.png"
                )
                if os.path.exists(sprite_path):
                    image = pygame.image.load(sprite_path)
                    image = pygame.transform.scale(image, (80, 24))
                    self.player_sprites.append(image)
                    
            # Try to load target sprites
            for i in range(1, 8):
                sprite_path = os.path.join(
                    "assets/balloon-flat-asset-pack/png/balloon-sprites/red-plain/",
                    f"red-plain-{i}.png"
                )
                if os.path.exists(sprite_path):
                    image = pygame.image.load(sprite_path)
                    image = pygame.transform.scale(image, (30, 52))
                    self.target_sprites.append(image)
                    
        except Exception as e:
            print(f"Could not load sprites: {e}")
            # Continue with basic rendering
            
    def _draw_background(self):
        """Draw background elements if available"""
        # Simple gradient background as fallback
        pass

    def close(self):
        if self.pygame_initialized:
            pygame.quit()
            self.pygame_initialized = False
