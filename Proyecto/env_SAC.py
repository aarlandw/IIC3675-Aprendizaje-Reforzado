import pygame
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from random import randrange
from math import sin, cos, pi, sqrt
from random import uniform, choice

class droneEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    def __init__(self,render_mode = None, render_every_frame=False, mouse_target=False):
        super().__init__()
        pygame.init()

        self.WIDTH, self.HEIGHT = 800, 800
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.FPS = 60
        self.FramePerSec = pygame.time.Clock()
        self.myfont = pygame.font.Font("assets/fonts/Roboto-Regular.ttf", 30)
       
        self.render_every_frame = render_every_frame
        self.mouse_target = mouse_target

        self.render_mode = render_mode
        # Física
        self.FPS = 60
        self.gravity = 0.08
        self.thruster_amplitude = 0.04
        self.diff_amplitude = 0.003
        self.thruster_mean = 0.04
        self.mass = 1
        self.arm = 25

        # Initialize variables
        (self.a, self.ad, self.add) = (0, 0, 0)
        (self.x, self.xd, self.xdd) = (400, 0, 0)
        (self.y, self.yd, self.ydd) = (400, 0, 0)
        self.xt = randrange(200, 600)
        self.yt = randrange(200, 600)

        #Obstaculo
        self.a_pajaro, self.b_pajaro = randrange(-3, 3),randrange(0, 800)
        self.x_pajaro = 0
        self.y_pajaro = self.a_pajaro*self.x_pajaro+self.b_pajaro
        self.a_pajaro2, self.b_pajaro2 = choice([x * 0.1 for x in range(-20, 21)]),randrange(-800, 800)
        self.x_pajaro2 = 799
        self.y_pajaro2 = self.a_pajaro2*self.x_pajaro2+self.b_pajaro2

        # Initialize game variables
        self.target_counter = 0
        self.reward = 0
        self.time = 0
        self.time_limit = 20
        if self.mouse_target is True:
            self.time_limit = 1000

        # Animaciones
        self.step_counter = 0
        self.target_counter = 0
        self.time = 0

        # Sprites
        self.load_assets()

        # Espacios
        # 2 action thrust amplitude and thrust difference in float values between -1 and 1
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))
        # 12 observations: angle_to_up, velocity, angle_velocity, distance_to_target, angle_to_target, angle_target_and_velocity, 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)

        # episodio recompensa
        self.episode_rewards = 0
        self.reset()






    def load_assets(self):
        # Player sprites
        self.player_width = 80
        self.player = []
        for i in range(1, 5):
            img = pygame.image.load(f"assets/balloon-flat-asset-pack/png/objects/drone-sprites/drone-{i}.png")
            img = pygame.transform.scale(img, (self.player_width, int(self.player_width * 0.3)))
            self.player.append(img)

        # Target sprites
        self.target_width = 30
        self.target = []
        for i in range(1, 8):
            img = pygame.image.load(f"assets/balloon-flat-asset-pack/png/balloon-sprites/red-plain/red-plain-{i}.png")
            img = pygame.transform.scale(img, (self.target_width, int(self.target_width * 1.73)))
            self.target.append(img)

        # Background elements
        self.cloud1 = pygame.image.load("assets/balloon-flat-asset-pack/png/background-elements/cloud-1.png")
        self.cloud2 = pygame.image.load("assets/balloon-flat-asset-pack/png/background-elements/cloud-2.png")
        self.sun = pygame.image.load("assets/balloon-flat-asset-pack/png/background-elements/sun.png")


        self.cloud1.set_alpha(124)
        self.cloud2.set_alpha(124)
        self.sun.set_alpha(124)

        self.x_cloud1, self.y_cloud1, self.speed_cloud1 = 150, 200, 0.3
        self.x_cloud2, self.y_cloud2, self.speed_cloud2 = 400, 500, -0.2

        # Obstaculo
        self.pajaro = pygame.image.load("assets/sprites/pajaro.png")
        self.pajaro2 = pygame.image.load("assets/sprites/pajaro2.png")


        
    def reset(self, seed=None, options=None):
        # Reset variables
        (self.a, self.ad, self.add) = (0, 0, 0)
        (self.x, self.xd, self.xdd) = (400, 0, 0)
        (self.y, self.yd, self.ydd) = (400, 0, 0)
        self.xt = randrange(200, 600)
        self.yt = randrange(200, 600)

        self.episode_rewards = 0
        self.target_counter = 0
        self.reward = 0
        self.time = 0

        #Obstaculo
        self.a_pajaro, self.b_pajaro = randrange(-3, 3),randrange(0, 800)
        self.x_pajaro = 0
        self.y_pajaro = self.a_pajaro*self.x_pajaro+self.b_pajaro
        self.a_pajaro2, self.b_pajaro2 = choice([x * 0.1 for x in range(-10, 11)]),randrange(-800, 800)
        self.x_pajaro2 = 799
        self.y_pajaro2 = self.a_pajaro2*self.x_pajaro2+self.b_pajaro2

        obs = self._get_obs()     
        return obs, {}

    def _get_obs(self):
        """
        Calculates the observations

        Returns:
            np.ndarray: The normalized observations:
            - angle_to_up : angle between the drone and the up vector (to observe gravity)
            - velocity : velocity of the drone
            - angle_velocity : angle of the velocity vector
            - distance_to_target : distance to the target
            - angle_to_target : angle between the drone and the target
            - angle_target_and_velocity : angle between the to_target vector and the velocity vector
            - distance_to_target : distance to the target (HERE TWICE BY MISTAKE)
        """
        angle_to_up = self.a / 180 * pi
        velocity = sqrt(self.xd**2 + self.yd**2)
        angle_velocity = self.ad
        #Target
        distance_to_target = (
            sqrt((self.xt - self.x) ** 2 + (self.yt - self.y) ** 2) / 500
        )
        angle_to_target = np.arctan2(self.yt - self.y, self.xt - self.x)
        # Angle between the to_target vector and the velocity vector
        angle_target_and_velocity = np.arctan2(
            self.yt - self.y, self.xt - self.x
        ) - np.arctan2(self.yd, self.xd)
        distance_to_target = (
            sqrt((self.xt - self.x) ** 2 + (self.yt - self.y) ** 2) / 500
        )
        # Obstacles
        # 1
        distance_to_obstacle1 = (sqrt((self.x_pajaro - self.x) ** 2 + (self.y_pajaro - self.y) ** 2) / 500)
        angle_to_obstacle1 = np.arctan2(self.y_pajaro - self.y, self.x_pajaro - self.x)
        angle_obstacle1_and_velocity = np.arctan2(self.y_pajaro - self.y, self.x_pajaro - self.x) - np.arctan2(self.yd, self.xd)
        distance_to_obstacle1 = (sqrt((self.x_pajaro - self.x) ** 2 + (self.y_pajaro - self.y) ** 2) / 500)
        # 2
        distance_to_obstacle2 = (sqrt((self.x_pajaro2 - self.x) ** 2 + (self.y_pajaro2 - self.y) ** 2) / 500)
        angle_to_obstacle2 = np.arctan2(self.y_pajaro2 - self.y, self.x_pajaro2 - self.x)
        angle_obstacle2_and_velocity = np.arctan2(self.y_pajaro2 - self.y, self.x_pajaro2 - self.x) - np.arctan2(self.yd, self.xd)
        distance_to_obstacle2 = (sqrt((self.x_pajaro2 - self.x) ** 2 + (self.y_pajaro2 - self.y) ** 2) / 500)
        return np.array(
            [
                angle_to_up,
                velocity,
                angle_velocity,
                distance_to_target,
                angle_to_target,
                angle_target_and_velocity,
                distance_to_obstacle1,
                angle_to_obstacle1,
                angle_obstacle1_and_velocity,
                distance_to_obstacle2,
                angle_to_obstacle2,
                angle_obstacle2_and_velocity,

            ]
        ).astype(np.float32)

    def step(self, action):
        # Game loop
        self.reward = 0.0
        (action0, action1) = (action[0], action[1])

        # Act every 5 frames
        for _ in range(5):
            self.time += 1 / 60

            if self.mouse_target is True:
                self.xt, self.yt = pygame.mouse.get_pos()
            
            #Obstaculo
            # 1
            self.x_pajaro += 1
            self.y_pajaro = self.a_pajaro*self.x_pajaro+self.b_pajaro

            if self.x_pajaro < 0 or self.HEIGHT< self.x_pajaro or self.y_pajaro < 0 or self.WIDTH < self.y_pajaro:
                self.a_pajaro, self.b_pajaro = randrange(-3, 3),randrange(0, 800)
                self.x_pajaro = 0
                self.y_pajaro = self.a_pajaro*self.x_pajaro+self.b_pajaro
            
                        
            # 2
            self.x_pajaro2 -= 1
            self.y_pajaro2 = self.a_pajaro2*self.x_pajaro2+self.b_pajaro2

            if self.x_pajaro2 < 0 or self.HEIGHT < self.x_pajaro2 or self.y_pajaro2 < 0 or self.WIDTH < self.y_pajaro2:
                self.a_pajaro2, self.b_pajaro2 = choice([x * 0.1 for x in range(-10, 11)]),randrange(-800, 800)
                self.x_pajaro2 = 799
                self.y_pajaro2 = self.a_pajaro2*self.x_pajaro2+self.b_pajaro2

                

                
            # Initialize accelerations
            self.xdd = 0
            self.ydd = self.gravity
            self.add = 0
            thruster_left = self.thruster_mean
            thruster_right = self.thruster_mean

            thruster_left += action0 * self.thruster_amplitude
            thruster_right += action0 * self.thruster_amplitude
            thruster_left += action1 * self.diff_amplitude
            thruster_right -= action1 * self.diff_amplitude

            # Calculating accelerations with Newton's laws of motions
            self.xdd += (
                -(thruster_left + thruster_right) * sin(self.a * pi / 180) / self.mass
            )
            self.ydd += (
                -(thruster_left + thruster_right) * cos(self.a * pi / 180) / self.mass
            )
            self.add += self.arm * (thruster_right - thruster_left) / self.mass

            self.xd += self.xdd
            self.yd += self.ydd
            self.ad += self.add
            self.x += self.xd
            self.y += self.yd
            self.a += self.ad

            dist = sqrt((self.x - self.xt) ** 2 + (self.y - self.yt) ** 2)

            #distancia a obstaculos
            dist_obst1 = sqrt((self.x - self.x_pajaro) ** 2 + (self.y - self.y_pajaro) ** 2)
            dist_obst2 = sqrt((self.x - self.x_pajaro2) ** 2 + (self.y - self.y_pajaro2) ** 2)
            # Reward per step survived
            self.reward += 1 / 60
            # Penalty according to the distance to target
            self.reward -= dist / (100 * 60)

            if dist < 50:
                # Reward if close to target
                self.xt = randrange(200, 600)
                self.yt = randrange(200, 600)
                self.reward += 100
                self.target_counter += 1

            # If out of time
            if self.time > self.time_limit:
                done = True
                break

            # If too far from target (crash)
            elif dist > 1000 or self.x_position < 0 or self.y_position < 0 or self.x_position > self.WIDTH or self.y_position > self.HEIGHT:
                self.reward -= 1000
                done = True
                break

            elif dist_obst1<60 or dist_obst2<60:
                self.reward -= 150 #500
                done = True
                break               

            else:
                done = False        

            if self.render_every_frame is True:
                self.render("yes")



        reward = self.reward
        obs = self._get_obs()
        info = {}
        self.episode_rewards += self.reward
        if done:
            info["episode"] = {"r": self.episode_rewards, "l": self.target_counter}
            self.episode_rewards = 0            
        return obs, reward, done, False, info

    def render(self, mode="human"):
        pygame.event.get()
        self.screen.fill((131, 176, 181))  # Fondo

        # Nubes y sol
        self.x_cloud1 += self.speed_cloud1
        if self.x_cloud1 > self.WIDTH:
            self.x_cloud1 = -self.cloud1.get_width()
        self.screen.blit(self.cloud1, (self.x_cloud1, self.y_cloud1))

        self.x_cloud2 += self.speed_cloud2
        if self.x_cloud2 < -self.cloud2.get_width():
            self.x_cloud2 = self.WIDTH
        self.screen.blit(self.cloud2, (self.x_cloud2, self.y_cloud2))

        self.screen.blit(self.sun, (630, -100))

        # Target animado
        target_sprite = self.target[self.step_counter % len(self.target)]
        self.screen.blit(
            target_sprite,
            (self.xt - target_sprite.get_width() // 2, self.yt - target_sprite.get_height() // 2)
        )

        # Player animado y rotado
        player_sprite = self.player[self.step_counter % len(self.player)]
        player_copy = pygame.transform.rotate(player_sprite, self.a)
        self.screen.blit(
            player_copy,
            (self.x - player_copy.get_width() // 2, self.y - player_copy.get_height() // 2)
        )

        #obstaculo
        pajaro_escalado = pygame.transform.scale(self.pajaro, (60, 60))
        self.screen.blit(pajaro_escalado, (self.x_pajaro, self.y_pajaro))
        pajaro_escalado2 = pygame.transform.scale(self.pajaro2, (70, 70))
        self.screen.blit(pajaro_escalado2, (self.x_pajaro2, self.y_pajaro2))


        # Texto
        collected = self.myfont.render(f"Collected: {self.target_counter}", True, (255, 255, 255))
        time_text = self.myfont.render(f"Time: {int(self.time)}", True, (255, 255, 255))
        self.screen.blit(collected, (20, 20))
        self.screen.blit(time_text, (20, 50))

        pygame.display.update()
        self.FramePerSec.tick(self.FPS)
        if self.render_mode == "rgb_array":
            return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))
    def _close(self):
        pygame.quit()
