import pygame
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from random import randrange
from math import sin, cos, pi, sqrt
from random import uniform, choice


class DroneEnvironment(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, render_every_frame=False, mouse_target=False):
        super().__init__()
        pygame.init()

        # Display settings
        self.WIDTH, self.HEIGHT = 800, 800
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.FPS = 60
        self.FramePerSec = pygame.time.Clock()
        self.myfont = pygame.font.Font("assets/fonts/Roboto-Regular.ttf", 30)

        # Mode settings
        self.render_every_frame = render_every_frame
        self.mouse_target = mouse_target
        self.render_mode = render_mode

        # Physics constants
        self._init_physics_constants()

        # Game variables
        self._init_game_variables()

        # Load sprites
        self.load_assets()

        # Define gym spaces
        self._init_gym_spaces()

        # Reset to initial state
        self.reset()

    def _init_physics_constants(self):
        """Initialize physics constants"""
        self.gravity = 0.08
        self.thruster_amplitude = 0.04
        self.diff_amplitude = 0.003
        self.thruster_mean = 0.04
        self.mass = 1
        self.arm = 25

    def _init_game_variables(self):
        """Initialize game state variables"""
        # Drone state
        self.a = self.ad = self.add = 0  # angle, angular velocity, angular acceleration
        self.x = self.xd = self.xdd = 400  # x position, velocity, acceleration
        self.y = self.yd = self.ydd = 400  # y position, velocity, acceleration

        # Target
        self.xt = randrange(200, 600)
        self.yt = randrange(200, 600)

        # Obstacles
        self._init_obstacles()

        # Game counters
        self.target_counter = 0
        self.reward = 0
        self.time = 0
        self.time_limit = 1000 if self.mouse_target else 20
        self.step_counter = 0
        self.episode_rewards = 0

    def _init_obstacles(self):
        """Initialize obstacle positions and movement"""
        # Obstacle 1 (moves right)
        self.a_pajaro, self.b_pajaro = randrange(-3, 3), randrange(0, 800)
        self.x_pajaro = 0
        self.y_pajaro = self.a_pajaro * self.x_pajaro + self.b_pajaro

        # Obstacle 2 (moves left)
        self.a_pajaro2, self.b_pajaro2 = choice(
            [x * 0.1 for x in range(-20, 21)]
        ), randrange(-800, 800)
        self.x_pajaro2 = 799
        self.y_pajaro2 = self.a_pajaro2 * self.x_pajaro2 + self.b_pajaro2

    def _init_gym_spaces(self):
        """Initialize action and observation spaces"""
        # 2D continuous actions: [thrust_amplitude, thrust_difference]
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))

        # 12D observations: drone state + target info + obstacle info
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )

    def load_assets(self):
        # Player sprites
        self.player_width = 80
        self.player = []
        for i in range(1, 5):
            img = pygame.image.load(
                f"assets/balloon-flat-asset-pack/png/objects/drone-sprites/drone-{i}.png"
            )
            img = pygame.transform.scale(
                img, (self.player_width, int(self.player_width * 0.3))
            )
            self.player.append(img)

        # Target sprites
        self.target_width = 30
        self.target = []
        for i in range(1, 8):
            img = pygame.image.load(
                f"assets/balloon-flat-asset-pack/png/balloon-sprites/red-plain/red-plain-{i}.png"
            )
            img = pygame.transform.scale(
                img, (self.target_width, int(self.target_width * 1.73))
            )
            self.target.append(img)

        # Background elements
        self.cloud1 = pygame.image.load(
            "assets/balloon-flat-asset-pack/png/background-elements/cloud-1.png"
        )
        self.cloud2 = pygame.image.load(
            "assets/balloon-flat-asset-pack/png/background-elements/cloud-2.png"
        )
        self.sun = pygame.image.load(
            "assets/balloon-flat-asset-pack/png/background-elements/sun.png"
        )

        self.cloud1.set_alpha(124)
        self.cloud2.set_alpha(124)
        self.sun.set_alpha(124)

        self.x_cloud1, self.y_cloud1, self.speed_cloud1 = 150, 200, 0.3
        self.x_cloud2, self.y_cloud2, self.speed_cloud2 = 400, 500, -0.2

        # Obstaculo
        self.pajaro = pygame.image.load("assets/sprites/pajaro.png")
        self.pajaro2 = pygame.image.load("assets/sprites/pajaro2.png")

    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        # Reset drone state
        self.a = self.ad = self.add = 0
        self.x, self.xd, self.xdd = 400, 0, 0
        self.y, self.yd, self.ydd = 400, 0, 0

        # Reset target
        self.xt = randrange(200, 600)
        self.yt = randrange(200, 600)

        # Reset game counters
        self.episode_rewards = 0
        self.target_counter = 0
        self.reward = 0
        self.time = 0

        # Reset obstacles
        self._init_obstacles()

        return self._get_obs(), {}

    def _calculate_distances(self):
        """Calculate distances to target and obstacles"""
        # Distance to target
        target_dist = sqrt((self.xt - self.x) ** 2 + (self.yt - self.y) ** 2)

        # Distance to obstacles
        obstacle1_dist = sqrt(
            (self.x_pajaro - self.x) ** 2 + (self.y_pajaro - self.y) ** 2
        )
        obstacle2_dist = sqrt(
            (self.x_pajaro2 - self.x) ** 2 + (self.y_pajaro2 - self.y) ** 2
        )

        return target_dist, obstacle1_dist, obstacle2_dist

    def _calculate_angles(self):
        """Calculate angles for observations"""
        # Drone orientation and velocity
        angle_to_up = self.a / 180 * pi
        velocity = sqrt(self.xd**2 + self.yd**2)
        angle_velocity = self.ad

        # Target angles
        angle_to_target = np.arctan2(self.yt - self.y, self.xt - self.x)
        angle_target_and_velocity = angle_to_target - np.arctan2(self.yd, self.xd)

        # Obstacle angles
        angle_to_obstacle1 = np.arctan2(self.y_pajaro - self.y, self.x_pajaro - self.x)
        angle_obstacle1_and_velocity = angle_to_obstacle1 - np.arctan2(self.yd, self.xd)

        angle_to_obstacle2 = np.arctan2(
            self.y_pajaro2 - self.y, self.x_pajaro2 - self.x
        )
        angle_obstacle2_and_velocity = angle_to_obstacle2 - np.arctan2(self.yd, self.xd)

        return (
            angle_to_up,
            velocity,
            angle_velocity,
            angle_to_target,
            angle_target_and_velocity,
            angle_to_obstacle1,
            angle_obstacle1_and_velocity,
            angle_to_obstacle2,
            angle_obstacle2_and_velocity,
        )

    def _get_obs(self):
        """Get normalized observations"""
        target_dist, obstacle1_dist, obstacle2_dist = self._calculate_distances()
        angles = self._calculate_angles()

        # Normalize distances
        target_dist_norm = target_dist / 500
        obstacle1_dist_norm = obstacle1_dist / 500
        obstacle2_dist_norm = obstacle2_dist / 500

        return np.array(
            [
                angles[0],  # angle_to_up
                angles[1],  # velocity
                angles[2],  # angle_velocity
                target_dist_norm,  # distance_to_target
                angles[3],  # angle_to_target
                angles[4],  # angle_target_and_velocity
                obstacle1_dist_norm,  # distance_to_obstacle1
                angles[5],  # angle_to_obstacle1
                angles[6],  # angle_obstacle1_and_velocity
                obstacle2_dist_norm,  # distance_to_obstacle2
                angles[7],  # angle_to_obstacle2
                angles[8],  # angle_obstacle2_and_velocity
            ]
        ).astype(np.float32)

    def step(self, action):
        """Main step function - simplified and clean"""
        self.reward = 0.0
        done = False

        # Simulate 5 physics frames per action
        for _ in range(5):
            self.time += 1 / 60

            # Handle mouse target mode
            if self.mouse_target:
                self.xt, self.yt = pygame.mouse.get_pos()

            # Update game state
            self._update_obstacles()
            self._apply_physics(action)

            # Calculate distances
            target_dist, obstacle1_dist, obstacle2_dist = self._calculate_distances()

            # Check for target collection
            self._check_target_reached(target_dist)

            # Check termination conditions
            if self._check_termination_conditions(
                target_dist, obstacle1_dist, obstacle2_dist
            ):
                done = True
                break

            # Render if needed
            if self.render_every_frame:
                self.render("yes")

        # Prepare return values
        obs = self._get_obs()
        info = {}
        self.episode_rewards += self.reward

        if done:
            info["episode"] = {"r": self.episode_rewards, "l": self.target_counter}
            self.episode_rewards = 0

        return obs, self.reward, done, False, info

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
            (
                self.xt - target_sprite.get_width() // 2,
                self.yt - target_sprite.get_height() // 2,
            ),
        )

        # Player animado y rotado
        player_sprite = self.player[self.step_counter % len(self.player)]
        player_copy = pygame.transform.rotate(player_sprite, self.a)
        self.screen.blit(
            player_copy,
            (
                self.x - player_copy.get_width() // 2,
                self.y - player_copy.get_height() // 2,
            ),
        )

        # obstaculo
        pajaro_escalado = pygame.transform.scale(self.pajaro, (60, 60))
        self.screen.blit(pajaro_escalado, (self.x_pajaro, self.y_pajaro))
        pajaro_escalado2 = pygame.transform.scale(self.pajaro2, (70, 70))
        self.screen.blit(pajaro_escalado2, (self.x_pajaro2, self.y_pajaro2))

        # Texto
        collected = self.myfont.render(
            f"Collected: {self.target_counter}", True, (255, 255, 255)
        )
        time_text = self.myfont.render(f"Time: {int(self.time)}", True, (255, 255, 255))
        self.screen.blit(collected, (20, 20))
        self.screen.blit(time_text, (20, 50))

        pygame.display.update()
        self.FramePerSec.tick(self.FPS)
        if self.render_mode == "rgb_array":
            return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))

    def _close(self):
        pygame.quit()

    def _apply_physics(self, action):
        """Apply physics simulation for one frame"""
        action0, action1 = action[0], action[1]

        # Calculate thruster forces
        thruster_left = (
            self.thruster_mean
            + action0 * self.thruster_amplitude
            + action1 * self.diff_amplitude
        )
        thruster_right = (
            self.thruster_mean
            + action0 * self.thruster_amplitude
            - action1 * self.diff_amplitude
        )

        # Calculate accelerations using Newton's laws
        thrust_total = thruster_left + thruster_right
        self.xdd = -(thrust_total) * sin(self.a * pi / 180) / self.mass
        self.ydd = self.gravity - (thrust_total) * cos(self.a * pi / 180) / self.mass
        self.add = self.arm * (thruster_right - thruster_left) / self.mass

        # Update velocities and positions
        self.xd += self.xdd
        self.yd += self.ydd
        self.ad += self.add
        self.x += self.xd
        self.y += self.yd
        self.a += self.ad

    def _update_obstacles(self):
        """Update obstacle positions"""
        # Obstacle 1 (moves right)
        self.x_pajaro += 1
        self.y_pajaro = self.a_pajaro * self.x_pajaro + self.b_pajaro

        # Reset if out of bounds
        if (
            self.x_pajaro < 0
            or self.x_pajaro > self.HEIGHT
            or self.y_pajaro < 0
            or self.y_pajaro > self.WIDTH
        ):
            self.a_pajaro, self.b_pajaro = randrange(-3, 3), randrange(0, 800)
            self.x_pajaro = 0
            self.y_pajaro = self.a_pajaro * self.x_pajaro + self.b_pajaro

        # Obstacle 2 (moves left)
        self.x_pajaro2 -= 1
        self.y_pajaro2 = self.a_pajaro2 * self.x_pajaro2 + self.b_pajaro2

        # Reset if out of bounds
        if (
            self.x_pajaro2 < 0
            or self.x_pajaro2 > self.HEIGHT
            or self.y_pajaro2 < 0
            or self.y_pajaro2 > self.WIDTH
        ):
            self.a_pajaro2, self.b_pajaro2 = choice(
                [x * 0.1 for x in range(-10, 11)]
            ), randrange(-800, 800)
            self.x_pajaro2 = 799
            self.y_pajaro2 = self.a_pajaro2 * self.x_pajaro2 + self.b_pajaro2

    def _check_target_reached(self, target_dist):
        """Check if target is reached and handle target collection"""
        if target_dist < 50:
            # Spawn new target
            self.xt = randrange(200, 600)
            self.yt = randrange(200, 600)
            self.reward += 1.0
            self.target_counter += 1
            return True
        return False

    def _check_termination_conditions(
        self, target_dist, obstacle1_dist, obstacle2_dist
    ):
        """Check if episode should terminate"""
        # Time limit
        if self.time > self.time_limit:
            return True

        # Out of bounds or too far from target
        if (
            target_dist > 1000
            or self.x < 0
            or self.y < 0
            or self.x > self.WIDTH
            or self.y > self.HEIGHT
        ):
            self.reward -= 1.0
            return True

        # Hit obstacles
        if obstacle1_dist < 60 or obstacle2_dist < 60:
            self.reward -= 1.0
            return True

        return False
