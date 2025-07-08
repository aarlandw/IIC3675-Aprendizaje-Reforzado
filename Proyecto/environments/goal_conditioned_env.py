import pygame
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from random import randrange
from math import sin, cos, pi, sqrt
from random import uniform, choice


class GoalConditionedDroneEnvironment(gym.Env):
    """
    Goal-conditioned version of the drone environment for HER training.

    Key changes from original:
    - Observation space includes achieved_goal and desired_goal
    - Reward function is goal-conditioned
    - Info dict contains goal information for HER
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        render_mode=None,
        render_every_frame=False,
        mouse_target=False,
        penalty_system="minimal",
    ):
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

        # Reward system configuration
        self.penalty_system = penalty_system  # "none", "minimal", "medium", "full"

        # Physics constants
        self._init_physics_constants()

        # Game variables
        self._init_game_variables()

        # Load sprites
        self.load_assets()

        # Define gym spaces for HER
        self._init_her_spaces()

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
        self.a = self.ad = self.add = 0
        self.x = self.xd = self.xdd = 400
        self.y = self.yd = self.ydd = 400

        # Goal (target) - HER specific
        self.goal = np.array([400.0, 400.0])  # [x, y] position

        # Obstacles
        self._init_obstacles()

        # Game counters
        self.target_counter = 0
        self.reward = 0
        self.time = 0
        self.time_limit = 1000 if self.mouse_target else 20
        self.step_counter = 0
        self.episode_rewards = 0

        # HER specific
        self.goal_threshold = 50.0  # Distance threshold for success

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

    def _init_her_spaces(self):
        """Initialize action and observation spaces for HER"""
        # Action space remains the same
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))

        # HER requires Dict observation space
        self.observation_space = spaces.Dict(
            {
                # Core drone observations (reduced from 12 to focus on essential info)
                "observation": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
                ),
                # Current position (what we achieved)
                "achieved_goal": spaces.Box(
                    low=0, high=800, shape=(2,), dtype=np.float32
                ),
                # Target position (what we desire)
                "desired_goal": spaces.Box(
                    low=0, high=800, shape=(2,), dtype=np.float32
                ),
            }
        )

    def load_assets(self):
        """Load game assets (same as original)"""
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

        # Obstacles
        self.pajaro = pygame.image.load("assets/sprites/pajaro.png")
        self.pajaro2 = pygame.image.load("assets/sprites/pajaro2.png")

    def _sample_goal(self):
        """Sample a random goal position"""
        return np.array(
            [np.random.uniform(200, 600), np.random.uniform(200, 600)],  # x  # y
            dtype=np.float32,
        )

    def _get_achieved_goal(self):
        """Get current drone position as achieved goal"""
        return np.array([self.x, self.y], dtype=np.float32)

    def _get_core_observation(self):
        """Get core drone state observations (simplified for HER)"""
        # Focus on essential state: angle, velocities, and goal direction
        angle_to_up = self.a / 180 * pi
        velocity = sqrt(self.xd**2 + self.yd**2) / 100  # Normalized
        angle_velocity = self.ad / 10  # Normalized

        # Direction and distance to goal
        goal_direction = np.arctan2(self.goal[1] - self.y, self.goal[0] - self.x)
        goal_distance = (
            sqrt((self.goal[0] - self.x) ** 2 + (self.goal[1] - self.y) ** 2) / 500
        )

        # Velocity alignment with goal direction
        velocity_direction = np.arctan2(self.yd, self.xd) if velocity > 0.01 else 0
        goal_velocity_alignment = goal_direction - velocity_direction

        return np.array(
            [
                angle_to_up,
                velocity,
                angle_velocity,
                goal_distance,
                goal_direction,
                goal_velocity_alignment,
            ],
            dtype=np.float32,
        )

    def _get_obs(self):
        """Get HER-compatible observation"""
        return {
            "observation": self._get_core_observation(),
            "achieved_goal": self._get_achieved_goal(),
            "desired_goal": self.goal.copy(),
        }

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        HER-compatible reward function with configurable penalty systems.
        This is called both during normal training and by HER for relabeling.
        """
        distance = np.linalg.norm(achieved_goal - desired_goal)

        # Base sparse reward: +1 if close enough, 0 otherwise
        base_reward = 1.0 if distance < self.goal_threshold else 0.0

        # Apply penalties based on system configuration
        # Only for real episodes with crashes (not HER relabeling)
        penalty = 0.0
        if hasattr(info, "get") and info.get("is_crash", False):
            if self.penalty_system == "none":
                penalty = 0.0  # Pure sparse rewards
            elif self.penalty_system == "minimal":
                penalty = -0.1  # Small crash penalty
            elif self.penalty_system == "medium":
                penalty = -0.3  # Medium crash penalty
            elif self.penalty_system == "full":
                penalty = -1.0  # Full crash penalty (same magnitude as success)

        return np.array(base_reward + penalty, dtype=np.float32)

    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        # Reset drone state
        self.a = self.ad = self.add = 0
        self.x, self.xd, self.xdd = 400, 0, 0
        self.y, self.yd, self.ydd = 400, 0, 0

        # Sample new goal
        self.goal = self._sample_goal()

        # Reset game counters
        self.episode_rewards = 0
        self.target_counter = 0
        self.reward = 0
        self.time = 0

        # Reset obstacles
        self._init_obstacles()

        return self._get_obs(), {}

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

    def _check_termination_conditions(self):
        """Check if episode should terminate and return termination reason"""

        # Out of bounds
        if self.x < 0 or self.y < 0 or self.x > self.WIDTH or self.y > self.HEIGHT:
            return True, "out_of_bounds"

        # Obstacles
        # Calculate distances
        obstacle1_dist = sqrt(
            (self.x_pajaro - self.x) ** 2 + (self.y_pajaro - self.y) ** 2
        )
        obstacle2_dist = sqrt(
            (self.x_pajaro2 - self.x) ** 2 + (self.y_pajaro2 - self.y) ** 2
        )

        if obstacle1_dist < 60 or obstacle2_dist < 60:
            return True, "obstacle_collision"

        # Time limit
        if self.time > self.time_limit:
            return True, "time_limit"

        return False, None

    def step(self, action):
        """Main step function for HER environment"""
        done = False
        termination_reason = None

        # Simulate 5 physics frames per action
        for _ in range(5):
            self.time += 1 / 60

            # Handle mouse target mode
            if self.mouse_target:
                mouse_pos = pygame.mouse.get_pos()
                self.goal = np.array(mouse_pos, dtype=np.float32)

            # Update game state
            self._update_obstacles()
            self._apply_physics(action)

            # Check termination conditions
            done, termination_reason = self._check_termination_conditions()
            if done:
                break

            # Render if needed
            if self.render_every_frame:
                self.render()

        # Get observations
        obs = self._get_obs()
        achieved_goal = obs["achieved_goal"]
        desired_goal = obs["desired_goal"]

        # Determine if this is a crash/failure (for penalty system)
        is_crash = termination_reason in ["out_of_bounds", "obstacle_collision"]

        # Compute reward using HER-compatible function
        reward_info = {"is_crash": is_crash, "termination_reason": termination_reason}
        reward = self.compute_reward(achieved_goal, desired_goal, reward_info)

        # Track cumulative episode reward
        self.episode_rewards += reward

        # Track target collection for statistics
        if reward > 0:
            self.target_counter += 1
            # Sample new goal when current one is reached
            self.goal = self._sample_goal()

        # Prepare info dict for HER
        info = {
            "achieved_goal": achieved_goal,
            "desired_goal": desired_goal,
            "is_success": reward > 0,
            "is_crash": is_crash,
            "termination_reason": termination_reason,
        }

        # Add episode info when done
        if done:
            info["episode"] = {"r": self.episode_rewards, "l": self.target_counter}

        return obs, reward, done, False, info

    def render(self, mode="human"):
        """Render the environment (same as original)"""
        pygame.event.get()
        self.screen.fill((131, 176, 181))

        # Clouds and sun
        self.x_cloud1 += self.speed_cloud1
        if self.x_cloud1 > self.WIDTH:
            self.x_cloud1 = -self.cloud1.get_width()
        self.screen.blit(self.cloud1, (self.x_cloud1, self.y_cloud1))

        self.x_cloud2 += self.speed_cloud2
        if self.x_cloud2 < -self.cloud2.get_width():
            self.x_cloud2 = self.WIDTH
        self.screen.blit(self.cloud2, (self.x_cloud2, self.y_cloud2))

        self.screen.blit(self.sun, (630, -100))

        # Target (goal) animated
        target_sprite = self.target[self.step_counter % len(self.target)]
        self.screen.blit(
            target_sprite,
            (
                self.goal[0] - target_sprite.get_width() // 2,
                self.goal[1] - target_sprite.get_height() // 2,
            ),
        )

        # Drone animated and rotated
        player_sprite = self.player[self.step_counter % len(self.player)]
        player_copy = pygame.transform.rotate(player_sprite, self.a)
        self.screen.blit(
            player_copy,
            (
                self.x - player_copy.get_width() // 2,
                self.y - player_copy.get_height() // 2,
            ),
        )

        # Obstacles
        pajaro_escalado = pygame.transform.scale(self.pajaro, (60, 60))
        self.screen.blit(pajaro_escalado, (self.x_pajaro, self.y_pajaro))
        pajaro_escalado2 = pygame.transform.scale(self.pajaro2, (70, 70))
        self.screen.blit(pajaro_escalado2, (self.x_pajaro2, self.y_pajaro2))

        # Text
        collected = self.myfont.render(
            f"Targets Reached: {self.target_counter}", True, (255, 255, 255)
        )
        time_text = self.myfont.render(f"Time: {int(self.time)}", True, (255, 255, 255))
        goal_text = self.myfont.render(
            f"Goal: ({int(self.goal[0])}, {int(self.goal[1])})", True, (255, 255, 255)
        )
        self.screen.blit(collected, (20, 20))
        self.screen.blit(time_text, (20, 50))
        self.screen.blit(goal_text, (20, 80))

        pygame.display.update()
        self.FramePerSec.tick(self.FPS)
        if self.render_mode == "rgb_array":
            return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))

    def close(self):
        """Clean up pygame"""
        pygame.quit()
