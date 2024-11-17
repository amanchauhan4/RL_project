# DroneNet.py

import gymnasium as gym
from gymnasium import spaces

import numpy as np
import math
from matplotlib import pyplot as plt

class DroneNet(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(DroneNet, self).__init__()

        # Physical constants and parameters
        self.g = 9.81  # Gravity (m/s^2)
        self.md = 1.0  # Mass of each drone (kg)
        self.m = 1.0   # Mass of the falling object (kg)
        self.n = 5.0   # Number of drones
        self.mu = self.m / self.n / self.md
        self.l = 3     # Length parameter of the net (m)
        self.I = 1 / 3.0  # Moment of inertia
        self.r = 0.2       # Radius parameter
        self.dt = 1e-3     # Time step (s)
        self.t = 0         # Current timestep
        self.t_limit = 40000  # Maximum timesteps per episode
        self.Fmax = 20        # Maximum thrust force
        self.Mmax = 100       # Maximum moment
        self.z_max = 100      # Maximum altitude
        self.theta_max = math.pi / 2.0  # Maximum angle theta
        self.sigma_max = math.pi        # Maximum bank angle sigma

        # Desired final state
        self.z_final = -2.0
        self.sigma_final = math.pi / 180.0 * 30.0  # 30 degrees in radians
        self.theta_final = math.atan(self.mu / (1 + self.mu) / math.tan(self.sigma_final))
        self.z_dot_final = 0.0
        self.theta_dot_final = 0.0
        self.sigma_dot_final = 0.0

        self.x_final = np.array([
            self.z_final,
            self.theta_final,
            self.sigma_final,
            self.z_dot_final,
            self.theta_dot_final,
            self.sigma_dot_final
        ])

        # Define observation and action spaces
        high_obs = np.array([
            self.z_max,
            self.theta_max,
            self.sigma_max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max
        ])

        self.action_space = spaces.Box(
            low=np.array([0.0, -self.Mmax]),
            high=np.array([self.Fmax, self.Mmax]),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-high_obs,
            high=high_obs,
            dtype=np.float32
        )

        self.state = None
        self.viewer = None

    def step(self, action):
        z, theta, sigma, z_dot, theta_dot, sigma_dot = self.state

        # Clip actions to their respective limits
        F = np.clip(action[0], 0.0, self.Fmax)
        M_control = np.clip(action[1], -self.Mmax, self.Mmax)

        # Dynamics calculations
        M_matrix = np.array([
            [1 + self.mu, (self.l / 2.0 - self.r * theta) * math.cos(theta), 0],
            [(self.l / 2.0 - self.r * theta) * math.cos(theta), (self.l / 2.0 - self.r * theta) ** 2, 0],
            [0, 0, self.I]
        ])

        v = np.array([
            F * math.cos(sigma) - self.g * (1 + self.mu) + (theta_dot ** 2) * (math.sin(theta) * (self.l / 2.0 - self.r * theta) + self.r * math.cos(theta)),
            F * (self.l / 2.0 - self.r * theta) * math.cos(theta + sigma) - self.r * (self.l / 2.0 - self.r * theta) * theta_dot ** 2
            - z_dot * theta_dot * (math.sin(theta) * (self.l / 2.0 - self.r * theta) + self.r * math.cos(theta))
            - self.g * (self.l / 2.0 - self.r * theta) * math.cos(theta)
            - 2 * (self.l / 2.0 - self.r * theta) * (-self.r * theta_dot) * theta_dot
            - z_dot * math.cos(theta) * (-self.r * theta_dot) + z_dot * math.sin(theta) * theta_dot * (self.l / 2.0 - self.r * theta),
            M_control
        ])

        try:
            ddx = np.linalg.solve(M_matrix, v)
        except np.linalg.LinAlgError:
            ddx = np.zeros(3)  # Handle singular matrix

        z_ddot, theta_ddot, sigma_ddot = ddx

        # Update state using Euler integration
        z += self.dt * z_dot
        theta += self.dt * theta_dot
        sigma += self.dt * sigma_dot
        z_dot += self.dt * z_ddot
        theta_dot += self.dt * theta_ddot
        sigma_dot += self.dt * sigma_ddot

        self.state = (z, theta, sigma, z_dot, theta_dot, sigma_dot)

        # Compute individual state errors
        z_error = z - self.z_final
        theta_error = theta - self.theta_final
        sigma_error = sigma - self.sigma_final
        z_dot_error = z_dot - self.z_dot_final
        theta_dot_error = theta_dot - self.theta_dot_final
        sigma_dot_error = sigma_dot - self.sigma_dot_final

        # Assign weights to each error component
        w_z = 1.0
        w_theta = 1.0
        w_sigma = 1.0
        w_z_dot = 0.1
        w_theta_dot = 0.1
        w_sigma_dot = 0.1

        # Compute the weighted sum of absolute errors
        state_error = (
                w_z * abs(z_error) +
                w_theta * abs(theta_error) +
                w_sigma * abs(sigma_error) +
                w_z_dot * abs(z_dot_error) +
                w_theta_dot * abs(theta_dot_error) +
                w_sigma_dot * abs(sigma_dot_error)
        )

        # Compute progress
        if self.prev_state_error is not None:
            progress = self.prev_state_error - state_error
        else:
            progress = 0.0  # No progress at the first step

        # Update the previous state error
        self.prev_state_error = state_error

        # Penalty for large actions
        action_penalty = 0.01 * (abs(F) + abs(M_control))

        # Compute reward
        reward = -state_error + 10.0 * progress - action_penalty

        # Initialize termination flags
        terminated = False
        truncated = False

        # Check for out-of-bounds conditions
        if z < -self.z_max or z > self.z_max or theta > self.theta_max or theta < 0 or sigma < -self.sigma_max or sigma > self.sigma_max:
            terminated = True
            reward -= 1000.0  # Penalty for going out of bounds

        # Increment timestep
        self.t += 1
        if self.t >= self.t_limit:
            truncated = True

        # Adjusted success condition
        success_threshold = 1e-1
        if state_error < success_threshold:
            reward += 5000.0  # Reward for reaching goal
            terminated = True

        obs = np.array([z, theta, sigma, z_dot, theta_dot, sigma_dot])

        return obs, reward, terminated, truncated, {'state_error': state_error}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize state near starting position with small random noise
        self.state = np.random.normal(
            loc=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            scale=np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
        )
        self.t = 0
        obs = np.array(self.state)
        self.prev_state_error = None  # Reset previous state error
        return obs, {}

    def render(self, mode='human'):
        z, theta, sigma, z_dot, theta_dot, sigma_dot = self.state

        if self.viewer is None:
            plt.ion()
            self.fig, self.ax = plt.subplots()
            self.ax.set_xlim(-5, 5)
            self.ax.set_ylim(-5, 5)
            self.ax.set_xlabel('X Position')
            self.ax.set_ylabel('Y Position')
            self.ax.set_title('DroneNet Simulation')
            self.line, = self.ax.plot([], [], 'b-', linewidth=2)  # Net
            self.object_point, = self.ax.plot([], [], 'ro', markersize=8)  # Object
            self.time_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes)
            self.viewer = True

        # Compute net endpoints based on theta
        x1 = -self.l / 2 * math.cos(theta)
        y1 = z + self.l / 2 * math.sin(theta)
        x2 = self.l / 2 * math.cos(theta)
        y2 = z + self.l / 2 * math.sin(theta)

        self.line.set_data([x1, x2], [y1, y2])
        self.object_point.set_data([0], [z])
        self.time_text.set_text(f'Time: {self.t * self.dt:.2f}s')

        plt.draw()
        plt.pause(0.05)

    def close(self):
        if self.viewer:
            plt.ioff()
            plt.show()
            plt.close(self.fig)
            self.viewer = None
