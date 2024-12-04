import gymnasium
from gymnasium import spaces
import math
import numpy as np


class DroneNet(gymnasium.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.g = 9.81  # gravity
        self.md = 1.0
        self.m = 1.0
        self.n = 5
        self.mu = self.m / self.n / self.md
        self.l = 3
        self.I = 1 / 3.0
        self.r = 0.2
        self.dt = 1e-3
        self.t = 0  # timestep
        self.t_limit = 30000
        self.Fmax = 20
        self.Mmax = 100
        self.Fdot_max = 3.0
        self.Mdot_max = 3.0
        self.z_max = 100
        self.theta_max = math.pi / 2.0  # 90 degrees
        self.sigma_max = math.pi / 2.0  # 90 degrees

        # Define boundary conditions
        self.z_initial = self.r
        self.theta_initial = 0.0
        self.sigma_initial = 0.0
        self.z_dot_initial = -1.0
        self.theta_dot_initial = -self.z_dot_initial / self.l * 2
        self.sigma_dot_initial = 0.0

        self.z_final = -2.0
        self.sigma_final = math.radians(30.0)  # 30 degrees in radians
        self.theta_final = math.atan(self.mu / (1 + self.mu) / math.tan(self.sigma_final))
        self.z_dot_final = 0.0
        self.theta_dot_final = 0.0
        self.sigma_dot_final = 0.0

        self.x_initial = np.array([
            self.z_initial,
            self.theta_initial,
            self.sigma_initial,
            self.z_dot_initial,
            self.theta_dot_initial,
            self.sigma_dot_initial,
            self.g * (1 + self.mu) / np.cos(self.sigma_initial) * self.md,
            0.0
        ])
        self.x_final = np.array([
            self.z_final,
            self.theta_final,
            self.sigma_final,
            self.z_dot_final,
            self.theta_dot_final,
            self.sigma_dot_final,
            self.g * (1 + self.mu) / np.cos(self.sigma_final) * self.md,
            0.0
        ])

        high = np.array([
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max
        ])

        self.high_action = np.array([1.0, 1.0])
        self.low_action = np.array([-1.0, -1.0])

        self.action_space = spaces.Box(self.low_action, self.high_action, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.viewer = None
        self.state = None

    def step(self, action):
        # Unpack the state
        z, theta, sigma, z_dot, theta_dot, sigma_dot, Force, Moment = self.state

        # Calculate control inputs
        alpha = Force / self.md
        M_control = Moment

        # Define the mass matrix M and vector v (forces)
        M = np.zeros((3, 3))
        M[0, 0] = 1 + self.mu
        M[0, 1] = (self.l / 2.0 - self.r * theta) * math.cos(theta)
        M[1, 0] = M[0, 1]  # Symmetric matrix
        M[1, 1] = (self.l / 2.0 - self.r * theta) ** 2
        M[2, 2] = self.I

        v = np.zeros(3)
        v[0] = alpha * math.cos(sigma) - self.g * (1 + self.mu) + theta_dot ** 2 * (
                math.sin(theta) * (self.l / 2.0 - self.r * theta) + self.r * math.cos(theta)
        )
        v[1] = (
                alpha * (self.l / 2.0 - self.r * theta) * math.cos(theta + sigma)
                - self.r * (self.l / 2.0 - self.r * theta) * theta_dot ** 2
                - z_dot * theta_dot * (
                        math.sin(theta) * (self.l / 2.0 - self.r * theta) + self.r * math.cos(theta)
                )
                - self.g * (self.l / 2.0 - self.r * theta) * math.cos(theta)
                - 2 * (self.l / 2.0 - self.r * theta) * (-self.r * theta_dot) * theta_dot
                - z_dot * math.cos(theta) * (-self.r * theta_dot)
                + z_dot * math.sin(theta) * theta_dot * (self.l / 2.0 - self.r * theta)
        )
        v[2] = M_control

        # Calculate rate of change of force and moment
        Fdot = action[0] * self.Fdot_max
        Mdot = action[1] * self.Mdot_max

        # Solve for accelerations
        ddx = np.linalg.inv(M).dot(v)
        z_ddot, theta_ddot, sigma_ddot = ddx

        # Update state variables
        z += self.dt * z_dot
        theta += self.dt * theta_dot
        sigma += self.dt * sigma_dot
        z_dot += self.dt * z_ddot
        theta_dot += self.dt * theta_ddot
        sigma_dot += self.dt * sigma_ddot
        Force += Fdot * self.dt
        Moment += Mdot * self.dt

        # Store the previous state error
        prev_state_error = getattr(self, 'prev_state_error', None)
        current_state = np.array([z, theta, sigma, z_dot, theta_dot, sigma_dot, Force, Moment])
        state_error = np.linalg.norm(current_state - self.x_final)

        # Reward function
        if prev_state_error is not None:
            # Calculate the change in state error
            error_delta = prev_state_error - state_error
            reward = 5*error_delta  # Positive if error decreases
        else:
            reward = 0.0  # No reward for the first step

        # Penalty terms
        # Penalize theta approaching theta_max (90 degrees)
        k_theta = 2.0
        n_theta = 2
        theta_penalty = k_theta * (theta / self.theta_max) ** n_theta

        # Penalize sigma being negative or exceeding sigma_max
        k_sigma = 2.0
        n_sigma = 2
        if sigma < 0:
            sigma_penalty = k_sigma * (-sigma / self.sigma_max) ** n_sigma
        elif sigma > self.sigma_max:
            sigma_penalty = k_sigma * ((sigma - self.sigma_max) / self.sigma_max) ** n_sigma
        else:
            sigma_penalty = 0.0

        # Reward for sigma within [0, sigma_max]
        k_sigma_reward = 1.0
        n_sigma_reward = 1
        if 0 <= sigma <= self.sigma_max:
            sigma_reward = k_sigma_reward * (sigma / self.sigma_max) ** n_sigma_reward
        else:
            sigma_reward = 0.0

        # Total penalty
        total_penalty = theta_penalty + sigma_penalty

        # Update reward
        reward = reward - total_penalty + sigma_reward

        # Save the current state error for the next step
        self.prev_state_error = state_error

        # Apply penalties for constraint violations
        constraint_violation = False
        if (
                z < -self.z_max
                or z > self.z_max
                or theta > self.theta_max
                or theta < 0
                or sigma < -self.sigma_max
                or sigma > self.sigma_max
                or Force < 0
                or Force > self.Fmax
                or abs(Moment) > self.Mmax
        ):
            constraint_violation = True
            reward -= 10.0  # Small penalty for violating constraints

        # Check if the agent has reached the goal
        success = False
        if state_error < 0.1:  # Adjust threshold as needed
            reward += 100.0  # Large reward for reaching the goal
            success = True

        # Update termination conditions
        terminated = constraint_violation or success
        truncated = self.t >= self.t_limit

        # Increment time step
        self.t += 1

        # Update the state
        self.state = (z, theta, sigma, z_dot, theta_dot, sigma_dot, Force, Moment)

        # Prepare observation
        obs = np.array([z, theta, sigma, z_dot, theta_dot, sigma_dot, Force, Moment], dtype=np.float32)
        info = {
            'state_error': state_error,
            'constraint_violation': constraint_violation,
            'success': success,
            'theta_penalty': theta_penalty,
            'sigma_penalty': sigma_penalty,
            'sigma_reward': sigma_reward,
            'total_penalty': total_penalty,
        }

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # Initialize the state
        self.state = self.x_initial.copy()
        self.steps_beyond_done = None
        self.t = 0  # timestep
        self.prev_state_error = None  # Reset previous state error

        # Prepare observation
        z, theta, sigma, z_dot, theta_dot, sigma_dot, Force, Moment = self.state
        obs = np.array([z, theta, sigma, z_dot, theta_dot, sigma_dot, Force, Moment], dtype=np.float32)

        return obs, {}  # Return observation and empty info dictionary

    def render(self, mode='human', close=False):
        pass
