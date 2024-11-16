# DroneNet.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform


class DroneNet(gym.Env):
    """
    Custom Gymnasium environment representing a system of quadrotors connected via an elastic net,
    aiming to capture a falling object and bring it to rest in minimum time while avoiding collisions.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, n_drones=5):
        # Physical constants and parameters
        self.g = 9.81          # Gravity acceleration (m/s^2)
        self.md = 1.0          # Mass of each drone (kg)
        self.m = 75.0          # Mass of the falling object (kg), e.g., a person
        self.n = n_drones      # Number of drones
        self.l = 10.0          # Initial length parameter of the net (m)
        self.k_net = 5000.0    # Elastic constant of the net
        self.c_net = 100.0     # Damping coefficient of the net
        self.dt = 1e-2         # Time step for simulation (s)
        self.t_limit = 5000    # Maximum number of time steps per episode

        # Control input constraints
        self.Fmax = 500.0      # Maximum thrust force (N)
        self.phi_max = np.pi / 6   # Max roll angle (30 degrees)
        self.theta_max = np.pi / 6 # Max pitch angle (30 degrees)

        # State variable constraints
        self.z_max = 100.0                    # Maximum altitude (m)
        self.x_range = 50.0                   # Maximum horizontal range (m)
        self.safe_distance = 0.5              # Minimum safe distance between drones (m)

        # Max net force
        self.max_net_force = 10000.0          # Increased to handle stiffer net and heavier object

        # Desired final state
        self.z_final = 20.0                   # Desired final altitude (m) after capture
        self.z_dot_final = 0.0                # Desired final altitude velocity (m/s)

        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([0.0, -self.phi_max, -self.theta_max] * self.n),
            high=np.array([self.Fmax, self.phi_max, self.theta_max] * self.n),
            dtype=np.float32
        )

        high = np.full((self.n * 6 + 6), np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # Initialize state and time step
        self.state = None
        self.t = 0

        # For rendering
        self.fig = None
        self.ax = None
        self.lines = []

    def step(self, action):
        # Unpack the state variables
        drone_states = self.state['drones']
        object_state = self.state['object']

        # Update drones' states based on actions
        for i, drone in enumerate(drone_states):
            F = action[i * 3]
            phi = action[i * 3 + 1]
            theta = action[i * 3 + 2]

            # Clip the actions to the allowed ranges
            F = np.clip(F, 0.0, self.Fmax)
            phi = np.clip(phi, -self.phi_max, self.phi_max)
            theta = np.clip(theta, -self.theta_max, self.theta_max)

            # Compute acceleration
            ax = F * np.sin(theta) / self.md
            ay = -F * np.sin(phi) * np.cos(theta) / self.md
            az = F * np.cos(phi) * np.cos(theta) / self.md - self.g

            # Update velocities
            drone['vx'] += ax * self.dt
            drone['vy'] += ay * self.dt
            drone['vz'] += az * self.dt

            # Update positions
            drone['x'] += drone['vx'] * self.dt
            drone['y'] += drone['vy'] * self.dt
            drone['z'] += drone['vz'] * self.dt

        # Initialize net forces
        for drone in drone_states:
            drone['fx'] = 0.0
            drone['fy'] = 0.0
            drone['fz'] = 0.0

        fx_net = 0.0
        fy_net = 0.0
        fz_net = 0.0

        # Compute forces between drones and object
        for drone in drone_states:
            dx = drone['x'] - object_state['x']
            dy = drone['y'] - object_state['y']
            dz = drone['z'] - object_state['z']
            distance = np.sqrt(dx**2 + dy**2 + dz**2)
            rest_length = self.l / 2.0

            # Compute the relative velocities
            dvx = drone['vx'] - object_state['vx']
            dvy = drone['vy'] - object_state['vy']
            dvz = drone['vz'] - object_state['vz']

            # Elastic force magnitude
            f_elastic = -self.k_net * (distance - rest_length)
            # Damping force magnitude
            relative_velocity = (dx * dvx + dy * dvy + dz * dvz) / (distance + 1e-6)
            f_damping = -self.c_net * relative_velocity

            # Total force magnitude
            f_total = f_elastic + f_damping
            f_total = np.clip(f_total, -self.max_net_force, self.max_net_force)

            # Compute force components
            if distance != 0:
                fx = f_total * dx / distance
                fy = f_total * dy / distance
                fz = f_total * dz / distance
            else:
                fx = fy = fz = 0.0

            # Apply forces to object
            fx_net += fx
            fy_net += fy
            fz_net += fz

            # Apply equal and opposite forces to drones
            drone['fx'] += -fx
            drone['fy'] += -fy
            drone['fz'] += -fz

        # Compute forces between drones (net connections)
        positions = np.array([[d['x'], d['y'], d['z']] for d in drone_states])
        velocities = np.array([[d['vx'], d['vy'], d['vz']] for d in drone_states])
        distances_matrix = squareform(pdist(positions))
        for i in range(self.n):
            for j in range(i + 1, self.n):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dz = positions[i, 2] - positions[j, 2]
                distance = distances_matrix[i, j]
                rest_length = self.l / self.n

                # Relative velocities
                dvx = velocities[i, 0] - velocities[j, 0]
                dvy = velocities[i, 1] - velocities[j, 1]
                dvz = velocities[i, 2] - velocities[j, 2]

                # Elastic force magnitude
                f_elastic = -self.k_net * (distance - rest_length)
                # Damping force magnitude
                relative_velocity = (dx * dvx + dy * dvy + dz * dvz) / (distance + 1e-6)
                f_damping = -self.c_net * relative_velocity

                # Total force magnitude
                f_total = f_elastic + f_damping
                f_total = np.clip(f_total, -self.max_net_force, self.max_net_force)

                # Compute force components
                if distance != 0:
                    fx = f_total * dx / distance
                    fy = f_total * dy / distance
                    fz = f_total * dz / distance
                else:
                    fx = fy = fz = 0.0

                # Apply forces to drones
                drone_states[i]['fx'] += -fx
                drone_states[i]['fy'] += -fy
                drone_states[i]['fz'] += -fz

                drone_states[j]['fx'] += fx
                drone_states[j]['fy'] += fy
                drone_states[j]['fz'] += fz

        # Update drones' velocities and positions due to net forces
        for drone in drone_states:
            ax_net = drone['fx'] / self.md
            ay_net = drone['fy'] / self.md
            az_net = drone['fz'] / self.md

            # Update velocities
            drone['vx'] += ax_net * self.dt
            drone['vy'] += ay_net * self.dt
            drone['vz'] += az_net * self.dt

            # Net forces have been applied, reset them
            drone['fx'] = 0.0
            drone['fy'] = 0.0
            drone['fz'] = 0.0

        # Total acceleration of object
        ax_obj = fx_net / self.m
        ay_obj = fy_net / self.m
        az_obj = fz_net / self.m - self.g

        # Update object's velocities
        object_state['vx'] += ax_obj * self.dt
        object_state['vy'] += ay_obj * self.dt
        object_state['vz'] += az_obj * self.dt

        # Update object's positions
        object_state['x'] += object_state['vx'] * self.dt
        object_state['y'] += object_state['vy'] * self.dt
        object_state['z'] += object_state['vz'] * self.dt

        # Collision avoidance check
        positions = np.array([[d['x'], d['y'], d['z']] for d in drone_states])
        distances = pdist(positions)
        min_distance = np.min(distances) if len(distances) > 0 else self.safe_distance

        # Calculate reward
        reward = 0.0
        terminated = False
        truncated = False

        # Penalize for drone collisions
        if min_distance < self.safe_distance:
            reward -= 500.0  # Collision penalty
            terminated = True

        # Penalize for drones going out of bounds
        for drone in drone_states:
            if (abs(drone['x']) > self.x_range or abs(drone['y']) > self.x_range or
                    drone['z'] < 0 or drone['z'] > self.z_max):
                terminated = True
                reward -= 500.0  # Out of bounds penalty
                break

        # Penalize if object goes out of bounds
        if (abs(object_state['x']) > self.x_range or abs(object_state['y']) > self.x_range or
                object_state['z'] < 0.0 or object_state['z'] > self.z_max):
            terminated = True
            reward -= 500.0  # Object out of bounds penalty

        # Reward shaping
        position_error = abs(object_state['z'] - self.z_final)
        reward -= position_error * 2.0  # Adjusted Position error penalty

        speed_error = abs(object_state['vz'] - self.z_dot_final)
        reward -= speed_error * 0.5  # Adjusted Speed error penalty

        # Encourage drones to be under the object
        horizontal_distance = 0.0
        for drone in drone_states:
            dx = drone['x'] - object_state['x']
            dy = drone['y'] - object_state['y']
            distance = np.sqrt(dx**2 + dy**2)
            horizontal_distance += distance
        reward -= horizontal_distance * 0.05  # Adjusted Distance penalty

        # Time penalty to encourage faster completion
        reward -= 0.05

        # Provide positive reward for lifting the object towards the desired altitude
        if object_state['z'] <= self.z_final and object_state['vz'] <= 0.0:
            reward += 50.0

        # Check for successful capture and lifting
        if (abs(object_state['z'] - self.z_final) < 0.5 and
                abs(object_state['vz'] - self.z_dot_final) < 0.5):
            reward += 5000.0
            terminated = True

        # Update time step and check time limit
        self.t += 1
        if self.t >= self.t_limit:
            truncated = True

        # Prepare observation
        obs = self._get_obs()

        # Prepare info with parameters for logging
        info = {
            'min_distance': min_distance,
            'reward': reward
        }

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0

        # Initialize drones' states
        drone_states = []
        angle_increment = 2 * np.pi / self.n
        radius = self.l / (2 * np.pi)
        for i in range(self.n):
            angle = i * angle_increment
            drone_states.append({
                'x': radius * np.cos(angle),
                'y': radius * np.sin(angle),
                'z': self.z_final,
                'vx': 0.0,
                'vy': 0.0,
                'vz': 0.0,
                'fx': 0.0,
                'fy': 0.0,
                'fz': 0.0
            })

        # Initialize object's state
        object_state = {
            'x': 0.0,
            'y': 0.0,
            'z': self.z_final + 30.0,  # Start higher to give the drones time to react
            'vx': 0.0,
            'vy': 0.0,
            'vz': -5.0  # Initial downward velocity
        }

        self.state = {'drones': drone_states, 'object': object_state}

        return self._get_obs(), {}

    def _get_obs(self):
        obs = []
        for drone in self.state['drones']:
            obs.extend([drone['x'], drone['y'], drone['z'],
                        drone['vx'], drone['vy'], drone['vz']])
        obj = self.state['object']
        obs.extend([obj['x'], obj['y'], obj['z'],
                    obj['vx'], obj['vy'], obj['vz']])
        return np.array(obs, dtype=np.float32)

    def render(self, mode='human', slow_motion=False):
        if self.fig is None:
            plt.ion()
            self.fig = plt.figure(figsize=(8, 6))
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.set_xlim(-self.x_range, self.x_range)
            self.ax.set_ylim(-self.x_range, self.x_range)
            self.ax.set_zlim(0, self.z_max)
            self.ax.set_xlabel('X Position')
            self.ax.set_ylabel('Y Position')
            self.ax.set_zlabel('Altitude (Z)')
            self.ax.set_title('DroneNet Simulation')

            # Initialize lines for drones and net
            for _ in range(self.n):
                line, = self.ax.plot([], [], [], 'bo')
                self.lines.append(line)
            self.obj_line, = self.ax.plot([], [], [], 'ro', markersize=8)

            # Lines for the net between drones
            self.net_lines = []
            for _ in range(self.n):
                net_line, = self.ax.plot([], [], [], 'k-')
                self.net_lines.append(net_line)

            # Lines from drones to object
            self.object_lines = []
            for _ in range(self.n):
                obj_line, = self.ax.plot([], [], [], 'r--')
                self.object_lines.append(obj_line)

        # Update drone positions
        for i, drone in enumerate(self.state['drones']):
            self.lines[i].set_data([drone['x']], [drone['y']])
            self.lines[i].set_3d_properties([drone['z']])

        # Update object position
        obj = self.state['object']
        self.obj_line.set_data([obj['x']], [obj['y']])
        self.obj_line.set_3d_properties([obj['z']])

        # Update net lines between drones
        for i in range(self.n):
            j = (i + 1) % self.n
            xs = [self.state['drones'][i]['x'], self.state['drones'][j]['x']]
            ys = [self.state['drones'][i]['y'], self.state['drones'][j]['y']]
            zs = [self.state['drones'][i]['z'], self.state['drones'][j]['z']]
            self.net_lines[i].set_data(xs, ys)
            self.net_lines[i].set_3d_properties(zs)

        # Update lines from drones to object
        for i, drone in enumerate(self.state['drones']):
            xs = [drone['x'], obj['x']]
            ys = [drone['y'], obj['y']]
            zs = [drone['z'], obj['z']]
            self.object_lines[i].set_data(xs, ys)
            self.object_lines[i].set_3d_properties(zs)

        plt.draw()
        plt.pause(0.05 if slow_motion else 0.001)

    def close(self):
        pass  # Keep the window open
