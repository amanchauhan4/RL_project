import gymnasium
from gymnasium import spaces
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

class DroneNet(gymnasium.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, render_mode=None):
        super().__init__()
        self.g = 9.81
        self.md = 1.0
        self.m = 1.0
        self.n = 5.0
        self.mu = self.m/self.n/self.md
        self.l = 6.0
        self.I = 1.0
        self.r = 0.2
        self.t = 0
        self.dt = 0.05
        self.t_limit = 3000
        self.Fmax = 20.0
        self.Mmax = 10.0

        # initial conditions
        self.z_initial = self.r
        self.theta_initial = 0.0
        self.sigma_initial = 0.0
        self.z_dot_initial = -0.5
        self.theta_dot_initial = -(self.z_dot_initial/self.l)*2.0
        self.sigma_dot_initial = 0.0

        # Desired final state
        self.z_final = -2.0
        self.sigma_final = math.radians(30.0)
        self.theta_final = math.atan(self.mu/(1+self.mu)/math.tan(self.sigma_final))
        self.z_dot_final = 0.0
        self.theta_dot_final = 0.0
        self.sigma_dot_final = 0.0
        self.x_final = np.array([self.z_final, self.theta_final, self.sigma_final,
                                 self.z_dot_final, self.theta_dot_final, self.sigma_dot_final])

        high = np.full((6,), 1000.0, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.state = None
        self.prev_action = np.zeros(2)
        self.render_mode = render_mode
        self.fig = None
        self.ax = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = (self.z_initial, self.theta_initial, self.sigma_initial,
                      self.z_dot_initial, self.theta_dot_initial, self.sigma_dot_initial)
        self.t = 0
        self.prev_action = np.zeros(2)
        if self.render_mode == 'human':
            self._init_render()
        obs = np.array(self.state, dtype=np.float32)
        return obs, {}

    def step(self, action):
        thrust = (action[0]*0.5 + 0.5)*self.Fmax
        M_control = action[1]*self.Mmax

        z, theta, sigma, z_dot, theta_dot, sigma_dot = self.state

        # Mass matrix and forces (no simplifications here)
        M_mat = np.zeros((3,3))
        M_mat[0,0] = 1+self.mu
        M_mat[0,1] = (self.l/2.0 - self.r*theta)*math.cos(theta)
        M_mat[1,0] = M_mat[0,1]
        M_mat[1,1] = (self.l/2.0 - self.r*theta)**2
        M_mat[2,2] = self.I
        M_mat[0,2] = 0.0
        M_mat[2,0] = 0.0

        alpha = thrust / self.md
        f0 = alpha*math.cos(sigma) - self.g*(1+self.mu) + theta_dot**2 * (math.sin(theta)*(self.l/2.0 - self.r*theta) + self.r*math.cos(theta))

        # The original f1 is unchanged
        f1 = alpha*(self.l/2.0-self.r*theta)*math.cos(theta+sigma)-self.r*(self.l/2.0-self.r*theta)*theta_dot**2 - z_dot*theta_dot*(math.sin(theta)*(self.l/2.0-self.r*theta)+self.r*math.cos(theta)) \
             - self.g*(self.l/2.0-self.r*theta)*math.cos(theta)-2*(self.l/2.0-self.r*theta)*(-self.r*theta_dot)*theta_dot - z_dot*math.cos(theta)*(-self.r*theta_dot) + z_dot*math.sin(theta)*theta_dot*(self.l/2.0-self.r*theta)

        f2 = M_control

        try:
            ddq = np.linalg.solve(M_mat, np.array([f0, f1, f2]))
        except np.linalg.LinAlgError:
            reward = -1000.0
            return np.array(self.state, dtype=np.float32), reward, True, True, {}

        z_ddot, theta_ddot, sigma_ddot = ddq

        # Update states
        z += self.dt*z_dot
        theta += self.dt*theta_dot
        sigma += self.dt*sigma_dot
        z_dot += self.dt*z_ddot
        theta_dot += self.dt*theta_ddot
        sigma_dot += self.dt*sigma_ddot

        self.state = (z, theta, sigma, z_dot, theta_dot, sigma_dot)
        self.t += 1

        # Compute error and reward
        error = np.array([
            z - self.z_final,
            theta - self.theta_final,
            sigma - self.sigma_final,
            z_dot - self.z_dot_final,
            theta_dot - self.theta_dot_final,
            sigma_dot - self.sigma_dot_final
        ])
        dist = np.linalg.norm(error)

        action_change_penalty = 0.01 * np.sum((action - self.prev_action)**2)
        self.prev_action = action.copy()

        angle_error = abs(theta - self.theta_final) + abs(sigma - self.sigma_final)
        reward = -dist - 0.5*angle_error - action_change_penalty

        terminated = False
        truncated = (self.t >= self.t_limit)

        if dist < 0.5:
            reward += 1000.0
            terminated = True

        if abs(z) > 100 or abs(theta) > math.pi/2 or abs(sigma) > math.pi:
            reward -= 500.0
            terminated = True

        if self.render_mode == 'human':
            self.render()

        obs = np.array(self.state, dtype=np.float32)
        return obs, reward, terminated, truncated, {}

    def _init_render(self):
        if self.fig is None or self.ax is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(5,5))
        self.ax.clear()
        self.ax.set_xlim(-5,5)
        self.ax.set_ylim(-3,1)
        self.ax.set_title('DroneNet Simulation')
        self.ax.set_xlabel('Horizontal (m)')
        self.ax.set_ylabel('Vertical (m)')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def render(self):
        if self.fig is None or self.ax is None:
            self._init_render()
        self.ax.clear()
        self.ax.set_xlim(-5,5)
        self.ax.set_ylim(-3,1)
        self.ax.set_title('DroneNet Simulation')
        self.ax.set_xlabel('Horizontal (m)')
        self.ax.set_ylabel('Vertical (m)')

        z, theta, sigma, z_dot, theta_dot, sigma_dot = self.state

        half_length = self.l/2.0
        x1 = -half_length * math.cos(theta)
        y1 = -half_length * math.sin(theta) + z
        x2 = half_length * math.cos(theta)
        y2 = half_length * math.sin(theta) + z

        drone_positions = []
        for i in range(int(self.n)):
            frac = i/(self.n-1)
            xd = x1 + frac*(x2 - x1)
            yd = y1 + frac*(y2 - y1)
            drone_positions.append((xd, yd))

        obj_x = 0.0
        obj_y = z

        net_line = Line2D([x1, x2], [y1, y2], color='black', linewidth=2)
        self.ax.add_line(net_line)

        for (dx, dy) in drone_positions:
            drone_circle = Circle((dx, dy), radius=0.1, color='blue')
            self.ax.add_patch(drone_circle)

        obj_circle = Circle((obj_x, obj_y), radius=self.r, color='red')
        self.ax.add_patch(obj_circle)

        self.ax.text(-4.5, 0.9, f'Sigma: {math.degrees(sigma):.1f} deg', fontsize=10)
        self.ax.text(-4.5, 0.7, f'Theta: {math.degrees(theta):.1f} deg', fontsize=10)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        if self.fig:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
