import gymnasium
from gymnasium import spaces
import math
import numpy as np

class DroneNet(gymnasium.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.g = 9.81  # gravity
        self.md = 1.0  # mass of a single drone
        self.m = 1.0   # mass of object
        self.n = 5.0   # number of drones
        self.mu = self.m/self.n/self.md  # mass ratio
        self.l = 6.0   # diameter of the net
        self.I = 1.0   # moment of inertia
        self.r = 0.2   # radius of object
        self.t = 0      # timestep
        self.dt = 0.05  # time step size
        self.t_limit = 3000  # max steps per episode
        self.Fmax = 20.0  # maximum thrust
        self.Mmax = 10.0  # maximum moment allowed

        # Initial conditions (could be made simpler for curriculum learning)
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

        # Initial state
        self.x_initial = np.array([self.z_initial, self.theta_initial, self.sigma_initial,
                                   self.z_dot_initial, self.theta_dot_initial, self.sigma_dot_initial])

        high = np.full((6,), 1000.0, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.x_initial.copy()
        self.t = 0
        # Return obs, info
        return np.array(self.state, dtype=np.float32), {}

    def step(self, action):
        # Map actions to thrust and moment
        thrust = (action[0]*0.5 + 0.5)*self.Fmax
        M_control = action[1]*self.Mmax

        z, theta, sigma, z_dot, theta_dot, sigma_dot = self.state

        # Mass matrix
        M_mat = np.zeros((3,3))
        M_mat[0,0] = 1+self.mu
        M_mat[0,1] = (self.l/2.0 - self.r*theta)*math.cos(theta)
        M_mat[1,0] = M_mat[0,1]
        M_mat[1,1] = (self.l/2.0 - self.r*theta)**2
        M_mat[2,2] = 0.5
        M_mat[0,2] = (math.sin(theta))/2.0
        M_mat[2,0] = math.sin(theta)

        alpha = thrust / self.md
        L_mid = (self.l/2.0 - self.r*theta)

        # Forces
        f0 = alpha*math.cos(sigma) - self.g*(1+self.mu) + theta_dot**2 * (math.sin(theta)*L_mid + self.r*math.cos(theta))
        f2 = M_control
        # Simpler approx for f1
        f1 = alpha * L_mid * math.cos(theta+sigma) - self.g * L_mid * math.cos(theta)

        try:
            ddq = np.linalg.solve(M_mat, np.array([f0, f1, f2]))
        except np.linalg.LinAlgError:
            # If singular, large penalty and terminate
            return np.array(self.state, dtype=np.float32), -1000.0, True, True, {}

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

        truncated = (self.t >= self.t_limit)

        # Calculate error
        error = np.array([z - self.z_final,
                          theta - self.theta_final,
                          sigma - self.sigma_final,
                          z_dot - self.z_dot_final,
                          theta_dot - self.theta_dot_final,
                          sigma_dot - self.sigma_dot_final])
        dist = np.linalg.norm(error)

        # Reward: focus more on angular accuracy if needed
        # Let's weight angular errors more heavily:
        ang_err = abs(theta - self.theta_final) + abs(sigma - self.sigma_final)
        weighted_dist = dist + 0.5 * ang_err  # add penalty for angle deviation

        reward = -weighted_dist
        terminated = False

        # Stricter success condition
        if dist < 0.05:
            reward += 1000.0
            terminated = True

        # Safety check
        if abs(z) > 100 or abs(theta) > math.pi/2 or abs(sigma) > math.pi:
            reward -= 500.0
            terminated = True

        return np.array(self.state, dtype=np.float32), reward, terminated, truncated, {}

    def render(self, mode='human'):
        pass
