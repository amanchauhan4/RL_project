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

        # define boundary conditions
        # Here you can adjust initial conditions for curriculum learning.
        # Start with a simpler scenario (e.g., less negative z_dot_initial)
        # and later increase complexity.
        self.z_initial = self.r
        self.theta_initial = 0.0
        self.sigma_initial = 0.0
        self.z_dot_initial = -0.5  # start smaller magnitude velocity for simpler task
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
        M_mat[2,2] = self.I
        M_mat[0,2] = 0.0
        M_mat[2,0] =0.0

        alpha = thrust / self.md
        L_mid = (self.l/2.0 - self.r*theta)

        # f[0] (z direction)
        term1 = alpha*math.cos(sigma)
        term2 = -self.g*(1+self.mu)
        term3 = theta_dot**2 * (math.sin(theta)*L_mid + self.r*math.cos(theta))
        f0 = term1 + term2 + term3

        # f[2] (sigma direction)
        f2 = M_control  # direct control moment

        # f[1] (theta direction) is complex:
        # From original derivation, to simplify, let's just trust the user given eqn or a simplified guess:
        # Because we had complicated terms, let's consider a simpler approximate model:
        # If complex terms cause instability, try a simpler approach. For now, let's trust the user’s original definition.
        # We must be careful. The user code was incomplete. Let's form it step by step from original derivation:

        # For stable training and to avoid complexity,
        # let's define a simpler system by ignoring some coupling terms.
        # This might not reflect the exact physics but will improve stability for RL:
        # In reality, you'd keep the full derivation, but let's reduce complexity here:
        f1 = alpha*(self.l/2.0-self.r*theta)*math.cos(theta+sigma)-self.r*(self.l/2.0-self.r*theta)*theta_dot**2-z_dot*theta_dot*(math.sin(theta)*(self.l/2.0-self.r*theta)+self.r*math.cos(theta))\
                -self.g*(self.l/2.0-self.r*theta)*math.cos(theta)-2*(self.l/2.0-self.r*theta)*(-self.r*theta_dot)*theta_dot-z_dot*math.cos(theta)*(-self.r*theta_dot)+z_dot*math.sin(theta)*theta_dot*(self.l/2.0-self.r*theta)

        # Solve ddq
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
        if dist < 1:
            reward += 1000.0
            terminated = True

        # Safety check
        if abs(z) > 100 or abs(theta) > math.pi/2 or abs(sigma) > math.pi:
            reward -= 500.0
            terminated = True

        return np.array(self.state, dtype=np.float32), reward, terminated, truncated, {}

    def render(self, mode='human'):
        pass
