"""
Cart pole swing-up:
Adapted from:
hardmaru: https://github.com/hardmaru/estool/blob/master/custom_envs/cartpole_swingup.py


Original version from:
https://github.com/zuoxingdong/DeepPILCO/blob/master/cartpole_swingup.py
hardmaru's changes:
More difficult, since dt is 0.05 (not 0.01), and only 200 timesteps
"""

import gym
from gym import spaces
from gym.utils import seeding
import logging
import math
import numpy as np

logger = logging.getLogger(__name__)


class DroneNet():
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.g = 9.81  # gravity
        self.md = 1.0
        self.m = 1.0
        self.n = 5.0
        self.mu = self.m/self.n/self.md
        self.l = 3
        self.I = 1/3.0
        self.r = 0.2
        self.t = 0  # timestep
        self.t_limit = 1000
        self.Fmax = 20
        self.Mmax = 100
        self.z_max = 100
        self.theta_max = math.pi/2.0
        self.sigma_max = math.pi 
        
        high = np.array([
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max])
        
        self.high_action = np.array([
            self.Fmax,
            self.Mmax])
        self.low_action = np.array([
            0.0,
            -self.Mmax])

        #self.action_space = spaces.Box(-1.0, 1.0, shape=(2,))
        self.action_space = spaces.Box(self.low_action,self.high_action)

        self.observation_space = spaces.Box(-high, high)

        self.seed()
        self.viewer = None
        self.state = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # Valid action
        #action = np.clip(action, -1.0, 1.0)[0]
        #action *= self.force_mag

        state = self.state
        z,theta,sigma,z_dot,theta_dot,sigma_dot = state

        alpha = action[0]/self.md
        M_control = action[1]
        M = np.zeros(3,3)

        M[0,0]=1+self.mu
        M[0,1]=(self.l/2.0-self.r*theta)*math.cos(theta)
        M[1,0]=math.cos(theta)*(self.l/2.0-self.r*theta)
        M[1,1]=(self.l/2.0-self.r*theta)**2
        M[2,2]=self.I
        v=np.zeros(3)
        v[0] = alpha*math.cos(sigma)-self.g*(1+self.mu)+theta_dot**2*(math.sin(theta)*(self.l/2.0-self.r*theta)+self.r*math.cos(theta))
        v[1]=alpha*(self.l/2.0-self.r*theta)*math.cos(theta+sigma)-self.r*(self.l/2.0-self.r*theta)*theta_dot**2-z_dot*theta_dot*(math.sin(theta)*(self.l/2.0-self.r*theta)+self.r*math.cos(theta))\
                -self.g*(self.l/2.0-self.r*theta)*math.cos(theta)-2*(self.l/2.0-self.r*theta)*(-self.r*theta_dot)*theta_dot-z_dot*math.cos(theta)*(-self.r*theta_dot)+z_dot*math.sin(theta)*theta_dot*(self.l/2.0-self.r*theta)
        
        v[2]=M_control

        ddx = np.matmul(np.linalg.inv(M),v)
        z_ddot = ddx[0]
        theta_ddot = ddx[1]
        sigma_ddot = ddx[2]
        
        
        z = z+self.dt*z_dot
        theta = theta+self.dt*theta_dot
        sigma = sigma+self.dt*sigma_dot
        z_dot = z_dot+self.dt*z_ddot
        theta_dot = theta_dot+self.dt*theta_ddot
        sigma_dot = sigma_dot+self.dt*sigma_ddot
        
        self.state=(z,theta,sigma,z_dot,theta_dot,sigma_dot)

        terminated = False
        truncated = False
        if z < -self.z_max or z> self.z_max or theta>self.theta_max or theta<0 or sigma<-self.sigma_max or sigma>self.sigma_max:
            terminated=True

        self.t += 1

        if self.t >= self.t_limit:
            truncated=True

        reward = -1.0

        obs = np.array([z,theta,sigma,z_dot,theta_dot,sigma_dot])

        return obs, reward, terminated, truncated

    def _reset(self):
        # self.state = self.np_random.normal(loc=np.array([0.0, 0.0, 30*(2*np.pi)/360, 0.0]), scale=np.array([0.0, 0.0, 0.0, 0.0]))
        self.state = np.random.normal(loc=np.array([0.0, 0.0, np.pi, 0.0]), scale=np.array([0.2, 0.2, 0.2, 0.2]))
        self.steps_beyond_done = None
        self.t = 0  # timestep
        x, x_dot, theta, theta_dot = self.state
        obs = np.array([x, x_dot, np.cos(theta), np.sin(theta), theta_dot])
        return obs

    def _render(self, mode='human', close=False):
        pass