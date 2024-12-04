
import gymnasium
from gymnasium import spaces
from gymnasium.utils import seeding
import logging
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
        self.mu = self.m/self.n/self.md
        self.l = 3
        self.I = 1/3.0
        self.r = 0.2
        self.dt = 1e-3
        self.t = 0  # timestep
        self.t_limit = 30000
        self.Fmax = 20
        self.Mmax = 100
        self.Fdot_max = 3
        self.Mdot_max = self.Fdot_max*self.l
        self.z_max = 100
        self.theta_max = math.pi/2.0
        self.sigma_max = math.pi
        #define boundary conditions
        self.z_initial = self.r
        self.theta_initial = 0.0
        self.sigma_intial = 0.0
        self.z_dot_intial = -1.0
        self.theta_dot_initial = -self.z_dot_intial/self.l*2
        self.sigma_dot_intial = 0.0

        self.z_final = -2.0
        self.sigma_final = math.pi/180.0*30.0
        self.theta_final = math.atan(self.mu/(1+self.mu)/math.tan(self.sigma_final))
        self.z_dot_final = 0.0
        self.theta_dot_final = 0.0
        self.sigma_dot_final = 0.0

        self.x_initial = np.array([self.z_initial,self.theta_initial,self.sigma_intial,self.z_dot_intial,self.theta_dot_initial,self.sigma_dot_intial,self.g*(1+self.mu)/np.cos(self.sigma_intial)*self.md,0.0])
        self.x_final = np.array([self.z_final,self.theta_final,self.sigma_final,self.z_dot_final,self.theta_dot_final,self.sigma_dot_final,self.g*(1+self.mu)/np.cos(self.sigma_final)*self.md,0.0])
        
        high = np.array([
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max])
        
        self.high_action = np.array([
            1.0,
            1.0])
        self.low_action = np.array([
            -1.0,
            -1.0])

        #self.action_space = spaces.Box(-1.0, 1.0, shape=(2,))
        self.action_space = spaces.Box(self.low_action,self.high_action)

        self.observation_space = spaces.Box(-high, high)

#        self.seed()
        self.viewer = None
        self.state = None

#    def seed(self, seed=None):
#        self.np_random, seed = seeding.np_random(seed)
#        return [seed]

    def step(self, action):
        # Valid action
        #action = np.clip(action, -1.0, 1.0)[0]
        #action *= self.force_mag

        state = self.state
        z,theta,sigma,z_dot,theta_dot,sigma_dot,Force,Moment = state
        alpha = Force/self.md
        M_control= Moment
        #alpha = self.Fmax*action[0]/self.md
        #M_control = self.Mmax*action[1]
        M = np.zeros((3,3))


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
        
        
        Fdot=action[0]*self.Fdot_max
        Mdot=action[1]*self.Mdot_max

        
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
        Force = Force+Fdot*self.dt
        Moment = Moment+Mdot*self.dt
        self.state=(z,theta,sigma,z_dot,theta_dot,sigma_dot,Force,Moment)

        terminated = False
        truncated = False
        reward = -1.0
        if z < -self.z_max or z> self.z_max or theta>self.theta_max or theta<0 or sigma<-self.sigma_max or sigma>self.sigma_max or Force<0 or Force>self.Fmax or abs(Moment)>self.Mmax :
            terminated=True
            reward+= -5000.0

        self.t += 1

        if self.t >= self.t_limit:
            truncated=True
        if np.linalg.norm(np.array([z,theta,sigma,z_dot,theta_dot,sigma_dot,Force,Moment])-self.x_final)<1.0:
            reward+=5000
            terminated=True

        obs = np.array([z,theta,sigma,z_dot,theta_dot,sigma_dot,Force,Moment])

        return obs, reward, terminated,truncated,{}

    def reset(self,seed=None,options=None):
        # self.state = self.np_random.normal(loc=np.array([0.0, 0.0, 30*(2*np.pi)/360, 0.0]), scale=np.array([0.0, 0.0, 0.0, 0.0]))
        #self.state = np.random.normal(loc=self.x_initial, scale=np.array([1e-3,1e-3, 1e-3, 1e-3,1e-3,1e-3,1e-3,1e-3]))
        self.state = self.x_initial
        self.steps_beyond_done = None
        self.t = 0  # timestep
        z,theta,sigma,z_dot,theta_dot,sigma_dot,Force,Moment = self.state
        obs = np.array([z,theta,sigma,z_dot,theta_dot,sigma_dot,Force,Moment])
        return obs,{}

    def render(self, mode='human', close=False):
        pass