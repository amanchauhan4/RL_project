#train the agent
import gymnasium
from gymnasium import spaces
from gymnasium.utils import seeding
import logging
import math
import numpy as np


logger = logging.getLogger(__name__)


from env_v1 import DroneNet
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.vec_env.util import copy_obs_dict, dict_to_obs, obs_space_info

# Instantiate the en
env = DroneNet()
env.reset()
# Define and Train the agent
# Parallel environments
#vec_env = make_vec_env(env, n_envs=4)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25000)
