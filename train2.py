import gymnasium as gym
import logging
import numpy as np
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from env_v2 import DroneNet  # Import your custom environment
from pathlib import Path
import math 
# Logger setup
logger = logging.getLogger(__name__)

# Instantiate the environment
env = DroneNet()

env.reset()

# Define and train the PPO agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25000)

# Save the model using pathlib to handle path more safely
#save_path = Path("D:/MSAAE/Fall_2024/Reinforcement_learning/drone_net_axisymmetric/RL_project/try2/ppo_dronenet_model")
#model.save(save_path)

# You can later load the model using the same path:
#model = PPO.load(save_path)

def collect_actions_vs_time(env, model, episodes=1):
    """
    Collect the actions taken by the agent over time for a number of episodes.
    """
    force = []  # List to store forces (or actions)
    moment = []  # List to store moments (or other actions)
    all_rewards = []
    z = []
    theta = []
    sigma = []
    for ep in range(episodes):
        obs,_ = env.reset()
       
        done = False
        while not done:
            # Get the action from the model
            
            action, _states = model.predict(obs)
            
            # Store the action components (assuming action is a tuple)
            force.append(action[0])  # action[0] for force
            moment.append(action[1])  # action[1] for moment (or another action dimension)
            
            # Step in the environment
            obs, rewards,terminated, truncated, info = env.step(action)
            all_rewards.append(rewards)
            z.append(obs[0])
            theta.append(obs[1])
            sigma.append(obs[2])
            done = terminated or truncated
    return force, moment,all_rewards,z,theta,sigma

# Collect actions taken during training
force, moment,all_rewards ,z,theta,sigma= collect_actions_vs_time(env, model, episodes=1)

# Plot actions vs time (timesteps)
plt.figure(figsize=(10, 6))

# Plot the force (action[0]) vs time
plt.subplot(3, 1, 1)  # Create a subplot for force
plt.plot(range(len(force)), force, marker='o', linestyle='-', color='b')
plt.xlabel('Timestep')
plt.ylabel('Force')
plt.title('Force Taken by the Agent Over Time')

# Plot the moment (action[1]) vs time
plt.subplot(3, 1, 2)  # Create a subplot for moment
plt.plot(range(len(moment)), moment, marker='o', linestyle='-', color='g')
plt.xlabel('Timestep')
plt.ylabel('Moment')
plt.title('Moment Taken by the Agent Over Time')

# Plot the moment (action[1]) vs time
plt.subplot(3, 1, 3)  # Create a subplot for moment
plt.plot(range(len(moment)), all_rewards, marker='o', linestyle='-', color='g')
plt.xlabel('Timestep')
plt.ylabel('Moment')
plt.title('Moment Taken by the Agent Over Time')

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()



# Plot actions vs time (timesteps)
plt.figure(figsize=(10, 6))

# Plot the force (action[0]) vs time
plt.subplot(3, 1, 1)  # Create a subplot for force
plt.plot(range(len(force)), z, marker='o', linestyle='-', color='b')
plt.xlabel('Timestep')
plt.ylabel('z')

# Plot the force (action[0]) vs time
plt.subplot(3, 1, 2)  # Create a subplot for force
plt.plot(range(len(force)), np.array(theta)*180.0/math.pi, marker='o', linestyle='-', color='b')
plt.xlabel('Timestep')
plt.ylabel('theta')

# Plot the force (action[0]) vs time
plt.subplot(3, 1, 3)  # Create a subplot for force
plt.plot(range(len(force)), np.array(sigma)*180.0/math.pi, marker='o', linestyle='-', color='b')
plt.xlabel('Timestep')
plt.ylabel('sigma')


plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()
