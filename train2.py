import gymnasium as gym
import logging
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import matplotlib.pyplot as plt
from env2 import DroneNet  # Import your custom environment
import math

# Instantiate the environment
env = DroneNet()

# Normalize observations (if necessary)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Create a vectorized environment
venv = DummyVecEnv([lambda: env])

# Normalize the observations and rewards
venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.)

# Define and train the PPO agent with adjusted hyperparameters
model = PPO(
    "MlpPolicy",
    venv,
    verbose=1,
    learning_rate=1e-4,  # Lower learning rate
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    tensorboard_log="./ppo_dronenet_tensorboard/"
)

# Create callbacks
eval_callback = EvalCallback(
    venv,
    best_model_save_path='./logs/best_model',
    log_path='./logs/',
    eval_freq=5000,
    deterministic=True,
    render=False
)

checkpoint_callback = CheckpointCallback(save_freq=5000, save_path='./logs/checkpoints/')

# Train the agent
total_timesteps = 100000  # Increase training time
model.learn(total_timesteps=total_timesteps, callback=[eval_callback, checkpoint_callback])

# Save the trained model
model.save("ppo_dronenet_model")

# Function to collect actions and constraints during evaluation
def collect_actions_and_constraints(env, model, episodes=1):
    force = []
    moment = []
    all_rewards = []
    z = []
    theta = []
    sigma = []
    constraints_violated = []
    time_steps = []
    total_rewards = []
    state_errors = []

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        episode_reward = 0
        while not done:
            # Normalize observation
            obs_norm = venv.normalize_obs(obs)

            # Get the action from the model
            action, _states = model.predict(obs_norm, deterministic=True)

            # Step in the environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store actions and observations
            force.append(action[0])
            moment.append(action[1])
            all_rewards.append(reward)
            z.append(obs[0])
            theta.append(obs[1])
            sigma.append(obs[2])
            state_errors.append(info.get('state_error', 0))
            time_steps.append(step)
            episode_reward += reward

            # Check for constraint violations
            violation = info.get('constraint_violation', False)
            constraints_violated.append(violation)

            # Print constraint information every 100 steps
            if step % 100 == 0:
                print(f"Step {step}: z={obs[0]:.2f}, theta={obs[1]:.2f}, sigma={obs[2]:.2f}, "
                      f"Force={obs[6]:.2f}, Moment={obs[7]:.2f}, Violation={violation}")

            step += 1

        total_rewards.append(episode_reward)

    return force, moment, all_rewards, z, theta, sigma, constraints_violated, time_steps, total_rewards, state_errors

# Collect actions and constraints during evaluation
force, moment, all_rewards, z, theta, sigma, constraints_violated, time_steps, total_rewards, state_errors = collect_actions_and_constraints(env, model, episodes=1)

# Plot total rewards over time
plt.figure()
plt.plot(total_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Reward per Episode')
plt.savefig('total_reward_per_episode.png')

# Plot state error over time
plt.figure()
plt.plot(state_errors)
plt.xlabel('Time Step')
plt.ylabel('State Error')
plt.title('State Error Over Time')
plt.savefig('state_error_over_time.png')

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

# Plot the rewards vs time
plt.subplot(3, 1, 3)  # Create a subplot for rewards
plt.plot(range(len(all_rewards)), all_rewards, marker='o', linestyle='-', color='r')
plt.xlabel('Timestep')
plt.ylabel('Reward')
plt.title('Reward Received by the Agent Over Time')

plt.tight_layout()  # Adjust layout to prevent overlap
plt.savefig('actions_vs_time.png')

# Plot states vs time (timesteps)
plt.figure(figsize=(10, 6))

# Plot the vertical position (z) vs time
plt.subplot(3, 1, 1)
plt.plot(range(len(z)), z, marker='o', linestyle='-', color='b')
plt.xlabel('Timestep')
plt.ylabel('z')
plt.title('Vertical Position Over Time')

# Plot theta vs time
plt.subplot(3, 1, 2)
plt.plot(range(len(theta)), np.array(theta) * 180.0 / math.pi, marker='o', linestyle='-', color='b')
plt.xlabel('Timestep')
plt.ylabel('Theta (degrees)')
plt.title('Theta Over Time')

# Plot sigma vs time
plt.subplot(3, 1, 3)
plt.plot(range(len(sigma)), np.array(sigma) * 180.0 / math.pi, marker='o', linestyle='-', color='b')
plt.xlabel('Timestep')
plt.ylabel('Sigma (degrees)')
plt.title('Sigma Over Time')

plt.tight_layout()
plt.savefig('states_vs_time.png')
