# Note: This code assumes that:
# 1) You have the `env_v1.py` environment file unchanged in the same directory.
# 2) You have stable_baselines3 and its dependencies installed.
# 3) You have matplotlib and numpy installed.
# 4) The code below is meant as a standalone training and comparison script (e.g. "train_comparison.py").
#
# The code:
# - Imports the DroneNet environment from env_v1.py
# - Trains multiple algorithms (PPO, A2C, TD3, SAC) under different hyperparameter settings
# - Runs evaluation episodes to collect actions, rewards, states, etc.
# - Creates comparison plots:
#    - Plots of total reward per episode for different algorithms and hyperparameters on the same figure.
#    - Plots of state error over time for different algorithms/hyperparameters.
#    - Plots of force/moment/reward over time.
#    - Plots of states (z, theta, sigma) over time.
#
# In this example, we vary learning_rate and gamma. You can add more variations as you see fit.
#
# IMPORTANT:
# - This code will run multiple training sessions for each combination of algorithm and hyperparameters.
#   This can be time-consuming. Adjust the number of timesteps, episodes, etc., as needed.
# - The plotting logic combines results into single figures with multiple lines. Different algorithms will
#   have different colors, and different hyperparameters will be represented by different line styles or markers.

import os
import math
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, A2C, TD3, SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env_v1 import DroneNet

#--------------------------
# Training and Evaluation Setup
#--------------------------

# Hyperparameter grids
algorithms = {
    "PPO": PPO,
    "A2C": A2C,
    "TD3": TD3,
    "SAC": SAC,
}

learning_rates = [1e-4, 5e-4]
gammas = [0.99, 0.995]

# Training settings
total_timesteps = 100000  # adjust as needed
eval_freq = 5000
episodes_to_evaluate = 1  # number of episodes to run for evaluation after training

#--------------------------
# Utility Functions
#--------------------------

def make_env():
    return DroneNet()

def train_and_evaluate(algorithm_cls, algo_name, lr, gamma):
    """
    Train a given algorithm with specified hyperparameters and evaluate it.
    Returns:
      - force, moment, all_rewards, z, theta, sigma, state_errors, total_rewards
        collected from evaluation episodes
    """
    # Create environment
    env = make_env()
    venv = DummyVecEnv([make_env])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.)

    # Default hyperparams for demonstration. Adjust as needed for each algorithm if required.
    common_kwargs = {
        "policy": "MlpPolicy",
        "env": venv,
        "verbose": 0,
        "learning_rate": lr,
        "gamma": gamma,
    }

    # Some algorithms require different kwargs
    if algo_name in ["TD3"]:
        # TD3 defaults
        model = algorithm_cls(**common_kwargs, learning_starts=1000, buffer_size=100000, batch_size=100)
    elif algo_name in ["SAC"]:
        model = algorithm_cls(**common_kwargs, learning_starts=1000, buffer_size=100000, batch_size=64)
    else:
        # PPO or A2C defaults
        model = algorithm_cls(**common_kwargs)

    # Set up callbacks
    eval_callback = EvalCallback(
        venv,
        best_model_save_path=f'./logs/{algo_name}_lr{lr}_g{gamma}/best_model',
        log_path=f'./logs/{algo_name}_lr{lr}_g{gamma}',
        eval_freq=eval_freq,
        deterministic=True,
        render=False
    )
    checkpoint_callback = CheckpointCallback(save_freq=eval_freq, save_path=f'./logs/{algo_name}_lr{lr}_g{gamma}/checkpoints/')

    # Train the model
    model.learn(total_timesteps=total_timesteps, callback=[eval_callback, checkpoint_callback])

    # Save model and normalization stats
    model.save(f"./models/{algo_name}_lr{lr}_g{gamma}")
    venv.save(f"./models/{algo_name}_lr{lr}_g{gamma}_vecnormalize.pkl")

    # Evaluate model
    return collect_actions_and_constraints(make_env(), model, venv, episodes=episodes_to_evaluate)


def collect_actions_and_constraints(env, model, venv, episodes=1):
    """
    Run evaluation episodes and collect relevant data.
    """
    force = []
    moment = []
    all_rewards = []
    z = []
    theta = []
    sigma = []
    state_errors = []
    total_rewards = []
    time_steps = []

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        step = 0
        episode_reward = 0
        while not done:
            # Normalize observation using venv's internal function
            # (We need to call the internal obs normalization if we want the same behavior as during training)
            obs_norm = venv.normalize_obs(obs)
            action, _states = model.predict(obs_norm, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            force.append(action[0])
            moment.append(action[1])
            all_rewards.append(reward)
            z.append(obs[0])
            theta.append(obs[1])
            sigma.append(obs[2])

            error = np.array([obs[0]-env.z_final,
                              obs[1]-env.theta_final,
                              obs[2]-env.sigma_final,
                              obs[3]-env.z_dot_final,
                              obs[4]-env.theta_dot_final,
                              obs[5]-env.sigma_dot_final])
            dist = np.linalg.norm(error)
            state_errors.append(dist)
            time_steps.append(step)
            episode_reward += reward
            step += 1

        total_rewards.append(episode_reward)

    return force, moment, all_rewards, z, theta, sigma, state_errors, total_rewards


#--------------------------
# Run Experiments
#--------------------------

# Data structure to store results for plotting
results = {
    # structure:
    # algo_name: {
    #   (lr, gamma): {
    #       "force": [...],
    #       "moment": [...],
    #       "rewards": [...],
    #       "z": [...],
    #       "theta": [...],
    #       "sigma": [...],
    #       "state_errors": [...],
    #       "total_rewards": [...]
    #   }
    # }
}

for algo_name, algo_cls in algorithms.items():
    results[algo_name] = {}
    for lr in learning_rates:
        for gm in gammas:
            print(f"Training {algo_name} with lr={lr}, gamma={gm}")
            force, moment, all_rewards, z, theta, sigma, state_errors, total_rewards = train_and_evaluate(algo_cls, algo_name, lr, gm)
            results[algo_name][(lr, gm)] = {
                "force": force,
                "moment": moment,
                "rewards": all_rewards,
                "z": z,
                "theta": theta,
                "sigma": sigma,
                "state_errors": state_errors,
                "total_rewards": total_rewards,
            }


#--------------------------
# Plotting
#--------------------------

# We'll create combined plots:
# 1) Total Reward per Episode (different algorithms and hyperparams)
# 2) State Error Over Time
# 3) Force/Moment/Reward Over Time
# 4) z, theta, sigma Over Time

# To differentiate algorithms, assign each algorithm a color
algo_colors = {
    "PPO": "blue",
    "A2C": "green",
    "TD3": "red",
    "SAC": "purple"
}

# For different hyperparams (lr, gamma), vary line styles or markers
line_styles = ["-", "--", "-.", ":"]
markers = ["o", "s", "d", "^"]
# We'll cycle through these for each (lr, gamma) combination
param_combos = [(lr, gm) for lr in learning_rates for gm in gammas]

# Create mapping from (lr,gamma) to style/marker
style_map = {}
for i, combo in enumerate(param_combos):
    style_map[combo] = (line_styles[i % len(line_styles)], markers[i % len(markers)])


# Plot total reward per episode
plt.figure()
for algo_name in results:
    for (lr, gm) in results[algo_name]:
        total_rewards = results[algo_name][(lr, gm)]["total_rewards"]
        style, marker = style_map[(lr, gm)]
        plt.plot(range(len(total_rewards)), total_rewards,
                 linestyle=style, marker=marker, color=algo_colors[algo_name],
                 label=f"{algo_name}, lr={lr}, gamma={gm}")
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Reward per Episode - Comparison')
plt.legend()
plt.savefig('comparison_total_reward_per_episode.png')


# Plot state error over time
plt.figure()
for algo_name in results:
    for (lr, gm) in results[algo_name]:
        state_errors = results[algo_name][(lr, gm)]["state_errors"]
        style, marker = style_map[(lr, gm)]
        plt.plot(range(len(state_errors)), state_errors,
                 linestyle=style, marker=marker, color=algo_colors[algo_name],
                 label=f"{algo_name}, lr={lr}, gamma={gm}")
plt.xlabel('Time Step')
plt.ylabel('State Error')
plt.title('State Error Over Time - Comparison')
plt.legend()
plt.savefig('comparison_state_error_over_time.png')


# Plot actions (force, moment) and rewards over time
plt.figure(figsize=(10, 10))

plt.subplot(3, 1, 1)
for algo_name in results:
    for (lr, gm) in results[algo_name]:
        force = results[algo_name][(lr, gm)]["force"]
        style, marker = style_map[(lr, gm)]
        plt.plot(range(len(force)), force, linestyle=style, marker=marker, color=algo_colors[algo_name],
                 label=f"{algo_name}, lr={lr}, gamma={gm}")
plt.xlabel('Timestep')
plt.ylabel('Force')
plt.title('Force Over Time')
plt.legend()

plt.subplot(3, 1, 2)
for algo_name in results:
    for (lr, gm) in results[algo_name]:
        moment = results[algo_name][(lr, gm)]["moment"]
        style, marker = style_map[(lr, gm)]
        plt.plot(range(len(moment)), moment, linestyle=style, marker=marker, color=algo_colors[algo_name],
                 label=f"{algo_name}, lr={lr}, gamma={gm}")
plt.xlabel('Timestep')
plt.ylabel('Moment')
plt.title('Moment Over Time')
plt.legend()

plt.subplot(3, 1, 3)
for algo_name in results:
    for (lr, gm) in results[algo_name]:
        rew = results[algo_name][(lr, gm)]["rewards"]
        style, marker = style_map[(lr, gm)]
        plt.plot(range(len(rew)), rew, linestyle=style, marker=marker, color=algo_colors[algo_name],
                 label=f"{algo_name}, lr={lr}, gamma={gm}")
plt.xlabel('Timestep')
plt.ylabel('Reward')
plt.title('Reward Over Time')
plt.legend()

plt.tight_layout()
plt.savefig('comparison_actions_rewards_over_time.png')


# Plot states (z, theta, sigma) over time
plt.figure(figsize=(10, 10))

plt.subplot(3, 1, 1)
for algo_name in results:
    for (lr, gm) in results[algo_name]:
        z = results[algo_name][(lr, gm)]["z"]
        style, marker = style_map[(lr, gm)]
        plt.plot(range(len(z)), z, linestyle=style, marker=marker, color=algo_colors[algo_name],
                 label=f"{algo_name}, lr={lr}, gamma={gm}")
plt.xlabel('Timestep')
plt.ylabel('z')
plt.title('Vertical Position Over Time')
plt.legend()

plt.subplot(3, 1, 2)
for algo_name in results:
    for (lr, gm) in results[algo_name]:
        theta = results[algo_name][(lr, gm)]["theta"]
        style, marker = style_map[(lr, gm)]
        plt.plot(range(len(theta)), np.array(theta)*180.0/math.pi, linestyle=style, marker=marker, color=algo_colors[algo_name],
                 label=f"{algo_name}, lr={lr}, gamma={gm}")
plt.xlabel('Timestep')
plt.ylabel('Theta (deg)')
plt.title('Theta Over Time')
plt.legend()

plt.subplot(3, 1, 3)
for algo_name in results:
    for (lr, gm) in results[algo_name]:
        sigma = results[algo_name][(lr, gm)]["sigma"]
        style, marker = style_map[(lr, gm)]
        plt.plot(range(len(sigma)), np.array(sigma)*180.0/math.pi, linestyle=style, marker=marker, color=algo_colors[algo_name],
                 label=f"{algo_name}, lr={lr}, gamma={gm}")
plt.xlabel('Timestep')
plt.ylabel('Sigma (deg)')
plt.title('Sigma Over Time')
plt.legend()

plt.tight_layout()
plt.savefig('comparison_states_over_time.png')

print("Training and comparison plots completed successfully.")
