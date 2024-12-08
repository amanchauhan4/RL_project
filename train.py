# train_comparison.py

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
gammas = [0.99, 0.995, 0.95]  # Added gamma=0.95 as per your request

# Training settings
total_timesteps = 20000  # adjust as needed
eval_freq = 5000
episodes_to_evaluate = 1  # number of episodes to run for evaluation after training

# Ensure directories exist
os.makedirs("./models", exist_ok=True)
os.makedirs("./logs", exist_ok=True)

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
    model_log_dir = f'./logs/{algo_name}_lr{lr}_g{gamma}'
    os.makedirs(model_log_dir, exist_ok=True)
    eval_callback = EvalCallback(
        venv,
        best_model_save_path=os.path.join(model_log_dir, 'best_model'),
        log_path=model_log_dir,
        eval_freq=eval_freq,
        deterministic=True,
        render=False
    )
    checkpoint_callback = CheckpointCallback(save_freq=eval_freq, save_path=os.path.join(model_log_dir, 'checkpoints/'))

    # Train the model
    model.learn(total_timesteps=total_timesteps, callback=[eval_callback, checkpoint_callback])

    # Save model and normalization stats
    model_save_path = f"./models/{algo_name}_lr{lr}_g{gamma}"
    model.save(model_save_path)
    venv.save(f"{model_save_path}_vecnormalize.pkl")

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

            error = np.array([
                obs[0] - env.z_final,
                obs[1] - env.theta_final,
                obs[2] - env.sigma_final,
                obs[3] - env.z_dot_final,
                obs[4] - env.theta_dot_final,
                obs[5] - env.sigma_dot_final
            ])
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

# Define colors for algorithms
algo_colors = {
    "PPO": "blue",
    "A2C": "green",
    "TD3": "red",
    "SAC": "purple"
}

# Define line styles and markers for hyperparameter combinations
line_styles = ["-", "--", "-.", ":"]
markers = ["o", "s", "d", "^"]
param_combos = [(lr, gm) for lr in learning_rates for gm in gammas]
style_map = {}
for i, combo in enumerate(param_combos):
    style_map[combo] = (line_styles[i % len(line_styles)], markers[i % len(markers)])

#------------------------------------------------
# 1) Comparison Plots for lr=1e-4 and gamma=0.99
#------------------------------------------------

fixed_lr = 1e-4
fixed_gamma = 0.99

# Extract data for the chosen hyperparams and all four algorithms
z_dict = {}
theta_dict = {}
sigma_dict = {}
force_dict = {}
moment_dict = {}

for algo_name in ["PPO", "A2C", "TD3", "SAC"]:
    if (fixed_lr, fixed_gamma) in results[algo_name]:
        z_dict[algo_name] = results[algo_name][(fixed_lr, fixed_gamma)]["z"]
        theta_dict[algo_name] = results[algo_name][(fixed_lr, fixed_gamma)]["theta"]
        sigma_dict[algo_name] = results[algo_name][(fixed_lr, fixed_gamma)]["sigma"]
        force_dict[algo_name] = results[algo_name][(fixed_lr, fixed_gamma)]["force"]
        moment_dict[algo_name] = results[algo_name][(fixed_lr, fixed_gamma)]["moment"]
    else:
        print(f"Warning: No data found for {algo_name} at lr={fixed_lr} and gamma={fixed_gamma}")

# Plot z vs time
plt.figure()
for algo_name in z_dict:
    plt.plot(range(len(z_dict[algo_name])), z_dict[algo_name], color=algo_colors[algo_name], label=algo_name)
plt.xlabel('Timestep')
plt.ylabel('z')
plt.title(f'z vs Time (lr={fixed_lr}, gamma={fixed_gamma})')
plt.legend()
plt.savefig('comparison_z_vs_time_lr1e-4_gamma0.99.png')
plt.close()

# Plot theta vs time (convert to degrees)
plt.figure()
for algo_name in theta_dict:
    theta_deg = np.array(theta_dict[algo_name]) * 180.0 / math.pi
    plt.plot(range(len(theta_deg)), theta_deg, color=algo_colors[algo_name], label=algo_name)
plt.xlabel('Timestep')
plt.ylabel('Theta (deg)')
plt.title(f'Theta vs Time (lr={fixed_lr}, gamma={fixed_gamma})')
plt.legend()
plt.savefig('comparison_theta_vs_time_lr1e-4_gamma0.99.png')
plt.close()

# Plot sigma vs time (convert to degrees)
plt.figure()
for algo_name in sigma_dict:
    sigma_deg = np.array(sigma_dict[algo_name]) * 180.0 / math.pi
    plt.plot(range(len(sigma_deg)), sigma_deg, color=algo_colors[algo_name], label=algo_name)
plt.xlabel('Timestep')
plt.ylabel('Sigma (deg)')
plt.title(f'Sigma vs Time (lr={fixed_lr}, gamma={fixed_gamma})')
plt.legend()
plt.savefig('comparison_sigma_vs_time_lr1e-4_gamma0.99.png')
plt.close()

# Plot force vs time
plt.figure()
for algo_name in force_dict:
    plt.plot(range(len(force_dict[algo_name])), force_dict[algo_name], color=algo_colors[algo_name], label=algo_name)
plt.xlabel('Timestep')
plt.ylabel('Force')
plt.title(f'Force vs Time (lr={fixed_lr}, gamma={fixed_gamma})')
plt.legend()
plt.savefig('comparison_force_vs_time_lr1e-4_gamma0.99.png')
plt.close()

# Plot moment vs time
plt.figure()
for algo_name in moment_dict:
    plt.plot(range(len(moment_dict[algo_name])), moment_dict[algo_name], color=algo_colors[algo_name], label=algo_name)
plt.xlabel('Timestep')
plt.ylabel('Moment')
plt.title(f'Moment vs Time (lr={fixed_lr}, gamma={fixed_gamma})')
plt.legend()
plt.savefig('comparison_moment_vs_time_lr1e-4_gamma0.99.png')
plt.close()

#------------------------------------------------
# 2) Hyperparameter Variation Plots for PPO
#------------------------------------------------

# Define PPO-specific color maps for varying learning rates and gammas
ppo_color_map_lr = {
    1e-4: "blue",
    5e-4: "red"
}

ppo_color_map_gamma = {
    0.99: "blue",
    0.995: "green",
    0.95: "red"
}

# Define hyperparameter combinations to compare
compare_lrs = [1e-4, 5e-4]
fixed_gamma_for_lr = 0.99

compare_gammas = [0.99, 0.995, 0.95]
fixed_lr_for_gamma = 1e-4

# Check that PPO results are available for these combinations
for lr_val in compare_lrs:
    if (lr_val, fixed_gamma_for_lr) not in results["PPO"]:
        print(f"Warning: No PPO data for lr={lr_val}, gamma={fixed_gamma_for_lr}")
for gm_val in compare_gammas:
    if (fixed_lr_for_gamma, gm_val) not in results["PPO"]:
        print(f"Warning: No PPO data for lr={fixed_lr_for_gamma}, gamma={gm_val}")

# Function to plot PPO data with given parameter variations
def plot_ppo_comparison(metric, ylabel, title_suffix, filename_suffix, conversion=lambda x: x):
    plt.figure()
    # Learning rate comparison at fixed gamma
    for lr_val in compare_lrs:
        if (lr_val, fixed_gamma_for_lr) in results["PPO"]:
            data = results["PPO"][(lr_val, fixed_gamma_for_lr)][metric]
            data_converted = conversion(data)
            plt.plot(range(len(data_converted)), data_converted, color=ppo_color_map_lr[lr_val],
                     label=f"lr={lr_val}, gamma={fixed_gamma_for_lr}")
    # Gamma comparison at fixed learning rate
    for gm_val in compare_gammas:
        if (fixed_lr_for_gamma, gm_val) in results["PPO"]:
            data = results["PPO"][(fixed_lr_for_gamma, gm_val)][metric]
            data_converted = conversion(data)
            plt.plot(range(len(data_converted)), data_converted, color=ppo_color_map_gamma[gm_val],
                     label=f"lr={fixed_lr_for_gamma}, gamma={gm_val}")
    plt.xlabel('Timestep')
    plt.ylabel(ylabel)
    plt.title(f'PPO: Effect of Hyperparameters on {title_suffix}')
    plt.legend()
    plt.savefig(f'ppo_{filename_suffix}.png')
    plt.close()

# Plot z vs time
plot_ppo_comparison(
    metric="z",
    ylabel="z",
    title_suffix="z",
    filename_suffix="z_vs_time_hyperparams",
    conversion=lambda x: x
)

# Plot theta vs time (convert to degrees)
plot_ppo_comparison(
    metric="theta",
    ylabel="Theta (deg)",
    title_suffix="Theta",
    filename_suffix="theta_vs_time_hyperparams",
    conversion=lambda x: np.array(x) * 180.0 / math.pi
)

# Plot sigma vs time (convert to degrees)
plot_ppo_comparison(
    metric="sigma",
    ylabel="Sigma (deg)",
    title_suffix="Sigma",
    filename_suffix="sigma_vs_time_hyperparams",
    conversion=lambda x: np.array(x) * 180.0 / math.pi
)

# Plot force vs time
plot_ppo_comparison(
    metric="force",
    ylabel="Force",
    title_suffix="Force",
    filename_suffix="force_vs_time_hyperparams",
    conversion=lambda x: x
)

# Plot moment vs time
plot_ppo_comparison(
    metric="moment",
    ylabel="Moment",
    title_suffix="Moment",
    filename_suffix="moment_vs_time_hyperparams",
    conversion=lambda x: x
)

print("Training and comparison plots completed successfully.")
