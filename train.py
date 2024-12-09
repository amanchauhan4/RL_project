import os
import math
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, A2C, TD3, SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env_v1 import DroneNet

#---------------------------------------
# Training and Evaluation Setup
#---------------------------------------

algorithms = {
    "PPO": PPO,
    "A2C": A2C,
    "TD3": TD3,
    "SAC": SAC,
}

# Hyperparameters to vary
learning_rates = [1e-4, 5e-4]
gammas = [0.99, 0.995]

# Increase total timesteps for more thorough training
total_timesteps = 60000
eval_freq = 2000
episodes_to_evaluate = 1

os.makedirs("./models", exist_ok=True)
os.makedirs("./logs", exist_ok=True)

#---------------------------------------
# Utility Functions
#---------------------------------------

def make_env():
    return DroneNet()

def collect_actions_and_constraints(env, model, venv, episodes=1):
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
            obs_norm = venv.normalize_obs(obs)
            action, _ = model.predict(obs_norm, deterministic=True)
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

def train_and_evaluate(algorithm_cls, algo_name, lr, gamma):
    env = make_env()
    venv = DummyVecEnv([make_env])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.)

    # Modify hyperparams especially for A2C to improve performance:
    # We'll give A2C a bigger n_steps and lower lr:
    if algo_name == "A2C":
        current_lr = 3e-5 if lr == 1e-4 else lr*0.3  # just a heuristic to lower A2C lr
        n_steps = 4096
    else:
        current_lr = lr
        n_steps = 2048

    # Common kwargs for on-policy algorithms (PPO, A2C)
    on_policy_kwargs = {
        "policy": "MlpPolicy",
        "env": venv,
        "verbose": 1,
        "learning_rate": current_lr,
        "gamma": gamma,
    }

    # Common kwargs for off-policy algorithms (TD3, SAC)
    off_policy_kwargs = {
        "policy": "MlpPolicy",
        "env": venv,
        "verbose": 1,
        "learning_rate": current_lr,
        "gamma": gamma,
        "learning_starts": 1000,
        "buffer_size": 100000,
    }

    # Construct model based on algorithm type
    if algo_name == "PPO":
        model = algorithm_cls(
            **on_policy_kwargs,
            n_steps=n_steps,
            batch_size=64,
            n_epochs=10,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=f"./{algo_name}_dronenet_tensorboard/"
        )
    elif algo_name == "A2C":
        model = algorithm_cls(
            **on_policy_kwargs,
            n_steps=n_steps,
            gae_lambda=0.95,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5
        )
    elif algo_name == "TD3":
        model = algorithm_cls(
            **off_policy_kwargs,
            batch_size=64,
            tau=0.005,
        )
    elif algo_name == "SAC":
        model = algorithm_cls(
            **off_policy_kwargs,
            batch_size=64,
            tau=0.02,
        )

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
    checkpoint_callback = CheckpointCallback(save_freq=eval_freq, save_path=os.path.join(model_log_dir, 'checkpoints'))

    print(f"Training {algo_name} with lr={current_lr}, gamma={gamma}")
    model.learn(total_timesteps=total_timesteps, callback=[eval_callback, checkpoint_callback])

    # Save model and normalization stats
    model_save_path = f"./models/{algo_name}_lr{lr}_g{gamma}"
    model.save(model_save_path)
    venv.save(f"{model_save_path}_vecnormalize.pkl")

    # Evaluate model
    force, moment, all_rewards, z, theta, sigma, state_errors, total_rewards = collect_actions_and_constraints(make_env(), model, venv, episodes=episodes_to_evaluate)

    return {
        "force": force,
        "moment": moment,
        "rewards": all_rewards,
        "z": z,
        "theta": theta,
        "sigma": sigma,
        "state_errors": state_errors,
        "total_rewards": total_rewards
    }

#---------------------------------------
# Run Experiments for All Algorithms and All Hyperparams
#---------------------------------------

results = {
    # Structure:
    # algo_name: {
    #   (lr, gamma): {... results ...}
    # }
}

for algo_name, algo_cls in algorithms.items():
    results[algo_name] = {}
    for lr in learning_rates:
        for gm in gammas:
            res = train_and_evaluate(algo_cls, algo_name, lr, gm)
            results[algo_name][(lr, gm)] = res

#---------------------------------------
# Plotting Hyperparameter Comparisons
#---------------------------------------
# We will create separate plots for each algorithm, showing the effect of hyperparameters.
# For each algorithm, we make a plot for z, theta, sigma, force, moment.

algo_colors = {
    (1e-4, 0.99): "blue",
    (1e-4, 0.995): "green",
    (5e-4, 0.99): "red",
    (5e-4, 0.995): "purple"
}

def plot_metric_per_algo(algo_name, metric, ylabel, conversion=lambda x: x):
    # metric is one of "z", "theta", "sigma", "force", "moment"
    # conversion is a function to convert the metric if needed (e.g., radians to degrees)
    plt.figure()
    for (lr, gm), data in results[algo_name].items():
        y_data = conversion(data[metric])
        plt.plot(range(len(y_data)), y_data, color=algo_colors[(lr, gm)], label=f"lr={lr}, gamma={gm}")
    plt.xlabel('Timestep')
    plt.ylabel(ylabel)
    plt.title(f'{ylabel} vs Time - {algo_name} (Hyperparam Effects)')
    plt.legend()
    plot_filename = f'{algo_name}_hyperparam_{metric}_vs_time.png'
    plt.savefig(plot_filename)
    plt.close()

# For theta and sigma, we convert from radians to degrees
rad_to_deg = lambda x: np.array(x)*180.0/math.pi

for algo_name in algorithms.keys():
    plot_metric_per_algo(algo_name, "z", "z")
    plot_metric_per_algo(algo_name, "theta", "Theta (deg)", conversion=rad_to_deg)
    plot_metric_per_algo(algo_name, "sigma", "Sigma (deg)", conversion=rad_to_deg)
    plot_metric_per_algo(algo_name, "force", "Force")
    plot_metric_per_algo(algo_name, "moment", "Moment")

print("Training and hyperparameter comparison plots completed successfully.")

# At this point, we have:
# - Trained all algorithms (PPO, A2C, TD3, SAC) with varying hyperparameters (lr, gamma).
# - Created separate plots per algorithm for z, theta, sigma, force, and moment vs time.
# - Improved A2C by adjusting its learning rate and n_steps, which should yield better results.
#
# The intuition for making A2C work better was to provide more stable learning signals and more data per update step,
# as well as decreasing the learning rate to ensure more careful policy updates.
