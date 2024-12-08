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

# Common hyperparameters
learning_rate = 1e-4
gamma = 0.99
total_timesteps = 200000
eval_freq = 5000
episodes_to_evaluate = 1

os.makedirs("./models", exist_ok=True)
os.makedirs("./logs", exist_ok=True)

#---------------------------------------
# Utility Functions
#---------------------------------------

def make_env():
    return DroneNet()

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
            # Normalize observation to match training conditions
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

def train_and_evaluate(algorithm_cls, algo_name):
    """
    Train and evaluate a given algorithm with similar parameters.
    Returns the evaluation results.
    """
    env = make_env()
    venv = DummyVecEnv([make_env])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.)

    # Common kwargs for on-policy algorithms (PPO, A2C)
    on_policy_kwargs = {
        "policy": "MlpPolicy",
        "env": venv,
        "verbose": 1,
        "learning_rate": learning_rate,
        "gamma": gamma,
    }

    # Common kwargs for off-policy algorithms (TD3, SAC)
    off_policy_kwargs = {
        "policy": "MlpPolicy",
        "env": venv,
        "verbose": 1,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "learning_starts": 1000,
        "buffer_size": 100000,
    }

    # Construct model based on algorithm type
    if algo_name == "PPO":
        model = algorithm_cls(
            **on_policy_kwargs,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log="./ppo_dronenet_tensorboard/"
        )
    elif algo_name == "A2C":
        # A2C does not use n_epochs, but we can set n_steps and batch_size similarly
        model = algorithm_cls(
            **on_policy_kwargs,
            n_steps=2048,
            gae_lambda=0.95,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5
        )
    elif algo_name == "TD3":
        # TD3 specific defaults
        model = algorithm_cls(
            **off_policy_kwargs,
            batch_size=64,
            tau=0.005,
        )
    elif algo_name == "SAC":
        # SAC specific defaults
        model = algorithm_cls(
            **off_policy_kwargs,
            batch_size=64,
            tau=0.02,
        )

    model_log_dir = f'./logs/{algo_name}'
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

    print(f"Training {algo_name} with lr={learning_rate}, gamma={gamma}")
    model.learn(total_timesteps=total_timesteps, callback=[eval_callback, checkpoint_callback])

    # Save model and normalization stats
    model_save_path = f"./models/{algo_name}_lr{learning_rate}_g{gamma}"
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
# Run Experiments for All Algorithms
#---------------------------------------

results = {}
for algo_name, algo_cls in algorithms.items():
    results[algo_name] = train_and_evaluate(algo_cls, algo_name)

#---------------------------------------
# Plotting Comparison
#---------------------------------------

algo_colors = {
    "PPO": "blue",
    "A2C": "green",
    "TD3": "red",
    "SAC": "purple"
}

# Plot z vs time comparison
plt.figure()
for algo_name in results:
    z = results[algo_name]["z"]
    plt.plot(range(len(z)), z, color=algo_colors[algo_name], label=algo_name)
plt.xlabel('Timestep')
plt.ylabel('z')
plt.title('z vs Time - All Algorithms')
plt.legend()
plt.savefig('comparison_z_vs_time_all_algos.png')
plt.close()

# Plot theta vs time comparison (in degrees)
plt.figure()
for algo_name in results:
    theta_deg = np.array(results[algo_name]["theta"]) * 180.0 / math.pi
    plt.plot(range(len(theta_deg)), theta_deg, color=algo_colors[algo_name], label=algo_name)
plt.xlabel('Timestep')
plt.ylabel('Theta (deg)')
plt.title('Theta vs Time - All Algorithms')
plt.legend()
plt.savefig('comparison_theta_vs_time_all_algos.png')
plt.close()

# Plot sigma vs time comparison (in degrees)
plt.figure()
for algo_name in results:
    sigma_deg = np.array(results[algo_name]["sigma"]) * 180.0 / math.pi
    plt.plot(range(len(sigma_deg)), sigma_deg, color=algo_colors[algo_name], label=algo_name)
plt.xlabel('Timestep')
plt.ylabel('Sigma (deg)')
plt.title('Sigma vs Time - All Algorithms')
plt.legend()
plt.savefig('comparison_sigma_vs_time_all_algos.png')
plt.close()

# Plot force vs time
plt.figure()
for algo_name in results:
    force = results[algo_name]["force"]
    plt.plot(range(len(force)), force, color=algo_colors[algo_name], label=algo_name)
plt.xlabel('Timestep')
plt.ylabel('Force')
plt.title('Force vs Time - All Algorithms')
plt.legend()
plt.savefig('comparison_force_vs_time_all_algos.png')
plt.close()

# Plot moment vs time
plt.figure()
for algo_name in results:
    moment = results[algo_name]["moment"]
    plt.plot(range(len(moment)), moment, color=algo_colors[algo_name], label=algo_name)
plt.xlabel('Timestep')
plt.ylabel('Moment')
plt.title('Moment vs Time - All Algorithms')
plt.legend()
plt.savefig('comparison_moment_vs_time_all_algos.png')
plt.close()

# Plot reward vs time
plt.figure()
for algo_name in results:
    rew = results[algo_name]["rewards"]
    plt.plot(range(len(rew)), rew, color=algo_colors[algo_name], label=algo_name)
plt.xlabel('Timestep')
plt.ylabel('Reward')
plt.title('Reward vs Time - All Algorithms')
plt.legend()
plt.savefig('comparison_reward_vs_time_all_algos.png')
plt.close()

# Plot state error over time
plt.figure()
for algo_name in results:
    errs = results[algo_name]["state_errors"]
    plt.plot(range(len(errs)), errs, color=algo_colors[algo_name], label=algo_name)
plt.xlabel('Timestep')
plt.ylabel('State Error')
plt.title('State Error Over Time - All Algorithms')
plt.legend()
plt.savefig('comparison_state_error_over_time_all_algos.png')
plt.close()

# Plot total rewards per episode (though we have only 1 episode of evaluation)
plt.figure()
for algo_name in results:
    total_rewards = results[algo_name]["total_rewards"]
    plt.plot(range(len(total_rewards)), total_rewards, marker='o', color=algo_colors[algo_name], label=algo_name)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Reward per Episode - All Algorithms')
plt.legend()
plt.savefig('comparison_total_reward_per_episode_all_algos.png')
plt.close()

print("Training and comparison plots completed successfully for all algorithms.")
