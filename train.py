import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import math
import matplotlib.pyplot as plt
from env_v1 import DroneNet

# Instantiate environment
env = DroneNet()

venv = DummyVecEnv([lambda: env])
venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.)

model = PPO(
    "MlpPolicy",
    venv,
    verbose=1,
    learning_rate=1e-4,
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

eval_callback = EvalCallback(
    venv,
    best_model_save_path='./logs/best_model',
    log_path='./logs/',
    eval_freq=5000,
    deterministic=True,
    render=False
)

checkpoint_callback = CheckpointCallback(save_freq=5000, save_path='./logs/checkpoints/')

total_timesteps = 200000
model.learn(total_timesteps=total_timesteps, callback=[eval_callback, checkpoint_callback])

# Save model and VecNormalize stats
model.save("./models/PPO_drone_net")  # match the evaluation default name
venv.save("./models/vecnormalize.pkl")

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

    # For evaluation, we need the non-vec environment or to reset venv properly
    # We'll just reuse env directly (unwrapped)
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        step = 0
        episode_reward = 0
        while not done:
            # Normalize observation using venv's internal functions
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
            error = np.array([obs[0]-env.z_final, obs[1]-env.theta_final, obs[2]-env.sigma_final,
                              obs[3]-env.z_dot_final, obs[4]-env.theta_dot_final, obs[5]-env.sigma_dot_final])
            dist = np.linalg.norm(error)
            state_errors.append(dist)
            time_steps.append(step)
            episode_reward += reward

            step += 1

        total_rewards.append(episode_reward)

    return force, moment, all_rewards, z, theta, sigma, constraints_violated, time_steps, total_rewards, state_errors

force, moment, all_rewards, z, theta, sigma, constraints_violated, time_steps, total_rewards, state_errors = collect_actions_and_constraints(env, model, episodes=1)

plt.figure()
plt.plot(total_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Reward per Episode')
plt.savefig('total_reward_per_episode.png')

plt.figure()
plt.plot(state_errors)
plt.xlabel('Time Step')
plt.ylabel('State Error')
plt.title('State Error Over Time')
plt.savefig('state_error_over_time.png')

plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(range(len(force)), force, marker='o', linestyle='-', color='b')
plt.xlabel('Timestep')
plt.ylabel('Force')
plt.title('Force Over Time')

plt.subplot(3, 1, 2)
plt.plot(range(len(moment)), moment, marker='o', linestyle='-', color='g')
plt.xlabel('Timestep')
plt.ylabel('Moment')
plt.title('Moment Over Time')

plt.subplot(3, 1, 3)
plt.plot(range(len(all_rewards)), all_rewards, marker='o', linestyle='-', color='r')
plt.xlabel('Timestep')
plt.ylabel('Reward')
plt.title('Reward Over Time')
plt.tight_layout()
plt.savefig('actions_vs_time.png')

plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(range(len(z)), z, marker='o', linestyle='-', color='b')
plt.xlabel('Timestep')
plt.ylabel('z')
plt.title('Vertical Position Over Time')

plt.subplot(3, 1, 2)
plt.plot(range(len(theta)), np.array(theta)*180.0/math.pi, marker='o', linestyle='-', color='b')
plt.xlabel('Timestep')
plt.ylabel('Theta (deg)')
plt.title('Theta Over Time')

plt.subplot(3, 1, 3)
plt.plot(range(len(sigma)), np.array(sigma)*180.0/math.pi, marker='o', linestyle='-', color='b')
plt.xlabel('Timestep')
plt.ylabel('Sigma (deg)')
plt.title('Sigma Over Time')
plt.tight_layout()
plt.savefig('states_vs_time.png')
