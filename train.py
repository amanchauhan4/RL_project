# train.py

from DroneNet import DroneNet
import numpy as np
import optuna
from stable_baselines3 import PPO, SAC, TD3, DDPG
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback
import os
import torch

def make_env():
    return DroneNet()

def optimize_agent(trial):
    # Hyperparameters to tune
    algo = trial.suggest_categorical('algo', ['PPO', 'SAC', 'TD3', 'DDPG'])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float('gamma', 0.95, 0.9999)
    num_envs = 8  # Number of parallel environments

    # Initialize the environment
    env = SubprocVecEnv([make_env for _ in range(num_envs)])

    # Set algorithm-specific hyperparameters
    model_params = {
        'policy': 'MlpPolicy',
        'env': env,
        'learning_rate': learning_rate,
        'gamma': gamma,
        'verbose': 0,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    if algo == 'PPO':
        n_steps = trial.suggest_categorical('ppo_n_steps', [2048, 4096, 8192])
        batch_size = trial.suggest_categorical('ppo_batch_size', [64, 128, 256])
        ent_coef = trial.suggest_float('ppo_ent_coef', 0.0, 0.01, log=True)
        model_params.update({
            'n_steps': n_steps,
            'batch_size': batch_size,
            'ent_coef': ent_coef
        })
        model = PPO(**model_params)
    elif algo == 'SAC':
        batch_size = trial.suggest_categorical('sac_batch_size', [64, 128, 256])
        ent_coef = trial.suggest_float('sac_ent_coef', 0.0, 0.01, log=True)
        model_params.update({
            'batch_size': batch_size,
            'ent_coef': ent_coef
        })
        model = SAC(**model_params)
    elif algo == 'TD3':
        batch_size = trial.suggest_categorical('td3_batch_size', [64, 128, 256])
        model_params.update({
            'batch_size': batch_size,
        })
        model = TD3(**model_params)
    elif algo == 'DDPG':
        batch_size = trial.suggest_categorical('ddpg_batch_size', [64, 128, 256])
        model_params.update({
            'batch_size': batch_size,
        })
        model = DDPG(**model_params)

    # Configure logger for Tensorboard
    log_dir = f"./logs/trial_{trial.number}"
    os.makedirs(log_dir, exist_ok=True)
    new_logger = configure(log_dir, ["tensorboard"])
    model.set_logger(new_logger)

    # Train the model
    try:
        model.learn(total_timesteps=500000)
    except Exception as e:
        print(f"Training failed for trial with parameters: {trial.params}")
        raise e

    # Evaluate the model
    mean_reward = evaluate_model(model, num_episodes=5)

    # Clean up
    env.close()
    del model
    del env

    return mean_reward

def evaluate_model(model, num_episodes=5):
    # Use a non-vectorized environment for evaluation
    eval_env = DroneNet()
    all_rewards = []
    for episode in range(num_episodes):
        obs, _ = eval_env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            total_reward += reward
            done = terminated or truncated
        all_rewards.append(total_reward)
    eval_env.close()
    return np.mean(all_rewards)

if __name__ == '__main__':
    # Create directories to save models and logs
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)

    # Create Optuna study
    study = optuna.create_study(direction='maximize')
    study.optimize(optimize_agent, n_trials=150)  # Adjust n_trials as needed

    print("Best hyperparameters:", study.best_params)

    # Retrain the model with the best hyperparameters
    best_params = study.best_params
    algo = best_params.pop('algo')  # Remove 'algo' from best_params

    num_envs = 8  # Number of parallel environments
    env = SubprocVecEnv([make_env for _ in range(num_envs)])

    # Set up model_params
    model_params = {
        'policy': 'MlpPolicy',
        'env': env,
        'verbose': 1,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    if algo == 'PPO':
        n_steps = best_params.pop('ppo_n_steps')
        batch_size = best_params.pop('ppo_batch_size')
        ent_coef = best_params.pop('ppo_ent_coef')
        model_params.update({
            'n_steps': n_steps,
            'batch_size': batch_size,
            'ent_coef': ent_coef
        })
        model = PPO(**model_params)
    elif algo == 'SAC':
        batch_size = best_params.pop('sac_batch_size')
        ent_coef = best_params.pop('sac_ent_coef')
        model_params.update({
            'batch_size': batch_size,
            'ent_coef': ent_coef
        })
        model = SAC(**model_params)
    elif algo == 'TD3':
        batch_size = best_params.pop('td3_batch_size')
        model_params.update({
            'batch_size': batch_size,
        })
        model = TD3(**model_params)
    elif algo == 'DDPG':
        batch_size = best_params.pop('ddpg_batch_size')
        model_params.update({
            'batch_size': batch_size,
        })
        model = DDPG(**model_params)

    model_params.update(best_params)

    # Configure logger for Tensorboard
    log_dir = f"./logs/final_training_{algo}"
    os.makedirs(log_dir, exist_ok=True)
    new_logger = configure(log_dir, ["tensorboard"])
    model.set_logger(new_logger)

    # Callback for checkpointing
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=f"./models/{algo}_checkpoints",
        name_prefix=f"{algo}_drone_net_checkpoint"
    )

    # Train the agent
    model.learn(total_timesteps=6000000, callback=checkpoint_callback)

    # Save the final model
    model_path = f"./models/{algo}_drone_net"
    model.save(model_path)
    print(f"Training complete. Model saved to {model_path}.")

    # Clean up
    env.close()
