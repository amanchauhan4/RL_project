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
    num_envs = 8  # Increase number of environments to better utilize GPU and CPU

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
        batch_size = trial.suggest_categorical('ppo_batch_size', [1024, 2048, 4096])
        ent_coef = trial.suggest_float('ppo_ent_coef', 0.01, 0.1, log=True)  # Increased range
        # Ensure n_steps is multiple of num_envs
        if n_steps % num_envs != 0:
            n_steps = ((n_steps // num_envs) + 1) * num_envs
        # Ensure (n_steps * num_envs) is divisible by batch_size
        total_batch_size = n_steps * num_envs
        if total_batch_size % batch_size != 0:
            batch_size = total_batch_size // (total_batch_size // batch_size)
        model_params.update({
            'n_steps': n_steps,
            'batch_size': batch_size,
            'ent_coef': ent_coef
        })
        model = PPO(**model_params)
    elif algo == 'SAC':
        batch_size = trial.suggest_categorical('sac_batch_size', [512, 1024, 2048])
        ent_coef = trial.suggest_float('sac_ent_coef', 0.01, 0.1, log=True)  # Corrected key
        model_params.update({
            'batch_size': batch_size,
            'ent_coef': ent_coef
        })
        model = SAC(**model_params)
    elif algo == 'TD3':
        batch_size = trial.suggest_categorical('td3_batch_size', [512, 1024, 2048])
        model_params['batch_size'] = batch_size
        model = TD3(**model_params)
    elif algo == 'DDPG':
        batch_size = trial.suggest_categorical('ddpg_batch_size', [512, 1024, 2048])
        model_params['batch_size'] = batch_size
        model = DDPG(**model_params)

    # Configure logger for Tensorboard
    log_dir = f"./logs/trial_{trial.number}"
    os.makedirs(log_dir, exist_ok=True)
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    model.set_logger(new_logger)

    # Train the model
    try:
        model.learn(total_timesteps=500000)  # Increased total_timesteps for better training
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

    # Create Optuna study with RDB storage for parallel execution
    storage_name = "sqlite:///optuna_study.db"
    study = optuna.create_study(direction='maximize', storage=storage_name, load_if_exists=True)
    study.optimize(optimize_agent, n_trials=20, n_jobs=12)  # Reduced n_trials to 62

    print("Best hyperparameters:", study.best_params)

    # Retrain the model with the best hyperparameters
    best_params = study.best_params
    algo = best_params.pop('algo')  # Remove 'algo' from best_params

    num_envs = 16  # Further increase num_envs for final training

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
        # Ensure n_steps is multiple of num_envs
        if n_steps % num_envs != 0:
            n_steps = ((n_steps // num_envs) + 1) * num_envs
        # Ensure (n_steps * num_envs) is divisible by batch_size
        total_batch_size = n_steps * num_envs
        if total_batch_size % batch_size != 0:
            batch_size = total_batch_size // (total_batch_size // batch_size)
        model_params.update({
            'n_steps': n_steps,
            'batch_size': batch_size,
            'ent_coef': ent_coef
        })
    elif algo == 'SAC':
        batch_size = best_params.pop('sac_batch_size')
        ent_coef = best_params.pop('sac_ent_coef')
        model_params.update({
            'batch_size': batch_size,
            'ent_coef': ent_coef
        })
    elif algo == 'TD3':
        batch_size = best_params.pop('td3_batch_size')
        model_params['batch_size'] = batch_size
    elif algo == 'DDPG':
        batch_size = best_params.pop('ddpg_batch_size')
        model_params['batch_size'] = batch_size

    model_params.update(best_params)

    # Checkpoint directory
    checkpoint_dir = f"./models/{algo}_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Check for existing checkpoints
    checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith(f"{algo}_drone_net_checkpoint")])
    if checkpoint_files:
        # Load the latest checkpoint
        last_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[-1])
        print(f"Loading model from checkpoint {last_checkpoint}")
        model = None
        if algo == 'PPO':
            model = PPO.load(last_checkpoint, env=env)
        elif algo == 'SAC':
            model = SAC.load(last_checkpoint, env=env)
        elif algo == 'TD3':
            model = TD3.load(last_checkpoint, env=env)
        elif algo == 'DDPG':
            model = DDPG.load(last_checkpoint, env=env)
        model.set_env(env)
    else:
        # Initialize the model
        if algo == 'PPO':
            model = PPO(**model_params)
        elif algo == 'SAC':
            model = SAC(**model_params)
        elif algo == 'TD3':
            model = TD3(**model_params)
        elif algo == 'DDPG':
            model = DDPG(**model_params)

    # Configure logger for Tensorboard
    log_dir = f"./logs/final_training_{algo}"
    os.makedirs(log_dir, exist_ok=True)
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    model.set_logger(new_logger)

    # Create a callback that saves the model every n steps
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,  # Save every 100,000 steps
        save_path=checkpoint_dir,
        name_prefix=f"{algo}_drone_net_checkpoint"
    )

    # Determine remaining timesteps
    total_timesteps = 5000000
    if model.num_timesteps > 0:
        remaining_timesteps = total_timesteps - model.num_timesteps
        if remaining_timesteps <= 0:
            print("Training already completed.")
            remaining_timesteps = 0
    else:
        remaining_timesteps = total_timesteps

    if remaining_timesteps > 0:
        # Train the agent with the checkpoint callback
        model.learn(total_timesteps=remaining_timesteps, callback=checkpoint_callback)
        # Save the final model
        model_path = f"./models/{algo}_drone_net"
        model.save(model_path)
        print(f"Training complete. Model saved to {model_path}.")
    else:
        print("No training needed.")

    # Clean up
    env.close()
