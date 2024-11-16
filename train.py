# train.py

from DroneNet import DroneNet
import numpy as np
import optuna
from stable_baselines3 import PPO, SAC, TD3, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import torch

def optimize_agent(trial):
    # Hyperparameters to tune
    algo = trial.suggest_categorical('algo', ['PPO', 'SAC', 'TD3', 'DDPG'])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float('gamma', 0.95, 0.9999)
    n_steps = trial.suggest_categorical('n_steps', [1024, 2048, 4096])
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    ent_coef = trial.suggest_float('ent_coef', 1e-8, 1e-2, log=True)

    # Initialize the environment
    env = DummyVecEnv([lambda: DroneNet()])

    # Choose algorithm based on hyperparameter
    if algo == 'PPO':
        model = PPO(
            'MlpPolicy',
            env,
            learning_rate=learning_rate,
            gamma=gamma,
            n_steps=n_steps,
            batch_size=batch_size,
            ent_coef=ent_coef,
            verbose=0,
            device='cuda' if torch.cuda.is_available() else 'cpu'  # Explicitly set device
        )
    elif algo == 'SAC':
        model = SAC(
            'MlpPolicy',
            env,
            learning_rate=learning_rate,
            gamma=gamma,
            batch_size=batch_size,
            ent_coef=ent_coef,
            verbose=0,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    elif algo == 'TD3':
        model = TD3(
            'MlpPolicy',
            env,
            learning_rate=learning_rate,
            gamma=gamma,
            batch_size=batch_size,
            verbose=0,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    elif algo == 'DDPG':
        model = DDPG(
            'MlpPolicy',
            env,
            learning_rate=learning_rate,
            gamma=gamma,
            batch_size=batch_size,
            verbose=0,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

    # Train the model
    try:
        model.learn(total_timesteps=50000)
    except Exception as e:
        print(f"Training failed for trial with parameters: {trial.params}")
        raise e

    # Evaluate the model
    mean_reward = evaluate_model(model, num_episodes=5)

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
    study.optimize(optimize_agent, n_trials=20)

    print("Best hyperparameters:", study.best_params)

    # Retrain the model with the best hyperparameters
    best_params = study.best_params
    algo = best_params.pop('algo')  # Remove 'algo' from best_params

    env = DummyVecEnv([lambda: DroneNet()])

    # Choose algorithm based on the best hyperparameter
    if algo == 'PPO':
        model = PPO(
            'MlpPolicy',
            env,
            **best_params,
            verbose=1,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    elif algo == 'SAC':
        model = SAC(
            'MlpPolicy',
            env,
            **best_params,
            verbose=1,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    elif algo == 'TD3':
        model = TD3(
            'MlpPolicy',
            env,
            **best_params,
            verbose=1,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    elif algo == 'DDPG':
        model = DDPG(
            'MlpPolicy',
            env,
            **best_params,
            verbose=1,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

    # Train the agent
    model.learn(total_timesteps=500000)

    # Save the trained model
    model_path = f"./models/{algo}_drone_net"
    model.save(model_path)

    print(f"Training complete. Model saved to {model_path}.")
