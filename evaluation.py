# evaluate.py

import time
from DroneNet import DroneNet
from stable_baselines3 import PPO, SAC, TD3, DDPG
import matplotlib.pyplot as plt
import os

def evaluate_model(model, env):
    obs, _ = env.reset()
    done = False
    start_time = time.time()
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        env.render()
        time.sleep(0.05)
        if time.time() - start_time > 20:
            break
    env.close()

if __name__ == '__main__':
    # Initialize environment
    env = DroneNet()

    # Choose the algorithm to load
    algo = 'PPO'  # Change this based on the trained model

    # Define the path to the trained model
    model_path = f'./models/{algo}_drone_net.zip'

    if not os.path.exists(model_path):
        print(f"Trained model not found at {model_path}. Please train the model first.")
        exit(1)

    # Load the trained model
    if algo == 'PPO':
        model = PPO.load(model_path, env=env)
    elif algo == 'SAC':
        model = SAC.load(model_path, env=env)
    elif algo == 'TD3':
        model = TD3.load(model_path, env=env)
    elif algo == 'DDPG':
        model = DDPG.load(model_path, env=env)
    else:
        print(f"Algorithm '{algo}' is not supported.")
        exit(1)

    # Evaluate the model by rendering a single episode
    evaluate_model(model, env)
