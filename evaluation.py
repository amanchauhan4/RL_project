# evaluate.py

import time
import argparse
import os
from stable_baselines3 import PPO, SAC, TD3, DDPG
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from env_v1 import DroneNet

def evaluate_model(model, env, vec_env=False, max_duration=20, sleep_time=0.05):
    """
    Evaluate the model on a single episode.

    If vec_env=True, env is a vectorized environment and env.reset() returns obs only.
    If vec_env=False, env.reset() returns (obs, info).
    """
    if vec_env:
        obs = env.reset()
    else:
        obs, info = env.reset()

    done = False
    start_time = time.time()

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        # If vectorized env, step returns obs only, else (obs, reward, done, info)
        if vec_env:
            obs, reward, done, info = env.step(action)
            # Some vec envs might not have "terminated" and "truncated", but combined done.
        else:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        env.render()

        if time.time() - start_time > max_duration:
            break
        time.sleep(sleep_time)

    env.close()

def load_model(algo, model_path, env):
    if algo == 'PPO':
        model = PPO.load(model_path, env=env)
    elif algo == 'SAC':
        model = SAC.load(model_path, env=env)
    elif algo == 'TD3':
        model = TD3.load(model_path, env=env)
    elif algo == 'DDPG':
        model = DDPG.load(model_path, env=env)
    else:
        raise ValueError(f"Algorithm '{algo}' is not supported.")
    return model

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained RL model on the DroneNet environment.")
    parser.add_argument('--algo', type=str, default='PPO', help='RL algorithm used (PPO, SAC, TD3, DDPG).')
    parser.add_argument('--model-dir', type=str, default='./models', help='Directory where the model is stored.')
    parser.add_argument('--model-name', type=str, default='PPO_drone_net.zip', help='Name of the model file.')
    parser.add_argument('--no-normalize', action='store_true', help='If set, do not attempt to load VecNormalize stats.')

    args = parser.parse_args()

    algo = args.algo
    model_path = os.path.join(args.model_dir, args.model_name)
    norm_path = os.path.join(args.model_dir, 'vecnormalize.pkl')

    if not os.path.exists(model_path):
        print(f"Trained model not found at {model_path}. Please train the model first.")
        return

    if not args.no_normalize and os.path.exists(norm_path):
        # VecEnv
        raw_env = DroneNet()
        venv = DummyVecEnv([lambda: raw_env])
        venv = VecNormalize.load(norm_path, venv)
        venv.training = False
        venv.norm_reward = False
        model = load_model(algo, model_path, venv)
        # Here vec_env=True since we're using a VecNormalize environment
        evaluate_model(model, venv, vec_env=True)

    else:
        # No normalization
        env = DroneNet()
        model = load_model(algo, model_path, env)
        # vec_env=False in this case
        evaluate_model(model, env, vec_env=False)

if __name__ == '__main__':
    main()
