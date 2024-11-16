# evaluation.py

import numpy as np
import matplotlib.pyplot as plt
from DroneNet import DroneNet
from stable_baselines3 import PPO

def evaluate_policy(env, model, episodes=10, render=False):
    total_rewards = []
    success_count = 0
    capture_times = []
    energy_consumptions = []

    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        energy = 0.0

        while not done:
            # Get action from model
            action, _states = model.predict(obs, deterministic=True)
            action = action.flatten()

            # Accumulate energy consumption
            energy += np.sum(np.abs(action)) * env.dt

            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

            if render:
                env.render(slow_motion=True)

            steps += 1

            # Check for successful capture
            if terminated and reward >= 5000.0:
                success_count += 1
                capture_times.append(steps * env.dt)

        total_rewards.append(total_reward)
        energy_consumptions.append(energy)

        print(f"Episode {episode + 1} Summary:")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Steps: {steps}")
        print(f"  Energy Consumption: {energy:.2f} J")
        print("=" * 50)

    avg_reward = np.mean(total_rewards)
    success_rate = (success_count / episodes) * 100
    avg_capture_time = np.mean(capture_times) if capture_times else None
    avg_energy = np.mean(energy_consumptions)

    print(f"Average Reward over {episodes} episodes: {avg_reward:.2f}")
    print(f"Success Rate: {success_rate:.1f}%")
    if avg_capture_time:
        print(f"Average Capture Time: {avg_capture_time:.2f} s")
    else:
        print("Average Capture Time: N/A")
    print(f"Average Energy Consumption: {avg_energy:.2f} J")

    if render:
        # Keep the window open after the evaluation concludes
        print("Press any key to close the rendering window.")
        plt.show()

if __name__ == "__main__":
    env = DroneNet(n_drones=5)

    # Load the trained model
    model = PPO.load("ppo_drone_net_enhanced")

    evaluate_policy(env, model, episodes=10, render=True)
