import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import glob
import subprocess
from stable_baselines3 import PPO, A2C, TD3, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env_v1 import DroneNet

# Choose which trained model to load
MODEL_NAME = "PPO"  # or "A2C", "TD3", "SAC"
MODEL_PATH = f"./models/{MODEL_NAME}_lr0.0001_g0.99"  # adjust if you saved differently

# Load environment and VecNormalize
env = DroneNet()
venv = DummyVecEnv([lambda: env])
venv = VecNormalize.load(MODEL_PATH + "_vecnormalize.pkl", venv)
venv.training = False
venv.norm_reward = False

# Load model
if MODEL_NAME == "PPO":
    model = PPO.load(MODEL_PATH, env=venv)
elif MODEL_NAME == "A2C":
    model = A2C.load(MODEL_PATH, env=venv)
elif MODEL_NAME == "TD3":
    model = TD3.load(MODEL_PATH, env=venv)
elif MODEL_NAME == "SAC":
    model = SAC.load(MODEL_PATH, env=venv)
else:
    raise ValueError("Unknown MODEL_NAME")

# Create a folder for frames
os.makedirs("video_frames", exist_ok=True)
for f in glob.glob("video_frames/*.png"):
    os.remove(f)  # clean previous frames

obs, info = env.reset()
done = False
step = 0
episode_reward = 0

# Collect force for display
force_values = []
moment_values = []
z_values = []
theta_values = []
sigma_values = []

while not done:
    obs_norm = venv.normalize_obs(obs)
    action, _ = model.predict(obs_norm, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    z, theta, sigma, z_dot, theta_dot, sigma_dot = obs
    force = (action[0]*0.5 + 0.5)*env.Fmax  # reconstruct actual force from action
    moment = action[1]*env.Mmax

    force_values.append(force)
    moment_values.append(moment)
    z_values.append(z)
    theta_values.append(math.degrees(theta))
    sigma_values.append(math.degrees(sigma))

    episode_reward += reward

    # Render the environment to a figure
    # We'll save the figure with overlays of text
    fig, ax = plt.subplots(figsize=(5,5))
    ax.set_xlim(-5,5)
    ax.set_ylim(-3,1)
    ax.set_title('DroneNet Simulation')
    ax.set_xlabel('Horizontal (m)')
    ax.set_ylabel('Vertical (m)')

    # Draw the net
    half_length = env.l/2.0
    x1 = -half_length * math.cos(theta)
    y1 = -half_length * math.sin(theta) + z
    x2 = half_length * math.cos(theta)
    y2 = half_length * math.sin(theta) + z

    # Draw net line
    ax.plot([x1, x2], [y1, y2], color='black', linewidth=2)
    # Drones along the net
    for i in range(int(env.n)):
        frac = i/(env.n-1)
        xd = x1 + frac*(x2 - x1)
        yd = y1 + frac*(y2 - y1)
        ax.add_patch(plt.Circle((xd, yd), radius=0.1, color='blue'))

    # Object
    obj_x = 0.0
    obj_y = z
    ax.add_patch(plt.Circle((obj_x, obj_y), radius=env.r, color='red'))

    # Add text annotations of states and force
    ax.text(-4.5, 0.9, f'Step: {step}', fontsize=10)
    ax.text(-4.5, 0.7, f'Z: {z:.2f} m', fontsize=10)
    ax.text(-4.5, 0.5, f'Theta: {theta_values[-1]:.2f} deg', fontsize=10)
    ax.text(-4.5, 0.3, f'Sigma: {sigma_values[-1]:.2f} deg', fontsize=10)
    ax.text(-4.5, 0.1, f'Force: {force:.2f} N', fontsize=10)
    ax.text(-4.5, -0.1, f'Moment: {moment:.2f} Nm', fontsize=10)
    ax.text(-4.5, -0.3, f'Reward: {reward:.2f}', fontsize=10)

    # Save frame
    frame_path = f"video_frames/frame_{step:04d}.png"
    plt.savefig(frame_path)
    plt.close(fig)

    step += 1

print(f"Episode finished after {step} steps with total reward {episode_reward}")

# OPTIONAL: Combine frames into a video using ffmpeg (if installed)
# Creates a video at 20 fps
video_name = f"drone_catching_{MODEL_NAME}.mp4"
cmd = f"ffmpeg -r 20 -i video_frames/frame_%04d.png -pix_fmt yuv420p -y {video_name}"
subprocess.run(cmd, shell=True, check=True)

print(f"Video saved as {video_name}")
