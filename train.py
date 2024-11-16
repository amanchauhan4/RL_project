# train.py

from DroneNet import DroneNet
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# Create the environment
env = DroneNet(n_drones=5)

# Create a callback to save the model periodically
checkpoint_callback = CheckpointCallback(save_freq=50000, save_path='./logs/',
                                         name_prefix='ppo_drone_net')

# Define and Train the agent
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=1e-4,
    n_steps=2048,
    batch_size=64,
    gae_lambda=0.95,
    gamma=0.99,
    ent_coef=0.01,
    clip_range=0.2,
    n_epochs=10,
    tensorboard_log="./ppo_drone_net_tensorboard/"
)

# Train the agent
model.learn(total_timesteps=3000000, callback=checkpoint_callback)

# Save the trained model
model.save("ppo_drone_net_enhanced")
