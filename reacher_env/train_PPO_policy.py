import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# import torch
# print(torch.cuda.is_available())  # should return True
# print(torch.cuda.get_device_name(0))  # prints your GPU name
# Create the environment with rendering
env = gym.make('Reacher-v5', render_mode="human")

# You must wrap the environment for vectorized training (no need for 'human' render mode here)
vec_env = make_vec_env(lambda: gym.make('Reacher-v5', max_episode_steps=150), n_envs=1)

# Create the model
model = PPO("MlpPolicy", vec_env, verbose=1, device="cpu")
# print(next(model.policy.parameters()).device)  # should say cuda:0

# Train the model
model.learn(total_timesteps=1_000_000)
model.save("ppo_reacher_apple")
# Evaluate the trained agent with rendering
# obs, _ = env.reset()
# done = False

# while not done:
#     action, _ = model.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, info = env.step(action)
#     env.render()
#     done = terminated or truncated
# env.close()