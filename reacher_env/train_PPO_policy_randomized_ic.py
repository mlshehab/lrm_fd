import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import mujoco
# import torch
# print(torch.cuda.is_available())  # should return True
# print(torch.cuda.get_device_name(0))  # prints your GPU name
# Create the environment with rendering

class ForceRandomizedReacher(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        # Extract custom overrides (if any)
        qpos_override = kwargs.pop("qpos_override", None)
        qvel_override = kwargs.pop("qvel_override", None)

        # Call base reset
        obs, info = self.env.reset(**kwargs)

        model = self.unwrapped.model
        data = self.unwrapped.data

        # Override joint positions
        if qpos_override is not None:
            data.qpos[0] = qpos_override[0]
            data.qpos[1] = qpos_override[1]
        else:
            data.qpos[0] = np.random.uniform(-np.pi, np.pi)
            data.qpos[1] = np.random.uniform(-np.pi, np.pi)

        # Override velocities
        if qvel_override is not None:
            data.qvel[0] = qvel_override[0]
            data.qvel[1] = qvel_override[1]
        else:
            data.qvel[0] = np.random.uniform(-0.5, 0.5)
            data.qvel[1] = np.random.uniform(-0.5, 0.5)

        mujoco.mj_forward(model, data)

        return self.unwrapped._get_obs(), info



env = gym.make('Reacher-v5', max_episode_steps=150)
env = ForceRandomizedReacher(env)  # Wrap it
# You must wrap the environment for vectorized training (no need for 'human' render mode here)
vec_env = make_vec_env(lambda: ForceRandomizedReacher(gym.make('Reacher-v5', max_episode_steps=150)), n_envs=1)

# Create the model
model = PPO("MlpPolicy", vec_env, verbose=1, device="cpu")
# print(next(model.policy.parameters()).device)  # should say cuda:0

# Train the model
model.learn(total_timesteps=10_000_000)
model.save("ppo_reacher_randomized_ic")
# Evaluate the trained agent with rendering
# obs, _ = env.reset()
# done = False

# while not done:
#     action, _ = model.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, info = env.step(action)
#     env.render()
#     done = terminated or truncated
# env.close()