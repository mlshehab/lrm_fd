import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import mujoco
from gymnasium.spaces import MultiDiscrete
# import os
# print("Number of logical CPUs:", os.cpu_count())
# class DiscreteReacherActionWrapper(gym.ActionWrapper):
#     def __init__(self, env):
#         super().__init__(env)
#         # 5 discrete actions for each joint
#         self.action_space = MultiDiscrete([5, 5])
#         # Define the mapping from discrete to continuous values
#         self.action_values = np.linspace(-1.0, 1.0, 5)  # [-1.0, -0.5, 0.0, 0.5, 1.0]

#     def action(self, action):
#         # Map [0,1,2,3,4] to [-1.0,-0.5,0.0,0.5,1.0] for each joint
#         mapped_action = self.action_values[np.array(action)]
#         return mapped_action.astype(np.float32)
    

class DiscreteReacherActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # 5 discrete actions for each joint
        self.action_space = MultiDiscrete([3, 3])
        # Define the mapping from discrete to continuous values
        self.action_values = np.linspace(-1.0, 1.0, 3)  # [-1.0, -0.5, 0.0, 0.5, 1.0]

    def action(self, action):
        # Map [0,1,2,3,4] to [-1.0,-0.5,0.0,0.5,1.0] for each joint
        mapped_action = self.action_values[np.array(action)]
        return mapped_action.astype(np.float32)
    
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




# # You must wrap the environment for vectorized training (no need for 'human' render mode here)
# vec_env = make_vec_env(lambda: DiscreteReacherActionWrapper(ForceRandomizedReacher(gym.make('Reacher-v5', max_episode_steps=250))), n_envs=8)

# # Create the model
# model = PPO("MlpPolicy", vec_env, verbose=1, device="cpu")
# # print(next(model.policy.parameters()).device)  # should say cuda:0

# # Train the model
# model.learn(total_timesteps=10_000_000)
# model.save("ppo_reacher_randomized_ic_discrete")

if __name__ == "__main__":

    env = gym.make('Reacher-v5', render_mode='human', xml_file="./reacher.xml", max_episode_steps=150)
    env = ForceRandomizedReacher(env)  # Wrap it
    env = DiscreteReacherActionWrapper(env)

    # Evaluate the trained agent with rendering
    model = PPO.load("ppo_reacher_randomized_ic_discrete_5_actions", device="cpu")

    obs, _ = env.reset()
    done = False

    total_reward = 0
    steps = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        print("action", action)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        env.render()
        done = terminated or truncated

    print(f"Average reward per step: {total_reward:.3f}")
    env.close()