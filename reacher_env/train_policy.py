import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import mujoco
from gymnasium.spaces import MultiDiscrete
import os
 

class DiscreteReacherActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.discrete_vals = np.array([-1.0,-0.5,0.0,0.5,1.0])
        self.action_space = MultiDiscrete([5, 5])

    def action(self, action):
        # Map [0, 1, 2] to [-1, 0, +1]
        # mapped_action = np.array(action) - 1/
        return self.discrete_vals[np.array(action)]
    

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



if __name__ == "__main__":
    # You must wrap the environment for vectorized training (no need for 'human' render mode here)
    vec_env = make_vec_env(lambda: DiscreteReacherActionWrapper(ForceRandomizedReacher(gym.make('Reacher-v5', max_episode_steps=150))), n_envs=50)

    # Create the model
    model = PPO("MlpPolicy", vec_env, verbose=1, device="cpu",tensorboard_log="./ppo_reacher_tensorboard/")
    # print(next(model.policy.parameters()).device)  # should say cuda:0

    # Train the model
    model.learn(total_timesteps=20_000_000)
    model.save("ppo_reacher_randomized_ic_discrete_5_actions")




