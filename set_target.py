import gymnasium as gym
import numpy as np
env = gym.make("Reacher-v5", render_mode="human",max_episode_steps=100)
import time
# Reset with a custom target
target = [10, 10]  # x, y between -1 and 1
# Forcefully set a new target (not recommended, but possible)

import mujoco

# After env.reset(), modify the target's qpos (joint positions)
def set_target_position(env, x, y):
    model = env.unwrapped.model
    data = env.unwrapped.data
    
    # Find the joint IDs for target_x and target_y
    target_x_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_x")
    target_y_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_y")
    
    # Set the target's position
    data.qpos[target_x_id] = x
    data.qpos[target_y_id] = y

# Usage:
set_target_position(env, -0.15, 0.15)  # Moves the target to (0.15, -0.15)

# if hasattr(env.unwrapped, "target"):
#     env.unwrapped.target = np.array([-0.2, 0.6], dtype=np.float32)
observation, info = env.reset()

curr_counter = 0
for _ in range(1000):
    time.sleep(0.2)
    curr_counter += 1
    action = env.action_space.sample()  # Replace with your policy
    observation, reward, terminated, truncated, info = env.step(action)
    if curr_counter == 50:
        set_target_position(env, 0.15,0.15) 
    
    if terminated or truncated:
        curr_counter = 0
        # Reset with a new target
        # new_target = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
        new_target = target
        observation, info = env.reset()
        set_target_position(env, -0.15, 0.15)  # Moves the target to (0.15, -0.15)

env.close()