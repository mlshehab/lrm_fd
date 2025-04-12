import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
from gymnasium.wrappers import TimeLimit
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

# Load the environment with rendering
env = gym.make("Reacher-v5", render_mode="human", max_episode_steps=150, xml_file="./reacher.xml")

# Reset the environment
obs, _ = env.reset()


# Load model
model = PPO.load("ppo_reacher", device="cpu")

target_green = [0.12, -0.1]
target_red = [0.12, 0.1]
target_yellow = [-0.12, 0.1]

target_dict = {"green": target_green, "red": target_red, "yellow": target_yellow}

targets_goals = ["green", "red", "yellow"]



current_target = target_dict[targets_goals[0]]
set_target_position(env, current_target[0], current_target[1])

for _ in range(500):
    
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    env.render()
    # Print XY coordinates of end-effector (fingertip)
    # Get end-effector position from environment's state
    fingertip_xy = env.unwrapped.get_body_com("fingertip")[:2]

    distance_to_target = np.sqrt(obs[8]**2 + obs[9]**2)

    if distance_to_target < 0.02:
        print("Target reached")
        targets_goals.append(targets_goals.pop(0))
        print(f"Targets remaining: {targets_goals}")
        current_target = target_dict[targets_goals[0]]
        set_target_position(env, current_target[0], current_target[1])



    # print(f"End-Effector XY coordinates: {fingertip_xy.shape}")
    # time.sleep(1)

    if terminated or truncated:
        
        obs, info = env.reset()
        targets_goals = ["green", "red", "yellow"]
        current_target = target_dict[targets_goals[0]]
        set_target_position(env, current_target[0], current_target[1])


# Close the environment
env.close()