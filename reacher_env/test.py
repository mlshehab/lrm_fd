import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np

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

def randomize_arm_position(env):
    model = env.unwrapped.model
    data = env.unwrapped.data

    qpos = data.qpos.copy()
    qvel = data.qvel.copy()

    qpos[0:2] = np.random.uniform(low=-0.1, high=0.1, size=2)
    qvel[0:2] = np.random.uniform(low=-0.01, high=0.01, size=2)

    # Proper way to update joint states in Gymnasium
    env.unwrapped.set_state(qpos, qvel)

    # Forward the simulation manually (no .sim.forward())
    # mujoco.mj_forward(model, data)
# Load the environment with rendering
# env = gym.make("Reacher-v5", render_mode="human", max_episode_steps=150, xml_file="./reacher.xml")

class ForceRandomizedReacher(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # NUKE the default state and force randomization
        model = self.unwrapped.model
        data = self.unwrapped.data
        
        # Randomize ARM JOINTS (qpos[0] and qpos[1])
        data.qpos[0] = np.random.uniform(-1.0, 1.0)  # Joint 0
        data.qpos[1] = np.random.uniform(-1.0, 1.0)  # Joint 1
        
        # Randomize ARM VELOCITIES (qvel[0] and qvel[1])
        # data.qvel[0] = np.random.uniform(-0.5, 0.5)
        # data.qvel[1] = np.random.uniform(-0.5, 0.5)
        
        # CRITICAL: Update physics *immediately* after changing state
        mujoco.mj_forward(model, data)
        
        # Return new observation
        return self.unwrapped._get_obs(), info

# Initialize environment with FORCED randomization
env = gym.make("Reacher-v5", render_mode="human", max_episode_steps=150)
env = ForceRandomizedReacher(env)  # Wrap it
print(f"The env starting is: {env.unwrapped.model.nq}")
# Test randomization
# for _ in range(50):
#     obs, _ = env.reset()
#     x = env.unwrapped.get_body_com("fingertip")[:2]
#     print(f"Arm joint angles: {x}")  # Should differ each time
# env = gym.make("Reacher-v5", render_mode="human",max_episode_steps=150)

# Reset the environment
obs, _ = env.reset()
# randomize_arm_position(env)

# Load model
model = PPO.load("ppo_reacher", device="cpu")

target_green = [0.12, -0.1]
target_red = [0.12, 0.1]
target_yellow = [-0.12, 0.1]

target_dict = {"green": target_green, "red": target_red, "yellow": target_yellow}

targets_goals = ["green", "red", "yellow"]



current_target = target_dict[targets_goals[0]]
# set_target_position(env, current_target[0], current_target[1])

t = 0
for _ in range(1000):
    
    action, _ = model.predict(obs, deterministic=False)
    # action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    env.render()
    # Print XY coordinates of end-effector (fingertip)
    # Get end-effector position from environment's state
    fingertip_xy = env.unwrapped.get_body_com("fingertip")[:2]
  
    distance_to_target = np.sqrt(obs[8]**2 + obs[9]**2)

    # if distance_to_target < 0.02:
    #     print("Target reached")
    #     targets_goals.append(targets_goals.pop(0))
    #     print(f"Targets remaining: {targets_goals}")
    #     current_target = target_dict[targets_goals[0]]
    #     set_target_position(env, current_target[0], current_target[1])



    # print(f"End-Effector XY coordinates: {fingertip_xy.shape}")
    # time.sleep(1)

    if terminated or truncated:
        
        obs, info = env.reset()
        x = env.unwrapped.get_body_com("fingertip")[:2]
        print(f"Arm joint: {x}")
        # t = 0
        # randomize_arm_position(env)
        # targets_goals = ["green", "red", "yellow"]
        # current_target = target_dict[targets_goals[0]]
        # set_target_position(env, current_target[0], current_target[1])


# Close the environment
env.close()