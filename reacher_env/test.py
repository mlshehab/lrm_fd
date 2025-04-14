import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import time
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
            data.qpos[0] = -1.5707 / 2
            data.qpos[1] = 0.0

        # Override velocities
        if qvel_override is not None:
            data.qvel[0] = qvel_override[0]
            data.qvel[1] = qvel_override[1]
        else:
            data.qvel[0] = np.random.uniform(-0.5, 0.5)
            data.qvel[1] = np.random.uniform(-0.5, 0.5)

        mujoco.mj_forward(model, data)

        return self.unwrapped._get_obs(), info
    
# class ForceRandomizedReacher(gym.Wrapper):
#     def __init__(self, env):
#         super().__init__(env)
        
#     def reset(self, **kwargs):
#         obs, info = self.env.reset(**kwargs)
        
#         # NUKE the default state and force randomization
#         model = self.unwrapped.model
#         data = self.unwrapped.data
        
#         # Randomize ARM JOINTS (qpos[0] and qpos[1])
#         # data.qpos[0] = np.random.uniform(-1.0, 1.0)  # Joint 0
#         data.qpos[0] =  -1.5707/2
#         # data.qpos[1] = np.random.uniform(-1.0, 1.0)  # Joint 1
#         data.qpos[1] = 0.0
        
#         # Randomize ARM VELOCITIES (qvel[0] and qvel[1])
#         data.qvel[0] = np.random.uniform(-0.5, 0.5)
#         data.qvel[1] = np.random.uniform(-0.5, 0.5)
        
#         # CRITICAL: Update physics *immediately* after changing state
#         mujoco.mj_forward(model, data)
        
#         # Return new observation
#         return self.unwrapped._get_obs(), info

# Initialize environment with FORCED randomization
env = gym.make("Reacher-v5", render_mode="human", max_episode_steps=150,xml_file="./reacher.xml")
env = ForceRandomizedReacher(env)  # Wrap it

# Test randomization
# for _ in range(50):
#     obs, _ = env.reset()
#     x = env.unwrapped.get_body_com("fingertip")[:2]
#     print(f"Arm joint angles: {x}")  # Should differ each time
# env = gym.make("Reacher-v5", render_mode="human",max_episode_steps=150)



# Load model
# model = PPO.load("ppo_reacher", device="cpu")

# def inverse_kinematics(x, y, L1=0.1, L2=0.1):
#     D = np.sqrt(x**2 + y**2)
#     if D > (L1 + L2):
#         raise ValueError("Target is out of reach!")

#     # Elbow angle (theta2)
#     cos_theta2 = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
#     theta2 = np.arccos(np.clip(cos_theta2, -1.0, 1.0))

#     # Shoulder angle (theta1)
#     k1 = L1 + L2 * np.cos(theta2)
#     k2 = L2 * np.sin(theta2)
#     theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)

#     return theta1, theta2

def inverse_kinematics(x, y, L1=0.1, L2=0.11):
    """
    Calculate joint angles for a 2-link planar arm to reach target position (x, y)
    
    Args:
        x, y: Target end-effector position
        L1: Length of first link (default 0.1 as in Reacher-v2)
        L2: Length of second link (default 0.1 as in Reacher-v2)
        
    Returns:
        (theta1, theta2): Joint angles in radians
    """
    # Calculate theta2 using cosine law
    D = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
    
    # Handle possible numerical errors (keep D within [-1, 1])
    D = np.clip(D, -1.0, 1.0)
    
    # Two possible solutions for theta2 (elbow up and elbow down)
    theta2 = np.arccos(D)
    
    # Calculate theta1
    theta1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(theta2), L1 + L2 * np.cos(theta2))
    
    # Return angles in range [-π, π]
    theta1 = np.arctan2(np.sin(theta1), np.cos(theta1))
    theta2 = np.arctan2(np.sin(theta2), np.cos(theta2))
    
    return theta1, theta2

target_green = [0.12, -0.1]
target_red = [0.12, 0.1]
target_yellow = [-0.12, 0.1]

# Reset the environment
theta1, theta2 = inverse_kinematics(target_yellow[0], target_yellow[1])
obs, _ = env.reset(qpos_override=[theta1, theta1])

target_dict = {"green": target_green, "red": target_red, "yellow": target_yellow}

targets_goals = ["green", "red", "yellow"]



current_target = target_dict[targets_goals[0]]
# set_target_position(env, current_target[0], current_target[1])

t = 0
for _ in range(1000):
    
    # action, _ = model.predict(obs, deterministic=False)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    env.render()
    if t == 0:
        time.sleep(1)
        t += 1
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
        
        obs, info = env.reset(qpos_override=[theta1, theta2])
        t = 0
        x = env.unwrapped.get_body_com("fingertip")[:2]
        print(f"Arm joint: {x}")
        # t = 0
        # randomize_arm_position(env)
        # targets_goals = ["green", "red", "yellow"]
        # current_target = target_dict[targets_goals[0]]
        # set_target_position(env, current_target[0], current_target[1])


# Close the environment
env.close()