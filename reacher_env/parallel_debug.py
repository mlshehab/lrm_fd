import numpy as np
from tqdm import tqdm
from collections import Counter
from scipy.stats import entropy
import gymnasium as gym
from stable_baselines3 import PPO
import mujoco
import time
import itertools
import pickle
import matplotlib.pyplot as plt
import os
from datetime import datetime
from multiprocessing import Pool

from train_PPO_policy_randomized_ic_discrete import DiscreteReacherActionWrapper

# === Inverse Kinematics ===
def inverse_kinematics(x, y, L1=0.1, L2=0.11):
    D = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
    D = np.clip(D, -1.0, 1.0)
    theta2 = np.arccos(D)
    theta1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(theta2), L1 + L2 * np.cos(theta2))
    theta1 = np.arctan2(np.sin(theta1), np.cos(theta1))
    theta2 = np.arctan2(np.sin(theta2), np.cos(theta2))
    return theta1, theta2

# === Set Target Position ===
def set_target_position(env, x, y):
    model = env.unwrapped.model
    data = env.unwrapped.data
    target_x_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_x")
    target_y_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_y")
    data.qpos[target_x_id] = x
    data.qpos[target_y_id] = y

# === Run a Single Trajectory ===
from simulator import ReacherDiscretizerUniform, ReacherDiscreteSimulator, ForceRandomizedReacher

from train_PPO_policy_randomized_ic_discrete import DiscreteReacherActionWrapper

def run_single_trajectory(_):
    try:
        max_len = 150
        target_dict = {"blue": [0.1, -0.11], "red": [0.1, 0.11], "yellow": [-0.1, 0.11]}
        target_goals = ["blue", "red", "yellow"]
        starting_state = [0.1, 0.0]

        env = gym.make("Reacher-v5", xml_file="./reacher.xml")
        env = DiscreteReacherActionWrapper(env)
        env = ForceRandomizedReacher(env)
        policy = PPO.load("ppo_reacher_randomized_ic_discrete_5_actions", device="cpu")
        rd = ReacherDiscretizerUniform(target_dict=target_dict)
        rds = ReacherDiscreteSimulator(env, policy, rd, target_goals)
        rds.sample_trajectory(starting_state=starting_state, len_traj=max_len, render=False, threshold=0.02)
    except Exception as e:
        print(f"Error in run_single_trajectory: {e}")

# === Parallel Execution Entry Point ===
if __name__ == "__main__":
    os.makedirs("debug_results", exist_ok=True)
    with Pool(processes=os.cpu_count()) as pool:
        pool.map(run_single_trajectory, range(1000))
