import numpy as np
from tqdm import tqdm
from collections import Counter
from scipy.stats import entropy
import gymnasium as gym
from stable_baselines3 import PPO # type: ignore
import mujoco
import time
import itertools
import pickle
import matplotlib.pyplot as plt
import os

from train_PPO_policy_randomized_ic_discrete import DiscreteReacherActionWrapper

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



def set_target_position(env, x, y):


    model = env.unwrapped.model
    data = env.unwrapped.data
    
    # Find the joint IDs for target_x and target_y
    target_x_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_x")
    target_y_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_y")
    
    # Set the target's position
    data.qpos[target_x_id] = x
    data.qpos[target_y_id] = y


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



class ReacherDiscretizer:
    """Discretize the Reacher environment using joint angles (theta) and angular velocities.

    State   : (theta1, theta2, theta1_dot, theta2_dot)
    Action  : continuous torques on joint‑0 and joint‑1 discretised into bins.
    Label L : maps a state index to a region label (B / R / Y / I) by computing the
               end‑effector (x, y) via forward kinematics and comparing with targets.
    """

    def __init__(
        self,
        target_dict,
        theta_grid_size=np.deg2rad(5),      # 5° resolution ≈ 0.087 rad
        vel_grid_size=1.0,                  # rad s⁻¹ resolution
        action_grid_size=0.1,               # torque resolution
        theta_bound=np.pi,                  # joint limits [‑π, π]
        vel_bound=14,                      # assume |q̇| ≤ 1  (override per‑env)
        action_bound=1.0,                   # assume |τ| ≤ 1
        link_lengths=(0.1, 0.11)            # l1, l2  (m)
    ) -> None:
        self.target_dict = target_dict
        self.theta_grid_size = theta_grid_size
        self.vel_grid_size = vel_grid_size
        self.action_grid_size = action_grid_size
        self.link_lengths = link_lengths

        # Build bins
        self._build_theta_bins()
        # self.theta1_bins = np.arange(-theta_bound, theta_bound + theta_grid_size, theta_grid_size)
        # self.theta2_bins = np.arange(-theta_bound, theta_bound + theta_grid_size, theta_grid_size)
        self.theta1dot_bins = np.arange(-vel_bound, vel_bound + vel_grid_size, vel_grid_size)
        self.theta2dot_bins = np.arange(-vel_bound, vel_bound + vel_grid_size, vel_grid_size)

        # self.action_bins = np.arange(-action_bound, action_bound + action_grid_size, action_grid_size)
        self.action_bins = np.array([-1,-0.5,0,0.5,1])
        # Index maps
        self.state_to_idx, self.idx_to_state = self._build_state_maps()
        self.action_to_idx, self.idx_to_action = self._build_action_maps()
        # print(f"The action_to_idx is {self.action_to_idx}")
        # time.sleep(100)
        # Transition containers (counts & probabilities)
        self.n_states = len(self.state_to_idx)
        self.n_actions = len(self.action_to_idx)
     

    # ---------------------------------------------------------------------
    #                             BUILD MAPS
    # ---------------------------------------------------------------------
    def _build_theta_bins(self):
        blue_solutions = self._double_inverse_kinematics(self.target_dict["blue"][0], self.target_dict["blue"][1])
        red_solutions = self._double_inverse_kinematics(self.target_dict["red"][0], self.target_dict["red"][1])
        yellow_solutions = self._double_inverse_kinematics(self.target_dict["yellow"][0], self.target_dict["yellow"][1])
        solutions = [blue_solutions, red_solutions, yellow_solutions]
        
        self.solutions = {
            "blue": blue_solutions,
            "red": red_solutions,
            "yellow": yellow_solutions
        }
       
        x_grid = [-np.pi]
        y_grid = [-np.pi]
        
        # Add points for each solution
        for sols in solutions:
            for theta1, theta2 in sols:
                # Calculate half of the grid size in radians
                half_size = self.theta_grid_size
                
                # Add points around each solution
                x_grid.extend([theta1 - half_size, theta1 + half_size])
                y_grid.extend([theta2 - half_size, theta2 + half_size])
        
        # Add the upper bound
        x_grid.append(np.pi)
        y_grid.append(np.pi)
        
        # Sort and remove duplicates
        x_grid = np.array(sorted(list(set(x_grid))))
        y_grid = np.array(sorted(list(set(y_grid))))
        
        # print("x_grid:", np.round(np.array(x_grid),3))
        # print("y_grid:", np.round(np.array(y_grid),3))
        
        self.theta1_bins = x_grid
        self.theta2_bins = y_grid
        
    def _build_state_maps(self):
        """Return ({(i,j,k,l): s_idx}, {s_idx: (i,j,k,l)})."""
        indices = itertools.product(
            range(len(self.theta1_bins)),
            range(len(self.theta2_bins)),
            range(len(self.theta1dot_bins)),
            range(len(self.theta2dot_bins)),
        )
        state_to_idx = {}
        idx_to_state = {}
        for s_idx, key in enumerate(indices):
            state_to_idx[key] = s_idx
            idx_to_state[s_idx] = key
        return state_to_idx, idx_to_state

    def _build_action_maps(self):
        indices = itertools.product(range(len(self.action_bins)), repeat=2)
        action_to_idx = {}
        idx_to_action = {}
        for a_idx, key in enumerate(indices):
            action_to_idx[key] = a_idx
            idx_to_action[a_idx] = key
        return action_to_idx, idx_to_action

    # ---------------------------------------------------------------------
    #                             DISCRETISERS
    # ---------------------------------------------------------------------
    def discretize_state(self, state):
        (theta1, theta2, theta1dot, theta2dot) = state
        i = np.digitize(theta1, self.theta1_bins) - 1
        j = np.digitize(theta2, self.theta2_bins) - 1
        k = np.digitize(theta1dot, self.theta1dot_bins) - 1
        l = np.digitize(theta2dot, self.theta2dot_bins) - 1
        return (i, j, k, l)

    def discretize_action(self, act , already_discretized = True):
        if already_discretized:
            a0_idx = act[0]
            a1_idx = act[1]
        else:
            a0_idx = np.digitize(act[0], self.action_bins) - 1
            a1_idx = np.digitize(act[1], self.action_bins) - 1
        return (a0_idx, a1_idx)

    # ---------------------------------------------------------------------
    #                       UTILS / REVERSE MAPPING
    # ---------------------------------------------------------------------
    def midpoint_from_state_idx(self, state_idx):
        i, j, k, l = self.idx_to_state[state_idx]
        theta1_mid = self.theta1_bins[i] + self.theta_grid_size / 2
        theta2_mid = self.theta2_bins[j] + self.theta_grid_size / 2
        theta1dot_mid = self.theta1dot_bins[k] + self.vel_grid_size / 2
        theta2dot_mid = self.theta2dot_bins[l] + self.vel_grid_size / 2
        return theta1_mid, theta2_mid, theta1dot_mid, theta2dot_mid

    # ---------------------------------------------------------------------
    #                               LABELING
    # ---------------------------------------------------------------------
    def L(self, state):
        """Return colour label based on whether joint angles are within solution boxes for targets."""
        theta1, theta2, _ , _ = state
        
        # Check if joint angles are within solution boxes for each target
        for colour, solutions in self.solutions.items():
            for sol in solutions:
                theta1_sol, theta2_sol = sol
                if (abs(theta1 - theta1_sol) < self.theta_grid_size and 
                    abs(theta2 - theta2_sol) < self.theta_grid_size):
                    return colour[0].upper()  # 'blue' -> 'B'
        return 'I'  # intermediate / none
    
    def st4obs(self, obs):
        th1, th2 , th1dot, th2dot = (np.arctan2(obs[2],obs[0]), np.arctan2(obs[3],obs[1]) , obs[6] , obs[7])
        return (th1, th2 , th1dot, th2dot)

    # ---------------------------------------------------------------------
    #                           FORWARD KINEMATICS
    # ---------------------------------------------------------------------
    def _forward_kinematics(self, theta1, theta2):
        l1, l2 = self.link_lengths
        # Planar 2‑link arm anchored at origin (0,0)
        x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
        y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)
        return x, y
    
    def _double_inverse_kinematics( self, x, y):

        """Solves inverse kinematics for a given (x, y) target."""
        l1, l2 =  self.link_lengths
        # Using cosine law to compute the angle configurations (elbow-up and elbow-down)
        D = (x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2)
        D = np.clip(D, -1.0, 1.0)  # Ensure D is within [-1, 1] to avoid math errors
        theta2_up = np.arccos(D)
        theta2_down = -theta2_up
        
        theta1_up = np.arctan2(y, x) - np.arctan2(l2 * np.sin(theta2_up), l1 + l2 * np.cos(theta2_up))
        theta1_down = np.arctan2(y, x) - np.arctan2(l2 * np.sin(theta2_down), l1 + l2 * np.cos(theta2_down))
        
        return [(theta1_up, theta2_up), (theta1_down, theta2_down)]


class ReacherDiscretizerUniform:
    """Discretize the Reacher environment using joint angles (theta) and angular velocities.

    State   : (theta1, theta2, theta1_dot, theta2_dot)
    Action  : continuous torques on joint‑0 and joint‑1 discretised into bins.
    Label L : maps a state index to a region label (B / R / Y / I) by computing the
               end‑effector (x, y) via forward kinematics and comparing with targets.
    """

    def __init__(
        self,
        target_dict,
        theta_grid_size=np.deg2rad(15),      # 5° resolution ≈ 0.087 rad
        vel_grid_size=1.0,                  # rad s⁻¹ resolution
        action_grid_size=0.2,               # torque resolution
        theta_bound=np.pi,                  # joint limits [‑π, π]
        vel_bound=14,                      # assume |q̇| ≤ 1  (override per‑env)
        action_bound=1.0,                   # assume |τ| ≤ 1
        link_lengths=(0.1, 0.11)            # l1, l2  (m)
    ) -> None:
        self.target_dict = target_dict
        self.theta_grid_size = theta_grid_size
        self.vel_grid_size = vel_grid_size
        self.action_grid_size = action_grid_size
        self.link_lengths = link_lengths

        # Build bins
        self._build_theta_bins()
        self.theta1_bins = np.arange(-theta_bound, theta_bound + theta_grid_size, theta_grid_size)
        self.theta2_bins = np.arange(-theta_bound, theta_bound + theta_grid_size, theta_grid_size)
        self.theta1dot_bins = np.arange(-vel_bound, vel_bound + vel_grid_size, vel_grid_size)
        self.theta2dot_bins = np.arange(-vel_bound, vel_bound + vel_grid_size, vel_grid_size)

        # self.action_bins = np.arange(-action_bound, action_bound + action_grid_size, action_grid_size)
        self.action_bins = np.array([-1,-0.5,0,0.5,1])
        # Index maps
        self.state_to_idx, self.idx_to_state = self._build_state_maps()
        self.action_to_idx, self.idx_to_action = self._build_action_maps()

        # Transition containers (counts & probabilities)
        self.n_states = len(self.state_to_idx)
        self.n_actions = len(self.action_to_idx)
     

    # ---------------------------------------------------------------------
    #                             BUILD MAPS
    # ---------------------------------------------------------------------
    def _build_theta_bins(self):
        blue_solutions = self._double_inverse_kinematics(self.target_dict["blue"][0], self.target_dict["blue"][1])
        red_solutions = self._double_inverse_kinematics(self.target_dict["red"][0], self.target_dict["red"][1])
        yellow_solutions = self._double_inverse_kinematics(self.target_dict["yellow"][0], self.target_dict["yellow"][1])
        solutions = [blue_solutions, red_solutions, yellow_solutions]
        
        self.solutions = {
            "blue": blue_solutions,
            "red": red_solutions,
            "yellow": yellow_solutions
        }
       
        # x_grid = [-np.pi]
        # y_grid = [-np.pi]
        
        # # Add points for each solution
        # for sols in solutions:
        #     for theta1, theta2 in sols:
        #         # Calculate half of the grid size in radians
        #         half_size = self.theta_grid_size
                
        #         # Add points around each solution
        #         x_grid.extend([theta1 - half_size, theta1 + half_size])
        #         y_grid.extend([theta2 - half_size, theta2 + half_size])
        
        # # Add the upper bound
        # x_grid.append(np.pi)
        # y_grid.append(np.pi)
        
        # # Sort and remove duplicates
        # x_grid = np.array(sorted(list(set(x_grid))))
        # y_grid = np.array(sorted(list(set(y_grid))))
        
        # print("x_grid:", np.round(np.array(x_grid),3))
        # print("y_grid:", np.round(np.array(y_grid),3))
        
        # self.theta1_bins = x_grid
        # self.theta2_bins = y_grid
        
    def _build_state_maps(self):
        """Return ({(i,j,k,l): s_idx}, {s_idx: (i,j,k,l)})."""
        indices = itertools.product(
            range(len(self.theta1_bins)),
            range(len(self.theta2_bins)),
            range(len(self.theta1dot_bins)),
            range(len(self.theta2dot_bins)),
        )
        state_to_idx = {}
        idx_to_state = {}
        for s_idx, key in enumerate(indices):
            state_to_idx[key] = s_idx
            idx_to_state[s_idx] = key
        return state_to_idx, idx_to_state

    def _build_action_maps(self):
        indices = itertools.product(range(len(self.action_bins)), repeat=2)
        action_to_idx = {}
        idx_to_action = {}
        for a_idx, key in enumerate(indices):
            action_to_idx[key] = a_idx
            idx_to_action[a_idx] = key
        return action_to_idx, idx_to_action

    # ---------------------------------------------------------------------
    #                             DISCRETISERS
    # ---------------------------------------------------------------------
    def discretize_state(self, state):
        (theta1, theta2, theta1dot, theta2dot) = state
        i = np.digitize(theta1, self.theta1_bins) - 1
        j = np.digitize(theta2, self.theta2_bins) - 1
        k = np.digitize(theta1dot, self.theta1dot_bins) - 1
        l = np.digitize(theta2dot, self.theta2dot_bins) - 1
        return (i, j, k, l)

    def discretize_action(self, act , already_discretized = True):
        if already_discretized:
            a0_idx = act[0]
            a1_idx = act[1]
        else:
            a0_idx = np.digitize(act[0], self.action_bins) - 1
            a1_idx = np.digitize(act[1], self.action_bins) - 1
        return (a0_idx, a1_idx)

    # ---------------------------------------------------------------------
    #                       UTILS / REVERSE MAPPING
    # ---------------------------------------------------------------------
    def midpoint_from_state_idx(self, state_idx):
        i, j, k, l = self.idx_to_state[state_idx]
        theta1_mid = self.theta1_bins[i] + self.theta_grid_size / 2
        theta2_mid = self.theta2_bins[j] + self.theta_grid_size / 2
        theta1dot_mid = self.theta1dot_bins[k] + self.vel_grid_size / 2
        theta2dot_mid = self.theta2dot_bins[l] + self.vel_grid_size / 2
        return theta1_mid, theta2_mid, theta1dot_mid, theta2dot_mid

    # ---------------------------------------------------------------------
    #                               LABELING
    # ---------------------------------------------------------------------
    def L(self, state):
        """Return colour label based on whether joint angles are within solution boxes for targets."""
        # theta1, theta2, theta1dot , theta2dot = state
        
        discrete_state = self.discretize_state(state)
        # print(f"The discrete state is {self.state_to_idx[discrete_state]}")
        # find the center of the discrete state
        theta1, theta2 , _,_  = self.midpoint_from_state_idx(self.state_to_idx[discrete_state])
   


        # Check if joint angles are within solution boxes for each target
        for colour, solutions in self.solutions.items():
            for sol in solutions:
                theta1_sol, theta2_sol = sol
                if (abs(theta1 - theta1_sol) < self.theta_grid_size and 
                    abs(theta2 - theta2_sol) < self.theta_grid_size):
                    return colour[0].upper()  # 'blue' -> 'B'
        return 'I'  # intermediate / none

    # ---------------------------------------------------------------------
    #                           FORWARD KINEMATICS
    # ---------------------------------------------------------------------
    def _forward_kinematics(self, theta1, theta2):
        l1, l2 = self.link_lengths
        # Planar 2‑link arm anchored at origin (0,0)
        x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
        y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)
        return x, y
    
    def _double_inverse_kinematics( self, x, y):

        """Solves inverse kinematics for a given (x, y) target."""
        l1, l2 =  self.link_lengths
        # Using cosine law to compute the angle configurations (elbow-up and elbow-down)
        D = (x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2)
        D = np.clip(D, -1.0, 1.0)  # Ensure D is within [-1, 1] to avoid math errors
        theta2_up = np.arccos(D)
        theta2_down = -theta2_up
        
        theta1_up = np.arctan2(y, x) - np.arctan2(l2 * np.sin(theta2_up), l1 + l2 * np.cos(theta2_up))
        theta1_down = np.arctan2(y, x) - np.arctan2(l2 * np.sin(theta2_down), l1 + l2 * np.cos(theta2_down))
        
        return [(theta1_up, theta2_up), (theta1_down, theta2_down)]


class ReacherDiscreteSimulator():
    
    def __init__(self, env, policy, rd, target_goals):
        self.env = env
        self.policy = policy
        self.rd = rd
        self.target_goals = target_goals
        self.state_action_counts = {}  # Dictionary to track actions for (label, state)
        self.state_action_probs = {}
        self.state_label_counts = {}
        self.n_actions = self.rd.n_actions
        self.n_states = self.rd.n_states


    def remove_consecutive_duplicates(self, s):
        elements = s.split(',')
        if not elements:
            return s  # Handle edge case
        result = [elements[0]]
        for i in range(1, len(elements)):
            if elements[i] != elements[i - 1]:
                result.append(elements[i])
        return ','.join(result)

    def st4obs(self, obs):
        th1, th2 , th1dot, th2dot = (np.arctan2(obs[2],obs[0]), np.arctan2(obs[3],obs[1]) , obs[6] , obs[7])
        return (th1, th2 , th1dot, th2dot)

    def sample_trajectory(self, starting_state, len_traj,render=False, threshold=0.02 ):

        if not hasattr(self, 'history'):
            self.history = []
        
        theta1, theta2 = inverse_kinematics(starting_state[0], starting_state[1])

        obs, _ = self.env.reset(qpos_override=[theta1, theta2])
        current_target = self.rd.target_dict[self.target_goals[0]]

        set_target_position(self.env, current_target[0], current_target[1])
        
        continuous_state = self.st4obs(obs)
        discrete_state_tuple = self.rd.discretize_state(continuous_state)
        discrete_state_idx = self.rd.state_to_idx[discrete_state_tuple]
        label = self.rd.L(continuous_state) + ','
        next_continuous_state = continuous_state


        for t in range(len_traj):
            continuous_action, _ = self.policy.predict(obs, deterministic=False)
            self.history.append((next_continuous_state, continuous_action))
            obs, reward, terminated, truncated, info = self.env.step(continuous_action)

            if render:
                # print(f"The trajectory is {t} of {len_traj}")
                self.env.render()
                # time.sleep(0.05)
                 
            discrete_action_tuple = self.rd.discretize_action(continuous_action)
            discrete_action_idx = self.rd.action_to_idx[discrete_action_tuple]
            # print(f"The model action is {continuous_action} and the discrete action is {discrete_action_idx}")
            compressed_label = self.remove_consecutive_duplicates(label)
            # print(f"\rCurrent label: {compressed_label}", end="", flush=True)
            if discrete_state_idx == 141766:
                if compressed_label == 'R,I,B,I,':
                    print(f"\nFOUND OUR STATE! The label is: {compressed_label}")
            # if compressed_label == 'R,I,B,I,':
            #     print(f"The discrete state is: {discrete_state_idx}")
            if discrete_state_idx not in self.state_action_counts:
                self.state_action_counts[discrete_state_idx] = []

            label_exists = False
            for i, (existing_label, counter) in enumerate(self.state_action_counts[discrete_state_idx]):
                if existing_label == compressed_label:
                    counter[discrete_action_idx] += 1
                    label_exists = True
                    break
            
            if not label_exists:
                self.state_action_counts[discrete_state_idx].append((compressed_label, Counter({discrete_action_idx: 1})))

            next_continuous_state = self.st4obs(obs)
           
            next_discrete_state_tuple = self.rd.discretize_state(next_continuous_state)
            try:
                next_discrete_state_idx = self.rd.state_to_idx[next_discrete_state_tuple]
            except KeyError:
                # print(f"State {next_discrete_state_tuple} not found in state_to_idx")
                print(f"The state is: (θ1: {next_continuous_state[0]:.3f}, θ2: {next_continuous_state[1]:.3f}, θ1_dot: {next_continuous_state[2]:.3f}, θ2_dot: {next_continuous_state[3]:.3f})")
                return

            l = self.rd.L(next_continuous_state)

            label = label + l + ','
            discrete_state_idx = next_discrete_state_idx

            target_label = self.target_goals[0][0].upper()

            if l == target_label:
                self.target_goals.append(self.target_goals.pop(0))
                current_target = self.rd.target_dict[self.target_goals[0]]
                set_target_position(self.env, current_target[0], current_target[1])

            if terminated or truncated:
                
                
                # Add current state and action to history
                
                
                # Plot the history when trajectory ends
                if terminated or truncated:
                    # Extract data for plotting
                    theta1dots = [state[2] for state, _ in self.history]
                    theta2dots = [state[3] for state, _ in self.history]
                    actions = [DiscreteReacherActionWrapper.action_to_idx[tuple(action)] for _, action in self.history]
                    
                    # Create the plot
                    plt.figure(figsize=(12, 6))
                    plt.subplot(2, 1, 1)
                    plt.plot(theta1dots, label='θ1_dot')
                    plt.plot(theta2dots, label='θ2_dot')
                    plt.ylabel('Angular Velocity')
                    plt.legend()
                    
                    plt.subplot(2, 1, 2)
                    plt.plot(actions, label='Discrete Actions')
                    plt.ylabel('Action Index')
                    plt.xlabel('Time Step')
                    plt.legend()
                    
                    plt.tight_layout()
                    plt.show()
                    
                    # Clear history for next trajectory
                    self.history = []

                # print(f"The trajectory has terminated or truncated")
                # print(f"The label is: {compressed_label}")
                self.target_goals_reset()
                # self.env.close()  
                break

    def target_goals_reset(self):
        self.target_goals =  ["blue", "red", "yellow"]      

    def sample_dataset(self, starting_states, number_of_trajectories, max_trajectory_length):
        # for each starting state
        for state in starting_states:
            # for each length trajectory
            # for l in range(max_trajectory_length):
                # sample (number_of_trajectories) trajectories of length l 
            for i in  range(number_of_trajectories):
                # print(f"Sampling trajectory {i} of {number_of_trajectories}")
                self.sample_trajectory(starting_state= state,len_traj= max_trajectory_length)



    def compute_action_distributions(self):
        """
        Converts each action Counter into a probability distribution over all possible actions.

        Returns:
            dict: {state: [(label, action_probs)]}, where action_probs is a numpy array of size (self.n_actions,).
        """
        # state_action_probs = {}

        for state, label_counts in self.state_action_counts.items():
            self.state_action_probs[state] = {}
            self.state_label_counts[state] = {}

            for label, action_counter in label_counts:
                total_actions = sum(action_counter.values())  # Total samples for this label-state pair
                action_probs = np.zeros(self.rd.n_actions)  # Initialize with zeros for all actions
                
                if total_actions > 0:
                    for action, count in action_counter.items():
                        action_probs[action] = count / total_actions  # Normalize

                self.state_action_probs[state][label] = action_probs
                self.state_label_counts[state][label] = total_actions

    
    
from train_PPO_policy_randomized_ic_discrete import DiscreteReacherActionWrapper

if __name__ == "__main__":

    max_len = 150

    render = True
    video =False

    if render:
        env = gym.make("Reacher-v5", render_mode="human",  max_episode_steps=max_len,xml_file="./reacher.xml")
    else:
        env = gym.make("Reacher-v5",  max_episode_steps=max_len,xml_file="./reacher.xml")

    if video:
        env = gym.make("Reacher-v5",  render_mode="rgb_array", max_episode_steps=max_len,xml_file="./reacher.xml")
        env = gym.wrappers.RecordVideo(env, video_folder="./videos", name_prefix="reacher_run_BB")
    
    
    env = DiscreteReacherActionWrapper(env)
    env = ForceRandomizedReacher(env)  # Wrap it
    

    target_blue = [0.1, -0.11]
    target_red = [0.1, 0.11]
    target_yellow = [-0.1, 0.11]
    target_random_1 = [0.1,0.0]
    target_random_2 = [0.14,0.05]
     


    target_dict = {"blue": target_blue, "red": target_red, "yellow": target_yellow}

    targets_goals = ["blue", "red", "yellow"]

    rd = ReacherDiscretizerUniform(target_dict=target_dict)

    print(rd.n_states)
    print(rd.n_actions)
    # Initialize empty lists for x and y grid points
    
    print("The number of states is ", rd.n_states)
    print("The number of actions is ", rd.n_actions)
    policy = PPO.load("ppo_reacher_randomized_ic_discrete_5_actions", device="cpu")
    rds = ReacherDiscreteSimulator(env, policy, rd, targets_goals)

    start = time.time()
    
    n_traj = 10_000
    starting_states = [target_random_1, target_red, target_blue, target_yellow]
    for t in range(100):
        rds.sample_trajectory(starting_state= target_red, len_traj= max_len, render=render, threshold=0.02)
    # rds.sample_dataset(starting_states=starting_states, number_of_trajectories= n_traj, max_trajectory_length=max_len)
    # end = time.time()

    # elapsed_time = end - start
    # hours, rem = divmod(elapsed_time, 3600)
    # minutes, seconds = divmod(rem, 60)
    # print(f"Simulating the dataset took {int(hours)} hour {int(minutes)} minute {seconds:.2f} sec.")

     

    # rds.compute_action_distributions()

     

    # rds.policy = None  # Drop the PPO policy before saving
    # with open(f"./objects/object{n_traj}_{max_len}.pkl", "wb") as foo:
    #     pickle.dump(rds, foo)
   
 
    # print(f"The object has been saved to ./objects/object{n_traj}_{max_len}.pkl")        


    # with open("./objects/object10_10.pkl", "rb") as foo:
    #     rds = pickle.load(foo)

    # # print(rds.state_action_probs)
