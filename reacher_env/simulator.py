import numpy as np
from tqdm import tqdm
from collections import Counter
from scipy.stats import entropy
import gymnasium as gym
from stable_baselines3 import PPO # type: ignore
import mujoco
import time
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
    def __init__(self, target_dict, xy_grid_size=0.01,
                 action_grid_size=0.1,
                 xy_bound=0.22,
                 action_bound=1):
        self.target_dict = target_dict
        self.xy_grid_size = xy_grid_size
        self.action_grid_size = action_grid_size
        self.xy_bound = xy_bound

        # State and action bins
        self.x_bins = np.arange(-xy_bound, xy_bound + xy_grid_size, xy_grid_size)
        self.y_bins = np.arange(-xy_bound, xy_bound + xy_grid_size, xy_grid_size)
        self.action_bins = np.arange(-action_bound, action_bound + action_grid_size, action_grid_size)

        # Index maps
        self.state_to_idx, self.idx_to_state = self._build_state_maps()
        self.action_to_idx, self.idx_to_action = self._build_action_maps()

        # Transition counts and probabilities
        self.n_states = len(self.state_to_idx)
        self.n_actions = len(self.action_to_idx)
        self.transition_counts = np.zeros((self.n_actions, self.n_states, self.n_states))
        self.transition_matrices = None

    def _build_state_maps(self):
        states = [(i, j) for i in range(len(self.x_bins)) for j in range(len(self.y_bins))]
        state_to_idx = {state: idx for idx, state in enumerate(states)}
        idx_to_state = {idx: state for state, idx in state_to_idx.items()}
        return state_to_idx, idx_to_state

    def _build_action_maps(self):
        actions = [(i, j) for i in range(len(self.action_bins)) for j in range(len(self.action_bins))]
        action_to_idx = {act: idx for idx, act in enumerate(actions)}
        idx_to_action = {idx: act for act, idx in action_to_idx.items()}
        return action_to_idx, idx_to_action

    def discretize_xy(self, xy):
        x_idx = np.digitize(xy[0], self.x_bins) - 1
        y_idx = np.digitize(xy[1], self.y_bins) - 1
        return (x_idx, y_idx)

  
    def discretize_action(self, act):
        a0_idx = np.digitize(act[0], self.action_bins) - 1
        a1_idx = np.digitize(act[1], self.action_bins) - 1
        return (a0_idx, a1_idx)
    
    def midpoint_from_idx(self, x_idx, y_idx):
        x_mid = self.x_bins[x_idx] + self.xy_grid_size / 2
        y_mid = self.y_bins[y_idx] + self.xy_grid_size / 2
        return (x_mid, y_mid)

    def L(self, state_idx, threshold=0.02):
        x_idx , y_idx = self.idx_to_state[state_idx]
        x_mid, y_mid = self.midpoint_from_idx(x_idx, y_idx)

        target_blue = self.target_dict["blue"]
        target_red = self.target_dict["red"]
        target_yellow = self.target_dict["yellow"]

        if np.linalg.norm(np.array(target_blue) - np.array([x_mid, y_mid])) < threshold:
            return 'B'
        elif np.linalg.norm(np.array(target_red) - np.array([x_mid, y_mid])) < threshold:
            return 'R'
        elif np.linalg.norm(np.array(target_yellow) - np.array([x_mid, y_mid])) < threshold:
            return 'Y'
        else:
            return 'I'


class ReacherDiscreteSimulator():
    
    def __init__(self, env, policy, rd, target_goals):
        self.env = env
        self.policy = policy
        self.rd = rd
        self.target_goals = target_goals
        self.state_action_counts = {}  # Dictionary to track actions for (label, state)
        self.state_action_probs = {}
        self.state_label_counts = {}


    def remove_consecutive_duplicates(self, s):
        elements = s.split(',')
        if not elements:
            return s  # Handle edge case
        result = [elements[0]]
        for i in range(1, len(elements)):
            if elements[i] != elements[i - 1]:
                result.append(elements[i])
        return ','.join(result)



    def sample_trajectory(self, starting_state, len_traj, threshold=0.02):

        current_target = self.rd.target_dict[self.target_goals[0]]

        set_target_position(self.env, current_target[0], current_target[1])
        # starting state is a tuple [x,y] of x y position of the end effector
        continuous_state = starting_state
        discrete_state_tuple = self.rd.discretize_xy(continuous_state)
        discrete_state_idx = self.rd.state_to_idx[discrete_state_tuple]
        
        label = self.rd.L(discrete_state_idx) + ','
        theta1, theta2 = inverse_kinematics(starting_state[0], starting_state[1])

        
        obs, _ = self.env.reset(qpos_override=[theta1, theta1])
        sim = 0
        print(f"Simulation -- {sim}")

        for _ in range(len_traj):
            
            continuous_action, _ = self.policy.predict(obs, deterministic=False)
   
            # Log the continuous action for debugging
        
            discrete_action_tuple = self.rd.discretize_action(continuous_action)
            discrete_action_idx = self.rd.action_to_idx[discrete_action_tuple]

            # Sample a next state 
            obs, reward, terminated, truncated, info = self.env.step(continuous_action)
            
            self.env.render()

            distance_to_target = np.sqrt(obs[8]**2 + obs[9]**2)

            if distance_to_target < threshold:
                print("Target reached")
                self.target_goals.append(self.target_goals.pop(0))
                # print(f"Targets remaining: {self.target_goals}")
                current_target = self.rd.target_dict[self.target_goals[0]]
                set_target_position(env, current_target[0], current_target[1])
        
            # Compress the label
            compressed_label = self.remove_consecutive_duplicates(label)

            # Ensure state exists in dictionary
            if discrete_state_idx not in self.state_action_counts:
                self.state_action_counts[discrete_state_idx] = []

            # Check if this label already exists for the state
            label_exists = False
            for i, (existing_label, counter) in enumerate(self.state_action_counts[discrete_state_idx]):
                if existing_label == compressed_label:
                    counter[discrete_action_idx] += 1  # Update action count
                    label_exists = True
                    break
            
            # If the label was not found, add a new entry
            if not label_exists:
                self.state_action_counts[discrete_state_idx].append((compressed_label, Counter({discrete_action_idx: 1})))


            next_xy = self.env.unwrapped.get_body_com("fingertip")[:2]
            next_discrete_state_tuple = self.rd.discretize_xy(next_xy)
            next_discrete_state_idx = self.rd.state_to_idx[next_discrete_state_tuple]

            l = self.rd.L(next_discrete_state_idx)
           
            label = label + l + ','

            if terminated or truncated:
                obs, info = env.reset(qpos_override=[theta1, theta2])
                self.env.render()
                time.sleep(1)
                current_target = self.rd.target_dict[self.target_goals[0]]
                set_target_position(self.env, current_target[0], current_target[1])
                print(f"The label is {compressed_label}")  
                label = self.rd.L(discrete_state_idx) + ','  

                sim += 1
                print(f"Simulation -- {sim}")

        self.env.close()           
            

    def sample_dataset(self, starting_states, number_of_trajectories, max_trajectory_length):
        # for each starting state
        for state in tqdm(starting_states):
            # for each length trajectory
            for l in range(max_trajectory_length):
                # sample (number_of_trajectories) trajectories of length l 
                for _ in range(number_of_trajectories):
                    self.sample_trajectory(starting_state= state,len_traj= l)



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
                action_probs = np.zeros(self.n_actions)  # Initialize with zeros for all actions
                
                if total_actions > 0:
                    for action, count in action_counter.items():
                        action_probs[action] = count / total_actions  # Normalize

                self.state_action_probs[state][label] = action_probs
                self.state_label_counts[state][label] = total_actions

    def group_similar_policies(self, state, metric="TV", threshold=0.05):
        """
        Groups labels based on similar action distributions for each state.

        Args:
            metric (str): Similarity metric to use. Options: 'KL', 'TV', 'L1'.
            threshold (float): Maximum allowed difference to consider policies similar.

        Returns:
            dict: {state: {policy_signature: [labels]}}
        """
        grouped_traces = {}

        def similarity(p1, p2, metric):
            """Computes similarity based on chosen metric."""
            if metric == "KL":
                p1 = np.clip(p1, 1e-10, 1)  # Avoid division by zero
                p2 = np.clip(p2, 1e-10, 1)
                return entropy(p1, p2)  # KL divergence
            elif metric == "TV":
                return 0.5 * np.sum(np.abs(p1 - p2))  # Total Variation Distance
            elif metric == "L1":
                return np.linalg.norm(p1 - p2, ord=1)  # 1-norm distance
            else:
                raise ValueError("Unsupported metric! Choose from 'KL', 'TV', 'L1'.")

        if state not in self.state_action_probs:
            raise ValueError(f"State {state} not found in state_action_probs.")

        # Iterate over the labels for the given state
        for label, action_probs in self.state_action_probs[state]:
            matched = False

            # Compare against existing policy groups
            for existing_policy in grouped_traces:
                if similarity(existing_policy, action_probs, metric) < threshold:
                    grouped_traces[existing_policy].append(label)
                    matched = True
                    break
            
            # If no match, create a new group with this action_probs
            if not matched:
                grouped_traces[tuple(action_probs)] = [label]

        return grouped_traces
    

if __name__ == "__main__":
    env = gym.make("Reacher-v5", render_mode="human", max_episode_steps=150,xml_file="./reacher.xml")
    env = ForceRandomizedReacher(env)  # Wrap it

    target_blue = [0.12, -0.1]
    target_red = [0.12, 0.1]
    target_yellow = [-0.12, 0.1]

    target_dict = {"blue": target_blue, "red": target_red, "yellow": target_yellow}

    targets_goals = ["blue", "red", "yellow"]

    rd = ReacherDiscretizer(target_dict=target_dict)
    policy = PPO.load("ppo_reacher_randomized_ic", device="cpu")
    rds = ReacherDiscreteSimulator(env, policy, rd, targets_goals)
    rds.sample_trajectory(starting_state=target_yellow, len_traj=1500)



    # ds.sample_dataset(starting_states=[(0,0)], number_of_trajectories=10, max_trajectory_length=10)

