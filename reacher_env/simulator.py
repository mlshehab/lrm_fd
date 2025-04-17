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
        vel_grid_size=4,                  # rad s⁻¹ resolution
        action_grid_size=0.5,               # torque resolution
        theta_bound=np.pi,                  # joint limits [‑π, π]
        vel_bound=20.0,                      # assume |q̇| ≤ 1  (override per‑env)
        action_bound=1.0,                   # assume |τ| ≤ 1
        link_lengths=(0.1, 0.11),            # l1, l2  (m)
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

        self.action_bins = np.arange(-action_bound, action_bound + action_grid_size, action_grid_size)

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
        x_grid = [-np.pi]
        y_grid = [-np.pi]
        
        # Add points for each solution
        for sols in solutions:
            for theta1, theta2 in sols:
                # Calculate half of the grid size in radians
                half_size = np.deg2rad(5)/2
                
                # Add points around each solution
                x_grid.extend([theta1 - half_size, theta1 + half_size])
                y_grid.extend([theta2 - half_size, theta2 + half_size])
        
        # Add the upper bound
        x_grid.append(np.pi)
        y_grid.append(np.pi)
        
        # Sort and remove duplicates
        x_grid = np.array(sorted(list(set(x_grid))))
        y_grid = np.array(sorted(list(set(y_grid))))
        
        print("x_grid:", np.round(np.array(x_grid),3))
        print("y_grid:", np.round(np.array(y_grid),3))
        
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

    def discretize_action(self, act):
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
    def L(self, state_idx, threshold=0.02):
        """Return colour label based on distance of end‑effector to targets."""
        theta1, theta2, _, _ = self.midpoint_from_state_idx(state_idx)
        x, y = self._forward_kinematics(theta1, theta2)
        # x,y = eef_pos
        for colour, target_xy in self.target_dict.items():
            if np.linalg.norm(np.array(target_xy) - np.array([x,y])) < threshold:
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

    def sample_trajectory(self, starting_state, len_traj, threshold=0.02):

        
        theta1, theta2 = inverse_kinematics(starting_state[0], starting_state[1])

        obs, _ = self.env.reset(qpos_override=[theta1, theta2])
        current_target = self.rd.target_dict[self.target_goals[0]]

        set_target_position(self.env, current_target[0], current_target[1])
        
        
        # starting state is a tuple [x,y] of x y position of the end effector
        # th1, th2 , th1dot, th2dot = (np.atan2(obs[2],obs[0]), np.atan2(obs[3],obs[1]) , obs[6] , obs[7])

        continuous_state = self.st4obs(obs)
        # print(f"Continuous state: {continuous_state}")
        discrete_state_tuple = self.rd.discretize_state(continuous_state)
        discrete_state_idx = self.rd.state_to_idx[discrete_state_tuple]
        # eef_pos = self.env.unwrapped.get_body_com("fingertip")[:2]
        label = self.rd.L(discrete_state_idx) + ','
        

       
        for t in range(len_traj):
            # print(f"Sampling trajectory {t} of {len_traj}")
            continuous_action, _ = self.policy.predict(obs, deterministic=False)
   
            # Sample a next state 
            obs, reward, terminated, truncated, info = self.env.step(continuous_action)
    
            # self.env.render()
            # time.sleep(0.1)
            # Log the continuous action for debugging
        
            discrete_action_tuple = self.rd.discretize_action(continuous_action)
            discrete_action_idx = self.rd.action_to_idx[discrete_action_tuple]


            distance_to_target = np.sqrt(obs[8]**2 + obs[9]**2)

            if distance_to_target < threshold:
                # print("Target reached")
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


            next_continuous_state = self.st4obs(obs)
            next_discrete_state_tuple = self.rd.discretize_state(next_continuous_state)
            next_discrete_state_idx = self.rd.state_to_idx[next_discrete_state_tuple]

            # eef_pos = self.env.unwrapped.get_body_com("fingertip")[:2]
            l = self.rd.L(next_discrete_state_idx)
           
            label = label + l + ','
            discrete_state_idx = next_discrete_state_idx

            if terminated or truncated:
                # print(f"The compressed label is {compressed_label}")
                self.env.close()  
                break

    def target_goals_reset(self):
        self.target_goals =  ["blue", "red", "yellow"]      

    def sample_dataset(self, starting_states, number_of_trajectories, max_trajectory_length):
        # for each starting state
        for state in tqdm(starting_states):
            # for each length trajectory
            # for l in range(max_trajectory_length):
                # sample (number_of_trajectories) trajectories of length l 
            for i in tqdm(range(number_of_trajectories)):
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
    env = gym.make("Reacher-v5",  max_episode_steps=200,xml_file="./reacher.xml")
    env = ForceRandomizedReacher(env)  # Wrap it

    target_blue = [0.12, -0.1]
    target_red = [0.12, 0.1]
    target_yellow = [-0.12, 0.1]
    target_random_1 = [0.1,0.0]
    target_random_2 = [0.14,0.05]

 
    # Create figure and axis
    




    target_dict = {"blue": target_blue, "red": target_red, "yellow": target_yellow}

    targets_goals = ["blue", "red", "yellow"]

    rd = ReacherDiscretizer(target_dict=target_dict)

    print(rd.n_states)
    print(rd.n_actions)
    # Initialize empty lists for x and y grid points
    
    # print("The number of states is ", rd.n_states)
    # print("The number of actions is ", rd.n_actions)
    # policy = PPO.load("ppo_reacher_randomized_ic", device="cpu")
    # rds = ReacherDiscreteSimulator(env, policy, rd, targets_goals)

    # start = time.time()
    # max_len = 200
    # n_traj = 50_000
    # starting_states = [target_random_1, target_blue, target_red, target_yellow]

    # rds.sample_dataset(starting_states=starting_states, number_of_trajectories= n_traj, max_trajectory_length=max_len)
    # end = time.time()

    # elapsed_time = end - start
    # hours, rem = divmod(elapsed_time, 3600)
    # minutes, seconds = divmod(rem, 60)
    # print(f"Simulating the dataset took {int(hours)} hour {int(minutes)} minute {seconds:.2f} sec.")

     

    # rds.compute_action_distributions()

    # for key, value in rds.state_action_counts.items():
    #     if len(value) > 2:
    #         print(f"{key}: {value}")

    # rds.policy = None  # Drop the PPO policy before saving
    # with open(f"./objects/object{n_traj}_{max_len}.pkl", "wb") as foo:
    #     pickle.dump(rds, foo)
   
 
    # print(f"The object has been saved to ./objects/object{n_traj}_{max_len}.pkl")        


    # with open("./objects/object10_10.pkl", "rb") as foo:
    #     rds = pickle.load(foo)

    # # print(rds.state_action_probs)
