import numpy as np
from tqdm import tqdm
from collections import Counter
from scipy.stats import entropy


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


class DiscreteSimulator():
    
    def __init__(self, env, L, policy):
        self.env = env
        self.L = L
        self.policy = policy
    
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

    def sample_trajectory(self, starting_state, len_traj):
        
        state = starting_state
        label = self.L[state] + ','
        theta1, theta2 = inverse_kinematics(state[0], state[1])
        obs, _ = self.env.reset(qpos_override=[theta1, theta1])

        for _ in range(len_traj):
            
            action = self.policy.predict(obs, deterministic=False)
            
            # Sample a next state 
            obs, reward, terminated, truncated, info = self.env.step(action)
            # traj.append((state,a,next_state))

            # Compress the label
            compressed_label = self.remove_consecutive_duplicates(label)

            # Ensure state exists in dictionary
            if state not in self.state_action_counts:
                self.state_action_counts[state] = []

            # Check if this label already exists for the state
            label_exists = False
            for i, (existing_label, counter) in enumerate(self.state_action_counts[state]):
                if existing_label == compressed_label:
                    counter[a] += 1  # Update action count
                    label_exists = True
                    break
            
            # If the label was not found, add a new entry
            if not label_exists:
                self.state_action_counts[state].append((compressed_label, Counter({a: 1})))

            l = self.L[next_state]
            label = label + l + ','
            u = u_from_obs(label, self.rm)
    
            state = next_state


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