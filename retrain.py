import os
import sys

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Append the parent directory to sys.path
sys.path.append(parent_dir)
import pandas as pd
import numpy as np
from utils.mdp import MDP, MDPRM
from reward_machine.reward_machine import RewardMachine
import scipy.linalg
import time 
from scipy.special import softmax, logsumexp
from tqdm import tqdm
import pprint
import xml.etree.ElementTree as ET
from collections import deque
from dynamics.BlockWorldMDP import BlocksWorldMDP, infinite_horizon_soft_bellman_iteration
from utils.ne_utils import get_label, u_from_obs,save_tree_to_text_file, collect_state_traces_iteratively, get_unique_traces,group_traces_by_policy
from utils.sat_utils import *
from datetime import timedelta
import time
from collections import Counter, defaultdict
from scipy.stats import entropy  # For KL divergence
from tqdm import tqdm

import pickle 

class Simulator():
    
    def __init__(self, rm , mdp, L, policy):
        self.rm = rm
        self.mdp = mdp
        self.L = L
        self.policy = policy
        self.n_states = self.mdp.n_states
        self.n_actions = self.mdp.n_actions
        self.n_nodes = self.rm.n_states
        self.state_action_counts = {}  # Dictionary to track actions for (label, state)
        self.state_action_probs = {}

    def sample_next_state(self, state, action):
        """ Generic next state computation. """
        transition_probs = self.mdp.P[action][state, :]
        return np.random.choice(np.arange(self.n_states), p=transition_probs)

    def compute_action_distributions(self):
        """
        Converts each action Counter into a probability distribution over all possible actions.

        Returns:
            dict: {state: [(label, action_probs)]}, where action_probs is a numpy array of size (self.n_actions,).
        """
        # state_action_probs = {}

        for state, label_counts in self.state_action_counts.items():
            self.state_action_probs[state] = []

            for label, action_counter in label_counts:
                total_actions = sum(action_counter.values())  # Total samples for this label-state pair
                action_probs = np.zeros(self.n_actions)  # Initialize with zeros for all actions
                
                if total_actions > 0:
                    for action, count in action_counter.items():
                        action_probs[action] = count / total_actions  # Normalize

                self.state_action_probs[state].append((label, action_probs))

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


class BlockworldSimulator(Simulator):
    def __init__(self, rm, mdp, L, policy, state2index, index2state):
        super().__init__(rm, mdp, L, policy)
        self.state2index = state2index
        self.index2state = index2state

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
       
        # find the state in the reward machine
        # traj = []

        state = starting_state
        label = self.L[state] + ','
        u = u_from_obs(label,rm)
        
        for _ in range(len_traj):
            idx = u * self.n_states + state
            action_dist = self.policy[u*self.n_states + state,:]

            # Sample an action from the action distribution
            a = np.random.choice(np.arange(self.n_actions), p=action_dist)
            
            # Sample a next state 
            next_state = self.sample_next_state(state, a)
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


            # Debugging
            # if self.L[state] in {'A', 'B', 'C'}:
            #     print(f"Passed through {self.L[state]} !!!")


            l = self.L[next_state]
            label = label + l + ','
            u = u_from_obs(label,rm)
    
            state = next_state


    def sample_dataset(self, starting_states, number_of_trajectories, max_trajectory_length):
        # for each starting state
        for state in tqdm(starting_states):
            # for each length trajectory
            for l in range(max_trajectory_length):
                # sample (number_of_trajectories) trajectories of length l 
                for _ in range(number_of_trajectories):
                    self.sample_trajectory(starting_state= state,len_traj= l)



                
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
        

import argparse

# Define a function to handle command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Automate script with depth option")
    parser.add_argument("-depth", type=int, help="Set the depth", required=True)
    return parser.parse_args()

if __name__ == '__main__':



    bw = BlocksWorldMDP(num_piles=3)
    

    transition_matrices,s2i, i2s = bw.extract_transition_matrices_v2()
    n_states = bw.num_states
    n_actions = bw.num_actions

    print(bw)

    P = []

    for a in range(n_actions):
        # print(f"The matrix shape is: {transition_matrices[a,:,:]}")
        P.append(transition_matrices[a,:,:])

    mdp = MDP(n_states=n_states, n_actions=n_actions,P = P,gamma = 0.9,horizon=10)

    rm = RewardMachine("./rm_examples/dynamic_stacking.txt")
    print(f"rm.delta_u = {rm.delta_u}")


    policy = {}
    for rms in range(rm.n_states):
        policy[rms] = f"p{rms}"
    
    # policy[2] = policy[3]

    print("The policy is: ", policy)
  
    L = {}

    print(f"The number of states is: {len(s2i.keys())}")

    # for state_index in range(bw.num_states):
    #     state_tuple = i2s[state_index]
    #     L[state_index] = get_label(state_tuple)

    target_state_1 = ((0,1,2),(),())
    target_state_2 = ((),(2,1,0),())
    target_state_3 = ((),(),(2,1,0))
    bad_state = ((0,),(1,),(2,))

    for state_index in range(bw.num_states):
        if state_index == s2i[target_state_1]:
            L[state_index] = 'A'
        elif state_index == s2i[target_state_2]:
            L[state_index] = 'B'
        elif state_index == s2i[target_state_3]:
            L[state_index] = 'C'
        else:
            L[state_index] = 'I'

    
    mdpRM = MDPRM(mdp,rm,L)
    mdp_ =  mdpRM.construct_product()
 

    soft_policy = np.load("soft_policy.npy")



    # bws = BlockworldSimulator(rm = rm,mdp = mdp,L = L,policy = soft_policy,state2index=s2i,index2state=i2s)
    # # bws.sample_trajectory(starting_state=0,len_traj=9)
    
    starting_states = [s2i[target_state_1], s2i[target_state_2], s2i[target_state_3], 4, 24]

    


    # Save the object to a file
    # with open("object.pkl", "wb") as f:
    #     pickle.dump(bws, f)

    # Load the object back

    with open("./objects/object1000000_13.pkl", "rb") as f:
        bws = pickle.load(f)

    
    start = time.time()
    n_traj = 2000000
    max_len = 15
    bws.sample_dataset(starting_states=starting_states, number_of_trajectories= n_traj, max_trajectory_length=max_len)
    end = time.time()
    print(f"Simulating the dataset took {end - start} sec.")
    
    bws.compute_action_distributions()
    # Save the object to a file
    with open(f"./objects/object{1000000+n_traj}_{max_len}.pkl", "wb") as f:
        pickle.dump(bws, f)

    # # Load the object back
    # with open("object.pkl", "rb") as f:
    #     bws = pickle.load(f)

    # print(bws)
    # print(bws.state_action_counts)


    
  
    data = [] 
    for key, item in bws.state_action_probs.items():
        # print(f"The state is: {key}\n")
        for label, action_prob in item:
            u = u_from_obs(label,rm)
            policy = np.round(action_prob,3)
            
            true_policy = np.round(soft_policy[u*bws.n_states + key,:],3)
            kl_div = similarity(policy,true_policy,'KL')
            l1_norm = similarity(policy,true_policy,'L1')
            tv_distance = similarity(policy,true_policy,'TV')

            data.append([key, label, policy, true_policy, kl_div, l1_norm, tv_distance])
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=["State", "Label", "Policy", "True Policy", "KL Divergence", "L1 Norm", "TV Distance"])
    # Save as Excel


    # Compute min and max values for KL Divergence, L1 Norm, and TV Distance
    min_values = df[["KL Divergence", "L1 Norm", "TV Distance"]].min()
    max_values = df[["KL Divergence", "L1 Norm", "TV Distance"]].max()

    # Create a DataFrame for min and max values
    summary_df = pd.DataFrame({
        "State": ["Min", "Max"],
        "Label": ["-", "-"],
        "Policy": ["-", "-"],
        "True Policy": ["-", "-"],
        "KL Divergence": [min_values["KL Divergence"], max_values["KL Divergence"]],
        "L1 Norm": [min_values["L1 Norm"], max_values["L1 Norm"]],
        "TV Distance": [min_values["TV Distance"], max_values["TV Distance"]]
    })

    # Append the summary row to the original DataFrame
    df = pd.concat([df, summary_df], ignore_index=True)

    # Save DataFrame to CSV
    df.to_excel(f"./results/policy_comparison_nt_{1000000+n_traj}_ml_{max_len}.xlsx", index=False)
  

    
    
