import os
import sys
from scipy.optimize import minimize_scalar
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
import matplotlib.pyplot as plt
import pickle 

def f(epsilon_1, n1, n2, A, epsilon):
    term1 = np.maximum(1 - ((2**A - 2) * np.exp((-n1 * epsilon_1**2) / (2 ))), 0)
    term2 = np.maximum(1 - ((2**A - 2) * np.exp((-n2 * (epsilon - epsilon_1)**2) / (2 ))), 0)
    return term1 * term2

def find_optimal_epsilon1(n1, n2, A, epsilon):
    """
    Find the epsilon_1 value that maximizes f(epsilon_1) using scipy's optimize.
    
    Args:
        n1 (int): First sample size
        n2 (int): Second sample size 
        A (int): Size of action space
        epsilon (float): Total epsilon value
        
    Returns:
        float: The optimal epsilon_1 value that maximizes f(epsilon_1)
    """
    # Define objective function to minimize (negative of f since we want to maximize f)
    def objective(eps1):
        return -f(eps1, n1, n2, A, epsilon)
    
    # Find minimum of negative f (maximum of f) in range [0, epsilon]
    result = minimize_scalar(objective, bounds=(0, epsilon), method='bounded')
    
    return result.x



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

def solve_sat_instance(bws, counter_examples, epsilon, rm, p_threshold=0.8):
    """
    Solve SAT instance for given counter examples, filtering by probability threshold
    Returns all SAT solutions found
    """
    kappa = 3
    AP = 4
    total_variables = kappa**2*AP
    total_constraints = 0

    # Initialize SAT variables
    B = [[[Bool('x_%s_%s_%s'%(i,j,k)) for j in range(kappa)]for i in range(kappa)]for k in range(AP)]
    B_ = element_wise_or_boolean_matrices([b_k for b_k in B])
    x = [False]*kappa
    x[0] = True

    B_T = transpose_boolean_matrix(B_)
    powers_B_T = [boolean_matrix_power(B_T,k) for k in range(1,kappa)]
    powers_B_T_x = [boolean_matrix_vector_multiplication(B,x) for B in powers_B_T]
    powers_B_T_x.insert(0, x)
    OR_powers_B_T_x = element_wise_or_boolean_vectors(powers_B_T_x)
    
    s = Solver()

    # Add constraints
    for ap in range(AP):
        for i in range(kappa):
            for j in range(kappa):
                s.add(Implies(B[ap][i][j], B[ap][j][j]))

    for k in range(AP):
        s.add(one_entry_per_row(B[k]))

    proposition2index = {'A':0,'B':1,'C':2,'I':3}

    def prefix2indices(s):
        out = []
        for l in s.split(','):
            if l:
                out.append(proposition2index[l])
        return out

    # Filter counter examples by probability threshold
    filtered_counter_examples = {}
    prob_values = []  # Store probability values for visualization

    wrong_ce_counts = 0

    for state, ce_set in counter_examples.items():
        filtered_ce = []
        for ce in ce_set:
            n1 = bws.state_label_counts[state][ce[0]] 
            n2 = bws.state_label_counts[state][ce[1]]
            A = bws.n_actions
            # optimal_epsilon1 = find_optimal_epsilon1(n1, n2, A, epsilon)
            optimal_epsilon1 = epsilon/2
            prob = f(optimal_epsilon1, n1, n2, A, epsilon)
            prob_values.append(prob)
            
            if prob > p_threshold:
                filtered_ce.append((ce, prob))  # Store probability with counter example
                
        if filtered_ce:
            filtered_counter_examples[state] = filtered_ce

    # Add C4 constraints for filtered counter examples
    for state in filtered_counter_examples.keys():
        ce_set = filtered_counter_examples[state]
        total_constraints += len(ce_set)
        
        for ce, prob in ce_set:
            if u_from_obs(ce[0],rm) == u_from_obs(ce[1],rm):
                wrong_ce_counts += 1

            p1 = prefix2indices(ce[0])
            p2 = prefix2indices(ce[1])

            sub_B1 = bool_matrix_mult_from_indices(B,p1, x)
            sub_B2 = bool_matrix_mult_from_indices(B,p2, x)
            res_ = element_wise_and_boolean_vectors(sub_B1, sub_B2)

            # print(f"Negative example at state {state}:")
            # print(f"  Prefix 1: {ce[0]}   -- Count: {bws.state_label_counts[state][ce[0]]}")
            # print(f"  Prefix 2: {ce[1]}   -- Count: {bws.state_label_counts[state][ce[1]]}")
            # print(f"  Probability: {prob:.4f}")

            for elt in res_:
                s.add(Not(elt))

    # Find all solutions
    solutions = []
    start = time.time()
    while s.check() == sat:
        m = s.model()
        solution = []
        for ap in range(AP):
            r = [[m.evaluate(B[ap][i][j]) for j in range(kappa)] for i in range(kappa)]
            solution.append(r)
        solutions.append(solution)

        block_clause = []
        for ap in range(AP):
            for i in range(kappa):
                for j in range(kappa):
                    block_clause.append(B[ap][i][j] != m.evaluate(B[ap][i][j], model_completion=True))
        s.add(Or(block_clause))

    end = time.time()
    solve_time = end - start
    
    return solutions, total_constraints, len(filtered_counter_examples), solve_time, prob_values, wrong_ce_counts
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

    # start = time.time()
    # n_traj = 200000
    # max_len = 15
    # bws.sample_dataset(starting_states=starting_states, number_of_trajectories= n_traj, max_trajectory_length=max_len)
    # end = time.time()
    # print(f"Simulating the dataset took {end - start} sec.")


    # Save the object to a file
    # with open("object.pkl", "wb") as f:
    #     pickle.dump(bws, f)

    # Load the object back
    with open("object1000000_13.pkl", "rb") as foo:
        bws = pickle.load(foo)

    
    # epsilon_vals = [0.001, 0.002, 0.005, 0.01, 0.02,0.05, 0.1, 0.2, 0.5]
    epsilon_vals = [0.1,0.2]
    metrics = ["L1","KL"]
    # metrics = ["TV", "KL", "L1"]
   
    # print(f"The count of state 51 is: {bws.state_label_counts[51]}")

    # Run SAT solver for each metric and threshold
    results = {metric: [] for metric in metrics}
    for metric in tqdm(metrics):
        for epsilon in epsilon_vals:
            state_traces_dict = {}
            for state, label_dists in bws.state_action_probs.items():
                if len(label_dists) > 1:
                    gpd = bws.group_similar_policies(state, metric=metric, threshold=epsilon)
                    grouped_lists = list(gpd.values())
                    state_traces_dict[state] = grouped_lists
            
            counter_examples = generate_combinations(state_traces_dict)
            solutions, n_constraints, n_states, solve_time, prob_values, wrong_ce_counts = solve_sat_instance(bws, counter_examples, epsilon,rm,p_threshold=0.95)
            results[metric].append({
                'epsilon': epsilon,
                'solutions': solutions,
                'n_solutions': len(solutions),
                'n_constraints': n_constraints, 
                'n_states': n_states,
                'solve_time': solve_time,
                'prob_values': prob_values,
                'wrong_ce_counts': wrong_ce_counts
            })


    for solution in results["L1"][0]["solutions"]:
        print("\nSolution matrices:")
        for i, matrix in enumerate(solution):
            print(f"\nMatrix {i} ({['A', 'B', 'C', 'I'][i]}):")
            for row in matrix:
                print("  " + " ".join("1" if x else "0" for x in row))


    # Visualization
    plt.figure(figsize=(15, 12))

    # Plot 1: Number of solutions vs epsilon for each metric
    plt.subplot(3, 2, 1)
    for metric in metrics:
        plt.plot(epsilon_vals, [r['n_solutions'] for r in results[metric]], marker='o', label=metric)
    plt.xlabel('Epsilon')
    plt.ylabel('Number of Solutions')
    plt.title('Solutions vs Epsilon')
    plt.legend()
    plt.grid(True)

    # Plot 2: Solve time vs epsilon
    plt.subplot(3, 2, 2)
    for metric in metrics:
        plt.plot(epsilon_vals, [r['solve_time'] for r in results[metric]], marker='o', label=metric)
    plt.xlabel('Epsilon')
    plt.ylabel('Solve Time (s)')
    plt.title('Solve Time vs Epsilon')
    plt.legend()
    plt.grid(True)

    # Plot 3: Number of constraints vs epsilon
    plt.subplot(3, 2, 3)
    for metric in metrics:
        plt.plot(epsilon_vals, [r['n_constraints'] for r in results[metric]], marker='o', label=metric)
    plt.xlabel('Epsilon')
    plt.ylabel('Number of Constraints')
    plt.title('Constraints vs Epsilon')
    plt.legend()
    plt.grid(True)

    # Plot 4: Wrong counter examples count vs epsilon
    plt.subplot(3, 2, 4)
    for metric in metrics:
        plt.plot(epsilon_vals, [r['wrong_ce_counts'] for r in results[metric]], marker='o', label=metric)
    plt.xlabel('Epsilon')
    plt.ylabel('Wrong Counter Examples')
    plt.title('Wrong Counter Examples vs Epsilon')
    plt.legend()
    plt.grid(True)

    # Plot 5: Probability distribution violin plot
    plt.subplot(3, 2, (5, 6))
    prob_data = []
    labels = []
    for metric in metrics:
        for r in results[metric]:
            if r['prob_values']:  # Only add if there are probability values
                prob_data.append(r['prob_values'])
                labels.append(f"{metric}\nε={r['epsilon']:.3f}")

    plt.violinplot(prob_data)
    plt.xticks(range(1, len(labels) + 1), labels, rotation=45)
    plt.ylabel('Probability Values')
    plt.title('Distribution of Probability Values')

    plt.tight_layout()
    plt.savefig('sat_results_analysis.png')
    plt.close()



