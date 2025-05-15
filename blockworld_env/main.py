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
import argparse
import multiprocessing as mp
 

from simulator import BlockworldSimulator

from bwe_helpers import parse_args, generate_label_combinations, generate_policy_comparison_report, solve_sat_instance


import config

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--depth', type=int, default=20)
    parser.add_argument('--n_traj', type=int, default=4_000_000)
    parser.add_argument('--n_procs', type=int, default=int(mp.cpu_count()))
    parser.add_argument('--adv', action='store_true')
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()
    
    bw = BlocksWorldMDP(num_piles=config.NUM_PILES)

    transition_matrices, s2i, i2s = bw.extract_transition_matrices()
    n_states = bw.num_states
    n_actions = bw.num_actions

    P = []

    for a in range(n_actions):
       
        P.append(transition_matrices[a,:,:])

    mdp = MDP(n_states=n_states, n_actions=n_actions,P = P,gamma = config.GAMMA,horizon=config.HORIZON)

    if args.adv:
        policy_path = config.POLICY_PATH_ADV
        rm = RewardMachine(config.RM_PATH_ADV)
        L = {
        s2i[config.TARGET_STATE_1]: 'A',
        s2i[config.TARGET_STATE_2]: 'B',
        s2i[config.BAD_STATE]: 'D',
        }
        
        for s in range(n_states):
            if s not in L:
                L[s] = 'I'
        
    else:
        policy_path = config.POLICY_PATH
        rm = RewardMachine(config.RM_PATH)
        L = {
        s2i[config.TARGET_STATE_1]: 'A',
        s2i[config.TARGET_STATE_2]: 'B',
        s2i[config.TARGET_STATE_3]: 'C'
        }
        for s in range(n_states):
            if s not in L:
                L[s] = 'I'

    
    soft_policy = np.load("./"+ policy_path + ".npy")

    

    bws = BlockworldSimulator(rm = rm,mdp = mdp, L = L, policy = soft_policy, state2index=s2i, index2state=i2s)
    
    
    starting_states = [s2i[config.TARGET_STATE_1], s2i[config.TARGET_STATE_2], s2i[config.TARGET_STATE_3], 4, 24]

    start = time.time()
    max_len = args.depth
    n_traj = args.n_traj
    
    bws.sample_dataset(starting_states=starting_states, number_of_trajectories= n_traj, max_trajectory_length=max_len)
    end = time.time()

    elapsed_time = end - start
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Simulating the dataset took {int(hours)} hour {int(minutes)} minute {seconds:.2f} sec.")

    bws.compute_action_distributions()

    # Save the object to a file
    if args.save:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        with open(f"./objects/object{n_traj}_{max_len}_{timestamp}.pkl", "wb") as foo:
            pickle.dump(bws, foo)
        print(f"The object has been saved to ./objects/object{n_traj}_{max_len}_{timestamp}.pkl")

        # generate_policy_comparison_report(bws, rm, soft_policy, n_traj, max_len, timestamp)
  
    
    counter_examples = generate_label_combinations(bws)



    p_threshold = 0.95
    metric = "L1"
    kappa = 3
    AP = 4
 
    if args.adv:
        proposition2index = { 'A': 0,'B': 1,'D': 2,'I': 3}
    else:
        proposition2index = {'A': 0,'B': 1,'C': 2,'I': 3 }
    
    solutions, n_constraints, n_states, solve_time, prob_values, wrong_ce_counts = \
               solve_sat_instance(bws, counter_examples,rm, metric, kappa, AP, proposition2index, p_threshold=p_threshold)

    print(f"The number of constraints is: {n_constraints}")
    print(f"The number of solutions is: {len(solutions)}")
    # for solution in solutions:
    #     print("\nSolution matrices:")
    #     for i, matrix in enumerate(solution):
    #         print(f"\nMatrix {i} ({['A', 'B', 'C', 'I'][i]}):")
    #         for row in matrix:
    #             print("  " + " ".join("1" if x else "0" for x in row))
    print(f"The wrong counter example counts are: {wrong_ce_counts}")
    hours, rem = divmod(solve_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"The solve time is: {int(hours)} hour {int(minutes)} minute {seconds:.2f} sec.")
    # print(f"The count of state 51 is: {bws.state_label_counts[51]}")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    solutions_text_path = f"./objects/solutions_{args.n_traj}_{args.depth}_{timestamp}.txt"

    if args.save:
        with open(solutions_text_path, "w") as f:
            f.write(f"Solutions for n_traj={args.n_traj}, depth={args.depth}\n")
            f.write("=" * 50 + "\n\n")
            for i, solution in enumerate(solutions):
                f.write(f"Solution {i+1}:\n")
                for j, matrix in enumerate(solution):
                    f.write(f"\nMatrix {j} ({['A', 'B', 'C', 'I'][j]}):\n")
                    for row in matrix:
                        f.write("  " + " ".join("1" if x else "0" for x in row) + "\n")
                f.write("\n" + "-" * 30 + "\n\n")

    print(f"[Main] Saved solutions in readable format to {solutions_text_path}")