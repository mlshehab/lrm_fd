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
from dynamics.GridWorld import BasicGridWorld

from simulator import GridworldSimulator

from gwe_helpers import parse_args, generate_label_combinations, solve_sat_instance


import config

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--depth', type=int, default=10)
    parser.add_argument('--n_traj', type=int, default=2500)
    parser.add_argument('--save', action='store_true', default=False)
    args = parser.parse_args()
    
    grid_size = config.GRID_SIZE
    wind = config.WIND
    discount = config.GAMMA
    horizon = config.HORIZON   
 
    gw = BasicGridWorld(grid_size,wind,discount,horizon)
    
    n_states = gw.n_states
    n_actions = gw.n_actions

    P = []

    for a in range(n_actions):
        P.append(gw.transition_probability[:,a,:])

    mdp = MDP(n_states=n_states, n_actions=n_actions, P = P,gamma = gw.discount,horizon= config.HORIZON)

    
   
    rm = RewardMachine(config.RM_PATH)

    L = {}
    # The grid numbering and labeling is :
    # 0 4 8 12    D D C C 
    # 1 5 9 13    D D C C 
    # 2 6 10 14   A A B B
    # 3 7 11 15   A A B B
    
    L[2], L[6], L[3], L[7] = 'A', 'A', 'A', 'A'
    L[0], L[4], L[8], L[12] = 'D', 'D', 'C', 'C'
    L[1], L[5], L[9], L[13] = 'D', 'D', 'C', 'C'
    L[10], L[14] = 'B', 'B'
    L[11], L[15] = 'B', 'B'
        
    
    soft_policy = np.load(config.POLICY_PATH + ".npy")

    print(f"The number of trajectories is: {args.n_traj}")

    gws = GridworldSimulator(rm = rm,mdp = mdp, L = L, policy = soft_policy)
    
    starting_states = np.arange(n_states)
 
    start = time.time()
    max_len = args.depth
    n_traj = args.n_traj
    
    gws.sample_dataset(starting_states=starting_states, number_of_trajectories= n_traj, max_trajectory_length=max_len)
    end = time.time()

    elapsed_time = end - start
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Simulating the dataset took {int(hours)} hour {int(minutes)} minute {seconds:.2f} sec.")

    gws.compute_action_distributions()

    # Save the object to a file
    if args.save:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        with open(f"./objects/object{n_traj}_{max_len}_{timestamp}.pkl", "wb") as foo:
            pickle.dump(gws, foo)
        print(f"The object has been saved to ./objects/object{n_traj}_{max_len}_{timestamp}.pkl")

       
    counter_examples = generate_label_combinations(gws)

    p_threshold = 0.95
    metric = "L1"

     
    kappa = 3
    AP = 4
     
    proposition2index = {'A': 0,'B': 1,'C': 2,'D': 3 }
    
    solutions, n_constraints, n_states, solve_time, prob_values, wrong_ce_counts = \
               solve_sat_instance(gws, counter_examples,rm, metric, kappa, AP, proposition2index, p_threshold=p_threshold)

    print(f"The number of constraints is: {n_constraints}")
    print(f"The number of solutions is: {len(solutions)}")
    
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