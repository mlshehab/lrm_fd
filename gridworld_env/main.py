import os
import sys

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Append the parent directory to sys.path
sys.path.append(parent_dir)

import numpy as np
import time
import pickle
import argparse
from utils.mdp import MDP, MDPRM
from reward_machine.reward_machine import RewardMachine
from dynamics.GridWorld import BasicGridWorld
from simulator import GridworldSimulator
from gwe_helpers import generate_label_combinations, solve_sat_instance, maxsat_clauses, solve_with_clauses, prepare_sat_problem, constrtuct_product_policy
import config
from gwe_helpers import perfrom_policy_rollout, perfrom_policy_rollout_IRL
from gwe_helpers import construct_learned_product_policy
from tqdm import tqdm

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_rmm_learning', action='store_true', default=False)
    parser.add_argument('--depth', type=int, default=10)
    parser.add_argument('--n_traj', type=int, default=2500)
    parser.add_argument('--umax', type=int, default=4)
    parser.add_argument('--AP', type=int, default=4)
    parser.add_argument('--print_solutions', action='store_true', default=False)
    parser.add_argument('--use_maxsat', action='store_true', default=False)
    parser.add_argument('--use_irl', action='store_true', default=False)
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
        
    invL = {'A':[2,6,3,7], 'B':[10,11,14,15], 'C':[8,9,12,13], 'D':[0,1,4,5]}

    soft_policy = np.load(config.POLICY_PATH + ".npy")


    gws = GridworldSimulator(rm = rm,mdp = mdp, L = L, policy = soft_policy)
    
    starting_states = np.arange(n_states)
 
    
    max_len = args.depth
    n_traj = args.n_traj
    
    np.random.seed(config.SEED)
    gws.sample_dataset(starting_states=starting_states, number_of_trajectories= n_traj, max_trajectory_length=max_len, seed = config.SEED)
 
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

     
    umax = args.umax
    AP = args.AP   
    
    proposition2index = {'A': 0,'B': 1,'C': 2,'D': 3 }
    
    if args.run_rmm_learning:
        print(f"Running RMM learning with MAX-SAT and umax = {umax}")
        c4_clauses = prepare_sat_problem(gws, counter_examples, p_threshold)
        maxsat_clauses = maxsat_clauses(c4_clauses, umax, AP, proposition2index)
        solutions = solve_with_clauses(maxsat_clauses, umax, AP, proposition2index, print_solutions = args.print_solutions)
        print(f"Total # clauses: {len(c4_clauses)}")
        print(f"# used clauses: {len(maxsat_clauses)}")
        print(f"The number of solutions is: {solutions}")
             
    else:    
        # do evaluation instead  
        if args.umax == 4:
            rm_maxsat = rm
        elif args.umax == 3:
            rm_maxsat = RewardMachine(config.RM_PATH_MAXSAT_3)
        elif args.umax == 2:
            rm_maxsat = RewardMachine(config.RM_PATH_MAXSAT_2)
        else:
            raise ValueError(f"Invalid umax: {args.umax}")
        

        max_len = config.DEPTH_FOR_CONSTRUCTING_PRODUCT_POLICY
        
        if args.use_irl:
            learned_product_policy = np.load(config.IRL_POLICY_PATH + ".npy")

        else:
            learned_product_policy = construct_learned_product_policy(mdp, rm_maxsat, max_len, soft_policy, rm, invL, L)
      
        # perform policy rollout
        it = config.ROLLOUTS
        total_reward = 0.0
        for _ in tqdm(range(it)):
    
            if args.use_irl:
                total_reward += perfrom_policy_rollout_IRL(gws,config.ROLLOUT_LENGTH, rm, learned_product_policy, seed = config.SEED)
            else:
                total_reward += perfrom_policy_rollout(gws,config.ROLLOUT_LENGTH, rm_maxsat, rm, learned_product_policy, seed = config.SEED)
        
        print(f"The average reward is: {total_reward / it}")
        
    

    
