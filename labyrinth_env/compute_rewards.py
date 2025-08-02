import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm

import os
import sys
from scipy.optimize import minimize_scalar
# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Append the parent directory to sys.path
sys.path.append(parent_dir)

# Example matrices
from dynamics.BlockWorldMDP import BlocksWorldMDP, infinite_horizon_soft_bellman_iteration
from utils.mdp import MDP, MDPRM
from reward_machine.reward_machine import RewardMachine
import matplotlib.pyplot as plt
import time
import config

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
import labyrinth_with_stay
from le_helpers import generate_label_combinations
from collections import Counter
from le_helpers import solve_sat_instance
from utils.ne_utils import u_from_obs
from main import LabyrinthEnvSimulator as lenvs
import argparse

def d(P):
    m,n = P.shape
    dP = np.zeros((m,m*n))
    for i in range(m):
        dP[i,n*i:n*(i+1)] = P[i]
    return dP

def create_index_to_tuple_dict(mdp_states, rm_states, actions):
    # Compute the total number of elements
    total_elements = mdp_states**2 * rm_states**2 * actions
    
    # Initialize the dictionary
    index_to_tuple = {}
    
    # Iterate over all possible combinations and fill the dictionary
    index = 0
    for a in range(actions):
        for u in range(rm_states):
            for s in range(mdp_states):
                for u_prime in range(rm_states):
                    for s_prime in range(mdp_states):
                        index_to_tuple[index] = (s, u, a, s_prime, u_prime)
                        index += 1
                        
    return index_to_tuple


def get_u_ap_tuple(j, rm_states, ap_list):

    ap_len = len(ap_list)
    u = j // ap_len  # Calculate the RM state (u)
    ap = j % ap_len  # Calculate the AP index
    return (u, ap)


def construct_product_policy_from_trajectories(lb, rm, trajs, L, epsilon=0.05):
    """
    Build a product‐MDP policy from trajectories, then guarantee that
    every probability is >= epsilon by smoothing.

    Args:
        lb:       underlying MDP, with n_states and n_actions
        rm:       reward‐machine, with n_states (nodes)
        trajs:    list of dicts, each with 'states' and 'actions'
        L:        list or dict mapping state index -> label string
        epsilon:  lower‐bound for each probability (must satisfy epsilon * n_actions < 1)

    Returns:
        policy:   array of shape (n_states * n_nodes, n_actions), row‐stochastic
                  with ∀i,j: policy[i,j] ≥ epsilon
    """
    n_states  = lb.n_states
    n_actions = lb.n_actions
    n_nodes   = rm.n_states

    # 1) Count visits:
    policy = np.zeros((n_states * n_nodes, n_actions), dtype=float)
    
    for traj in trajs:
        label = ""
        for s, a in zip(traj['states'], traj['actions']):
            label += L[s] + ","
            comp = lenvs.remove_consecutive_duplicates(label)
            u = u_from_obs(comp, rm)
            idx = u * n_states + s
            policy[idx, a] += 1

    # 2) Normalize counts into a distribution, with uniform fallback for empty rows:
    row_sums = policy.sum(axis=1, keepdims=True)
    zero_rows = (row_sums == 0).flatten()
    # for non‐zero rows:
    nonzero = ~zero_rows
    policy[nonzero] /= row_sums[nonzero]
    # for zero rows, assign uniform:
    policy[zero_rows] = 1.0 / n_actions

    # 3) Smooth so that every entry ≥ epsilon, while keeping each row sum = 1
    if epsilon * n_actions >= 1.0:
        raise ValueError(f"epsilon * n_actions must be < 1 (got {epsilon * n_actions:.3f})")

    #   p' = (1 − K*ε) * p + ε
    policy = (1.0 - n_actions * epsilon) * policy + epsilon
    print(f"The minimum value in the policy is: {policy.min()}")
    return policy



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--umax", type=int, default=3, help="The number of states in the reward machine")
    args = parser.parse_args()

    GEN_DIR_NAME = './data/mouse_data/'

    TRAJS_DIR_NAME = GEN_DIR_NAME + 'water_restricted_mice_trajs.pickle'
    lb = labyrinth_with_stay.LabyrinthEnv()
    P_a = lb.get_transition_mat()
    
    n_states = P_a.shape[0] # num states in this env
    n_actions = P_a.shape[-1] # num actions in this env
    
    print(f"The number of states in the labyrinth env is: {n_states}")
    print(f"The number of actions in the labyrinth env is: {n_actions}")
 


 

    # Load trajectories
    trajs = pd.read_pickle(TRAJS_DIR_NAME)
    # Save trajectories to a text file
    traj_file = "trajectories.txt"
    with open(traj_file, 'w') as f:
        f.write("Trajectories:\n")
        f.write("=" * 50 + "\n\n")
        
        for i, traj in enumerate(trajs):
            f.write(f"Trajectory {i+1}:\n")
            f.write("States: " + " -> ".join(map(str, traj['states'])) + "\n")
            f.write("Actions: " + " -> ".join(map(str, traj['actions'])) + "\n\n")
            
        f.write("=" * 50 + "\n")
        f.write(f"Total trajectories: {len(trajs)}\n")
    
    print(f"Trajectories saved to {traj_file}")
     
   

    P = []

    for a in range(n_actions):
    
        P.append(P_a[:,:,a])

    mdp = MDP(n_states=n_states, n_actions=n_actions,P = P,gamma = config.GAMMA,horizon=config.HORIZON)

    if args.umax == 3:
        rm = RewardMachine(config.RM_PATH_MOD)
    elif args.umax == 2:
        rm = RewardMachine(config.RM_PATH)
    else:
        raise ValueError(f"Invalid umax: {args.umax}")


    print(f"rm.delta_u = {rm.delta_u}")

    L = {}

    for state in range(n_states):
        if state == lb.home_state:
            L[state] = 'H'
        elif state == lb.water_port:
            L[state] = 'W'
        else:
            L[state] = 'I'

    mdpRM = MDPRM(mdp,rm,L)
    mdp_ =  mdpRM.construct_product()

    soft_policy = construct_product_policy_from_trajectories(lb, rm, trajs, L) 
     

    P = mdp_.P[0]
    E = np.eye(mdp_.n_states)

    for a in range(1,mdp_.n_actions):
        P = np.vstack((P,mdp_.P[a]))
        E = np.vstack((E, np.eye(mdp_.n_states)))

     

    Psi = d(P)

    # A = np.hstack((Psi, -E + config.GAMMA*P))
    # # # print(f"The shape of A is {A.shape}")
    # # # print(f"The shape of soft_policy is {soft_policy.shape}")
    # # print(f"The shape of mdp_.n_states is {mdp_.n_states}")

    # # b = np.log(soft_policy)[:mdp_.n_states,:]
    # b = np.log(soft_policy)
    # # bb = np.log(soft_policy)[:mdp_.n_states,:]

    # # b = b.reshape((A.shape[0],1))
    # b = b.flatten('F')[:,None]

    # start = time.time()
    # x = np.linalg.lstsq(A,b, rcond = None)
    # end = time.time()
    # print(f"This took a total of {end - start} secs.")
    # print(f"The residual is: {x[1]}")
    # print(f"x[0].shape is {x[0].shape} , A.shape is {A.shape} , b.shape is {b.shape}")
    # print(f"The residual is: {np.linalg.norm(A@x[0]-b)}")
    
    
    AP = ['H','W','I']
    ap2index = {'H':0,'W':1,'I':2}
    row_F = mdp.n_states**2*rm.n_states**2*mdp.n_actions
    col_F = rm.n_states*len(AP)
    F = np.zeros(shape = (row_F,col_F))

    index_to_tuple = create_index_to_tuple_dict(mdp_states = mdp.n_states , rm_states= rm.n_states, actions = mdp.n_actions)


    for j in range(col_F):
        
        u_j, ap_j = get_u_ap_tuple(j, rm.n_states, AP)
       
        for i in range(row_F):
            (s,u,a,s_prime, u_prime) = index_to_tuple[i]

            L_s_prime = L[s_prime]

            if u == u_j and L_s_prime == AP[ap_j]:
                F[i,j] = 1.0

    A = np.hstack((Psi@F, -E + config.GAMMA*P))

    b = np.log(soft_policy) 

    b = b.flatten('F')[:,None]

    
    x = np.linalg.lstsq(A,b, rcond = None)
     
     
    reward_vec = x[0][:F.shape[1]]
    # Print the reward vector in a more readable way: node <u> AP '<ap>': <reward>
    reward_vec_shifted = np.round(reward_vec + abs(reward_vec.min()), decimals=3)
    for j in range(F.shape[1]):
        u_j, ap_j = get_u_ap_tuple(j, rm.n_states, AP)
        ap_label = AP[ap_j]
        print(f"node {u_j} AP '{ap_label}': {reward_vec_shifted[j][0]}")

    # # now we need a state action state reward for the product MDP
    # reward = np.zeros((mdp_.n_states, mdp_.n_actions, mdp_.n_states))
    # # print(f"Reward: {reward.shape}, S: {mdp.n_states}, A: {mdp.n_actions}, RM: {rm.n_states}")
    
    # # print(f"The shape of reward is: {reward_vec.shape}")
    # for bar_s in range(mdp_.n_states):
    #     for a in range(mdp_.n_actions):
    #         for bar_s_prime in range(mdp_.n_states):
    #             (s,u) = mdpRM.su_pair_from_s(bar_s)
    #             (s_prime,u_prime) = mdpRM.su_pair_from_s(bar_s_prime)

    #             is_possible = mdp_.P[a][bar_s][bar_s_prime] > 0.0
                
    #             lsp = L[s_prime]
    #             ap_index = ap2index[lsp]

    #             # if is_possible:

    #             reward[bar_s,a,bar_s_prime] = reward_vec[u * len(AP) + ap_index][0]


    # q_soft,v_soft , soft_policy_special_reward = infinite_horizon_soft_bellman_iteration(mdp_,reward,logging = False)

    # out = np.log(soft_policy_special_reward).flatten('F')[:,None]
   
    
 
    # print(f"The norm difference between the policies is: {np.linalg.norm(soft_policy_special_reward - soft_policy)}")
    # uniform_policy = np.ones((mdp_.n_states, mdp_.n_actions)) / mdp_.n_actions
    # print(f"The norm difference between the original and uniform policies is: {np.linalg.norm(uniform_policy - soft_policy)}")
    
    