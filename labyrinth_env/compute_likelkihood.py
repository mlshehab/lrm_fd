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
from MaxEntIRL import InfiniteHorizonMaxEntIRL, InfiniteHorizonMaxEntIRLWeighted
from dynamics.GridWorld import BasicGridWorld # type: ignore

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
    # print(f"The shape of the policy is: {policy.shape}")
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



def compute_likelihood(policy, traj,L, rm, is_irl=False) -> float:
    """Computes the likelihood of the observed data (visit_counts, action_counts) under the given policy."""
    
    states, actions = traj['states'], traj['actions']



    # Compute log-likelihood
    log_likelihood = 0.0
    T = len(states)

    # starting_state = states[0]
    # label = L[starting_state] + ','
    # comp = lenvs.remove_consecutive_duplicates(label)
    # u = u_from_obs(comp, rm)
    label = ''

    for s, a in zip(states, actions):
        label += L[s] + ","
        comp = lenvs.remove_consecutive_duplicates(label)
        u = u_from_obs(comp, rm)
        idx = u * n_states + s
        if not is_irl:
            log_likelihood += np.log(policy[idx, a])
        else:
            log_likelihood += np.log(policy[s, a])  
    
    return log_likelihood


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--umax", type=int, default=2, help="The number of states in the reward machine")
    args = parser.parse_args()

    GEN_DIR_NAME = './data/mouse_data/'

    TRAJS_DIR_NAME = GEN_DIR_NAME + 'water_restricted_mice_trajs.pickle'
    lb = labyrinth_with_stay.LabyrinthEnv()
    P_a = lb.get_transition_mat()
    
    print(f"The first action is: {P_a[:,:,0]}")
    input()
    
    n_states = P_a.shape[0] # num states in this env
    n_actions = P_a.shape[-1] # num actions in this env
    
    print(f"The number of states in the labyrinth env is: {n_states}")
    print(f"The number of actions in the labyrinth env is: {n_actions}")
 

    # Load trajectories
    trajs = pd.read_pickle(TRAJS_DIR_NAME)

   

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

    rm_2 = RewardMachine(config.RM_PATH)
    rm_3 = RewardMachine(config.RM_PATH_MOD)

    # Manually split the trajectories into training and testing sets
    split_ratio = config.TRAIN_TEST_SPLIT
    split_index = int(len(trajs) * split_ratio)

    train_trajs = trajs[:split_index]
    test_trajs = trajs[split_index:]

   
    soft_policy_2 = construct_product_policy_from_trajectories(lb, rm_2, train_trajs, L) 
    soft_policy_3 = construct_product_policy_from_trajectories(lb, rm_3, train_trajs, L) 

  
    uniform_policy = np.ones((mdp.n_states * rm.n_states, mdp.n_actions)) / mdp.n_actions
    
    # print(f"The log likelihood of the soft policy is: {compute_likelihood(soft_policy, trajs[1],L, rm)}")
  
    # create trajectories compatible with the IRL Algorithm

    expert_trajs = []

    rm = RewardMachine(config.RM_PATH)

    action_counts = {0:0, 1:0, 2:0, 3:0}
    # for tj in test_trajs:
    for tj in train_trajs:
        states, actions = tj['states'], tj['actions']
        t_len = len(states)
        traj = []
        for i in range(t_len):  # Enough steps to simulate stationarity
            state = states[i]
            action = actions[i]
            action_counts[action] += 1
            traj.append((state, int(action)))
 
        # print(compressed_label)
        expert_trajs.append(traj)

    print(f"The action counts are: {action_counts}")

    # Train Infinite Horizon MaxEnt IRL
    landmark_states=[lb.home_state, lb.water_port]

    irl_dense = InfiniteHorizonMaxEntIRLWeighted(mdp, expert_trajs, is_featurized=False, n_iter=50, lr=0.2)
    irl_featurized = InfiniteHorizonMaxEntIRLWeighted(mdp, expert_trajs, is_featurized=True, landmark_states=landmark_states, n_iter=50, lr=0.2)

    # learned_weights = irl_dense.train(verbose=True)
    # learned_weights_featurized = irl_featurized.train(verbose=True)
    
    # irl_policy_dense = irl_dense.soft_value_iteration()
    # irl_policy_featurized = irl_featurized.soft_value_iteration()

    
    policies_dir = './policies'
    if not os.path.exists(policies_dir):
        os.makedirs(policies_dir)

    # np.save(os.path.join(policies_dir, 'IRL_soft_policy_dense_weighted.npy'), irl_policy_dense)
    # np.save(os.path.join(policies_dir, 'IRL_soft_policy_featurized_weighted.npy'), irl_policy_featurized)

    irl_policy_dense = np.load(os.path.join(policies_dir, 'IRL_soft_policy_dense_weighted.npy'))
    irl_policy_featurized = np.load(os.path.join(policies_dir, 'IRL_soft_policy_featurized_weighted.npy'))
    
    epsilon = 0.05
    # n_actions = mdp.n_actions
    # irl_policy_dense = (1.0 - n_actions * epsilon) * irl_policy_dense + epsilon
    irl_policy_featurized = (1.0 - n_actions * epsilon) * irl_policy_featurized + epsilon
 

    # # Compute the average likelihood over test trajectories for each method

    # Initialize variables to store total log likelihoods
    total_log_likelihood_dense = 0.0
    total_log_likelihood_featurized = 0.0
    total_log_likelihood_uniform = 0.0
    total_log_likelihood_soft = 0.0

    # # Load or define uniform and soft policies
    # uniform_policy = np.ones((mdp.n_states, mdp.n_actions)) / mdp.n_actions
    # soft_policy = soft_policy_2

    # Iterate over each test trajectory
    for traj in test_trajs:
        # Compute log likelihood for dense policy
        total_log_likelihood_dense += compute_likelihood(irl_policy_dense, traj, L, rm, is_irl=True)
        
        # Compute log likelihood for featurized policy
        total_log_likelihood_featurized += compute_likelihood(irl_policy_featurized, traj, L, rm, is_irl=True)

        # Compute log likelihood for uniform policy
        total_log_likelihood_uniform += compute_likelihood(uniform_policy, traj, L, rm, is_irl=False)

        # Compute log likelihood for soft policy
        total_log_likelihood_soft += compute_likelihood(soft_policy_2, traj, L, rm, is_irl=False)

    # Calculate average log likelihoods
    average_log_likelihood_dense = total_log_likelihood_dense / len(test_trajs)
    average_log_likelihood_featurized = total_log_likelihood_featurized / len(test_trajs)
    average_log_likelihood_uniform = total_log_likelihood_uniform / len(test_trajs)
    average_log_likelihood_soft = total_log_likelihood_soft / len(test_trajs)

    # Print the average log likelihoods
    print(f"The average log likelihood of the dense IRL policy is: {average_log_likelihood_dense}")
    print(f"The average log likelihood of the featurized IRL policy is: {average_log_likelihood_featurized}")
    print(f"The average log likelihood of the uniform policy is: {average_log_likelihood_uniform}")
    print(f"The average log likelihood of the soft policy is: {average_log_likelihood_soft}")