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
from dynamics.BlockWorldMDP import BlocksWorldMDP, infinite_horizon_soft_bellman_iteration
from utils.ne_utils import u_from_obs

from MaxEntIRL import InfiniteHorizonMaxEntIRLWeighted, InfiniteHorizonMaxEntIRL


def d(P):
    m,n = P.shape
    dP = np.zeros((m,m*n))
    for i in range(m):
        dP[i,n*i:n*(i+1)] = P[i]
    return dP


def get_u_ap_tuple(j, rm_states, ap_list):
    
    ap_len = len(ap_list)
    u = j // ap_len  # Calculate the RM state (u)
    ap = j % ap_len  # Calculate the AP index
    return (u, ap)


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


def learn_reward_function(policy, rm, mdp, L, invL):

    
    mdpRM = MDPRM(mdp,rm,L)
    mdp_ =  mdpRM.construct_product()

  
    P = mdp_.P[0]
    E = np.eye(mdp_.n_states)

    for a in range(1,mdp_.n_actions):
        P = np.vstack((P,mdp_.P[a]))
        E = np.vstack((E, np.eye(mdp_.n_states)))

    Psi = d(P)

    AP = ['A','B','C','D']
    ap2index = {'A':0,'B':1,'C':2,'D':3}

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

    b = np.log(policy) 

    b = b.flatten('F')[:,None]

    
    x = np.linalg.lstsq(A,b, rcond = None)
    residual = np.linalg.norm(A @ x[0] - b)
    # print(f"Residual of least squares solution: {residual}")
     
     
    reward_vec = x[0][:F.shape[1]]
    # Print the reward vector in a more readable way: node <u> AP '<ap>': <reward>
    reward_vec_shifted = reward_vec + abs(reward_vec.min())
    # print(f"The reward vector is: {reward_vec_shifted}")
    # for j in range(F.shape[1]):
    #     u_j, ap_j = get_u_ap_tuple(j, rm.n_states, AP)
    #     ap_label = AP[ap_j]
    #     print(f"node {u_j} AP '{ap_label}': {reward_vec_shifted[j][0]}")

    return reward_vec_shifted


def learn_policy_from_reward_function(reward_vec, rm, mdp, L, AP):
    ap2index = {ap: idx for idx, ap in enumerate(AP)}  # Define ap2index
    
    mdpRM = MDPRM(mdp,rm,L)
    mdp_ =  mdpRM.construct_product()
    
    reward_t = np.zeros((mdp_.n_states, mdp_.n_actions, mdp_.n_states))

    for bar_s in range(mdp_.n_states):
        for a in range(mdp_.n_actions):
            for bar_s_prime in range(mdp_.n_states):
                (s,u) = mdpRM.su_pair_from_s(bar_s)
                (s_prime,u_prime) = mdpRM.su_pair_from_s(bar_s_prime)

                reward_t[bar_s, a, bar_s_prime] = reward_vec[u*len(AP) + ap2index[L[s_prime]]][0]

    q_soft,v_soft , soft_policy = infinite_horizon_soft_bellman_iteration(mdp_,reward_t,logging = False)

    
    return soft_policy
    
    
def compute_likelihood(policy, traj,L, rm, gws, is_irl=False) -> float:
    """Computes the likelihood of the observed data (visit_counts, action_counts) under the given policy."""
  
    states, actions = traj['states'], traj['actions']

    # Compute log-likelihood
    log_likelihood = 0.0
    T = len(states)

    label = ''
    n_states = gws.n_states

    
    for s, a in zip(states, actions):
        label += L[s] + ","
        comp = gws.remove_consecutive_duplicates(label)
        u = u_from_obs(comp, rm)
        
        idx = u * n_states + s
        if not is_irl:
            log_likelihood += np.log(policy[idx, a])
        else:
            log_likelihood += np.log(policy[s, a]) 
   
    
    return log_likelihood    
    

from MaxEntIRL import remove_consecutive_duplicates

def generate_expert_trajs_for_irl(env, rm, L, n_traj, t_len, soft_optimal_policy):
    expert_trajs = []

    for _ in range(n_traj):
        traj = []
        state = np.random.randint(0, env.n_states)
        label = L[state] + ','
        compressed_label = remove_consecutive_duplicates(label)
        u = u_from_obs(label, rm)
        for _ in range(t_len):  # Enough steps to simulate stationarity
            idx = u * env.n_states + state
            action_dist = soft_optimal_policy[idx,:]
            
            # Sample an action from the action distribution
            a = np.random.choice(np.arange(env.n_actions), p=action_dist)

            traj.append((state, a))

            transition_probs = env.transition_probability[:,a,:]
            next_state = np.random.choice(np.arange(env.n_states), p=transition_probs[state])

            # Compress the label
            compressed_label = remove_consecutive_duplicates(label)
            l = L[next_state]
            label = label + l + ','
            u = u_from_obs(label, rm)
            
            state = next_state
        # print(compressed_label)
        expert_trajs.append(traj)
    
    return expert_trajs
    


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
    AP = ['A','B','C','D']
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

    
    max_len = args.depth
    n_traj = args.n_traj

  
    
    rm_maxsat_4 = rm

    rm_maxsat_3 = RewardMachine(config.RM_PATH_MAXSAT_3)
    rm_maxsat_2 = RewardMachine(config.RM_PATH_MAXSAT_2)
    rm_maxsat_1 = RewardMachine(config.RM_PATH_MAXSAT_1)
   
    
    

    max_len = config.DEPTH_FOR_CONSTRUCTING_PRODUCT_POLICY
    
    
    learned_policy_IRL = np.load(config.IRL_POLICY_PATH + ".npy")

    learned_product_policy_4 = construct_learned_product_policy(mdp, rm_maxsat_4, max_len, soft_policy, rm, invL, L)
    learned_product_policy_3 = construct_learned_product_policy(mdp, rm_maxsat_3, max_len, soft_policy, rm, invL, L)
    learned_product_policy_2 = construct_learned_product_policy(mdp, rm_maxsat_2, max_len, soft_policy, rm, invL, L)
    learned_product_policy_1 = construct_learned_product_policy(mdp, rm_maxsat_1, max_len, soft_policy, rm, invL, L)
       
    print("Learning reward functions for different RM sizes ...")
    learned_reward_vec_4 = learn_reward_function(learned_product_policy_4, rm_maxsat_4, mdp, L, invL)
    learned_reward_vec_3 = learn_reward_function(learned_product_policy_3, rm_maxsat_3, mdp, L, invL)
    learned_reward_vec_2 = learn_reward_function(learned_product_policy_2, rm_maxsat_2, mdp, L, invL)
    learned_reward_vec_1 = learn_reward_function(learned_product_policy_1, rm_maxsat_1, mdp, L, invL)
    
    n_traj_irl = 2500
    max_len_irl = 15

    expert_trajs_irl = generate_expert_trajs_for_irl(gw, rm, L, n_traj_irl, max_len_irl, soft_policy)


   
    irl_dense = InfiniteHorizonMaxEntIRLWeighted(mdp, expert_trajs_irl, is_featurized=False,invL=invL, n_iter=100, lr=0.1)
    irl_featurized = InfiniteHorizonMaxEntIRLWeighted(mdp, expert_trajs_irl, is_featurized=True,invL=invL, n_iter= 100, lr=0.1)


    
    learned_irl_weights_dense = irl_dense.train(verbose=True)
    learned_irl_weights_featurized = irl_featurized.train(verbose=True)

    

    irl_policy_dense = irl_dense.soft_value_iteration()
    irl_policy_featurized = irl_featurized.soft_value_iteration()

    # Save the IRL policies to disk
    np.save("./policies/irl_policy_dense.npy", irl_policy_dense)
    np.save("./policies/irl_policy_featurized.npy", irl_policy_featurized)

    print("IRL policies saved to disk")

    # Create a new mdp with shuffled rooms 
    Lt = {}

    # The grid numbering and labeling is :
    # 0 4 8 12    B B A A 
    # 1 5 9 13    B B A A 
    # 2 6 10 14   C C D D
    # 3 7 11 15   C C D D
    
    Lt[0], Lt[4], Lt[1], Lt[5] = 'B', 'B', 'B', 'B'
    Lt[8], Lt[12], Lt[9], Lt[13] = 'A', 'A', 'A', 'A'
    Lt[2], Lt[6], Lt[3], Lt[7] = 'C', 'C', 'C', 'C'
    Lt[10], Lt[14], Lt[11], Lt[15] = 'D', 'D', 'D', 'D'
        
    invLt = {'A':[8, 9, 12, 13], 'B':[0, 1, 4, 5], 'C':[2, 6, 3, 7], 'D':[10, 11, 14, 15]}
    

    irl_dense_transfer = InfiniteHorizonMaxEntIRLWeighted(mdp, expert_trajs_irl, is_featurized=False,invL=invLt, n_iter=100, lr=0.1)
    irl_featurized_transfer = InfiniteHorizonMaxEntIRLWeighted(mdp, expert_trajs_irl, is_featurized=True,invL=invLt, n_iter=100, lr=0.1)


    # force the learned weights to be the same as the learned weights for the original environment
    irl_dense_transfer.weights = learned_irl_weights_dense
    irl_featurized_transfer.weights = learned_irl_weights_featurized
 


    irl_policy_dense_transfer = irl_dense_transfer.soft_value_iteration()
    irl_policy_featurized_transfer = irl_featurized_transfer.soft_value_iteration()



    mdpRMt = MDPRM(mdp,rm,Lt)
    mdp_t =  mdpRMt.construct_product()
    
    reward_t = np.zeros((mdp_t.n_states, mdp_t.n_actions, mdp_t.n_states))

    for bar_s in range(mdp_t.n_states):
        for a in range(mdp_t.n_actions):
            for bar_s_prime in range(mdp_t.n_states):
                (s,u) = mdpRMt.su_pair_from_s(bar_s)
                (s_prime,u_prime) = mdpRMt.su_pair_from_s(bar_s_prime)

                if u == 3 and Lt[s_prime] == 'D':
                    reward_t[bar_s, a, bar_s_prime] = config.REWARD_PARAMETER
    
    q_soft,v_soft , soft_policy_transfer_gt = infinite_horizon_soft_bellman_iteration(mdp_t,reward_t,logging = False)
    
    learned_policy_4 = learn_policy_from_reward_function(learned_reward_vec_4, rm_maxsat_4, mdp, Lt, AP)
    learned_policy_3 = learn_policy_from_reward_function(learned_reward_vec_3, rm_maxsat_3, mdp, Lt, AP)
    learned_policy_2 = learn_policy_from_reward_function(learned_reward_vec_2, rm_maxsat_2, mdp, Lt, AP)
    learned_policy_1 = learn_policy_from_reward_function(learned_reward_vec_1, rm_maxsat_1, mdp, Lt, AP)

    simulator = GridworldSimulator(rm=rm, mdp=mdp, L=Lt, policy=soft_policy_transfer_gt)

    starting_states = np.arange(n_states)

    # traj = simulator.sample_trajectory_for_transferability(starting_state=1, len_traj=15)

    n_trajectories = 100
    dataset = simulator.sample_dataset_for_transferability(starting_states= starting_states, n_trajectories=n_trajectories, max_trajectory_length=15, seed=123)
    

    # Compute the likelihood of the trajectory under a uniform policy
    n_states = mdp.n_states
    n_actions = mdp.n_actions
    uniform_policy = np.full((n_states, n_actions), 1.0 / n_actions)

    ll_gt = 0.0
    ll_4 = 0.0
    ll_3 = 0.0
    ll_2 = 0.0
    ll_1 = 0.0
    # ll_irl = 0.0
    ll_irl_dense_transfer = 0.0
    ll_irl_featurized_transfer = 0.0
    ll_irl_basic_transfer = 0.0
    ll_uniform = 0.0
    
    for traj in tqdm(dataset):
        ll_gt += compute_likelihood(soft_policy_transfer_gt, traj, Lt, rm, simulator, is_irl=False)
        ll_4 += compute_likelihood(learned_policy_4, traj, Lt, rm_maxsat_4, simulator, is_irl=False)
        ll_3 += compute_likelihood(learned_policy_3, traj, Lt, rm_maxsat_3, simulator, is_irl=False)
        ll_2 += compute_likelihood(learned_policy_2, traj, Lt, rm_maxsat_2, simulator, is_irl=False)
        ll_1 += compute_likelihood(learned_policy_1, traj, Lt, rm_maxsat_1, simulator, is_irl=False)
        ll_irl_dense_transfer += compute_likelihood(irl_policy_dense_transfer, traj, Lt, rm, simulator, is_irl=True)
        ll_irl_featurized_transfer += compute_likelihood(irl_policy_featurized_transfer, traj, Lt, rm, simulator, is_irl=True)
        ll_uniform += compute_likelihood(uniform_policy, traj, Lt, rm, simulator, is_irl=True)

    print(f"The likelihood of the trajectory under true: {ll_gt/n_trajectories}")
    print(f"The likelihood of the trajectory under learned umax=4: {ll_4/n_trajectories}")
    print(f"The likelihood of the trajectory under learned umax=3: {ll_3/n_trajectories}")
    print(f"The likelihood of the trajectory under learned umax=2: {ll_2/n_trajectories}")
    print(f"The likelihood of the trajectory under learned umax=1: {ll_1/n_trajectories}")
    print(f"The likelihood of the trajectory under a uniform policy: {ll_uniform/n_trajectories}")
    print(f"The likelihood of the trajectory under IRL dense transfer: {ll_irl_dense_transfer/n_trajectories}")
    print(f"The likelihood of the trajectory under IRL featurized transfer: {ll_irl_featurized_transfer/n_trajectories}")
    print(f"The likelihood of the trajectory under IRL basic transfer: {ll_irl_basic_transfer/n_trajectories}")





   
    
    
    
