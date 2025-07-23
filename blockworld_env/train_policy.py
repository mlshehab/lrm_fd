import os
import sys

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Append the parent directory to sys.path
sys.path.append(parent_dir)
import numpy as np
from utils.mdp import MDP, MDPRM
from reward_machine.reward_machine import RewardMachine
from dynamics.BlockWorldMDP import BlocksWorldMDP, infinite_horizon_soft_bellman_iteration
import argparse
import config

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()    
    parser.add_argument('--rm', type=str, default='stack')
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()

    
    # Create policies directory if it doesn't exist
    os.makedirs(os.path.join(os.path.dirname(__file__), "policies"), exist_ok=True)

    bw = BlocksWorldMDP(num_piles=config.NUM_PILES)
    
    transition_matrices,s2i, i2s = bw.extract_transition_matrices()
    n_states = bw.num_states
    n_actions = bw.num_actions

    P = []

    for a in range(n_actions):
        P.append(transition_matrices[a,:,:])

    mdp = MDP(n_states=n_states, n_actions=n_actions,P = P,gamma = config.GAMMA,horizon=config.HORIZON)

    if args.rm == 'stack-adv':
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
    elif args.rm  == 'stack-extra':
        policy_path = config.POLICY_PATH_EXTRA
        rm = RewardMachine(config.RM_PATH_EXTRA)
        L = {
        s2i[config.TARGET_STATE_1]: 'A',
        s2i[config.TARGET_STATE_2]: 'B',
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
  

    
    mdpRM = MDPRM(mdp,rm,L)
    mdp_ =  mdpRM.construct_product()
 
    reward = np.zeros((mdp_.n_states, mdp_.n_actions, mdp_.n_states))

    if args.rm == 'stack-adv':
        # adv policy reward
        for bar_s in range(mdp_.n_states):
            for a in range(mdp_.n_actions):
                for bar_s_prime in range(mdp_.n_states):
                    (s,u) = mdpRM.su_pair_from_s(bar_s)
                    (s_prime,u_prime) = mdpRM.su_pair_from_s(bar_s_prime)

                    if u == 2:

                        reward[bar_s, a, bar_s_prime] = config.REWARD_PARAMETER_ADV_1
                    
                    if u == 0 and u_prime == 3 and L[s_prime] == 'D':
                        reward[bar_s, a, bar_s_prime] = config.REWARD_PARAMETER_ADV_2

                    if u == 1 and u_prime == 3 and L[s_prime] == 'D':
                        reward[bar_s, a, bar_s_prime] = config.REWARD_PARAMETER_ADV_2

    elif args.rm == 'stack-extra':
        for bar_s in range(mdp_.n_states):
            for a in range(mdp_.n_actions):
                for bar_s_prime in range(mdp_.n_states):
                    (s,u) = mdpRM.su_pair_from_s(bar_s)
                    (s_prime,u_prime) = mdpRM.su_pair_from_s(bar_s_prime)

                    if u == 3 and L[s_prime] == 'D':
                        reward[bar_s, a, bar_s_prime] = config.REWARD_PARAMETER
    else:
        
        for bar_s in range(mdp_.n_states):
            for a in range(mdp_.n_actions):
                for bar_s_prime in range(mdp_.n_states):
                    (s,u) = mdpRM.su_pair_from_s(bar_s)
                    (s_prime,u_prime) = mdpRM.su_pair_from_s(bar_s_prime)

                    is_possible = mdp_.P[a][bar_s][bar_s_prime] > 0.0

                    if u == 2 and L[s_prime] == 'C':
                        reward[bar_s, a, bar_s_prime] = config.REWARD_PARAMETER


    if not args.rm == 'stack-extra':
        q_soft,v_soft , soft_policy = infinite_horizon_soft_bellman_iteration(mdp_,reward,logging = False)
        print(f"The shape of the policy is: {soft_policy.shape}")
    
    else:
         
        soft_policy = np.random.randn(mdp_.n_states,mdp_.n_actions)
        print(f"The shape of the policy is: {soft_policy.shape}")
        soft_policy = np.exp(soft_policy) / np.sum(np.exp(soft_policy), axis=1, keepdims=True)

    
    if args.save:
        np.save(os.path.join(os.path.dirname(__file__), policy_path + ".npy"), soft_policy)