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

import config

if __name__ == '__main__':
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

    rm = RewardMachine(config.RM_PATH)
 
    L = {}

    for state_index in range(bw.num_states):
        if state_index == s2i[config.TARGET_STATE_1]:
            L[state_index] = 'A'
        elif state_index == s2i[config.TARGET_STATE_2]:
            L[state_index] = 'B'
        elif state_index == s2i[config.TARGET_STATE_3]:
            L[state_index] = 'C'
        else:
            L[state_index] = 'I'

    
    mdpRM = MDPRM(mdp,rm,L)
    mdp_ =  mdpRM.construct_product()
    # now we need a state action state reward for the product MDP
    reward = np.zeros((mdp_.n_states, mdp_.n_actions, mdp_.n_states))
    print(f"Reward: {reward.shape}, S: {mdp.n_states}, A: {mdp.n_actions}, RM: {rm.n_states}")

    for bar_s in range(mdp_.n_states):
        for a in range(mdp_.n_actions):
            for bar_s_prime in range(mdp_.n_states):
                (s,u) = mdpRM.su_pair_from_s(bar_s)
                (s_prime,u_prime) = mdpRM.su_pair_from_s(bar_s_prime)

                is_possible = mdp_.P[a][bar_s][bar_s_prime] > 0.0

                if u == 2 and L[s_prime] == 'C':
                    reward[bar_s, a, bar_s_prime] = 100.0
                
    q_soft,v_soft , soft_policy = infinite_horizon_soft_bellman_iteration(mdp_,reward,logging = True)
    print(f"The shape of the policy is: {soft_policy.shape}")
    np.save(os.path.join(os.path.dirname(__file__), "policies", "soft_policy.npy"), soft_policy)