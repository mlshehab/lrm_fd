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
from dynamics.GridWorld import BasicGridWorld
import argparse
import config
from gwe_helpers import infinite_horizon_soft_policy_evaluation

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()    
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()

    # Create policies directory if it doesn't exist
    os.makedirs(os.path.join(os.path.dirname(__file__), "policies"), exist_ok=True)
     
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

    mdp = MDP(n_states=n_states, n_actions=n_actions,P = P,gamma = config.GAMMA,horizon=config.HORIZON)

     
    policy_path = config.POLICY_PATH
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
    
  
    mdpRM = MDPRM(mdp,rm,L)
    mdp_ =  mdpRM.construct_product()
 
    reward = np.zeros((mdp_.n_states, mdp_.n_actions, mdp_.n_states))

    
     
    for bar_s in range(mdp_.n_states):
        for a in range(mdp_.n_actions):
            for bar_s_prime in range(mdp_.n_states):
                (s,u) = mdpRM.su_pair_from_s(bar_s)
                (s_prime,u_prime) = mdpRM.su_pair_from_s(bar_s_prime)

                if u == 3 and L[s_prime] == 'D':
                    reward[bar_s, a, bar_s_prime] = config.REWARD_PARAMETER
    
   
    q_soft,v_soft , soft_policy = infinite_horizon_soft_bellman_iteration(mdp_,reward,logging = False)
         
    q_soft_e, v_soft_e = infinite_horizon_soft_policy_evaluation(mdp_,reward,soft_policy, logging = False)
    

    random_soft_policy = np.random.randn(mdp_.n_states,mdp_.n_actions)
    print(f"The shape of the policy is: {random_soft_policy.shape}")
    random_soft_policy = np.exp(random_soft_policy) / np.sum(np.exp(random_soft_policy), axis=1, keepdims=True)
    q_soft_e, v_soft_e = infinite_horizon_soft_policy_evaluation(mdp_,reward,random_soft_policy, logging = False)

    # print("v_soft:", np.round(v_soft, 4))
    # print("v_soft_e:", np.round(v_soft_e, 4))
    print(f"The norm of the difference between the two value functions is: {np.linalg.norm(v_soft - v_soft_e)}")

    if args.save:
        print(f"The policy is saved to {policy_path}.npy")
        np.save(os.path.join(os.path.dirname(__file__), policy_path + ".npy"), soft_policy)


    