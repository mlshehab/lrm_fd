import os
import sys

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Append the parent directory to sys.path
sys.path.append(parent_dir)

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




class Simulator():
    
    def __init__(self, rm , mdp, L, policy):
        self.rm = rm
        self.mdp = mdp
        self.L = L
        self.policy = policy
        self.n_states = self.mdp.n_states
        self.n_actions = self.mdp.n_actions
        self.n_nodes = self.rm.n_states

    def sample_next_state(self, state, action):
        """ Generic next state computation. """
        transition_probs = self.mdp.P[action][state, :]
        return np.random.choice(np.arange(self.n_states), p=transition_probs)



class BlockworldSimulator(Simulator):
    def __init__(self, rm, mdp, L, policy, state2index, index2state):
        super().__init__(rm, mdp, L, policy)
        self.state2index = state2index
        self.index2state = index2state

    def sample_trajectory(self, starting_state, len_traj):
       
        # find the state in the reward machine
        traj = []

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
            traj.append((state,a,next_state))

            # print(f"The state was: {self.index2state[state]}")
            # print(f"The action  was: {a}")
            # print(f"The next state is: {self.index2state[next_state]}\n")
            
            if self.L[state] == 'A':
                print("Passed through A !!!")
            elif self.L[state] == 'B':
                print("Passed through B !!!")
            elif self.L[state] == 'C':
                print("Passed through C !!!")


            l = self.L[next_state]
            label = label + l + ','
            u = u_from_obs(label,rm)
    
            state = next_state

import argparse

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
    # print(f"The shape of the policy is: {soft_policy.shape}")
    np.save("soft_policy.npy", soft_policy)
    # soft_policy = np.load("soft_policy.npy")
    bws = BlockworldSimulator(rm = rm,mdp = mdp,L = L,policy = soft_policy,state2index=s2i,index2state=i2s)
    bws.sample_trajectory(starting_state=0,len_traj=20)
    


