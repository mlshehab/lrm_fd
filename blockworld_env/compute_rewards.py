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

if __name__ == "__main__":
 
    bw = BlocksWorldMDP(num_piles = config.NUM_PILES)

    soft_policy = np.load(config.POLICY_PATH)
     
    n_states = bw.num_states
    n_actions = bw.num_actions

   
    transition_matrices,s2i, i2s = bw.extract_transition_matrices()
    n_states = bw.num_states
    n_actions = bw.num_actions

 

    P = []

    for a in range(n_actions):
        # print(f"The matrix shape is: {transition_matrices[a,:,:]}")
        P.append(transition_matrices[a,:,:])

    mdp = MDP(n_states=n_states, n_actions=n_actions,P = P,gamma = config.GAMMA,horizon=config.HORIZON)

    rm = RewardMachine(config.RM_PATH_WRONG)

    print(f"rm.delta_u = {rm.delta_u}")

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


    P = mdp_.P[0]
    E = np.eye(mdp_.n_states)

    for a in range(1,mdp_.n_actions):
        P = np.vstack((P,mdp_.P[a]))
        E = np.vstack((E, np.eye(mdp_.n_states)))

    print(f"The shape of P is {P.shape}")
    Psi = d(P)

    A = np.hstack((Psi, -E + config.GAMMA*P))
    print(f"The shape of A is {A.shape}")
    print(f"The shape of soft_policy is {soft_policy.shape}")
    print(f"The shape of mdp_.n_states is {mdp_.n_states}")

    # b = np.log(soft_policy)[:mdp_.n_states,:]
    b = np.log(soft_policy)
    # bb = np.log(soft_policy)[:mdp_.n_states,:]

    # b = b.reshape((A.shape[0],1))
    b = b.flatten('F')[:,None]
    



    # construct the feature matrix
    AP = ['A','B','C','I']

    row_F = mdp.n_states**2*rm.n_states**2*mdp.n_actions
    col_F = rm.n_states*len(AP)
    F = np.zeros(shape = (row_F,col_F))

    print(f"The shape of F is: {F.shape}")


    index_to_tuple = create_index_to_tuple_dict(mdp_states = mdp.n_states , rm_states= rm.n_states, actions = mdp.n_actions)


    for j in range(col_F):
        
        u_j, ap_j = get_u_ap_tuple(j, rm.n_states, AP)
       
        for i in range(row_F):
            (s,u,a,s_prime, u_prime) = index_to_tuple[i]

            L_s_prime = L[s_prime]

            if u == u_j and L_s_prime == AP[ap_j] and mdp.P[a][s,s_prime] > 0.0:
                F[i,j] = 1.0

    A = np.hstack((Psi@F, -E + config.GAMMA*P))

    b = np.log(soft_policy) 

  
    b = b.flatten('F')[:,None]

    start = time.time()
    x = np.linalg.lstsq(A,b, rcond = None)
    end = time.time()
    print(f"This took a total of {end - start} secs.")
    # print(f"The residual is: {x[1]}")
    # plt.hist(x[0][:291600])

    print(f"The residual is: {np.linalg.norm(A@x[0]-b)}")
    print(f"The shape of x is: {x[0].shape}")
    reward_vec = x[0][:F.shape[1]]
    print(f"The reward vector is: {np.round(reward_vec + abs(reward_vec.min()), decimals = 3)}")

    # now we need a state action state reward for the product MDP
    reward = np.zeros((mdp_.n_states, mdp_.n_actions, mdp_.n_states))
    # print(f"Reward: {reward.shape}, S: {mdp.n_states}, A: {mdp.n_actions}, RM: {rm.n_states}")


    ap2index = {'A':0,'B':1,'C':2,'I':3}

    for bar_s in range(mdp_.n_states):
        for a in range(mdp_.n_actions):
            for bar_s_prime in range(mdp_.n_states):
                (s,u) = mdpRM.su_pair_from_s(bar_s)
                (s_prime,u_prime) = mdpRM.su_pair_from_s(bar_s_prime)

                is_possible = mdp_.P[a][bar_s][bar_s_prime] > 0.0
                
                lsp = L[s_prime]
                ap_index = ap2index[lsp]

                # if is_possible:
                reward[bar_s,a,bar_s_prime] = reward_vec[u * len(AP) + ap_index]


    q_soft,v_soft , soft_policy_special_reward = infinite_horizon_soft_bellman_iteration(mdp_,reward,logging = False)

    out = np.log(soft_policy_special_reward).flatten('F')[:,None]

    print(f"The norm difference between the policies is: {np.linalg.norm(out - b)}")

    # print(f"The value of x is: {np.round(x[0][:F.shape[1]]+ abs(x[0][:F.shape[1]].min()),decimals = 3)}")
    # np.save('x.npy',x)


    # # Test the dictionary
    # print(index_to_tuple[0])  # Output: (0, 0, 0, 0, 0)
    # print(index_to_tuple[1])  # Output: (0, 0, 0, 1, 0)
    # print(index_to_tuple[5])  # Output: (0, 0, 0, 2, 1)














# for state in range(n_states):
#     r1 = soft_policy[state,:]
#     r2 = soft_policy[n_states + state,:]
#     r3 = soft_policy[2*n_states + state,:]
#     r4 = soft_policy[3*n_states + state,:]

#     # r1_2 = soft_policy_d_avd_2[state,:]
#     # r2_2 = soft_policy_d_avd_2[n_states + state,:]
#     # r3_2 = soft_policy_d_avd_2[2*n_states + state,:]
#     # r4_2 = soft_policy_d_avd_2[3*n_states + state,:]

#     # print(f"r1_1 = {r1_1} and r1_2 = {r1_2}")
#     if norm(r1-r3) <= 1e-6:
#         print(f"At state {state}, we have r1 = r3.")
#     if norm(r1-r4) <= 1e-6:
#         print(f"At state {state}, we have r1 = r4.")
#     if norm(r2-r3) <= 1e-6:
#         print(f"At state {state}, we have r2 = r3.")
#     if norm(r2-r4) <= 1e-6:
#         print(f"At state {state}, we have r2 = r4.")
#     if norm(r3-r4) <= 1e-6:
#         print(f"At state {state}, we have r3 = r4.")
 
# for node in range(3):
#     for s in range(n_states):
#     # for node in range(3):
#         print(f"Log policy at state = {s}, node = {node} is: {np.round(np.log(soft_policy[node*n_states + state,:]),decimals  = 3)}")

# for i in range(n_nodes):
#     matrices.append(soft_policy[i*n_states:(i+1)*n_states,:])
# print(len(matrices))

# # matrix1 = np.random.rand(10, 10)
# # matrix2 = np.random.rand(10, 10)
# # matrix3 = np.random.rand(10, 10)
# # matrix4 = np.random.rand(10, 10)

# # # Combine matrices into a list
# # matrices = [matrix1, matrix2, matrix3, matrix4]

# # Titles for each subplot
# titles = ['Matrix 1', 'Matrix 2', 'Matrix 3', 'Matrix 4']

# # Create a figure with subplots
# fig, axes = plt.subplots(1, 3, figsize=(20, 5))  # 1 row, 4 columns

# # Plot each matrix as a heatmap
# for i, ax in enumerate(axes):
#     heatmap = ax.imshow(matrices[i], cmap='viridis', aspect='auto')
#     ax.set_title(titles[i])
#     ax.axis('off')  # Turn off axis ticks and labels
#     fig.colorbar(heatmap, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)

# # Adjust layout
# plt.tight_layout()
# plt.show()
