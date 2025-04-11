import os
import sys
import gymnasium as gym
# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Append the parent directory to sys.path
sys.path.append(parent_dir)
import pandas as pd
import numpy as np
from utils.mdp import MDP, MDPRM
from dynamics.BlockWorldMDP import BlocksWorldMDP, infinite_horizon_soft_bellman_iteration

from reacher_env import state_to_idx, action_to_idx, discretize_xy, discretize_action, midpoint_from_idx, midpoint_from_action_idx
from reacher_env import idx_to_action, action_bins

if __name__ == '__main__':

    print(f"The action bins are: {action_bins}")

    transition_matrices = np.load("transition_matrices.npy")

    n_actions, n_states, n_states = transition_matrices.shape
    print(f"The shape of the transition matrices is: {transition_matrices.shape}")
    


   

    P = []

    for a in range(n_actions):
        # print(f"The matrix shape is: {transition_matrices[a,:,:]}")
        Pa = transition_matrices[a,:,:]
        print(f"We have {np.sum(Pa.sum(axis=1))}/{Pa.shape[0]} zero entries in the transition matrix for action {a}")
        # If there's a row in Pa that's all zeros, fix it with 1/n_states
        for i in range(Pa.shape[0]):
            if np.all(Pa[i, :] == 0):
                Pa[i, :] = 1 / n_states
        
        assert np.allclose(Pa.sum(axis=1), 1), "Transition matrix Pa is not Markovian"
        P.append(Pa)

    mdp = MDP(n_states=n_states, n_actions=n_actions,P = P,gamma = 0.9,horizon=10)

    
    target_state = (0.15,0.15)

    target_state_idx = state_to_idx[discretize_xy(target_state)]

    print(f"The target state index is: {target_state_idx}")
 
    # # now we need a state action state reward for the product MDP
    reward = np.zeros((mdp.n_states, mdp.n_actions, mdp.n_states))
    reward[target_state_idx,:,:] = 100.0
    # # print(f"Reward: {reward.shape}, S: {mdp.n_states}, A: {mdp.n_actions}, RM: {rm.n_states}")


                
    # q_soft,v_soft , soft_policy = infinite_horizon_soft_bellman_iteration(mdp,reward,logging = True)
    # print(f"The shape of the policy is: {soft_policy.shape}")
    # # Save the policy to a file
    # np.save("soft_policy.npy", soft_policy)
    # print("Soft policy has been saved to soft_policy.npy")

    # Load the policy from the file
    soft_policy = np.load("soft_policy.npy")
    print("Soft policy has been loaded from soft_policy.npy")

    # # Reset the environment
    env = gym.make('Reacher-v5',render_mode="human")
    obs, info = env.reset()

    # Simulate for 500 steps with a random policy
    for _ in range(1):
        current_xy = env.unwrapped.get_body_com("fingertip")[:2]
        current_state = discretize_xy(current_xy)
        current_state_idx = state_to_idx[current_state]

        action_dist = soft_policy[current_state_idx,:]
            # Sample an action from the action distribution
        a = np.random.choice(np.arange(n_actions), p=action_dist)
        a_disc = idx_to_action[a]
        a_continuous = midpoint_from_action_idx(a_disc)

        # Sample a random action
        obs, reward, terminated, truncated, info = env.step(a_continuous)


        env.render()
        # Print XY coordinates of end-effector (fingertip)
        # Get end-effector position from environment's state
        fingertip_xy = env.unwrapped.get_body_com("fingertip")[:2]
        print(f"End-Effector XY coordinates: {fingertip_xy}")
        # time.sleep(1)

        if terminated or truncated:
            obs, info = env.reset()

    # Close the environment
    env.close()