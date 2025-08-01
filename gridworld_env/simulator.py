
import os
import sys
from scipy.optimize import minimize_scalar
# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import numpy as np
from tqdm import tqdm
from collections import Counter
from scipy.stats import entropy
from utils.ne_utils import u_from_obs
from dynamics.GridWorld import BasicGridWorld
import config

class Simulator():
    
    def __init__(self, rm , mdp, L, policy):
        self.rm = rm
        self.mdp = mdp
        self.L = L
        self.policy = policy
        self.n_states = self.mdp.n_states
        self.n_actions = self.mdp.n_actions
        self.n_nodes = self.rm.n_states
        self.state_action_counts = {}  # Dictionary to track actions for (label, state)
        self.state_action_probs = {}
        self.state_label_counts = {}

    def sample_next_state(self, state, action):
        """ Generic next state computation. """
        transition_probs = self.mdp.P[action][state, :]
        return np.random.choice(np.arange(self.n_states), p=transition_probs)

    def compute_action_distributions(self):
        """
        Converts each action Counter into a probability distribution over all possible actions.

        Returns:
            dict: {state: [(label, action_probs)]}, where action_probs is a numpy array of size (self.n_actions,).
        """
        # state_action_probs = {}

        for state, label_counts in self.state_action_counts.items():
            self.state_action_probs[state] = {}
            self.state_label_counts[state] = {}

            for label, action_counter in label_counts:
                total_actions = sum(action_counter.values())  # Total samples for this label-state pair
                action_probs = np.zeros(self.n_actions)  # Initialize with zeros for all actions
                
                if total_actions > 0:
                    for action, count in action_counter.items():
                        action_probs[action] = count / total_actions  # Normalize

                self.state_action_probs[state][label] = action_probs
                self.state_label_counts[state][label] = total_actions

     


class GridworldSimulator(Simulator):
    def __init__(self, rm, mdp, L, policy):
        super().__init__(rm, mdp, L, policy)
   

    def remove_consecutive_duplicates(self, s):
        elements = s.split(',')
        if not elements:
            return s  # Handle edge case
        result = [elements[0]]
        for i in range(1, len(elements)):
            if elements[i] != elements[i - 1]:
                result.append(elements[i])
        return ','.join(result)

    def sample_trajectory(self, starting_state, len_traj):
       
 
        state = starting_state
        label = self.L[state] + ','
        compressed_label = self.remove_consecutive_duplicates(label)
        u = u_from_obs(label,self.rm)
   
        for _ in range(len_traj):
            idx = u * self.n_states + state
            action_dist = self.policy[idx,:]

            # Sample an action from the action distribution
            a = np.random.choice(np.arange(self.n_actions), p=action_dist)
            
            # Sample a next state 
            next_state = self.sample_next_state(state, a)
          

            # Compress the label
            compressed_label = self.remove_consecutive_duplicates(label)

           
            # Ensure state exists in dictionary
            if state not in self.state_action_counts:
                self.state_action_counts[state] = []

            # Check if this label already exists for the state
            label_exists = False
            for i, (existing_label, counter) in enumerate(self.state_action_counts[state]):
                if existing_label == compressed_label:
                    counter[a] += 1  # Update action count
                    label_exists = True
                    break
            
            # If the label was not found, add a new entry
            if not label_exists:
                self.state_action_counts[state].append((compressed_label, Counter({a: 1})))
            
            l = self.L[next_state]
            label = label + l + ','
            u = u_from_obs(label, self.rm)
            
            state = next_state
        # print(f"The trajectory is: {compressed_label}")

    def sample_dataset(self, starting_states, number_of_trajectories, max_trajectory_length, seed = None):
        if seed is not None:
            np.random.seed(seed)
        
        for _ in range(number_of_trajectories):
            ss = np.random.randint(0, len(starting_states))
            self.sample_trajectory(starting_state=ss, len_traj= max_trajectory_length)


    
from reward_machine.reward_machine import RewardMachine
import config
 
from utils.mdp import MDP

if __name__ == "__main__":

    
    rm = RewardMachine(config.RM_PATH)
     

    policy = np.load(config.POLICY_PATH + ".npy")
     
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
    

    print(f"The transition probability is: {gw.transition_probability[:,0,:]}")



    mdp = MDP(n_states=n_states, n_actions=n_actions, P = P,gamma = gw.discount,horizon=10)


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

    # simulator = GridworldSimulator(rm=rm, mdp=mdp, L=L, policy=policy)
    # simulator.sample_trajectory(starting_state=1, len_traj=10)



    
 

  