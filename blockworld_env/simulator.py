
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
from dynamics.BlockWorldMDP import BlocksWorldMDP

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

     


class BlockworldSimulator(Simulator):
    def __init__(self, rm, mdp, L, policy, state2index, index2state):
        super().__init__(rm, mdp, L, policy)
        self.state2index = state2index
        self.index2state = index2state

    def remove_consecutive_duplicates(self, s):
        elements = s.split(',')
        if not elements:
            return s  # Handle edge case
        result = [elements[0]]
        for i in range(1, len(elements)):
            if elements[i] != elements[i - 1]:
                result.append(elements[i])
        return ','.join(result)

    def sample_trajectory(self, len_traj):
       
        starting_state = np.random.randint(0, self.n_states)
        # if starting_state ==0 :
        #     print(f"The starting state is: {self.L[starting_state]}")
        state = starting_state
        label = self.L[state] + ','
        compressed_label = self.remove_consecutive_duplicates(label)
        u = u_from_obs(label,self.rm)
        # print(f"The initial state is: {state}, the initial label is: ({label}), the initial u is: {u}")
        
        for _ in range(len_traj):
            idx = u * self.n_states + state
            action_dist = self.policy[idx,:]

            
            # Sample an action from the action distribution
            a = np.random.choice(np.arange(self.n_actions), p=action_dist)
            
            # Sample a next state 
            next_state = self.sample_next_state(state, a)
            # traj.append((state,a,next_state))

            # Compress the label
            compressed_label = self.remove_consecutive_duplicates(label)

            # if state == 0:
            #     print(f"current state: {state}, current label: ({compressed_label}), current u is: {u}")
            #     print(f"action distribution: {np.round(action_dist, 3)}\n")

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

    def sample_dataset(self, starting_states, number_of_trajectories, max_trajectory_length):
        # for each starting state
        
        # for each length trajectory
        for l in range(max_trajectory_length):
            # sample (number_of_trajectories) trajectories of length l 
            for _ in range(number_of_trajectories):
                self.sample_trajectory(len_traj= l)



from reward_machine.reward_machine import RewardMachine
import config
from dynamics.BlockWorldMDP import BlocksWorldMDP
from utils.mdp import MDP

if __name__ == "__main__":

    
    rm = RewardMachine(config.RM_PATH)
    # print(f" The node is: {u_from_obs('A,I,B,I,C,I,A,',rm)}")

    policy = np.load(config.POLICY_PATH)
    # mdp = BlocksWorldMDP(num_piles=config.NUM_PILES)
    bw = BlocksWorldMDP(num_piles=config.NUM_PILES)

    transition_matrices, s2i, i2s = bw.extract_transition_matrices()

    L = {
        s2i[config.TARGET_STATE_1]: 'A',
        s2i[config.TARGET_STATE_2]: 'B',
        s2i[config.TARGET_STATE_3]: 'C'
    }
    n_states = bw.num_states
    for s in range(n_states):
        if s not in L:
            L[s] = 'I'

    n_states = bw.num_states
    n_actions = bw.num_actions

    P = []

    for a in range(n_actions):
       
        P.append(transition_matrices[a,:,:])

    mdp = MDP(n_states=n_states, n_actions=n_actions,P = P,gamma = config.GAMMA,horizon=config.HORIZON)

    rm = RewardMachine(config.RM_PATH)

    starting_states = [s2i[config.TARGET_STATE_1], s2i[config.TARGET_STATE_2], s2i[config.TARGET_STATE_3], 4, 24]
    simulator = BlockworldSimulator(rm=rm, mdp=mdp, L=L, policy=policy, state2index=s2i, index2state=i2s)
    # simulator.sample_dataset(starting_states, number_of_trajectories=1000, max_trajectory_length=100)
    simulator.sample_trajectory(starting_state=s2i[config.TARGET_STATE_1], len_traj=25)
