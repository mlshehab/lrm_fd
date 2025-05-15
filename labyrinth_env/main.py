import os
import sys

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Append the parent directory to sys.path
sys.path.append(parent_dir)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
import labyrinth_with_stay
from le_helpers import generate_label_combinations
from collections import Counter
from le_helpers import solve_sat_instance

class LabyrinthEnvSimulator():

    def __init__(self, lb, trajs, L):
        self.lb = lb
        self.trajs = trajs
        self.L = L

        self.n_states = self.lb.n_states
        self.n_actions = self.lb.n_actions
        self.actions = self.lb.action_space
        self.home_state = self.lb.home_state
        self.water_port = self.lb.water_port

        self.state_action_counts = {}
        self.state_action_probs = {}
        self.state_label_counts = {}

    @staticmethod
    def remove_consecutive_duplicates(s):
        elements = s.split(',')
        if not elements:
            return s  # Handle edge case
        result = [elements[0]]
        for i in range(1, len(elements)):
            if elements[i] != elements[i - 1]:
                result.append(elements[i])
        return ','.join(result)


    def get_state_action_counts(self):
         
        for t_idx, traj in enumerate(self.trajs):
            label = ''
            for i in range(len(traj['states'])):
                state, action = traj['states'][i], traj['actions'][i]
                label = label + self.L[state] + ','
       
                compressed_label = self.remove_consecutive_duplicates(label)
            

                if state not in self.state_action_counts:
                    self.state_action_counts[state] = []

                label_exists = False
                for i, (existing_label, counter) in enumerate(self.state_action_counts[state]):
                    if existing_label == compressed_label:
                        counter[action] += 1
                        label_exists = True
                        break
    
                if not label_exists:
                    self.state_action_counts[state].append((compressed_label, Counter({action: 1})))

            # print("Done with traj: ", t_idx)
            # print(f"The states are: {traj['states']}")
            # print(f"The actions are: {traj['actions']}")
            # print(f"The label is: {label}")
            # print(f"The compressed label is: {compressed_label}")
            # print(f"-------------------------------- \n\n")
           
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
         
        
if __name__ == "__main__":
    
    GEN_DIR_NAME = './data/mouse_data/'

    TRAJS_DIR_NAME = GEN_DIR_NAME + 'water_restricted_mice_trajs.pickle'
    lb = labyrinth_with_stay.LabyrinthEnv()
    P_a = lb.get_transition_mat()
    N_STATES = P_a.shape[0] # num states in this env


    # Load restricted training indices
    # restricted_train_indices = np.load(GEN_DIR_NAME + 'restricted_train_indices.npy')


    # Load trajectories
    trajs = pd.read_pickle(TRAJS_DIR_NAME)
    # print(len(trajs))


    L = {}

    for state in range(N_STATES):
        if state == lb.home_state:
            L[state] = 'H'
        elif state == lb.water_port:
            L[state] = 'W'
        else:
            L[state] = 'I'

    sim = LabyrinthEnvSimulator(lb, trajs, L)
    sim.get_state_action_counts()
    sim.compute_action_distributions()

    counter_examples = generate_label_combinations(sim)
    
    kappa = 2
    AP = 3
    alpha = 0.001
    solutions, total_constraints,  filtered_counter_examples , solve_time = solve_sat_instance(sim, counter_examples, kappa, AP, alpha)
    # print(f"The number of constraints is: {total_constraints}, { filtered_counter_examples }")
    print(f"The number of solutions is: {len(solutions)}")
    print(f"the solution is: {solutions}")

 