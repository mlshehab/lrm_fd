import numpy as np
import sys
sys.path.append('C:/Users/mlshehab/Desktop/lrm_fd')

from itertools import combinations
from scipy.stats import entropy

from utils.sat_utils import *
from utils.ne_utils import get_label, u_from_obs
import argparse
import pandas as pd
from z3 import Bool, Solver, Implies, Not, BoolRef, sat,print_matrix, Or, And, AtMost # type: ignore
from tqdm import tqdm
import time

def similarity(p1, p2, metric):
        """Computes similarity based on chosen metric."""
        if metric == "KL":
            p1 = np.clip(p1, 1e-10, 1)  # Avoid division by zero
            p2 = np.clip(p2, 1e-10, 1)
            return entropy(p1, p2)  # KL divergence
        elif metric == "TV":
            return 0.5 * np.sum(np.abs(p1 - p2))  # Total Variation Distance
        elif metric == "L1":
            return np.linalg.norm(p1 - p2, ord=1)  # 1-norm distance
        else:
            raise ValueError("Unsupported metric! Choose from 'KL', 'TV', 'L1'.")
        
def f(epsilon_1, n1, n2, A, epsilon):
    term1 =    ((2**A - 2) * np.exp((-n1 * epsilon_1**2) / (2 ))) 
    term2 =    ((2**A - 2) * np.exp((-n2 * (epsilon - epsilon_1)**2) / (2 ))) 
    return 1- term1 - term2


def generate_label_combinations(bws):
    """
    Generate a dictionary where each state maps to combinations of labels of length 2 corresponding to it.

    Args:
        bws (BlockworldSimulator): The BlockworldSimulator object.

    Returns:
        dict: {state: [label_combinations]}
    """
    

    label_combinations = {}

    for state, label_dists in bws.state_action_probs.items():
        
        labels = [label for label in label_dists.keys()]
        label_combinations[state] = list(combinations(labels, 2))


    return label_combinations


def solve_sat_instance(sim, counter_examples, kappa, AP, alpha ):
    """
    Solve SAT instance for given counter examples, filtering by probability threshold
    Returns all SAT solutions found
    """
    
    total_variables = kappa**2*AP
    total_constraints = 0

    # Initialize SAT variables
    B = [[[Bool('x_%s_%s_%s'%(i,j,k)) for j in range(kappa)]for i in range(kappa)]for k in range(AP)]
    B_ = element_wise_or_boolean_matrices([b_k for b_k in B])
    x = [False]*kappa
    x[0] = True

    B_T = transpose_boolean_matrix(B_)
    powers_B_T = [boolean_matrix_power(B_T,k) for k in range(1,kappa)]
    powers_B_T_x = [boolean_matrix_vector_multiplication(B,x) for B in powers_B_T]
    powers_B_T_x.insert(0, x)
    OR_powers_B_T_x = element_wise_or_boolean_vectors(powers_B_T_x)
    
    s = Solver()

    # Add constraints
    for ap in range(AP):
        for i in range(kappa):
            for j in range(kappa):
                s.add(Implies(B[ap][i][j], B[ap][j][j]))

    for k in range(AP):
        s.add(one_entry_per_row(B[k]))

    proposition2index = {'H':0,'W':1,'I':2}

    def prefix2indices(s):
        out = []
        for l in s.split(','):
            if l:
                out.append(proposition2index[l])
        return out

    # Filter counter examples by probability threshold
    filtered_counter_examples = {}
    

 
    n_total_previous = 0
    n_total_new = 0
    
    wrong_ce_counts_previous = 0
    wrong_ce_counts_new = 0
    
    for state, ce_set in tqdm(counter_examples.items()):
        filtered_ce_previous = []
        filtered_ce  = []
        wrong_examples = []

        
        for ce in ce_set:
            a = ce[0][-2:]
            b = ce[1][-2:]

            if not a == b:
                print("Something is wrong")
            # method prvious
             
            epsilon = similarity(sim.state_action_probs[state][ce[0]], sim.state_action_probs[state][ce[1]], metric = "L1")

            n1 = sim.state_label_counts[state][ce[0]] 
            n2 = sim.state_label_counts[state][ce[1]]
            A = sim.n_actions
            
            optimal_epsilon1 = epsilon/2
            prob = f(optimal_epsilon1, n1, n2, A, epsilon)
            
            
            if prob > 1 - alpha:
                filtered_ce.append((ce, prob))  # Store probability with counter example
                print(f"\nNegative Example Found:")
                print(f"State: {state}")
                print(f"Label 1: {ce[0]}")
                print(f"Label 2: {ce[1]}")
                print(f"Probability: {prob:.4f}")
                print(f"Sample sizes: n1={n1}, n2={n2}")
                print(f"Epsilon: {epsilon:.4f}")
                print("-" * 50)
                n_total_previous += 1
       
          
                
        if filtered_ce:
            filtered_counter_examples[state] = filtered_ce

    # Add C4 constraints for filtered counter examples
    # print(f"Previous method - Total examples: {n_total_previous}, Wrong examples: {wrong_ce_counts_previous}, Ratio: {wrong_ce_counts_previous/n_total_previous:.2f}")
    # print(f"New method - Total examples: {n_total_new}, Wrong examples: {wrong_ce_counts_new}, Ratio: {wrong_ce_counts_new/n_total_new:.2f}")
    wrong_ce_counts = 0
    for state in tqdm(filtered_counter_examples.keys()):
        ce_set = filtered_counter_examples[state]
        total_constraints += len(ce_set)
        
        for ce, prob in ce_set:

            # print(f"The counter example is: {ce}")
            # print(f"The probability is: {prob}")

            

            p1 = prefix2indices(ce[0])
            p2 = prefix2indices(ce[1])

            sub_B1 = bool_matrix_mult_from_indices(B,p1, x)
            sub_B2 = bool_matrix_mult_from_indices(B,p2, x)
            res_ = element_wise_and_boolean_vectors(sub_B1, sub_B2)

            for elt in res_:
                s.add(Not(elt))
    
   
    # Find all solutions
    solutions = []
    start = time.time()
    while s.check() == sat:
        m = s.model()
        solution = []
        for ap in range(AP):
            r = [[m.evaluate(B[ap][i][j]) for j in range(kappa)] for i in range(kappa)]
            solution.append(r)
        solutions.append(solution)

        block_clause = []
        for ap in range(AP):
            for i in range(kappa):
                for j in range(kappa):
                    block_clause.append(B[ap][i][j] != m.evaluate(B[ap][i][j], model_completion=True))
        s.add(Or(block_clause))

    end = time.time()
    solve_time = end - start
    
    return solutions, total_constraints, len(filtered_counter_examples), solve_time

    # return 0,0,0,0


