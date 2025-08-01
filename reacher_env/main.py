import numpy as np
from scipy.stats import entropy
import time
import os
import sys
from scipy.optimize import minimize_scalar
# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Append the parent directory to sys.path
sys.path.append(parent_dir)
from reward_machine.reward_machine import RewardMachine
from utils.sat_utils import *
from utils.ne_utils import get_label, u_from_obs
import argparse
import pandas as pd
from z3 import Optimize, Bool, Solver, Implies, Not, BoolRef, sat,print_matrix, Or, And, AtMost # type: ignore
from itertools import combinations
from tqdm import tqdm
import pickle
from simulator import ReacherDiscretizerUniform, ReacherDiscreteSimulator, ForceRandomizedReacher
import matplotlib.pyplot as plt
import config

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


def prepare_sat_problem(rds, counter_examples, alpha, ground_truth_rm):
    
     
    filtered = {}
    FP_count = 0
    for state, ces in tqdm(counter_examples.items()):
        kept = []
        for ce in ces:
            # previous‐method test
            eps = similarity(
                rds.state_action_probs[state][ce[0]],
                rds.state_action_probs[state][ce[1]],
                metric="L1")
            n1 = rds.state_label_counts[state][ce[0]]
            n2 = rds.state_label_counts[state][ce[1]]
            A = rds.rd.n_actions
            prob = f(eps/2, n1, n2, A, eps)

            if prob > 1 - alpha:
                if u_from_obs(ce[0],ground_truth_rm) == u_from_obs(ce[1],ground_truth_rm):
                    FP_count += 1
                kept.append(ce)
        if kept:
            filtered[state] = kept
    
    c4_clauses = []
    for state, ces in filtered.items():
        for ce in ces:
            c4_clauses.append(ce)
    
    return  c4_clauses, FP_count  


def solve_with_clauses(c4_clauses, print_solutions=False):
    """
    Given a list of C4 clauses, add all of them as constraints and
    enumerate all satisfying assignments. Returns the count of solutions.
    """
    # helper to convert prefixes to indices
    proposition2index = {'B': 0, 'R': 1, 'Y': 2, 'I': 3}
    def prefix2indices(lbl):
        return [proposition2index[p] for p in lbl.split(',') if p]

    # 1) SAT variables
    B = [[[Bool(f"x_{i}_{j}_{k}") for j in range(kappa)]
           for i in range(kappa)]
           for k in range(AP)]

    # fixed vector x (as before)
    x = [False] * kappa
    x[0] = True

    # 2) base solver with C1 & C2
    s = Solver()
    # C1: x[i][j][k] ⇒ x[j][j][k]
    for ap in range(AP):
        for i in range(kappa):
            for j in range(kappa):
                s.add(Implies(B[ap][i][j], B[ap][j][j]))
    # C2: exactly one entry per row
    for ap in range(AP):
        s.add(one_entry_per_row(B[ap]))

    # 3) Add all provided C4 clauses
    for ce in c4_clauses:
        p1 = prefix2indices(ce[0])
        p2 = prefix2indices(ce[1])
        sub1 = bool_matrix_mult_from_indices(B, p1, x)
        sub2 = bool_matrix_mult_from_indices(B, p2, x)
        res_ = element_wise_and_boolean_vectors(sub1, sub2)
        for elt in res_:
            s.add(Not(elt))

    # 4) enumerate all models
    count = 0
    while s.check() == sat:
        m = s.model()
        # Print the current solution
        if print_solutions:
            print("\nSolution found:")
            for ap in range(AP):
                print(f"\nAP {ap}:")
                for i in range(kappa):
                    row = []
                    for j in range(kappa):
                        val = m.evaluate(B[ap][i][j], model_completion=True)
                        row.append(1 if val else 0)
                    print(row)
        
        # block current model
        block_clause = []
        for ap in range(AP):
            for i in range(kappa):
                for j in range(kappa):
                    block_clause.append(
                        B[ap][i][j] != m.evaluate(B[ap][i][j], model_completion=True)
                    )
        s.add(Or(block_clause))
        count += 1
    
    return count

def maxsat_clauses(all_clauses):
    """
    Given a list of C4 clauses and the reward machine `rm`,
    build a MaxSAT problem that tries to include the largest possible
    subset of those clauses. Returns the list of clauses Z3 chose.
    """
    # helper to convert prefixes to indices
    proposition2index = {'B': 0, 'R': 1, 'Y': 2, 'I': 3}
    def prefix2indices(lbl):
        return [proposition2index[p] for p in lbl.split(',') if p]

    # 1) create the selector vars
    selectors = [Bool(f"sel_{i}") for i in range(len(all_clauses))]

    # 2) build an Optimize (MaxSAT) object
    opt = Optimize()

    # 3) HARD: C1 & C2 exactly as before
    B = [[[Bool(f"x_{i}_{j}_{k}") for j in range(kappa)]
           for i in range(kappa)]
           for k in range(AP)]
    x = [False]*kappa
    x[0] = True

    # C1
    for ap in range(AP):
        for i in range(kappa):
            for j in range(kappa):
                opt.add(Implies(B[ap][i][j], B[ap][j][j]))
    # C2
    for ap in range(AP):
        opt.add(one_entry_per_row(B[ap]))

    # 4) for each C4 clause, add a guarded hard‐part and a soft selector
    for idx, ce in enumerate(all_clauses):
        sel = selectors[idx]

        # build the actual Boolean constraints for this clause
        p1 = prefix2indices(ce[0])
        p2 = prefix2indices(ce[1])
        sub1 = bool_matrix_mult_from_indices(B, p1, x)
        sub2 = bool_matrix_mult_from_indices(B, p2, x)
        res_ = element_wise_and_boolean_vectors(sub1, sub2)

        # if sel is true, we must forbid every elt in res_
        for elt in res_:
            opt.add(Implies(sel, Not(elt)))

        # make sel a soft constraint of weight=1
        opt.add_soft(sel, weight=1, id=f"clause_{idx}")

    # 5) run MaxSAT
    opt.check()
    m = opt.model()

    # 6) collect which clauses were kept
    chosen = [all_clauses[i]
              for i, sel in enumerate(selectors)
              if m.evaluate(sel)]

    return chosen

 

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--alpha', type=float, default=0.0001)
    parser.add_argument('--print', action='store_true', default=False)
    args = parser.parse_args()
 
    rm = RewardMachine(config.RM_PATH)

    with open(config.SIM_DATA_PATH, "rb") as foo:
        rds = pickle.load(foo)

    counter_examples = generate_label_combinations(rds)

    kappa = config.KAPPA
    AP = config.AP
    
    alpha = args.alpha
    

    c4_clauses, FP_count = prepare_sat_problem(rds, counter_examples, alpha, rm)

  
    
    maxsat_clauses = maxsat_clauses(c4_clauses)
    print(f"The total number of negative examples is: {len(c4_clauses)}")
    print(f"The false positive rate is: {np.round(100*FP_count/len(c4_clauses), 3)}%")
    print(f"The number of solutions in the maxsat set is: {solve_with_clauses(maxsat_clauses, args.print)}")
       


 

