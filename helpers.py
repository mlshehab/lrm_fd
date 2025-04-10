import numpy as np
from scipy.stats import entropy
import time
from utils.sat_utils import *
from utils.ne_utils import get_label, u_from_obs
import argparse
import pandas as pd
from z3 import Bool, Solver, Implies, Not, BoolRef, sat,print_matrix, Or, And, AtMost # type: ignore
from itertools import combinations


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-depth', type=int, default=10)
    parser.add_argument('-n_traj', type=int, default=100)
    # Fix the save argument to properly handle boolean values
    parser.add_argument('-save', type=int, choices=[0, 1], default=1,
                       help='0 for False, 1 for True')
    return parser.parse_args()

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
    term1 = np.maximum(1 - ((2**A - 2) * np.exp((-n1 * epsilon_1**2) / (2 ))), 0)
    term2 = np.maximum(1 - ((2**A - 2) * np.exp((-n2 * (epsilon - epsilon_1)**2) / (2 ))), 0)
    return term1 * term2


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


def solve_sat_instance(bws, counter_examples, rm, metric, kappa, AP, p_threshold=0.8):
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

    proposition2index = {'A':0,'B':1,'C':2,'I':3}

    def prefix2indices(s):
        out = []
        for l in s.split(','):
            if l:
                out.append(proposition2index[l])
        return out

    # Filter counter examples by probability threshold
    filtered_counter_examples = {}
    prob_values = []  # Store probability values for visualization

    wrong_ce_counts = 0

    for state, ce_set in counter_examples.items():
        filtered_ce = []
        for ce in ce_set:
            epsilon = similarity(bws.state_action_probs[state][ce[0]], bws.state_action_probs[state][ce[1]], metric)

            n1 = bws.state_label_counts[state][ce[0]] 
            n2 = bws.state_label_counts[state][ce[1]]
            A = bws.n_actions
            
            optimal_epsilon1 = epsilon/2
            prob = f(optimal_epsilon1, n1, n2, A, epsilon)
            prob_values.append(prob)
            
            if prob > p_threshold:
                filtered_ce.append((ce, prob))  # Store probability with counter example
                
        if filtered_ce:
            filtered_counter_examples[state] = filtered_ce

    # Add C4 constraints for filtered counter examples
    for state in filtered_counter_examples.keys():
        ce_set = filtered_counter_examples[state]
        total_constraints += len(ce_set)
        
        for ce, prob in ce_set:

            # print(f"The counter example is: {ce}")
            # print(f"The probability is: {prob}")


            if u_from_obs(ce[0],rm) == u_from_obs(ce[1],rm):
                wrong_ce_counts += 1

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
    
    return solutions, total_constraints, len(filtered_counter_examples), solve_time, prob_values, wrong_ce_counts





def generate_policy_comparison_report(bws, rm, soft_policy, n_traj, max_len, timestamp):
        data = [] 
        for key, item in bws.state_action_probs.items():
            for label, action_prob in item:
                u = u_from_obs(label, rm)
                policy = np.round(action_prob, 3)
                
                true_policy = np.round(soft_policy[u * bws.n_states + key, :], 3)
                kl_div = similarity(policy, true_policy, 'KL')
                l1_norm = similarity(policy, true_policy, 'L1')
                tv_distance = similarity(policy, true_policy, 'TV')
                count = bws.state_label_counts[key][label]

                data.append([key, label, count, policy, true_policy, kl_div, l1_norm, tv_distance])
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=["State", "Label", "Count (n)", "Policy", "True Policy", "KL Divergence", "L1 Norm", "TV Distance"])

        # Compute min and max values for the last three columns
        min_values = df[["KL Divergence", "L1 Norm", "TV Distance"]].min()
        max_values = df[["KL Divergence", "L1 Norm", "TV Distance"]].max()

        # Create summary rows for min and max values
        summary_df = pd.DataFrame({
            "State": ["Min", "Max"],
            "Label": ["-", "-"],
            "Count (n)": ["-", "-"],
            "Policy": ["-", "-"],
            "True Policy": ["-", "-"],
            "KL Divergence": [min_values["KL Divergence"], max_values["KL Divergence"]],
            "L1 Norm": [min_values["L1 Norm"], max_values["L1 Norm"]],
            "TV Distance": [min_values["TV Distance"], max_values["TV Distance"]]
        })

        # Append summary rows to the DataFrame
        df = pd.concat([df, summary_df], ignore_index=True)

        # Save DataFrame to Excel
        df.to_excel(f"./results/test_policy_comparison_nt_{n_traj}_ml_{max_len}_{timestamp}.xlsx", index=False)