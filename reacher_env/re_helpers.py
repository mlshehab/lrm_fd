import numpy as np
from scipy.stats import entropy
import time
from utils.sat_utils import *
from utils.ne_utils import get_label, u_from_obs
import argparse
import pandas as pd
from z3 import Bool, Solver, Implies, Not, BoolRef, sat,print_matrix, Or, And, AtMost # type: ignore
from itertools import combinations
from tqdm import tqdm

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

        # if state == 141766:
        #     print("The list combinations are:")
        #     for combo in label_combinations[141766]:
        #         print(combo[0][-2:] , combo[1][-2:])

    return label_combinations


def solve_sat_instance(bws, counter_examples, rm, kappa, AP, alpha ):
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

    proposition2index = {'B':0,'R':1,'Y':2,'I':3}

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
        filtered_ce_new = []
        wrong_examples = []

        
        for ce in ce_set:
            a = ce[0][-2:]
            b = ce[1][-2:]
            if not a == b:
                print("Something is wrong")
            # method prvious
             
            epsilon = similarity(bws.state_action_probs[state][ce[0]], bws.state_action_probs[state][ce[1]], metric = "L1")

            n1 = bws.state_label_counts[state][ce[0]] 
            n2 = bws.state_label_counts[state][ce[1]]
            A = bws.rd.n_actions
            
            optimal_epsilon1 = epsilon/2
            prob = f(optimal_epsilon1, n1, n2, A, epsilon)
            
            
            if prob >  1 - alpha:
                filtered_ce_previous.append((ce, prob))  # Store probability with counter example
                n_total_previous += 1
                if u_from_obs(ce[0],rm) == u_from_obs(ce[1],rm):
                    wrong_ce_counts_previous += 1
                    wrong_examples.append({
                            'state': state,
                            'counter_example': ce,
                            'policy1': bws.state_action_probs[state][ce[0]],
                            'policy2': bws.state_action_probs[state][ce[1]],
                            'n1': n1,
                            'n2': n2
                        })
                # else:
                    # print(f"The correct counter example is: {ce}, with probability: {prob:.5f}")

          
            # method new
            t1 = np.sqrt(np.log(1/(alpha))/(2*n1))
            t2 = np.sqrt(np.log(1/(alpha))/(2*n2))
 

            for a in range(A):
                p1_hat = bws.state_action_probs[state][ce[0]][a]
                p2_hat = bws.state_action_probs[state][ce[1]][a]

                if p1_hat - p2_hat - (t1+t2) > 0:
                    filtered_ce_new.append((ce, p1_hat))
                    n_total_new += 1

                    if u_from_obs(ce[0],rm) == u_from_obs(ce[1],rm):
                        wrong_ce_counts_new += 1
                        # print("Im here")
                        # print(f"The wrong counter example is: {ce}")
                        # Store wrong examples in memory
                        
                    break

        # print(f"The total number of pairs for state {state} is: {len(ce_set)}. We got {len(filtered_ce_previous)} negative examples, {wrong_ce_counts_previous} of which are wrong.")                
        # Write all wrong examples to file at once
        if wrong_examples:
            # print(f"The wrong examples are: {len(wrong_examples)}")
            with open("wrong_examples_debug.txt", "a") as fooo:
                for example in wrong_examples:
                    fooo.write(f"State: {example['state']}\n")
                    fooo.write(f"Counter Example: {example['counter_example']}\n")
                    
                    # Find non-zero indices for policy 1
                    non_zero_indices_p1 = np.where(example['policy1'] > 0)[0]
                    fooo.write(f"Policy 1 ({example['counter_example'][0]}) non-zero actions:\n")
                    for idx in non_zero_indices_p1:
                        fooo.write(f"  Action {idx}: {np.round(example['policy1'][idx], 3)} (Policy 2: {np.round(example['policy2'][idx], 3)})\n")
                    
                    # Find non-zero indices for policy 2
                    non_zero_indices_p2 = np.where(example['policy2'] > 0)[0]
                    fooo.write(f"Policy 2 ({example['counter_example'][1]}) non-zero actions:\n")
                    for idx in non_zero_indices_p2:
                        fooo.write(f"  Action {idx}: {np.round(example['policy2'][idx], 3)} (Policy 1: {np.round(example['policy1'][idx], 3)})\n")
                    
                    fooo.write(f"n1: {example['n1']}, n2: {example['n2']}\n")
                    fooo.write("-" * 50 + "\n")

                
        if filtered_ce_new:
            filtered_counter_examples[state] = filtered_ce_new

    # Add C4 constraints for filtered counter examples
    print(f"Previous method - Total examples: {n_total_previous}, Wrong examples: {wrong_ce_counts_previous}, Ratio: {wrong_ce_counts_previous/n_total_previous:.2f}")
    print(f"New method - Total examples: {n_total_new}, Wrong examples: {wrong_ce_counts_new}, Ratio: {wrong_ce_counts_new/n_total_new:.2f}")
   
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