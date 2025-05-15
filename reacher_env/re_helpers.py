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
from z3 import Bool, Solver, Implies, Not, BoolRef, sat,print_matrix, Or, And, AtMost # type: ignore
from itertools import combinations
from tqdm import tqdm
import pickle
from simulator import ReacherDiscretizerUniform, ReacherDiscreteSimulator, ForceRandomizedReacher
import matplotlib.pyplot as plt

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

    # B_T = transpose_boolean_matrix(B_)
    # powers_B_T = [boolean_matrix_power(B_T,k) for k in range(1,kappa)]
    # powers_B_T_x = [boolean_matrix_vector_multiplication(B,x) for B in powers_B_T]
    # powers_B_T_x.insert(0, x)
    # OR_powers_B_T_x = element_wise_or_boolean_vectors(powers_B_T_x)
    
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
                else:
                    pass
                    
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

                
        if filtered_ce_previous:
            filtered_counter_examples[state] = filtered_ce_previous

    # Add C4 constraints for filtered counter examples
    print(f"Previous method - Total examples: {n_total_previous}, Wrong examples: {wrong_ce_counts_previous}, Ratio: {wrong_ce_counts_previous/n_total_previous:.4f}")
    print(f"New method - Total examples: {n_total_new}, Wrong examples: {wrong_ce_counts_new}, Ratio: {wrong_ce_counts_new/n_total_new:.4f}")
   
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





def prepare_sat_problem(rds, counter_examples, alpha):
    """
    Do all the one‐time work:
      1) build B, x, precompute powers, etc.
      2) create a 'base' z3.Solver() with C1 & C2 constraints
      3) compute filtered_counter_examples
      4) build the list of all C4 clauses (Not(elt)) but don’t add them yet
    Returns: (base_solver, c4_clauses, kappa, AP)
    """
    # 3) filter counter‐examples just once
    filtered = {}
 
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
                
                kept.append(ce)
        if kept:
            filtered[state] = kept
    
    c4_clauses = []
    for state, ces in filtered.items():
        for ce in ces:
            c4_clauses.append(ce)
            # p1 = prefix2indices(ce[0])
            # p2 = prefix2indices(ce[1])
            # sub1 = bool_matrix_mult_from_indices(B, p1, x)
            # sub2 = bool_matrix_mult_from_indices(B, p2, x)
            # for elt in element_wise_and_boolean_vectors(sub1, sub2):
            #     c4_clauses.append(Not(elt))
    # print(f"The number of C4 clauses is: {len(c4_clauses)}")
    return  c4_clauses   


def solve_for_beta( c4_clauses,  beta, rm):
    """
    Given the prepared base solver and the list of C4 clauses,
    sample u ~ Uniform[0,1], add C4 only if u>beta, then enumerate solutions.
    Returns the number of solutions found.
    """
    # helper to convert prefixes to indices
    proposition2index = {'B':0,'R':1,'Y':2,'I':3}
    def prefix2indices(lbl):
        return [proposition2index[p] for p in lbl.split(',') if p]

    # 1) SAT variables
    B = [[[Bool(f"x_{i}_{j}_{k}") for j in range(kappa)]
           for i in range(kappa)]
           for k in range(AP)]

    x = [False]*kappa
    x[0] = True

    # 2) base solver with C1 & C2
    s = Solver()
    # C1: x[i][j][k] ⇒ x[j][j][k]
    for ap in range(AP):
        for i in range(kappa):
            for j in range(kappa):
                s.add(Implies(B[ap][i][j], B[ap][j][j]))
    # C2: one entry per row
    for ap in range(AP):
        s.add(one_entry_per_row(B[ap]))

    n_ce_added = 0
    n_false = 0
    n_total_false = 0
    for ce in c4_clauses:
        if u_from_obs(ce[0],rm) == u_from_obs(ce[1],rm):
            n_total_false += 1
        u = np.random.rand()
        if u < beta:
            n_ce_added += 1
            # if u_from_obs(ce[0],rm) == u_from_obs(ce[1],rm):
            #     # print(f"A counter example wrongly included with beta={beta}.")
            #     n_false += 1
            p1 = prefix2indices(ce[0])
            p2 = prefix2indices(ce[1])
            sub1 = bool_matrix_mult_from_indices(B, p1, x)
            sub2 = bool_matrix_mult_from_indices(B, p2, x)
            res_ = element_wise_and_boolean_vectors(sub1, sub2)
            for elt in res_:
                s.add(Not(elt))

    # print(f"The number of C4 clauses added is: {n_ce_added}")
    # enumerate models
    count = 0
    
    while s.check() == sat:
        m = s.model()
        # block current model
        block_clause = []
        for ap in range(AP):
            for i in range(kappa):
                for j in range(kappa):
                    block_clause.append(B[ap][i][j] != m.evaluate(B[ap][i][j], model_completion=True))
        s.add(Or(block_clause))
        count += 1

    return count

from multiprocessing import Pool, cpu_count

def _worker_solve(args):
    """
    Unpack arguments and call solve_for_beta.
    We assume solve_for_beta signature is:
       solve_for_beta(c4_clauses, beta, rm)
    """
    c4_clauses, beta, rm = args
    return solve_for_beta(c4_clauses, beta, rm)

if __name__ == "__main__":
    # ────────────────────────────────────────────────────────────────
    # Example usage (e.g. in your main()):
    #
    # 1) do the one‐time prep

    rm = RewardMachine("../rm_examples/reacher.txt")


  
    with open("./objects/object_no_parallel_5_9.pkl", "rb") as foo:
        rds = pickle.load(foo)

    print(f"{rds.rd.n_actions}")


    # print(f"DEBUG: {np.round(rds.state_action_probs[16371]['I,B,'],3)}")
    # print(f"DEBUG: {np.round(rds.state_action_probs[16371]['I,B,I,R,I,Y,I,B,'],3)}")
    # time.sleep(1000)
    counter_examples = generate_label_combinations(rds)

    
    kappa = 3
    AP = 4
    alphas = [  0.00001, 0.0001,0.001,0.01,0.05]
    betas = [  0.02,0.05, 0.08, 0.1, 0.12, 0.15,0.2]
    TOTAL = 50
    
    results = {}

    # for alpha in alphas:
    #     print(f"\nPreparing for alpha={alpha}")
    #     # one‐time cost PER alpha
    #     c4_clauses = prepare_sat_problem(rds, counter_examples, alpha)

    #     for beta in betas:
    #         print(f"\n  Processing beta={beta}")

    #         # build the argument list for the pool
    #         # we call solve_for_beta(c4_clauses, beta, rm) TOTAL times
    #         args_list = [(c4_clauses, beta, rm) for _ in range(TOTAL)]

    #         # spawn a pool and map over args_list
    #         with Pool(processes=cpu_count()) as pool:
    #             num_sols_list = pool.map(_worker_solve, args_list)

    #         # now compute stats
    #         min_sols = min(num_sols_list)
    #         max_sols = max(num_sols_list)
    #         avg_sols = sum(num_sols_list) / len(num_sols_list)
    #         std_sols = (sum((x - avg_sols) ** 2 for x in num_sols_list) / len(num_sols_list)) ** 0.5

    #         results[(alpha, beta)] = {
    #             'min': min_sols,
    #             'max': max_sols,
    #             'avg': avg_sols,
    #             'std': std_sols,
    #             'values': num_sols_list
    #         }
    #         print(f"Results for alpha={alpha}, beta={beta}:")
    #         print(f"  Min solutions: {min_sols}")
    #         print(f"  Max solutions: {max_sols}")
    #         print(f"  Average solutions: {avg_sols:.2f}")
    #         print(f"  Standard deviation: {std_sols:.2f}")
    
    for alpha in tqdm(alphas):
        print(f"\nProcessing alpha={alpha}")
        c4_clauses = prepare_sat_problem(rds, counter_examples, alpha)
        
        for beta in betas:
            print(f"\nProcessing beta={beta}")
            num_sols_list = []
            
            for t in range(TOTAL):
                num_sols = solve_for_beta(c4_clauses, beta, rm)
                num_sols_list.append(num_sols)
                # print(f"Iteration {t}: {num_sols} solutions")
            
            # Calculate statistics
            min_sols = min(num_sols_list)
            max_sols = max(num_sols_list)
            avg_sols = sum(num_sols_list) / len(num_sols_list)
            std_sols = (sum((x - avg_sols) ** 2 for x in num_sols_list) / len(num_sols_list)) ** 0.5
            
            results[(alpha, beta)] = {
                'min': min_sols,
                'max': max_sols,
                'avg': avg_sols,
                'std': std_sols,
                'values': num_sols_list  # Store all values for histogram plotting
            }
        
            print(f"Results for alpha={alpha}, beta={beta}:")
            print(f"  Min solutions: {min_sols}")
            print(f"  Max solutions: {max_sols}")
            print(f"  Average solutions: {avg_sols:.2f}")
            print(f"  Standard deviation: {std_sols:.2f}")



    for alpha, beta in results:
         
        
        # Plot histogram
        plt.figure(figsize=(10, 6))
        plt.hist(results[(alpha, beta)]['values'], bins=40, alpha=0.7)
        plt.axvline(results[(alpha, beta)]['avg'], color='r', linestyle='dashed', linewidth=2, label=f'Average: {results[(alpha, beta)]["avg"]:.2f}')
        
        # Count occurrences of 4 solutions
        count_4 = results[(alpha, beta)]['values'].count(4)
        if count_4 > 0:
            print(f"4 solutions occurred {count_4} times")
            # Add arrow pointing to the 4-solutions bin
            plt.annotate(f'4 solutions\noccurred {count_4} times', 
                        xy=(4, count_4), 
                        xytext=(4, count_4 + 2),
                        arrowprops=dict(facecolor='black', shrink=0.05),
                        ha='center')
        
        plt.title(f'Distribution of Solutions (α={alpha}, β={beta})')
        plt.xlabel('Number of Solutions')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        # plt.show()

        # Save histogram to file
        plt.savefig(f'results/histogram_alpha_{alpha}_beta_{beta}.png')
        plt.close()
 


    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Save results dictionary to a pickle file
    with open('results/results.pkl', 'wb') as f:
        pickle.dump(results, f)



    