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
    parser.add_argument('-depth', type=int, required=True,
                        help='Depth must be provided as an integer.')
    parser.add_argument('-n_traj', type=int, required=True,
                        help='Number of trajectories must be provided as an integer.')
    parser.add_argument('-save', type=int, choices=[0, 1], required=True,
                       help='0 for False, 1 for True. This argument is required.')
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
 

    label_combinations = {}

    for state, label_dists in bws.state_action_probs.items():
        labels = [label for label in label_dists.keys()]
        label_combinations[state] = list(combinations(labels, 2))

    return label_combinations

def prefix2indices(s, proposition2index):
    out = []
    for l in s.split(','):
        if l:
            out.append(proposition2index[l])
    return out

def solve_sat_instance(bws, counter_examples, rm, metric, kappa, AP, proposition2index,  p_threshold=0.8):
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


    # Filter counter examples by probability threshold
    filtered_counter_examples = {}
    prob_values = []  # Store probability values for visualization

    wrong_ce_counts = 0
    # tot_ = 0
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
                # Save wrong counter example to file
                with open(f"./objects/wrong_counter_examples.txt", "a") as foo:
                    foo.write(f"State: {state}\n")
                    foo.write(f"Counter Example: {ce}\n")
                    foo.write(f"Probability: {prob}\n")
                    foo.write(f"Counts: {bws.state_label_counts[state][ce[0]]}, {bws.state_label_counts[state][ce[1]]}\n")
                    foo.write(f"Policy 1: {np.round(bws.state_action_probs[state][ce[0]], 3)}\n")
                    foo.write(f"Policy 2: {np.round(bws.state_action_probs[state][ce[1]], 3)}\n")
                    foo.write("-" * 50 + "\n")



            p1 = prefix2indices(ce[0], proposition2index)
            p2 = prefix2indices(ce[1], proposition2index)

            sub_B1 = bool_matrix_mult_from_indices(B,p1, x)
            sub_B2 = bool_matrix_mult_from_indices(B,p2, x)
            res_ = element_wise_and_boolean_vectors(sub_B1, sub_B2)

            for elt in res_:
                s.add(Not(elt))
            
    print(f"The total number of constraints is: {total_constraints}")
 
    solutions = []
    start = time.time()
    nsol = 0
    while s.check() == sat:
        nsol += 1
        print(f"The number of solutions is: {nsol}")
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




def infinite_horizon_soft_policy_evaluation(MDP, reward, pi, tol=1e-4, logging=True, log_iter=5):
    """
    Evaluate a fixed stochastic policy pi in a max-entropy RL setting.
    
    Arguments:
        MDP: object with attributes
            - gamma: discount factor
            - P: list of transition matrices P[a] of shape (n_states, n_states)
            - n_states: number of states
            - n_actions: number of actions
        reward: array of shape (n_states, n_actions, n_states), i.e., R[s, a, s']
        pi: array of shape (n_states, n_actions), policy probabilities π(a|s)
        tol: convergence threshold
        logging: whether to print progress
        log_iter: log every k iterations

    Returns:
        q_soft: shape (n_states, n_actions)
        v_soft: shape (n_states,)
    """
    print("Evaluating fixed stochastic policy with soft Bellman updates ...")
    
    gamma = MDP.gamma
    n_states = MDP.n_states
    n_actions = MDP.n_actions

    v_soft = np.zeros(n_states)
    q_soft = np.zeros((n_states, n_actions))

    delta = np.inf
    it = 0
    total_time = 0.0

    while delta > tol:
        it += 1
        start_time = time.time()

        # 1. Compute Q^\pi(s,a)
        for a in range(n_actions):
            P_a = MDP.P[a]                 # (n_states, n_states)
            r_a = reward[:, a, :]          # (n_states, n_states)
            expected_r = np.sum(P_a * r_a, axis=1)        # (n_states,)
            expected_v = gamma * P_a @ v_soft             # (n_states,)
            q_soft[:, a] = expected_r + expected_v        # (n_states,)

        # 2. Compute V^\pi(s) = E_{a ∼ π}[Q(s,a) - log π(a|s)]
        log_pi = np.log(pi + 1e-10)  # avoid log(0)
        v_new_soft = np.sum(pi * (q_soft - log_pi), axis=1)  # (n_states,)

        delta = np.linalg.norm(v_new_soft - v_soft)

        end_time = time.time()
        total_time += end_time - start_time

        if logging and it % log_iter == 0:
            print(f"Iter {it}: Δ={delta:.6f}, Time: {end_time - start_time:.2f}s, Total: {total_time:.2f}s")

        v_soft = v_new_soft

    return q_soft, v_soft
 