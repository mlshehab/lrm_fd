import numpy as np
from scipy.stats import entropy
import time
from utils.sat_utils import *
from utils.ne_utils import get_label, u_from_obs
import argparse
import pandas as pd
from z3 import Bool, Solver, Implies, Not, BoolRef, sat,print_matrix, Or, And, AtMost, Optimize # type: ignore
from itertools import combinations
from tqdm import tqdm
import config

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
            

 
    solutions = []
    start = time.time()
    nsol = 0
    while s.check() == sat:
        nsol += 1
        # print(f"The number of solutions is: {nsol}")
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
 


def prepare_sat_problem(gws, counter_examples, alpha):
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
                gws.state_action_probs[state][ce[0]],
                gws.state_action_probs[state][ce[1]],
                metric="L1")
            n1 = gws.state_label_counts[state][ce[0]]
            n2 = gws.state_label_counts[state][ce[1]]
            A = gws.n_actions
            prob = f(eps/2, n1, n2, A, eps)
            if prob > 1 - alpha:
                
                kept.append(ce)
        if kept:
            filtered[state] = kept
    
    c4_clauses = []
    
    for state, ces in filtered.items():
        for ce in ces:
            c4_clauses.append(ce)
            
    
    return  c4_clauses  



def maxsat_clauses(all_clauses, kappa, AP, proposition2index):
    """
    Given a list of C4 clauses and the reward machine `rm`,
    build a MaxSAT problem that tries to include the largest possible
    subset of those clauses. Returns the list of clauses Z3 chose.
    """
    # helper to convert prefixes to indices
 
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


def solve_with_clauses(c4_clauses, kappa, AP, proposition2index, print_solutions = False):
    """
    Given a list of C4 clauses, add all of them as constraints and
    enumerate all satisfying assignments. Returns the count of solutions.
    """
    # helper to convert prefixes to indices
    
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


from collections import deque

def get_prefixes_to_node(target_u, rm, max_depth=5):
    """
    Returns:
    - A set of tuples. Each tuple is a sequence of propositions that leads to target_u.
    """
    visited = set()
    prefixes = set()
    queue = deque()

    # Each item in the queue is: (current_u, path_so_far)
    queue.append((rm.u0, []))

    propositions = ['A', 'B', 'C', 'D']

    while queue:
        current_u, path = queue.popleft()

        if len(path) > max_depth:
            continue
        
        if current_u == target_u:
            prefixes.add(tuple(path))
            # Don't stop here — other shorter prefixes might exist

        if current_u not in rm.delta_u:
            continue

        for prop in propositions:
            prop_str = prop  # You can make this a comma string if multiple props are allowed
            u_prime = rm.get_next_state(current_u, prop_str)
          
            new_path = path + [prop_str]
            state_trace = (u_prime, tuple(new_path))
            if state_trace not in visited:
                visited.add(state_trace)
                queue.append((u_prime, new_path))

    return prefixes

def get_future_states(s, mdp):
    P = mdp.P
    post_state = []
    for Pa in P:
        for index in np.argwhere(Pa[s]> 0.0):
            post_state.append(index[0])     
    return list(set(post_state))


def get_prefixes_to_states(mdp, prefix, invL, L):
    if prefix:
        idx_ = 0
        
        p = prefix[idx_]
        
        prefix_state_dict = {}
        
        queue = invL[p].copy()  # Make a copy to avoid modifying the original
        idx_ += 1
        
        while idx_ < len(prefix):
            p = prefix[idx_]
            new_queue = []  # Create a new queue for the next iteration
            
            for q in queue:
                future_states = get_future_states(q, mdp)
                for s in future_states:
                    # print(f"s: {s}, L[s]: {L[int(s)]}, prefix[idx_]: {prefix[idx_]}")
                    if L[int(s)] == prefix[idx_]:
                        new_queue.append(int(s))  # Convert to regular int
            
            queue = list(set(new_queue))  # Remove duplicates and convert to regular ints
            idx_ += 1
        
        
        return queue
    else:
        return []


def from_paths_to_prefixes(path):
    prefix = ''
    for p in path:
        prefix += p+','
    return prefix

def construct_learned_product_policy(mdp, rm , max_depth, true_product_policy, true_rm, invL, L):
    n_states = mdp.n_states
    n_actions = mdp.n_actions
    n_nodes = rm.n_states
    
    product_policy = np.zeros((n_states*n_nodes, n_actions))

    for u in range(n_nodes):
        prefixes = get_prefixes_to_node(u, rm, max_depth)
        for prefix in prefixes:
            states = get_prefixes_to_states(mdp, prefix, invL, L)
            u_true = u_from_obs(from_paths_to_prefixes(prefix), true_rm)
            for state in states:
                product_policy[state + u*n_states, :] += true_product_policy[state + u_true*n_states, :]
    
    # Normalize the product policy by dividing each row by its sum
    row_sums = np.sum(product_policy, axis=1, keepdims=True)
    # Avoid division by zero by setting zero sums to one (this will result in a uniform distribution)
    row_sums[row_sums == 0] = 1
    product_policy = product_policy / row_sums
    
    for i, row in enumerate(product_policy):
        if np.sum(row) == 0:
            product_policy[i, :] = np.full(n_actions, 1.0 / n_actions)
    
    return product_policy





def constrtuct_product_policy(gws,states, c4_clauses, chosen_mask, rm, true_product_policy):
    
    # first the shape of the policy
    n_states = gws.n_states
    n_actions = gws.n_actions

    n_nodes = rm.n_states   
    
    product_policy = np.zeros((n_states*n_nodes, n_actions))
 
    for state , ce , idx in  zip(states, c4_clauses, chosen_mask):
        # fist find the corresponding rm node
        if idx == 1:
   
            u_0 = u_from_obs(ce[0], rm)
        
            u_1 = u_from_obs(ce[1], rm)

            product_state_0 = u_0 * n_states + state
            product_state_1 = u_1 * n_states + state


            product_policy[product_state_0, :] += gws.state_action_probs[state][ce[0]]
            product_policy[product_state_1, :] += gws.state_action_probs[state][ce[1]]

        else: # maxsat clause not chosen
            continue
    try:        
        product_policy = product_policy / np.sum(product_policy, axis=1, keepdims=True)
    except:
        pass

    
    # Find the rows in product_policy that are all NaNs

    for i, row in enumerate(product_policy):
        if np.all(np.isnan(row)):
            # extract the state and rm node
            product_policy[i, :] = true_product_policy[i, :]
            # state = i % n_states
            # rm_node = i // n_states

            # if rm_node == 0:
            #     product_policy[i, :] = true_product_policy[state, :]
            # else:
            #     product_policy[i, :] = 0


    # print(f"The product policy is: {product_policy}")
     
    return product_policy



def perfrom_policy_rollout(bws, len_traj, rm_learned, rm_true, policy, seed = None):

    if seed is not None:
        np.random.seed(seed)
        
    reward = 0.0
    state = np.random.randint(0, bws.n_states)
    label = bws.L[state] + ','
    compressed_label = bws.remove_consecutive_duplicates(label)

    # start of the synchronization
    u_learned = u_from_obs(label,rm_learned)
    u_true = u_from_obs(label,rm_true)
    
    true_product_state = u_true * bws.n_states + state

    # start of the rollout
    for _ in range(len_traj):
        idx = u_learned * bws.n_states + state
        action_dist = policy[idx,:]

        # Sample an action from the action distribution
        a = np.random.choice(np.arange(bws.n_actions), p=action_dist)
        
        # Sample a next state 
        next_state = bws.sample_next_state(state, a)

        # Compress the label
        compressed_label = bws.remove_consecutive_duplicates(label)
        # print(f"The compressed label is: {compressed_label}")
        l = bws.L[next_state]

        if u_true == 3 and l == 'D':
            reward += config.REWARD_PARAMETER

        label = label + l + ','
        u_learned = u_from_obs(label, rm_learned)
        u_true = u_from_obs(label, rm_true)
        

       

        state = next_state

    # print(f"The label is: {label}")
    return reward



def perfrom_policy_rollout_IRL(bws, len_traj, rm_true, policy, seed = None):

    if seed is not None:
        np.random.seed(seed)
        
    reward = 0.0
    state = np.random.randint(0, bws.n_states) # TODO: change this to the starting state
    label = bws.L[state] + ','
    compressed_label = bws.remove_consecutive_duplicates(label)

    # start of the synchronization
    
    u_true = u_from_obs(label,rm_true)
    
   
    # start of the rollout
    for _ in range(len_traj):
     
        action_dist = policy[state,:]

        # Sample an action from the action distribution
        a = np.random.choice(np.arange(bws.n_actions), p=action_dist)
        
        # Sample a next state 
        next_state = bws.sample_next_state(state, a)

        # Compress the label
        compressed_label = bws.remove_consecutive_duplicates(label)
        # print(f"The compressed label is: {compressed_label}")
        l = bws.L[next_state]

        if u_true == 3 and l == 'D':
            reward += config.REWARD_PARAMETER

        label = label + l + ','
       
        u_true = u_from_obs(label, rm_true)
    

        state = next_state

    # print(f"The label is: {label}")
    return reward
