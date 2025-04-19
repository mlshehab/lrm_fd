import os
import sys
from scipy.optimize import minimize_scalar
# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Append the parent directory to sys.path
sys.path.append(parent_dir)
import pandas as pd
import numpy as np
from utils.mdp import MDP, MDPRM
from reward_machine.reward_machine import RewardMachine
import scipy.linalg
import time 
from scipy.special import softmax, logsumexp
from tqdm import tqdm
import pprint
import xml.etree.ElementTree as ET
from collections import deque
from dynamics.BlockWorldMDP import BlocksWorldMDP, infinite_horizon_soft_bellman_iteration
from utils.ne_utils import get_label, u_from_obs,save_tree_to_text_file, collect_state_traces_iteratively, get_unique_traces,group_traces_by_policy
from utils.sat_utils import *
from datetime import timedelta
import time
from collections import Counter, defaultdict
from scipy.stats import entropy  # For KL divergence
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle 
import argparse

def f(epsilon_1, n1, n2, A, epsilon):
    term1 = np.maximum(1 - ((2**A - 2) * np.exp((-n1 * epsilon_1**2) / (2 ))), 0)
    term2 = np.maximum(1 - ((2**A - 2) * np.exp((-n2 * (epsilon - epsilon_1)**2) / (2 ))), 0)
    return term1 * term2


from simulators import Simulator, BlockworldSimulator
from  re_helpers import similarity, solve_sat_instance, parse_args, generate_policy_comparison_report
from  re_helpers import parse_args, generate_label_combinations
from simulator import ReacherDiscreteSimulator, ForceRandomizedReacher, ReacherDiscretizer




if __name__ == '__main__':

    # args = parse_args()
    # # print(f"The arguments are: {args}")
    # bw = BlocksWorldMDP(num_piles=3)
    # transition_matrices,s2i, i2s = bw.extract_transition_matrices_v2()
    # n_states = bw.num_states
    # n_actions = bw.num_actions

    # P = []

    # for a in range(n_actions):
    #     # print(f"The matrix shape is: {transition_matrices[a,:,:]}")
    #     P.append(transition_matrices[a,:,:])

    # mdp = MDP(n_states=n_states, n_actions=n_actions,P = P,gamma = 0.9,horizon=10)

    rm = RewardMachine("../rm_examples/reacher.txt")

    # L = {}

    # print(f"The number of states is: {len(s2i.keys())}")

    # target_state_1 = ((0,1,2),(),())
    # target_state_2 = ((),(2,1,0),())
    # target_state_3 = ((),(),(2,1,0))
    # bad_state = ((0,),(1,),(2,))

    # for state_index in range(bw.num_states):
    #     if state_index == s2i[target_state_1]:
    #         L[state_index] = 'A'
    #     elif state_index == s2i[target_state_2]:
    #         L[state_index] = 'B'
    #     elif state_index == s2i[target_state_3]:
    #         L[state_index] = 'C'
    #     else:
    #         L[state_index] = 'I'

    
    # soft_policy = np.load("soft_policy.npy")

    # bws = BlockworldSimulator(rm = rm,mdp = mdp,L = L,policy = soft_policy,state2index=s2i,index2state=i2s)
    
    
    # starting_states = [s2i[target_state_1], s2i[target_state_2], s2i[target_state_3], 4, 24]

    # start = time.time()
    # max_len = args.depth
    # n_traj = args.n_traj
    
    # bws.sample_dataset(starting_states=starting_states, number_of_trajectories= n_traj, max_trajectory_length=max_len)
    # end = time.time()

    # elapsed_time = end - start
    # hours, rem = divmod(elapsed_time, 3600)
    # minutes, seconds = divmod(rem, 60)
    # print(f"Simulating the dataset took {int(hours)} hour {int(minutes)} minute {seconds:.2f} sec.")

    # bws.compute_action_distributions()

    # Save the object to a file
    # if args.save:
    #     timestamp = time.strftime("%Y%m%d-%H%M%S")
    #     with open(f"./objects/object{n_traj}_{max_len}_{timestamp}.pkl", "wb") as foo:
    #         pickle.dump(bws, foo)
    #     print(f"The object has been saved to ./objects/object{n_traj}_{max_len}_{timestamp}.pkl")

    #     generate_policy_comparison_report(bws, rm, soft_policy, n_traj, max_len, timestamp)

    with open("./objects/object100000_250.pkl", "rb") as foo:
        rds = pickle.load(foo)
    print(f"{rds.rd.n_actions}")
    counter_examples = generate_label_combinations(rds)

    # for state, label_dists in bws.state_action_probs.items():
    #     print(f"State: {state}")
    #     for label, action_prob in label_dists.items():
    #         print(f"  Label: {label}")
    #         print(f"  Action Probabilities: {np.round(action_prob, 3)}")
    #     if state in counter_examples:
    #         print(f"  Combinations: {counter_examples[state]}")
    #     else:
    #         print("  Combinations: None")

    p_threshold = 0.95
    
    kappa = 3
    AP = 4
    alpha = 0.005
    solutions, total_constraints,  filtered_counter_examples , solve_time = solve_sat_instance(rds, counter_examples, rm, kappa, AP, alpha)
    print(f"The number of constraints is: {total_constraints}, { filtered_counter_examples }")
    print(f"The number of solutions is: {len(solutions)}")
    # for solution in solutions:
    #     print("\nSolution matrices:")
    #     for i, matrix in enumerate(solution):
    #         print(f"\nMatrix {i} ({['A', 'B', 'C', 'I'][i]}):")
    #         for row in matrix:
    #             print("  " + " ".join("1" if x else "0" for x in row))
    # print(f"The wrong counter example counts are: {wrong_ce_counts}")
    # hours, rem = divmod(solve_time, 3600)
    # minutes, seconds = divmod(rem, 60)
    # print(f"The solve time is: {int(hours)} hour {int(minutes)} minute {seconds:.2f} sec.")
    # print(f"The count of state 51 is: {bws.state_label_counts[51]}")

    # # Run SAT solver for each metric and threshold
    # results = {metric: [] for metric in metrics}
    # for metric in tqdm(metrics):
    #     for epsilon in tqdm(epsilon_vals):
    #         state_traces_dict = {}
    #         for state, label_dists in bws.state_action_probs.items():
    #             if len(label_dists) > 1:
    #                 gpd = bws.group_similar_policies(state, metric=metric, threshold=epsilon)
    #                 grouped_lists = list(gpd.values())
    #                 state_traces_dict[state] = grouped_lists
            
    #         counter_examples = generate_combinations(state_traces_dict)
    #         solutions, n_constraints, n_states, solve_time, prob_values, wrong_ce_counts = solve_sat_instance(bws, counter_examples, epsilon,rm,p_threshold=0.95)
    #         results[metric].append({
    #             'epsilon': epsilon,
    #             'solutions': solutions,
    #             'n_solutions': len(solutions),
    #             'n_constraints': n_constraints, 
    #             'n_states': n_states,
    #             'solve_time': solve_time,
    #             'prob_values': prob_values,
    #             'wrong_ce_counts': wrong_ce_counts
    #         })


    # # for solution in results["L1"][0]["solutions"]:
    # #     print("\nSolution matrices:")
    # #     for i, matrix in enumerate(solution):
    # #         print(f"\nMatrix {i} ({['A', 'B', 'C', 'I'][i]}):")
    # #         for row in matrix:
    # #             print("  " + " ".join("1" if x else "0" for x in row))



    # # # Save the results to a file
    # # with open("./objects/results.pkl", "wb") as f:
    # #     pickle.dump(results, f)



    # # Visualization
    # plt.figure(figsize=(15, 12))

    # # Plot 1: Number of solutions vs epsilon for each metric
    # plt.subplot(3, 2, 1)
    # for metric in metrics:
    #     plt.plot(epsilon_vals, [r['n_solutions'] for r in results[metric]], marker='o', label=metric)
    # plt.xlabel('Epsilon')
    # plt.ylabel('Number of Solutions')
    # plt.title('Solutions vs Epsilon')
    # plt.legend()
    # plt.grid(True)

    # # Plot 2: Solve time vs epsilon
    # plt.subplot(3, 2, 2)
    # for metric in metrics:
    #     plt.plot(epsilon_vals, [r['solve_time'] for r in results[metric]], marker='o', label=metric)
    # plt.xlabel('Epsilon')
    # plt.ylabel('Solve Time (s)')
    # plt.title('Solve Time vs Epsilon')
    # plt.legend()
    # plt.grid(True)

    # # Plot 3: Number of constraints vs epsilon
    # plt.subplot(3, 2, 3)
    # for metric in metrics:
    #     plt.plot(epsilon_vals, [r['n_constraints'] for r in results[metric]], marker='o', label=metric)
    # plt.xlabel('Epsilon')
    # plt.ylabel('Number of Constraints')
    # plt.title('Constraints vs Epsilon')
    # plt.legend()
    # plt.grid(True)

    # # Plot 4: Wrong counter examples count vs epsilon
    # plt.subplot(3, 2, 4)
    # for metric in metrics:
    #     plt.plot(epsilon_vals, [r['wrong_ce_counts'] for r in results[metric]], marker='o', label=metric)
    # plt.xlabel('Epsilon')
    # plt.ylabel('Wrong Counter Examples')
    # plt.title('Wrong Counter Examples vs Epsilon')
    # plt.legend()
    # plt.grid(True)

    # # Plot 5: Probability distribution violin plot
    # plt.subplot(3, 2, (5, 6))
    # prob_data = []
    # labels = []
    # for metric in metrics:
    #     for r in results[metric]:
    #         if r['prob_values']:  # Only add if there are probability values
    #             prob_data.append(r['prob_values'])
    #             labels.append(f"{metric}\nε={r['epsilon']:.3f}")

    # plt.violinplot(prob_data)
    # plt.xticks(range(1, len(labels) + 1), labels, rotation=45)
    # plt.ylabel('Probability Values')
    # plt.title('Distribution of Probability Values')

    # plt.tight_layout()
    # plt.savefig('./figures/sat_results_analysis.png')
    # plt.close()



