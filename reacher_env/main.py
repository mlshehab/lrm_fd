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
from simulator import ReacherDiscreteSimulator, ForceRandomizedReacher, ReacherDiscretizerUniform




if __name__ == '__main__':

  
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
    alpha = 0.0001
    solutions, total_constraints,  filtered_counter_examples , solve_time = solve_sat_instance(rds, counter_examples, rm, kappa, AP, alpha)
    print(f"The number of constraints is: {total_constraints}, { filtered_counter_examples }")
    print(f"The number of solutions is: {len(solutions)}")

    solutions_text_path = f"./objects/solutionsss.txt"
    with open(solutions_text_path, "w") as f:
        f.write(f"Solutions for n_traj={1000000}, depth={160}\n")
        f.write("=" * 50 + "\n\n")
        for i, solution in enumerate(solutions):
            f.write(f"Solution {i+1}:\n")
            for j, matrix in enumerate(solution):
                f.write(f"\nMatrix {j} ({['B', 'R', 'Y', 'I'][j]}):\n")
                for row in matrix:
                    f.write("  " + " ".join("1" if x else "0" for x in row) + "\n")
            f.write("\n" + "-" * 30 + "\n\n")

    print(f"[Main] Saved solutions in readable format to {solutions_text_path}")


