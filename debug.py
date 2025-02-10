import os
import sys

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Append the parent directory to sys.path
sys.path.append(parent_dir)

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



rm = RewardMachine("./rm_examples/dynamic_stacking.txt")

label1 = 'C,I,' # --> 2
label2 = 'C,I,C,' #--> 0
label3 = 'C,I,C,'
label4 = 'I,A,I,B,I,' #--> 1
label5 = 'I,A,I,' # --> 0 
print(f"The node with label = {label1} is u = {u_from_obs(label1,rm)}")
print(f"The node with label = {label2} is u = {u_from_obs(label2,rm)}")
print(f"The node with label = {label3} is u = {u_from_obs(label3,rm)}")
print(f"The node with label = {label4} is u = {u_from_obs(label4,rm)}")
print(f"The node with label = {label5} is u = {u_from_obs(label5,rm)}")