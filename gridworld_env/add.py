from collections import deque
import os
import sys
import time
import numpy as np

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Append the parent directory to sys.path
sys.path.append(parent_dir)
from reward_machine.reward_machine import RewardMachine

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
    


def from_paths_to_prefixes(path):
    prefix = ''
    for p in path:
        prefix += p+','
    return prefix


from reward_machine.reward_machine import RewardMachine
import config
from dynamics.GridWorld import BasicGridWorld
from utils.mdp import MDP

if __name__ == "__main__":

    grid_size = config.GRID_SIZE
    wind = config.WIND
    discount = config.GAMMA
    horizon = config.HORIZON   
 
    gw = BasicGridWorld(grid_size,wind,discount,horizon)
    
    n_states = gw.n_states
    n_actions = gw.n_actions

    P = []

    for a in range(n_actions):
        P.append(gw.transition_probability[:,a,:])

    mdp = MDP(n_states=n_states, n_actions=n_actions, P = P,gamma = gw.discount,horizon= config.HORIZON)

    rm = RewardMachine(config.RM_PATH_MAXSAT_3)
    target_node = 1
    max_depth = 6
    
    import time
    start_time = time.time()
    prefixes = get_prefixes_to_node(target_node, rm, max_depth)
    end_time = time.time()
    print(f"Time taken to compute prefixes: {end_time - start_time} seconds")

    print(f"Prefixes that reach node {target_node}: {len(prefixes)}")
    

    L = {} # Labeling of the states
    # The grid numbering and labeling is :
    # 0 4 8 12    D D C C 
    # 1 5 9 13    D D C C 
    # 2 6 10 14   A A B B
    # 3 7 11 15   A A B B
    
    L[2], L[6], L[3], L[7] = 'A', 'A', 'A', 'A'
    L[0], L[4], L[8], L[12] = 'D', 'D', 'C', 'C'
    L[1], L[5], L[9], L[13] = 'D', 'D', 'C', 'C'
    L[10], L[14] = 'B', 'B'
    L[11], L[15] = 'B', 'B'
    
    invL = {'A':[2,6,3,7], 'B':[10,11,14,15], 'C':[8,9,12,13], 'D':[0,1,4,5]}
    
    # Convert set to list to access first element
    
   
    prefixes_list = list(prefixes)
   
    import time
    start_time = time.time()
    for prefix in prefixes_list:
        
        states_to_reach = get_prefixes_to_states(mdp, prefix, invL, L)
    end_time = time.time()
    print(f"Time taken to compute states for prefixs: {end_time - start_time} seconds")



    path = prefixes_list[0]
    prefix = from_paths_to_prefixes(path)
    print(f"path: {path}")
    print(f"Prefix: {prefix}")
    print(f"States to reach: {states_to_reach}")
    time.sleep(4)

    