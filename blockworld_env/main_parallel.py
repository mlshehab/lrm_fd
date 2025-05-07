import os
import sys
from scipy.optimize import minimize_scalar
# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import multiprocessing as mp
import pickle
import numpy as np
from simulator import BlockworldSimulator
from dynamics.BlockWorldMDP import BlocksWorldMDP
from reward_machine.reward_machine import RewardMachine
from utils.mdp import MDP
import config
import time
import argparse
import os
from bwe_helpers import generate_label_combinations, solve_sat_instance

def sample_worker(proc_id, rm_path, policy_path, mdp, L, s2i, i2s, starting_states, n_traj, max_len, output_path):
    rm = RewardMachine(rm_path)
    policy = np.load(policy_path)
    simulator = BlockworldSimulator(rm=rm, mdp=mdp, L=L, policy=policy, state2index=s2i, index2state=i2s)
    simulator.sample_dataset(starting_states, number_of_trajectories=n_traj, max_trajectory_length=max_len)

    with open(output_path, "wb") as f:
        pickle.dump(simulator, f)
    print(f"[Process {proc_id}] Done and saved to {output_path}")

def merge_simulators(base_sim, others):
    for other in others:
        for state, other_label_counts in other.state_action_counts.items():
            if state not in base_sim.state_action_counts:
                base_sim.state_action_counts[state] = list(other_label_counts)
            else:
                for other_label, other_counter in other_label_counts:
                    found = False
                    for i, (existing_label, existing_counter) in enumerate(base_sim.state_action_counts[state]):
                        if existing_label == other_label:
                            base_sim.state_action_counts[state][i] = (
                                existing_label,
                                existing_counter + other_counter
                            )
                            found = True
                            break
                    if not found:
                        base_sim.state_action_counts[state].append((other_label, other_counter))

def parallel_sample_dataset(num_procs, total_trajectories, max_len, starting_states):

    bw = BlocksWorldMDP(num_piles=config.NUM_PILES)
    transition_matrices, s2i, i2s = bw.extract_transition_matrices()
    n_states = bw.num_states
    n_actions = bw.num_actions
    P = [transition_matrices[a, :, :] for a in range(n_actions)]

    mdp = MDP(n_states=n_states, n_actions=n_actions, P=P, gamma=config.GAMMA, horizon=config.HORIZON)

    L = {
        s2i[config.TARGET_STATE_1]: 'A',
        s2i[config.TARGET_STATE_2]: 'B',
        s2i[config.TARGET_STATE_3]: 'C'
    }

    for s in range(n_states):
        if s not in L:
            L[s] = 'I'

    trajs_per_proc = total_trajectories // num_procs
    output_files = [f"./objects/tmp_bws_{i}.pkl" for i in range(num_procs)]
    procs = []

    for i in range(num_procs):
        p = mp.Process(
            target=sample_worker,
            args=(i, config.RM_PATH, config.POLICY_PATH, mdp, L, s2i, i2s, starting_states, trajs_per_proc, max_len, output_files[i])
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    # Load and merge all simulators
    with open(output_files[0], "rb") as f:
        base_sim = pickle.load(f)

    other_sims = []
    for fpath in output_files[1:]:
        with open(fpath, "rb") as f:
            other_sims.append(pickle.load(f))

    merge_simulators(base_sim, other_sims)
    print(f"[Main] Merged {num_procs} simulators.")

    # Optionally delete temp files
    for fpath in output_files:
        os.remove(fpath)

    return base_sim

if __name__ == '__main__':
    # print(f"Number of available CPU cores: {mp.cpu_count()}")

    parser = argparse.ArgumentParser()
    parser.add_argument('--depth', type=int, default=20)
    parser.add_argument('--n_traj', type=int, default=10_000_000)
    parser.add_argument('--n_procs', type=int, default=int(mp.cpu_count()/2))
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()

    bw = BlocksWorldMDP(num_piles=config.NUM_PILES)
    transition_matrices, s2i, i2s = bw.extract_transition_matrices()

    starting_states = [s2i[config.TARGET_STATE_1], s2i[config.TARGET_STATE_2], s2i[config.TARGET_STATE_3], 4, 24]
    
    # Ensure the './objects/' directory exists
    os.makedirs('./objects', exist_ok=True)

    # Start the parallel sampling process
    start_time = time.time()
    bws = parallel_sample_dataset(args.n_procs, args.n_traj, args.depth, starting_states)
    elapsed = time.time() - start_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"[Main] Dataset sampled in {int(hours)} hour {int(minutes)} minute {seconds:.2f} sec.")

    bws.compute_action_distributions()

    if args.save:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        out_path = f"./objects/bws_merged_{args.n_traj}_{args.depth}_{timestamp}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(bws, f)
        print(f"[Main] Saved merged simulator to {out_path}")


    counter_examples = generate_label_combinations(bws)

    rm = RewardMachine(config.RM_PATH)

    p_threshold = 0.95
    metric = "L1"
    kappa = 3
    AP = 4
    
    solutions, n_constraints, n_states, solve_time, prob_values, wrong_ce_counts = solve_sat_instance(bws, counter_examples,rm, metric, kappa, AP, p_threshold=0.95)
    print(f"The number of wrong counter examples is: {wrong_ce_counts}")
    print(f"The number of constraints is: {n_constraints}")
    print(f"The number of solutions is: {len(solutions)}")
    # Save solutions to a text file in a readable format


    timestamp = time.strftime("%Y%m%d-%H%M%S")
    solutions_text_path = f"./objects/solutions_{args.n_traj}_{args.depth}_{timestamp}.txt"
    with open(solutions_text_path, "w") as f:
        f.write(f"Solutions for n_traj={args.n_traj}, depth={args.depth}\n")
        f.write("=" * 50 + "\n\n")
        for i, solution in enumerate(solutions):
            f.write(f"Solution {i+1}:\n")
            for j, matrix in enumerate(solution):
                f.write(f"\nMatrix {j} ({['A', 'B', 'C', 'I'][j]}):\n")
                for row in matrix:
                    f.write("  " + " ".join("1" if x else "0" for x in row) + "\n")
            f.write("\n" + "-" * 30 + "\n\n")
    print(f"[Main] Saved solutions in readable format to {solutions_text_path}")


 


    
