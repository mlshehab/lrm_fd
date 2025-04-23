import gymnasium as gym
import numpy as np
from tqdm import tqdm
# Environment setup
from scipy import sparse

from simulator import ReacherDiscretizerB, ForceRandomizedReacher, ReacherDiscreteSimulator
  
from simulator import ReacherDiscretizerA as ReacherDiscretizer

# Run simulations
if __name__ == "__main__":

    max_len = 250
    env = gym.make("Reacher-v5",   max_episode_steps=max_len,xml_file="./reacher.xml")
    env = ForceRandomizedReacher(env)  # Wrap it

    target_blue = [0.1, -0.11]
    target_red = [0.1, 0.11]
    target_yellow = [-0.1, 0.11]
    target_random_1 = [0.1,0.0]
    target_random_2 = [0.14,0.05]
     


    target_dict = {"blue": target_blue, "red": target_red, "yellow": target_yellow}

    targets_goals = ["blue", "red", "yellow"]

    rd = ReacherDiscretizer(target_dict=target_dict)

    n_states = rd.n_states
    n_actions = rd.n_actions
    print(rd.n_states)
    print(rd.n_actions)

    # transition_counts = np.zeros((n_actions, n_states, n_states), dtype=np.int32)
    transition_counts = [sparse.lil_matrix((n_states, n_states), dtype=np.int32) for _ in range(n_actions)]
    # print(transition_counts.shape)
     
    n_steps = int(100)
    obs, _ = env.reset()
    continuous_state = rd.st4obs(obs)
    discrete_state_tuple =  rd.discretize_state(continuous_state)
    discrete_state_idx =  rd.state_to_idx[discrete_state_tuple]     


    for _ in tqdm(range(n_steps)):
        action = env.action_space.sample()
        discrete_action_tuple = rd.discretize_action(action)
        discrete_action_idx = rd.action_to_idx[discrete_action_tuple]
        
        obs, reward, terminated, truncated, info =  env.step(action)

        next_continuous_state = rd.st4obs(obs)
        next_discrete_state_tuple = rd.discretize_state(next_continuous_state)
        try:
            next_discrete_state_idx = rd.state_to_idx[next_discrete_state_tuple]
        except KeyError:
            # print(f"State {next_discrete_state_tuple} not found in state_to_idx")
            print(f"The state is: (θ1: {next_continuous_state[0]:.3f}, θ2: {next_continuous_state[1]:.3f}, θ1_dot: {next_continuous_state[2]:.3f}, θ2_dot: {next_continuous_state[3]:.3f})")
            obs, _ = env.reset()
            continue
 

        transition_counts[discrete_action_idx][discrete_state_idx, next_discrete_state_idx] += 1
        discrete_state_idx = next_discrete_state_idx 

        if terminated or truncated:
            obs, _ = env.reset()

    # Normalize to form transition probability matrices
    # transition_matrices = transition_counts / np.maximum(transition_counts.sum(axis=2, keepdims=True), 1)

    # Save the transition matrices to a file
    # Suppose transition_counts is a list of sparse matrices
    for action, matrix in enumerate(transition_counts):
        sparse.save_npz(f"./matrices/transition_counts_action{action}.npz", matrix.tocsr())


    # To load:
    # matrix = sparse.load_npz("transition_counts_action0.npz")
    print("Transition matrices have been saved to transition_matrices.npy")

# print(f"The transition matrices are: {transition_matrices.shape}")



# # Example of how to print transition probabilities
# for action_idx in range(min(3, n_actions)):
#     print(f"\nTransition probabilities for action {action_idx}:")
#     for state_idx in range(min(3, n_states)):
#         probs = transition_matrices[action_idx, state_idx]
#         print(f"  From state {state_idx} probabilities: {probs[probs > 0]}")

    env.close()








# # Reset the environment
# obs, info = env.reset()

# # Simulate for 500 steps with a random policy
# for _ in range(500):
#     action = env.action_space.sample()  # Sample a random action
#     obs, reward, terminated, truncated, info = env.step(action)
#     # print(obs)

#     env.render()
#     # Print XY coordinates of end-effector (fingertip)
#     # Get end-effector position from environment's state
#     fingertip_xy = env.unwrapped.get_body_com("fingertip")[:2]
#     print(f"End-Effector XY coordinates: {fingertip_xy}")
#     # time.sleep(1)

#     if terminated or truncated:
#         obs, info = env.reset()

# # Close the environment
# env.close()