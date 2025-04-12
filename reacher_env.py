import gymnasium as gym
import numpy as np
from tqdm import tqdm
# Environment setup


# Discretization parameters
xy_grid_size = 0.01  # 2cm grid resolution
action_grid_size = 0.1  # discrete bins in [-1, 1]

# Define state discretization boundaries (workspace ~0.2 radius)
xy_bound = 0.22
x_bins = np.arange(-xy_bound, xy_bound + xy_grid_size, xy_grid_size)
y_bins = np.arange(-xy_bound, xy_bound + xy_grid_size, xy_grid_size)

# Define action discretization boundaries
action_bins = np.arange(-1, 1 + action_grid_size, action_grid_size)

# Helper function to discretize XY positions
def discretize_xy(xy):
    x_idx = np.digitize(xy[0], x_bins) - 1
    y_idx = np.digitize(xy[1], y_bins) - 1
    return (x_idx, y_idx)

def midpoint_from_idx(x_idx, y_idx):
    x_mid = x_bins[x_idx] + xy_grid_size / 2
    y_mid = y_bins[y_idx] + xy_grid_size / 2
    return (x_mid, y_mid)

def midpoint_from_action_idx(a):
    a0_idx, a1_idx = a
    a0_mid = action_bins[a0_idx] + action_grid_size / 2
    a1_mid = action_bins[a1_idx] + action_grid_size / 2
    return (a0_mid, a1_mid)

# Helper function to discretize actions
def discretize_action(act):
    a0_idx = np.digitize(act[0], action_bins) - 1
    a1_idx = np.digitize(act[1], action_bins) - 1
    return (a0_idx, a1_idx)

# Generate state and action indexing dictionaries
state_to_idx = {(i, j): idx for idx, (i, j) in enumerate([(x, y) for x in range(len(x_bins)) for y in range(len(y_bins))])}
idx_to_state = {idx: (i, j) for (i, j), idx in state_to_idx.items()}

action_to_idx = {(i, j): idx for idx, (i, j) in enumerate([(a0, a1) for a0 in range(len(action_bins)) for a1 in range(len(action_bins))])}
idx_to_action = {idx: (i, j) for (i, j), idx in action_to_idx.items()}

# Initialize transition matrices
n_states = len(state_to_idx)
n_actions = len(action_to_idx)
transition_counts = np.zeros((n_actions, n_states, n_states))
print(transition_counts.shape)
# Run simulations
if __name__ == "__main__":
    env = gym.make('Reacher-v5')
    n_steps = int(10000)
    obs, _ = env.reset()

    for _ in tqdm(range(n_steps)):
        action = env.action_space.sample()
        discrete_action = discretize_action(action)
        action_idx = action_to_idx[discrete_action]

        current_xy = env.unwrapped.get_body_com("fingertip")[:2]
        current_state = discretize_xy(current_xy)
        current_state_idx = state_to_idx[current_state]

        obs, reward, terminated, truncated, _ = env.step(action)

        next_xy = env.unwrapped.get_body_com("fingertip")[:2]
        next_state = discretize_xy(next_xy)
        next_state_idx = state_to_idx[next_state]

        transition_counts[action_idx, current_state_idx, next_state_idx] += 1

        if terminated or truncated:
            obs, _ = env.reset()

    # Normalize to form transition probability matrices
    transition_matrices = transition_counts / np.maximum(transition_counts.sum(axis=2, keepdims=True), 1)

    # Save the transition matrices to a file
    np.save("transition_matrices.npy", transition_matrices)
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