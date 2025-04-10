import gymnasium as gym
import matplotlib.pyplot as plt
import time
import numpy as np
# Create the Reacher environment
env = gym.make('Reacher-v5')

# Discretization parameters
xy_grid_size = 0.02  # 2cm grid resolution
action_grid_size = 0.2  # discrete bins in [-1, 1]

# Define state discretization boundaries (workspace ~0.2 radius)
xy_bound = 0.2
x_bins = np.arange(-xy_bound, xy_bound + xy_grid_size, xy_grid_size)
y_bins = np.arange(-xy_bound, xy_bound + xy_grid_size, xy_grid_size)

# Define action discretization boundaries
action_bins = np.arange(-1, 1 + action_grid_size, action_grid_size)

# Helper function to discretize XY positions
def discretize_xy(xy):
    x_idx = np.digitize(xy[0], x_bins) - 1
    y_idx = np.digitize(xy[1], y_bins) - 1
    return (x_idx, y_idx)

# Helper function to discretize actions
def discretize_action(act):
    a0_idx = np.digitize(act[0], action_bins) - 1
    a1_idx = np.digitize(act[1], action_bins) - 1
    return (a0_idx, a1_idx)

# Initialize transition matrices
state_shape = (len(x_bins), len(y_bins))
action_shape = (len(action_bins), len(action_bins))

# Transition counts
transitions = {}

# Run simulations
n_steps = 5000
obs, _ = env.reset()

for _ in range(n_steps):
    action = env.action_space.sample()
    discrete_action = discretize_action(action)

    # Current discretized state
    current_xy = env.unwrapped.get_body_com("fingertip")[:2]
    current_state = discretize_xy(current_xy)

    obs, reward, terminated, truncated, _ = env.step(action)

    next_xy = env.unwrapped.get_body_com("fingertip")[:2]
    next_state = discretize_xy(next_xy)

    key = (current_state, discrete_action)

    if key not in transitions:
        transitions[key] = {}

    transitions[key][next_state] = transitions[key].get(next_state, 0) + 1

    if terminated or truncated:
        obs, _ = env.reset()

# Normalize to form transition probabilities
transition_matrices = {}
for key, next_states in transitions.items():
    total_transitions = sum(next_states.values())
    transition_matrices[key] = {state: count / total_transitions for state, count in next_states.items()}

# Example of how to print transition probabilities
for key, probs in list(transition_matrices.items())[:5]:  # Print first 5 entries
    print(f"From state {key[0]} with action {key[1]}:")
    for next_state, prob in probs.items():
        print(f"  to state {next_state} probability {prob:.2f}")

env.close()

# Reset the environment
# obs, info = env.reset()

# Simulate for 500 steps with a random policy
# for _ in range(500):
    action = env.action_space.sample()  # Sample a random action
    obs, reward, terminated, truncated, info = env.step(action)
    # print(obs)

    env.render()
    # Print XY coordinates of end-effector (fingertip)
    # Get end-effector position from environment's state
    fingertip_xy = env.unwrapped.get_body_com("fingertip")[:2]
    print(f"End-Effector XY coordinates: {fingertip_xy}")
    time.sleep(1)

    if terminated or truncated:
        obs, info = env.reset()

# Close the environment
# env.close()