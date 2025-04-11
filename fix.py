from reacher_env import x_bins, y_bins, action_bins, discretize_xy, discretize_action, midpoint_from_idx, midpoint_from_action_idx
from reacher_env import state_to_idx, idx_to_state, action_to_idx, idx_to_action
import numpy as np

# print(f"The x bins are: {x_bins}" )
# print(f"The y bins are: {y_bins}")
print(f"The action bins are: {action_bins}")


a1 = (0,0)
a2 = (0,1)
a3 = (1,0)
a4 = (1,1)
a5 = (-0.6,0.6)
a6 = (0.6,-0.6)
a7 = (-0.6,-0.6)
a8 = (0.6,0.6)

actions = [a1, a2, a3, a4, a5, a6, a7, a8]

for action in actions:
    discrete_action = discretize_action(action)
    print(f"Action: {action}, Discretized Action: {discrete_action}")

# x = np.array([0.2, 6.4, 3.0, 1.6,120])
# bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])
# inds = np.digitize(x, bins)
# print(inds)
# print("Keys and items in action_to_idx:")
# for key, item in action_to_idx.items():
#     print(f"Key: {key}, Item: {item}")



