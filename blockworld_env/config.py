

RM_PATH = "../rm_examples/dynamic_stacking.txt"

NUM_PILES = 3

POLICY_PATH = "./policies/soft_policy.npy"

TARGET_STATE_1 = ((0,1,2),(),())
TARGET_STATE_2 = ((),(1,2,0),())
TARGET_STATE_3 = ((),(),(2,1,0))
BAD_STATE = ((0,),(1,),(2,))



GAMMA = 0.9
HORIZON = 10
