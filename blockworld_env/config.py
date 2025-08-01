

RM_PATH = "../rm_examples/dynamic_stacking.txt"
RM_PATH_WRONG = "../rm_examples/dynamic_stacking_wrong.txt"
RM_PATH_ADV = "../rm_examples/adv_stacking.txt"
RM_PATH_EXTRA = "../rm_examples/stack_extra.txt"

NUM_PILES = 3


TARGET_STATE_1 = ((0,1,2),(),())
TARGET_STATE_2 = ((),(1,2,0),())
TARGET_STATE_3 = ((),(),(2,1,0))
TARGET_STATE_4 = ((),(0,1,2),())

BAD_STATE = ((0,),(1,),(2,))

GAMMA = 0.99
HORIZON = 10

REWARD_PARAMETER = 15.0
REWARD_PARAMETER_ADV_1 = 1.0
REWARD_PARAMETER_ADV_2 = 0.2

POLICY_PATH = f"policies/soft_policy_{REWARD_PARAMETER}"
POLICY_PATH_ADV = f"policies/soft_policy_adv_{REWARD_PARAMETER_ADV_1}_{REWARD_PARAMETER_ADV_2}"
POLICY_PATH_EXTRA = f"policies/soft_policy_extra"


import numpy as np
SEED = 43# 43
 