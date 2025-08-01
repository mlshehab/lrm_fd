import numpy as np
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from dynamics.GridWorld import BasicGridWorld

from utils.ne_utils import u_from_obs
from reward_machine.reward_machine import RewardMachine
import config
from tqdm import tqdm
import pickle

class InfiniteHorizonMaxEntIRL:
    def __init__(self, env, expert_trajs, n_iter=100, lr=0.1):
        self.env = env
        self.expert_trajs = expert_trajs
        self.n_states = env.n_states
        self.n_actions = env.n_actions
        self.discount = env.discount
        self.n_iter = n_iter
        self.lr = lr
        
        # Learnable reward weights: one per state
        self.weights = np.random.randn(self.n_states)
        
        # Expert feature expectations: one-hot per state
        self.expert_features = self._compute_expert_features()
    
    def _compute_expert_features(self):
        features = np.zeros(self.n_states)
        for traj in self.expert_trajs:
            for state, _ in traj:
                features[state] += 1
        return features / len(self.expert_trajs)
    
    def soft_value_iteration(self, tol=1e-4, max_iter=1000):
        V = np.zeros(self.n_states)
        Q = np.zeros((self.n_states, self.n_actions))
        policy = np.zeros((self.n_states, self.n_actions))
        
        for _ in range(max_iter):
            V_old = V.copy()
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    next_probs = self.env.transition_probability[s, a]
                    Q[s, a] = self.weights[s] + self.discount * np.dot(next_probs, V)
                V[s] = logsumexp(Q[s])
            if np.max(np.abs(V - V_old)) < tol:
                break

        for s in range(self.n_states):
            policy[s] = np.exp(Q[s] - V[s])
            policy[s] /= policy[s].sum()

        return policy

    def compute_state_visitation_frequencies(self, policy, tol=1e-6, max_iter=10000):
        D = np.ones(self.n_states) / self.n_states  # Uniform init
        for _ in range(max_iter):
            D_next = np.zeros(self.n_states)
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    next_probs = self.env.transition_probability[s, a]
                    D_next += D[s] * policy[s, a] * next_probs
            if np.linalg.norm(D_next - D, 1) < tol:
                break
            D = D_next
        return D

    def train(self):
        for _ in tqdm(range(self.n_iter)):
            policy = self.soft_value_iteration()
            D = self.compute_state_visitation_frequencies(policy)
            grad = D - self.expert_features
            self.weights -= self.lr * grad
        return self.weights

def remove_consecutive_duplicates(s):
    elements = s.split(',')
    if not elements:
        return s  # Handle edge case
    result = [elements[0]]
    for i in range(1, len(elements)):
        if elements[i] != elements[i - 1]:
            result.append(elements[i])
    return ','.join(result)

# Environment
grid_size = 4
reward_states = [0, 14]
gt_reward = np.zeros(grid_size**2)
gt_reward[reward_states[0]] = -1
gt_reward[reward_states[1]] = 5

env = BasicGridWorld(grid_size=grid_size, wind=0.1, discount=0.9, horizon=10)

# Expert policy and trajectories
soft_optimal_policy = np.load('./policies/soft_patrol_policy.npy')

L = {}
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

expert_trajs = []

rm = RewardMachine(config.RM_PATH)

for _ in range(2500):
    traj = []
    state = np.random.randint(0, env.n_states)
    label = L[state] + ','
    compressed_label = remove_consecutive_duplicates(label)
    u = u_from_obs(label, rm)
    for _ in range(15):  # Enough steps to simulate stationarity
        idx = u * env.n_states + state
        action_dist = soft_optimal_policy[idx,:]
        
        # Sample an action from the action distribution
        a = np.random.choice(np.arange(env.n_actions), p=action_dist)


        traj.append((state, a))

        transition_probs = env.transition_probability[:,a,:]
        next_state = np.random.choice(np.arange(env.n_states), p=transition_probs[state])

        # Compress the label
        compressed_label = remove_consecutive_duplicates(label)
        l = L[next_state]
        label = label + l + ','
        u = u_from_obs(label, rm)
        
        state = next_state
    # print(compressed_label)
    expert_trajs.append(traj)


# Train Infinite Horizon MaxEnt IRL
irl = InfiniteHorizonMaxEntIRL(env, expert_trajs, n_iter=200, lr=0.1)
learned_weights = irl.train()

policy = irl.soft_value_iteration()

np.save('./policies/IRL_soft_policy.npy', policy)



