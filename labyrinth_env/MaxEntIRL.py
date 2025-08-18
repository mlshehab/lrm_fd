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

class InfiniteHorizonMaxEntIRLWeighted:
    """
    Infinite-Horizon MaxEnt IRL with discounted occupancy matching.

    Pass an MDP that already represents the state space you want to learn on.
    (If you need RM-aware behavior, construct a product MDP first and pass it here.)

    Parameters
    ----------
    mdp : object with fields {n_states, n_actions, P (list of A matrices SxS), gamma}
    expert_trajs : list[list[tuple[int,int]]]
        Each trajectory is a list of (state, action) pairs in this MDP's state space.
    is_featurized : bool
        If True, uses a 3-dim landmark feature map by default (H, W, Other).
    landmark_states : list[int] | None
        Two landmark states for the default 3-feature map.
    feature_fn : callable | None
        Optional custom feature map: phi = feature_fn(state) -> 1D np.ndarray.
        Overrides `is_featurized/landmark_states` if provided.
    n_iter : int
    lr : float
    """

    def __init__(
        self,
        mdp,
        expert_trajs,
        is_featurized=False,
        landmark_states=None,
        feature_fn=None,
        n_iter=100,
        lr=0.1,
    ):
        self.mdp = mdp
        self.expert_trajs = expert_trajs
        self.S = mdp.n_states
        self.A = mdp.n_actions
        self.gamma = mdp.gamma
        self.n_iter = n_iter
        self.lr = lr

        # Feature definition
        self.is_featurized = is_featurized
        self.landmark_states = landmark_states
        self.feature_fn = feature_fn

        if self.feature_fn is not None:
            # Infer feature dimension from any valid state (assume 0 exists)
            try:
                self.n_features = int(np.asarray(self.feature_fn(0)).size)
            except Exception as e:
                raise ValueError("feature_fn(state) must return a 1D array-like.") from e
        elif self.is_featurized:
            if not (isinstance(self.landmark_states, (list, tuple)) and len(self.landmark_states) == 2):
                raise ValueError("Provide landmark_states=[s_home, s_water] (two ints) for featurized IRL.")
            self.n_features = 3
        else:
            # Dense one-hot per state
            self.n_features = self.S

        # Parameters
        self.weights = np.zeros(self.n_features, dtype=float)

        # Empirical start-state distribution μ0 from expert data
        self.mu0 = self._empirical_mu0()

        # Discounted expert feature expectations (scaled to sum to 1 across time)
        self.expert_features = self._compute_expert_features_discounted()

    # -------------------- Features --------------------

    def _phi(self, s: int) -> np.ndarray:
        """Feature vector ϕ(s)."""
        if self.feature_fn is not None:
            return np.asarray(self.feature_fn(s), dtype=float)

        if self.is_featurized:
            f = np.zeros(3, dtype=float)
            if s == self.landmark_states[0]:
                f[0] = 1.0
            elif s == self.landmark_states[1]:
                f[1] = 1.0
            else:
                f[2] = 1.0
            return f

        # Dense one-hot
        f = np.zeros(self.S, dtype=float)
        f[s] = 1.0
        return f
    
    def _r(self, s: int) -> float:
        return float(np.dot(self.weights, self._phi(s)))

    # -------------------- Expert stats --------------------

    def _empirical_mu0(self) -> np.ndarray:
        mu0 = np.zeros(self.S, dtype=float)
        for traj in self.expert_trajs:
            if not traj:
                continue
            s0 = int(traj[0][0])
            mu0[s0] += 1.0
        total = mu0.sum()
        if total == 0:
            return np.ones(self.S) / self.S
        return mu0 / total

    def _compute_expert_features_discounted(self) -> np.ndarray:
        """
        (1-γ) * E_{τ~expert} [ sum_t γ^t ϕ(s_t) ]
        """
        feat = np.zeros(self.n_features, dtype=float)
        g = self.gamma
        for traj in self.expert_trajs:
            w = 1.0
            for (s, _a) in traj:
                feat += w * self._phi(int(s))
                w *= g
        # Average over trajectories and scale by (1-γ) so totals align with discounted occupancy
        N = max(1, len(self.expert_trajs))
        feat *= (1.0 - g) / N
        return feat

    # -------------------- Soft value iteration --------------------

    def soft_value_iteration(self, tol=1e-6, max_iter=10_000):
        S, A, g = self.S, self.A, self.gamma
        V = np.zeros(S, dtype=float)
        Q = np.zeros((S, A), dtype=float)

        for _ in range(max_iter):
            V_old = V.copy()
            # Q(s,a) = r(s) + γ * Σ P(s'|s,a) V(s')
            for s in range(S):
                r_s = self._r(s)
                for a in range(A):
                    Q[s, a] = r_s + g * np.dot(self.mdp.P[a][s, :], V_old)

            # V(s) = logsumexp_a Q(s,a)
            # Stable: subtract row max
            row_max = Q.max(axis=1)
            V = row_max + np.log(np.exp(Q - row_max[:, None]).sum(axis=1))

            if np.max(np.abs(V - V_old)) < tol:
                break

        # π(a|s) = softmax(Q(s,·))
        row_max = Q.max(axis=1)
        logits = Q - row_max[:, None]
        exp_logits = np.exp(logits)
        policy = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        return policy

    # -------------------- Discounted occupancy --------------------

    def discounted_occupancy(self, policy, tol=1e-10, max_steps=10000):
        """
        Returns d_π ∈ R^S where d_π = (1-γ) * Σ_t γ^t Pr(s_t=· | μ0, π)
        This sums to 1 and is comparable to discounted expert features.
        """
        S, A, g = self.S, self.A, self.gamma

        # p_{t+1} = p_t * P_π
        # where P_π[s, s'] = Σ_a π[a|s] P_a[s, s']
        # We'll iterate forward and accumulate the discounted sum.
        p_t = self.mu0.copy()
        d = np.zeros(S, dtype=float)

        for t in range(max_steps):
            d += (g**t) * p_t

            # compute next distribution
            p_next = np.zeros(S, dtype=float)
            # loop form to avoid giant dense (SxSxA) tensor
            for s in range(S):
                for a in range(A):
                    if policy[s, a] == 0.0:
                        continue
                    p_sa = p_t[s] * policy[s, a]
                    if p_sa != 0.0:
                        p_next += p_sa * self.mdp.P[a][s, :]

            if np.linalg.norm(p_next - p_t, 1) < tol:
                # Geometric tail: add remaining mass analytically
                d += (g**(t + 1)) / (1.0 - g) * p_next
                break

            p_t = p_next

        d *= (1.0 - g)  # normalize so sum(d)=1
        return d

    def _expected_features_under_policy(self, d_occ: np.ndarray) -> np.ndarray:
        feat = np.zeros(self.n_features, dtype=float)
        for s in range(self.S):
            feat += d_occ[s] * self._phi(s)
        return feat

    # -------------------- Training --------------------

    def train(self, verbose=False):
        for it in range(self.n_iter):
            policy = self.soft_value_iteration()
            d_occ = self.discounted_occupancy(policy)
            e_pi = self._expected_features_under_policy(d_occ)

            # Gradient descent on (E_pi - E_exp)
            grad = e_pi - self.expert_features
            self.weights -= self.lr * grad

            if verbose and (it % 10 == 0 or it == self.n_iter - 1):
                print(f"[IRL] iter={it:03d}  ||grad||={np.linalg.norm(grad):.4e}")

        return self.weights

    # -------------------- Convenience --------------------

    def policy(self):
        """Return current soft-optimal policy for the current weights."""
        return self.soft_value_iteration()

class InfiniteHorizonMaxEntIRL:
    """
    Infinite Horizon MaxEnt IRL
    """
    def __init__(self, mdp, expert_trajs, is_featurized = False, landmark_states = None, n_iter=100, lr=0.1):
        self.mdp = mdp
        self.expert_trajs = expert_trajs
        self.n_states = mdp.n_states
        self.n_actions = mdp.n_actions
        self.discount = mdp.gamma
        self.n_iter = n_iter
        self.lr = lr

        self.is_featurized = is_featurized
        if is_featurized:
            self.n_features = 3
        else:
            self.n_features = self.n_states
       
        self.weights = np.random.randn(self.n_features)

        if is_featurized:
            assert landmark_states is not None, "Landmark states must be provided for featurized IRL"
            self.landmark_states = landmark_states
        else:
            self.landmark_states = None

        # Expert feature expectations: one-hot per state
        self.expert_features = self._compute_expert_features()
        # print(f"The expert features are: {self.expert_features}")

    def _compute_expert_features(self):
        features = np.zeros(self.n_features)
        if self.is_featurized:
            
            for traj in self.expert_trajs:
                for state, _ in traj:
                    if state == self.landmark_states[0]:
                        features[0] += 1
                    elif state == self.landmark_states[1]:
                        features[1] += 1
                    else:
                        features[2] += 1
        else:
            features = np.zeros(self.n_states)
            for traj in self.expert_trajs:
                for state, _ in traj:
                    features[state] += 1
        return features / len(self.expert_trajs)
    
    def feature_vector(self, state):
        features = np.zeros(self.n_features)
        if state == self.landmark_states[0]:
            features[0] += 1
        elif state == self.landmark_states[1]:
            features[1] += 1
        else:
            features[2] += 1
        return features


    def soft_value_iteration(self, tol=1e-4, max_iter=1000):
        V = np.zeros(self.n_states)
        Q = np.zeros((self.n_states, self.n_actions))
        policy = np.zeros((self.n_states, self.n_actions))
        
        for _ in range(max_iter):
            V_old = V.copy()
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    next_probs = self.mdp.P[a][s,:]
                    if self.is_featurized:
                        Q[s, a] = np.dot(self.weights, self.feature_vector(s)) + self.discount * np.dot(next_probs, V)
                    else:
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
                    next_probs = self.mdp.P[a][s,:]
                    D_next += D[s] * policy[s, a] * next_probs
            if np.linalg.norm(D_next - D, 1) < tol:
                break
            D = D_next
        return D
    
    def _expected_features_under_policy(self, D):
        """
        Turn state visitation frequencies D (size n_states) into
        expected feature counts (size n_features) using ϕ(s).
        """
        feat = np.zeros(self.n_features)
        for s in range(self.n_states):
            feat += D[s] * self.feature_vector(s)
        return feat
    
    def train(self):
        for _ in tqdm(range(self.n_iter)):
            policy = self.soft_value_iteration()
            D = self.compute_state_visitation_frequencies(policy)

            if self.is_featurized:
                grad = self._expected_features_under_policy(D) - self.expert_features
                # print(f"The gradient is: {np.linalg.norm(grad)}")
                self.weights -= self.lr * grad
            else:
                grad = D - self.expert_features
                print(f"The gradient is: {np.linalg.norm(grad)}")
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


if __name__ == "__main__":
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



