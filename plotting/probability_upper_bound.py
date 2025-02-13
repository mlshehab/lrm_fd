import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize_scalar
# Define parameters
n1, n2 = 50000, 1  # Large values for n1 and n2
A = 9  # Value for |A|
epsilon = 0.1  # Given epsilon

# Define the function f(epsilon_1)
def f(epsilon_1, n1, n2, A, epsilon):
    term1 = np.maximum(1 - ((2**A - 2) * np.exp((-n1 * epsilon_1**2) / (2 ))), 0)
    term2 = np.maximum(1 - ((2**A - 2) * np.exp((-n2 * (epsilon - epsilon_1)**2) / (2 ))), 0)
    return term1 * term2



def find_optimal_epsilon1(n1, n2, A, epsilon):
    """
    Find the epsilon_1 value that maximizes f(epsilon_1) using scipy's optimize.
    
    Args:
        n1 (int): First sample size
        n2 (int): Second sample size 
        A (int): Size of action space
        epsilon (float): Total epsilon value
        
    Returns:
        float: The optimal epsilon_1 value that maximizes f(epsilon_1)
    """
    # Define objective function to minimize (negative of f since we want to maximize f)
    def objective(eps1):
        return -f(eps1, n1, n2, A, epsilon)
    
    # Find minimum of negative f (maximum of f) in range [0, epsilon]
    result = minimize_scalar(objective, bounds=(0, epsilon), method='bounded')
    
    return result.x



# Generate values for epsilon_1 in a larger range
epsilon_1_values = np.linspace(0, epsilon, 200)
f_values = f(epsilon_1_values, n1, n2, A, epsilon)

# Find optimal epsilon_1
optimal_epsilon1 = find_optimal_epsilon1(n1, n2, A, epsilon)
optimal_f = f(optimal_epsilon1, n1, n2, A, epsilon)
print("The optimal f is ", optimal_f)
# Plot the function with LaTeX labels and title
plt.figure(figsize=(8, 5))
plt.plot(epsilon_1_values, f_values, label=r'$f(\epsilon_1)$', color='b')
# Draw a vertical line at epsilon/2 and optimal point
plt.axvline(x=epsilon/2, color='r', linestyle='--', label=r'$\epsilon/2$')
plt.axvline(x=optimal_epsilon1, color='g', linestyle='--', label=rf'$\epsilon_1^*={optimal_epsilon1:.4f}$')

plt.xlabel(r'$\epsilon_1$', fontsize=14)
plt.ylabel(r'$f(\epsilon_1)$', fontsize=14)
plt.title(rf'Plot of $f(\epsilon_1)$ with $n_1={n1}$, $n_2={n2}$, |\mathcal{{A}}|=9, $\epsilon=0.1$', fontsize=12)
plt.legend()
plt.grid(True)

# Create figures directory if it doesn't exist
if not os.path.exists('figures'):
    os.makedirs('figures')

# Save the figure
plt.savefig('../figures/probability_bound.png', dpi=300, bbox_inches='tight')
plt.show()
