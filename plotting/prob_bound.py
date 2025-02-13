import numpy as np
import matplotlib.pyplot as plt

# Define parameters
n1, n2 = 30000, 20000  # Large values for n1 and n2
A = 9  # Value for |A|
epsilon = 0.1  # Given epsilon

# Define the function f(epsilon_1)
def f(epsilon_1, n1, n2, A, epsilon):
    term1 = 1 - 2 * np.exp(-n1 * epsilon_1**2 / (2 * A))
    term2 = 1 - 2 * np.exp(-n2 * (epsilon - epsilon_1)**2 / (2 * A))
    return term1 * term2

# Generate values for epsilon_1 in a larger range
epsilon_1_values = np.linspace(0, epsilon, 200)
f_values = f(epsilon_1_values, n1, n2, A, epsilon)

# Plot the function with LaTeX labels and title
plt.figure(figsize=(8, 5))
plt.plot(epsilon_1_values, f_values, label=r'$f(\epsilon_1)$', color='b')
# Draw a vertical line at epsilon/2
plt.axvline(x=epsilon/2, color='r', linestyle='--', label=r'$\epsilon/2$')

plt.xlabel(r'$\epsilon_1$', fontsize=14)
plt.ylabel(r'$f(\epsilon_1)$', fontsize=14)
plt.title(r'Plot of $f(\epsilon_1)$ with $n_1={30000}$, $n_2=20000$, $|\mathcal{A}|=9$, $\epsilon=0.1$', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
