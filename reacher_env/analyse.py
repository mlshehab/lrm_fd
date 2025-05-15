
import pickle
with open('results/results.pkl', 'rb') as f:
    results = pickle.load(f)

# Find the pair with highest frequency of 4
max_freq = 0
best_pair = None

for alpha, beta in results:
    values = results[(alpha, beta)]['values']
    freq_4 = values.count(4)
    if freq_4 > max_freq:
        max_freq = freq_4
        best_pair = (alpha, beta)

# Print the best pair
print(f"Best pair: alpha = {best_pair[0]}, beta = {best_pair[1]}")

# Write results to file
with open('results/analysis_output.txt', 'w') as f:
    f.write(f"Best pair: alpha = {best_pair[0]}, beta = {best_pair[1]}\n")
    f.write(f"Frequency of 4: {max_freq}\n")
    f.write(f"All values: {results[best_pair]['values']}\n")
