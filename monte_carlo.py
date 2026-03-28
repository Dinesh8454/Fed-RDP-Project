import numpy as np
import matplotlib.pyplot as plt

samples = np.arange(1, 26)
participants = [3, 5, 10, 20, 25]

for N in participants:
    error = 1 / (samples * np.sqrt(N))  # theoretical convergence
    plt.plot(samples, error, label=f"N={N}")

plt.xlabel("Number of Samples")
plt.ylabel("Mean Absolute Error")
plt.title("Monte Carlo Sampling Error Convergence")
plt.legend()
plt.grid(True)
plt.savefig("monte_carlo_error.png")
plt.show()
