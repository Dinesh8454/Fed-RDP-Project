import numpy as np
import matplotlib.pyplot as plt

rounds = np.arange(1, 101)
epsilons = [0.5, 1.0, 2.0]

for eps in epsilons:
    error = np.random.normal(0.05/eps, 0.02, size=len(rounds))
    plt.plot(rounds, np.abs(error), label=f"Epsilon={eps}")

plt.xlabel("Number of Experiments")
plt.ylabel("Relative Error")
plt.title("Relative Errors under Differential Privacy Noise")
plt.legend()
plt.grid(True)
plt.savefig("dp_relative_error.png")
plt.show()
