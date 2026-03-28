import matplotlib.pyplot as plt
import numpy as np

malicious = [1, 2, 3]
fedavg = [0.9, 0.8, 0.7]
fedrdp_a = [0.95, 0.92, 0.88]
fedrdp_b = [0.97, 0.94, 0.89]

x = np.arange(len(malicious))
w = 0.25

plt.bar(x-w, fedavg, w, label="Fed-AVG")
plt.bar(x, fedrdp_a, w, label="Fed-RDP-a")
plt.bar(x+w, fedrdp_b, w, label="Fed-RDP-b")

plt.xticks(x, malicious)
plt.xlabel("Number of Malicious Clients")
plt.ylabel("Weighted Sum")
plt.title("Comparison of Weighted Sums")
plt.legend()
plt.savefig("weighted_sum_comparison.png")
plt.show()
