import re
import matplotlib.pyplot as plt

log_file = "outputs_base_mnist/ledger.log"

rounds = []
stability = []

with open(log_file, "r") as f:
    for line in f:
        if "stability" in line:
            r = int(re.search(r"Round (\d+)", line).group(1))
            s = float(re.search(r"stability: ([0-9.]+)", line).group(1))
            rounds.append(r)
            stability.append(s)

plt.plot(rounds, stability, marker="o")
plt.xlabel("Round")
plt.ylabel("Stability Score (STD)")
plt.title("Client Contribution Stability Across Rounds")
plt.grid(True)
plt.savefig("stability_score.png")
plt.show()
