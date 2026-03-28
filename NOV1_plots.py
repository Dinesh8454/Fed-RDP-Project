import json
import matplotlib.pyplot as plt
import sys

folder = sys.argv[1]
ledger_file = f"{folder}/ledger.json"

rounds = []
losses = []

with open(ledger_file, "r") as f:
    for line in f:
        entry = json.loads(line)   # ✅ fix
        if "loss" in entry and entry["loss"] is not None:
            rounds.append(entry["round"])
            losses.append(entry["loss"])

plt.plot(rounds, losses, marker="o")
plt.xlabel("Round")
plt.ylabel("Loss")
plt.title("Training Loss per Round")
plt.grid(True)
plt.savefig(f"{folder}/loss_plot.png")
plt.show()
