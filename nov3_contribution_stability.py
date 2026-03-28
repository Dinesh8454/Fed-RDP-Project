import json
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# -------------------------------
# Arguments
# -------------------------------
if len(sys.argv) < 2:
    print("Usage: python nov3_contribution_stability.py <output_dir>")
    sys.exit(1)

outdir = sys.argv[1]
ledger_path = os.path.join(outdir, "ledger.json")

if not os.path.exists(ledger_path):
    print("ledger.json not found")
    sys.exit(1)

# -------------------------------
# Load ledger
# -------------------------------
with open(ledger_path, "r") as f:
    ledger = json.load(f)

if "_default" not in ledger:
    print("Unsupported ledger format (missing _default)")
    sys.exit(1)

entries = ledger["_default"].values()

# -------------------------------
# Collect scores per client
# -------------------------------
client_scores = defaultdict(list)

for entry in entries:
    if not isinstance(entry, dict):
        continue

    if "client" in entry and "score" in entry:
        client_scores[entry["client"]].append(entry["score"])

# -------------------------------
# Compute stability (std dev)
# -------------------------------
clients = []
stabilities = []

for cid, scores in client_scores.items():
    if len(scores) >= 2:
        clients.append(cid)
        stabilities.append(np.std(scores))

if not stabilities:
    print("No valid stability data to plot.")
    sys.exit(0)

# -------------------------------
# Plot
# -------------------------------
plt.figure(figsize=(8, 4))
plt.bar(clients, stabilities)
plt.xlabel("Client ID")
plt.ylabel("Score Stability (Std Dev)")
plt.title("Client Contribution Stability")
plt.tight_layout()

save_path = os.path.join(outdir, "client_stability.png")
plt.savefig(save_path)
plt.close()

print(f"Stability plot saved to: {save_path}")
