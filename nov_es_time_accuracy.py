import json
import sys
import os
import matplotlib.pyplot as plt

def load_ledger_metrics(outdir):
    ledger_path = os.path.join(outdir, "ledger.json")

    rounds = []
    accs = []
    times = []

    with open(ledger_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            entry = json.loads(line)   # ✅ JSONL SAFE

            if "round" in entry:
                rounds.append(entry["round"])
            if "avg_acc" in entry:
                accs.append(entry["avg_acc"])
            if "time" in entry:
                times.append(entry["time"])

    return rounds, accs, times


# ---------------- MAIN ----------------
base_dir = sys.argv[1]
es_dir = sys.argv[2]

r_b, a_b, t_b = load_ledger_metrics(base_dir)
r_e, a_e, t_e = load_ledger_metrics(es_dir)

# -------- Accuracy Plot --------
plt.figure(figsize=(7,5))
plt.plot(r_b, a_b, marker="o", label="Base Training")
plt.plot(r_e, a_e, marker="s", label="Early Stopping")
plt.xlabel("Round")
plt.ylabel("Avg Accuracy")
plt.title("Accuracy vs Round (Early Stopping vs Base)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("nov_es_accuracy.png")
plt.show()

# -------- Time Plot --------
plt.figure(figsize=(7,5))
plt.plot(r_b, t_b, marker="o", label="Base Training")
plt.plot(r_e, t_e, marker="s", label="Early Stopping")
plt.xlabel("Round")
plt.ylabel("Training Time (sec)")
plt.title("Training Time vs Round")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("nov_es_time.png")
plt.show()

print("Saved: nov_es_accuracy.png, nov_es_time.png")