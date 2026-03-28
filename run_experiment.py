# run_experiment.py
import argparse, copy, time, os
import ray, torch, numpy as np
from client import Client
from contribut_eval import compute_contribs_by_similarity, monte_carlo_least_core
from reputation import update_score
from ledger import log as ledger_log
import matplotlib.pyplot as plt

# ------------------ ARGUMENTS ------------------
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='mnist', choices=['mnist','cifar10','svhn'])
parser.add_argument('--clients', type=int, default=3)
parser.add_argument('--rounds', type=int, default=10)
parser.add_argument('--local-epochs', type=int, default=1)
parser.add_argument('--malicious', type=int, default=-1)
parser.add_argument('--dp', action='store_true')
parser.add_argument('--contrib', default='similarity', choices=['similarity','leastcore'])
parser.add_argument('--outdir', default='outputs')

# 🔷 NOVELTY-3: EARLY STOPPING
parser.add_argument("--early-stop", action="store_true",
                    help="Enable early stopping")
parser.add_argument("--patience", type=int, default=3,
                    help="Rounds to wait before stopping")
parser.add_argument("--min-delta", type=float, default=0.001,
                    help="Minimum accuracy improvement")

# 🔷 NOVELTY-4: NON-IID DATA
parser.add_argument("--noniid", action="store_true",
                    help="Enable Non-IID Dirichlet label skew")

args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)
ray.init(ignore_reinit_error=True)

# ------------------ AGGREGATION ------------------
def weighted_aggregate(state_dicts, scores):
    agg = copy.deepcopy(state_dicts[0])
    for k in agg:
        agg[k] = torch.zeros_like(agg[k])

    total = float(sum(scores)) + 1e-12
    for i, sd in enumerate(state_dicts):
        w = scores[i] / total
        for k in agg:
            agg[k] += sd[k].float() * w
    return agg

# ------------------ CLIENTS ------------------
clients = [
    Client.remote(
        i,
        dataset=args.dataset,
        malicious_id=args.malicious,
        num_clients=args.clients,
        noniid=args.noniid   # 🔷 Novelty-4 hook
    )
    for i in range(args.clients)
]

# initialize global model from client 0
_ = ray.get(clients[0].train.remote(local_epochs=0))

# reputation scores
scores = [1.0] * args.clients

# logging arrays
history_acc = []
history_scores = []

# 🔷 EARLY STOP VARIABLES
best_acc = 0.0
no_improve_rounds = 0

# ================== TRAINING LOOP ==================
for r in range(1, args.rounds + 1):
    print(f"\n[Server] Round {r} starting ...", flush=True)
    t0 = time.time()

    # send round info to clients
    for c in clients:
        c.current_round = r
        c.max_rounds = args.rounds

    # local training
    futures = [
        clients[i].train.remote(
            local_epochs=args.local_epochs,
            score=scores[i],
            dp=args.dp
        )
        for i in range(args.clients)
    ]
    results = ray.get(futures)

    state_dicts = [res[0] for res in results]
    deltas      = [res[1] for res in results]
    metrics     = [res[2] for res in results]

    # aggregate
    global_state = weighted_aggregate(state_dicts, scores)

    # ledger logging
    for i, sd in enumerate(state_dicts):
        ledger_log(
            round_no=r,
            client_id=i,
            state_dict=sd,
            score=scores[i],
            contrib=0.0,
            loss=metrics[i].get('loss', None)
        )

    # broadcast global model
    [c.set_global.remote(global_state) for c in clients]

    # contribution evaluation
    if args.contrib == 'similarity':
        contribs = compute_contribs_by_similarity(deltas)
    else:
        contribs = monte_carlo_least_core(
            state_dicts,
            None,
            None,
            M=100
        )

    # reputation update
    new_scores = []
    for i in range(args.clients):
        s_new = update_score(scores[i], contribs[i], args.clients)
        new_scores.append(s_new)
    scores = new_scores

    print(f"[Server] Round {r} contribs: {[round(c,3) for c in contribs]}")
    print(f"[Server] Round {r} scores:   {[round(s,3) for s in scores]}")

    # evaluate global model
    accs = ray.get([c.eval_global.remote() for c in clients])
    round_acc = np.mean(accs)

    history_acc.append(round_acc)
    history_scores.append(scores.copy())

    print(f"[Server] Round {r} avg client acc: {round_acc:.4f}")
    print(f"[Server] Round {r} done in {time.time() - t0:.2f}s")

    # ================== EARLY STOP LOGIC ==================
    if args.early_stop:
        if round_acc > best_acc + args.min_delta:
            best_acc = round_acc
            no_improve_rounds = 0
        else:
            no_improve_rounds += 1

        if no_improve_rounds >= args.patience:
            print(f"[Server] Early stopping triggered at round {r}")
            break

# ================== PLOTS ==================
plt.figure()
plt.plot(range(1, len(history_acc) + 1), history_acc, marker='o')
plt.xlabel("Round")
plt.ylabel("Avg client accuracy")
plt.title(f"{args.dataset} - avg accuracy")
plt.grid(True)
plt.savefig(os.path.join(args.outdir, "accuracy_vs_round.png"))

arr = np.array(history_scores)
if arr.size:
    plt.figure(figsize=(6, 3))
    for i in range(arr.shape[1]):
        plt.plot(range(1, arr.shape[0] + 1), arr[:, i], label=f"c{i}")
    plt.legend()
    plt.xlabel("Round")
    plt.ylabel("Score")
    plt.savefig(os.path.join(args.outdir, "scores_vs_round.png"))

ray.shutdown()
print(f"\n[Server] Saved plots & summary to: {args.outdir}/ and ledger.json")
print("[Server] Training complete! Ray shutdown successful.")
