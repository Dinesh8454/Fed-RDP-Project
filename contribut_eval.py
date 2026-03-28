import random
import numpy as np
import torch
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

# ======================================================
# Store contribution history per client
# ======================================================
CLIENT_SCORE_HISTORY = defaultdict(list)

# ======================================================
# Stability score (lower = more stable)
# ======================================================
def compute_stability_score(client_id):
    scores = CLIENT_SCORE_HISTORY[client_id]
    if len(scores) < 2:
        return 0.0
    return np.std(scores)

# ======================================================
# Fast proxy: gradient / weight-change similarity
# ======================================================
def compute_contribs_by_similarity(deltas):
    mats = []
    for d in deltas:
        arr = d.numpy().reshape(1, -1).astype(float)
        arr[~np.isfinite(arr)] = 0.0
        mats.append(arr)

    agg = np.sum(mats, axis=0)
    if not np.isfinite(agg).all():
        agg[~np.isfinite(agg)] = 0.0

    sims = [max(float(cosine_similarity(m, agg)[0, 0]), 0.0) for m in mats]
    arr = np.array(sims, dtype=float)

    return (arr / arr.sum()).tolist() if arr.sum() > 0 else [1.0 / len(deltas)] * len(deltas)

# ======================================================
# Paper-faithful (slower): Monte-Carlo least-core
# ======================================================
def _aggregate_subset(state_dicts, idxs):
    import copy
    agg = copy.deepcopy(state_dicts[0])
    for k in agg:
        agg[k] = torch.zeros_like(agg[k])
    for i in idxs:
        for k in agg:
            agg[k] += state_dicts[i][k]
    for k in agg:
        agg[k] /= len(idxs)
    return agg

def _eval_state(model_class, state_dict, val_loader, device="cpu"):
    m = model_class().to(device)
    m.load_state_dict(state_dict)
    m.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            p = m(x).argmax(1)
            correct += (p == y).sum().item()
            total += y.size(0)
    return correct / max(1, total)

def monte_carlo_least_core(state_dicts, model_class, val_loader, M=100):
    """
    Approximate least-core via M random coalition samples.
    Returns a list of non-negative weights that sum to 1.
    """
    try:
        import cvxpy as cp
    except Exception:
        # ---- fallback: similarity proxy ----
        deltas = []
        with torch.no_grad():
            mean = {}
            for k in state_dicts[0]:
                mean[k] = sum(sd[k] for sd in state_dicts) / len(state_dicts)
            for sd in state_dicts:
                flat = torch.cat([(sd[k] - mean[k]).view(-1).cpu() for k in sd])
                deltas.append(flat)
        return compute_contribs_by_similarity(deltas)

    n = len(state_dicts)
    vN = _eval_state(
        model_class,
        _aggregate_subset(state_dicts, list(range(n))),
        val_loader
    )

    samples = []
    for _ in range(M):
        s = random.randint(1, n - 1)
        idxs = random.sample(range(n), s)
        vS = _eval_state(
            model_class,
            _aggregate_subset(state_dicts, idxs),
            val_loader
        )
        samples.append((idxs, vS))

    x = cp.Variable(n)
    eps = cp.Variable()

    cons = [cp.sum(x[idxs]) >= vS - eps for idxs, vS in samples]
    cons += [cp.sum(x) == vN, x >= 0, eps >= 0]

    prob = cp.Problem(cp.Minimize(eps), cons)
    prob.solve(solver=cp.SCS, verbose=False)

    xs = np.maximum(np.array(x.value).flatten(), 0)
    s = xs.sum()

    return (xs / s).tolist() if s > 0 else [1.0 / n] * n
