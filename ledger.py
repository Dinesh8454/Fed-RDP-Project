# ledger.py
import time, pickle, hashlib, json, os

# Prefer tinydb if available, otherwise fallback to JSON append
try:
    from tinydb import TinyDB
    _db = TinyDB("ledger.json")
    def _write_record(rec):
        _db.insert(rec)
except Exception:
    # fallback: append to JSON lines file
    def _write_record(rec):
        with open("ledger.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")

def _hash_state(sd, limit=10):
    # small deterministic fingerprint: take first `limit` elements from each param
    mini = {k: (v.cpu().numpy().flatten()[:limit].tolist() if hasattr(v, "cpu") else None) for k,v in sd.items()}
    return hashlib.sha256(pickle.dumps(mini)).hexdigest()

def log(round_no, client_id, state_dict, score, contrib, loss, ts=None):
    """
    Standardized ledger log function expected by run_experiment.py
    Arguments:
      round_no (int), client_id (int), state_dict (OrderedDict), score (float),
      contrib (float), loss (float or None)
    """
    rec = {
        "round": int(round_no),
        "client": int(client_id),
        "hash": _hash_state(state_dict),
        "score": float(score),
        "contrib": float(contrib),
        "loss": None if loss is None else float(loss),
        "ts": float(time.time() if ts is None else ts)
    }
    _write_record(rec)
