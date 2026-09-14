import argparse
import glob
from pathlib import Path

import numpy as np

from learning.local_structural_map.tools.io_utils import load_map_npz, write_json


def features(paths):
    xs = []
    for p in paths:
        t = load_map_npz(p)["tensor"]
        # Compact deterministic diagnostic features, not a production model.
        xs.append(np.concatenate([t.mean(axis=(1, 2)), t.std(axis=(1, 2)), t.max(axis=(1, 2))]))
    return np.asarray(xs, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-glob", required=True)
    parser.add_argument("--b-glob", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()
    a = [Path(p) for p in sorted(glob.glob(args.a_glob))]
    b = [Path(p) for p in sorted(glob.glob(args.b_glob))]
    if not a or not b:
        write_json(args.output, {"status": "UNAVAILABLE", "reason": "both sources required", "a_count": len(a), "b_count": len(b)})
        return
    x = np.concatenate([features(a), features(b)], axis=0)
    y = np.concatenate([np.zeros(len(a), dtype=np.int64), np.ones(len(b), dtype=np.int64)])
    rng = np.random.default_rng(123)
    idx = rng.permutation(len(y))
    split = max(2, int(0.7 * len(y)))
    train = idx[:split]
    val = idx[split:]
    mu = x[train].mean(axis=0, keepdims=True)
    sig = x[train].std(axis=0, keepdims=True) + 1e-6
    x = (x - mu) / sig
    w = np.zeros((x.shape[1], 2), dtype=np.float32)
    b0 = np.zeros((2,), dtype=np.float32)
    losses = []
    for _ in range(args.steps):
        logits = x[train] @ w + b0
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        loss = -np.log(probs[np.arange(len(train)), y[train]] + 1e-8).mean()
        losses.append(float(loss))
        probs[np.arange(len(train)), y[train]] -= 1.0
        probs /= len(train)
        w -= 0.1 * (x[train].T @ probs)
        b0 -= 0.1 * probs.sum(axis=0)
    pred_train = np.argmax(x[train] @ w + b0, axis=1)
    pred_val = np.argmax(x[val] @ w + b0, axis=1)
    write_json(args.output, {
        "status": "PASS",
        "a_count": len(a),
        "b_count": len(b),
        "feature_dim": int(x.shape[1]),
        "train_count": int(len(train)),
        "val_count": int(len(val)),
        "train_accuracy": float(np.mean(pred_train == y[train])),
        "val_accuracy": float(np.mean(pred_val == y[val])) if len(val) else None,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
    })


if __name__ == "__main__":
    main()
