import argparse
import glob
from pathlib import Path

import numpy as np

from learning.local_structural_map.tools.io_utils import load_map_npz, write_json


class MixedSampleLoader:
    def __init__(self, paths_a: list[Path], paths_b: list[Path], batch_size: int, seed: int = 7) -> None:
        self.paths = [(p, 0) for p in paths_a] + [(p, 1) for p in paths_b]
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

    def batch(self) -> tuple[np.ndarray, np.ndarray, list[str]]:
        idx = self.rng.choice(len(self.paths), size=self.batch_size, replace=True)
        xs = []
        ys = []
        sources = []
        for i in idx:
            path, label = self.paths[int(i)]
            data = load_map_npz(path)
            xs.append(data["tensor"])
            ys.append(label)
            sources.append(data["source"])
        return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.int64), sources


class TinyCNN:
    def __init__(self, channels: int, hidden: int = 4, seed: int = 11) -> None:
        rng = np.random.default_rng(seed)
        self.w = (rng.normal(0, 0.05, size=(hidden, channels, 3, 3))).astype(np.float32)
        self.b = np.zeros((hidden,), dtype=np.float32)
        self.fc_w = (rng.normal(0, 0.05, size=(hidden, 2))).astype(np.float32)
        self.fc_b = np.zeros((2,), dtype=np.float32)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict]:
        pad = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="constant")
        n, c, h, w = x.shape
        hidden = self.w.shape[0]
        conv = np.zeros((n, hidden, h, w), dtype=np.float32)
        for oy in range(3):
            for ox in range(3):
                patch = pad[:, :, oy : oy + h, ox : ox + w]
                conv += np.einsum("nchw,kc->nkhw", patch, self.w[:, :, oy, ox], optimize=True)
        conv += self.b.reshape(1, hidden, 1, 1)
        relu = np.maximum(conv, 0.0)
        pooled = relu.mean(axis=(2, 3))
        logits = pooled @ self.fc_w + self.fc_b
        return logits, {"x": x, "pad": pad, "conv": conv, "relu": relu, "pooled": pooled}

    def step(self, x: np.ndarray, y: np.ndarray, lr: float = 0.2) -> tuple[float, dict]:
        logits, cache = self.forward(x)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        loss = -np.log(probs[np.arange(len(y)), y] + 1e-8).mean()
        dlogits = probs
        dlogits[np.arange(len(y)), y] -= 1.0
        dlogits /= len(y)
        grad_fc_w = cache["pooled"].T @ dlogits
        grad_fc_b = dlogits.sum(axis=0)
        dpooled = dlogits @ self.fc_w.T
        drelu = dpooled[:, :, None, None] / (x.shape[2] * x.shape[3])
        dconv = drelu * (cache["conv"] > 0)
        grad_b = dconv.sum(axis=(0, 2, 3))
        grad_w = np.zeros_like(self.w)
        h, w = x.shape[2:]
        for oy in range(3):
            for ox in range(3):
                patch = cache["pad"][:, :, oy : oy + h, ox : ox + w]
                grad_w[:, :, oy, ox] = np.einsum("nkhw,nchw->kc", dconv, patch, optimize=True)
        before = [p.copy() for p in [self.w, self.b, self.fc_w, self.fc_b]]
        self.w -= lr * grad_w
        self.b -= lr * grad_b
        self.fc_w -= lr * grad_fc_w
        self.fc_b -= lr * grad_fc_b
        grad_norm = float(np.sqrt(np.sum(grad_w**2) + np.sum(grad_b**2) + np.sum(grad_fc_w**2) + np.sum(grad_fc_b**2)))
        update_norm = float(np.sqrt(sum(np.sum((a - b) ** 2) for a, b in zip([self.w, self.b, self.fc_w, self.fc_b], before))))
        return float(loss), {"grad_norm": grad_norm, "update_norm": update_norm, "finite": bool(np.isfinite(loss) and np.isfinite(grad_norm))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lamp-glob", required=True)
    parser.add_argument("--mtare-glob", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    lamp = [Path(p) for p in sorted(glob.glob(args.lamp_glob))]
    mtare = [Path(p) for p in sorted(glob.glob(args.mtare_glob))]
    if not lamp or not mtare:
        raise SystemExit("both lamp and mtare samples are required")
    first = load_map_npz(lamp[0])["tensor"]
    loader = MixedSampleLoader(lamp, mtare, args.batch_size)
    model = TinyCNN(first.shape[0])
    losses = []
    grad_norms = []
    update_norms = []
    mixed_batches = 0
    for _ in range(args.steps):
        x, y, sources = loader.batch()
        if "LAMP" in sources and "M-TARE" in sources:
            mixed_batches += 1
        loss, info = model.step(x, y)
        losses.append(loss)
        grad_norms.append(info["grad_norm"])
        update_norms.append(info["update_norm"])
    result = {
        "status": "PASS",
        "batch_shape": [args.batch_size, *first.shape],
        "input_channels": int(first.shape[0]),
        "steps": args.steps,
        "initial_loss": float(losses[0]),
        "final_loss": float(losses[-1]),
        "nan_or_inf": bool(not np.isfinite(losses).all()),
        "gradient_exists": bool(max(grad_norms) > 0),
        "parameter_updated": bool(max(update_norms) > 0),
        "mixed_batches": int(mixed_batches),
        "losses": losses,
    }
    write_json(args.output, result)


if __name__ == "__main__":
    main()
