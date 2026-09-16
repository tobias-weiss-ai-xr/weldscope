"""Fit multinomial logistic regression on simulated weld features.

The ground truth is the sim recipe (config/sim.json): the defect active at the
mid-time of each feature window. Features are reproduced in numpy with the
same formulas as crates/features, driven by the same KeyholeModel depth
dynamics as crates/sim.
"""
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WIN = 80
RATE = 80_000.0
N_FRAMES = 20_000  # 250 ms weld
SPEC_BINS = 2048


def hash01(k):
    """Deterministic [0,1) per-frame LCG; bit-exact mirror of KeyholeModel::hash01."""
    s = (k * 2654435761 + 40503) & ((1 << 64) - 1)
    return ((s >> 16) & 0x7FFF) / 32767.0


def depth_series(cfg):
    zs = []
    for k in range(N_FRAMES):
        t = k / RATE
        z = cfg["depth0_bins"] + cfg["osc_amp_bins"] * math.sin(
            2 * math.pi * cfg["osc_hz"] * t
        )
        for d in cfg["defects"]:
            t0, t1, kind = d  # sim.json: [start, end, kind]
            if not (t0 <= t <= t1):
                continue
            p = min(1.0, max(0.0, (t - t0) / (t1 - t0)))
            if kind == "spatter":
                z += (hash01(k) * 2.0 - 1.0) * 30.0  # steady ±30 jitter
            elif kind == "pore":
                z *= 1.0 - 0.35 * (0.5 + 0.5 * math.sin(p * 12.0))
            elif kind == "incomplete":
                z *= (0.80 - 0.45 * p)  # onset dip 0.80x, floor 0.35z
            elif kind == "humping":
                z += 20.0 * math.sin(p * 24.0)  # fast ±20 oscillation, no dropout
        zs.append(min(max(z, 1.0), SPEC_BINS * 0.45))
    return np.array(zs, dtype=np.float32)


def label_at(defects, t):
    for d in defects:
        t0, t1, kind = d
        if t0 <= t <= t1:
            return kind
    return "ok"


def window_features(cfg, zs):
    feats, labels = [], []
    for i in range(WIN, len(zs)):
        w = zs[i - WIN : i]
        x = np.zeros(8, dtype=np.float32)
        x[0] = w.mean()
        x[1] = w.std()
        x[2] = w.min()
        x[3] = w.max()
        x[4] = (w > 250.0).mean()
        x[5] = (np.abs(np.diff(w)) > 15.0).mean()  # mirrors crates/features spatter_rate (spike_delta=15)
        x[6] = np.sum((w[:-1] > 250.0) & (w[1:] <= 250.0)).astype(np.float32)
        x[7] = w.reshape(-1, 8).mean(axis=1).std()
        t = (i - WIN / 2) / RATE
        feats.append(x)
        labels.append(label_at(cfg["defects"], t))
    return np.vstack(feats), np.array(labels)


if __name__ == "__main__":
    cfg = json.loads((ROOT / "config" / "sim.json").read_text())
    zs = depth_series(cfg)
    X, y = window_features(cfg, zs)
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(C=3.0, max_iter=2000, random_state=cfg["seed"])
    clf.fit(X, y)
    print("classes:", list(clf.classes_))
    print("train acc:", round(clf.score(X, y), 3))
    model = {
        "classes": list(clf.classes_),
        "weights": clf.coef_.tolist(),
        "bias": clf.intercept_.tolist(),
    }
    out = ROOT / "ai" / "weights.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(model, indent=1))
    print("wrote", out)
