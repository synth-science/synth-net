"""Minimal, reproducible demo: how documented-sign pooling (variant B) fares when
the keying of the Prolific validation study is perturbed.

    python documented_keying_noise_validation.py

Perturbation per draw (seeded): a fraction u of reverse-keyed item memberships is
unmarked; optionally 3% of positive memberships are marked reverse and 3% of
scales are inverted ('full' regime) or not ('missing_only'). 50 draws per u.
Scores: Pearson r, MAE and sign-error rate (|empirical r| > .10) of the cosine
between pooled scale vectors against the empirical scale correlations.

Writes results/keying_noise/documented_only_validation.{csv,png}
and prints the table.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scale_pooling as sp  # noqa: E402
from keying_noise_simulation import OUT_NOISE, REGIMES, U_GRID, corrupt  # noqa: E402
from run_pooling_comparison import DATASETS, SIGN_MIN_R, Dataset  # noqa: E402

N_DRAWS = 50
SEED = 20260918


def score_documented(ds: Dataset, flags: dict[str, np.ndarray]) -> dict[str, float]:
    """Pool every scale with pool_documented under `flags`, return accuracy metrics."""
    vec = {s: sp.pool_documented(np.stack([ds.emb[i] for i in ds.items[s]]), flags[s]) for s in ds.scale_names}
    pred = np.array([sp.cosine(vec[a], vec[b]) for a, b in zip(ds.pairs.scale_a, ds.pairs.scale_b)])
    emp = ds.pairs.empirical_r.to_numpy()
    big = np.abs(emp) > SIGN_MIN_R
    return dict(
        pearson_r=float(np.corrcoef(pred, emp)[0, 1]),
        mae=float(np.mean(np.abs(pred - emp))),
        sign_error_rate=float(np.mean(np.sign(pred[big]) != np.sign(emp[big]))),
    )


def main():
    os.makedirs(OUT_NOISE, exist_ok=True)
    rng = np.random.default_rng(SEED)
    ds = Dataset("validation_prolific", DATASETS["validation_prolific"])
    true_flags = {s: ds.rev_flags[s].copy() for s in ds.scale_names}

    rows = []
    for regime, spec in REGIMES.items():
        for u in U_GRID:
            for draw in range(N_DRAWS):
                flags, _ = corrupt(true_flags, u, spec["f"], spec["s"], rng)
                rows.append(dict(regime=regime, u=u, draw=draw, **score_documented(ds, flags)))
    draws = pd.DataFrame(rows)

    def q(x, p):
        return np.percentile(x, p)

    summary = (
        draws.groupby(["regime", "u"])
        .agg(
            pearson_r=("pearson_r", "mean"),
            pearson_r_lo=("pearson_r", lambda x: q(x, 2.5)),
            pearson_r_hi=("pearson_r", lambda x: q(x, 97.5)),
            mae=("mae", "mean"),
            mae_lo=("mae", lambda x: q(x, 2.5)),
            mae_hi=("mae", lambda x: q(x, 97.5)),
            sign_error_rate=("sign_error_rate", "mean"),
            sign_error_lo=("sign_error_rate", lambda x: q(x, 2.5)),
            sign_error_hi=("sign_error_rate", lambda x: q(x, 97.5)),
        )
        .reset_index()
    )
    summary.round(2).to_csv(os.path.join(OUT_NOISE, "documented_only_validation.csv"), index=False)

    pd.set_option("display.width", 200)
    print(f"Variant B (documented) on the validation study, {N_DRAWS} draws per cell, mean [2.5%, 97.5%]:\n")
    for regime, spec in REGIMES.items():
        print(f"regime {regime} (f = {spec['f']}, s = {spec['s']})")
        g = summary[summary.regime == regime]
        for _, r in g.iterrows():
            print(
                f"  u = {r.u:.2f}   r = {r.pearson_r:.2f} [{r.pearson_r_lo:.2f}, {r.pearson_r_hi:.2f}]   "
                f"MAE = {r.mae:.2f} [{r.mae_lo:.2f}, {r.mae_hi:.2f}]   "
                f"sign err = {r.sign_error_rate:.2f} [{r.sign_error_lo:.2f}, {r.sign_error_hi:.2f}]"
            )
        print()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for regime, color in [("missing_only", "#1b6ca8"), ("full", "#d1495b")]:
        g = summary[summary.regime == regime].sort_values("u")
        ax.plot(g.u, g.pearson_r, marker="o", color=color, label=f"{regime} (f = {REGIMES[regime]['f']}, s = {REGIMES[regime]['s']})")
        ax.fill_between(g.u, g.pearson_r_lo, g.pearson_r_hi, color=color, alpha=0.2, linewidth=0)
    ax.set_xlabel("u = fraction of reverse-keyed memberships unmarked")
    ax.set_ylabel("Pearson r with empirical scale r")
    ax.set_title("Documented-sign pooling (B), validation study")
    ax.set_xticks(U_GRID)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_NOISE, "documented_only_validation.png"), dpi=150)
    print("wrote", os.path.join(OUT_NOISE, "documented_only_validation.{csv,png}"))


if __name__ == "__main__":
    main()
