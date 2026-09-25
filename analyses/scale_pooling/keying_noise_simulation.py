"""Keying-noise simulation: how robust is each pooling variant to wrong keying?

    python keying_noise_simulation.py

Starting from the documented keying of each validation dataset, the keying is
corrupted in three ways per draw:
  (1) a fraction u of reverse-keyed item memberships is unmarked (set positive),
      u in U_GRID;
  (2) a fraction f of positive memberships is marked reverse;
  (3) the entire keying of a fraction s of scales is inverted.
Two regimes: 'full' (f = s = .03) and 'missing_only' (f = s = 0). N_DRAWS random
corruptions per u and regime. Each draw pools scales with plain, documented (B),
aligned (C) and hyperplane_old (D) from the corrupted keying and scores the
pooled cosines against the empirical scale correlations (Pearson, MAE, sign-error
rate on |r| > .10) and on abs_cross_instrument retrieval (rho, P@5, P@10, NDCG@10,
means over query scales). For C, the fraction of corrupted memberships restored to
the true documented sign and the fraction of uncorrupted memberships wrongly
flipped are recorded as well.

Outputs: results/keying_noise/
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scale_pooling as sp  # noqa: E402
from run_pooling_comparison import DATASETS, OUT, RELEVANT_R, SIGN_MIN_R, Dataset, dcg  # noqa: E402

OUT_NOISE = os.path.join(OUT, "keying_noise")
U_GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
REGIMES = {"full": dict(f=0.03, s=0.03), "missing_only": dict(f=0.0, s=0.0)}
N_DRAWS = 50
SEED = 20260918
VARIANTS = ["plain", "documented", "aligned", "hyperplane_old"]
ACC_METRICS = ["pearson_r", "mae", "sign_error_rate"]
RET_METRICS = ["rho", "p_at_5", "p_at_10", "ndcg_at_10"]


class Scorer:
    """Precomputes everything per dataset so scoring a draw is a few numpy ops."""

    def __init__(self, ds: Dataset):
        self.ds = ds
        self.scales = ds.scale_names
        self.pos = {s: k for k, s in enumerate(self.scales)}
        pairs = ds.pairs
        self.ia = np.array([self.pos[a] for a in pairs.scale_a])
        self.ib = np.array([self.pos[b] for b in pairs.scale_b])
        self.emp = pairs.empirical_r.to_numpy()
        self.big = np.abs(self.emp) > SIGN_MIN_R
        # retrieval: cross-instrument database per query
        same = np.array([ds.scale_instrument[a] == ds.scale_instrument[b] for a, b in zip(pairs.scale_a, pairs.scale_b)])
        q = np.concatenate([self.ia, self.ib])
        d = np.concatenate([self.ib, self.ia])
        e = np.abs(np.concatenate([self.emp, self.emp]))
        keep = ~np.concatenate([same, same])
        q, d, e = q[keep], d[keep], e[keep]
        self.queries = []
        for qi in np.unique(q):
            m = q == qi
            emp_abs = e[m]
            ideal = dcg(np.sort(emp_abs)[::-1][:10])
            self.queries.append((qi, d[m], emp_abs, emp_abs >= RELEVANT_R, ideal))
        # item matrices per scale
        self.Xs = {s: np.stack([ds.emb[i] for i in ds.items[s]]) for s in self.scales}

    def pool(self, variant: str, flags: dict[str, np.ndarray]):
        fn = sp.POOLERS[variant]
        V = np.stack([fn(self.Xs[s], flags[s]) for s in self.scales])
        return sp.unit(V)

    def score(self, V: np.ndarray) -> dict[str, float]:
        C = V @ V.T
        pred = C[self.ia, self.ib]
        ok = np.isfinite(pred)
        out = dict(
            pearson_r=float(np.corrcoef(pred[ok], self.emp[ok])[0, 1]),
            mae=float(np.mean(np.abs(pred[ok] - self.emp[ok]))),
            sign_error_rate=float(np.mean(np.sign(pred[self.big & ok]) != np.sign(self.emp[self.big & ok]))),
        )
        rho, p5, p10, nd = [], [], [], []
        for qi, dbi, emp_abs, rel, ideal in self.queries:
            score = np.abs(C[qi, dbi])
            order = np.argsort(-score, kind="stable")
            rho.append(_spearman(score, emp_abs))
            p5.append(rel[order][:5].mean())
            p10.append(rel[order][:10].mean())
            nd.append(dcg(emp_abs[order][:10]) / ideal if ideal > 0 else np.nan)
        out.update(rho=float(np.nanmean(rho)), p_at_5=float(np.mean(p5)), p_at_10=float(np.mean(p10)), ndcg_at_10=float(np.nanmean(nd)))
        return out


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return np.nan
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def corrupt(true_flags: dict[str, np.ndarray], u: float, f: float, s: float, rng: np.random.Generator):
    """Return (corrupted flags, per-membership corruption type array dict)."""
    scales = list(true_flags)
    # membership index
    mem = [(sc, k) for sc in scales for k in range(len(true_flags[sc]))]
    is_rev = np.array([true_flags[sc][k] for sc, k in mem])
    rev_idx = np.where(is_rev)[0]
    pos_idx = np.where(~is_rev)[0]
    n_unmark = int(round(u * len(rev_idx)))
    n_mark = int(round(f * len(pos_idx)))
    n_invert = int(round(s * len(scales)))
    unmark = set(rng.choice(rev_idx, n_unmark, replace=False).tolist()) if n_unmark else set()
    mark = set(rng.choice(pos_idx, n_mark, replace=False).tolist()) if n_mark else set()
    inverted = set(rng.choice(len(scales), n_invert, replace=False).tolist()) if n_invert else set()
    inverted_scales = {scales[i] for i in inverted}
    flags = {sc: true_flags[sc].copy() for sc in scales}
    ctype = {sc: np.array(["none"] * len(true_flags[sc]), dtype=object) for sc in scales}
    for j, (sc, k) in enumerate(mem):
        if j in unmark:
            flags[sc][k] = False
            ctype[sc][k] = "unmarked"
        elif j in mark:
            flags[sc][k] = True
            ctype[sc][k] = "falsely_marked"
    for sc in inverted_scales:
        flags[sc] = ~flags[sc]
        ctype[sc] = np.where(ctype[sc] == "none", "inverted_scale", ctype[sc] + "+inverted_scale")
    return flags, ctype


def aligned_restoration(scorer: Scorer, true_flags, flags, ctype) -> dict[str, float]:
    """Fraction of corrupted memberships C restored to the true sign, and fraction
    of uncorrupted memberships it wrongly flipped; also broken down by type."""
    rows = []
    for sc in scorer.scales:
        _, d = sp.pool_aligned(scorer.Xs[sc], flags[sc], return_details=True)
        final_rev = d["signs"] < 0
        for k in range(len(flags[sc])):
            rows.append((ctype[sc][k], bool(final_rev[k] == true_flags[sc][k])))
    df = pd.DataFrame(rows, columns=["ctype", "correct"])
    corrupted = df[df.ctype != "none"]
    clean = df[df.ctype == "none"]
    out = dict(
        n_corrupted=len(corrupted),
        restored_frac=float(corrupted.correct.mean()) if len(corrupted) else np.nan,
        n_clean=len(clean),
        wrongly_flipped_frac=float((~clean.correct).mean()) if len(clean) else np.nan,
    )
    for t in ["unmarked", "falsely_marked", "inverted_scale"]:
        sub = df[df.ctype == t]
        out[f"restored_frac_{t}"] = float(sub.correct.mean()) if len(sub) else np.nan
        out[f"n_{t}"] = len(sub)
    return out


def summarize(draws: pd.DataFrame, metrics: list[str], keys: list[str]) -> pd.DataFrame:
    rows = []
    for k, g in draws.groupby(keys, sort=False):
        row = dict(zip(keys, k))
        row["n_draws"] = len(g)
        for m in metrics:
            x = g[m].to_numpy(dtype=float)
            x = x[~np.isnan(x)]
            if len(x) == 0:  # e.g. no falsely-marked items in the missing_only regime
                row[f"{m}_mean"] = row[f"{m}_p2.5"] = row[f"{m}_p97.5"] = np.nan
                continue
            row[f"{m}_mean"] = float(np.mean(x))
            row[f"{m}_p2.5"] = float(np.percentile(x, 2.5))
            row[f"{m}_p97.5"] = float(np.percentile(x, 97.5))
        rows.append(row)
    return pd.DataFrame(rows)


def plot(summary: pd.DataFrame, dataset: str, path: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    colors = {"plain": "#888888", "documented": "#1b6ca8", "aligned": "#2a9d5c", "hyperplane_old": "#d1495b"}
    for ax, (regime, spec) in zip(axes, REGIMES.items()):
        sub = summary[(summary.dataset == dataset) & (summary.regime == regime)]
        for v in VARIANTS:
            g = sub[sub.variant == v].sort_values("u")
            ax.plot(g.u, g["pearson_r_mean"], marker="o", color=colors[v], label=v)
            ax.fill_between(g.u, g["pearson_r_p2.5"], g["pearson_r_p97.5"], color=colors[v], alpha=0.18, linewidth=0)
        ax.set_title(f"{regime}: f = {spec['f']}, s = {spec['s']}")
        ax.set_xlabel("u = fraction of reverse-keyed memberships unmarked")
        ax.set_xticks(U_GRID)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Pearson r with empirical scale r")
    axes[0].legend(frameon=False)
    fig.suptitle(f"Keying noise: {dataset} (mean and 2.5/97.5 percentiles over {N_DRAWS} draws)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUT_NOISE, exist_ok=True)
    rng = np.random.default_rng(SEED)
    draw_rows, rest_rows = [], []
    for key, tag in DATASETS.items():
        print("==", key)
        ds = Dataset(key, tag)
        sc = Scorer(ds)
        true_flags = {s: ds.rev_flags[s].copy() for s in sc.scales}
        for regime, spec in REGIMES.items():
            for u in U_GRID:
                for draw in range(N_DRAWS):
                    flags, ctype = corrupt(true_flags, u, spec["f"], spec["s"], rng)
                    base = dict(dataset=key, regime=regime, u=u, f=spec["f"], s=spec["s"], draw=draw)
                    for v in VARIANTS:
                        draw_rows.append({**base, "variant": v, **sc.score(sc.pool(v, flags))})
                    rest_rows.append({**base, **aligned_restoration(sc, true_flags, flags, ctype)})
                print(f"   {regime} u={u} done")
    draws = pd.DataFrame(draw_rows)
    rest = pd.DataFrame(rest_rows)
    draws.to_csv(os.path.join(OUT_NOISE, "keying_noise_draws.csv"), index=False)
    rest.to_csv(os.path.join(OUT_NOISE, "aligned_restoration_draws.csv"), index=False)
    summary = summarize(draws, ACC_METRICS + RET_METRICS, ["dataset", "regime", "u", "f", "s", "variant"])
    summary.to_csv(os.path.join(OUT_NOISE, "keying_noise_summary.csv"), index=False)
    rest_cols = ["restored_frac", "wrongly_flipped_frac", "restored_frac_unmarked", "restored_frac_falsely_marked", "restored_frac_inverted_scale"]
    rest_summary = summarize(rest, rest_cols, ["dataset", "regime", "u", "f", "s"])
    rest_summary.to_csv(os.path.join(OUT_NOISE, "aligned_restoration_summary.csv"), index=False)
    for key in DATASETS:
        plot(summary, key, os.path.join(OUT_NOISE, f"keying_noise_{key}.png"))

    # markdown
    md = ["# Keying-noise simulation", "", "See `keying_noise_simulation.py` docstring for the design. "
          f"{N_DRAWS} draws per u and regime. Values are mean [2.5th, 97.5th percentile] over draws.", ""]
    md.append("Note: the pilot's 'documented' keying is itself derived from the empirical data (see summary.md), "
              "so for the pilot the true keying is by construction consistent with the empirical correlations.")
    md.append("")
    for key in DATASETS:
        md.append(f"## {key}")
        md.append("")
        md.append(f"![]({os.path.basename(os.path.join(OUT_NOISE, f'keying_noise_{key}.png'))})")
        md.append("")
        for regime, spec in REGIMES.items():
            md.append(f"### regime `{regime}` (f = {spec['f']}, s = {spec['s']})")
            md.append("")
            sub = summary[(summary.dataset == key) & (summary.regime == regime)]
            hdr = ["u", "variant"] + ACC_METRICS + RET_METRICS
            lines = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
            for _, r in sub.sort_values(["u", "variant"], key=lambda c: c.map({v: i for i, v in enumerate(VARIANTS)}) if c.name == "variant" else c).iterrows():
                cells = [f"{r.u:.2f}", r.variant] + [
                    f"{r[m + '_mean']:.3f} [{r[m + '_p2.5']:.3f}, {r[m + '_p97.5']:.3f}]" for m in ACC_METRICS + RET_METRICS
                ]
                lines.append("| " + " | ".join(cells) + " |")
            md.extend(lines)
            md.append("")
            md.append("Variant C (aligned) restoration:")
            md.append("")
            rs = rest_summary[(rest_summary.dataset == key) & (rest_summary.regime == regime)].sort_values("u")
            hdr = ["u", "restored (all corrupted)", "restored (unmarked)", "restored (falsely marked)", "restored (inverted scale)", "wrongly flipped (clean)"]
            lines = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
            for _, r in rs.iterrows():
                def fmt(m):
                    return "n/a" if np.isnan(r[m + "_mean"]) else f"{r[m + '_mean']:.3f} [{r[m + '_p2.5']:.3f}, {r[m + '_p97.5']:.3f}]"
                lines.append("| " + " | ".join([f"{r.u:.2f}", fmt("restored_frac"), fmt("restored_frac_unmarked"), fmt("restored_frac_falsely_marked"), fmt("restored_frac_inverted_scale"), fmt("wrongly_flipped_frac")]) + " |")
            md.extend(lines)
            md.append("")
    with open(os.path.join(OUT_NOISE, "keying_noise_summary.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("wrote", OUT_NOISE)


if __name__ == "__main__":
    main()
