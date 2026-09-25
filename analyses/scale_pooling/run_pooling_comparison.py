"""Compare ways of pooling SurveyBot3000 item embeddings into scale vectors.

Re-runnable from scratch:

    python run_pooling_comparison.py

Inputs (in data/, copied from github.com/rubenarslan/surveybot3000 data/intermediate;
the validation-study embeddings come from github.com/synth-science/surveybot3000):

    data/ItemSimilarityTraining-20240502-trial12.raw.<dataset>.{mapping2,scales,human,machine,scale_correlations}.feather
    data/embeddings_surveybot3000.feather

Outputs go to results/.

What is compared (see scale_pooling.py for A-E):
    A plain, B documented, C aligned (+ C_loo, leave-one-out variant), D hyperplane_old,
    E positive_only  -> cosine between pooled scale vectors
    F_composite      -> composite correlation from the predicted item correlation
                        (cosine) matrix with documented keying
    F_paper          -> the paper's own scale-level prediction, i.e. the stored
                        synthetic_r (Pearson across embedding dimensions between
                        keyed mean embedding vectors); reproduced here as a check
"""

from __future__ import annotations

import io
import os
import sys
import urllib.request
from itertools import combinations

import numpy as np
import pandas as pd
import pyarrow.feather as feather
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scale_pooling as sp  # noqa: E402

MODEL = "ItemSimilarityTraining-20240502-trial12"
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "results")
VAL_EMB_URL = (
    "https://github.com/synth-science/surveybot3000/raw/refs/heads/main/"
    "validation_study/embeddings_surveybot3000.feather"
)
VAL_EMB_CACHE = os.path.join(DATA, "embeddings_surveybot3000.feather")

DATASETS = {
    "pilot_holdout": "osf-bainbridge-2021-s2-0",
    "validation_prolific": "validation-study-2024-11-01",
}
VARIANTS = ["plain", "documented", "aligned", "aligned_loo", "hyperplane_old", "positive_only"]
ALL_PREDICTORS = VARIANTS + ["F_composite", "F_paper"]
RELEVANT_R = 0.40
SIGN_MIN_R = 0.10
N_BOOT = 2000
SEED = 20260918


# --------------------------------------------------------------------------- IO
def read_feather(name: str) -> pd.DataFrame:
    return feather.read_table(os.path.join(DATA, f"{MODEL}.raw.{name}.feather")).to_pandas()


def load_validation_embeddings() -> pd.DataFrame:
    """Return DataFrame (dims x items) like the pilot 'machine' feather."""
    os.makedirs(os.path.dirname(VAL_EMB_CACHE), exist_ok=True)
    if not os.path.exists(VAL_EMB_CACHE):
        local = os.path.expanduser(
            "~/research/surveybot3000-1/validation_study/embeddings_surveybot3000.feather"
        )
        if os.path.exists(local):
            data = open(local, "rb").read()
        else:
            data = urllib.request.urlopen(VAL_EMB_URL, timeout=120).read()
        with open(VAL_EMB_CACHE, "wb") as fh:
            fh.write(data)
    t = feather.read_table(VAL_EMB_CACHE).to_pandas()
    E = np.stack(t["embeddings"].values).astype(float)  # items x dims
    return pd.DataFrame(E.T, columns=t["id"].tolist())


# ------------------------------------------------------------ paper's procedures
def find_reverse_items_by_first_item(rs: np.ndarray, items: list[str], first_item_key: float):
    """Port of global_functions.R::find_reverse_items_by_first_item.
    rs: empirical item correlation matrix restricted to `items` (same order)."""
    col = rs[1:, 0]
    if first_item_key == 1:
        reversed_idx = [i + 1 for i in np.where(col < 0)[0]]
    else:
        reversed_idx = [0] + [i + 1 for i in np.where(col > 0)[0]]
    return [items[i] for i in reversed_idx]


def keyed_row_means(df: pd.DataFrame, items: list[str], reverse_items: list[str]) -> np.ndarray:
    """Port of calculate_row_means: reverse items via max(all selected)+1-x, then
    rowMeans(na.rm=TRUE)."""
    sub = df[items].astype(float).copy()
    if reverse_items:
        mx = np.nanmax(sub.to_numpy())
        for it in reverse_items:
            sub[it] = mx + 1 - sub[it]
    return sub.mean(axis=1, skipna=True).to_numpy()


def pearson_pairwise(a: np.ndarray, b: np.ndarray) -> float:
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 3:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def build_scale_items(mapping: pd.DataFrame, scales: pd.DataFrame) -> dict[str, list[str]]:
    """Port of items_by_scale: parent scales (scale_1 == '') take every item of
    the construct; facet scales match on scale_1 too. Item order = mapping order."""
    m = mapping.rename(columns={"scale0": "scale_0", "scale1": "scale_1"}).copy()
    m["scale_1"] = m["scale_1"].fillna("__NA__")
    out = {}
    for _, s in scales.iterrows():
        if s["scale_1"] == "":
            sel = m[(m.instrument == s.instrument) & (m.scale_0 == s.scale_0)]
        else:
            sel = m[(m.instrument == s.instrument) & (m.scale_0 == s.scale_0) & (m.scale_1 == s.scale_1)]
        out[s["scale"]] = sel["variable"].tolist()
    return out


# ------------------------------------------------------------------ datasets
class Dataset:
    def __init__(self, key: str, tag: str):
        self.key, self.tag = key, tag
        self.mapping = read_feather(f"{tag}.mapping2")
        self.scales = read_feather(f"{tag}.scales")
        self.human = read_feather(f"{tag}.human")
        self.pairs = read_feather(f"{tag}.scale_correlations")
        if key == "pilot_holdout":
            self.machine = read_feather(f"{tag}.machine")
        else:
            self.machine = load_validation_embeddings()
        self.machine = self.machine.astype(float)
        self.item_text = dict(zip(self.mapping.variable, self.mapping.item_text))
        self.emb = {v: self.machine[v].to_numpy() for v in self.machine.columns}

        self.items = build_scale_items(self.mapping, self.scales)
        used = set(self.pairs.scale_a) | set(self.pairs.scale_b)
        self.items = {k: v for k, v in self.items.items() if k in used and len(v) > 0}
        self.scale_names = sorted(self.items)

        # reverse flags per scale
        self.reverse: dict[str, list[str]] = {}
        self.keying_source = None
        if "keyed" in self.mapping.columns:
            self.keying_source = "documented per-item keying (mapping2.keyed)"
            keyed = dict(zip(self.mapping.variable, self.mapping.keyed))
            for s, its in self.items.items():
                self.reverse[s] = [i for i in its if keyed[i] == -1]
        else:
            self.keying_source = (
                "derived: hand-coded key of the first item per scale (scales.keyed) + sign of "
                "the empirical correlation of every other item with that first item "
                "(global_functions.R::find_reverse_items_by_first_item)"
            )
            hc = self.human.astype(float).corr(method="pearson")  # pairwise complete
            first_key = dict(zip(self.scales.scale, self.scales.keyed))
            for s, its in self.items.items():
                rs = hc.loc[its, its].to_numpy()
                self.reverse[s] = find_reverse_items_by_first_item(rs, its, first_key[s])
        self.rev_flags = {
            s: np.array([i in set(self.reverse[s]) for i in its], bool) for s, its in self.items.items()
        }
        # unit item matrix and cosine matrix
        self.item_ids = self.mapping.variable.tolist()
        X = np.stack([self.emb[i] for i in self.item_ids])
        self.norms = np.linalg.norm(X, axis=1)
        self.X = sp.unit(X)
        self.idx = {i: k for k, i in enumerate(self.item_ids)}
        self.C = self.X @ self.X.T  # predicted item correlation (cosine) matrix
        self.H = self.human.astype(float).corr(method="pearson").loc[self.item_ids, self.item_ids].to_numpy()
        self.instrument = dict(zip(self.mapping.variable, self.mapping.instrument))
        self.scale_instrument = {s: self.instrument[its[0]] for s, its in self.items.items()}

    def item_evidence(self, scale: str, item: str) -> tuple[float, float]:
        """Mean empirical r and mean cosine of the RAW item with the other items of
        the scale, the others signed by their documented keys. Positive means the
        raw item runs with the construct; negative means it runs against it."""
        others = [i for i in self.items[scale] if i != item]
        if not others:
            return np.nan, np.nan
        signs = np.array([-1.0 if o in set(self.reverse[scale]) else 1.0 for o in others])
        io = [self.idx[o] for o in others]
        ii = self.idx[item]
        return float(np.nanmean(self.H[ii, io] * signs)), float(np.mean(self.C[ii, io] * signs))

    # -- reproduce the paper's empirical and synthetic scale correlations
    def reproduce_paper(self) -> pd.DataFrame:
        rows = []
        for _, p in self.pairs.iterrows():
            a, b = p.scale_a, p.scale_b
            ha = keyed_row_means(self.human, self.items[a], self.reverse[a])
            hb = keyed_row_means(self.human, self.items[b], self.reverse[b])
            ma = keyed_row_means(self.machine, self.items[a], self.reverse[a])
            mb = keyed_row_means(self.machine, self.items[b], self.reverse[b])
            rows.append(
                dict(
                    scale_a=a,
                    scale_b=b,
                    empirical_r_stored=p.empirical_r,
                    empirical_r_repro=pearson_pairwise(ha, hb),
                    F_paper_stored=p.synthetic_r,
                    F_paper_repro=pearson_pairwise(ma, mb),
                )
            )
        return pd.DataFrame(rows)

    # -- pooled vectors and diagnostics
    def pool_all(self):
        vecs = {v: {} for v in VARIANTS}
        aligned_rows, hyper_rows = [], []
        de_maxdiff = 0.0
        for s in self.scale_names:
            its = self.items[s]
            Xs = np.stack([self.emb[i] for i in its])
            rev = self.rev_flags[s]
            vecs["plain"][s] = sp.pool_plain(Xs, rev)
            vecs["documented"][s] = sp.pool_documented(Xs, rev)
            vecs["positive_only"][s] = sp.pool_positive_only(Xs, rev)
            v, d = sp.pool_aligned(Xs, rev, return_details=True)
            vecs["aligned"][s] = v
            vecs["aligned_loo"][s] = sp.pool_aligned(Xs, rev, leave_one_out=True)
            vh, dh = sp.pool_hyperplane_old(Xs, rev, return_details=True)
            vecs["hyperplane_old"][s] = vh
            for k, it in enumerate(its):
                emp_ev, cos_ev = self.item_evidence(s, it)
                aligned_rows.append(
                    dict(
                        dataset=self.key,
                        scale=s,
                        item=it,
                        item_text=self.item_text[it],
                        n_items=len(its),
                        documented_reverse=bool(rev[k]),
                        final_reverse=bool(d["signs"][k] < 0),
                        disagrees=bool(d["disagrees"][k]),
                        cos_to_centroid=float(d["cos_to_centroid"][k]),
                        mean_emp_r_raw_vs_scale=emp_ev,
                        mean_cos_raw_vs_scale=cos_ev,
                        hyperplane_reflected=bool(dh["reflected"][k]),
                        hyperplane_cos_to_pos_c=float(dh["cos_to_pos_c"][k]),
                        n_iter=d["n_iter"],
                        converged=d["converged"],
                    )
                )
            v, d = vh, dh
            n_rev = int(rev.sum())
            hyper_rows.append(
                dict(
                    dataset=self.key,
                    scale=s,
                    n_items=len(its),
                    n_reverse=n_rev,
                    all_reverse=bool(n_rev == len(its)),
                    degenerate=bool(d["degenerate"]),
                    n_reflected=int(d["reflected"].sum()),
                    n_reverse_not_reflected=int(d["not_reflected"].sum()),
                    unreflected_items="; ".join(it for it, f in zip(its, d["not_reflected"]) if f),
                    unreflected_cos_to_pos_c="; ".join(
                        f"{c:.3f}" for c, f in zip(d["cos_to_pos_c"], d["not_reflected"]) if f
                    ),
                )
            )
            # sanity: D == E whenever every reverse item was reflected (and not degenerate)
            if not d["degenerate"] and d["not_reflected"].sum() == 0:
                de_maxdiff = max(de_maxdiff, float(np.abs(v - vecs["positive_only"][s]).max()))
        return vecs, pd.DataFrame(aligned_rows), pd.DataFrame(hyper_rows), de_maxdiff

    def composite_r(self, a: str, b: str) -> float:
        """Composite correlation from the predicted item correlation matrix."""
        ia = [self.idx[i] for i in self.items[a]]
        ib = [self.idx[i] for i in self.items[b]]
        ka = np.where(self.rev_flags[a], -1.0, 1.0)
        kb = np.where(self.rev_flags[b], -1.0, 1.0)
        num = ka @ self.C[np.ix_(ia, ib)] @ kb
        va = ka @ self.C[np.ix_(ia, ia)] @ ka
        vb = kb @ self.C[np.ix_(ib, ib)] @ kb
        if va <= 0 or vb <= 0:
            return np.nan
        return float(num / np.sqrt(va * vb))

    def pair_predictions(self, vecs) -> pd.DataFrame:
        rows = []
        for _, p in self.pairs.iterrows():
            a, b = p.scale_a, p.scale_b
            r = dict(
                dataset=self.key,
                scale_a=a,
                scale_b=b,
                same_instrument=self.scale_instrument[a] == self.scale_instrument[b],
                empirical_r=p.empirical_r,
                pairwise_n=p.pairwise_n,
                F_paper=p.synthetic_r,
                F_composite=self.composite_r(a, b),
            )
            for v in VARIANTS:
                r[v] = sp.cosine(vecs[v][a], vecs[v][b])
            rows.append(r)
        return pd.DataFrame(rows)


# ------------------------------------------------------------------ metrics
def accuracy_metrics(pp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    y = pp.empirical_r.to_numpy()
    for v in ALL_PREDICTORS:
        x = pp[v].to_numpy()
        ok = ~np.isnan(x)
        big = ok & (np.abs(y) > SIGN_MIN_R)
        rows.append(
            dict(
                dataset=pp.dataset.iloc[0],
                variant=v,
                n_pairs=int(ok.sum()),
                pearson_r=stats.pearsonr(x[ok], y[ok])[0],
                spearman_rho=stats.spearmanr(x[ok], y[ok])[0],
                mae=float(np.mean(np.abs(x[ok] - y[ok]))),
                rmse=float(np.sqrt(np.mean((x[ok] - y[ok]) ** 2))),
                sign_error_rate=float(np.mean(np.sign(x[big]) != np.sign(y[big]))),
                n_pairs_sign=int(big.sum()),
            )
        )
    return pd.DataFrame(rows)


def dcg(gains: np.ndarray) -> float:
    return float(np.sum(gains / np.log2(np.arange(2, len(gains) + 2))))


def retrieval_per_query(pp: pd.DataFrame, predictor: str, mode: str, cross_instrument: bool = False) -> pd.DataFrame:
    """One row per query scale: rho of the two rankings, P@5, P@10, NDCG@10.
    cross_instrument=True restricts the database to scales from other instruments."""
    cols = ["query", "db", "empirical_r", predictor, "same_instrument"]
    long = pd.concat(
        [
            pp.rename(columns={"scale_a": "query", "scale_b": "db"})[cols],
            pp.rename(columns={"scale_b": "query", "scale_a": "db"})[cols],
        ]
    ).dropna(subset=[predictor])
    if cross_instrument:
        long = long[~long.same_instrument]
    rows = []
    for q, g in long.groupby("query"):
        emp = np.abs(g.empirical_r.to_numpy())
        pred = g[predictor].to_numpy()
        score = np.abs(pred) if mode == "abs" else pred
        order = np.argsort(-score, kind="stable")
        emp_sorted = emp[order]
        rel = emp >= RELEVANT_R
        ideal = np.sort(emp)[::-1][:10]
        idcg = dcg(ideal)
        rows.append(
            dict(
                query=q,
                n_db=len(g),
                n_relevant=int(rel.sum()),
                rho=stats.spearmanr(score, emp)[0] if len(g) > 2 else np.nan,
                p_at_5=float(rel[order][:5].mean()),
                p_at_10=float(rel[order][:10].mean()),
                ndcg_at_10=dcg(emp_sorted[:10]) / idcg if idcg > 0 else np.nan,
            )
        )
    return pd.DataFrame(rows)


def bootstrap_mean(x: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    means = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def retrieval_metrics(pp: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary, per_query = [], []
    for v in ALL_PREDICTORS:
        for mode, cross in [("signed", False), ("abs", False), ("abs_cross_instrument", True)]:
            pq = retrieval_per_query(pp, v, mode.split("_")[0], cross_instrument=cross)
            pq.insert(0, "mode", mode)
            pq.insert(0, "variant", v)
            pq.insert(0, "dataset", pp.dataset.iloc[0])
            per_query.append(pq)
            row = dict(
                dataset=pp.dataset.iloc[0],
                variant=v,
                mode=mode,
                n_queries=len(pq),
                n_queries_no_relevant=int((pq.n_relevant == 0).sum()),
            )
            for m in ["rho", "p_at_5", "p_at_10", "ndcg_at_10"]:
                mean, lo, hi = bootstrap_mean(pq[m].to_numpy(dtype=float), rng)
                row[m] = mean
                row[f"{m}_lo"] = lo
                row[f"{m}_hi"] = hi
            summary.append(row)
    return pd.DataFrame(summary), pd.concat(per_query, ignore_index=True)


# ------------------------------------------------------------------ report
def md_table(df: pd.DataFrame, cols: list[str], fmt: dict[str, str] | None = None) -> str:
    fmt = fmt or {}
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            val = r[c]
            if isinstance(val, float):
                cells.append(fmt.get(c, "{:.3f}").format(val))
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    md = ["# Pooling comparison: reverse-keyed items in scale vectors", ""]
    md.append(
        "Generated by `run_pooling_comparison.py`. Pooling functions live in "
        "`scale_pooling.py` (numpy only)."
    )
    md.append("")
    md.append(
        "Variants: A `plain`, B `documented`, C `aligned` (plus `aligned_loo`, the leave-one-out "
        "version of C, an extra), D `hyperplane_old`, E `positive_only`, `F_composite` (composite "
        "correlation from the cosine item matrix with documented keying), `F_paper` (the paper's "
        "stored scale-level synthetic r). A-E are cosines between pooled vectors."
    )
    md.append("")
    md.append("Files in this folder:")
    md.append("")
    md.append("- `accuracy_metrics.csv`: accuracy table per dataset and variant")
    md.append("- `retrieval_metrics.csv`, `retrieval_per_query.csv`: retrieval means with bootstrap CIs, and per-query values")
    md.append("- `scale_pair_predictions.csv`: every scale pair with the empirical r and all predictions")
    md.append("- `aligned_items.csv`: every item-scale membership with keying, C and D diagnostics and empirical evidence; `aligned_sign_flips.csv` is the subset C flipped")
    md.append("- `hyperplane_scales.csv`: per-scale D diagnostics")
    md.append("- `paper_reproduction.csv`: stored vs re-computed empirical and synthetic scale r")
    md.append("- `dataset_info.csv`: sizes, embedding norms, keying source")
    md.append("")

    all_pp, all_acc, all_ret, all_pq, all_aligned, all_hyper, repro_rows, ds_info = [], [], [], [], [], [], [], []
    for key, tag in DATASETS.items():
        print(f"== {key}")
        ds = Dataset(key, tag)
        n_scales = len(ds.scale_names)
        n_items = len(ds.item_ids)
        n_rev = sum(int(f.sum()) for f in ds.rev_flags.values())
        n_scale_items = sum(len(v) for v in ds.items.values())
        ds_info.append(
            dict(
                dataset=key,
                n_respondents=len(ds.human),
                n_items=n_items,
                n_scales=n_scales,
                n_scale_pairs=len(ds.pairs),
                n_scale_item_memberships=n_scale_items,
                n_reverse_memberships=n_rev,
                embedding_norm_min=float(ds.norms.min()),
                embedding_norm_max=float(ds.norms.max()),
                keying_source=ds.keying_source,
            )
        )

        rep = ds.reproduce_paper()
        rep.insert(0, "dataset", key)
        repro_rows.append(rep)
        r_emp = stats.pearsonr(rep.empirical_r_stored, rep.empirical_r_repro)[0]
        d_emp = float(np.abs(rep.empirical_r_stored - rep.empirical_r_repro).max())
        r_syn = stats.pearsonr(rep.F_paper_stored, rep.F_paper_repro)[0]
        d_syn = float(np.abs(rep.F_paper_stored - rep.F_paper_repro).max())
        print(f"   reproduction: empirical r={r_emp:.5f} maxdiff={d_emp:.2e}; F_paper r={r_syn:.5f} maxdiff={d_syn:.2e}")

        vecs, aligned, hyper, de_maxdiff = ds.pool_all()
        pp = ds.pair_predictions(vecs)
        all_pp.append(pp)
        acc = accuracy_metrics(pp)
        all_acc.append(acc)
        ret, pq = retrieval_metrics(pp, rng)
        all_ret.append(ret)
        all_pq.append(pq)
        all_aligned.append(aligned)
        all_hyper.append(hyper)

        # B vs F_paper agreement, D vs E
        r_b_fp = stats.pearsonr(pp.documented, pp.F_paper)[0]
        d_b_fp = float(np.abs(pp.documented - pp.F_paper).max())
        de_all = float(np.abs(pp.hyperplane_old - pp.positive_only).max())
        n_all_rev = int(hyper.all_reverse.sum())
        n_no_rev = int((hyper.n_reverse == 0).sum())
        n_unref = int(((hyper.n_reverse_not_reflected > 0) & ~hyper.all_reverse).sum())
        n_with_rev = int((hyper.n_reverse > 0).sum())
        n_mixed = n_with_rev - n_all_rev
        flips = aligned[aligned.disagrees]

        md.append(f"## Dataset: {key}")
        md.append("")
        md.append(
            f"{len(ds.human)} respondents, {n_items} items, {n_scales} scales, {len(ds.pairs)} scale pairs "
            f"(parent-facet pairs excluded, as in the paper). Item-scale memberships: {n_scale_items}, "
            f"of which {n_rev} reverse-keyed."
        )
        md.append("")
        md.append(f"Keying source: {ds.keying_source}.")
        md.append("")
        md.append(
            f"Checks: stored embeddings have norms in [{ds.norms.min():.6f}, {ds.norms.max():.6f}]. "
            f"Reproduction of the paper's empirical scale r: Pearson {r_emp:.5f}, max abs diff {d_emp:.1e}. "
            f"Reproduction of the paper's synthetic scale r (F_paper): Pearson {r_syn:.5f}, max abs diff {d_syn:.1e}. "
            f"B (documented cosine) vs F_paper: Pearson {r_b_fp:.5f}, max abs diff {d_b_fp:.1e}. "
            f"D vs E on scales where every reverse item was reflected: max abs vector diff {de_maxdiff:.1e}; "
            f"max abs cosine diff over all pairs {de_all:.3f}."
        )
        md.append("")
        if n_scales < 20:
            md.append(f"**Only {n_scales} scales: treat retrieval numbers as descriptive.**")
            md.append("")
        md.append("### Accuracy against empirical scale correlations")
        md.append("")
        md.append(
            md_table(
                acc,
                ["variant", "n_pairs", "pearson_r", "spearman_rho", "mae", "rmse", "sign_error_rate", "n_pairs_sign"],
            )
        )
        md.append("")
        md.append(
            f"Sign-error rate is restricted to pairs with |empirical r| > {SIGN_MIN_R} (n_pairs_sign)."
        )
        md.append("")
        md.append("### Retrieval (each scale as query, other scales as database)")
        md.append("")
        md.append(
            f"Relevant = |empirical r| >= {RELEVANT_R}. NDCG@10 gain = |empirical r|. rho = Spearman between the "
            "predicted score and |empirical r| over the database. Mean over query scales with bootstrap 95% "
            f"interval ({N_BOOT} resamples of query scales). Modes: `signed` ranks by the signed prediction, "
            "`abs` by its absolute value, `abs_cross_instrument` additionally drops database scales from the "
            "query's own instrument (closer to real synth-net use). n_queries_no_relevant = queries for which no "
            "database scale reaches the relevance threshold (their P@k is 0 under every variant)."
        )
        md.append("")
        rt = ret.copy()
        for m in ["rho", "p_at_5", "p_at_10", "ndcg_at_10"]:
            rt[m] = rt.apply(lambda r: f"{r[m]:.3f} [{r[m + '_lo']:.3f}, {r[m + '_hi']:.3f}]", axis=1)
        md.append(
            md_table(
                rt.sort_values(["mode", "variant"], key=lambda s: s.map({v: i for i, v in enumerate(ALL_PREDICTORS)}) if s.name == "variant" else s),
                ["mode", "variant", "n_queries", "n_queries_no_relevant", "rho", "p_at_5", "p_at_10", "ndcg_at_10"],
            )
        )
        md.append("")
        md.append("### Item-level sign geometry (context for C and D)")
        md.append("")
        ri = aligned[aligned.documented_reverse]
        pi = aligned[~aligned.documented_reverse]
        md.append(
            f"Mean raw cosine of each item with the other items of its scale (others signed by documentation): "
            f"for the {len(ri)} documented-reverse items it is negative in {100 * (ri.mean_cos_raw_vs_scale < 0).mean():.0f}% "
            f"of cases (median {ri.mean_cos_raw_vs_scale.median():.3f}); for the {len(pi)} positive items it is positive in "
            f"{100 * (pi.mean_cos_raw_vs_scale > 0).mean():.0f}% (median {pi.mean_cos_raw_vs_scale.median():.3f}). "
            f"Empirically the reverse items' raw correlation is negative in {100 * (ri.mean_emp_r_raw_vs_scale < 0).mean():.0f}% "
            "of cases. So the model usually gets the direction of reverse items right, but with much weaker magnitude "
            "than for positive items, which is why D's 'cosine with the positive centroid < 0' test often fails."
        )
        md.append("")
        md.append("### Variant C: items whose aligned sign disagrees with documentation")
        md.append("")
        md.append(
            f"{len(flips)} of {len(aligned)} item-scale memberships changed sign "
            f"({int(flips.documented_reverse.sum())} documented-reverse items became positive, "
            f"{int((~flips.documented_reverse).sum())} documented-positive items became reverse). "
            f"{int((~aligned.groupby('scale').converged.first()).sum())} scales did not converge within 10 iterations. "
            "`cos_to_centroid` is the cosine of the documented-signed item with the final centroid. "
            "`mean_emp_r_raw` / `mean_cos_raw` are the mean empirical correlation / mean cosine of the RAW item with "
            "the other items of the scale (those signed by documentation): positive = the raw item runs with the "
            "construct. If `documented_reverse` is True and `mean_emp_r_raw` is negative, the documentation is right "
            "and the model has the sign wrong; if `mean_emp_r_raw` agrees with the model instead, the documentation "
            "is the likelier error."
        )
        md.append("")
        if len(flips):
            # documentation is right if sign(mean_emp_r_raw) == documented sign
            doc_sign = np.where(flips.documented_reverse, -1.0, 1.0)
            doc_right = np.sign(flips.mean_emp_r_raw_vs_scale.to_numpy()) == doc_sign
            md.append(
                f"Verdict by empirical evidence: {int(doc_right.sum())} of {len(flips)} flips contradict the "
                f"empirical keying (model sign errors), {int((~doc_right).sum())} agree with it (the documented "
                "keying is the likelier error)."
            )
            md.append("")
            fl = flips.rename(columns={"mean_emp_r_raw_vs_scale": "mean_emp_r_raw", "mean_cos_raw_vs_scale": "mean_cos_raw"})
            md.append(
                md_table(
                    fl.sort_values("cos_to_centroid"),
                    ["scale", "item", "item_text", "documented_reverse", "cos_to_centroid", "mean_emp_r_raw", "mean_cos_raw"],
                )
            )
            md.append("")
        md.append("### Variant D: reverse items not reflected")
        md.append("")
        md.append(
            f"{n_no_rev} scales have no reverse item (D = plain mean = B there). {n_all_rev} scales have ONLY reverse "
            f"items; D falls back to the plain mean for these, which points AGAINST the construct (B negates it). "
            f"Of the remaining {n_mixed} mixed scales, {n_unref} have at least one reverse item that was NOT "
            "reflected because its cosine with the positive centroid was already >= 0. Such an item is either a "
            "model sign error (the model already places it with the positive items) or a documentation error; "
            "see `aligned_items.csv` (columns hyperplane_reflected, mean_emp_r_raw_vs_scale) for the evidence."
        )
        md.append("")
        unref_items = aligned[aligned.documented_reverse & ~aligned.hyperplane_reflected & ~aligned.scale.isin(hyper.scale[hyper.degenerate])]
        if len(unref_items):
            emp_neg = unref_items.mean_emp_r_raw_vs_scale < 0
            md.append(
                f"At item level, {len(unref_items)} reverse items were not reflected. Their raw empirical correlation with "
                f"the rest of the scale is negative for {int(emp_neg.sum())} (documentation right, model places the item "
                f"on the wrong side) and positive for {int((~emp_neg).sum())} (documented keying likely wrong). "
                f"Median cosine with the positive centroid: {unref_items.hyperplane_cos_to_pos_c.median():.3f}."
            )
            md.append("")
        unref = hyper[(hyper.n_reverse_not_reflected > 0) & ~hyper.all_reverse]
        if len(unref):
            md.append(
                md_table(
                    unref,
                    ["scale", "n_items", "n_reverse", "n_reflected", "n_reverse_not_reflected", "unreflected_items", "unreflected_cos_to_pos_c"],
                )
            )
            md.append("")

    pd.concat(all_pp).to_csv(os.path.join(OUT, "scale_pair_predictions.csv"), index=False)
    pd.concat(all_acc).to_csv(os.path.join(OUT, "accuracy_metrics.csv"), index=False)
    pd.concat(all_ret).to_csv(os.path.join(OUT, "retrieval_metrics.csv"), index=False)
    pd.concat(all_pq).to_csv(os.path.join(OUT, "retrieval_per_query.csv"), index=False)
    pd.concat(all_aligned).to_csv(os.path.join(OUT, "aligned_items.csv"), index=False)
    pd.concat(all_aligned).query("disagrees").to_csv(os.path.join(OUT, "aligned_sign_flips.csv"), index=False)
    pd.concat(all_hyper).to_csv(os.path.join(OUT, "hyperplane_scales.csv"), index=False)
    pd.concat(repro_rows).to_csv(os.path.join(OUT, "paper_reproduction.csv"), index=False)
    pd.DataFrame(ds_info).to_csv(os.path.join(OUT, "dataset_info.csv"), index=False)
    with open(os.path.join(OUT, "summary.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
