"""Pooling item embeddings into one vector per scale, with different treatments of
reverse-keyed items.

Self-contained: depends on numpy only. Intended to be copied unchanged into
synth-net.

Every function takes

    item_vectors  : array (n_items, dim). Item embeddings; they are L2-normalised
                    to unit length inside every variant before anything else.
    reverse_flags : boolean array (n_items,). True where the documentation says
                    the item is reverse-keyed relative to the scale's construct.

and returns one scale vector of shape (dim,). The returned vector is NOT
re-normalised, so callers that need cosine similarity should normalise it
themselves (see `cosine`).

Variants
--------
plain          A. mean of unit item vectors, keying ignored.
documented     B. reverse items multiplied by -1, then mean.
aligned        C. sign-alignment iteration seeded with the documented keys.
hyperplane_old D. the method currently in synth-net (Supplementary Note 2):
                  reflect documented-reverse items through the hyperplane
                  between the positive and negative centroids, but only those
                  whose cosine with the positive centroid is negative.
positive_only  E. mean of documented positively keyed items only. Equivalent to
                  D whenever D reflects every reverse item.

`pool_aligned` and `pool_hyperplane_old` accept `return_details=True` to also
return a dict of per-item diagnostics.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "unit",
    "cosine",
    "pool_plain",
    "pool_documented",
    "pool_aligned",
    "pool_hyperplane_old",
    "pool_positive_only",
    "POOLERS",
    "pool_scale",
]


def unit(X: np.ndarray) -> np.ndarray:
    """L2-normalise rows of X (2-D) or X itself (1-D). Zero vectors stay zero."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        n = np.linalg.norm(X)
        return X / n if n > 0 else X
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors (0.0 if either is zero)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


def _prepare(item_vectors, reverse_flags):
    X = unit(np.atleast_2d(np.asarray(item_vectors, dtype=float)))
    if reverse_flags is None:
        rev = np.zeros(X.shape[0], dtype=bool)
    else:
        rev = np.asarray(reverse_flags, dtype=bool).reshape(-1)
    if rev.shape[0] != X.shape[0]:
        raise ValueError(
            f"reverse_flags has length {rev.shape[0]} but there are {X.shape[0]} items"
        )
    return X, rev


# --------------------------------------------------------------------------- A
def pool_plain(item_vectors, reverse_flags=None) -> np.ndarray:
    """A. Mean of unit item vectors; keying ignored."""
    X, _ = _prepare(item_vectors, reverse_flags)
    return X.mean(axis=0)


# --------------------------------------------------------------------------- B
def pool_documented(item_vectors, reverse_flags) -> np.ndarray:
    """B. Multiply documented reverse items by -1, then mean."""
    X, rev = _prepare(item_vectors, reverse_flags)
    signs = np.where(rev, -1.0, 1.0)
    return (X * signs[:, None]).mean(axis=0)


# --------------------------------------------------------------------------- C
def pool_aligned(
    item_vectors,
    reverse_flags,
    max_iter: int = 10,
    return_details: bool = False,
    leave_one_out: bool = False,
):
    """C. Sign-alignment iteration.

    Signs start from the documentation. Repeat: compute the signed centroid, set
    each item's sign to the sign of its dot product with the centroid, until no
    sign changes (at most `max_iter` iterations). An item whose dot product is
    exactly zero keeps its current sign.

    Note that with unit vectors an item's own contribution to the centroid is
    always +1/n in its favour, so an item flips only if the sum of its signed
    cosines with the *other* items is below -1. `leave_one_out=True` drops the
    item's own vector from the centroid it is compared with, which flips an item
    as soon as that sum is below 0 (more sensitive, also more eager).

    Returns the mean vector, and with `return_details=True` a tuple
    (vector, details) where details has
        'signs'               final signs (+1/-1) per item
        'disagrees'           bool per item: final sign != documented sign
        'cos_to_centroid'     cosine of each *documented-signed* item vector
                              with the final centroid (negative = disagreement)
        'n_iter'              iterations run
        'converged'           whether signs stopped changing within max_iter
    """
    X, rev = _prepare(item_vectors, reverse_flags)
    doc_signs = np.where(rev, -1.0, 1.0)
    signs = doc_signs.copy()
    converged = False
    n_iter = 0
    n = X.shape[0]
    for n_iter in range(1, max_iter + 1):
        signed_sum = (X * signs[:, None]).sum(axis=0)
        if leave_one_out and n > 1:
            # centroid of the other items only (own signed vector removed)
            dots = (X @ signed_sum - signs) / (n - 1)
        else:
            dots = X @ signed_sum / n
        new_signs = np.where(dots > 0, 1.0, np.where(dots < 0, -1.0, signs))
        if np.array_equal(new_signs, signs):
            converged = True
            break
        signs = new_signs
    centroid = (X * signs[:, None]).mean(axis=0)
    if not return_details:
        return centroid
    nc = np.linalg.norm(centroid)
    cos_doc = (X * doc_signs[:, None]) @ centroid / nc if nc > 0 else np.zeros(X.shape[0])
    details = {
        "signs": signs,
        "disagrees": signs != doc_signs,
        "cos_to_centroid": cos_doc,
        "n_iter": n_iter,
        "converged": converged,
    }
    return centroid, details


# --------------------------------------------------------------------------- D
def pool_hyperplane_old(item_vectors, reverse_flags, return_details: bool = False):
    """D. Hyperplane reflection, as currently implemented in synth-net.

    pos_c, neg_c = centroids of documented positive / reverse items
    u        = (pos_c - neg_c) / |pos_c - neg_c|
    midpoint = (pos_c + neg_c) / 2
    dist_i   = (v_i - midpoint) . u
    Items that are documented reverse AND have negative cosine with pos_c are
    reflected: v_i -> v_i - 2 * dist_i * u. Then the mean of all items.
    If there are no reverse items, or every item is reverse, the plain mean is
    returned and nothing is reflected.

    With `return_details=True` returns (vector, details) where details has
        'reflected'      bool per item: was reflected
        'not_reflected'  bool per item: documented reverse but NOT reflected
        'cos_to_pos_c'   cosine of each item with the positive centroid
        'degenerate'     True when the plain-mean fallback was used
    """
    X, rev = _prepare(item_vectors, reverse_flags)
    n = X.shape[0]
    n_rev = int(rev.sum())
    reflected = np.zeros(n, dtype=bool)
    cos_pos = np.full(n, np.nan)
    degenerate = n_rev == 0 or n_rev == n
    if not degenerate:
        pos_c = X[~rev].mean(axis=0)
        neg_c = X[rev].mean(axis=0)
        d = pos_c - neg_c
        nd = np.linalg.norm(d)
        npos = np.linalg.norm(pos_c)
        if nd == 0 or npos == 0:
            degenerate = True
    if degenerate:
        vec = X.mean(axis=0)
    else:
        u = d / nd
        midpoint = (pos_c + neg_c) / 2.0
        dist = (X - midpoint) @ u
        cos_pos = X @ pos_c / npos
        reflected = rev & (cos_pos < 0)
        Xr = X.copy()
        Xr[reflected] = X[reflected] - 2.0 * dist[reflected, None] * u
        vec = Xr.mean(axis=0)
    if not return_details:
        return vec
    details = {
        "reflected": reflected,
        "not_reflected": rev & ~reflected,
        "cos_to_pos_c": cos_pos,
        "degenerate": degenerate,
    }
    return vec, details


# --------------------------------------------------------------------------- E
def pool_positive_only(item_vectors, reverse_flags) -> np.ndarray:
    """E. Mean of documented positively keyed items only.

    Falls back to the plain mean when every item is reverse-keyed (mirrors D).
    """
    X, rev = _prepare(item_vectors, reverse_flags)
    if rev.all():
        return X.mean(axis=0)
    return X[~rev].mean(axis=0)


POOLERS = {
    "plain": pool_plain,
    "documented": pool_documented,
    "aligned": pool_aligned,
    "hyperplane_old": pool_hyperplane_old,
    "positive_only": pool_positive_only,
}


def pool_scale(item_vectors, reverse_flags, method: str = "documented") -> np.ndarray:
    """Dispatch on `method` (one of POOLERS) and return the scale vector."""
    try:
        fn = POOLERS[method]
    except KeyError as e:
        raise ValueError(f"unknown method {method!r}; choose from {sorted(POOLERS)}") from e
    return fn(item_vectors, reverse_flags)
