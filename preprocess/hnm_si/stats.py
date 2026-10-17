"""Train-only oil reference, Mahalanobis distance, SI cutoff and quota sampling."""
import numpy as np


def ledoit_wolf_reference(pool):
    """Fit the oil-pixel SI covariance (Ledoit-Wolf shrinkage) from a (N, 3) pool."""
    from sklearn.covariance import LedoitWolf
    pool = np.asarray(pool, dtype=np.float64)
    if pool.ndim != 2 or not np.isfinite(pool).all():
        raise ValueError("Oil pool must be a finite N x D array")
    lw = LedoitWolf(store_precision=True).fit(pool)
    return {"median": np.median(pool, axis=0), "mean": pool.mean(axis=0),
            "cov": lw.covariance_, "precision": lw.precision_,
            "shrinkage": float(lw.shrinkage_), "n": int(len(pool)),
            "cond": float(np.linalg.cond(lw.covariance_))}


def mahalanobis(points, centers, precision):
    """d = sqrt((p - f)^T P (p - f)) row-wise; NaN rows stay NaN."""
    diff = np.asarray(points, dtype=np.float64) - np.asarray(centers, dtype=np.float64)
    d2 = np.einsum("ni,ij,nj->n", diff, precision, diff)
    return np.sqrt(np.where(np.isnan(d2), np.nan, np.maximum(d2, 0.0)))


def lower_cutoff(distances, q):
    """Distance at or below which candidates are dropped (closest to oil). None if q == 0."""
    d = np.asarray(distances, dtype=np.float64)
    d = d[np.isfinite(d)]
    if q <= 0 or len(d) == 0:
        return None
    return float(np.quantile(d, q))


def outlier_scenes(scenes, sigma):
    """Scenes whose candidate count exceeds mean + sigma * std (original 3-sigma rule)."""
    if sigma is None:
        return set()
    names, counts = np.unique(np.asarray(scenes), return_counts=True)
    if len(names) < 3:
        return set()
    threshold = counts.mean() + sigma * counts.std(ddof=1)
    return {str(n) for n, c in zip(names, counts) if c > threshold}


def stratified_select(keys_by_scene, target, seed):
    """Pick ``target`` unique items spreading evenly over scenes, without replacement.

    Round-robin over a seeded scene order and a seeded per-scene order, so the result
    is deterministic and any scene with spare candidates fills shortfalls elsewhere.
    Returns fewer than ``target`` only when the whole pool is smaller.
    """
    rng = np.random.default_rng(seed)
    scenes = sorted(keys_by_scene)
    queues = {s: [keys_by_scene[s][i] for i in rng.permutation(len(keys_by_scene[s]))]
              for s in scenes}
    order = [scenes[i] for i in rng.permutation(len(scenes))]
    picked, depth = [], 0
    while len(picked) < target:
        progressed = False
        for s in order:
            if depth < len(queues[s]):
                picked.append(queues[s][depth])
                progressed = True
                if len(picked) == target:
                    break
        if not progressed:
            break
        depth += 1
    return picked
