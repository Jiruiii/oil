"""Deterministic hard-negative and facility-location selection."""
import numpy as np


def facility_location(features, k, batch_size=256):
    n = len(features)
    if k <= 0 or n == 0:
        return []
    if k >= n:
        return list(range(n))
    f = np.asarray(features, dtype=np.float32)
    if f.ndim != 2 or not np.isfinite(f).all():
        raise ValueError("Expected finite N x D embeddings")
    f = f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)
    first = int(np.argmax(f @ f.mean(axis=0)))
    selected = [first]
    available = np.ones(n, dtype=bool)
    available[first] = False
    best_similarity = f @ f[first]
    for _ in range(1, k):
        best_gain, best_idx = -np.inf, -1
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            similarity = f @ f[start:stop].T
            gains = np.maximum(similarity - best_similarity[:, None], 0).sum(axis=0)
            gains[~available[start:stop]] = -np.inf
            j = int(np.argmax(gains))
            if gains[j] > best_gain:
                best_gain, best_idx = float(gains[j]), start + j
        selected.append(best_idx)
        available[best_idx] = False
        best_similarity = np.maximum(best_similarity, f @ f[best_idx])
    return selected


def select_negatives(scores, embeddings, count, hard_fraction=.5):
    scores = np.asarray(scores)
    embeddings = np.asarray(embeddings)
    if count < 0 or not 0 <= hard_fraction <= 1:
        raise ValueError("Invalid sampling count/fraction")
    if (scores.ndim != 1 or embeddings.ndim != 2 or len(scores) != len(embeddings)
            or not np.isfinite(scores).all() or not np.isfinite(embeddings).all()):
        raise ValueError("Scores and embeddings must align and be finite")
    count = min(count, len(scores))
    hard_count = int(count * hard_fraction)
    order = np.argsort(-scores, kind="stable")
    hard = order[:hard_count].tolist()
    remaining = order[hard_count:]
    diverse = [int(remaining[i]) for i in
               facility_location(embeddings[remaining], count - hard_count)]
    return hard, diverse
