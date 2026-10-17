"""Band mapping, spectral indices and pixel-validity rules (numpy only)."""
import re

import numpy as np

BAND_NAMES = ("B01", "B02", "B03", "B04", "B08", "B8A", "B11", "B12")
SI_NAMES = ("MNDWI", "NDI_B1_B11", "SR_B1B2_B3B8A")
# Bands read by the three indices; every one must be valid for a pixel to count.
SI_BANDS = ("B01", "B02", "B03", "B8A", "B11")
SENSOR_RE = re.compile(r"_(S2|L8|L9)_[0-9A-Za-z]+$")


def sensor_of(scene):
    m = SENSOR_RE.search(scene)
    return m[1] if m else "unknown"


def band_index(names=BAND_NAMES):
    """Map band name -> 0-based channel index of the eight-band VRT."""
    names = list(names)
    if len(set(names)) != len(names):
        raise ValueError("Duplicate band names")
    missing = [b for b in SI_BANDS if b not in names]
    if missing:
        raise ValueError(f"Missing bands required for spectral indices: {missing}")
    return {b: i for i, b in enumerate(names)}


def model_input(raw, scale, mean, std):
    """Replicate the segmentation loader: /scale, clean nodata, mean/std normalize.

    raw: (C, H, W). mean/std: length-C sequences from the probe's own training run.
    """
    a = raw.astype(np.float32) / np.float32(scale)
    np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    a[(a > 100.0) | (a < -100.0)] = 0.0
    m = np.asarray(mean, dtype=np.float32)[:, None, None]
    s = np.asarray(std, dtype=np.float32)[:, None, None]
    return (a - m) / s


def model_valid_mask(raw):
    """Pixels the segmentation model actually saw: finite and not all-zero nodata."""
    finite = np.isfinite(raw).all(axis=0)
    nonzero = (np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0) != 0).any(axis=0)
    return finite & nonzero


def reflectance(raw, scale, offset=0.0):
    """Physical reflectance for spectral indices: (DN - offset) / scale.

    The delivered Sen2Like rasters carry a +1000 DN additive offset (sea pixels floor at
    DN 1001 in every band and sensor), so indices need ``offset=1000``. The segmentation
    input deliberately stays ``DN / scale`` without it, exactly as the probe was trained.
    """
    return (raw.astype(np.float32) - np.float32(offset)) / np.float32(scale)


def compute_si(refl, bands, den_floor=1e-6, clip=10.0):
    """Return ((3, H, W) indices, (H, W) validity) from physical reflectance.

    Unlike the original, invalid pixels are never zero-filled: a pixel is valid only
    when each required band is finite, positive and plausible and every denominator
    exceeds ``den_floor``. Invalid entries are NaN so callers cannot mistake them
    for a real index value.
    """
    b = {name: refl[i] for name, i in bands.items() if name in SI_BANDS}
    ok = np.ones(refl.shape[1:], dtype=bool)
    for v in b.values():
        ok &= np.isfinite(v) & (v > 0) & (v <= 100.0)
    den = np.stack([b["B03"] + b["B11"], b["B01"] + b["B11"], b["B03"] + b["B8A"]])
    ok &= (den > den_floor).all(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        si = np.stack([
            (b["B03"] - b["B11"]) / den[0],
            (b["B01"] - b["B11"]) / den[1],
            (b["B01"] + b["B02"]) / den[2],
        ]).astype(np.float32)
    si = np.clip(si, -clip, clip)
    ok &= np.isfinite(si).all(axis=0)
    si[:, ~ok] = np.nan
    return si, ok
