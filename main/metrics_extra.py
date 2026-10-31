"""Extra evaluation atoms shared by the reconstruction and the offline diagnostics.

boundary_counts  Boundary IoU (Cheng et al., CVPR 2021) restricted to scored pixels: the
                 oil pixels within ``d`` px of the oil contour, for ground truth and for the
                 prediction; Boundary IoU = |G_d & P_d| / |G_d | P_d|, pooled over scenes.
                 Pixels outside the scored area count as "unknown", never as a contour, so
                 the edge of the tested tiles does not create a false boundary.
bg_tile_counts   false-positive pixels on an oil-free background tile; added to the oil
                 union these give the background-inclusive Oil IoU
                 TP / (TP + FN + FP_reconstruction + FP_background_tiles).
"""
import cv2
import numpy as np


def boundary_band(region, valid, d):
    """Pixels of ``region`` (inside ``valid``) within ``d`` px of its contour."""
    region = region & valid
    if not region.any():
        return region
    ys, xs = np.nonzero(valid)
    y0, y1 = max(0, ys.min() - d - 1), min(valid.shape[0], ys.max() + d + 2)
    x0, x1 = max(0, xs.min() - d - 1), min(valid.shape[1], xs.max() + d + 2)
    sub_r, sub_v = region[y0:y1, x0:x1], valid[y0:y1, x0:x1]
    # Unknown (unscored) pixels are treated as region so they never form a contour.
    dist = cv2.distanceTransform((sub_r | ~sub_v).astype(np.uint8), cv2.DIST_L2, 5)
    band = np.zeros_like(region)
    band[y0:y1, x0:x1] = sub_r & (dist <= d)
    return band


def boundary_counts(gt_oil, pred_oil, valid, d):
    """(intersection, union) of the GT and predicted boundary bands of width ``d``."""
    g = boundary_band(gt_oil, valid, d)
    p = boundary_band(pred_oil, valid, d)
    return int((g & p).sum()), int((g | p).sum())


def bg_tile_counts(pred_oil, gt_raw, oil_value=0):
    """(false-positive px, scored px) on one background tile; GT oil there is an error."""
    if (gt_raw == oil_value).any():
        raise ValueError("A background tile contains ground-truth oil")
    return int(pred_oil.sum()), int(pred_oil.size)
