"""Remeasure a declared radial shaft using its original finite ink track.

The short arrow detector supplies ownership and a search direction only.
Neither CAD coordinates nor radius/length nominals enter this observation.
"""
from __future__ import annotations

import math
import cv2
import numpy as np


def radial_shaft_track(gray, observation, records=()):
    """Return a stable long-shaft ray, or retain the unmodified observation.

    Only isolated thin ink runs are measured. OCR boxes belonging to this
    radius are explicit occlusions, not ink support. A long missing run or
    competing tracks cannot be bridged. This is not a circle-center proof:
    the caller still checks two separate shafts and their finite termination.
    """
    tip = np.asarray(observation.get("arrowhead", {}).get("tip_px"), float)
    direction = np.asarray(observation.get("arrowhead", {}).get("direction_px"), float)
    if (tip.shape != (2,) or direction.shape != (2,) or
            not np.isfinite(np.r_[tip, direction]).all() or np.linalg.norm(direction) <= 0):
        return None
    backward = -direction/np.linalg.norm(direction)
    normal = np.array([-backward[1], backward[0]])
    _, ink = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    height, width = gray.shape
    own_boxes = []
    for row in records:
        if row.get("id") != observation.get("record_id"):
            continue
        box = np.asarray(row.get("box"), float)
        if box.ndim == 2 and box.shape[1] == 2 and np.isfinite(box).all():
            own_boxes.append((box.min(axis=0)-3., box.max(axis=0)+3.))
    maximum = min(2048., math.hypot(height, width)*.8)
    measured, distances = [], []
    missing = 0.
    skipped = 0
    ambiguous = 0
    for travel in np.arange(24., maximum, 3.):
        point = tip+travel*backward
        if not (2 <= point[0] < width-2 and 2 <= point[1] < height-2):
            break
        if any(np.all(point >= low) and np.all(point <= high) for low, high in own_boxes):
            skipped += 1
            continue
        extent = min(24., max(5., travel*math.sin(math.radians(3.))))
        offsets = np.arange(-math.ceil(extent), math.ceil(extent)+1)
        pixels = np.rint(point+offsets[:, None]*normal).astype(int)
        if ((pixels < 0).any() or (pixels[:, 0] >= width).any() or (pixels[:, 1] >= height).any()):
            break
        occupied = ink[pixels[:, 1], pixels[:, 0]].astype(bool)
        starts = np.flatnonzero(occupied & ~np.r_[False, occupied[:-1]])
        ends = np.flatnonzero(occupied & ~np.r_[occupied[1:], False])
        choices = [(a, b) for a, b in zip(starts, ends)
                   if a > 0 and b < len(offsets)-1 and 1 <= b-a+1 <= 7]
        if len(choices) == 1:
            a, b = choices[0]
            measured.append(point+offsets[a:b+1].mean()*normal)
            distances.append(float(travel))
            missing = 0.
        else:
            ambiguous += int(len(choices) > 1)
            missing += 3.
            if missing > 15. and len(measured) >= 12:
                break
            # Do not search through unrelated empty regions for a remote line.
            if travel > 90. and len(measured) < 12:
                break
    if len(measured) < 32 or np.ptp(distances) < 120.:
        return None
    points = np.asarray(measured)
    center = points.mean(axis=0)
    _, _, axes = np.linalg.svd(points-center, full_matrices=False)
    actual = axes[0]
    if actual@backward < 0:
        actual = -actual
    perpendicular = np.array([-actual[1], actual[0]])
    residual = np.abs((points-center)@perpendicular)
    p90 = float(np.quantile(residual, .9))
    angle = math.degrees(math.acos(float(np.clip(actual@backward, -1., 1.))))
    tip_gap = abs(float((tip-center)@perpendicular))
    observed_fraction = len(measured)/max(1, len(measured)+ambiguous)
    if p90 > 2. or angle > 3. or tip_gap > 4. or observed_fraction < .7:
        return None
    origin = center+float((tip-center)@actual)*actual
    return {"origin_px": origin.tolist(), "direction_px": (-actual).tolist(),
        "evidence": {"method": "finite_original_radial_shaft_track_v1", "verified": True,
            "sample_count": len(measured), "observed_span_px": float(np.ptp(distances)),
            "fit_p90_residual_px": p90, "arrow_tip_line_gap_px": tip_gap,
            "short_direction_change_degrees": angle, "own_ocr_occluded_sample_count": skipped,
            "ambiguous_sample_count": ambiguous, "nominal_used": False,
            "cad_geometry_used": False, "coordinates_locked": False}}
