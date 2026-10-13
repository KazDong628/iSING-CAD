"""Bounded original-pixel observations of oblique two-arrow dimensions.

This module has no CAD, scale, nominal-value or reference-geometry input.
An observation is not a numeric binding: the caller must independently prove
which finite source objects the observed arrow tips actually measure.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .dimension_evidence import _dimension_station_arrow


_MAX_RECORDS = 64
_MAX_SHAFTS = 48
_MAX_DIRECTIONS = 24
_MAX_ARROWS_PER_SHAFT = 24


def _box(row):
    try:
        box = np.asarray(row.get("box"), dtype=float)
    except (TypeError, ValueError):
        return None
    return box if box.shape == (4, 2) and np.isfinite(box).all() else None


def _eligible(row):
    parsed = row.get("parsed") or {}
    # The numeric nominal is deliberately neither inspected nor returned.
    return parsed.get("kind") == "length" and not parsed.get("reference") and _box(row) is not None


def _label_distance(box, direction, cross, lo, hi):
    normal = np.array([-direction[1], direction[0]])
    along, across = box @ direction, box @ normal
    cross_size = max(8., float(np.ptp(across)))
    along_size = max(8., float(np.ptp(along)))
    normal_gap = max(float(across.min())-cross, cross-float(across.max()), 0.)
    along_gap = max(lo-float(along.max()), float(along.min())-hi, 0.)
    return normal_gap/cross_size+along_gap/along_size, normal_gap, along_gap, cross_size, along_size


def _shaft_hypotheses(masked, local_box):
    """Hough proposes source strokes; it never establishes an arrow pair."""
    size = max(12., float(np.linalg.norm(np.ptp(local_box, axis=0))))
    edge = cv2.Canny(masked, 70, 180)
    raw = cv2.HoughLinesP(edge, 1., np.pi/720, threshold=18,
        minLineLength=max(18, int(size*.32)), maxLineGap=8)
    if raw is None:
        return []
    hypotheses = []
    for row in raw[:, 0]:
        segment = np.asarray(row, dtype=float).reshape(2, 2)
        delta = segment[1]-segment[0]
        length = float(np.linalg.norm(delta))
        if length <= 0:
            continue
        direction = delta/length
        if direction[0] < 0:
            direction = -direction
        # Axis dimensions already have their independent observer.
        if min(abs(direction)) < math.sin(math.radians(3.)):
            continue
        normal = np.array([-direction[1], direction[0]])
        cross = float(segment.mean(axis=0) @ normal)
        along = segment @ direction
        distance, normal_gap, along_gap, cross_size, along_size = _label_distance(
            local_box, direction, cross, float(along.min()), float(along.max()))
        if normal_gap > cross_size or along_gap > along_size*1.5:
            continue
        hypotheses.append(dict(segment=segment, direction=direction, length=length, distance=distance))
    # Deduplicate the two edges and fragments of the SAME physical shaft.
    # A busy ROI outside the declared bound remains unknown, not a chosen ID.
    hypotheses.sort(key=lambda h: (-h["length"], h["distance"]))
    unique = []
    for item in hypotheses:
        if any(abs(float(item["direction"] @ other["direction"])) > math.cos(math.radians(1.5)) and
               abs(float((item["segment"].mean(axis=0)-other["segment"].mean(axis=0)) @
                         np.array([-other["direction"][1], other["direction"][0]]))) <= 3.
               for other in unique):
            continue
        unique.append(item)
        if len(unique) > _MAX_SHAFTS:
            return []
    return unique


def _rotate_roi(gray, direction):
    """Rigid original-resolution rotation, with an exact inverse point map."""
    normal = np.array([-direction[1], direction[0]])
    basis = np.stack([direction, normal])
    h, w = gray.shape
    corners = np.array([[0., 0.], [w-1., 0.], [0., h-1.], [w-1., h-1.]]) @ basis.T
    low, high = np.floor(corners.min(axis=0))-3., np.ceil(corners.max(axis=0))+3.
    affine = np.column_stack([basis, -low])
    size = tuple((high-low+1).astype(int))
    rotated = cv2.warpAffine(gray, affine, size, flags=cv2.INTER_LINEAR, borderValue=255)
    def to_frame(points):
        return np.asarray(points, float) @ basis.T-low
    def from_frame(points):
        return (np.asarray(points, float)+low) @ basis
    return rotated, to_frame, from_frame


def _width_profile(gray, cross, stations, extent):
    """Same contiguous filled-ink width as the existing station-arrow check."""
    result = np.zeros(len(stations), dtype=float)
    iy = int(round(cross))
    low, high = max(0, iy-extent), min(gray.shape[0], iy+extent+1)
    if high <= low:
        return result
    center = iy-low
    for index, station in enumerate(stations):
        ix = int(station)
        if not 0 <= ix < gray.shape[1]:
            continue
        ink = gray[low:high, ix] < 170
        near = [j for j in range(max(0, center-2), min(len(ink), center+3)) if ink[j]]
        if not near:
            continue
        start = min(near, key=lambda j: abs(j-center))
        a = b = start
        while a > 0 and ink[a-1]:
            a -= 1
        while b+1 < len(ink) and ink[b+1]:
            b += 1
        result[index] = b-a+1
    return result


def _shaft_arrows(rotated, cross, low, high, label_size):
    """Prefilter taper signatures, then run the original pixel verifier."""
    stations = np.arange(max(0, int(math.floor(low))), min(rotated.shape[1], int(math.ceil(high))+1))
    if len(stations) < 12:
        return []
    lengths = sorted({max(7., min(48., label_size*f)) for f in (.08, .12, .18, .24, .32, .42)})
    proposals = set()
    for length in lengths:
        profile = _width_profile(rotated, cross, stations, max(7, int(length*.65)))
        for sign in (-1., 1.):
            offsets = np.rint(sign*length*np.array([.13, .40, .70, 1.25, 1.60])).astype(int)
            indices = np.arange(len(stations))[:, None]-offsets
            inside = ((indices >= 0) & (indices < len(stations))).all(axis=1)
            widths = profile[np.clip(indices, 0, len(stations)-1)]
            narrow, middle, body, shaft, tail = widths.T
            shaft = np.maximum(np.maximum(shaft, tail), 1.)
            good = (inside & (narrow >= shaft+1) & (middle >= 2) &
                    (body >= np.maximum(4., 1.8*shaft)) & (middle >= .45*body) &
                    (middle <= 1.35*body) & (narrow <= .72*body) & (body <= .85*length))
            proposals.update((int(stations[i]), sign) for i in np.flatnonzero(good))
    # The bound is on physical taper groups, rather than every adjacent raster
    # offset which describes one arrowhead. No nominal controls this grouping.
    verified = []
    for station, sign in sorted(proposals):
        arrow = _dimension_station_arrow(rotated, "x", cross, station, sign, label_size)
        if arrow is None:
            continue
        # The shared station verifier permits an absent continuation at a
        # separately observed extension station. Here no extension is assumed:
        # both thin-shaft samples behind the filled body must actually exist.
        narrow, middle, body, shaft, tail = arrow["cross_section_widths_px"]
        if min(shaft, tail) <= 0 or not narrow < middle <= body:
            continue
        # Five isolated cross sections can accidentally sample two crossings
        # or two different heads on a hatched contour. Require one connected,
        # progressively widening filled body between those cross sections.
        fractions = np.arange(.10, .701, .05)
        taper_stations = np.rint(station+arrow["sample_tip_shift_px"]-
            sign*arrow["length_px"]*fractions).astype(int)
        taper = _width_profile(rotated, cross, taper_stations,
                               max(7, int(arrow["length_px"]*.65)))
        correlation = float(np.corrcoef(fractions, taper)[0, 1]) if np.ptp(taper) > 0 else 0.
        if (not np.isfinite(correlation) or correlation < .75 or taper.min() <= 0
                or taper.max() > 1.4*body or np.diff(taper).min() < -max(2., .25*body)):
            continue
        arrow = dict(arrow)
        arrow["filled_taper_profile"] = dict(verified=True, fractions=fractions.tolist(),
            widths_px=taper.tolist(), increasing_width_correlation=correlation,
            shaft_continuation_observed=True)
        arrow["tip_px"] = [float(station+arrow["sample_tip_shift_px"]), float(cross)]
        nearby = next((a for a in verified if a["direction_px"] == arrow["direction_px"] and
            abs(a["tip_px"][0]-arrow["tip_px"][0]) <= max(7., .6*arrow["length_px"])), None)
        if nearby is None:
            verified.append(arrow)
        elif arrow["score"] > nearby["score"]:
            nearby.clear(); nearby.update(arrow)
    if len(verified) > _MAX_ARROWS_PER_SHAFT:
        return []
    return sorted(verified, key=lambda a: a["tip_px"][0])


def _same_pair(first, second):
    a, b = np.asarray(first["endpoints_source_px"]), np.asarray(second["endpoints_source_px"])
    arrow_lengths = [arrow["length_px"] for row in (first, second)
                     for arrow in row["evidence"]["endpoint_arrows"]]
    # Two nearby Hough edges can locate the same physical tip a few raster
    # pixels apart. This is deduplication within an observed head footprint,
    # not a numeric dimensional accuracy allowance or a resized image.
    tip_window = max(5., .35*max(arrow_lengths))
    return min(float(np.linalg.norm(a-b, axis=1).max()),
               float(np.linalg.norm(a-b[::-1], axis=1).max())) <= tip_window


def oblique_dimension_witnesses(gray, records):
    """Return uniquely label-owned, independently verified oblique arrow pairs.

    The observation is bounded to an OCR-sized ROI at native pixel resolution.
    Unknown/incomplete/ambiguous pairs return no witness and remain unbound.
    Endpoints and direction are mapped back into the original source image.
    """
    gray = np.asarray(gray)
    if gray.ndim != 2 or not gray.size or gray.dtype != np.uint8:
        return []
    records = list(records)
    eligible = [row for row in records if _eligible(row)]
    if len(eligible) > _MAX_RECORDS:
        return []
    candidates = []
    for row in eligible:
        box = _box(row)
        size = np.maximum(np.ptp(box, axis=0), 8.)
        # An oversized OCR extent cannot grow this into an unbounded full-image
        # search. It stays unknown rather than silently rescaling pixel evidence.
        if max(size) > 320:
            continue
        padding = np.minimum(np.maximum(size*2.8, 80.), 320.)
        roi_low = np.maximum(np.floor(box.min(axis=0)-padding).astype(int), 0)
        roi_high = np.minimum(np.ceil(box.max(axis=0)+padding).astype(int), gray.shape[::-1])
        roi = gray[roi_low[1]:roi_high[1], roi_low[0]:roi_high[0]]
        if not roi.size:
            continue
        masked = roi.copy()
        for other in records:
            other_box = _box(other)
            if other_box is not None:
                cv2.fillPoly(masked, [np.rint(other_box-roi_low).astype(np.int32)], 255)
        hypotheses = _shaft_hypotheses(masked, box-roi_low)
        directions = []
        for hypothesis in hypotheses:
            if not any(abs(float(hypothesis["direction"] @ direction)) > math.cos(math.radians(1.5))
                       for direction in directions):
                directions.append(hypothesis["direction"])
        if len(directions) > _MAX_DIRECTIONS:
            continue
        own = []
        for direction in directions:
            rotated, to_frame, from_frame = _rotate_roi(roi, direction)
            rotated_box = to_frame(box-roi_low)
            along_size = max(8., float(np.ptp(rotated_box[:, 0])))
            cross_size = max(8., float(np.ptp(rotated_box[:, 1])))
            shaft_crosses = []
            for h in hypotheses:
                if abs(float(h["direction"] @ direction)) <= math.cos(math.radians(1.5)):
                    continue
                cross = float(to_frame(h["segment"])[:, 1].mean())
                if not any(abs(cross-old) <= 3. for old in shaft_crosses):
                    shaft_crosses.append(cross)
            for cross in shaft_crosses:
                low = max(0., float(rotated_box[:, 0].min())-max(80., along_size*2.8))
                high = min(rotated.shape[1]-1., float(rotated_box[:, 0].max())+max(80., along_size*2.8))
                arrows = _shaft_arrows(rotated, cross, low, high, float(np.linalg.norm(size)))
                for first, second in zip(arrows[:-1], arrows[1:]):
                    lo, hi = first["tip_px"][0], second["tip_px"][0]
                    if (first["direction_px"][0] != -second["direction_px"][0] or
                            hi-lo < max(12., 1.2*max(first["length_px"], second["length_px"]))):
                        continue
                    gap = max(lo-float(rotated_box[:, 0].max()), float(rotated_box[:, 0].min())-hi, 0.)
                    if gap > along_size*1.5:
                        continue
                    # Both heads must share an actual uninterrupted source shaft.
                    samples = np.arange(int(math.ceil(lo)), int(math.floor(hi))+1)
                    support = _width_profile(rotated, cross, samples, 3) > 0
                    if len(support) < 12 or float(support.mean()) < .90:
                        continue
                    endpoints = from_frame([[lo, cross], [hi, cross]])+roi_low
                    normal = np.array([-direction[1], direction[0]])
                    source_cross = float(endpoints.mean(axis=0) @ normal)
                    along = endpoints @ direction
                    cost, _, _, _, _ = _label_distance(box, direction, source_cross, along.min(), along.max())
                    competing = []
                    # A radius label can own a nearby single arrow which
                    # happens to align with another leader or hatch stroke.
                    # Include all parsed dimension-label kinds in ownership
                    # competition, even though only lengths can emit rows.
                    for other in records:
                        if other["id"] == row["id"]:
                            continue
                        if (_box(other) is None or (other.get("parsed") or {}).get("kind") not in
                                {"length", "radius", "diameter", "angle"}):
                            continue
                        other_cost, normal_gap, along_gap, height, width = _label_distance(
                            _box(other), direction, source_cross, along.min(), along.max())
                        if normal_gap <= height and along_gap <= width*1.5 and other_cost <= cost+.20:
                            competing.append(other["id"])
                    if competing:
                        continue
                    source_arrows = []
                    for arrow in (first, second):
                        item = {**arrow, "tip_px": (from_frame([arrow["tip_px"]])[0]+roi_low).tolist(),
                                "direction_px": (np.asarray(arrow["direction_px"]) @
                                    np.stack([direction, normal])).tolist()}
                        source_arrows.append(item)
                    witness = dict(record_id=row["id"], endpoints_source_px=endpoints.tolist(),
                        direction_px=direction.tolist(), arrow_pair_verified=True,
                        evidence=dict(method="bounded_original_pixel_oblique_opposing_arrows_v1",
                            verified=True, endpoint_arrows=source_arrows,
                            shaft_supported_fraction=float(support.mean()),
                            arrow_directions_opposed=True, same_source_shaft=True,
                            taper_tip_localization_quality=max(
                                a["cross_section_widths_px"][0]/a["cross_section_widths_px"][2]
                                for a in (first, second)),
                            label_association_unique=True, association_distance_label_heights=float(cost),
                            label_interval_gap_px=float(gap), roi_source_px=[roi_low.tolist(), roi_high.tolist()],
                            arrow_search_label_size_px=float(np.linalg.norm(size)),
                            rigid_rotation_only=True, original_pixel_resolution=True,
                            nominal_used_to_rank=False, nominal_used_to_set_span=False,
                            contour_contacts_verified=False, requires_contact_tip_recheck=True,
                            endpoint_search_uncertainty_px=[min(48., max(7., 2*a["length_px"]))
                                for a in (first, second)],
                            uncertainty_is_certified_accuracy_bound=False,
                            numeric_binding_certified=False, ground_truth_used=False))
                    same = next((old for old in own if _same_pair(witness, old)), None)
                    if same is None:
                        own.append(witness)
                    elif (witness["evidence"]["taper_tip_localization_quality"] <
                          same["evidence"]["taper_tip_localization_quality"]):
                        # Among observations of the SAME two physical heads,
                        # prefer narrowing closer to their tips over a wide
                        # interior-body station. No dimension/span participates.
                        same.clear(); same.update(witness)
        if len(own) == 1:
            candidates.append(own[0])
    # A physical pair cannot be simultaneously owned by two source labels.
    return [row for row in candidates if not any(other["record_id"] != row["record_id"] and
        _same_pair(row, other) for other in candidates)]


def refine_oblique_arrow_contacts(gray, witness, contacts_source_px):
    """Recheck a caller's two finite source-contour contacts at original pixels.

    Raster taper searches can locate the interior of a head rather than its
    exact contour-contact tip. This optional observation accepts only contacts
    on the same original shaft, inside the already observed head neighbourhood,
    with a fresh filled-taper and thin-shaft check at EACH contact. It neither
    selects CAD objects nor uses a nominal to move/choose the source contacts.
    The caller still has to certify unique finite-object ownership/semantics.
    """
    gray = np.asarray(gray)
    proof = witness.get("evidence") or {}
    try:
        contacts = np.asarray(contacts_source_px, float)
        old = np.asarray(witness["endpoints_source_px"], float)
        direction = np.asarray(witness["direction_px"], float)
        low, high = np.asarray(proof["roi_source_px"], int)
        old_arrows = proof["endpoint_arrows"]
        label_size = float(proof["arrow_search_label_size_px"])
    except (KeyError, TypeError, ValueError):
        return None
    if (gray.ndim != 2 or gray.dtype != np.uint8 or contacts.shape != (2, 2)
            or old.shape != (2, 2) or direction.shape != (2,) or len(old_arrows) != 2
            or not np.isfinite(contacts).all() or witness.get("arrow_pair_verified") is not True
            or proof.get("verified") is not True or not .999 <= np.linalg.norm(direction) <= 1.001):
        return None
    if np.linalg.norm(contacts-old[::-1], axis=1).sum() < np.linalg.norm(contacts-old, axis=1).sum():
        contacts = contacts[::-1]
    windows = [min(48., max(7., 2*a["length_px"])) for a in old_arrows]
    if any(np.linalg.norm(point-tip) > window for point, tip, window in zip(contacts, old, windows)):
        return None
    if (np.any(low < 0) or np.any(high > gray.shape[::-1]) or np.any(high <= low)
            or np.any(contacts < low) or np.any(contacts >= high)):
        return None
    normal = np.array([-direction[1], direction[0]])
    # This is a recheck of one observed physical shaft, not another Hough
    # search or permission to move its line towards an expected dimension.
    cross = float((old @ normal).mean())
    if np.max(np.abs(contacts @ normal-cross)) > 3.:
        return None
    rotated, to_frame, from_frame = _rotate_roi(gray[low[1]:high[1], low[0]:high[0]], direction)
    framed = to_frame(contacts-low)
    local_cross = float(framed[:, 1].mean())
    if framed[1, 0]-framed[0, 0] < 12.:
        return None
    arrows = []
    for contact, previous in zip(framed, old_arrows):
        sign = float(np.sign(np.asarray(previous["direction_px"]) @ direction))
        arrow = _dimension_station_arrow(rotated, "x", local_cross, float(contact[0]), sign, label_size)
        if arrow is None:
            return None
        narrow, middle, body, shaft, tail = arrow["cross_section_widths_px"]
        if min(shaft, tail) <= 0 or not narrow < middle <= body:
            return None
        fractions = np.arange(.10, .701, .05)
        stations = np.rint(contact[0]+arrow["sample_tip_shift_px"]-
            sign*arrow["length_px"]*fractions).astype(int)
        taper = _width_profile(rotated, local_cross, stations, max(7, int(arrow["length_px"]*.65)))
        correlation = float(np.corrcoef(fractions, taper)[0, 1]) if np.ptp(taper) > 0 else 0.
        if (not np.isfinite(correlation) or correlation < .75 or taper.min() <= 0
                or taper.max() > 1.4*body or np.diff(taper).min() < -max(2., .25*body)):
            return None
        arrows.append({**arrow, "tip_px": (from_frame([contact])[0]+low).tolist(),
            "direction_px": (direction*sign).tolist(),
            "filled_taper_profile": dict(verified=True, fractions=fractions.tolist(), widths_px=taper.tolist(),
                increasing_width_correlation=correlation, shaft_continuation_observed=True),
            "tip_station_source": "caller_finite_source_contour_contact_rechecked_in_original_pixels"})
    if np.dot(arrows[0]["direction_px"], arrows[1]["direction_px"]) > -.999:
        return None
    stations = np.arange(math.ceil(framed[0, 0]), math.floor(framed[1, 0])+1)
    support = _width_profile(rotated, local_cross, stations, 3) > 0
    if len(support) < 12 or float(support.mean()) < .90:
        return None
    return {**witness, "endpoints_source_px": contacts.tolist(), "evidence": {**proof,
        "endpoint_arrows": arrows, "shaft_supported_fraction": float(support.mean()),
        "contour_contacts_verified": False, "contact_pixel_tapers_verified": True,
        "requires_contact_tip_recheck": False,
        "source_object_ownership_requires_independent_check": True,
        "candidate_contact_source": "caller_finite_source_contours",
        "numeric_binding_certified": False, "ground_truth_used": False}}
