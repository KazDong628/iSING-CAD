"""Bounded source-ink snapping of approximate multimodal arrow locations.

The nominal radius and CAD parameters are never used to rank a location. Every
snapped hypothesis must pass the existing complete original-pixel verifier.
"""
from __future__ import annotations

import math
import time

import cv2
import numpy as np


def source_label_shaft_ownership(gray, record, tip, direction, *, inspect_parallel_family=False):
    """Reject a repetitive hatch stroke merely crossing the OCR rectangle.

    A true long radius leader may pass its text on the way to the centre, so
    continuation alone is never rejection evidence. Require both continuous
    ink through and beyond the box and at least three approximately equally
    spaced parallel source lines including this stroke. No CAD or R is used.
    """
    from .constraint_binding import _box
    result = {"method": "source_label_repetitive_crossing_stroke_v1",
              "checked": False, "repetitive_label_crossing": False,
              "nominal_used_to_rank": False, "ground_truth_used": False}
    box = _box(record)
    if gray is None or box is None:
        return result
    tip, direction = np.asarray(tip, float), np.asarray(direction, float)
    if tip.shape != (2,) or direction.shape != (2,) or not np.isfinite([tip, direction]).all():
        return result
    magnitude = float(np.linalg.norm(direction))
    if magnitude < 1e-8:
        return result
    away = -direction/magnitude
    low, high = box.min(axis=0), box.max(axis=0)
    size = max(12., float(np.linalg.norm(high-low)))
    near, far = 0., math.inf
    for axis in range(2):
        if abs(away[axis]) < 1e-8:
            if not low[axis] <= tip[axis] <= high[axis]:
                return result
        else:
            first, last = sorted(((low[axis]-tip[axis])/away[axis], (high[axis]-tip[axis])/away[axis]))
            near, far = max(near, first), min(far, last)
    if not math.isfinite(far) or far-near < max(12., .20*size):
        return result
    extension = max(18., min(64., .35*size))
    normal = np.array([-away[1], away[0]])
    points = tip+np.linspace(near, far+extension, max(8, int(math.ceil(far+extension-near))+1))[:, None]*away
    pixels = np.rint(points[:, None, :]+np.arange(-2., 3.)[None, :, None]*normal).astype(int)
    x, y = pixels[:, :, 0], pixels[:, :, 1]
    valid = (x >= 0) & (x < gray.shape[1]) & (y >= 0) & (y < gray.shape[0])
    ink = np.zeros(x.shape, bool)
    ink[valid] = gray[y[valid], x[valid]] < 170
    support = ink.any(axis=1)
    longest = running = 0
    for present in support:
        running = 0 if present else running+1
        longest = max(longest, running)
    maximum_gap = max(5., min(12., size*.10))
    result.update(checked=True, label_and_extension_support=float(support.mean()),
                  maximum_unobserved_run_px=longest, maximum_allowed_gap_px=maximum_gap,
                  extension_beyond_label_px=extension)
    through_label = bool(support.mean() >= .88 and longest <= maximum_gap)
    if not through_label and not inspect_parallel_family:
        return result
    offsets = []
    for segment in native_radius_leader_segments(gray, record):
        delta = segment[1]-segment[0]
        length = float(np.linalg.norm(delta))
        if length < max(30., .45*size) or abs(float(np.dot(delta/length, away))) < math.cos(math.radians(5.)):
            continue
        offsets.append(float(np.dot(segment.mean(axis=0)-tip, normal)))
    groups = []
    for offset in sorted(offsets):
        if groups and offset-float(np.mean(groups[-1])) <= 8.:
            groups[-1].append(offset)
        else:
            groups.append([offset])
    centers = [float(np.median(group)) for group in groups]
    anchor = min(centers, key=abs) if centers else None
    result["parallel_stroke_offsets_px"] = centers
    if anchor is None or abs(anchor) > 8.:
        return result
    others = [offset-anchor for offset in centers if abs(offset-anchor) >= max(12., .1*size)]
    family = None
    for i, first in enumerate(others):
        for second in others[i+1:]:
            pitch = min(abs(first), abs(second))
            mismatch = abs(abs(first)-abs(second)) if first*second < 0 else abs(max(abs(first), abs(second))-2*pitch)
            if mismatch <= max(5., .15*pitch):
                family = {"pitch_px": pitch, "offsets_px": [anchor, anchor+first, anchor+second],
                          "spacing_mismatch_px": mismatch}
                break
        if family is not None:
            break
    if family is not None:
        result["parallel_family"] = family
        if through_label:
            result.update(repetitive_label_crossing=True,
                          reason="repetitive_source_stroke_crosses_label_without_unique_leader_ownership")
    return result


def _source_text_pixels(gray, record):
    """Observe OCR-region ink after removing long straight source strokes.

    This is geometric text adjacency, not another OCR or a numeric parser.
    Rotated glyphs remain in their original image coordinates. A missing or
    line-only crop cannot provide ownership evidence.
    """
    from .constraint_binding import _box
    box = _box(record)
    if gray is None or gray.ndim != 2 or box is None:
        return None
    low = np.maximum(0, np.floor(box.min(axis=0))).astype(int)
    high = np.minimum([gray.shape[1], gray.shape[0]], np.ceil(box.max(axis=0))).astype(int)
    width, height = high-low
    if min(width, height) < 3 or width*height > 250_000:
        return None
    ink = np.uint8(gray[low[1]:high[1], low[0]:high[0]] < 170)*255
    polygon = np.zeros_like(ink)
    cv2.fillPoly(polygon, [np.rint(box-low).astype(np.int32)], 255)
    ink &= polygon
    # Only strokes spanning most of the OCR box are removed: a short letter
    # stem is not sufficient evidence that it is a leader or hatch stroke.
    diagonal = float(np.linalg.norm(high-low))
    segments = cv2.HoughLinesP(ink, 1., np.pi/360, threshold=max(20, int(.18*diagonal)),
                              minLineLength=max(24., .65*diagonal), maxLineGap=6)
    cleaned = ink.copy()
    for line in ([] if segments is None else segments[:32]):
        x, y, end_x, end_y = map(int, line[0])
        cv2.line(cleaned, (x, y), (end_x, end_y), 0, 5)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, 8)
    selected = np.zeros_like(cleaned, bool)
    for index in range(1, count):
        component_width, component_height, area = stats[index, 2:5]
        if area >= 4 and min(component_width, component_height) >= 2:
            selected |= labels == index
    yy, xx = np.where(selected)
    if len(xx) < 20:
        return None
    points = np.c_[xx+low[0], yy+low[1]].astype(float)
    # Bound diagnostic memory and cost without favouring a particular shaft.
    if len(points) > 4096:
        points = points[np.linspace(0, len(points)-1, 4096, dtype=int)]
    return points


def source_arrow_label_attachment(gray, record, observation, *, text_points=None):
    """Measure source text/shaft adjacency after the complete shaft verifier.

    Close text alone is never a binding certificate. These measurements may
    reject a conflicting or repetitive claim, but cannot create an arrow.
    """
    from .constraint_binding import _box, _label_ray_entry
    result = {"method": "source_glyph_directed_shaft_adjacency_v1", "checked": False,
              "nominal_used_to_rank": False, "ground_truth_used": False}
    box = _box(record)
    try:
        arrow = observation["arrowhead"]
        tip = np.asarray(arrow["tip_px"], float)
        unit = np.asarray(arrow["direction_px"], float)
    except (KeyError, TypeError, ValueError):
        return result
    if (gray is None or gray.ndim != 2 or box is None or tip.shape != (2,) or
            unit.shape != (2,) or not np.isfinite([tip, unit]).all()):
        return result
    magnitude = float(np.linalg.norm(unit))
    if magnitude < 1e-8:
        return result
    unit /= magnitude
    low, high = box.min(axis=0), box.max(axis=0)
    ray = _label_ray_entry(tip, -unit, low-2., high+2., float(max(gray.shape))*2)
    result.update(directed_ray_reaches_text=ray is not None,
                  arrow_tip_inside_text_box=bool(np.all(tip >= low) and np.all(tip <= high)),
                  label_to_tip_gap_px=ray,
                  full_source_shaft_verified=(observation.get("shaft_evidence") or {}).get("verified") is True)
    points = _source_text_pixels(gray, record) if text_points is None else text_points
    if points is None:
        result["reason"] = "source_text_ink_unavailable"
        return result
    normal = np.array([-unit[1], unit[0]])
    distances = np.abs((points-tip)@normal)
    behind = (points-tip)@(-unit)
    scale = max(12., float(min(high-low)))
    cost = float(np.median(distances))/scale
    result.update(checked=True, source_text_sample_count=len(points),
                  median_text_to_shaft_px=float(np.median(distances)),
                  text_adjacency_normalization_px=scale, normalized_text_to_shaft=cost,
                  text_behind_arrow_fraction=float(np.mean(behind >= 0)),
                  strong_text_adjacency=bool(cost <= .35 and np.mean(behind >= 0) >= .9))
    return result


def recover_source_attached_radius_arrow(gray, record, seed, boundary_points, band,
                                         contours=None, *, verifier=None):
    """Revisit one weak source tip using OCR-facing ink, never an R or CAD fit.

    A localizer may select an annotation/extension stroke beside the actual
    radius leader. Search a bounded one-pixel tip neighbourhood and the OCR
    edge facing it; every survivor passes the unchanged full arrow and shaft
    verifier. This is a fallback only when the existing tip lacks a close
    original-ink contour intersection or glyph attachment.
    """
    from scipy.spatial import cKDTree
    from .constraint_binding import _box
    if verifier is None:
        from .constraint_binding import verify_source_arrow_proposal
        verifier = verify_source_arrow_proposal
    if (gray is None or gray.ndim != 2 or record.get("parsed", {}).get("kind") != "radius"
            or not seed.get("arrowhead_verified") or
            not (seed.get("shaft_evidence") or {}).get("verified")):
        return None
    box = _box(record)
    ink = _source_text_pixels(gray, record)
    try:
        tip = np.asarray(seed["arrowhead"]["tip_px"], float)
        direction = np.asarray(seed["arrowhead"]["direction_px"], float)
        boundary = np.asarray(boundary_points, float)
    except (KeyError, TypeError, ValueError):
        return None
    if (box is None or ink is None or tip.shape != (2,) or direction.shape != (2,) or
            boundary.ndim != 2 or boundary.shape[1] != 2 or not len(boundary) or
            not np.isfinite(tip).all() or not np.isfinite(direction).all() or
            not np.isfinite(boundary).all()):
        return None
    magnitude = float(np.linalg.norm(direction))
    if magnitude < 1e-8:
        return None
    direction /= magnitude
    low, high = box.min(axis=0), box.max(axis=0)
    width = float(min(high-low))
    if width < 8:
        return None
    # The closest facing OCR-box edge is determined from the observed source
    # tip alone. Source text and source ink rank the search, not the nominal.
    outside = np.maximum(low-tip, 0.)+np.maximum(tip-high, 0.)
    axis = int(np.argmax(outside))
    if outside[axis] < 4.:
        return None
    side = high[axis] if tip[axis] > high[axis] else low[axis]
    sign = 1. if tip[axis] > high[axis] else -1.
    cross = 1-axis
    across = np.arange(math.floor(low[cross]), math.ceil(high[cross])+1, 4, dtype=float)
    stations = side+sign*np.array([4., 7., 10.])
    text_points = ink[np.linspace(0, len(ink)-1, min(512, len(ink)), dtype=int)]
    tree = cKDTree(boundary)
    width_observed = max((float(value) for value in
                          (seed.get("arrowhead") or {}).get("cross_section_widths_px", [])
                          if isinstance(value, (int, float)) and math.isfinite(value)), default=8.)
    tip_shift = int(round(max(6., min(8., .75*width_observed))))
    options = []
    for x in range(int(round(tip[0]))-tip_shift, int(round(tip[0]))+tip_shift+1):
        for y in range(int(round(tip[1]))-tip_shift, int(round(tip[1]))+tip_shift+1):
            target = np.array([float(x), float(y)])
            gap = float(tree.query(target)[0])
            if gap > 3.:
                continue
            for station in stations:
                for offset in across:
                    shaft = np.empty(2, float)
                    shaft[axis], shaft[cross] = station, offset
                    delta = target-shaft
                    length = float(np.linalg.norm(delta))
                    if length < 8.:
                        continue
                    unit = delta/length
                    if float(np.dot(unit, direction)) < math.cos(math.radians(40.)):
                        continue
                    normal = np.array([-unit[1], unit[0]])
                    attachment = float(np.median(np.abs((text_points-target) @ normal)))/width
                    if attachment > .38:
                        continue
                    points = shaft+np.linspace(0., 1., 16)[:, None]*delta
                    strip = np.rint(points[:, None, :]+np.arange(-2., 3.)[None, :, None]*normal).astype(int)
                    px, py = strip[:, :, 0], strip[:, :, 1]
                    inside = (px >= 0) & (px < gray.shape[1]) & (py >= 0) & (py < gray.shape[0])
                    source = gray[np.clip(py, 0, gray.shape[0]-1), np.clip(px, 0, gray.shape[1]-1)] < 170
                    if float(np.mean(np.any(inside & source, axis=1))) < .875:
                        continue
                    options.append((gap, attachment, shaft.tolist(), target.tolist()))
    if not options:
        return None
    options.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
    verified = []
    for gap, attachment, shaft, target in options[:256]:
        proposal = {"record_id": record.get("id"), "shaft_px": shaft, "tip_px": target}
        evidence = verifier(gray, record, proposal, boundary, band, contours)
        if evidence is None:
            continue
        source_attachment = source_arrow_label_attachment(gray, record, evidence, text_points=ink)
        first = (evidence.get("contour_visibility") or {}).get("first_intersection_to_target_px")
        if (not source_attachment.get("strong_text_adjacency") or first is None or first > 1. or
                float(evidence.get("arrow_tip_to_arc_gap_px", math.inf)) > 3.):
            continue
        verified.append((float(first), source_attachment["normalized_text_to_shaft"],
                         float(evidence["arrow_tip_to_arc_gap_px"]), evidence, proposal))
    if not verified:
        return None
    first, attachment, gap, evidence, proposal = min(verified, key=lambda row: row[:3])
    return {**evidence, "method": "source_ocr_edge_arrow_with_full_shaft_locally_verified",
            "proposal_origin": "bounded_original_source_ocr_edge_search",
            "model_proposal_used": False,
            "source_pixel_localization": {"method": "bounded_original_source_ocr_edge_search",
                                          "seed_tip_px": tip.tolist(),
                                          "verified_full_shaft_and_arrow": True,
                                          "strong_source_text_adjacency": True,
                                          "first_source_contour_intersection_gap_px": first,
                                          "source_tip_to_current_contour_px": gap,
                                          "searched_coordinate_count": len(options),
                                          "full_verification_limit": 256,
                                          "nominal_used_to_rank": False,
                                          "ground_truth_used": False}}


def _same_source_arrow(first, second, band):
    """Group one physical taper, requiring direction AND shaft agreement."""
    try:
        a, b = first["arrowhead"], second["arrowhead"]
        p, q = np.asarray(a["tip_px"], float), np.asarray(b["tip_px"], float)
        u, v = np.asarray(a["direction_px"], float), np.asarray(b["direction_px"], float)
        u /= np.linalg.norm(u); v /= np.linalg.norm(v)
        radius = max(4., min(12., .4*min(float(a.get("length_px", 0)), float(b.get("length_px", 0)))))
        delta = q-p
        lateral = max(abs(float(u[0]*delta[1]-u[1]*delta[0])),
                      abs(float(v[0]*delta[1]-v[1]*delta[0])))
        return bool(np.dot(u, v) >= math.cos(math.radians(8.)) and np.linalg.norm(p-q) <= radius
                    and lateral <= max(3., min(4., .5*float(band))))
    except (KeyError, TypeError, ValueError, FloatingPointError):
        return False


def same_original_ink_arrow_shaft(gray, first, second):
    """Prove two nearby tip hypotheses follow one wide source-ink shaft.

    Check the observed arrow body, not the narrowing tip wedge. The source
    arrow verifier measures middle and body widths at .42 and .72 of its
    length; this check samples within those same stations. A successful proof
    may group duplicate detections, while preserving every target hypothesis.
    """
    if gray is None or gray.ndim != 2:
        return False
    try:
        a, b = first["arrowhead"], second["arrowhead"]
        p, q = np.asarray(a["tip_px"], float), np.asarray(b["tip_px"], float)
        u, v = np.asarray(a["direction_px"], float), np.asarray(b["direction_px"], float)
        if (not all(arr.shape == (2,) and np.isfinite(arr).all() for arr in (p, q, u, v)) or
                min(np.linalg.norm(u), np.linalg.norm(v)) < 1e-8):
            return False
        u /= np.linalg.norm(u); v /= np.linalg.norm(v)
        lengths = [float(a.get("length_px", 0)), float(b.get("length_px", 0))]
        length = min(lengths)
        if length < 16. or float(u@v) < math.cos(math.radians(8.)):
            return False
        widths = []
        for arrow in (a, b):
            values = [float(value) for value in arrow.get("cross_section_widths_px") or []
                      if isinstance(value, (int, float)) and not isinstance(value, bool)
                      and math.isfinite(value) and value > 0]
            if not values:
                return False
            widths.append(max(values))
        lateral = max(abs(float(np.cross(u, q-p))), abs(float(np.cross(v, q-p))))
        if (lateral > .65*min(widths) or np.linalg.norm(q-p) > min(8., .75*min(widths))):
            return False
        # Two separate parallel ink strokes can have matching direction and
        # nearby tips. A single rounded pixel on a slanted, antialiased shaft
        # is not a white channel, so sample a one-pixel axial band and require
        # a connected ink path across each section. The .42 and .72 stations
        # are the observed middle/body widths from _arrowhead_evidence; .30
        # may still be inside the taper where the two detected edges diverge.
        axis = u+v
        axis /= np.linalg.norm(axis)
        for fraction in (.42, .52, .62, .72):
            depth = fraction*length
            ends = (p-depth*u, q-depth*v)
            span = np.linspace(*ends, max(9, int(math.ceil(lateral*2))+1))
            points = np.rint(span[None, :, :] +
                              np.asarray([-1., 0., 1.])[:, None, None]*axis).astype(int)
            x, y = points[:, :, 0], points[:, :, 1]
            if (np.any(x < 0) or np.any(x >= gray.shape[1]) or
                    np.any(y < 0) or np.any(y >= gray.shape[0])):
                return False
            dark = gray[y, x] < 170
            if float(np.mean(np.any(dark, axis=0))) < .9:
                return False
            # Column support alone could join two dark parallel shafts across
            # a persistent white slit. The original pixels must also form an
            # eight-connected path across this narrow 2-D source-ink band.
            components, labels = cv2.connectedComponents(dark.astype(np.uint8), connectivity=8)
            if not any(np.any(labels[:, 0] == label) and np.any(labels[:, -1] == label)
                       for label in range(1, components)):
                return False
        return True
    except (KeyError, TypeError, ValueError, FloatingPointError):
        return False


def resolve_source_arrow_ownership(gray, records, observations, *, band=4.):
    """Resolve complete physical-arrow claims globally using source ink only.

    Two nearby genuine labels remain unresolved if both are adjacent to the
    same shaft. A winner requires full directed shaft proof, actual text ink,
    a poor competing attachment and a declared margin. Separate arrows from
    one label are retained, including arrows to different contour objects.
    """
    import copy
    rows = {row["id"]: row for row in records}
    if len(observations) > 1024:
        rejected = [{"record_id": key, "reason": "source_arrow_ownership_budget_exceeded"}
                    for key in sorted({row.get("record_id") for row in observations if row.get("record_id")})]
        return [], rejected, {"method": "global_complete_source_arrow_ownership_v1", "status": "budget_exceeded",
            "input_observation_count": len(observations), "maximum_observations": 1024,
            "accepted_observation_count": 0, "ground_truth_used": False, "nominal_used_to_rank": False}
    claimed = {row.get("record_id") for row in observations}
    pixels = {key: _source_text_pixels(gray, row) for key, row in rows.items() if key in claimed}
    accepted, rejected, groups = [], [], []
    for original in observations:
        row = copy.deepcopy(original)
        record = rows.get(row.get("record_id"))
        if record is None:
            rejected.append({"record_id": row.get("record_id"), "reason": "source_label_missing"})
            continue
        attachment = source_arrow_label_attachment(gray, record, row, text_points=pixels[record["id"]])
        row["source_text_shaft_attachment"] = attachment
        reason = None
        if not attachment.get("full_source_shaft_verified"):
            reason = "complete_source_shaft_not_verified"
        elif not attachment.get("directed_ray_reaches_text"):
            reason = "source_arrow_direction_does_not_reach_label"
        elif attachment.get("arrow_tip_inside_text_box"):
            reason = "source_arrow_tip_inside_text_requires_review"
        else:
            ownership = row.get("source_label_association") or {}
            # Inspect parallel families only for the surviving poor glyph
            # attachments. Running native Hough for every discarded detector
            # hypothesis would repeat the same expensive crop unnecessarily.
            if (attachment.get("checked") and not attachment.get("strong_text_adjacency")
                    and not ownership.get("parallel_family")):
                ownership = source_label_shaft_ownership(gray, record, row["arrowhead"]["tip_px"],
                    row["arrowhead"]["direction_px"], inspect_parallel_family=True)
                row["source_label_association"] = ownership
            if (ownership.get("parallel_family") and attachment.get("checked") and
                    not attachment.get("strong_text_adjacency")):
                reason = "repetitive_shaft_without_source_glyph_adjacency"
        if reason:
            rejected.append({"record_id": row["record_id"], "reason": reason,
                             "previous_evidence_method": row.get("method"),
                             "arrowhead": row.get("arrowhead"), "source_text_shaft_attachment": attachment,
                             "source_label_association": row.get("source_label_association")})
            continue
        # Complete-link grouping prevents a series of neighboring arrowheads
        # from becoming one taper merely through transitive proximity.
        # Repeated preflight may carry a verified shaft back into the graph.
        # Its two edge detections can then differ laterally by a few pixels,
        # exceeding the narrow same-tip rule despite following one broad,
        # continuous original-ink stroke. Require the independent full-ink
        # bridge at several shaft depths before treating them as one arrow;
        # target alternatives are preserved in the group union below.
        group = next((g for g in groups if all(
            _same_source_arrow(row, old, band) or
            same_original_ink_arrow_shaft(gray, row, old) for old in g)), None)
        if group is None:
            groups.append([row])
        else:
            group.append(row)
    conflicts = []
    for index, group in enumerate(groups):
        physical_id = f"source-arrow-{index:03d}"
        by_record = {}
        for row in group:
            key = row["record_id"]
            source_crossing = (row.get("contour_visibility") or {}).get("first_intersection_to_target_px")
            cost = (source_crossing is None or source_crossing > 1.,
                    row["source_text_shaft_attachment"].get("normalized_text_to_shaft", math.inf))
            old = by_record.get(key)
            old_crossing = (old.get("contour_visibility") or {}).get("first_intersection_to_target_px") if old else None
            old_cost = (old_crossing is None or old_crossing > 1.,
                        (old.get("source_text_shaft_attachment") or {}).get("normalized_text_to_shaft", math.inf)) if old else None
            if old is None or cost < old_cost:
                by_record[key] = row
        choices = sorted(by_record.values(), key=lambda row: (
            row["source_text_shaft_attachment"].get("normalized_text_to_shaft", math.inf), row["record_id"]))
        winner = choices[0] if len(choices) == 1 else None
        if len(choices) > 1:
            best, runner = choices[:2]
            first, second = best["source_text_shaft_attachment"], runner["source_text_shaft_attachment"]
            gap = second.get("normalized_text_to_shaft", math.inf)-first.get("normalized_text_to_shaft", math.inf)
            resolved = (all(row["source_text_shaft_attachment"].get("checked") for row in choices)
                        and first.get("strong_text_adjacency") is True
                        and not any(row["source_text_shaft_attachment"].get("strong_text_adjacency") for row in choices[1:])
                        and math.isfinite(gap) and gap > .15)
            if resolved:
                winner = best
            conflicts.append({"physical_arrow_id": physical_id,
                "record_ids": [row["record_id"] for row in choices],
                "status": "source_ownership_resolved" if resolved else "source_ownership_ambiguous",
                "selected_record_id": winner["record_id"] if winner else None,
                "required_normalized_margin": .15, "observed_normalized_margin": gap if math.isfinite(gap) else None,
                "attachments": [{"record_id": row["record_id"], **row["source_text_shaft_attachment"]} for row in choices]})
        for row in choices:
            if row is winner:
                # Duplicate observations of one physical arrow may disagree
                # about which contour object its tip reaches. Deduplicate the
                # arrow, never erase that target ambiguity by picking the row
                # with the most convenient glyph distance.
                targets = {}
                duplicates = [old for old in group if old["record_id"] == row["record_id"]]
                for duplicate in duplicates:
                    for target in duplicate.get("target_candidates") or []:
                        key = target.get("entity_id")
                        old = targets.get(key)
                        if key is not None and (old is None or float(target.get("tip_gap_px", math.inf)) <
                                                float(old.get("tip_gap_px", math.inf))):
                            targets[key] = copy.deepcopy(target)
                row["target_candidates"] = sorted(targets.values(), key=lambda target: (
                    float(target.get("tip_gap_px", math.inf)), target["entity_id"]))
                row["source_arrow_ownership"] = {"physical_arrow_id": physical_id,
                    "status": "globally_unique_source_claim", "competing_record_ids": sorted(by_record),
                    "duplicate_observation_count": len(duplicates)-1,
                    "duplicate_target_ambiguity_preserved": len(targets) > 1,
                    "nominal_used_to_rank": False, "ground_truth_used": False}
                accepted.append(row)
            else:
                rejected.append({"record_id": row["record_id"], "physical_arrow_id": physical_id,
                    "reason": "source_arrow_claim_owned_by_other_label" if winner else "shared_source_arrow_ownership_ambiguous",
                    "selected_record_id": winner["record_id"] if winner else None,
                    "arrowhead": row["arrowhead"], "source_text_shaft_attachment": row["source_text_shaft_attachment"],
                    "previous_evidence_method": row.get("method")})
    audit = {"method": "global_complete_source_arrow_ownership_v1", "input_observation_count": len(observations),
             "physical_arrow_count": len(groups), "accepted_observation_count": len(accepted),
             "rejected_claim_count": len(rejected), "shared_arrow_conflicts": conflicts,
             "record_observation_counts_before": {key: sum(row.get("record_id") == key for row in observations)
                                                  for key in sorted(claimed) if key is not None},
             "record_observation_counts_after": {key: sum(row.get("record_id") == key for row in accepted)
                                                 for key in sorted(claimed) if key is not None},
             "nominal_used_to_rank": False, "ground_truth_used": False,
             "scope": "Original text ink, directed complete shafts and physical arrow ownership only; no target radius or GT."}
    return accepted, rejected, audit


def native_radius_leader_segments(gray, record, records=(), *, limit=96):
    """Observe short label-adjacent strokes without shrinking the source image.

    These are detection hypotheses only. They must subsequently pass the full
    source arrow, label ray and shaft verifier, including unmasked text strokes.
    """
    from .constraint_binding import _box
    if gray is None or gray.ndim != 2 or record.get("parsed", {}).get("kind") != "radius":
        return []
    box = _box(record)
    if box is None:
        return []
    low, high = box.min(axis=0), box.max(axis=0)
    padding = min(384., max(48., float(np.linalg.norm(high-low))))
    left, top = np.maximum(0, np.floor(low-padding)).astype(int)
    right, bottom = np.minimum([gray.shape[1], gray.shape[0]], np.ceil(high+padding)).astype(int)
    if right <= left or bottom <= top or (right-left)*(bottom-top) > 2_000_000:
        return []
    original = gray[top:bottom, left:right]
    masked = original.copy()
    for row in records or (record,):
        other = _box(row)
        if other is not None:
            cv2.fillPoly(masked, [np.rint(other-[left, top]).astype(np.int32)], 255)
    segments = []
    for observed in (masked, original):
        raw = cv2.HoughLinesP(cv2.Canny(observed, 70, 180), 1., np.pi/720,
                              threshold=22, minLineLength=22, maxLineGap=6)
        for item in ([] if raw is None else raw[:500]):
            points = np.asarray(item[0], float).reshape(2, 2)+[left, top]
            if not any(np.max(np.linalg.norm(points-old, axis=1)) < 1.5 or
                       np.max(np.linalg.norm(points[::-1]-old, axis=1)) < 1.5 for old in segments):
                segments.append(points)
    return sorted(segments, key=lambda pair: -float(np.linalg.norm(pair[1]-pair[0])))[:min(96, max(1, limit))]


def verify_source_hough_leader(gray, record, segment, boundary_points, band, contours=None, *, verifier=None):
    """Apply exactly the proposal channel's full pixel proof to a Hough stroke."""
    if gray is None or gray.ndim != 2:
        return None
    if verifier is None:
        from .constraint_binding import verify_source_arrow_proposal
        verifier = verify_source_arrow_proposal
    try:
        points = np.asarray(segment, float)
    except (TypeError, ValueError):
        return None
    if points.shape != (2, 2) or not np.isfinite(points).all():
        return None
    proposal = {"record_id": record.get("id"), "shaft_px": points[0].tolist(), "tip_px": points[1].tolist()}
    evidence = verifier(gray, record, proposal, boundary_points, band, contours)
    if evidence is None:
        return None
    return {**evidence, "method": "source_hough_arrow_with_full_shaft_locally_verified",
            "proposal_origin": "source_hough", "model_proposal_used": False,
            "crossing_admission": "detected_directed_source_arrow_with_full_shaft_and_label_ray"}


def localize_ocr_radius_arrows(gray, record, records, boundary_points, band, contours=None, *,
                              verifier=None, seed_limit=24, time_limit_seconds=8.):
    """Snap bounded OCR-local ink seeds independently of provider proposals.

    An approximate Hough endpoint can lie on the arrow body or one shaft edge.
    Reuse the same bounded pixel snap as a model seed, followed by unchanged
    complete shaft/arrow and actual-glyph attachment checks. Distinct survivors
    remain separate observations for the global ownership/target audit.
    """
    from .constraint_binding import _box, _label_ray_entry
    started = time.monotonic()
    maximum_seeds = min(24, max(0, int(seed_limit)))
    maximum_seconds = min(8., max(0., float(time_limit_seconds)))
    audit = {"method": "bounded_ocr_local_source_seeds_v1", "record_id": record.get("id"),
             "status": "not_run", "maximum_seed_count": maximum_seeds,
             "maximum_hypotheses_per_seed": 24, "maximum_native_segments": 96,
             "maximum_roi_pixels": 2_000_000, "time_limit_seconds": maximum_seconds,
             "attempted_seed_count": 0, "accepted_observation_count": 0,
             "candidate_seed_count": 0, "seed_attempts": [],
             "api_proposal_used": False, "nominal_used_to_rank": False, "ground_truth_used": False}
    def finish(result, status):
        exhausted = status in {"time_budget_exhausted", "seed_budget_exhausted"}
        audit.update(status=status, verified_observation_count=len(result),
                     accepted_observation_count=0 if exhausted else len(result),
                     uninspected_seed_count=max(0, audit["candidate_seed_count"]-audit["attempted_seed_count"]),
                     elapsed_seconds=round(time.monotonic()-started, 6))
        if exhausted:
            # Unvisited seeds may contain a competing physical arrow. A valid
            # partial observation is diagnostic, never a uniqueness proof.
            audit.update(acceptance_withheld_reason="uninspected_source_seed_competitors",
                         diagnostic_only_observations=result)
            return [], audit
        return result, audit
    box = _box(record)
    boundary = np.asarray(boundary_points, float)
    if (gray is None or gray.ndim != 2 or box is None or
            record.get("parsed", {}).get("kind") != "radius" or
            boundary.ndim != 2 or boundary.shape[1] != 2 or not len(boundary) or
            not np.isfinite(boundary).all()):
        return finish([], "invalid_source_input")
    text_points = _source_text_pixels(gray, record)
    if text_points is None:
        return finish([], "source_text_ink_unavailable")
    low, high = box.min(axis=0), box.max(axis=0)
    size = max(12., float(np.linalg.norm(high-low)))
    padding = min(384., max(48., size))
    left, top = np.maximum(0, np.floor(low-padding)).astype(int)
    right, bottom = np.minimum([gray.shape[1], gray.shape[0]], np.ceil(high+padding)).astype(int)
    audit["seed_roi_px"] = [int(left), int(top), int(right), int(bottom)]
    prelocalization_band = max(10., band*1.7)+.65*max(7., min(90., size*.52))
    # Boundary proximity only bounds hypotheses. A current fitted radius or an
    # OCR numeric value never ranks seeds or establishes their acceptance.
    from scipy.spatial import cKDTree
    tree = cKDTree(boundary)
    seeds = []
    for segment in native_radius_leader_segments(gray, record, records):
        for shaft, tip in (segment, segment[::-1]):
            label_gap = float(np.linalg.norm(np.maximum(low-shaft, 0.)+np.minimum(high-shaft, 0.)))
            direction = tip-shaft
            length = float(np.linalg.norm(direction))
            if label_gap > max(18., size*.75) or length < 6.:
                continue
            direction /= length
            ray_gap = _label_ray_entry(shaft, -direction, low-2., high+2., max(18., size*.75))
            target_gap = float(tree.query(tip)[0])
            if ray_gap is None or target_gap > prelocalization_band:
                continue
            proposal = {"record_id": record.get("id"), "shaft_px": shaft.tolist(), "tip_px": tip.tolist()}
            seeds.append((label_gap+.5*target_gap, proposal))
    seeds.sort(key=lambda item: (item[0], item[1]["shaft_px"], item[1]["tip_px"]))
    audit["candidate_seed_count"] = len(seeds)
    accepted = []
    for _, proposal in seeds[:maximum_seeds]:
        if time.monotonic()-started >= maximum_seconds:
            return finish(accepted, "time_budget_exhausted")
        audit["attempted_seed_count"] += 1
        receipt = {"seed_segment_px": [proposal["shaft_px"], proposal["tip_px"]]}
        evidence = localize_source_arrow_proposal(gray, record, proposal, boundary, band, contours,
                                                 verifier=verifier)
        if evidence is None:
            receipt["status"] = "full_source_verification_failed"
        else:
            attachment = source_arrow_label_attachment(gray, record, evidence, text_points=text_points)
            receipt["text_shaft_attachment"] = attachment
            if not attachment.get("strong_text_adjacency"):
                receipt["status"] = "source_text_shaft_attachment_insufficient"
            else:
                receipt.update(status="locally_verified_pending_global_ownership",
                               tip_px=evidence["arrowhead"]["tip_px"])
                accepted.append({**evidence, "method": "ocr_local_source_arrow_with_full_shaft_verified",
                    "proposal_origin": "bounded_ocr_local_source_seeds", "model_proposal_used": False,
                    "source_text_shaft_attachment": attachment,
                    "source_seed": {"segment_px": receipt["seed_segment_px"],
                                    "api_proposal_used": False, "nominal_used_to_rank": False}})
        audit["seed_attempts"].append(receipt)
    return finish(accepted, "seed_budget_exhausted" if len(seeds) > maximum_seeds else "completed")


def source_arrow_hypotheses(gray, record, proposal, *, limit=24):
    """Find nearby native-resolution shaft strokes, retaining tight pixel bounds."""
    from .constraint_binding import _box, _label_ray_entry
    if (record.get("parsed", {}).get("kind") != "radius" or not isinstance(proposal, dict) or
            proposal.get("record_id", record.get("id")) != record.get("id")):
        return []
    box = _box(record)
    try:
        tip = np.asarray(proposal.get("tip_px"), float)
        shaft = np.asarray(proposal.get("shaft_px"), float)
    except (TypeError, ValueError, AttributeError):
        return []
    if (box is None or tip.shape != (2,) or shaft.shape != (2,) or
            not np.isfinite([tip, shaft]).all() or gray is None or gray.ndim != 2):
        return []
    if any(not (0 <= p[0] < gray.shape[1] and 0 <= p[1] < gray.shape[0]) for p in (tip, shaft)):
        return []
    direction = tip - shaft
    length = float(np.linalg.norm(direction))
    if length < 6.:
        return []
    direction /= length
    low, high = box.min(axis=0), box.max(axis=0)
    label_size = max(12., float(np.linalg.norm(high-low)))
    maximum_shift = min(48., max(16., .35*label_size))
    pad = maximum_shift + 12.
    left, top = np.maximum(0, np.floor(np.minimum(low, tip)-pad)).astype(int)
    right, bottom = np.minimum([gray.shape[1], gray.shape[0]], np.ceil(np.maximum(high, tip)+pad)).astype(int)
    if right <= left or bottom <= top or (right-left)*(bottom-top) > 2_000_000:
        return []
    crop = gray[top:bottom, left:right].copy()
    unmasked = crop.copy()
    # Text strokes must not become the replacement shaft. The OCR rectangle
    # can cover part of a diagonal leader, so extrapolation is still verified.
    cv2.fillPoly(crop, [np.rint(box-[left, top]).astype(np.int32)], 255)
    segments = []
    # An axis-aligned OCR box around rotated text may also contain the real
    # leader. Retain an unmasked detector pass as hypotheses only; letter
    # strokes still cannot pass the complete arrow/shaft verifier by assertion.
    for observed in (crop, unmasked):
        raw_segments = cv2.HoughLinesP(cv2.Canny(observed, 70, 180), 1., np.pi/1440,
                                      threshold=12, minLineLength=12, maxLineGap=5)
        if raw_segments is not None:
            segments.extend(raw_segments[:500])
    if not segments:
        return []
    choices = []
    for raw in segments:
        a, b = np.asarray(raw[0], float).reshape(2, 2)+[left, top]
        unit = b-a
        segment_length = float(np.linalg.norm(unit))
        if segment_length < 12.:
            continue
        unit /= segment_length
        if np.dot(unit, direction) < 0:
            unit = -unit
        alignment = float(np.dot(unit, direction))
        if alignment < math.cos(math.radians(24.)):
            continue
        projected = a + unit*np.dot(tip-a, unit)
        lateral_shift = float(np.linalg.norm(projected-tip))
        if lateral_shift > maximum_shift:
            continue
        along = float(np.dot(projected-a, (b-a)/segment_length))
        if along < -maximum_shift or along > segment_length + maximum_shift:
            continue
        ray = _label_ray_entry(projected, -unit, low-2., high+2., length+2*label_size)
        if ray is None or ray < 6.:
            continue
        label_exit = projected-unit*ray
        snapped_shaft = label_exit + unit*min(12., max(3., .08*label_size))
        if float(np.linalg.norm(snapped_shaft-shaft)) > label_size:
            continue
        # The original tip estimate and observed endpoints provide bounded
        # alternatives; downstream taper verification decides the exact tip.
        targets = [projected, *sorted((a, b), key=lambda point: float(np.linalg.norm(point-tip)))]
        for snapped_tip in targets:
            tip_shift = float(np.linalg.norm(snapped_tip-tip))
            if tip_shift > maximum_shift or np.dot(snapped_tip-snapped_shaft, unit) < 6.:
                continue
            candidate = {"tip_px": snapped_tip.tolist(), "shaft_px": snapped_shaft.tolist()}
            score = tip_shift/maximum_shift + (1-alignment)*4 + 1/max(segment_length, 1)
            choices.append((score, candidate, {"method": "native_source_hough_shaft_snap",
                "observed_segment_px": [a.tolist(), b.tolist()],
                "original_proposal": {"record_id": record.get("id"), "tip_px": tip.tolist(), "shaft_px": shaft.tolist()},
                "tip_shift_px": tip_shift, "maximum_tip_shift_px": maximum_shift,
                "shaft_shift_px": float(np.linalg.norm(snapped_shaft-shaft)),
                "direction_change_deg": math.degrees(math.acos(min(1., max(-1., alignment)))),
                "nominal_used_to_rank": False, "ground_truth_used": False}))
    result = []
    for _, candidate, audit in sorted(choices, key=lambda row: row[0]):
        if any(math.dist(candidate["tip_px"], old["proposal"]["tip_px"]) < 1.5 and
               math.dist(candidate["shaft_px"], old["proposal"]["shaft_px"]) < 1.5 for old in result):
            continue
        result.append({"proposal": candidate, "localization": audit})
        if len(result) >= min(24, max(1, limit)):
            break
    return result


def localize_source_arrow_proposal(gray, record, proposal, boundary_points, band, contours=None, *, verifier=None):
    """Verify directly, then search bounded ink corrections without weaker gates."""
    if verifier is None:
        from .constraint_binding import verify_source_arrow_proposal
        verifier = verify_source_arrow_proposal
    direct = verifier(gray, record, proposal, boundary_points, band, contours)
    if direct is not None:
        return direct
    accepted = []
    for candidate in source_arrow_hypotheses(gray, record, proposal):
        evidence = verifier(gray, record, candidate["proposal"], boundary_points, band, contours)
        if evidence is not None:
            accepted.append({**evidence, "source_pixel_localization": candidate["localization"]})
    if not accepted:
        return None
    # More than one distinct arrow endpoint is unresolved; do not select by R
    # or by whichever proposed target makes the downstream geometry easier.
    best = min(accepted, key=lambda row: (float(row["score"]),
               row["source_pixel_localization"]["tip_shift_px"]))
    def same_arrow(row):
        # The taper detector samples several sections of one filled arrow;
        # opposite Hough shaft edges can move its reported tip within that
        # same observed arrowhead. This groups observations only, not gates.
        target_band = max(10., float(band)*1.7, .5*max(
            float(best["arrowhead"].get("length_px", 0)), float(row["arrowhead"].get("length_px", 0))))
        return math.dist(best["arrowhead"]["tip_px"], row["arrowhead"]["tip_px"]) <= target_band
    if any(not same_arrow(row) for row in accepted):
        return None
    best["source_pixel_localization"]["locally_verified_alternative_count"] = len(accepted)
    return best
