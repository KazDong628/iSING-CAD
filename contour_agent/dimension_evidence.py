"""Image measurement scale from dimension lines and OCR evidence, without case tables.

Diameter dimensions on a half section encode radial positions D/2. Multiple
independent labels must agree on one pixel/mm scale before it is considered usable.
An independent linear-dimension fallback estimates scale without an axis origin.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
import cv2
import numpy as np
from .ocr import canonical_records


def _axis_lines(gray: np.ndarray, axis: str, *, minimum_span: int | None = None) -> list[dict]:
    h, w = gray.shape
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    extent = w if axis == "x" else h
    length = max(25, int(extent * .025)) if minimum_span is None else max(5, int(minimum_span))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1) if axis == "x" else (1, length))
    opened = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)
    n, _, stats, _ = cv2.connectedComponentsWithStats(opened, 8)
    result = []
    for x, y, sw, sh, area in stats[1:]:
        span, thickness = (sw, sh) if axis == "x" else (sh, sw)
        if span < length or thickness > max(12, extent * .01):
            continue
        lo, hi, cross = (x, x + sw - 1, y + (sh - 1) / 2) if axis == "x" else (y, y + sh - 1, x + (sw - 1) / 2)
        result.append({"lo": float(lo), "hi": float(hi), "cross": float(cross), "span": float(span), "thickness": int(thickness)})
    return result


def _label_span(line, along, cross_size, crossing_lines):
    """Separate a chained dimension stroke using observed extension stations.

    This operates only on source geometry. The nominal value and any candidate
    scale are deliberately unavailable here: they cannot choose the endpoints.
    Intersections inside the label's projected box are not segment boundaries.
    """
    result = dict(line)
    margin = max(3., float(line.get("thickness", 1)) * 1.5)
    stations = [other for other in crossing_lines
                if other["lo"] - margin <= line["cross"] <= other["hi"] + margin
                and line["lo"] - margin <= other["cross"] <= line["hi"] + margin
                and other["span"] > cross_size * 2]
    before = [other for other in stations if other["cross"] < float(min(along)) - margin]
    after = [other for other in stations if other["cross"] > float(max(along)) + margin]
    supports = {}
    if before:
        supports["lo"] = max(before, key=lambda other: other["cross"])
    if after:
        supports["hi"] = min(after, key=lambda other: other["cross"])
    for endpoint, other in supports.items():
        # Do not move an ordinary endpoint merely to a thick crossing's centre.
        # Splitting requires clear continuation past that observed station.
        if abs(line[endpoint] - other["cross"]) > max(margin, other.get("thickness", 1)) * 2:
            result.setdefault("unsegmented_stroke", dict(line))
            result[endpoint] = float(other["cross"])
    result["span"] = result["hi"] - result["lo"] + 1
    if "unsegmented_stroke" in result:
        result["extension_stations"] = supports
        result["segmentation_method"] = "nearest_observed_extensions_outside_label"
    return result


def _linear_witnesses(gray, records, *, split_chains=False):
    """Independent non-diameter spans distinguish D from D/2 semantics.

    A diameter endpoint regression alone cannot determine whether the drawing
    shows a half section. Do not infer that semantic choice from filenames.
    """
    witnesses = []
    lines_by_axis = {axis: _axis_lines(gray, axis) for axis in ("x", "y")}
    by_record = {}
    for axis in ("x", "y"):
        lines = lines_by_axis[axis]
        for row in records:
            parsed = row["parsed"]
            if parsed["kind"] != "length" or not parsed["nominal"] or parsed["nominal"] <= 0 or not row.get("box"):
                continue
            box = np.asarray(row["box"], float)
            along, cross = (box[:, 0], box[:, 1]) if axis == "x" else (box[:, 1], box[:, 0])
            cross_size = max(8., float(np.ptp(cross)))
            along_size = max(8., float(np.ptp(along)))
            options = []
            for line in lines:
                delta = line["cross"] - float(max(cross))
                if not -.3 * cross_size <= delta <= 1.3 * cross_size:
                    continue
                if not line["lo"] <= float(np.mean(along)) <= line["hi"]:
                    continue
                if line["span"] < max(along_size * 1.5, max(gray.shape) * .06):
                    continue
                options.append((abs(delta) / cross_size, line))
            if options:
                distance, line = min(options, key=lambda pair: pair[0])
                if split_chains:
                    line = _label_span(line, along, cross_size, lines_by_axis["y" if axis == "x" else "x"])
                by_record.setdefault(row["id"], []).append({"record_id": row["id"], "text": row["text"], "axis": axis,
                                  "nominal": parsed["nominal"], "span_px": line["hi"] - line["lo"],
                                  "pixels_per_mm": (line["hi"] - line["lo"]) / parsed["nominal"],
                                  "line": line, "box": row["box"],
                                  "association_distance_label_heights": float(distance)})
    for options in by_record.values():
        options.sort(key=lambda item: item["association_distance_label_heights"])
        # One label cannot independently describe both axes. Prefer the source
        # stroke directly alongside it over a remote orthogonal material edge.
        # Near ties remain explicitly ambiguous rather than picked by scale.
        best = options[0]["association_distance_label_heights"]
        retained = [item for item in options if item["association_distance_label_heights"] < best + .25]
        for item in retained:
            item["axis_association"] = {
                "method": "nearest_source_stroke_normalized_label_distance",
                "ambiguity_margin_label_heights": .25,
                "ambiguous": len(retained) > 1,
                "alternatives": [{"axis": other["axis"],
                                  "distance_label_heights": other["association_distance_label_heights"],
                                  "retained": other in retained} for other in options],
            }
            witnesses.append(item)
    return witnesses


def _dimension_station_arrow(gray, axis, cross, station, sign, label_size):
    """Observe a filled dimension arrow at an independently detected station.

    Unlike a radius leader, a dimension shaft can continue beyond its arrow
    tip (in particular when arrows are outside a short measured interval).
    Its narrowing tip, widening body and thinner shaft are still required.
    A plain crossing or an undifferentiated thick line cannot establish this.
    No nominal length or candidate scale participates in the observation.
    """
    oriented = gray if axis == "y" else gray.T
    h, w = oriented.shape

    def width_at(y, extent):
        iy, ix = int(round(y)), int(round(cross))
        if not 0 <= iy < h:
            return 0.
        left, right = max(0, ix-extent), min(w, ix+extent+1)
        ink = oriented[iy, left:right] < 170
        central = [i for i in range(max(0, ix-left-2), min(len(ink), ix-left+3)) if ink[i]]
        if not central:
            return 0.
        index = min(central, key=lambda i: abs(i-(ix-left)))
        lo = hi = index
        while lo > 0 and ink[lo-1]:
            lo -= 1
        while hi+1 < len(ink) and ink[hi+1]:
            hi += 1
        return float(hi-lo+1)

    best = None
    for length in sorted({max(7., min(48., label_size*f)) for f in (.08, .12, .18, .24, .32, .42)}):
        extent = max(7, int(length*.65))
        for shift in (-2., 0., 2.):
            tip = station+shift
            widths = [width_at(tip-sign*length*f, extent) for f in (.13, .40, .70, 1.25, 1.60)]
            narrow, middle, body, shaft, tail = widths
            shaft = max(shaft, tail, 1.)
            if not (narrow >= shaft+1 and middle >= 2 and body >= max(4., 1.8*shaft)
                    and .45*body <= middle <= 1.35*body and narrow <= .72*body
                    and body <= .85*length):
                continue
            score = body/shaft + body/max(narrow, 1.) - abs(shift)/length
            item = {"verified": True, "method": "dimension_station_directed_filled_taper",
                    "tip_px": [cross, station] if axis == "y" else [station, cross],
                    "direction_px": [0., float(sign)] if axis == "y" else [float(sign), 0.],
                    "sample_tip_shift_px": shift, "length_px": length,
                    "cross_section_widths_px": widths, "score": float(score)}
            if best is None or score > best["score"]:
                best = item
    return best


def _local_binding_witnesses(gray, row):
    """Return source-arrow-confirmed intervals, including external-arrow spans.

    Window and morphology lengths depend on the OCR box, never the nominal
    value. Both stations must be observed perpendicular strokes with opposing
    arrows on one common shaft. Ambiguous intervals are retained for review.
    """
    box = np.asarray(row["box"], float)
    low, high = box.min(axis=0), box.max(axis=0)
    size = np.maximum(high-low, 8.)
    padding = np.minimum(np.maximum(size*2.2, 80.), 320.)
    roi_low = np.maximum(np.floor(low-padding).astype(int), 0)
    roi_high = np.minimum(np.ceil(high+padding).astype(int), np.asarray(gray.shape[::-1]))
    roi = gray[roi_low[1]:roi_high[1], roi_low[0]:roi_high[0]]
    if not roi.size:
        return []
    lines_by_axis = {}
    for axis, k in (("x", 0), ("y", 1)):
        local = _axis_lines(roi, axis, minimum_span=max(10, int(min(size)*.22)))
        lines_by_axis[axis] = [{**line, "lo": line["lo"]+float(roi_low[k]),
                               "hi": line["hi"]+float(roi_low[k]),
                               "cross": line["cross"]+float(roi_low[1-k])} for line in local]
    candidates = []
    search_complete = True
    for axis, k in (("x", 0), ("y", 1)):
        cross_size, along_size = size[1-k], size[k]
        lines = [line for line in lines_by_axis[axis]
                 if -.3*cross_size <= line["cross"]-high[1-k] <= 1.3*cross_size
                 and line["lo"]-along_size <= box[:, k].mean() <= line["hi"]+along_size]
        # Bounded local hypotheses. If a busy text/line neighbourhood exceeds
        # the bound, do not silently claim that the remaining pair is unique.
        if len(lines) > 12:
            search_complete = False
            continue
        for line in lines:
            # Local opening can connect the arrow body to its shaft. Its full
            # component width must not let an adjacent OCR glyph masquerade
            # as an extension which actually reaches the shaft.
            margin = min(4., max(2., line["thickness"]*.5))
            stations = [other for other in lines_by_axis["y" if axis == "x" else "x"]
                        if other["lo"]-margin <= line["cross"] <= other["hi"]+margin
                        and line["lo"]-margin <= other["cross"] <= line["hi"]+margin
                        and other["span"] >= max(16., cross_size*.4)
                        and low[k]-along_size*2 <= other["cross"] <= high[k]+along_size*2]
            stations.sort(key=lambda item: item["cross"])
            if len(stations) > 12:
                search_complete = False
                continue
            arrows = {}
            for station in stations:
                for sign in (-1., 1.):
                    arrows[(station["cross"], sign)] = _dimension_station_arrow(
                        gray, axis, line["cross"], station["cross"], sign,
                        float(np.linalg.norm(size)))
            # Adjacent observed stations only: an intervening extension may
            # delimit a chained dimension and must not be skipped by ranking.
            for first, second in zip(stations[:-1], stations[1:]):
                lo, hi = first["cross"], second["cross"]
                if hi-lo < max(12., line["thickness"]*4):
                    continue
                label_gap = max(lo-high[k], low[k]-hi, 0.)
                if label_gap > along_size*1.25:
                    continue
                pairs = [(arrows[(lo, sign)], arrows[(hi, -sign)]) for sign in (-1., 1.)]
                pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
                if not pairs:
                    continue
                a, b = max(pairs, key=lambda pair: pair[0]["score"]+pair[1]["score"])
                interval = {**line, "lo": lo, "hi": hi, "span": hi-lo+1,
                            "unsegmented_stroke": dict(line),
                            "extension_stations": {"lo": first, "hi": second},
                            "segmentation_method": "opposing_source_arrows_at_observed_extensions"}
                distance = abs(line["cross"]-high[1-k])/cross_size
                candidates.append({"record_id": row["id"], "text": row["text"], "axis": axis,
                                   "nominal": row["parsed"]["nominal"], "span_px": hi-lo,
                                   "pixels_per_mm": (hi-lo)/row["parsed"]["nominal"],
                                   "line": interval, "box": row["box"],
                                   "association_distance_label_heights": float(distance),
                                   "binding_evidence": {"method": "ocr_local_opposing_dimension_arrows_v1",
                                       "verified": True, "nominal_used_to_rank": False,
                                       "endpoint_arrows": [a, b], "label_interval_gap_px": label_gap,
                                       "roi_px": [roi_low.tolist(), roi_high.tolist()],
                                       "scale_estimation_eligible": False}})
    # Do not use the expected nominal span to choose between competing strokes.
    # Deduplicate only the same observed station pair on the same shaft.
    unique = {}
    for item in candidates:
        line = item["line"]
        key = (item["axis"], line["lo"], line["hi"], line["cross"])
        unique.setdefault(key, item)
    result = list(unique.values())
    for item in result:
        item["binding_evidence"]["verified"] = search_complete
        item["binding_evidence"]["search_complete"] = search_complete
        item["binding_evidence"]["ambiguous"] = len(result) > 1 or not search_complete
        item["binding_evidence"]["alternative_intervals"] = len(result)
        item["axis_association"] = {"method": "opposing_source_arrows_and_extension_stations",
                                    "ambiguous": len(result) > 1 or not search_complete}
    return result


def _long_binding_witnesses(gray, legacy, crossing_by_axis):
    """Recover distant arrow stations on an already label-owned long shaft.

    A legitimate short extension at a remote center may be shorter than the
    OCR label. Legacy span splitting excluded it and selected a different
    crossing farther away. Observe both arrows on the full shaft instead;
    neither the nominal nor the predicted CAD supplies an endpoint.
    """
    result = []
    by_record = {}
    for witness in legacy:
        if witness.get("axis_association", {}).get("ambiguous"):
            continue
        axis = witness["axis"]; k = 0 if axis == "x" else 1
        box = np.asarray(witness["box"], float)
        size = np.maximum(np.ptp(box, axis=0), 8.)
        line = witness["line"].get("unsegmented_stroke", witness["line"])
        margin = min(4., max(2., line["thickness"]*.5))
        stations = [other for other in crossing_by_axis[axis]
            if other["lo"]-margin <= line["cross"] <= other["hi"]+margin
            and line["lo"]-margin <= other["cross"] <= line["hi"]+margin
            and other["span"] >= max(16., float(size[1-k])*.4)]
        stations.sort(key=lambda row: row["cross"])
        if len(stations) > 64:
            continue
        for first, second in zip(stations[:-1], stations[1:]):
            low, high = first["cross"], second["cross"]
            if high-low < max(12., 4*line["thickness"]):
                continue
            gap = max(low-float(box[:, k].max()), float(box[:, k].min())-high, 0.)
            if gap > size[k]*1.25:
                continue
            for sign in (-1., 1.):
                a = _dimension_station_arrow(gray, axis, line["cross"], low, sign, float(np.linalg.norm(size)))
                b = _dimension_station_arrow(gray, axis, line["cross"], high, -sign, float(np.linalg.norm(size)))
                if a is None or b is None:
                    continue
                item = {**witness, "line": {**line, "lo": low, "hi": high, "span": high-low+1,
                    "unsegmented_stroke": dict(line), "extension_stations": {"lo": first, "hi": second},
                    "segmentation_method": "full_shaft_opposing_source_arrows_at_finite_extensions"},
                    "span_px": high-low, "pixels_per_mm": (high-low)/witness["nominal"],
                    "binding_evidence": {"method": "full_shaft_opposing_dimension_arrows_v1",
                        "verified": True, "nominal_used_to_rank": False, "endpoint_arrows": [a, b],
                        "label_interval_gap_px": gap, "search_complete": True,
                        "scale_estimation_eligible": False}}
                by_record.setdefault(witness["record_id"], {})[(axis, low, high, line["cross"])] = item
    for options in by_record.values():
        for item in options.values():
            item["binding_evidence"].update(ambiguous=len(options) > 1, alternative_intervals=len(options))
            item["axis_association"] = {"method": "full_shaft_opposing_arrows_and_finite_extensions",
                                        "ambiguous": len(options) > 1}
            result.append(item)
    return result


def binding_linear_witnesses(gray, records):
    """Binding evidence with short dimensions kept separate from scale evidence.

    Original long-span witnesses remain available. A locally observed pair of
    opposing arrows supersedes that record's unsplit stroke; competing pairs
    stay explicitly ambiguous and must not become hard constraints. Endpoint
    attachment to actual CAD objects remains the caller's independent duty.
    """
    legacy = _linear_witnesses(gray, records, split_chains=True)
    observed = []
    for row in records:
        parsed = row.get("parsed") or {}
        if (parsed.get("kind") != "length" or not row.get("box")
                or not isinstance(parsed.get("nominal"), (int, float))
                or not np.isfinite(parsed["nominal"]) or parsed["nominal"] <= 0):
            continue
        observed.extend(_local_binding_witnesses(gray, row))
    recovered = {item["record_id"] for item in observed}
    remaining = [item for item in legacy if item["record_id"] not in recovered]
    if remaining:
        observed.extend(_long_binding_witnesses(gray, remaining, {
            "x": _axis_lines(gray, "y"), "y": _axis_lines(gray, "x")}))
        recovered = {item["record_id"] for item in observed}
    return [item for item in legacy if item["record_id"] not in recovered] + observed


def _estimate_diameter_scale(gray, records, witnesses):
    candidates = []
    for axis in ("x", "y"):
        lines = _axis_lines(gray, axis)
        crossing_lines = _axis_lines(gray, "y" if axis == "x" else "x")
        bindings = []
        for row in records:
            parsed = row["parsed"]
            if parsed["kind"] != "diameter" or not parsed["nominal"] or parsed["nominal"] <= 0 or not row.get("box"):
                continue
            box = np.asarray(row["box"], dtype=float)
            along, cross = (box[:, 0], box[:, 1]) if axis == "x" else (box[:, 1], box[:, 0])
            center = float(np.mean(along))
            cross_lo, cross_hi = float(min(cross)), float(max(cross))
            cross_size = max(8, cross_hi - cross_lo)
            along_size = max(8, float(max(along) - min(along)))
            options = []
            for line in lines:
                # Dimension text normally lies just above/left of its dimension line.
                delta = line["cross"] - cross_hi
                if not (-.3 * cross_size <= delta <= 1.3 * cross_size):
                    continue
                if not line["lo"] - along_size * .3 <= center <= line["hi"] + along_size * .3:
                    continue
                if line["span"] < along_size * 1.4:
                    continue
                score = abs(delta) / cross_size + .15 * along_size / line["span"]
                options.append((score, line))
            if options:
                _, line = min(options, key=lambda item: item[0])
                line = dict(line)
                # Adjacent dimensions may share one uninterrupted stroke; use
                # the first long extension line beyond the label to split it.
                intersections = [other["cross"] for other in crossing_lines
                                 if other["lo"] - 3 <= line["cross"] <= other["hi"] + 3
                                 and max(along) + along_size * .25 < other["cross"] <= line["hi"] + 4
                                 and other["span"] > cross_size * 2]
                if intersections:
                    nearest = min(intersections)
                    if line["hi"] - nearest > max(gray.shape) * .015:
                        line["stroke_hi"] = line["hi"]
                        line["hi"] = nearest
                bindings.append({"record_id": row["id"], "text": row["text"], "nominal": float(parsed["nominal"]),
                                 "box": row["box"], "line": line})
        for endpoint in ("hi", "lo"):
            for first, second in combinations(bindings, 2):
                d1, d2 = first["nominal"] / 2, second["nominal"] / 2
                if abs(d2 - d1) < max(5, max(d1, d2) * .08):
                    continue
                slope = (second["line"][endpoint] - first["line"][endpoint]) / (d2 - d1)
                if not .1 <= abs(slope) <= 1000:
                    continue
                intercept = first["line"][endpoint] - slope * d1
                # Tolerance covers scan edge thickness and arrow-tip attachment.
                tolerance_px = max(4., .006 * max(gray.shape))
                inliers = [b for b in bindings if abs(b["line"][endpoint] - (intercept + slope * b["nominal"] / 2)) <= tolerance_px]
                distinct = len(set(b["nominal"] for b in inliers))
                if distinct < 3:
                    continue
                xs = np.array([b["nominal"] / 2 for b in inliers])
                ys = np.array([b["line"][endpoint] for b in inliers])
                design = np.column_stack([xs, np.ones(len(xs))])
                scale, origin = np.linalg.lstsq(design, ys, rcond=None)[0]
                residuals = ys - (scale * xs + origin)
                if abs(scale) < .1:
                    continue
                span = float(np.ptp(ys))
                if span < (gray.shape[1] if axis == "x" else gray.shape[0]) * .12:
                    continue
                rms = float(np.sqrt(np.mean(residuals ** 2)))
                candidates.append({"axis": axis, "pixels_per_mm": abs(float(scale)), "axis_origin_px": float(origin),
                                   "direction": 1 if scale > 0 else -1, "endpoint": endpoint, "bindings": inliers,
                                   "distinct_dimensions": distinct, "rms_residual_px": rms, "span_px": span,
                                   "score": distinct + span / max(gray.shape) - rms / tolerance_px})
    if not candidates:
        return {"status": "unresolved", "pixels_per_mm": None, "method": "diameter_extension_line_consensus-v1",
                "issues": ["未找到至少三个相互一致的直径标注及尺寸线端点，不能可靠换算毫米。"]}
    # A coherent diameter regression is still ambiguous by a factor of two.
    # Require independent linear dimensions before assigning physical units.
    verified = []
    for candidate in candidates:
        interpretations = []
        for semantic, factor in (("half_section_radius_station", 1.), ("full_diameter_span", .5)):
            scale = candidate["pixels_per_mm"] * factor
            matching = [w for w in witnesses if abs(w["pixels_per_mm"] / scale - 1) <= .035]
            distinct = len(set(w["record_id"] for w in matching))
            if distinct >= 2 or (distinct == 1 and matching[0]["span_px"] >= max(gray.shape) * .3):
                interpretations.append({**candidate, "pixels_per_mm": scale,
                                        "diameter_semantics": semantic, "linear_witnesses": matching})
        if len(interpretations) == 1:
            verified.extend(interpretations)
    if not verified:
        best = max(candidates, key=lambda c: c["score"])
        return {**best, "status": "ambiguous", "pixels_per_mm": None,
                "method": "diameter_extension_line_consensus-v2",
                "diameter_semantics": "unresolved_half_or_full", "linear_witnesses": witnesses,
                "hypotheses_px_per_mm": [best["pixels_per_mm"], best["pixels_per_mm"] / 2],
                "issues": ["直径标注存在半剖面与完整直径的二倍比例歧义；缺少足够的独立线性尺寸证据，采用像素单位。"]}
    candidates = verified
    best = max(candidates, key=lambda c: c["score"])
    alternate = [c for c in candidates if c["distinct_dimensions"] >= best["distinct_dimensions"] and
                 abs(c["pixels_per_mm"] / best["pixels_per_mm"] - 1) > .04]
    ambiguous = bool(alternate)
    return {**best, "status": "ambiguous" if ambiguous else "resolved", "method": "diameter_extension_line_consensus-v2",
            "alternatives": [{k: c[k] for k in ("axis", "pixels_per_mm", "distinct_dimensions", "rms_residual_px", "span_px", "score")} for c in sorted(alternate, key=lambda c: -c["score"])[:5]],
            "issues": ["存在不同尺寸比例的竞争解释。"] if ambiguous else [],
            "scope": "Scale inferred from original-image dimension lines, not GT; isotropic scan scaling assumed."}


def _linear_scale_consensus(witnesses):
    """Require three independent lengths/lines and a strict 3% ratio spread.

    Each accepted window is measured against its minimum scale, not a fitted
    centre with a +/-3% allowance. Duplicate OCR records on the same detected
    line cannot supply independent geometry. Distinct valid scale modes are
    returned as ambiguous even when one has more supporting dimensions.
    """
    usable = [w for w in witnesses if np.isfinite(w["pixels_per_mm"]) and w["pixels_per_mm"] > 0]
    usable.sort(key=lambda w: (w["pixels_per_mm"], w["record_id"], w["axis"]))
    clusters = []
    for seed in usable:
        group = [w for w in usable if 1 <= w["pixels_per_mm"] / seed["pixels_per_mm"] <= 1.03]
        nominal_count = len({w["nominal"] for w in group})
        record_count = len({w["record_id"] for w in group})
        line_count = len({(w["axis"], w["line"]["lo"], w["line"]["hi"], w["line"]["cross"]) for w in group})
        if min(nominal_count, record_count, line_count) < 3:
            continue
        # Give each physical line one vote when calculating the central scale.
        physical = {}
        for witness in group:
            key = (witness["axis"], witness["line"]["lo"], witness["line"]["hi"], witness["line"]["cross"])
            physical.setdefault(key, []).append(witness["pixels_per_mm"])
        central = float(np.median([np.median(values) for values in physical.values()]))
        ratios = [w["pixels_per_mm"] for w in group]
        clusters.append({"pixels_per_mm": central, "min_pixels_per_mm": min(ratios),
                         "max_pixels_per_mm": max(ratios), "ratio_spread": max(ratios)/min(ratios)-1,
                         "distinct_dimensions": nominal_count, "distinct_records": record_count,
                         "distinct_physical_lines": line_count,
                         "measurement_axes": sorted({w["axis"] for w in group}), "bindings": group})
    clusters.sort(key=lambda c: (-c["distinct_dimensions"], -c["distinct_physical_lines"], c["ratio_spread"]))
    modes = []
    for cluster in clusters:
        # Overlapping windows are one mode only when their combined scale range
        # still satisfies the same 3% rule. Do not hide a near-threshold conflict.
        same_mode = any(max(cluster["max_pixels_per_mm"], mode["max_pixels_per_mm"]) /
                        min(cluster["min_pixels_per_mm"], mode["min_pixels_per_mm"]) <= 1.03
                        for mode in modes)
        if not same_mode:
            modes.append(cluster)
    return modes


def estimate_scale(image_path: str | Path, document: dict) -> dict:
    """Cross-check diameter evidence against independent linear consensus.

    Diameter endpoints can include a datum-symbol tail or the next chained
    dimension. A valid three-line length consensus must not be used only to
    select the D versus D/2 interpretation while its actual scale is ignored.
    The existing 3% maximum/minimum rule applies to the combined evidence.
    """
    gray = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError("无法读取尺寸证据图像")
    records = canonical_records(document)
    # Scale consensus keeps complete detected strokes. Source-derived chain
    # segmentation is available to the binding stage, where both stations must
    # also connect to contour nodes; a bare crossing cannot establish scale.
    witnesses = _linear_witnesses(gray, records)
    diameter = _estimate_diameter_scale(gray, records, witnesses)
    clusters = _linear_scale_consensus(witnesses)
    cross_check = None
    if diameter["status"] == "resolved":
        if not clusters:
            return diameter
        if len(clusters) == 1:
            linear = clusters[0]
            spread = max(diameter["pixels_per_mm"], linear["max_pixels_per_mm"]) / min(
                diameter["pixels_per_mm"], linear["min_pixels_per_mm"]) - 1
            cross_check = {"status": "consistent" if spread <= .03 else "conflict",
                           "diameter_pixels_per_mm": diameter["pixels_per_mm"],
                           "linear_pixels_per_mm": linear["pixels_per_mm"],
                           "linear_min_pixels_per_mm": linear["min_pixels_per_mm"],
                           "linear_max_pixels_per_mm": linear["max_pixels_per_mm"],
                           "combined_ratio_spread": float(spread), "maximum_ratio_spread": .03,
                           "linear_distinct_dimensions": linear["distinct_dimensions"],
                           "linear_distinct_physical_lines": linear["distinct_physical_lines"],
                           "linear_record_ids": [w["record_id"] for w in linear["bindings"]]}
            if cross_check["status"] == "consistent":
                return {**diameter, "cross_evidence_consistency": cross_check}
        else:
            cross_check = {"status": "multiple_linear_modes", "maximum_ratio_spread": .03,
                           "diameter_pixels_per_mm": diameter["pixels_per_mm"]}
    if not clusters:
        return {**diameter, "linear_witnesses": witnesses,
                "linear_consensus": {"status": "insufficient_evidence", "required_distinct_dimensions": 3,
                                     "required_distinct_physical_lines": 3, "maximum_ratio_spread": .03}}
    common = {"method": "linear_dimension_consensus-v1", "scale_kind": "linear_dimension_consensus",
              "axis": None, "axis_origin_px": None, "direction": None,
              "rotation_axis_resolved": False, "diameter_semantics": "not_inferred_linear_only",
              "diameter_evidence": diameter, "linear_witnesses": witnesses,
              "assumptions": ["isotropic_image_scaling", "dimension_nominals_in_millimetres"],
              "scope": "Scale from original-image linear dimension spans only; no rotation-axis origin inferred. Isotropic scan scaling assumed."}
    if cross_check:
        common["cross_evidence_consistency"] = cross_check
    if len(clusters) > 1:
        return {**common, "status": "ambiguous", "pixels_per_mm": None, "bindings": [],
                "alternatives": clusters,
                "issues": ["独立线性尺寸存在多个满足三个不同名义值和尺寸线的一致比例簇，不能唯一确定毫米比例。"]}
    best = clusters[0]
    issues = []
    if cross_check and cross_check["status"] == "conflict":
        common["discarded_diameter_reason"] = "diameter_endpoints_conflict_with_independent_linear_consensus"
        issues.append("直径端点拟合与至少三个独立线性尺寸的比例不满足同一3%范围；保留冲突证据，采用独立线性尺寸共识，未推断旋转轴原点。")
    return {**common, **best, "status": "resolved", "alternatives": [], "issues": issues}
