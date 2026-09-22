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


def _axis_lines(gray: np.ndarray, axis: str) -> list[dict]:
    h, w = gray.shape
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    extent = w if axis == "x" else h
    length = max(25, int(extent * .025))
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
