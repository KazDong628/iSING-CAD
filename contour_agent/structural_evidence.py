"""Conservative source-pixel checks for optional geometric relations.

A fitted graph suggests where to inspect, never supplies its own proof. These
checks neither read reference CAD nor use nominal dimensions. Missing/ambiguous
strokes stay unresolved instead of becoming hard solver constraints.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .dimension_evidence import _axis_lines


def source_ink(gray, records):
    cleaned = gray.copy()
    text_mask = np.zeros_like(gray, np.uint8)
    for row in records:
        try:
            box = np.asarray(row.get("box"), float)
            if box.ndim == 2 and box.shape[1] == 2 and len(box) >= 3 and np.isfinite(box).all():
                cv2.fillPoly(cleaned, [np.rint(box).astype(np.int32)], 255)
                cv2.fillPoly(text_mask, [np.rint(box).astype(np.int32)], 255)
        except (TypeError, ValueError):
            continue
    # Loose OCR polygons sometimes cover a real long boundary. Retain its
    # actually observed ink only when most of that same continuous axial stroke
    # lies outside text. A glyph wholly inside an OCR box is never restored.
    for axis in ("x", "y"):
        for line in _axis_lines(gray, axis):
            if line["span"] < max(32., max(gray.shape) * .04):
                continue
            half = line["thickness"] / 2
            limits = ([line["lo"], line["cross"] - half, line["hi"] + 1, line["cross"] + half + 1]
                      if axis == "x" else
                      [line["cross"] - half, line["lo"], line["cross"] + half + 1, line["hi"] + 1])
            x0, y0, x1, y1 = [int(round(v)) for v in limits]
            x0, y0, x1, y1 = max(0,x0), max(0,y0), min(gray.shape[1],x1), min(gray.shape[0],y1)
            observed = gray[y0:y1,x0:x1] < 170
            outside = text_mask[y0:y1,x0:x1] == 0
            if observed.any() and float(np.count_nonzero(observed & outside)) / np.count_nonzero(observed) >= .7:
                region = cleaned[y0:y1,x0:x1]
                region[observed] = gray[y0:y1,x0:x1][observed]
    _, ink = cv2.threshold(cleaned, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return cleaned, ink > 0


def axis_stroke_evidence(samples, axis, lines, band):
    """Require one measured axial ink strip covering the material-edge proposal."""
    along, across = (0, 1) if axis == "horizontal" else (1, 0)
    low, high = float(samples[:, along].min()), float(samples[:, along].max())
    span = high - low
    result = {"method": "unique_source_axis_stroke_v1", "verified": False,
              "span_px": span, "proposal_band_px": band,
              "reason": "source_axis_stroke_unavailable"}
    if span < max(24., 4 * band):
        result["reason"] = "source_axis_span_too_short"
        return result
    options = []
    for line in lines:
        half = max(1., float(line["thickness"])) / 2
        gap = np.maximum(np.abs(samples[:, across] - line["cross"]) - half, 0.)
        coverage = max(0., min(high, line["hi"]) - max(low, line["lo"])) / span
        # Endpoint-sized raster omissions are allowed, but an isolated dimension
        # crossing or short hatch fragment cannot certify an entire boundary.
        if coverage >= .94 and float(gap.max()) <= band:
            options.append({**line, "covered_fraction": coverage,
                            "maximum_projection_gap_px": float(gap.max())})
    result["supporting_strokes"] = options
    if len(options) != 1:
        if options:
            result["reason"] = "multiple_parallel_source_strokes"
        return result
    result.update(verified=True, reason=None, observed_station_px=options[0]["cross"])
    return result


def _ink_trace(ink, samples, band):
    """Measure single connected ink runs across normals, reject local crossings."""
    distances = np.r_[0., np.cumsum(np.linalg.norm(np.diff(samples, axis=0), axis=1))]
    span = min(96., float(distances[-1]) * .45)
    if span < max(24., 4 * band):
        return {"verified": False, "reason": "insufficient_local_source_span", "span_px": span}
    stations = np.linspace(1.5, span, 32)
    seed = np.c_[np.interp(stations, distances, samples[:, 0]),
                 np.interp(stations, distances, samples[:, 1])]
    tangent = np.gradient(seed, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1)[:, None], 1e-9)
    normals = np.c_[-tangent[:, 1], tangent[:, 0]]
    extent = max(4, int(math.ceil(band)))
    offsets = np.arange(-extent, extent + 1, dtype=float)
    measured, kept, widths = [], [], []
    for index, (point, normal) in enumerate(zip(seed, normals)):
        cross = np.rint(point + offsets[:, None] * normal).astype(int)
        valid = ((cross[:, 0] >= 0) & (cross[:, 0] < ink.shape[1]) &
                 (cross[:, 1] >= 0) & (cross[:, 1] < ink.shape[0]))
        if not valid.all():
            continue
        occupied = ink[cross[:, 1], cross[:, 0]]
        starts = np.flatnonzero(occupied & ~np.r_[False, occupied[:-1]])
        ends = np.flatnonzero(occupied & ~np.r_[occupied[1:], False])
        # Never choose the run nearest the fitted curve from several alternatives:
        # that would make the prior decide which hatch/dimension stroke is truth.
        if len(starts) != 1 or starts[0] == 0 or ends[0] == len(offsets) - 1:
            continue
        width = int(ends[0] - starts[0] + 1)
        if width > max(7., band):
            continue
        measured.append(point + float(offsets[starts[0]:ends[0] + 1].mean()) * normal)
        kept.append(index)
        widths.append(width)
    result = {"method": "unambiguous_normal_ink_runs_quadratic_tangent_v1", "verified": False,
              "span_px": span, "sample_count": len(stations), "unambiguous_samples": len(kept),
              "reason": "source_junction_strokes_ambiguous"}
    if len(kept) < 24 or min(kept, default=32) > 3 or max(kept, default=0) < 28:
        return result
    points = np.asarray(measured)
    parameter = stations[kept] / span
    # Estimate the derivative at the joint from source ink on two independently
    # chosen scales. A corner hidden by one broad fit must not pass as tangency.
    def fit(keep):
        coefficients = np.polynomial.polynomial.polyfit(parameter[keep], points[keep], 2)
        fitted = np.polynomial.polynomial.polyval(parameter[keep], coefficients).T
        residual = np.linalg.norm(points[keep] - fitted, axis=1)
        direction = coefficients[1]
        direction /= max(float(np.linalg.norm(direction)), 1e-9)
        return direction, float(np.percentile(residual, 90))
    full, residual = fit(np.ones(len(points), bool))
    short_mask = parameter <= .75
    if int(short_mask.sum()) < 16:
        return result
    short, short_residual = fit(short_mask)
    stability = math.degrees(math.acos(float(np.clip(full @ short, -1., 1.))))
    result.update(tangent_direction_px=full.tolist(), fit_residual_p90_px=residual,
                  short_fit_residual_p90_px=short_residual, scale_disagreement_degrees=stability)
    if max(residual, short_residual) > max(1.25, float(np.median(widths)) / 2):
        result["reason"] = "source_local_curve_fit_unstable"
    elif stability > 2.:
        result["reason"] = "source_tangent_scale_disagreement"
    else:
        result.update(verified=True, reason=None)
    return result


def structural_evidence(gray, records, graph, transform, sample, band):
    """Return source-validated copies of the graph's structural hypotheses."""
    cleaned, ink = source_ink(gray, records)
    lines = {"horizontal": _axis_lines(cleaned, "x"), "vertical": _axis_lines(cleaned, "y")}
    entities = {e["id"]: e for e in graph.get("entities", [])}
    origin, x_point, y_point = transform([[0., 0.], [1., 0.], [0., 1.]])
    x_axis, y_axis = x_point - origin, y_point - origin
    aligned = (abs(x_axis[1]) <= 1e-6 * max(np.linalg.norm(x_axis), 1e-9) and
               abs(y_axis[0]) <= 1e-6 * max(np.linalg.norm(y_axis), 1e-9))
    result = []
    trace_cache = {}
    for index, original in enumerate(graph.get("relations", [])):
        kind, entity_ids = original.get("type"), original.get("entities", [])
        if kind not in {"horizontal", "vertical", "tangent"} or not all(e in entities for e in entity_ids):
            continue
        item = {"id": original.get("id", f"rel{index:03d}"), "type": kind,
                "entities": entity_ids, "nodes": original.get("nodes", []),
                "source": "geometry_hypothesis", "required": False, "local_reliable": False}
        evidence = {"verified": False, "reason": "relation_entity_type_mismatch"}
        if kind in lines and len(entity_ids) == 1 and entities[entity_ids[0]]["type"] == "LINE":
            evidence = (axis_stroke_evidence(sample(entities[entity_ids[0]], transform), kind, lines[kind], band)
                        if aligned else {"verified": False, "reason": "source_coordinate_axes_not_aligned"})
        elif kind == "tangent" and len(entity_ids) == 2 and len(set(entity_ids)) == 2:
            first, second = [entities[e] for e in entity_ids]
            shared = ({first.get("start_node"), first.get("end_node")} &
                      {second.get("start_node"), second.get("end_node")}) - {None}
            evidence = {"verified": False, "reason": "tangent_requires_unique_shared_node"}
            if len(shared) == 1:
                node = next(iter(shared))
                traces = []
                for entity in (first, second):
                    at_end = entity.get("end_node") == node
                    key = entity["id"], at_end
                    if key not in trace_cache:
                        points = sample(entity, transform)
                        trace_cache[key] = _ink_trace(ink, points[::-1] if at_end else points, band)
                    traces.append(trace_cache[key])
                evidence = {"method": "two_sided_source_tangent_v1", "verified": False,
                            "shared_node": node, "sides": traces, "tolerance_degrees": 3.,
                            "reason": "source_tangent_evidence_insufficient"}
                if all(t["verified"] for t in traces):
                    cosine = float(np.dot(traces[0]["tangent_direction_px"], traces[1]["tangent_direction_px"]))
                    # Both traces run AWAY from a shared node, so a smooth
                    # continuation must be opposite, not merely parallel.
                    angle = math.degrees(math.acos(float(np.clip(-cosine, -1., 1.))))
                    evidence.update(observed_deviation_degrees=angle, verified=angle <= 3.,
                                    reason=None if angle <= 3. else "source_junction_is_not_tangent")
                item["nodes"] = [node]
        item.update(local_reliable=bool(evidence["verified"]), evidence=evidence)
        result.append(item)
    return result
