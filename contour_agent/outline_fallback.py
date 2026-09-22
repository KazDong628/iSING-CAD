"""Phase-free, source-only fallback for material silhouette proposals.

The legacy entry-point name ``extract_unhatched`` does not imply that an
unhatched outline is solved. This implementation requires directional hatch
evidence, but deliberately makes no global hatch-spacing/phase assumption.
All proposals remain drafts; coordinates are original image pixels.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .raster import _angle_difference, _remove_local_detours, _write_image


def _text_mask(document, shape, scale):
    """Use only declared source OCR polygons, never execute document content."""
    mask = np.zeros(shape, np.uint8)
    count = 0
    if not isinstance(document, dict):
        return mask, count
    for record in document.get("records", []):
        if not isinstance(record, dict):
            continue
        try:
            points = np.asarray(record.get("box", []), dtype=float)
            if points.shape != (4, 2) or not np.isfinite(points).all():
                continue
            points = np.rint(points * scale).astype(np.int32)
            # Excessively large OCR regions must not suppress a whole view.
            if abs(cv2.contourArea(points)) > mask.size * .03:
                continue
            cv2.fillPoly(mask, [points], 255)
            count += 1
        except (TypeError, ValueError, OverflowError):
            continue
    return mask, count


def extract_unhatched(image_path, ocr_document=None, output_dir=None, *, max_dimension=2600):
    """Propose closed hatch-supported silhouettes without periodic alignment.

    Source OCR only excludes segment midpoints inside text polygons. Original
    ink remains intact for enclosure detection. No dimensions, templates,
    filenames, reference geometry, learned case constants or provider calls
    are involved. Blank/unhatched images yield an explicit unresolved result.
    """
    with Image.open(Path(image_path)) as source:
        original_width, original_height = source.size
        if original_width * original_height > 80_000_000:
            raise ValueError("Image exceeds the 80 million pixel processing limit")
        rgb = np.asarray(source.convert("RGB"))
    scale = min(1., max_dimension / max(original_width, original_height))
    resized = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else rgb
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    text_mask, text_count = _text_mask(ocr_document, gray.shape, scale)
    detected = cv2.createLineSegmentDetector(cv2.LSD_REFINE_NONE).detect(gray)[0]
    segments = detected.reshape(-1, 4) if detected is not None else np.empty((0, 4))
    delta = segments[:, 2:] - segments[:, :2]
    lengths = np.linalg.norm(delta, axis=1)
    angles = np.degrees(np.arctan2(delta[:, 1], delta[:, 0])) % 180
    midpoint = np.rint((segments[:, :2] + segments[:, 2:]) / 2).astype(int)
    outside_text = text_mask[np.clip(midpoint[:, 1], 0, height-1), np.clip(midpoint[:, 0], 0, width-1)] == 0
    minimum = max(12., math.hypot(width, height) * .006)
    oblique = outside_text & (lengths >= minimum) & (_angle_difference(angles, 0) > 12) & (_angle_difference(angles, 90) > 12)
    histogram, _ = np.histogram(angles[oblique], bins=180, range=(0, 180), weights=lengths[oblique])
    dominant = float(np.argmax(histogram) + .5) if oblique.any() else 135.
    seed = oblique & (_angle_difference(angles, dominant) <= 2)
    if seed.any():
        dominant = float(np.average(angles[seed], weights=lengths[seed]))
    selected = outside_text & (lengths >= max(9., minimum * .65)) & (_angle_difference(angles, dominant) <= 2)
    hatch = np.zeros_like(ink)
    for segment in segments[selected]:
        x1, y1, x2, y2 = np.rint(segment).astype(int)
        cv2.line(hatch, (x1, y1), (x2, y2), 255, 3)
    barrier = cv2.dilate(ink, np.ones((3, 3), np.uint8))
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(255-barrier, connectivity=4)
    near_hatch = cv2.dilate(hatch, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    near_barrier = cv2.dilate(barrier, np.ones((3, 3), np.uint8))
    hatch_contact = np.bincount(labels[(near_hatch > 0) & (barrier == 0)], minlength=component_count)
    boundary_contact = np.bincount(labels[(near_barrier > 0) & (barrier == 0)], minlength=component_count)
    # Distinguish genuine long hatch sides from many hatch endpoints touching
    # an exterior dimension wedge. No periodic line spacing is assumed.
    support_offsets = [[] for _ in range(component_count)]
    normal = np.array([-math.sin(math.radians(dominant)), math.cos(math.radians(dominant))])
    for segment, length in zip(segments[selected], lengths[selected]):
        samples = max(8, int(length/2))
        centres = segment[:2] + np.linspace(.08, .92, samples)[:, None] * (segment[2:]-segment[:2])
        offset = float(((segment[:2]+segment[2:])/2) @ normal)
        for sign in (-1, 1):
            probe = np.rint(centres + sign*5*normal).astype(int)
            cell_ids = labels[np.clip(probe[:, 1], 0, height-1), np.clip(probe[:, 0], 0, width-1)]
            values, counts = np.unique(cell_ids, return_counts=True)
            for label, count in zip(values, counts):
                if label and count >= max(5, samples*.3):
                    support_offsets[label].append(offset)
    ys, xs = np.nonzero(barrier == 0)
    normal_offsets = xs*normal[0] + ys*normal[1]
    minimum_offset = np.full(component_count, np.inf)
    maximum_offset = np.full(component_count, -np.inf)
    np.minimum.at(minimum_offset, labels[ys, xs], normal_offsets)
    np.maximum.at(maximum_offset, labels[ys, xs], normal_offsets)
    normal_spans = maximum_offset-minimum_offset
    chosen, loose = [], []
    image_area = width * height
    if selected.sum() >= 12:
        for label in range(1, component_count):
            x, y, w, h, area = stats[label]
            if area < 12 or area > image_area * .18 or x <= 0 or y <= 0 or x+w >= width or y+h >= height:
                continue
            offsets = support_offsets[label]
            supported_strip = len(offsets) >= 2 and max(offsets)-min(offsets) >= 6
            if hatch_contact[label] >= 20 and hatch_contact[label] / max(1, boundary_contact[label]) >= .32:
                loose.append(label)
                if supported_strip:
                    chosen.append(label)
    # Recover boundary slivers with a single hatch side only when their normal
    # width fits locally observed inter-hatch cells; broad external wedges fail.
    strip_width = float(np.percentile(normal_spans[chosen], 90)) if chosen else 0.
    strip_limit = min(math.hypot(width, height)*.06, strip_width*1.4)
    if chosen:
        chosen = [label for label in loose if label in chosen or normal_spans[label] <= strip_limit]
    material = np.isin(labels, chosen).astype(np.uint8) * 255
    # Local closure repairs omitted narrow cells; no periodic pitch is fitted.
    closing = max(11, int(math.hypot(width, height) * .006) | 1)
    material = cv2.morphologyEx(material, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing, closing)))
    contours, _ = cv2.findContours(material, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    candidates = []
    primary_mask = np.zeros_like(ink)
    polyline, raw_polyline, cleanup = [], [], []
    evidence = {"method": "phase-free-directional-hatch-enclosed-cells-v1", "processing_scale": scale,
                "global_hatch_phase_required": False, "source_ocr_polygons": text_count,
                "detected_line_segments": len(segments), "hatch_segments": int(selected.sum()),
                "detected_hatch_angle_deg": dominant, "selected_material_cells": len(chosen),
                "connected_material_candidates": len(contours), "morphological_gap_closure_px": closing/scale,
                "observed_strip_width_px": strip_width/scale, "maximum_recovered_strip_width_px": strip_limit/scale,
                "confidence_kind": "heuristic evidence, not calibrated accuracy probability"}
    for index, contour in enumerate(contours[:8]):
        if cv2.contourArea(contour) < max(image_area * .001, cv2.contourArea(contours[0]) * .04):
            continue
        island = np.zeros_like(ink)
        cv2.drawContours(island, [contour], -1, 255, cv2.FILLED)
        island = cv2.dilate(island, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        outer, _ = cv2.findContours(island, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = max(outer, key=cv2.contourArea)
        simplified = cv2.approxPolyDP(contour, 1.2, True).reshape(-1, 2)
        raw = [[round(float(x)/scale, 4), round(float(y)/scale, 4)] for x, y in simplified]
        raw.append(raw[0])
        simplified, changes = _remove_local_detours(simplified, closing)
        closed = [[round(float(x)/scale, 4), round(float(y)/scale, 4)] for x, y in simplified]
        closed.append(closed[0])
        x, y, w, h = cv2.boundingRect(contour)
        candidates.append({"id": f"material-{index}", "polyline_px": closed, "raw_polyline_px": raw,
                           "area_px": cv2.contourArea(contour)/(scale*scale),
                           "area_ratio": cv2.contourArea(contour)/image_area,
                           "bounds_px": [x/scale, y/scale, (x+w)/scale, (y+h)/scale]})
        if index == 0 and cv2.contourArea(contour) > image_area * .003:
            polyline, raw_polyline, cleanup, primary_mask = closed, raw, changes, island
    confidence = 0.
    issues = ["Fallback silhouette is a draft; dimensions and detailed geometry are not verified."]
    if polyline:
        edge = np.zeros_like(ink)
        cv2.polylines(edge, [np.rint(np.asarray(polyline)*scale).astype(np.int32)], True, 255, 1)
        distance = cv2.distanceTransform(255-ink, cv2.DIST_L2, 3)[edge > 0]
        support = float(np.mean(distance <= 3))
        dominance = candidates[0]["area_px"] / max(1, sum(c["area_px"] for c in candidates))
        confidence = min(.8, .35*support + .25*dominance + .2*min(1, selected.sum()/35))
        evidence.update(edge_ink_support=support, max_edge_to_ink_px=float(distance.max())/scale,
                        mean_edge_to_ink_px=float(distance.mean())/scale, closure_gap_px=0.,
                        local_detour_shortcuts=len(cleanup), polyline_vertices=len(polyline)-1,
                        cleanup_scope="Local returning detours removed; raw vertices retained. Real narrow grooves may be suppressed.")
        if len(candidates) > 1 and candidates[1]["area_px"] > candidates[0]["area_px"]*.2:
            issues.append("Multiple substantial material islands retained; largest alone may be incomplete.")
    else:
        issues.append("No sufficiently large directional-hatch-supported enclosed region found; unhatched outlines are unresolved.")
    result = {"status": "needs_review", "polyline_px": polyline, "raw_polyline_px": raw_polyline,
              "candidates": candidates, "primary_candidate_id": "material-0" if polyline else None,
              "image_size": {"width": original_width, "height": original_height},
              "confidence": round(confidence, 4), "evidence": evidence, "issues": issues,
              "geometry_scope": "source-pixel material silhouette draft; no dimensional binding or engineering acceptance", "artifacts": {}}
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_image(output_dir/"material-mask.png", primary_mask)
        _write_image(output_dir/"all-material-cells.png", material)
        _write_image(output_dir/"hatch-evidence.png", hatch)
        overlay = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
        if polyline:
            cv2.polylines(overlay, [np.rint(np.asarray(polyline)*scale).astype(np.int32)], True, (0, 0, 230), 3)
        _write_image(output_dir/"overlay.png", overlay)
        points = " ".join(f"{x},{y}" for x, y in polyline)
        (output_dir/"candidate.svg").write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {original_width} {original_height}"><polygon points="{points}" fill="#dbe9e2" stroke="#087f72" stroke-width="2"/></svg>', encoding="utf-8")
        result["artifacts"] = {"mask": "material-mask.png", "hatch": "hatch-evidence.png", "overlay": "overlay.png", "svg": "candidate.svg"}
        (output_dir/"result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
