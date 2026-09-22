"""Generic hatch-supported material silhouette proposals from drawing images.

This module reads source pixels only. It has no case IDs, dimensional template,
reference-coordinate access or API call. Coordinates are original-image pixels.
The result is a raster candidate, not dimension-certified engineering geometry.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def _write_image(path, image):
    suffix = path.suffix.lower()
    okay, encoded = cv2.imencode(suffix, image)
    if not okay:
        raise ValueError("Could not encode raster diagnostic")
    path.write_bytes(encoded.tobytes())


def _angle_difference(a, b):
    return np.abs((a - b + 90) % 180 - 90)


def _remove_local_detours(points, hatch_pitch):
    """Suppress short returning annotation notches, preserving the raw proposal.

    A shortcut requires both surrounding tangents to agree with its direction
    and a path/chord ratio above 1.8. This preserves ordinary straight-to-arc
    corners but may erase real narrow grooves; it is disclosed pixel cleanup.
    """
    points = np.asarray(points, dtype=float)
    changes = []
    limit = max(12., float(hatch_pitch or 25) * 1.7)
    for _ in range(40):
        best = None
        count = len(points)
        if count < 8:
            break
        for i in range(count):
            incoming = points[i] - points[(i-1) % count]
            path_length = 0.
            for step in range(1, min(28, count//3)):
                j, previous = (i+step) % count, (i+step-1) % count
                path_length += float(np.linalg.norm(points[j] - points[previous]))
                if path_length > limit * 4:
                    break
                if step < 2:
                    continue
                chord = points[j] - points[i]
                length = float(np.linalg.norm(chord))
                if not 3 < length < limit or path_length < 1.8 * length:
                    continue
                outgoing = points[(j+1) % count] - points[j]
                if np.dot(incoming, chord) < .45 * np.linalg.norm(incoming) * length:
                    continue
                if np.dot(outgoing, chord) < .45 * np.linalg.norm(outgoing) * length:
                    continue
                score = path_length - length
                if best is None or score > best[0]:
                    best = score, i, step, length, path_length
        if best is None:
            break
        _, i, step, chord_length, path_length = best
        remove = {(i+k) % count for k in range(1, step)}
        changes.append({"chord_px": chord_length, "removed_path_px": path_length, "vertices_removed": len(remove)})
        points = np.array([point for index, point in enumerate(points) if index not in remove])
    return points, changes


def extract_main_profile(image_path: str | Path, output_dir: str | Path | None = None, *, max_dimension=2200) -> dict:
    """Return one closed main silhouette candidate and transparent evidence.

    ``polyline_px`` uses original image x-right/y-down pixels. Failure yields an
    empty polyline and needs_review. Confidence is a heuristic evidence score,
    never a calibrated probability or geometry-accuracy claim.
    """
    image_path = Path(image_path)
    with Image.open(image_path) as source:
        source = source.convert("RGB")
        original_width, original_height = source.size
        if original_width * original_height > 80_000_000:
            raise ValueError("Image exceeds the 80 million pixel processing limit")
        rgb = np.asarray(source)
    scale = min(1.0, max_dimension / max(original_width, original_height))
    resized = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else rgb
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_NONE)
    detected = lsd.detect(gray)[0]
    segments = detected.reshape(-1, 4) if detected is not None else np.empty((0, 4))
    delta = segments[:, 2:] - segments[:, :2]
    lengths = np.linalg.norm(delta, axis=1)
    angles = np.degrees(np.arctan2(delta[:, 1], delta[:, 0])) % 180
    min_length = max(12.0, math.hypot(width, height) * .009)
    oblique = (lengths >= min_length) & (_angle_difference(angles, 0) > 12) & (_angle_difference(angles, 90) > 12)
    histogram, _ = np.histogram(angles[oblique], bins=180, range=(0, 180), weights=lengths[oblique])
    dominant = float(np.argmax(histogram) + .5) if oblique.any() else 135.0
    direction_seed = oblique & (_angle_difference(angles, dominant) <= 2.0)
    if direction_seed.any():
        dominant = float(np.average(angles[direction_seed], weights=lengths[direction_seed]))
    normal = np.array([-math.sin(math.radians(dominant)), math.cos(math.radians(dominant))])
    offsets = ((segments[:, :2] + segments[:, 2:]) / 2) @ normal
    pitch, coherence, phase = None, 0.0, 0.0
    selected = oblique & (_angle_difference(angles, dominant) <= 1.5)
    if selected.sum() >= 8:
        offset_hist = np.bincount(np.rint(offsets[selected] - offsets.min()).astype(int), weights=lengths[selected])
        offset_hist = np.convolve(offset_hist, np.ones(3), mode="same")
        autocorrelation = np.correlate(offset_hist, offset_hist, mode="full")[len(offset_hist)-1:]
        lo, hi = 8, min(len(autocorrelation)-1, int(math.hypot(width, height) * .07))
        peaks = [i for i in range(lo+1, hi) if autocorrelation[i] >= autocorrelation[i-1] and autocorrelation[i] > autocorrelation[i+1]]
        if peaks:
            strongest = max(peaks, key=lambda i: autocorrelation[i])
            if autocorrelation[strongest] / max(1, autocorrelation[0]) > .3:
                periods = np.linspace(strongest-1.5, strongest+1.5, 151)
                complex_votes = np.exp(2j * math.pi * offsets[selected, None] / periods[None, :])
                votes = (complex_votes * lengths[selected, None]).sum(axis=0) / lengths[selected].sum()
                best = int(np.argmax(np.abs(votes)))
                pitch, coherence = float(periods[best]), float(abs(votes[best]))
                phase = float(np.angle(votes[best]) * pitch / (2 * math.pi))
                phase_error = np.abs((offsets-phase+pitch/2) % pitch-pitch/2)
                selected = (lengths >= 7) & (_angle_difference(angles, dominant) <= 2.5) & (phase_error <= max(2.5, pitch*.055))
    hatch = np.zeros_like(ink)
    for line in segments[selected]:
        x1, y1, x2, y2 = np.rint(line).astype(int)
        cv2.line(hatch, (x1, y1), (x2, y2), 255, 3)

    # A small gap closure repairs antialias cracks while keeping the original
    # ink mask separate for later boundary support measurements.
    barrier = cv2.dilate(ink, np.ones((3, 3), np.uint8))
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(255 - barrier, connectivity=4)
    near_hatch = cv2.dilate(hatch, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    near_barrier = cv2.dilate(barrier, np.ones((3, 3), np.uint8))
    hatch_contact = np.bincount(labels[(near_hatch > 0) & (barrier == 0)].ravel(), minlength=component_count)
    boundary_contact = np.bincount(labels[(near_barrier > 0) & (barrier == 0)].ravel(), minlength=component_count)
    hatch_sides = np.ones(component_count, dtype=int) * 2
    cell_normal_span = np.zeros(component_count)
    if pitch:
        ys, xs = np.nonzero((near_hatch > 0) & (barrier == 0))
        lattice_index = np.rint((xs * normal[0] + ys * normal[1] - phase) / pitch).astype(int)
        lattice_index -= lattice_index.min(initial=0)
        span = lattice_index.max(initial=0) + 1
        pair_counts = np.bincount(labels[ys, xs] * span + lattice_index, minlength=component_count * span).reshape(component_count, span)
        hatch_sides = (pair_counts >= 6).sum(axis=1)
        ys, xs = np.nonzero(barrier == 0)
        pixel_offsets = xs * normal[0] + ys * normal[1]
        min_offsets, max_offsets = np.full(component_count, np.inf), np.full(component_count, -np.inf)
        np.minimum.at(min_offsets, labels[ys, xs], pixel_offsets)
        np.maximum.at(max_offsets, labels[ys, xs], pixel_offsets)
        cell_normal_span = max_offsets - min_offsets
    chosen = []
    image_area = width * height
    for label in range(1, component_count):
        x, y, w, h, area = stats[label]
        if area < 12 or area > image_area * .22 or x <= 0 or y <= 0 or x + w >= width or y + h >= height:
            continue
        contact = hatch_contact[label]
        # Interior inter-hatch cells have long parallel supported boundaries;
        # external dimension boxes only touch occasional hatch endpoints.
        strip_cell = not pitch or hatch_sides[label] >= 2 or cell_normal_span[label] <= pitch * 1.1
        if contact >= 12 and contact / max(1, boundary_contact[label]) >= .24 and strip_cell:
            chosen.append(label)
    material = np.isin(labels, chosen).astype(np.uint8) * 255
    closing = max(11, int((pitch or 20) * .95) | 1)
    material = cv2.morphologyEx(material, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing, closing)))
    contours, _ = cv2.findContours(material, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    candidate = np.zeros_like(ink)
    issues = []
    polyline = []
    raw_polyline = []
    candidates = []
    confidence = 0.0
    evidence = {"method": "dominant-hatch-supported-enclosed-white-cells-v1", "processing_scale": scale,
                "detected_line_segments": len(segments), "detected_hatch_angle_deg": dominant,
                "hatch_segments": int(selected.sum()), "hatch_spacing_px": pitch / scale if pitch else None,
                "hatch_lattice_coherence": coherence, "enclosed_cells": component_count - 1,
                "selected_material_cells": len(chosen), "connected_material_candidates": len(contours),
                "morphological_gap_closure_px": closing / scale,
                "confidence_kind": "heuristic evidence, not calibrated accuracy probability"}
    for index, item in enumerate(contours[:8]):
        area = float(cv2.contourArea(item))
        if area < max(image_area * .001, cv2.contourArea(contours[0]) * .04):
            continue
        island_mask = np.zeros_like(ink)
        cv2.drawContours(island_mask, [item], -1, 255, cv2.FILLED)
        # Compensate the one-pixel barrier dilation and inner stroke edge in
        # every retained island, using the same rule as the primary contour.
        island_mask = cv2.dilate(island_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        island_contours, _ = cv2.findContours(island_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        item = max(island_contours, key=cv2.contourArea)
        area = float(cv2.contourArea(item))
        points = cv2.approxPolyDP(item, 1.2, True).reshape(-1, 2)
        closed = [[round(float(x)/scale, 4), round(float(y)/scale, 4)] for x, y in points]
        if closed:
            closed.append(closed[0])
        x, y, w, h = cv2.boundingRect(item)
        candidates.append({"id": f"material-{index}", "polyline_px": closed,
                           "area_px": area / (scale * scale), "area_ratio": area / image_area,
                           "bounds_px": [x/scale, y/scale, (x+w)/scale, (y+h)/scale]})
    if contours and cv2.contourArea(contours[0]) > image_area * .003:
        cv2.drawContours(candidate, contours[:1], -1, 255, cv2.FILLED)
        candidate = cv2.dilate(candidate, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        final, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = max(final, key=cv2.contourArea)
        simplified = cv2.approxPolyDP(contour, 1.2, True).reshape(-1, 2)
        raw_polyline = [[round(float(x) / scale, 4), round(float(y) / scale, 4)] for x, y in simplified]
        raw_polyline.append(raw_polyline[0])
        simplified, cleanup = _remove_local_detours(simplified, pitch)
        polyline = [[round(float(x) / scale, 4), round(float(y) / scale, 4)] for x, y in simplified]
        polyline.append(polyline[0])
        if candidates:
            candidates[0]["raw_polyline_px"] = raw_polyline
            candidates[0]["polyline_px"] = polyline
        # Score the exported cleaned outline rather than the pre-cleanup mask;
        # a shortcut unsupported by source ink must reduce evidence support.
        exported_edge = np.zeros_like(ink)
        cv2.polylines(exported_edge, [np.rint(simplified).astype(np.int32)], True, 255, 1)
        edge_y, edge_x = np.nonzero(exported_edge)
        distance = cv2.distanceTransform(255 - ink, cv2.DIST_L2, 3)
        support_distances = distance[edge_y, edge_x]
        support = float(np.mean(support_distances <= 3))
        area = float(cv2.contourArea(contour))
        dominance = area / max(1, sum(cv2.contourArea(c) for c in contours))
        confidence = float(np.clip(.3 * support + .2 * min(1, selected.sum() / 35) + .2 * dominance + .2 * coherence, 0, .9))
        evidence.update(mask_area_ratio=area / image_area, edge_ink_support=support,
                        max_edge_to_ink_px=float(support_distances.max()) / scale,
                        mean_edge_to_ink_px=float(support_distances.mean()) / scale,
                        dominant_component_fraction=min(1.0, dominance), closure_gap_px=0.0,
                        polyline_vertices=len(polyline) - 1,
                        local_detour_shortcuts=len(cleanup),
                        max_shortcut_chord_px=max((c["chord_px"] for c in cleanup), default=0) / scale,
                        cleanup_scope="Local returning detours removed; raw_polyline_px retained. Real narrow grooves may also be suppressed.")
        if support < .7:
            issues.append("Some inferred boundary lacks nearby source ink; shape may bridge gaps or miss details.")
        if selected.sum() < 12:
            issues.append("Insufficient repeated hatch-line evidence.")
        if dominance < .65:
            issues.append("Multiple material regions compete; the largest candidate may omit another relevant region.")
        if len(candidates) > 1 and candidates[1]["area_px"] > candidates[0]["area_px"] * .2:
            issues.append("Multiple substantial material islands retained in candidates; no assumption that largest alone is the complete part.")
    else:
        issues.append("No sufficiently large hatch-supported closed region found.")
    result = {"status": "candidate" if polyline and confidence >= .7 and not issues else "needs_review",
              "polyline_px": polyline, "raw_polyline_px": raw_polyline,
              "candidates": candidates, "primary_candidate_id": "material-0" if polyline else None,
              "image_size": {"width": original_width, "height": original_height},
              "confidence": round(confidence, 4), "evidence": evidence, "issues": issues,
              "geometry_scope": "source-pixel main material silhouette; no dimensional binding or engineering acceptance", "artifacts": {}}
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_image(output_dir / "material-mask.png", candidate)
        _write_image(output_dir / "all-material-cells.png", material)
        _write_image(output_dir / "hatch-evidence.png", hatch)
        overlay = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
        boundary, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if polyline:
            cv2.polylines(overlay, [np.rint(np.asarray(polyline) * scale).astype(np.int32)], True, (0, 0, 230), 3)
        _write_image(output_dir / "overlay.png", overlay)
        points = " ".join(f"{x},{y}" for x, y in polyline)
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {original_width} {original_height}"><polygon points="{points}" fill="#dbe9e2" stroke="#087f72" stroke-width="2"/></svg>'
        (output_dir / "candidate.svg").write_text(svg, encoding="utf-8")
        result["artifacts"] = {"mask": "material-mask.png", "hatch": "hatch-evidence.png", "overlay": "overlay.png", "svg": "candidate.svg"}
        (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
