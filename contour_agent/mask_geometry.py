"""Convert source-aligned segmentation masks into pixel contour proposals.

No model, OCR, template, reference geometry or case identity is consumed here.
The caller must remove letterbox padding before supplying a mask. Contours are
the outer boundaries of thresholded material components, not dimension-certified
CAD geometry. Interior holes are diagnosed but are intentionally not exported.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
from shapely import contains_xy
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from shapely.validation import explain_validity


def _size(original_size):
    values = (original_size.get("width"), original_size.get("height")) if isinstance(original_size, dict) else original_size
    if not isinstance(values, (tuple, list)) or len(values) != 2:
        raise ValueError("original_size must be (width, height) or a width/height mapping")
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v <= 0 for v in values):
        raise ValueError("Original image dimensions must be positive integers")
    width, height = map(int, values)
    if width*height > 80_000_000:
        raise ValueError("Original image exceeds the 80 million pixel limit")
    return width, height


def _normalize(mask):
    array = np.asarray(mask)
    if array.ndim != 2 or not array.size or min(array.shape) < 2:
        raise ValueError("mask must be a nonempty two-dimensional array at least 2 by 2")
    if array.size > 80_000_000:
        raise ValueError("Mask exceeds the 80 million pixel processing limit")
    if not (np.issubdtype(array.dtype, np.number) or array.dtype == np.bool_) or np.iscomplexobj(array):
        raise ValueError("mask must contain real numeric probabilities or binary values")
    values = array.astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("mask contains nonfinite values")
    if np.all((values == 0) | (values == 255)) and values.max() > 1:
        return values/255, "binary"
    if values.min() < 0 or values.max() > 1:
        raise ValueError("Probability values must be in [0,1]; binary masks may use 0/255")
    return values, "binary" if np.all((values == 0) | (values == 1)) else "probability"


def _closed(points):
    result = [[float(x), float(y)] for x, y in points]
    if result and result[-1] != result[0]:
        result.append(result[0].copy())
    return result


def _polygon(points):
    if len(points) < 3 or len(np.unique(points, axis=0)) < 3:
        return None, "Degenerate contour with fewer than three distinct vertices"
    polygon = Polygon(points)
    if not polygon.is_valid or not polygon.exterior.is_simple or polygon.area <= 0:
        return None, explain_validity(polygon)
    return polygon, None


def _write_png(path, array):
    success, encoded = cv2.imencode(".png", array)
    if not success:
        raise ValueError("Could not encode mask diagnostic")
    path.write_bytes(encoded.tobytes())


def _pixel_cell_parts(region):
    """Exact union of occupied unit cells; never buffer, close gaps or move ink."""
    rectangles = []
    for y, row in enumerate(region):
        changes = np.diff(np.pad(row.astype(np.int8), (1, 1)))
        for start, end in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
            if len(rectangles) >= 250_000:
                raise ValueError("Exact pixel-cell boundary exceeds the 250000 run limit")
            rectangles.append(box(start-.5, y-.5, end-.5, y+.5))
    material = unary_union(rectangles)
    parts = list(material.geoms) if material.geom_type == "MultiPolygon" else [material]
    if any(part.geom_type != "Polygon" or not part.is_valid for part in parts):
        raise ValueError("Exact pixel-cell union did not yield valid polygon components")
    if not math.isclose(sum(part.area for part in parts), int(region.sum()), rel_tol=0, abs_tol=1e-7):
        raise ValueError("Exact pixel-cell union failed foreground area conservation")
    return sorted(parts, key=lambda part: (-part.area, part.bounds)), len(rectangles)


def extract_mask_profile(mask, original_size, output_dir=None, *, threshold=.5,
                         min_component_area_ratio=.001, significant_island_ratio=.04,
                         simplify_tolerance_px=1.0):
    """Return the ``extract_main_profile`` interface from a segmentation mask.

    ``original_size`` is (width, height), or a mapping with those keys. Resized
    mask pixel centres map by ``(coordinate + .5) * original/mask - .5``.
    There is no crop or letterbox offset inference. ``simplify_tolerance_px``
    uses original-image pixels. Self-touching pixel-centre contours fall back
    to the exact union of occupied pixel cells. Corner-touching parts remain
    separate; no mask pixel, gap, bridge or connection is invented. Invalid
    simplifications revert to valid raw contours.

    Confidence describes only uncalibrated model foreground probabilities.
    Binary masks have no available model confidence (the numeric field is 0).
    A successful proposal remains ``needs_review`` regardless of confidence.
    """
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("threshold must be strictly between zero and one")
    for name, value in (("min_component_area_ratio", min_component_area_ratio), ("significant_island_ratio", significant_island_ratio)):
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0,1]")
    if not math.isfinite(simplify_tolerance_px) or simplify_tolerance_px < 0:
        raise ValueError("simplify_tolerance_px must be finite and nonnegative")
    original_width, original_height = _size(original_size)
    probability, input_kind = _normalize(mask)
    height, width = probability.shape
    sx, sy = original_width/width, original_height/height
    binary = (probability >= threshold).astype(np.uint8)
    foreground_count = int(binary.sum())
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    # Candidate-export filtering is not a connectivity test: even a small
    # detached material fragment must prevent a complete-part claim.
    count4, _, stats4, _ = cv2.connectedComponentsWithStats(binary, connectivity=4)
    areas4 = sorted((int(a) for a in stats4[1:, cv2.CC_STAT_AREA]), reverse=True)
    areas8 = sorted((int(a) for a in stats[1:, cv2.CC_STAT_AREA]), reverse=True)
    meaningful4 = [a for a in areas4 if a >= 3]
    meaningful8 = [a for a in areas8 if a >= 3]
    order = sorted(range(1, count), key=lambda label: (-stats[label, cv2.CC_STAT_AREA], label))
    largest_area = int(stats[order[0], cv2.CC_STAT_AREA]) if order else 0
    minimum_area = max(3, math.ceil(binary.size*min_component_area_ratio), math.ceil(largest_area*significant_island_ratio))
    candidates, rejected, conversions = [], [], []
    retained_pixels = 0
    retained_mask = np.zeros(binary.shape, np.uint8)
    for label in order:
        x, y, w, h, area = map(int, stats[label])
        if area < minimum_area:
            continue
        region = (labels[y:y+h, x:x+w] == label).astype(np.uint8)
        contours, hierarchy = cv2.findContours(region, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        hierarchy = hierarchy[0] if hierarchy is not None else []
        exteriors = [i for i, relation in enumerate(hierarchy) if relation[3] == -1]
        holes = [i for i, relation in enumerate(hierarchy) if relation[3] != -1]
        points = contours[exteriors[0]].reshape(-1, 2).astype(float) if len(exteriors) == 1 else np.empty((0,2))
        raw_polygon, reason = _polygon((points+[x,y]+.5)*[sx,sy]-.5)
        outlines = [(points, region, len(holes), sum(abs(cv2.contourArea(contours[i])) for i in holes), "pixel_center_contour")]
        if reason:
            try:
                parts, run_count = _pixel_cell_parts(region)
                ys, xs = np.nonzero(region)
                outlines = []
                for part in parts:
                    part_region = region
                    if len(parts) > 1:
                        selected = contains_xy(part, xs, ys)
                        part_region = np.zeros_like(region)
                        part_region[ys[selected], xs[selected]] = 1
                    outlines.append((np.asarray(part.exterior.coords)[:-1], part_region, len(part.interiors),
                                     sum(Polygon(ring).area for ring in part.interiors), "exact_foreground_pixel_cells"))
                conversions.append({"component_label":label,"trigger":reason,"method":"union_of_occupied_pixel_row_runs",
                                    "pixel_area_before":area,"pixel_cell_area_after":sum(part.area for part in parts),
                                    "part_count":len(parts),"row_run_count":run_count,"foreground_pixels_changed":0,
                                    "bridges_added":0,"mask_morphology_applied":False,
                                    "scope":"Boundary representation fallback; exact occupied cells, not shape repair or reference fitting"})
            except (ValueError, ArithmeticError) as error:
                rejected.append({"component_label":label,"area_mask_px":area,"reason":str(error)})
                continue
        for local_points, part_region, hole_count, hole_area, representation in outlines:
            points = (local_points+[x,y]+.5)*[sx,sy]-.5
            raw_polygon, reason = _polygon(points)
            if reason:
                rejected.append({"component_label":label,"area_mask_px":int(part_region.sum()),"reason":reason})
                continue
            proposed = cv2.approxPolyDP(points.astype(np.float32), simplify_tolerance_px, True).reshape(-1, 2).astype(float)
            polygon, simplification_reason = _polygon(proposed)
            if simplification_reason:
                proposed, polygon = points, raw_polygon
            part_area = int(part_region.sum())
            component_values = probability[y:y+h, x:x+w][part_region > 0]
            boundary = (part_region > 0) & (cv2.erode(part_region, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0) == 0)
            boundary_values = probability[y:y+h, x:x+w][boundary]
            candidate = {"id": f"material-{len(candidates)}", "component_label": label,
                         "polyline_px": _closed(proposed), "raw_polyline_px": _closed(points),
                         "area_px": float(polygon.area), "area_ratio": float(polygon.area/(original_width*original_height)),
                         "material_pixel_area_px": float(part_area*sx*sy), "material_pixels_mask": part_area,
                         "bounds_px": list(map(float, polygon.bounds)), "valid_closed_polygon": True,
                         "self_intersection": False, "hole_count": hole_count, "boundary_representation":representation,
                         "omitted_hole_contour_area_px": float(hole_area*sx*sy),
                         "touches_image_border": bool(x == 0 or y == 0 or x+w == width or y+h == height),
                         "simplification_reverted": bool(simplification_reason),
                         "simplification_max_boundary_deviation_px": float(raw_polygon.boundary.hausdorff_distance(polygon.boundary)),
                         "foreground_probability_mean": float(component_values.mean()) if input_kind == "probability" else None,
                         "foreground_probability_p10": float(np.percentile(component_values, 10)) if input_kind == "probability" else None,
                         "boundary_probability_mean": float(boundary_values.mean()) if input_kind == "probability" else None}
            candidates.append(candidate)
            retained_pixels += part_area
            retained_mask[y:y+h, x:x+w][part_region > 0] = 255
    candidates.sort(key=lambda candidate: -candidate["material_pixels_mask"])
    for index,candidate in enumerate(candidates): candidate["id"] = f"material-{index}"
    primary = candidates[0] if candidates else None
    connectivity = {"components_4": count4-1, "components_8": count-1,
                    "component_areas_4_mask_px": areas4, "component_areas_8_mask_px": areas8,
                    "material_component_minimum_mask_px": 3,
                    "meaningful_components_4": len(meaningful4), "meaningful_components_8": len(meaningful8),
                    "corner_only_connections": count4-count,
                    "tiny_noise_pixels": sum(a for a in areas4 if a < 3),
                    "single_material_region": len(meaningful4) == 1,
                    "complete_exterior_candidate": bool(primary and len(meaningful4) == 1 and len(candidates) == 1),
                    "exported_primary_only": True,
                    "scope": "Four-neighbour material connectivity before candidate filtering; ignores only components below three mask pixels. A connected exterior does not establish boundary, hole or dimensional accuracy."}
    issues = ["Segmentation contour is a source-pixel draft; mask confidence does not establish dimensional or manufacturing accuracy."]
    if not primary:
        issues.append("No sufficiently large valid material exterior was extracted.")
    if len(candidates) > 1:
        issues.append("Multiple significant material islands retained; the primary contour alone may not describe the complete part.")
    if len(meaningful4) > 1:
        issues.append("Material mask contains disconnected or corner-only-connected regions before candidate filtering; exporting its largest contour is an incomplete main-profile draft.")
    if any(c["hole_count"] for c in candidates):
        issues.append("Interior holes are present but only exterior silhouettes are exported; hole boundaries must be handled separately.")
    if any(c["touches_image_border"] for c in candidates):
        issues.append("Material reaches the image border; the closed contour may describe a clipped view.")
    if rejected:
        issues.append("Invalid or self-touching mask components were rejected without automatic topological repair.")
    if conversions:
        issues.append("A self-touching pixel-centre trace was replaced by exact occupied-pixel-cell boundaries; foreground mask, gaps and connections were preserved. This does not correct model segmentation errors.")
    confidence = primary["foreground_probability_mean"] if primary and input_kind == "probability" else 0.
    evidence = {"method": "segmentation-mask-exterior-contours-v2", "input_kind": input_kind,
                "mask_size": {"width": width, "height": height}, "threshold": float(threshold),
                "coordinate_mapping": {"kind": "pixel_center_resize", "scale_x": sx, "scale_y": sy,
                                       "formula": "original=(mask+0.5)*scale-0.5",
                                       "letterbox_handling": "not performed; caller must provide an unpadded mask"},
                "foreground_pixels": foreground_count, "foreground_coverage": foreground_count/binary.size,
                "retained_foreground_fraction": retained_pixels/foreground_count if foreground_count else 0.,
                "primary_foreground_fraction": primary["material_pixels_mask"]/foreground_count if primary else 0.,
                "connected_material_candidates": count-1, "retained_candidates": len(candidates),
                "connectivity": connectivity,
                "minimum_component_pixels": minimum_area, "rejected_components": rejected,
                "discarded_foreground_pixels": foreground_count-retained_pixels,
                "interior_holes_exported": False, "holes_omitted": sum(c["hole_count"] for c in candidates),
                "simplification_tolerance_original_px": float(simplify_tolerance_px),
                "ambiguous_probability_fraction": float(np.mean(np.abs(probability-threshold) <= .1)) if input_kind == "probability" else None,
                "confidence_kind": "uncalibrated model foreground probability; not correctness probability" if input_kind == "probability" else "unavailable for binary input; numeric confidence is zero",
                "self_intersections_checked": True, "topological_repair_applied": False,
                "boundary_representation_fallback":bool(conversions),"boundary_conversions":conversions,
                "foreground_mask_modified":False,"pixel_cell_boundary_convention":"Occupied cell centred at (x,y) spans [x-0.5,x+0.5] by [y-0.5,y+0.5]; the same declared resize transform applies.",
                "dimensions_verified": False, "engineering_certified": False}
    result = {"status": "needs_review", "polyline_px": primary["polyline_px"] if primary else [],
              "raw_polyline_px": primary["raw_polyline_px"] if primary else [], "candidates": candidates,
              "primary_candidate_id": primary["id"] if primary else None,
              "image_size": {"width": original_width, "height": original_height},
              "confidence": float(confidence), "evidence": evidence, "issues": issues,
              "geometry_scope": "source-pixel material exterior silhouettes only; no hole boundaries, dimensional binding or engineering acceptance",
              "artifacts": {}}
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_png(output_dir/"material-mask.png", retained_mask)
        _write_png(output_dir/"threshold-mask.png", binary*255)
        overlay = cv2.cvtColor(255-binary*180, cv2.COLOR_GRAY2BGR)
        polygons = []
        for index, candidate in enumerate(candidates):
            image_points = np.asarray(candidate["polyline_px"])
            mask_points = (image_points+.5)/[sx, sy]-.5
            cv2.polylines(overlay, [np.rint(mask_points).astype(np.int32)], True, (0,0,230) if index == 0 else (210,90,0), 1)
            coordinates = " ".join(f"{px:.6g},{py:.6g}" for px, py in candidate["polyline_px"])
            polygons.append(f'<polygon points="{coordinates}" fill="none" stroke="#087f72" stroke-width="1"/>')
        _write_png(output_dir/"mask-overlay.png", overlay)
        (output_dir/"candidate.svg").write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {original_width} {original_height}">{"".join(polygons)}</svg>', encoding="utf-8")
        result["artifacts"] = {"mask": "material-mask.png", "threshold_mask": "threshold-mask.png", "overlay": "mask-overlay.png", "svg": "candidate.svg"}
        (output_dir/"result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return result
