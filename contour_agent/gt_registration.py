"""Register an explicitly supplied CAD reference for TRAINING LABELS ONLY.

This module must not be imported by prediction or provider code. The input
reference is never repaired or deformed to fit the drawing. Registration quality
is image agreement, not independent geometric accuracy or human review.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from shapely.geometry import Polygon


_ORIENTATIONS = (
    ("identity", [[1, 0], [0, 1]]), ("quarter_turn", [[0, -1], [1, 0]]),
    ("half_turn", [[-1, 0], [0, -1]]), ("three_quarter_turn", [[0, 1], [-1, 0]]),
    ("reflect_x", [[-1, 0], [0, 1]]), ("reflect_y", [[1, 0], [0, -1]]),
    ("swap_xy", [[0, 1], [1, 0]]), ("swap_reflect", [[0, -1], [-1, 0]]),
)


def _ring(polygon_xy):
    points = np.asarray(polygon_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not 4 <= len(points) <= 300_000:
        raise ValueError("polygon_xy must be a closed Nx2 ring with 4..300000 points")
    if not np.isfinite(points).all() or not np.array_equal(points[0], points[-1]):
        raise ValueError("polygon_xy must be finite and explicitly closed")
    polygon = Polygon(points)
    if not polygon.is_valid or not polygon.exterior.is_simple or polygon.area <= 0:
        raise ValueError("polygon_xy must be a valid, non-self-intersecting exterior")
    return points


def _sample(points, count=600):
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    positions = np.r_[0, np.cumsum(lengths)]
    samples = np.linspace(0, positions[-1], count, endpoint=False)
    return np.c_[np.interp(samples, positions, points[:, 0]),
                 np.interp(samples, positions, points[:, 1])]


def _transform(points, matrix):
    return points @ matrix[:, :2].T + matrix[:, 2]


def _matrix(value):
    result = np.asarray(value, dtype=float)
    if result.shape != (2, 3) or not np.isfinite(result).all():
        raise ValueError("Each initial transform must be a finite 2x3 CAD-mm to source-pixel matrix")
    if abs(np.linalg.det(result[:, :2])) < 1e-12:
        raise ValueError("Initial transform must be invertible")
    return result


def _is_similarity(matrix):
    gram = matrix[:, :2].T @ matrix[:, :2]
    return bool(np.allclose(gram, np.eye(2)*np.trace(gram)/2, rtol=1e-5, atol=1e-8))


def _read_hint(hint, size):
    if hint is None:
        return np.zeros((size[1], size[0]), np.uint8)
    if isinstance(hint, (str, Path)):
        with Image.open(hint) as image:
            values = np.asarray(image.convert("L"))
    else:
        values = np.asarray(hint)
    if values.ndim != 2 or not values.size or values.size > 80_000_000:
        raise ValueError("foreground_hint must be a nonempty 2D source-aligned mask")
    if not np.isfinite(values).all():
        raise ValueError("foreground_hint contains nonfinite values")
    binary = (values > .5 if values.max() <= 1 else values > 127).astype(np.uint8)
    return cv2.resize(binary, size, interpolation=cv2.INTER_NEAREST)


def _components(binary):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    order = sorted(range(1, count), key=lambda i: (-stats[i, cv2.CC_STAT_AREA], i))
    if not order:
        return []
    minimum = max(25, stats[order[0], cv2.CC_STAT_AREA]*.08)
    return [(labels == i).astype(np.uint8) for i in order[:3] if stats[i, cv2.CC_STAT_AREA] >= minimum]


def _distance(binary):
    return cv2.distanceTransform((1-binary).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE).astype(float)


def _lookup(field, points, outside=100.):
    return map_coordinates(field, [points[:, 1], points[:, 0]], order=1,
                           mode="constant", cval=outside, prefilter=False)


def _render(points, shape):
    result = np.zeros(shape, np.uint8)
    # Safe drawing bound; clipping is only rasterization, never geometry repair.
    bounded = np.rint(np.clip(points, -1_000_000, 1_000_000)).astype(np.int32)
    cv2.fillPoly(result, [bounded.reshape(-1, 1, 2)], 1)
    return result


def _write_png(path, array):
    success, encoded = cv2.imencode(".png", array)
    if not success:
        raise ValueError("Could not encode registration diagnostic")
    path.write_bytes(encoded.tobytes())


def _global_edge_initializers(points, ink_dist, resize_xy, pixels_per_mm, allow_reflection):
    """Scan a fixed reference boundary over the image at source OCR scale.

    This is a training-label alignment search, not a source-only predictor.
    The reference's shape is fixed; only its similarity pose is searched. A
    partial foreground hint cannot restrict the translation search window.
    """
    if pixels_per_mm is None or not math.isfinite(pixels_per_mm) or pixels_per_mm <= 0:
        return []
    proximity = np.exp(-ink_dist/2.5).astype(np.float32)
    height, width = ink_dist.shape
    hypotheses = []
    for name, orientation in _ORIENTATIONS:
        orientation = np.asarray(orientation, float)
        if not allow_reflection and np.linalg.det(orientation) < 0:
            continue
        for factor in (.98, 1., 1.02):
            linear = (orientation*pixels_per_mm*factor)*resize_xy[:, None]
            rotated = points@linear.T
            lo = np.floor(rotated.min(axis=0))-2
            size = np.ceil(rotated.max(axis=0)-lo).astype(int)+3
            if min(size) < 8 or size[0] > width or size[1] > height:
                continue
            boundary = np.zeros((int(size[1]), int(size[0])), np.float32)
            cv2.polylines(boundary, [np.rint(rotated-lo).astype(np.int32)], True, 1., 1)
            perimeter = float(boundary.sum())
            if perimeter < 20:
                continue
            response = cv2.matchTemplate(proximity, boundary, cv2.TM_CCORR)/perimeter
            for peak_index in range(2):
                _, value, _, location = cv2.minMaxLoc(response)
                x, y = location
                hypotheses.append({"matrix": np.c_[linear, np.array([x, y])-lo],
                                   "source": f"global_edge:{name}:ocr_scale_{factor:g}:peak_{peak_index}",
                                   "provided": False, "similarity": True, "hint_index": None,
                                   "global_edge_scan_score": float(np.clip(value, 0, 1)),
                                   "global_search": True})
                # Preserve a spatially distinct second basin, not neighboring
                # pixels from the same peak. All orientations are still searched.
                rx, ry = max(6, int(size[0]*.08)), max(6, int(size[1]*.08))
                response[max(0, y-ry):min(response.shape[0], y+ry+1),
                         max(0, x-rx):min(response.shape[1], x+rx+1)] = -1
    hypotheses.sort(key=lambda item: -item["global_edge_scan_score"])
    # Bounded fine fitting: keep one top placement from every orientation,
    # then distinct remaining translation/scale hypotheses up to 16 total.
    kept, orientations = [], set()
    for item in hypotheses:
        orientation_name = item["source"].split(":")[1]
        if orientation_name not in orientations:
            kept.append(item)
            orientations.add(orientation_name)
    for item in hypotheses:
        if len(kept) >= 16:
            break
        if any(np.max(np.abs(item["matrix"]-other["matrix"])) < 3. for other in kept):
            continue
        kept.append(item)
    return kept


def register_training_polygon(image_path, polygon_xy, output_dir=None, *,
                              foreground_hint=None, ocr_document=None,
                              initial_transforms=None, allow_reflection=True,
                              max_dimension=1200):
    """Create an automatically registered DXF-derived training mask.

    All input/output matrices map original CAD mm to original image pixel
    centres. ``initial_transforms`` accepts matrices or {matrix, source} maps;
    supplied transforms are evaluated unchanged and separately refined when
    they are similarities. General affine calibrations are only evaluated fixed.
    ``foreground_hint`` must already correspond to the complete source view
    (resizing is supported; unknown crops/letterboxing are not).

    An exterior only is filled; any holes must be handled by a future explicit
    multi-ring label policy. Every result is an automatic, unreviewed label.
    """
    if isinstance(max_dimension, bool) or not isinstance(max_dimension, int) or not 128 <= max_dimension <= 2400:
        raise ValueError("max_dimension must be an integer in 128..2400")
    points = _ring(polygon_xy)
    image_path = Path(image_path)
    with Image.open(image_path) as image:
        width, height = image.size
        if width*height > 80_000_000 or min(width, height) < 2:
            raise ValueError("Source image size is unsupported")
        image_rgb = np.asarray(image.convert("RGB"))
    scale = min(1., max_dimension/max(width, height))
    working_size = (max(2, round(width*scale)), max(2, round(height*scale)))
    # OpenCV resize has separate centre mappings when integer dimensions round.
    resize_xy = np.array([working_size[0]/width, working_size[1]/height])
    image_work = cv2.resize(image_rgb, working_size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image_work, cv2.COLOR_RGB2GRAY)
    _, ink = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    ink_dist = _distance(ink)
    hint = _read_hint(foreground_hint, working_size)
    hint_components = _components(hint)
    samples = _sample(points)
    centre = (points.min(axis=0)+points.max(axis=0))/2
    component_fields = []
    for component in hint_components:
        boundary = component-cv2.erode(component, np.ones((3, 3), np.uint8))
        component_fields.append(_distance(boundary))

    def to_work(matrix):
        return np.c_[matrix[:, :2]*resize_xy[:, None], (matrix[:, 2]+.5)*resize_xy-.5]

    def to_original(matrix):
        return np.c_[matrix[:, :2]/resize_xy[:, None], (matrix[:, 2]+.5)/resize_xy-.5]

    initial, issues = [], []
    if initial_transforms is not None:
        if not isinstance(initial_transforms, (tuple, list)) or len(initial_transforms) > 32:
            raise ValueError("initial_transforms must be a sequence of at most 32 transforms")
        for index, item in enumerate(initial_transforms):
            supplied = _matrix(item.get("matrix", item.get("transform_2x3")) if isinstance(item, dict) else item)
            if not allow_reflection and np.linalg.det(supplied[:, :2]) < 0:
                issues.append(f"Supplied transform {index} omitted because reflection is disabled")
                continue
            initial.append({"matrix": to_work(supplied), "source": str(item.get("source", f"provided_{index}")) if isinstance(item, dict) else f"provided_{index}",
                            "provided": True, "similarity": _is_similarity(supplied), "hint_index": None})

    scale_prior = None
    if ocr_document is not None:
        from .dimension_evidence import estimate_scale
        evidence = estimate_scale(image_path, ocr_document)
        if evidence.get("status") == "resolved":
            scale_prior = float(evidence["pixels_per_mm"])
    for ci, component in enumerate(hint_components):
        ys, xs = np.nonzero(component)
        span = np.array([xs.max()-xs.min(), ys.max()-ys.min()], float)
        target_centre = np.array([(xs.max()+xs.min())/2, (ys.max()+ys.min())/2])
        for name, orientation in _ORIENTATIONS:
            orientation = np.asarray(orientation, float)
            if not allow_reflection and np.linalg.det(orientation) < 0:
                continue
            rotated = (points-centre) @ orientation.T
            extent = np.ptp(rotated*resize_xy, axis=0)
            estimate = float(np.dot(extent, span)/np.dot(extent, extent))
            scales = [("bbox", estimate)]
            if scale_prior and abs(math.log(scale_prior/estimate)) > .015:
                scales.append(("ocr_scale", scale_prior))
            for scale_source, candidate_scale in scales:
                linear = (orientation*candidate_scale)*resize_xy[:, None]
                initial.append({"matrix": np.c_[linear, target_centre-linear@centre],
                                "source": f"hint_{ci}:{name}:{scale_source}", "provided": False,
                                "similarity": True, "hint_index": ci})
    global_initial = _global_edge_initializers(points, ink_dist, resize_xy, scale_prior, allow_reflection)
    initial.extend(global_initial)
    base_result = {"method": "dxf-training-registration-v2", "usage": "training_label_or_evaluation_only",
                   "reviewed": False, "label_source": "dxf_registered_automatically",
                   "image_size": {"width": width, "height": height},
                   "working_size": {"width": working_size[0], "height": working_size[1]},
                   "source_image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                   "reference_polygon_sha256": hashlib.sha256(points.astype("<f8").tobytes()).hexdigest(),
                   "global_search": {"method": "full_frame_boundary_chamfer_at_source_ocr_scale",
                                     "pixels_per_mm": scale_prior, "candidate_count": len(global_initial)},
                   "geometry_changed": False, "holes_policy": "supplied_exterior_only",
                   "quality_scope": "image agreement heuristic; not independent accuracy or human verification"}
    if not initial or not ink.any() or int(gray.max()) == int(gray.min()):
        return {**base_result, "status": "needs_review", "registered_polyline_px": [],
                "transform_2x3": None, "similarity_transform": None, "quality": {},
                "candidates": [], "artifacts": {}, "issues": issues+["No source foreground initializer/calibration or no source ink; registration not attempted"]}

    def quality(matrix):
        placed = _transform(points, matrix)
        placed_samples = _transform(samples, matrix)
        inside = ((placed_samples[:, 0] >= -.5) & (placed_samples[:, 0] <= working_size[0]-.5)
                  & (placed_samples[:, 1] >= -.5) & (placed_samples[:, 1] <= working_size[1]-.5))
        rendered = _render(placed, gray.shape)
        total = int(rendered.sum())
        distances = _lookup(ink_dist, placed_samples)
        edge_support = float(np.mean(distances <= 3.))
        hint_total, overlap = int(hint.sum()), int((rendered & hint).sum())
        ious = [float((rendered & c).sum()/max(1, (rendered | c).sum())) for c in hint_components]
        hint_iou = max(ious, default=0.)
        # Missing hints are missing evidence, not zero-quality geometry.
        edge_closeness = float(np.mean(np.exp(-distances/3)))
        image_score = .65*edge_support+.35*edge_closeness
        score = (.7*image_score+.3*hint_iou if hint_total else image_score)*float(inside.mean())
        return {"frame_inside_ratio": float(inside.mean()), "edge_support": edge_support,
                "edge_support_tolerance_working_px": 3., "edge_closeness": edge_closeness,
                "hint_precision": float(overlap/total) if hint_total and total else None,
                "hint_coverage": float(overlap/hint_total) if hint_total else None,
                "best_component_hint_iou": hint_iou if hint_total else None,
                "foreground_fraction": float(total/rendered.size), "score": float(np.clip(score, 0, 1))}

    candidates = []
    def append_candidate(matrix, item, variant, nfev=0):
        result = {"candidate_id": len(candidates), "source": item["source"], "provided": item["provided"],
                  "variant": variant, "refinement_evaluations": int(nfev),
                  "transform_kind": "similarity" if item["similarity"] else "provided_affine_fixed",
                  "transform_2x3": to_original(matrix).tolist(), "quality": quality(matrix)}
        if item.get("global_search"):
            result["global_edge_scan_score"] = item["global_edge_scan_score"]
        candidates.append(result)

    for item in initial:
        initial_matrix = item["matrix"]
        append_candidate(initial_matrix, item, "unoptimized")
        if not item["similarity"]:
            continue
        # Refine in original-coordinate similarity parameters, preserving exact
        # isotropy despite integer working-image resize dimensions.
        original_matrix = to_original(initial_matrix)
        initial_scale = math.sqrt(abs(np.linalg.det(original_matrix[:, :2])))
        basis = original_matrix[:, :2]/initial_scale
        t0 = initial_matrix[:, :2]@centre+initial_matrix[:, 2]
        hint_index = item["hint_index"]
        if hint_index is None and component_fields and not item.get("global_search"):
            hint_index = int(np.argmin([np.mean(np.minimum(_lookup(field, _transform(samples, initial_matrix)), 40)) for field in component_fields]))
        hint_field = component_fields[hint_index] if hint_index is not None else None
        extent = np.ptp(_transform(points, initial_matrix), axis=0)
        translation_bound = max(12., float(max(extent))*(.045 if item["provided"] else .22))
        log_bound, angle_bound = (.05, math.radians(3)) if item["provided"] else (.38, math.radians(12))
        if item.get("global_search"):
            # OCR already supplies physical scale; a partial weak mask must not
            # pull this full-frame initializer onto a smaller material fragment.
            log_bound, angle_bound = .045, math.radians(3)
            translation_bound = max(8., float(max(extent))*.04)

        def parameters_matrix(params):
            log_scale, angle, tx, ty = params
            co, si = math.cos(angle), math.sin(angle)
            linear = (np.array([[co, -si], [si, co]]) @ basis)*(initial_scale*math.exp(log_scale))
            linear *= resize_xy[:, None]
            return np.c_[linear, t0+[tx, ty]-linear@centre]

        def residual(params):
            placed = _transform(samples, parameters_matrix(params))
            image_residual = np.minimum(_lookup(ink_dist, placed), 35.)
            if hint_field is not None:
                hint_residual = np.minimum(_lookup(hint_field, placed), 35.)*.35
                return np.r_[image_residual, hint_residual]
            return image_residual

        limits = np.array([log_bound, angle_bound, translation_bound, translation_bound])
        optimized = least_squares(residual, np.zeros(4), bounds=(-limits, limits), loss="soft_l1",
                                  f_scale=3., max_nfev=38, diff_step=None, ftol=1e-5, xtol=1e-5)
        append_candidate(parameters_matrix(optimized.x), item, "refined", optimized.nfev)

    # Prefer supplied unchanged calibration in an exact quality tie.
    candidates.sort(key=lambda c: (-c["quality"]["score"], not c["provided"], c["variant"] != "unoptimized", c["candidate_id"]))
    best = candidates[0]
    best_matrix = np.asarray(best["transform_2x3"])
    registered = _transform(points, best_matrix)
    # Ignore equivalent matrices from neighboring initializers in ambiguity:
    # compare filled silhouettes, which also handles exact shape symmetries.
    best_mask_work = _render((registered+.5)*resize_xy-.5, gray.shape)
    alternatives = []
    for candidate in candidates[1:]:
        candidate_points = _transform(points, to_work(np.asarray(candidate["transform_2x3"])))
        candidate_mask = _render(candidate_points, gray.shape)
        agreement = float((best_mask_work & candidate_mask).sum()/max(1, (best_mask_work | candidate_mask).sum()))
        candidate["iou_with_selected"] = agreement
        if agreement < .90:
            alternatives.append(candidate)
    ambiguity_margin = best["quality"]["score"]-alternatives[0]["quality"]["score"] if alternatives else 1.
    result_quality = {**best["quality"], "ambiguity_margin": float(ambiguity_margin),
                      "ambiguity_competitor_id": alternatives[0]["candidate_id"] if alternatives else None}
    if ambiguity_margin < .025:
        issues.append("A geometrically different placement has similar image agreement; registration is ambiguous")
    if result_quality["frame_inside_ratio"] < .995:
        issues.append("Reference boundary is partly outside the source frame")
    if result_quality["edge_support"] < .65:
        issues.append("Limited source-ink boundary support; automatic mask requires review")
    if not hint.any():
        issues.append("No independent source foreground hint was available")
    issues.append("Automatically registered exterior training label; interior holes and human verification are not supplied")
    result = {**base_result, "status": "registration_candidate", "selected_candidate_id": best["candidate_id"],
              "transform_2x3": best["transform_2x3"],
              "similarity_transform": best["transform_2x3"] if best["transform_kind"] == "similarity" else None,
              "registered_polyline_px": registered.tolist(), "quality": result_quality,
              "candidates": candidates, "issues": issues, "artifacts": {}}
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        original_mask = _render(registered, (height, width))*255
        mask_path, overlay_path = output/"mask.png", output/"overlay.png"
        _write_png(mask_path, original_mask)
        overlay = image_work.copy()
        material = best_mask_work.astype(bool)
        overlay[material] = (overlay[material]*.73+np.array([40, 170, 240])*.27).astype(np.uint8)
        outline = np.rint((registered+.5)*resize_xy-.5).astype(np.int32)
        cv2.polylines(overlay, [outline], True, (225, 40, 50), 2, cv2.LINE_AA)
        _write_png(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        result["artifacts"] = {"mask": str(mask_path.resolve()), "overlay": str(overlay_path.resolve()),
                               "report": str((output/"registration.json").resolve())}
        (output/"registration.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
