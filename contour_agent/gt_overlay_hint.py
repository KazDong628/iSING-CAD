"""Offline training-registration hints from an official GT package overlay.

Shared grayscale drawing background establishes overlay-to-source alignment.
Colored GT strokes yield only a localization hint; final supervision must still
render the DXF. This module is never a runtime prediction input or a gold score.
"""
from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

import cv2
import numpy as np
from PIL import Image


def _eligible(name):
    stem = PurePosixPath(name).stem.lower()
    return (PurePosixPath(name).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
            and "overlay" in stem
            and not re.search(r"zoom|debug|label|dimension|(?:^|[_-])dim(?:[_-]|$)|detail|upper[_-]chain|lower[_-]chain|central|comparison|(?:^|[_-])vs(?:[_-]|$)", stem))


def _rank(candidate):
    name = str(candidate.get("member") or candidate["path"]).lower()
    return (0 if "main_overlay" in name or "overlay_full" in name else 1, name)


def _candidates(reference_source):
    reference = Path(reference_source["path"]).resolve()
    if reference.suffix.lower() == ".zip":
        with zipfile.ZipFile(reference) as archive:
            result = [{"path": str(reference), "member": info.filename} for info in archive.infolist()
                      if _eligible(info.filename) and info.file_size <= 40_000_000
                      and ".." not in PurePosixPath(info.filename.replace("\\", "/")).parts]
    else:
        # Select exactly this GT package, not the entire GT or a prior vis tree.
        package = reference.parent
        for ancestor in reference.parents:
            if ancestor.name.casefold() == "gt":
                relative = reference.relative_to(ancestor)
                if len(relative.parts) > 1:
                    package = ancestor / relative.parts[0]
                break
        result = [{"path": str(path.resolve()), "member": None} for path in package.rglob("*")
                  if path.is_file() and _eligible(path.name) and path.stat().st_size <= 40_000_000
                  and path.resolve().is_relative_to(package.resolve())]
    return sorted(result, key=_rank)[:3]


def _read(candidate):
    if candidate.get("member"):
        with zipfile.ZipFile(candidate["path"]) as archive:
            payload = archive.read(candidate["member"])
    else:
        payload = Path(candidate["path"]).read_bytes()
    with Image.open(BytesIO(payload)) as image:
        if image.width * image.height > 80_000_000:
            raise ValueError("Image exceeds processing limit")
        rgb = np.asarray(image.convert("RGB"))
    return rgb, sha256(payload).hexdigest()


def _resize(rgb, limit):
    image = Image.fromarray(rgb)
    image.thumbnail((limit, limit), Image.Resampling.LANCZOS)
    resized = np.asarray(image)
    scaling = np.diag([image.width / rgb.shape[1], image.height / rgb.shape[0], 1.])
    return resized, scaling


def _color_mask(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return ((hsv[:, :, 1] > 255 * .3) & (hsv[:, :, 2] > 50)).astype(np.uint8) * 255


def _colored_region(rgb):
    reduced, scale = _resize(rgb, 1536)
    strokes = _color_mask(reduced)
    if np.count_nonzero(strokes) < 40:
        return None
    connected = cv2.dilate(strokes, np.ones((3, 3), np.uint8))
    connected = cv2.morphologyEx(connected, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if not .01 <= area / strokes.size <= .85:
        return None
    mask = np.zeros(strokes.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
    colored_support = int(np.count_nonzero((strokes > 0) & (mask > 0)))
    if colored_support < 40 or area < colored_support * 2:
        return None
    return mask, scale, {"colored_pixels": int(np.count_nonzero(strokes)),
                         "selected_stroke_pixels": colored_support, "filled_area_pixels": int(np.count_nonzero(mask)),
                         "overlay_mask_size": {"width": mask.shape[1], "height": mask.shape[0]},
                         "saturation_minimum": .3, "dilation_kernel": 3, "closing_kernel": 5}


def overlay_hint(reference_source, source_image, output_dir=None):
    """Return uint8 source-prepared mask + provenance, or None if no strong match.

    SIFT is bounded to 1000-pixel images, 3500 features, and three package images.
    The final hint uses the source's <=1536 preparation frame. It is explicitly
    unsuitable as a final label; callers render reference DXF geometry instead.
    """
    attempts = []
    try:
        source, source_hash = _read({"path": str(Path(source_image).resolve())})
        source_small, source_scale = _resize(source, 1000)
        source_prepared, prepared_scale = _resize(source, 1536)
        sift = cv2.SIFT_create(nfeatures=3500, contrastThreshold=.025)
        source_gray = cv2.cvtColor(source_small, cv2.COLOR_RGB2GRAY)
        source_keys, source_desc = sift.detectAndCompute(source_gray, None)
        if source_desc is None or len(source_keys) < 20:
            return None
        accepted = []
        for candidate in _candidates(reference_source):
            attempt = {"overlay_source": candidate, "status": "rejected"}
            attempts.append(attempt)
            try:
                overlay, overlay_hash = _read(candidate)
                region = _colored_region(overlay)
                if region is None:
                    attempt["reason"] = "no_large_closed_colored_region"
                    continue
                overlay_small, overlay_scale = _resize(overlay, 1000)
                gray = cv2.cvtColor(overlay_small, cv2.COLOR_RGB2GRAY)
                # Registration features come from the shared drawing background,
                # avoiding colored GT lines as the matching evidence.
                feature_mask = 255 - cv2.dilate(_color_mask(overlay_small), np.ones((3, 3), np.uint8))
                overlay_keys, overlay_desc = sift.detectAndCompute(gray, feature_mask)
                if overlay_desc is None:
                    attempt["reason"] = "no_overlay_features"
                    continue
                pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(overlay_desc, source_desc, k=2)
                matches = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < .72 * pair[1].distance]
                attempt["ratio_matches"] = len(matches)
                if len(matches) < 20:
                    attempt["reason"] = "too_few_matches"
                    continue
                origin = np.float32([overlay_keys[match.queryIdx].pt for match in matches])
                target = np.float32([source_keys[match.trainIdx].pt for match in matches])
                affine, inlier_flags = cv2.estimateAffinePartial2D(origin, target, method=cv2.RANSAC,
                                                                 ransacReprojThreshold=3., maxIters=3000,
                                                                 confidence=.995, refineIters=10)
                if affine is None or inlier_flags is None or not np.isfinite(affine).all():
                    attempt["reason"] = "affine_not_found"
                    continue
                inliers = inlier_flags[:, 0].astype(bool)
                count, ratio = int(inliers.sum()), float(inliers.mean())
                error = np.linalg.norm(origin @ affine[:, :2].T + affine[:, 2] - target, axis=1)[inliers]
                if not count:
                    attempt["reason"] = "no_inliers"
                    continue
                median, p95 = float(np.median(error)), float(np.quantile(error, .95))
                span_a, span_b = np.ptp(origin[inliers], axis=0), np.ptp(target[inliers], axis=0)
                coverage_a = float(np.prod(span_a) / (gray.shape[0] * gray.shape[1]))
                coverage_b = float(np.prod(span_b) / (source_gray.shape[0] * source_gray.shape[1]))
                scale = float(np.sqrt(np.linalg.det(affine[:, :2])))
                attempt.update(inliers=count, inlier_ratio=ratio, median_reprojection_px=median,
                               p95_reprojection_px=p95, overlay_inlier_bbox_fraction=coverage_a,
                               source_inlier_bbox_fraction=coverage_b)
                if count < 20 or ratio < .5 or median > 1.5 or p95 > 3. or coverage_a < .1 or coverage_b < .08 or not .25 <= scale <= 4:
                    attempt["reason"] = "registration_quality_below_threshold"
                    continue
                matrix = np.eye(3); matrix[:2] = affine
                original_matrix = np.linalg.inv(source_scale) @ matrix @ overlay_scale
                region_mask, region_scale, color_info = region
                prepared_matrix = prepared_scale @ original_matrix @ np.linalg.inv(region_scale)
                height, width = source_prepared.shape[:2]
                hint = cv2.warpAffine(region_mask, prepared_matrix[:2], (width, height), flags=cv2.INTER_NEAREST,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                fraction = float(np.count_nonzero(hint) / hint.size)
                if not .01 <= fraction <= .85:
                    attempt["reason"] = "registered_hint_outside_source_or_degenerate"
                    continue
                attempt["status"] = "accepted"
                provenance = {"method": "official_gt_overlay_background_sift_ransac",
                              "localization_only": True, "final_label_source": "rendered_gt_dxf_not_overlay_pixels",
                              "source_image": str(Path(source_image).resolve()), "source_image_sha256": source_hash,
                              "overlay_source": candidate, "overlay_sha256": overlay_hash,
                              "affine_overlay_to_source": original_matrix[:2].tolist(),
                              "affine_overlay_mask_to_prepared_source": prepared_matrix[:2].tolist(),
                              "source_size": {"width": source.shape[1], "height": source.shape[0]},
                              "overlay_size": {"width": overlay.shape[1], "height": overlay.shape[0]},
                              "prepared_size": {"width": width, "height": height},
                              "prepared_scale": {"x": float(prepared_scale[0, 0]), "y": float(prepared_scale[1, 1])},
                              "quality": {key: value for key, value in attempt.items() if key not in {"overlay_source", "status"}},
                              "quality_thresholds": {"min_inliers": 20, "min_inlier_ratio": .5,
                                                     "max_median_error_px": 1.5, "max_p95_error_px": 3.,
                                                     "matching_max_dimension": 1000},
                              "color_region": color_info, "hint_foreground_fraction": fraction,
                              "registration_reviewed": False, "engineering_certified": False}
                accepted.append((count * ratio / (1 + median), hint, provenance))
            except (OSError, ValueError, KeyError, cv2.error, zipfile.BadZipFile) as error:
                attempt["reason"] = type(error).__name__
        if not accepted:
            if output_dir:
                output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
                (output / "overlay-hint.json").write_text(json.dumps({"status": "unavailable", "attempts": attempts}, ensure_ascii=False, indent=2), encoding="utf-8")
            return None
        _, hint, provenance = max(accepted, key=lambda item: item[0])
        provenance["attempts"] = attempts
        if output_dir:
            output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
            Image.fromarray(hint).save(output / "overlay-hint-mask.png")
            preview = source_prepared.copy()
            selected = hint > 0
            preview[selected] = (preview[selected] * .6 + np.array([0, 160, 220]) * .4).astype(np.uint8)
            Image.fromarray(preview).save(output / "overlay-hint-preview.png")
            (output / "overlay-hint.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        return {"hint": hint, "provenance": provenance}
    except (OSError, ValueError, KeyError, cv2.error, zipfile.BadZipFile):
        return None
