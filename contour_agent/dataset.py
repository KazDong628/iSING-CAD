"""Read-only source inventory. Reference coordinates never enter runtime OCR data."""
from __future__ import annotations

import json
import math
import re
import zipfile
from pathlib import Path, PurePosixPath

CALIBRATION_CASE = "solid-arrow-ping__img_000293"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def resolve_inside(root: Path, relative: str | Path) -> Path:
    """Reject absolute paths, traversal, and symlinks escaping the dataset."""
    root = Path(root).resolve()
    value = str(relative).replace("\\", "/")
    if not value or value.startswith("/") or re.match(r"^[A-Za-z]:", value):
        raise ValueError("Expected a relative dataset path")
    if ".." in PurePosixPath(value).parts:
        raise ValueError("Dataset path traversal is forbidden")
    result = (root / value).resolve()
    if not result.is_relative_to(root):
        raise ValueError("Dataset path escapes its root")
    return result


def _read_json(path: Path) -> dict:
    if path.stat().st_size > 25_000_000:
        raise ValueError("JSON exceeds the 25 MB limit")
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object")
    return data


def read_ocr(path: Path) -> dict:
    """Validate OCR and assign stable host-controlled r000 record identifiers.

    Metadata paths and arbitrary extra fields are deliberately not forwarded.
    Reading OCR does not inspect any adjacent reference/GT file.
    """
    data = _read_json(Path(path))
    records = data.get("records")
    if not isinstance(records, list) or len(records) > 20_000:
        raise ValueError("OCR records must be a list with at most 20000 entries")
    meta = data.get("meta", {})
    if not isinstance(meta, dict):
        raise ValueError("OCR meta must be an object")
    if meta.get("box_coord_space", "original_image") != "original_image":
        raise ValueError("OCR boxes must use original_image coordinates")
    size = meta.get("original_size", {})
    if not isinstance(size, dict):
        raise ValueError("OCR original_size must be an object")
    clean_size = {}
    for key in ("width", "height"):
        if key in size:
            value = size[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 100_000:
                raise ValueError(f"Invalid OCR image {key}")
            clean_size[key] = value
    clean = []
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not isinstance(record.get("text"), str):
            raise ValueError(f"OCR record {index}: text must be a string")
        if len(record["text"]) > 8192:
            raise ValueError(f"OCR record {index}: text too long")
        box = record.get("box")
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError(f"OCR record {index}: box must contain four points")
        clean_box = []
        for point in box:
            if not isinstance(point, (list, tuple)) or len(point) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or abs(v) > 1_000_000 for v in point):
                raise ValueError(f"OCR record {index}: invalid box coordinate")
            clean_box.append([float(v) for v in point])
        score = record.get("score")
        if score is not None and (isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError(f"OCR record {index}: invalid score")
        clean.append({"id": f"r{index:03d}", "text": record["text"], "box": clean_box, "score": score})
    return {"meta": {"original_size": clean_size, "box_coord_space": "original_image"}, "records": clean}


def _case_token(case_id: str) -> str:
    match = re.search(r"img_0*(\d+)", case_id)
    return match.group(1) if match else case_id.lower().replace("-main", "").replace("-", "")


def _matches_case(path: Path, case_id: str) -> bool:
    token = _case_token(case_id)
    value = path.as_posix().lower()
    if token.isdigit():
        number = str(int(token))
        return bool(re.search(r"(?:solid|img_0*|gt_|/|^)(?:0*)" + re.escape(number) + r"(?=[^0-9]|$)", value))
    return token in re.sub(r"[-_]", "", value)


def _profile_rank(path: Path) -> tuple:
    name = path.name.lower()
    # Prefer named complete main profiles. Debug/measurement outputs are rejected.
    return (
        not ("main_profile" in name or "mainprofile" in name or "main_contour" in name),
        "scored_only" in name or "strict_body" in name,
        "design_nominal_reference" in name,
        -max([int(x) for x in re.findall(r"(?:_|-)v(\d+)", name)] or [0]),
        path.as_posix().lower(),
    )


def _legacy_relative(value: str) -> str:
    """Map stale manifest paths only via an explicit __dataset anchor."""
    value = value.replace("\\", "/")
    if "/__dataset/" in value:
        return value.rsplit("/__dataset/", 1)[1]
    return value


def build_catalog(dataset_root: Path) -> dict:
    root = Path(dataset_root).resolve()
    origin = resolve_inside(root, "origin")
    if not origin.is_dir():
        raise ValueError("Dataset must have an origin directory")
    warnings = []
    originals = {}
    for path in sorted(origin.iterdir(), key=lambda p: p.name.lower()):
        if path.suffix.lower() not in IMAGE_SUFFIXES | {".json"} or not path.is_file():
            continue
        path = resolve_inside(root, path.relative_to(root))
        item = originals.setdefault(path.stem, {"images": [], "ocr": None})
        if path.suffix.lower() == ".json":
            item["ocr"] = path
        else:
            item["images"].append(path)
    legacy = {}
    manifest = resolve_inside(root, "vis/splits.json")
    if manifest.is_file():
        try:
            for item in _read_json(manifest).get("records", []):
                if isinstance(item, dict) and isinstance(item.get("sample_id"), str):
                    legacy[item["sample_id"]] = item
        except (ValueError, OSError) as exc:
            warnings.append(f"Legacy inventory could not be read: {type(exc).__name__}")
    gt_root = resolve_inside(root, "GT")
    dxf_files = []
    zip_files = []
    if gt_root.is_dir():
        for path in sorted(gt_root.rglob("*")):
            if not path.is_file():
                continue
            path = resolve_inside(root, path.relative_to(root))
            if path.suffix.lower() == ".dxf" and not any(word in path.name.lower() for word in ("construction", "debug", "measurement_audit")):
                dxf_files.append(path)
            if path.suffix.lower() == ".zip":
                zip_files.append(path)
    cases = []
    for case_id, source in sorted(originals.items()):
        candidates = sorted([p for p in dxf_files if _matches_case(p, case_id)], key=_profile_rank)
        selected = candidates[0] if candidates else None
        mapping_source = "filename_inventory" if selected else None
        previous = legacy.get(case_id, {})
        if isinstance(previous.get("dxf"), str):
            try:
                mapped = resolve_inside(root, _legacy_relative(previous["dxf"]))
                if mapped in candidates:
                    selected, mapping_source = mapped, "legacy_inventory_path_remapped"
            except ValueError:
                warnings.append(f"Rejected unsafe legacy reference path for {case_id}")
        archive = None
        for zip_path in [p for p in zip_files if _matches_case(p, case_id)]:
            if selected is not None:
                break
            try:
                with zipfile.ZipFile(zip_path) as package:
                    members = sorted(info.filename for info in package.infolist() if info.filename.lower().endswith(".dxf") and not any(x in info.filename.lower() for x in ("construction", "debug")))
                    safe_members = []
                    for member in members:
                        resolve_inside(root, member)
                        safe_members.append(member)
                    if safe_members:
                        archive = {"path": zip_path.relative_to(root).as_posix(), "member": safe_members[0]}
                        break
            except (ValueError, OSError, zipfile.BadZipFile):
                warnings.append(f"Rejected invalid archive inventory for {case_id}")
        limitations = ["Reference inventory is not engineering certification."] if selected or archive else ["No reference DXF package is available."]
        if previous:
            limitations.append("Historical raster alignment status is retained as provenance, not a current geometric verdict.")
        if case_id == CALIBRATION_CASE:
            limitations += ["Calibration case: cannot establish held-out generalization.", "Reference includes an unlabeled fitted bridge and simplified tread; these are assumptions, not fully dimension-defined truth."]
        if archive:
            limitations.append("Reference exists only in an archive; no source files were extracted or modified.")
        images = source["images"]
        if len(images) > 1:
            warnings.append(f"Multiple source images for {case_id}; selected lexicographically")
        cases.append({
            "id": case_id,
            "image": images[0].relative_to(root).as_posix() if images else None,
            "ocr": source["ocr"].relative_to(root).as_posix() if source["ocr"] else None,
            "gt_dxf": selected.relative_to(root).as_posix() if selected else None,
            "gt_archive": archive,
            "gt_provenance": {"mapping_source": mapping_source, "certified": False, "legacy_raster_status": previous.get("status"), "reference_variants": [p.relative_to(root).as_posix() for p in candidates], "limitations": limitations},
            "split": "calibration" if case_id == CALIBRATION_CASE else "holdout",
            "supported_template": "solid293-v1" if case_id == CALIBRATION_CASE else None,
        })
    counts = {
        "total": len(cases), "paired": sum(bool(c["image"] and c["ocr"]) for c in cases),
        "supported": sum(c["supported_template"] is not None for c in cases),
        "unsupported": sum(c["supported_template"] is None for c in cases),
        "calibration": sum(c["split"] == "calibration" for c in cases),
        "holdout": sum(c["split"] == "holdout" for c in cases),
        "gt_dxf": sum(c["gt_dxf"] is not None for c in cases),
        "archived_gt_only": sum(c["gt_archive"] is not None for c in cases),
        "missing_gt": sum(not c["gt_dxf"] and not c["gt_archive"] for c in cases),
        "missing_gt_dxf": sum(c["gt_dxf"] is None for c in cases),
        "missing_image": sum(c["image"] is None for c in cases),
        "missing_ocr": sum(c["ocr"] is None for c in cases),
    }
    return {"cases": cases, "counts": counts, "warnings": sorted(set(warnings))}
