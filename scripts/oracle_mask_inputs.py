"""Freeze one registered DXF-GT material mask as a development input.

Only the original image, source OCR, and *raster mask* are copied to the
prediction input directory.  GT DXF geometry, registration transforms and
reference polygon coordinates stay outside that directory and must never be
sent to an online provider.  The reference file is opened here solely to
verify its hash before the mask is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import zipfile

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contour_agent.dataset import read_ocr


MAX_MANIFEST_BYTES = 30_000_000
MAX_REFERENCE_BYTES = 50_000_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    if path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError(f"Oversized JSON: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path.name}")
    return value


def _inside(path: str | Path, root: Path, description: str) -> Path:
    candidate = Path(path).resolve()
    if not candidate.is_relative_to(root.resolve()) or not candidate.is_file():
        raise ValueError(f"{description} must be an existing file inside {root}")
    return candidate


def _verify_hash(path: Path, expected: str, description: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", str(expected)):
        raise ValueError(f"Missing or invalid {description} SHA-256")
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{description} SHA-256 changed from frozen manifest")
    return actual


def _reference_hash(source: dict, dataset_root: Path) -> str:
    path = _inside(source["path"], dataset_root / "GT", "GT source")
    member = source.get("member")
    if member is None:
        if path.stat().st_size > MAX_REFERENCE_BYTES:
            raise ValueError("GT DXF exceeds size limit")
        return _sha256(path)
    pure = PurePosixPath(str(member).replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts or ":" in str(member) or "\x00" in str(member):
        raise ValueError("Unsafe GT archive member")
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(member)
        if info.file_size > MAX_REFERENCE_BYTES:
            raise ValueError("GT archive member exceeds size limit")
        digest = hashlib.sha256()
        with archive.open(info) as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


def _size(value: dict, description: str) -> tuple[int, int]:
    if not isinstance(value, dict):
        raise ValueError(f"Missing {description} dimensions")
    result = []
    for key in ("width", "height"):
        item = value.get(key)
        if isinstance(item, bool) or not isinstance(item, int) or not 1 <= item <= 100_000:
            raise ValueError(f"Invalid {description} {key}")
        result.append(item)
    return tuple(result)


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _verify_mask(path: Path, size: tuple[int, int], foreground: int) -> dict:
    with Image.open(path) as image:
        values = np.asarray(image.convert("L"))
    if values.shape != (size[1], size[0]) or not np.isin(values, [0, 255]).all():
        raise ValueError("Frozen mask has changed dimensions or is not binary")
    binary = (values == 255).astype(np.uint8)
    count = int(np.count_nonzero(binary))
    if not count or count == binary.size or count != foreground:
        raise ValueError("Frozen mask foreground count is inconsistent")
    component_count = cv2.connectedComponents(binary, connectivity=8)[0] - 1
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hole_count = 0 if hierarchy is None else sum(int(parent >= 0) for parent in hierarchy[0, :, 3])
    if component_count != 1 or hole_count:
        raise ValueError("Oracle mask is not a single connected, hole-free main profile")
    return {"foreground_pixels": count, "connected_components": component_count,
            "holes": hole_count, "contours": len(contours)}


def load_oracle_mask_case(manifest_path: str | Path, case_id: str,
                          output_inputs: str | Path) -> dict:
    """Validate and freeze an eligible registered GT mask; never run prediction.

    Returns local paths ``source_image``, ``source_ocr``, ``mask`` and
    ``input_manifest``.  Caller passes the first three to the normal pipeline;
    the manifest is local audit evidence, not a model or provider input.
    Existing nonempty output directories are rejected to preserve provenance.
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", case_id):
        raise ValueError("Unsafe case id")
    manifest_path = Path(manifest_path).resolve()
    frozen_root = manifest_path.parent
    source = _read_json(manifest_path)
    if (source.get("format_version") != "registered-dxf-gt-v1"
            or source.get("label_source") != "registered_dxf_gt"
            or source.get("ground_truth_used") is not True
            or source.get("split_claim") != "development_only_not_blind_test"):
        raise ValueError("Expected a frozen development registered-DXF-GT manifest")
    dataset_root = Path(source["dataset_root"]).resolve()
    if not dataset_root.is_dir() or frozen_root.is_relative_to(dataset_root):
        raise ValueError("Frozen labels must live outside the source dataset")
    if Path(source["output_dir"]).resolve() != frozen_root:
        raise ValueError("Frozen label directory does not match manifest")
    output = Path(output_inputs).resolve()
    if output.is_relative_to(dataset_root) or output.is_relative_to(frozen_root):
        raise ValueError("Oracle experiment inputs must be outside dataset and frozen label directory")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Use a new or empty input directory; oracle inputs are immutable")
    rows = [row for row in source.get("cases", []) if row.get("id") == case_id]
    if len(rows) != 1:
        raise ValueError("Case id must match exactly one frozen row")
    row = rows[0]
    if (row.get("label_source") != "registered_dxf_gt" or row.get("artifact_status") != "ready"
            or row.get("trainable") is not True or row.get("registration", {}).get("status") != "accepted"
            or row.get("reference_status") != "ready"):
        raise ValueError("Case does not have an accepted registered GT material mask")
    if row.get("registration", {}).get("failed_quality_checks"):
        raise ValueError("Frozen registration has failed quality checks")
    original_size = _size(row.get("original_size"), "original image")
    prepared_size = _size(row.get("prepared_size"), "prepared image")
    image = _inside(row["source_image"], dataset_root / "origin", "Original source image")
    ocr = _inside(row["source_ocr"], dataset_root / "origin", "Source OCR")
    prepared = _inside(row["image"], frozen_root / "images", "Frozen prepared image")
    mask = _inside(row["mask"], frozen_root / "masks", "Frozen GT mask")
    image_sha = _verify_hash(image, row["source_image_sha256"], "Original image")
    _verify_hash(ocr, row["source_ocr_sha256"], "Source OCR")
    _verify_hash(prepared, row["prepared_image_sha256"], "Prepared image")
    mask_sha = _verify_hash(mask, row["mask_sha256"], "GT mask")
    if image_sha != row.get("image_sha256"):
        raise ValueError("Source image identity differs from case row")
    if _image_size(image) != original_size or _image_size(prepared) != prepared_size:
        raise ValueError("Original or prepared image dimensions changed")
    ocr_document = read_ocr(ocr)
    ocr_size = ocr_document["meta"]["original_size"]
    if ocr_size and (ocr_size.get("width"), ocr_size.get("height")) != original_size:
        raise ValueError("OCR boxes are not in the original image coordinate system")
    mask_geometry = _verify_mask(mask, prepared_size, row["foreground_pixels"])
    reference = _read_json(frozen_root / "references" / f"{case_id}.json")
    polygons = reference.get("polygons", [])
    if (reference.get("status") != "ready" or reference.get("topology_repaired") is not False
            or reference.get("selection_ambiguous") is True or len(polygons) != 1
            or any(polygon.get("holes") for polygon in polygons)):
        raise ValueError("Reference is repaired, ambiguous, multi-component, or contains uncopied holes")
    if reference.get("source") != row.get("reference_source"):
        raise ValueError("Frozen reference source differs from registration source")
    reference_sha = row["registration"].get("source_gt_sha256")
    if reference.get("source_sha256") != reference_sha or _reference_hash(reference["source"], dataset_root) != reference_sha:
        raise ValueError("GT DXF bytes differ from frozen registration reference")
    transform = np.asarray(row["registration"].get("transform"), dtype=float)
    if transform.shape != (2, 3) or not np.isfinite(transform).all() or abs(np.linalg.det(transform[:, :2])) < 1e-12:
        raise ValueError("Invalid frozen GT-to-image registration")
    policy = source.get("quality_policy", {})
    quality = row["registration"].get("quality", {})
    for field in ("frame_inside_ratio", "edge_support", "hint_precision", "hint_coverage", "ambiguity_margin"):
        lower = policy.get(field + "_min")
        actual = quality.get(field)
        if not isinstance(lower, (int, float)) or not isinstance(actual, (int, float)) or not math.isfinite(actual) or actual < lower:
            raise ValueError(f"Frozen registration quality not accepted: {field}")
    base_path = frozen_root.parent / "data" / "manifest.json"
    _verify_hash(base_path, source["base_manifest_sha256"], "Base preparation manifest")

    output.mkdir(parents=True, exist_ok=True)
    copied_image = output / ("source-image" + image.suffix.lower())
    copied_ocr = output / "source-ocr.json"
    copied_mask = output / "oracle-mask.png"
    for old, new, digest in ((image, copied_image, image_sha),
                             (ocr, copied_ocr, row["source_ocr_sha256"]),
                             (mask, copied_mask, mask_sha)):
        shutil.copyfile(old, new)
        _verify_hash(new, digest, new.name)
    receipt = {
        "schema_version": "oracle-mask-input-v1",
        "case_id": case_id,
        "oracle_mask": True,
        "ground_truth_used_for_mask": True,
        "ground_truth_geometry_sent_to_provider": False,
        "calibration_or_development": True,
        "held_out": False,
        "source_split_label": row.get("split"),
        "split_interpretation": "Development experiment even when the inherited training split says test",
        "source_image": str(copied_image),
        "source_ocr": str(copied_ocr),
        "mask": str(copied_mask),
        "original_image_size": {"width": original_size[0], "height": original_size[1]},
        "mask_size": {"width": prepared_size[0], "height": prepared_size[1]},
        "mask_geometry": mask_geometry,
        "source_image_sha256": image_sha,
        "source_ocr_sha256": row["source_ocr_sha256"],
        "mask_sha256": mask_sha,
        "reference_dxf_sha256": reference_sha,
        "frozen_manifest_sha256": _sha256(manifest_path),
        "base_manifest_sha256": source["base_manifest_sha256"],
        "registration_quality": {key: quality[key] for key in (
            "frame_inside_ratio", "edge_support", "hint_precision", "hint_coverage", "ambiguity_margin")},
        "reference_components": 1,
        "reference_holes": 0,
        "reference_topology_repaired": False,
        "scope": "GT DXF raster mask is an oracle segmentation input only; DXF geometry and registration transform are withheld from prediction and online API.",
    }
    receipt_path = output / "input-manifest.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"source_image": copied_image, "source_ocr": copied_ocr,
            "mask": copied_mask, "input_manifest": receipt_path, "receipt": receipt}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_id")
    parser.add_argument("--manifest", type=Path,
                        default=Path("runtime/segmentation/gt-data-v3/manifest.json"))
    parser.add_argument("--output-inputs", type=Path, required=True)
    args = parser.parse_args()
    result = load_oracle_mask_case(args.manifest, args.case_id, args.output_inputs)
    print(json.dumps({"case_id": args.case_id, "input_manifest": str(result["input_manifest"]),
                      "oracle_mask": True, "held_out": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
