"""Frozen, source-only weak segmentation labels and leakage-aware grouping.

Only ``origin`` images and their paired OCR are read. No reference package,
rendered reference, previous prediction, or model output is used as a label.
These development splits are not a blind test of reconstruction accuracy.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

import cv2
import numpy as np
from PIL import Image

from .dataset import CALIBRATION_CASE, IMAGE_SUFFIXES, read_ocr
from .outline_fallback import extract_unhatched
from .raster import extract_main_profile


FORMAT_VERSION = "source-heuristic-segmentation-v1"
MAX_DIMENSION = 1536
SPLIT_CLAIM = "development_only_not_blind_test"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Path escapes its declared data directory")
    return resolved


def _inventory(dataset_root: Path) -> list[dict]:
    origin = _inside(dataset_root, dataset_root / "origin")
    if not origin.is_dir():
        raise ValueError("Dataset must have an origin directory")
    rows, seen = [], set()
    # Intentionally do not use build_catalog: it also inventories GT/vis.
    for candidate in sorted(origin.iterdir(), key=lambda path: path.name.casefold()):
        if candidate.suffix.lower() not in IMAGE_SUFFIXES or not candidate.is_file():
            continue
        image = _inside(origin, candidate)
        case_id = candidate.stem
        if case_id.casefold() in seen:
            raise ValueError("Multiple source images share one case ID")
        seen.add(case_id.casefold())
        ocr = _inside(origin, candidate.with_suffix(".json"))
        image_hash = _sha256(image)
        rows.append({"id": case_id, "source_image": str(image),
                     "image_sha256": image_hash, "source_image_sha256": image_hash,
                     "source_ocr": str(ocr) if ocr.is_file() else None,
                     "source_ocr_sha256": _sha256(ocr) if ocr.is_file() else None})
    if not rows:
        raise ValueError("No source images found in origin")
    return rows


def _phash(image: Image.Image) -> str:
    gray = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    low = cv2.dct(gray)[:8, :8].flatten()
    bits = low > np.median(low[1:])
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def _assign_groups(rows: list[dict], seed: int) -> None:
    """Connected components prevent transitive near-duplicate split leakage."""
    parents = list(range(len(rows)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for left, a in enumerate(rows):
        for right in range(left):
            b = rows[right]
            duplicate = a["image_sha256"] == b["image_sha256"]
            if not duplicate and a.get("phash") and b.get("phash"):
                ratio_a, ratio_b = a["aspect_ratio"], b["aspect_ratio"]
                duplicate = (max(ratio_a, ratio_b) / min(ratio_a, ratio_b) <= 1.03
                             and (int(a["phash"], 16) ^ int(b["phash"], 16)).bit_count() <= 4)
            if duplicate:
                parents[find(left)] = find(right)
    components = {}
    for index, row in enumerate(rows):
        components.setdefault(find(index), []).append(row)
    groups = list(components.values())
    rng = random.Random(seed)
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)
    targets = {"train": len(rows) * .7, "val": len(rows) * .15, "test": len(rows) * .15}
    counts = {split: 0 for split in targets}
    calibration_ids = {CALIBRATION_CASE, CALIBRATION_CASE + "-main", "293", "293-main"}
    forced = [group for group in groups if any(row["id"] in calibration_ids for row in group)]

    def assign(group, split):
        signature = "\n".join(sorted(row["id"] + ":" + row["image_sha256"] for row in group))
        group_id = "group-" + hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]
        for row in group:
            row.update(group=group_id, split=split)
        counts[split] += len(group)

    for group in forced:
        assign(group, "train")
    for group in groups:
        if group in forced:
            continue
        # Whole groups are allocated by count deficit, never split to hit a quota.
        split = min(targets, key=lambda candidate: sum(
            (counts[name] + (len(group) if name == candidate else 0) - targets[name]) ** 2
            for name in targets))
        assign(group, split)


def _source_signature(rows: list[dict]) -> list[dict]:
    fields = ("id", "source_image", "source_image_sha256", "source_ocr", "source_ocr_sha256")
    return [{field: row.get(field) for field in fields} for row in sorted(rows, key=lambda row: row["id"])]


def _reuse(manifest_path: Path, output_dir: Path, source_rows: list[dict], seed: int) -> dict:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = manifest["cases"]
        if (manifest["format_version"] != FORMAT_VERSION or manifest["seed"] != seed
                or manifest["max_dimension"] != MAX_DIMENSION
                or manifest["split_claim"] != SPLIT_CLAIM
                or not isinstance(rows, list)
                or _source_signature(rows) != _source_signature(source_rows)):
            raise ValueError("Frozen manifest settings or source hashes differ; use a new output directory")
        for row in rows:
            if row.get("artifact_status") == "ready" and not all(row.get(name) for name in ("image", "mask")):
                raise ValueError("Frozen manifest has incomplete ready artifacts")
            for field, hash_field in (("image", "prepared_image_sha256"), ("mask", "mask_sha256")):
                if row.get(field):
                    artifact = _inside(output_dir, Path(row[field]))
                    if not artifact.is_file() or _sha256(artifact) != row.get(hash_field):
                        raise ValueError("Frozen dataset artifact is missing or changed; use a new output directory")
        return manifest
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("Existing manifest is invalid; it will not be overwritten") from error


def prepare_dataset(dataset_root, output_dir, seed=42) -> dict:
    """Create or verify/reuse ``output_dir/manifest.json`` and return its dict.

    ``cases`` retains failed labels with ``trainable=False``. Ready images and
    binary masks have matching sizes (longest edge <=1536), and absolute paths
    inside output_dir. Reviews are separate: this function never reads, rewrites
    or removes reviews.json, nor silently refreshes a frozen weak label.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    dataset_root, output_dir = Path(dataset_root).resolve(), Path(output_dir).resolve()
    if output_dir.is_relative_to(dataset_root):
        raise ValueError("Prepared data must be outside the read-only source dataset")
    source_rows = _inventory(dataset_root)
    manifest_path = _inside(output_dir, output_dir / "manifest.json")
    if manifest_path.is_file():
        return _reuse(manifest_path, output_dir, source_rows, seed)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Output directory is not empty and has no frozen manifest; use a new directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = _inside(output_dir, output_dir / "images")
    masks_dir = _inside(output_dir, output_dir / "masks")
    images_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)
    rows = []
    for source in source_rows:
        row = {**source, "image": None, "mask": None, "group": None, "split": None,
               "label_source": "source_heuristic", "reviewed": False,
               "artifact_status": "failed", "trainable": False, "phash": None,
               "issues": [], "fallback_used": False}
        rows.append(row)
        try:
            with Image.open(source["source_image"]) as loaded:
                width, height = loaded.size
                if width * height > 80_000_000:
                    raise ValueError("Image exceeds processing pixel limit")
                image = loaded.convert("RGB")
            row.update(original_size={"width": width, "height": height},
                       aspect_ratio=width / height, phash=_phash(image))
            image.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.Resampling.LANCZOS)
            sx, sy = image.width / width, image.height / height
            row.update(prepared_size={"width": image.width, "height": image.height},
                       resize_scale={"x": sx, "y": sy})
            image_path = _inside(output_dir, images_dir / (source["id"] + ".png"))
            image.save(image_path, format="PNG")
            row.update(image=str(image_path), prepared_image_sha256=_sha256(image_path))
            extraction = extract_main_profile(source["source_image"])
            if not extraction.get("polyline_px"):
                document = read_ocr(Path(source["source_ocr"])) if source["source_ocr"] else {"records": []}
                extraction = extract_unhatched(source["source_image"], document)
                row["fallback_used"] = True
            row["extraction_status"] = extraction.get("status")
            row["issues"] = list(extraction.get("issues", []))
            polygon = np.asarray(extraction.get("polyline_px", []), dtype=float)
            if (polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3
                    or not np.isfinite(polygon).all()
                    or np.any(polygon < 0) or np.any(polygon[:, 0] > width) or np.any(polygon[:, 1] > height)):
                row["issues"].append("No valid source-heuristic polygon; no training mask created.")
                continue
            mask = np.zeros((image.height, image.width), dtype=np.uint8)
            cv2.fillPoly(mask, [np.rint(polygon * [sx, sy]).astype(np.int32)], 255)
            if not np.any(mask) or np.all(mask):
                row["issues"].append("Degenerate filled mask; no training mask created.")
                continue
            mask_path = _inside(output_dir, masks_dir / (source["id"] + ".png"))
            Image.fromarray(mask).save(mask_path, format="PNG")
            row.update(mask=str(mask_path), mask_sha256=_sha256(mask_path),
                       artifact_status="ready", trainable=True,
                       foreground_pixels=int(np.count_nonzero(mask)))
        except Exception as error:
            # Error categories suffice here; arbitrary OCR/extractor text must
            # not be treated as executable code or copied as an exception dump.
            row["issues"].append("Source preparation failed: " + type(error).__name__)
    _assign_groups(rows, seed)
    manifest = {
        "format_version": FORMAT_VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root), "output_dir": str(output_dir), "seed": seed,
        "max_dimension": MAX_DIMENSION, "split_claim": SPLIT_CLAIM,
        "label_source": "source_heuristic", "reviewed": False,
        "ground_truth_used": False, "previous_predictions_used": False,
        "grouping": {"exact_sha256": True, "phash_bits": 64, "phash_max_distance": 4,
                     "max_aspect_ratio_relative_difference": .03, "transitive_components": True},
        "split_policy": {"target_ratios": {"train": .7, "val": .15, "test": .15},
                         "calibration_case_forced_train": CALIBRATION_CASE,
                         "groups_are_indivisible": True},
        "summary": {"total": len(rows), "ready": sum(row["trainable"] for row in rows),
                    "failed": sum(not row["trainable"] for row in rows),
                    "groups": len({row["group"] for row in rows}),
                    "split_counts": dict(Counter(row["split"] for row in rows)),
                    "trainable_split_counts": dict(Counter(row["split"] for row in rows if row["trainable"]))},
        "cases": rows,
    }
    temporary = _inside(output_dir, output_dir / "manifest.json.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(manifest_path)
    return manifest
