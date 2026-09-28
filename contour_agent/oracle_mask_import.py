"""Convert an uploaded reference DXF to a source-aligned raster mask.

This development-only boundary accepts the reference in memory, writes it to a
temporary local file for the read-only DXF parser, and returns only pixels and
safe provenance. CAD coordinates and registration matrices never enter a job,
browser response, prediction module, or online provider request.
"""
from __future__ import annotations

import hashlib
import io
import math
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .dataset import read_ocr
from .dxf_supervision import load_reference_polygon
from .gt_registration import register_training_polygon


MAX_UPLOAD_BYTES = 20_000_000
QUALITY_LIMITS = {"frame_inside_ratio": .99, "edge_support": .55,
                  "ambiguity_margin": .025}


def generate_oracle_mask(dxf_bytes: bytes, image_path, ocr_path, hint_mask_path, mask_size,
                         *, temporary_root) -> tuple[bytes, dict]:
    """Rasterize one unambiguous closed DXF exterior at the review-mask size.

    The model mask is only a weak alignment hint. Source-image ink and OCR scale
    independently screen the registration; poor or ambiguous alignment fails
    closed so it cannot be mislabeled as a GT input.
    """
    if not dxf_bytes or len(dxf_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError("GT DXF 必须是非空且不超过20MB的文件。")
    width, height = (int(mask_size[0]), int(mask_size[1]))
    if min(width, height) < 2 or width * height > 80_000_000:
        raise ValueError("当前分割图尺寸不支持GT DXF配准。")
    with tempfile.TemporaryDirectory(prefix="gt-dxf-import-", dir=temporary_root) as directory:
        path = Path(directory) / "uploaded.dxf"
        path.write_bytes(dxf_bytes)
        reference = load_reference_polygon(path, repair_topology=False)
        if (reference.get("status") != "ready" or reference.get("topology_repaired")
                or reference.get("selection_ambiguous") or len(reference.get("polygons", [])) != 1
                or reference["polygons"][0].get("holes")):
            raise ValueError("GT DXF 必须包含唯一、无孔、未修补的闭合材料主轮廓；请检查DXF图层和连接。")
        document = read_ocr(ocr_path)
        evidence = register_training_polygon(image_path, reference["polygon_xy"],
                                             foreground_hint=hint_mask_path,
                                             ocr_document=document, max_dimension=1000)
    quality = evidence.get("quality") or {}
    failed = [name for name, threshold in QUALITY_LIMITS.items()
              if not isinstance(quality.get(name), (int, float))
              or not math.isfinite(quality[name]) or quality[name] < threshold]
    if evidence.get("status") != "registration_candidate" or failed:
        raise ValueError("GT DXF 与原图自动配准未通过：" + ("、".join(failed) if failed else "未得到有效位置")
                         + "。请确认DXF与本任务原图对应，不能把未核实的掩膜作为GT输入。")
    polygon = np.asarray(evidence.get("registered_polyline_px", []), dtype=float)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 4 or not np.isfinite(polygon).all():
        raise ValueError("GT DXF 配准后没有有效的材料边界。")
    with Image.open(image_path) as image:
        source_width, source_height = image.size
    points = (polygon + .5) * [width / source_width, height / source_height] - .5
    if not np.isfinite(points).all() or np.max(np.abs(points)) > 1_000_000:
        raise ValueError("GT DXF 配准坐标超出可栅格化范围。")
    mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 255)
    foreground = int(np.count_nonzero(mask))
    if foreground < 3 or foreground == mask.size:
        raise ValueError("GT DXF 生成的材料掩膜为空或占满整幅图。")
    encoded = io.BytesIO()
    Image.fromarray(mask).save(encoded, format="PNG")
    pixels = encoded.getvalue()
    receipt = {
        "schema_version": "job-oracle-dxf-import-v1", "source": "uploaded_gt_dxf",
        "source_gt_sha256": hashlib.sha256(dxf_bytes).hexdigest(),
        "mask_sha256": hashlib.sha256(pixels).hexdigest(),
        "source_image_sha256": hashlib.sha256(Path(image_path).read_bytes()).hexdigest(),
        "source_ocr_sha256": hashlib.sha256(Path(ocr_path).read_bytes()).hexdigest(),
        "mask_size": {"width": width, "height": height},
        "foreground_pixels": foreground,
        "registration_quality": {name: float(quality[name]) for name in QUALITY_LIMITS},
        "registration_quality_limits": dict(QUALITY_LIMITS),
        "ground_truth_used_for_mask": True,
        "ground_truth_dxf_coordinates_sent_to_provider": False,
        "ground_truth_dxf_coordinates_used_for_prediction": False,
        "calibration_or_development": True, "held_out": False,
        "scope": "Uploaded DXF was used locally to derive only a raster-mask input; registration agreement does not certify reference accuracy.",
    }
    return pixels, receipt
