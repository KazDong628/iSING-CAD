"""Persist a real source-contour CAD checkpoint before remote/measurement work.

Availability is independent of engineering acceptance. No default shape,
template, nominal dimension or reference geometry can create this checkpoint.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil

import ezdxf
import numpy as np
from shapely.geometry import Polygon


def _write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
    temporary.replace(path)


def preserve_source_draft(image_path, extraction, output, *, oracle_mask=False, gt_assisted=False):
    from .automatic import _verify_dxf_readback
    from .vectorize import _closed_ring, _line_entities

    output = Path(output)
    # Prefer the exact extracted ring. Invalid inputs must not be turned into a
    # rectangle/convex hull, or an empty DXF masquerading as a reconstruction.
    ring = _closed_ring(extraction.get("raw_polyline_px") or extraction["polyline_px"])
    polygon = Polygon(ring)
    if not polygon.is_valid or polygon.area <= 0:
        raise ValueError("Source draft requires a valid nondegenerate closed contour")
    origin = [float(ring[:, 0].min()), float(ring[:, 1].max())]
    mapped = np.column_stack((ring[:, 0] - origin[0], origin[1] - ring[:, 1]))
    entities = _line_entities(mapped)
    identity = {"source_image_sha256": hashlib.sha256(Path(image_path).read_bytes()).hexdigest(),
                "source_ring_sha256": hashlib.sha256(np.asarray(ring, dtype="<f8").tobytes()).hexdigest()}
    document = ezdxf.new("R2010")
    document.units = 0
    document.layers.new("AUTO_SOURCE_DRAFT", dxfattribs={"color": 3})
    for row in entities:
        document.modelspace().add_line(row["start"], row["end"], dxfattribs={"layer": "AUTO_SOURCE_DRAFT"})
    temporary = output / "source-draft-drawing.dxf.tmp"
    document.saveas(temporary)
    native = _verify_dxf_readback(ezdxf.readfile(temporary), entities, expected_units=0)
    if not native["passed"]:
        raise ValueError("Source draft DXF readback failed")
    temporary.replace(output / "source-draft-drawing.dxf")
    connectivity = extraction.get("evidence", {}).get("connectivity")
    complete = connectivity is None or connectivity.get("complete_exterior_candidate") is True
    quality = {"passed": False, "fallback_used": True, "representation": "raw_source_contour_lines",
               "method": "source-contour-checkpoint-v1", "primitive_attributes_verified": False}
    validation = {"passed": False, "geometry_valid": True, "artifact_available": True,
                  "entity_count": len(entities), "scaled_mm": False, "max_gap_mm": None,
                  "self_intersection": False, "dxf_readback": native,
                  "material_connectivity": connectivity, "complete_material_exterior": complete,
                  "dimensions_verified": False, "reference_verified": False,
                  "engineering_certified": False, "curve_fit": quality,
                  "issues": ["源轮廓像素单位草稿：尚未完成比例、LINE/ARC 属性识别及标注约束求解。"]}
    bounds = Polygon(mapped).bounds
    model = {"mode": "autonomous_image", "algorithm_version": "source-contour-checkpoint-v1",
             "automatic_completion": False, "completion_class": "source_contour_pixel_draft",
             "complete_material_exterior": complete, "manual_intervention": bool(extraction.get("review")) and not oracle_mask, "template_used": False,
             "ground_truth_used": bool(oracle_mask or gt_assisted), "oracle_mask_conditioned": bool(oracle_mask),
             "ground_truth_use": "raster_mask_input_only" if oracle_mask else "human_edited_gt_raster_mask" if gt_assisted else "none",
             "artifact_input_identity": identity, "extraction": extraction, "entities": entities,
             "scale": {"status": "unresolved", "pixels_per_mm": None, "issues": []},
             "coordinate_system": {"units": "pixel", "x": "image right", "y": "image up", "origin_source_px": origin},
             "polyline_px": ring.tolist(), "fitted_polyline_px": ring.tolist(),
             "bounds": dict(zip(("min_x", "min_y", "max_x", "max_y"), bounds)),
             "validation": validation, "curve_fit": quality, "issues": validation["issues"],
             "scope": "Finite source contour only; fitted object types, dimensions and reference accuracy remain unverified."}
    _write(output / "source-draft-model.json", model)
    _write(output / "source-draft-validation.json", validation)
    width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
    pad = max(width, height) * .05
    path = "M " + " L ".join(f"{x} {-y}" for x, y in mapped) + " Z"
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{-pad} {-bounds[3]-pad} {width+2*pad} {height+2*pad}">'
           f'<title>Source contour draft; pixel units; attributes unverified</title><path d="{path}" fill="#dbe9e2" '
           'stroke="#087f72" stroke-width="1.6" vector-effect="non-scaling-stroke"/></svg>')
    (output / "source-draft-preview.svg").write_text(svg, encoding="utf8")
    _write(output / "draft-availability.json", {"schema_version": "automatic-dxf-availability-v1",
           "available": True, "source_checkpoint": "source-draft-drawing.dxf", "current_artifact": None,
           "source_checkpoint_units": "pixel", "engineering_accepted": False, "input_identity": identity})
    # Publish the complete, read-back checkpoint before the first slow stage.
    for name in ("drawing.dxf", "preview.svg", "model.json", "validation.json"):
        if not (output / name).exists():
            shutil.copyfile(output / ("source-draft-" + name), output / name)
    return model


def dxf_availability(output):
    """Check artifact existence/readability without claiming geometric accuracy."""
    path = Path(output) / "drawing.dxf"
    result = {"available": False, "artifact": "drawing.dxf", "engineering_accepted": False,
              "scope": "Nonempty finite LINE/ARC artifact availability only; validity and annotation accuracy are separate."}
    if not path.is_file():
        return {**result, "reason": "missing_dxf"}
    try:
        document = ezdxf.readfile(path)
        entities = list(document.modelspace())
        if not entities:
            return {**result, "reason": "empty_dxf"}
        for row in entities:
            kind = row.dxftype()
            if kind == "LINE":
                values = (*row.dxf.start, *row.dxf.end)
            elif kind == "ARC" and row.dxf.radius > 0:
                values = (*row.dxf.center, row.dxf.radius, row.dxf.start_angle, row.dxf.end_angle)
            else:
                return {**result, "reason": "unsupported_or_invalid_primitive"}
            if not all(math.isfinite(float(value)) for value in values):
                return {**result, "reason": "nonfinite_geometry"}
        return {**result, "available": True, "readback_performed": True, "entity_count": len(entities),
                "units": "mm" if document.units == 4 else "pixel_or_unitless" if document.units == 0 else str(document.units),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    except Exception:
        return {**result, "reason": "dxf_readback_failed"}


def recover_source_draft(output, model, error, *, stage=None):
    """Restore the verified checkpoint, never treat failed attributes as passed."""
    from .automatic import _verify_dxf_readback

    output = Path(output)
    check = _verify_dxf_readback(ezdxf.readfile(output / "source-draft-drawing.dxf"),
                                 model["entities"], expected_units=0)
    if not check["passed"]:
        raise ValueError("Cannot restore an invalid source checkpoint")
    model = json.loads(json.dumps(model))
    model["initialization_recovery"] = {"status": "retained_source_draft", "failure_type": type(error).__name__, "failure_stage": stage,
                                        "source_checkpoint_verified": True, "engineering_accepted": False}
    model["issues"].append("后续初始化失败，已保留真实源轮廓 DXF 草稿；未认定标注、对象类型或精度通过。")
    for name in ("drawing.dxf", "preview.svg"):
        temporary = output / (name + ".tmp")
        shutil.copyfile(output / ("source-draft-" + name), temporary)
        temporary.replace(output / name)
    _write(output / "model.json", model)
    _write(output / "validation.json", model["validation"])
    _write(output / "draft-availability.json", {"schema_version": "automatic-dxf-availability-v1",
           "available": True, "source_checkpoint": "source-draft-drawing.dxf", "current_artifact": "drawing.dxf",
           "current_units": "pixel", "engineering_accepted": False,
           "initialization_failure_type": type(error).__name__, "input_identity": model["artifact_input_identity"]})
    return model
