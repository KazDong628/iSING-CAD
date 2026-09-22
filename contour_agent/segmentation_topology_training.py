"""Optional material-area regularizers and offline mask topology diagnostics.

The Euler regularizer is a soft local/global surrogate, not a guarantee of a
connected prediction. It matches each target's topology, including real holes;
it does not impose tubular skeletons or universally require one component.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage


def connectivity_metrics(mask, *, valid=None, erosion_radius=1, min_core_pixels=4):
    """Report pixel-grid topology without changing the supplied mask.

    Foreground/background use complementary 4/8 adjacency. Erosion splitting
    indicates a thin-neck candidate, never proof of a spurious connection.
    Declared ignored pixels make full-mask topology unverified rather than
    turning ignored pixels into background and manufacturing breaks.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not mask.size:
        raise ValueError("mask must be a nonempty two-dimensional array")
    if not isinstance(erosion_radius, int) or not 1 <= erosion_radius <= 8:
        raise ValueError("erosion_radius must be an integer between 1 and 8")
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != mask.shape:
            raise ValueError("valid and mask shapes differ")
    result = {"foreground_pixels": int(mask.sum()), "shape": list(mask.shape),
              "ignored_pixels": int((~valid).sum()) if valid is not None else 0,
              "topology_complete": valid is None or bool(valid.all()),
              "thin_neck_meaning": "Candidate from erosion splitting; legitimate thin material can also trigger this flag",
              "erosion_radius_px": erosion_radius}
    labels8 = None
    for foreground_adjacency, background_adjacency in ((4, 8), (8, 4)):
        structure = ndimage.generate_binary_structure(2, 1 if foreground_adjacency == 4 else 2)
        labels, count = ndimage.label(mask, structure)
        sizes = np.bincount(labels.ravel())[1:]
        bg_structure = ndimage.generate_binary_structure(2, 1 if background_adjacency == 4 else 2)
        background_labels, _ = ndimage.label(~np.pad(mask, 1), bg_structure)
        exterior = set(np.unique(np.concatenate((background_labels[0], background_labels[-1],
                                                  background_labels[:, 0], background_labels[:, -1]))))
        hole_ids = set(np.unique(background_labels)) - exterior - {0}
        result[f"components_{foreground_adjacency}"] = int(count)
        result[f"holes_{foreground_adjacency}"] = len(hole_ids)
        result[f"euler_{foreground_adjacency}"] = int(count - len(hole_ids))
        result[f"component_areas_{foreground_adjacency}"] = sorted(map(int, sizes), reverse=True)
        if foreground_adjacency == 8:
            labels8 = labels
    result["largest_component_fraction"] = (result["component_areas_8"][0] / int(mask.sum())) if mask.any() else 0.0
    result["diagonal_contact_sensitive"] = result["components_4"] != result["components_8"]
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3), bool), iterations=erosion_radius)
    cores, _ = ndimage.label(eroded, np.ones((3, 3), bool))
    core_sizes = np.bincount(cores.ravel())
    qualifying_cores = np.flatnonzero(core_sizes >= min_core_pixels)
    qualifying_cores = qualifying_cores[qualifying_cores != 0]
    original_core_counts = {}
    for core_id in qualifying_cores:
        original_id = int(labels8[tuple(np.argwhere(cores == core_id)[0])])
        original_core_counts[original_id] = original_core_counts.get(original_id, 0) + 1
    result["erosion_core_count"] = len(qualifying_cores)
    result["thin_neck_component_candidates"] = sum(count > 1 for count in original_core_counts.values())
    result["thin_neck_extra_cores"] = sum(max(0, count - 1) for count in original_core_counts.values())
    return result


def _soft_euler_map(probability, adjacency=8):
    import torch.nn.functional as functional
    p = functional.pad(probability, (1, 1, 1, 1))
    a, b, c, d = p[..., :-1, :-1], p[..., :-1, 1:], p[..., 1:, :-1], p[..., 1:, 1:]
    na, nb, nc, nd = 1-a, 1-b, 1-c, 1-d
    one = a*nb*nc*nd + na*b*nc*nd + na*nb*c*nd + na*nb*nc*d
    three = na*b*c*d + a*nb*c*d + a*b*nc*d + a*b*c*nd
    diagonal = a*nb*nc*d + na*b*c*nd
    return (one - three + (2 if adjacency == 4 else -2)*diagonal) / 4


def material_regularization(logits, target, *, boundary_weight=.2, topology_weight=.03):
    """Differentiable boundary + Euler terms; target < 0 means ignored.

    Only fully supervised neighborhoods contribute. Ignored logits have zero
    gradient. Compute in float32 even under mixed precision. Euler equality
    alone is insufficient (components and holes can cancel), so both a local
    map term and a global term are reported and independent topology is audited.
    """
    import torch
    import torch.nn.functional as functional
    if logits.shape != target.shape or logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("logits and target must have matching Bx1xHxW shapes")
    if boundary_weight < 0 or topology_weight < 0:
        raise ValueError("regularization weights must be nonnegative")
    valid = target >= 0
    observed = valid.to(torch.float32)
    p = logits.float().sigmoid()
    truth = target.float().clamp(0, 1)
    # Replacing ignored pixels removes them from every differentiable path.
    p = torch.where(valid, p, torch.zeros_like(p))
    truth = torch.where(valid, truth, torch.zeros_like(truth))
    def gradient(value):
        padded = functional.pad(value, (1, 1, 1, 1))
        return functional.max_pool2d(padded, 3, stride=1) + functional.max_pool2d(-padded, 3, stride=1)
    supported_boundary = (functional.avg_pool2d(functional.pad(observed, (1, 1, 1, 1), value=1), 3, stride=1) >= 1).float()
    predicted_boundary, target_boundary = gradient(p), gradient(truth)
    boundary = (torch.abs(predicted_boundary-target_boundary)*supported_boundary).sum() / supported_boundary.sum().clamp_min(1)
    # Sharpen without changing exact binary masks or threshold ordering.
    sharpened = p.pow(4) / (p.pow(4)+(1-p).pow(4)).clamp_min(1e-8)
    supported_cells = (functional.avg_pool2d(functional.pad(observed, (1, 1, 1, 1), value=1), 2, stride=1) >= 1).float()
    local_terms, global_terms = [], []
    for adjacency in (4, 8):
        delta = (_soft_euler_map(sharpened, adjacency)-_soft_euler_map(truth, adjacency))*supported_cells
        local_terms.append(delta.abs().sum()/supported_cells.sum().clamp_min(1))
        per_example = delta.sum((1, 2, 3)).abs()
        global_terms.append(torch.log1p(per_example).mean())
    local_euler = sum(local_terms)/2
    global_euler = sum(global_terms)/2
    topology = 4*local_euler + .1*global_euler
    total = boundary_weight*boundary + topology_weight*topology
    return total, {"boundary": boundary, "euler_local": local_euler,
                   "euler_global_log_error": global_euler, "topology": topology}


def audit_manifest(manifest, output=None):
    """Read validated derived masks; never alter source data or supervision."""
    import json
    from PIL import Image
    from .segmentation import read_manifest, write_json, _ignore_array
    rows, provenance = read_manifest(manifest)
    consumed = {row["id"] for row in rows}
    document = json.loads(Path(manifest).read_text(encoding="utf8"))
    all_rows = document if isinstance(document, list) else next(document[k] for k in ("cases", "rows", "samples") if isinstance(document.get(k), list))
    results = []
    for row in rows:
        mask = np.asarray(Image.open(row["mask"]).convert("L")) > 127
        ignored = _ignore_array(row, (mask.shape[1], mask.shape[0]))
        metrics = connectivity_metrics(mask, valid=None if ignored is None else ~ignored)
        flags = []
        if metrics["components_8"] != 1: flags.append("not_one_foreground_component")
        if metrics["holes_8"]: flags.append("holes_require_source_confirmation")
        if metrics["diagonal_contact_sensitive"]: flags.append("diagonal_only_connection")
        if metrics["thin_neck_component_candidates"]: flags.append("erosion_sensitive_neck")
        if not metrics["topology_complete"]: flags.append("ignore_region_prevents_complete_topology_claim")
        results.append({"id": row["id"], "split": row["split"], "mask_sha256": row["mask_sha256"],
                        "label_source": row["label_source"], "reviewed": row.get("reviewed", False),
                        "metrics": metrics, "review_flags": flags})
    excluded = [{"id": r.get("id", r.get("case_id")), "split": r.get("split"),
                 "status": "not_usable_in_frozen_manifest"} for r in all_rows if r.get("id", r.get("case_id")) not in consumed]
    report = {"format": "mask-connectivity-audit-v1", "read_only": True, "provenance": provenance,
              "total_cases": len(all_rows), "audited_cases": len(results), "excluded_cases": excluded,
              "review_flagged_cases": sum(bool(row["review_flags"]) for row in results), "cases": results,
              "meaning": "Offline GT-label quality audit; flags neither modify masks nor certify source registration or dimensional accuracy"}
    if output is not None: write_json(output, report)
    return report
