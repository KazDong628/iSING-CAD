"""Bounded source-only multi-resolution refinement; never force a single region.

The detail view proposes local probability changes, not replacement geometry.
Topology and source-ink gates can reject the whole proposal. Neither a passed
gate nor a connected mask establishes correct dimensions or reference accuracy.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


def detail_inference_size(model_size):
    return min(1024, max(model_size, int(np.ceil(model_size*4/3/32))*32))


def _components(mask, connectivity=4):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=connectivity)
    return n - 1, labels, stats


def connectivity(mask):
    mask = np.asarray(mask, dtype=bool)
    n4, _, stats = _components(mask, 4)
    n8, _, _ = _components(mask, 8)
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = sum(int(x[3] >= 0) for x in hierarchy[0]) if hierarchy is not None else 0
    areas = sorted(map(int, stats[1:, cv2.CC_STAT_AREA]), reverse=True)
    return {"components_4": n4, "components_8": n8, "corner_only_connections": n4 - n8,
            "foreground_pixels": int(mask.sum()), "component_areas": areas,
            "holes": holes, "single_material_region": n4 == 1,
            "meaning": "Pixel topology only; holes and thin connections require source interpretation."}


def _edge_distance(mask, distance):
    eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    values = distance[mask & (eroded == 0)]
    return {"mean_px": float(values.mean()) if values.size else None,
            "p95_px": float(np.percentile(values, 95)) if values.size else None}


def refine_probabilities(image, coarse, detail, *, model_size=768):
    """Return (probability, receipt), using a frozen 75/25 fusion near boundaries.

Existing small components are never deleted. Only newly created specks of at
most four pixels can be rolled back to their coarse probabilities. New joins
must survive one-pixel erosion: corner/single-pixel bridges are not accepted.
"""
    image = np.asarray(image)
    coarse = np.asarray(coarse, np.float32)
    detail = np.asarray(detail, np.float32)
    if image.ndim != 3 or image.shape[2] != 3 or coarse.ndim != 2 or min(coarse.shape) < 3 or image.shape[:2] != coarse.shape or detail.shape != coarse.shape:
        raise ValueError("Aligned RGB image and 2D probabilities required")
    if not isinstance(model_size, int) or isinstance(model_size, bool) or model_size < 32:
        raise ValueError("model_size must be a positive integer >= 32")
    if coarse.size > 1536**2 or not all(np.isfinite(a).all() and a.min() >= 0 and a.max() <= 1 for a in (coarse, detail)):
        raise ValueError("Finite probabilities in [0,1] on a bounded 1536 grid required")
    raw = coarse >= .5
    before = connectivity(raw)
    evidence = {"algorithm": "guarded-boundary-multiresolution-v1", "ground_truth_used": False,
                "status": "unchanged", "before": before, "after": before,
                "detail_weight": .25, "forced_bridges": 0, "existing_components_deleted": 0,
                "threshold": .5, "reasons": [], "model_size": model_size,
                "scope": "Source-only proposal; not a topology or dimensional correctness guarantee."}
    if not raw.any() or raw.all() or before["components_4"] > 64:
        evidence["reasons"] = ["Empty/full or excessively fragmented coarse prediction cannot support bounded refinement."]
        return coarse.copy(), evidence
    # Keep deep foreground/background unchanged. This also bounds new material.
    band_radius = max(2., 6. * max(raw.shape) / model_size)
    band = (distance_transform_edt(raw) + distance_transform_edt(~raw)) <= band_radius
    proposal = np.where(band, .75*coarse + .25*detail, coarse).astype(np.float32)
    candidate = proposal >= .5
    count, labels, stats = _components(candidate)
    reverted_specks = 0
    for label in range(1, min(count, 256)+1):
        if stats[label, cv2.CC_STAT_AREA] <= 4:
            region = labels == label
            if not np.any(raw[region]):
                proposal[region] = coarse[region]
                reverted_specks += int(region.sum())
    candidate = proposal >= .5
    after = connectivity(candidate)
    distance = distance_transform_edt(cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY) > 170)
    source_before, source_after = _edge_distance(raw, distance), _edge_distance(candidate, distance)
    changed = raw ^ candidate
    fraction = float(changed.sum()/max(1, raw.sum()))
    strong_removed = int(((coarse >= .8) & ~candidate).sum())
    reasons = []
    _, old_labels, old_stats = _components(raw)
    surviving = set(map(int, np.unique(old_labels[candidate]))) - {0}
    removed_components = sorted(set(range(1, before["components_4"]+1)) - surviving)
    if removed_components:
        reasons.append("Proposal deletes existing material components.")
    if fraction > .04: reasons.append("Proposal changes more than 4 percent of coarse material pixels.")
    if strong_removed > max(4, int(raw.sum()*.002)): reasons.append("Proposal removes too much confident foreground.")
    if after["components_4"] > before["components_4"] or after["components_8"] > before["components_8"]:
        reasons.append("Proposal introduces disconnected regions.")
    if after["holes"] != before["holes"]:
        reasons.append("Hole count changed; source-only fusion cannot certify filling or opening holes.")
    if not candidate.any(): reasons.append("Proposal erases material.")
    if source_after["mean_px"] is None or source_after["mean_px"] > source_before["mean_px"] + .05:
        reasons.append("Average source-ink boundary support deteriorates.")
    if source_after["p95_px"] is not None and source_after["p95_px"] > source_before["p95_px"] + .5:
        reasons.append("Tail source-ink boundary support deteriorates.")
    joins = []
    split_components = []
    # Counts alone can hide a split and a join cancelling each other. Build a
    # bounded overlap table so every old component and every new join is checked.
    # Larger proposals are already rejected by the component-increase guard;
    # do not allocate a potentially enormous label-pair matrix for that case.
    if after["components_4"] <= 64:
        _, new_labels, _ = _components(candidate)
        stride = after["components_4"] + 1
        pairs = old_labels.astype(np.int64) * stride + new_labels
        overlap = np.bincount(pairs.ravel(), minlength=(before["components_4"]+1)*stride).reshape(before["components_4"]+1, stride)
        split_components = [label for label in range(1, before["components_4"]+1)
                            if np.count_nonzero(overlap[label, 1:]) > 1]
        if split_components:
            reasons.append("Proposal splits existing material components.")
        joining_groups = [(group, list(map(int, np.flatnonzero(overlap[1:, group])+1)))
                          for group in range(1, after["components_4"]+1)
                          if np.count_nonzero(overlap[1:, group]) > 1]
    else:
        joining_groups = []
    if joining_groups:
        core = cv2.erode(candidate.astype(np.uint8), np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
        _, core_labels, _ = _components(core)
        for group, joined in joining_groups:
            shared = None
            for label in joined:
                members = set(map(int, np.unique(core_labels[(old_labels == label) & (new_labels == group)]))) - {0}
                shared = members if shared is None else shared & members
            robust = bool(shared)
            joins.append({"coarse_component_ids": joined, "survives_one_pixel_erosion": robust,
                          "areas": [int(old_stats[i, cv2.CC_STAT_AREA]) for i in joined]})
            if not robust: reasons.append("New connection collapses under one-pixel erosion.")
    evidence.update(proposed=after, proposal_changed_pixels=int(changed.sum()), changed_fraction=fraction,
                    boundary_band_radius_px=band_radius, source_boundary_before=source_before,
                    source_boundary_proposed=source_after, reverted_new_speck_pixels=reverted_specks,
                    joined_components=joins, proposed_deleted_component_ids=removed_components,
                    proposed_split_component_ids=split_components, component_overlap_checked=after["components_4"] <= 64,
                    reasons=reasons)
    if reasons:
        evidence["status"] = "rejected"
        evidence["changed_pixels"] = 0
        return coarse.copy(), evidence
    evidence.update(status="accepted" if changed.any() else "unchanged", after=after, changed_pixels=int(changed.sum()))
    return proposal, evidence
