"""Localize source-stroke failures without changing any acceptance threshold."""
from __future__ import annotations

import cv2
import numpy as np

from .ocr import canonical_records
from .topology import _StrokeEvidence, _sample_path
from .vectorize import _sample_entities


def source_support_diagnostics(image_path, document, baseline, graph):
    """Report each primitive and its ordered quarters in original-image pixels.

    These are diagnostics, not additional evidence of boundary identity, and
    never approve geometry. No reference CAD is read.
    """
    image = cv2.imdecode(np.fromfile(str(image_path), np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return {"status": "unavailable", "reason": "source_image_unavailable"}
    grid = float(graph.get("source_grid_pitch_px") or 1.)
    evidence = _StrokeEvidence(image, canonical_records(document), grid)
    scale = float(baseline["scale"]["pixels_per_mm"]) if graph["units"] == "mm" else 1.
    origin = np.asarray(graph["coordinate_system"]["origin_source_px"], float)
    rows = []
    for entity in graph.get("entities", []):
        points, _, _ = _sample_entities([entity], max_step_px=max(.25, grid / scale / 2))
        pixels = _sample_path(points * [scale, -scale] + origin, max(.5, 1 / evidence.scale))
        distances, supported = evidence.query(pixels)
        quarters = []
        for index, indices in enumerate(np.array_split(np.arange(len(pixels)), 4)):
            if not len(indices):
                continue
            quarters.append({"quarter": index + 1,
                             "stroke_supported_fraction": float(supported[indices].mean()),
                             "p90_edge_distance_px": float(np.quantile(distances[indices], .9))})
        row = {"entity_id": entity["id"], "stable_id": entity.get("stable_id"),
               "type": entity["type"], "sample_count": len(pixels),
               "stroke_supported_fraction": float(supported.mean()),
               "p90_edge_distance_px": float(np.quantile(distances, .9)),
               "quarters": quarters}
        binding = entity.get("radius_binding") or {}
        if binding.get("record_id"):
            row["record_id"] = binding["record_id"]
        rows.append(row)
    return {"status": "measured", "schema_version": "local-source-stroke-diagnostics-v1",
            "ground_truth_used": False, "acceptance_thresholds_changed": False,
            "near_original_px": evidence.near / evidence.scale,
            "entities": rows,
            "scope": "Source ink evidence only; quarters follow each primitive's start-to-end order."}
