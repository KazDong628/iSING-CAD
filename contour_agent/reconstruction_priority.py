"""Explicit source-first policy; reference and mask failures remain visible."""
from __future__ import annotations

import copy

SOURCE_FIRST = "source_annotations_before_mask_v1"


def source_first_model(model):
    result = copy.deepcopy(model)
    result["reconstruction_priority"] = {
        "policy": SOURCE_FIRST,
        "known_source_dimensions": "hard",
        "attribute_binding_order": ["radius", "angle", "other_source_dimensions", "unknown_attributes"],
        "verified_source_relations": "hard",
        "source_ink_and_native_geometry": "hard",
        "mask": "initial_topology_and_soft_localization",
        "reference_dxf": "independent_evaluation_only",
        "reference_thresholds_changed": False,
    }
    return result


def mask_is_hard(model):
    return (model.get("reconstruction_priority") or {}).get("policy") != SOURCE_FIRST


def mask_conflict_receipt(model, check):
    return {
        "policy": (model.get("reconstruction_priority") or {}).get("policy", "legacy_mask_fidelity"),
        "mask_fidelity_required_for_source_publication": mask_is_hard(model),
        "mask_representation_passed": check.get("passed"),
        "source_mask_conflict": check.get("applicable") is True and check.get("passed") is False,
        "mask_measurement_or_threshold_changed": False,
        "reference_accuracy_verified": False,
    }
