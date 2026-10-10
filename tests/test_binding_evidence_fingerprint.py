"""Only known wall-clock telemetry is excluded from source evidence identity."""
from copy import deepcopy

import pytest

from contour_agent.parametric_pipeline import _binding_evidence_sha256


def inventory():
    return {"units": "mm", "source_image_sha256": "image", "proposal_tolerance_px": 2.,
            "all_records": [{"id": "r1", "parsed": {"kind": "radius", "nominal": 43.}}],
            "all_candidates": [{"id": "c1", "record_id": "r1", "entities": ["g1"], "verified": True}],
            "source_arrow_ownership": {"ocr_local_candidate_searches": [
                {"record_id": "r1", "elapsed_seconds": .172, "time_limit_seconds": 8.,
                 "status": "completed", "search_complete": True, "truncated": False,
                 "verified": True, "targets": ["g1"], "tip_px": [10., 20.],
                 "seed_attempts": [{"status": "completed", "verified": True}]}]}}


def test_known_elapsed_wall_time_is_ignored_without_mutating_either_inventory():
    first = inventory()
    second = deepcopy(first)
    second["source_arrow_ownership"]["ocr_local_candidate_searches"][0]["elapsed_seconds"] = .187
    before = deepcopy(first), deepcopy(second)
    assert first != second
    assert _binding_evidence_sha256(first) == _binding_evidence_sha256(second)
    assert (first, second) == before


@pytest.mark.parametrize("key,value", [
    ("time_limit_seconds", 7.), ("deadline", "different"), ("status", "deadline_exhausted"),
    ("search_complete", False), ("truncated", True), ("verified", False),
    ("targets", ["g2"]), ("tip_px", [10., 20.01]), ("nominal", 44.),
    ("acceptance_withheld_reason", "uninspected_source_seed_competitors"),
    ("seed_attempts", [{"status": "inner_search_incomplete", "verified": False}]),
])
def test_search_semantics_and_evidence_changes_still_change_the_fingerprint(key, value):
    first = inventory()
    second = deepcopy(first)
    second["source_arrow_ownership"]["ocr_local_candidate_searches"][0][key] = value
    assert _binding_evidence_sha256(first) != _binding_evidence_sha256(second)


@pytest.mark.parametrize("path", ["record", "candidate", "ownership", "nested_seed"])
def test_elapsed_seconds_is_not_ignored_at_unrecognized_paths(path):
    first = inventory()
    second = deepcopy(first)
    owner = second["source_arrow_ownership"]
    target = {"record": second["all_records"][0], "candidate": second["all_candidates"][0],
              "ownership": owner, "nested_seed": owner["ocr_local_candidate_searches"][0]["seed_attempts"][0]}[path]
    target["elapsed_seconds"] = .187
    assert _binding_evidence_sha256(first) != _binding_evidence_sha256(second)


@pytest.mark.parametrize("defect", ["nominal", "source_image", "target", "geometry"])
def test_changed_source_binding_evidence_is_never_hidden(defect):
    first = inventory()
    second = deepcopy(first)
    if defect == "nominal": second["all_records"][0]["parsed"]["nominal"] = 44.
    elif defect == "source_image": second["source_image_sha256"] = "other"
    elif defect == "target": second["all_candidates"][0]["entities"] = ["g2"]
    else: second["all_candidates"][0]["geometry"] = {"start": [1., 2.]}
    assert _binding_evidence_sha256(first) != _binding_evidence_sha256(second)
