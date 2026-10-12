"""Fresh arrow observations must reach construction, without certifying the parent."""
from copy import deepcopy
import math

import pytest

from contour_agent import topology_editing as editing
from test_topology_editing import _local_fixture


def _source_case():
    arc = {"id": "arc-old", "type": "ARC", "start": [10., 0.], "end": [0., 10.],
           "center": [0., 0.], "radius": 10., "clockwise": False}
    line = {"id": "line-old", "type": "LINE", "start": [0., 10.], "end": [10., 0.]}
    graph = {"entities": [arc, line], "units": "mm", "source_grid_pitch_px": .2,
        "annotation_support": [{"record_id": "r-fresh", "kind": "radius", "nominal": 9.,
            "candidate_entity_id": "line-old", "status": "candidate_supported",
            "arrowhead_verified": True, "target_gap_px": 0.,
            "source_evidence": {"target_source_px": [5., 5.]}}]}
    record = {"record_id": "r-fresh", "text": "R12", "kind": "radius", "nominal": 12.,
        "leader_status": "directed_arrow_candidate", "leader": {
            "target_source_px": [math.sqrt(50.), math.sqrt(50.)], "arrowhead_verified": True}}
    return graph, [arc, line], [record]


def test_fresh_source_target_and_ocr_nominal_reach_kernel_without_mutating_parent():
    graph, source, inventory = _source_case()
    original = deepcopy(graph)
    fresh, audit = editing._fresh_edit_source_graph(graph, source, inventory, .2)
    support = fresh["annotation_support"][0]
    assert support["candidate_entity_id"] == "arc-old"  # Not positional g000 or stale line-old.
    force_arc, radius, binding = editing._radius_edit_evidence(fresh,
        {"action": "refit_chain_as_annotated_arc", "record_id": "r-fresh"}, inventory,
        ["arc-old"], lambda points: points)
    assert force_arc is True and radius == 12. and binding["nominal"] == 12.
    assert graph == original
    assert audit["certified_parent_mutated"] is False
    assert audit["numeric_binding_certified"] is False


def test_failed_source_recheck_revokes_stale_target_for_local_construction():
    graph, source, inventory = _source_case()
    original = deepcopy(graph)
    inventory[0]["leader"] = None
    fresh, audit = editing._fresh_edit_source_graph(graph, source, inventory, .2)
    assert fresh["annotation_support"][0]["status"] == "no_source_leader_candidate"
    with pytest.raises(ValueError, match="radius_annotation_not_supported_by_named_chain"):
        editing._radius_edit_evidence(fresh,
            {"action": "refit_chain_as_annotated_arc", "record_id": "r-fresh"}, inventory,
            ["line-old"], lambda points: points)
    assert graph == original
    assert audit["verified_arrow_proposal_count"] == 0


def test_executor_passes_rechecked_inventory_and_support_to_construction(tmp_path, monkeypatch):
    image, baseline, parent, bundle = _local_fixture(tmp_path)
    parent["graph"]["annotation_support"] = [{"record_id": "r-new", "kind": "radius",
        "nominal": 999., "status": "no_source_leader_candidate"}]
    inventory = [{"record_id": "r-new", "kind": "radius", "nominal": 12.,
        "leader": {"target_source_px": [200., 90.], "arrowhead_verified": True}}]
    bundle["annotation_inventory"] = [{"record_id": "r-new", "kind": "radius", "nominal": 999.}]
    original_parent, original_bundle = deepcopy(parent), deepcopy(bundle)
    monkeypatch.setattr(editing, "_annotation_inventory", lambda *args: (deepcopy(inventory), {}))
    calls = []
    def apply(graph, model, operation, current_inventory, **kwargs):
        calls.append(operation)
        assert current_inventory == inventory
        assert graph is not parent["graph"]
        support = graph["annotation_support"][0]
        assert support["record_id"] == "r-new" and support["nominal"] == 12.
        assert support["candidate_entity_id"] == "g004"
        assert support["arrowhead_verified"] is True
        raise ValueError("synthetic_kernel_rejection_after_fresh_evidence")
    monkeypatch.setattr(editing, "_apply_one", apply)
    rows, audit = editing.execute_topology_edits(image, {"records": []}, baseline, parent, bundle,
        [{"action": "refit_chain_as_annotated_arc", "entity_ids": ["g004"], "record_id": "r-new"}],
        tmp_path / "edits")
    assert len(calls) == 1 and rows == []
    assert audit["operations"][0]["reason"] == "synthetic_kernel_rejection_after_fresh_evidence"
    assert audit["source_inventory_refresh"]["status"] == "rechecked_before_construction"
    assert parent == original_parent and bundle == original_bundle
