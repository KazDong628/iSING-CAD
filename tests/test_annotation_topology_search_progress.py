"""A bounded search must spend numeric preflights on real source repairs."""
from copy import deepcopy

import numpy as np
import pytest

from contour_agent.topology_editing import _replacement_chain, propose_annotation_arc_edits
from contour_agent.topology_search import preflight_order


def _line(start, end):
    return {"type": "LINE", "start": list(start), "end": list(end), "fit_error_px": 0.}


def test_source_split_rejects_same_two_lines_without_claiming_annotation_repair():
    # A putative missing radius on a rectangular closure does not become an
    # ARC simply because the model requests a source split of its two LINEs.
    points = np.array([[0., 0.], [20., 0.], [20., 20.]])
    old = [_line(points[0], points[1]), _line(points[1], points[2])]
    original = deepcopy(old)
    with pytest.raises(ValueError, match="source_split_did_not_change_topology"):
        _replacement_chain(points, "split_chain_at_source_features", .2, old)
    assert old == original


def test_source_split_retains_actual_corner_missing_from_one_line():
    points = np.array([[0., 0.], [20., 0.], [20., 20.]])
    old = [_line(points[0], points[-1])]
    replacement = _replacement_chain(points, "split_chain_at_source_features", .2, old)
    assert [row["type"] for row in replacement] == ["LINE", "LINE"]


def test_same_arc_count_with_verified_radius_partition_is_still_a_candidate(monkeypatch):
    old = [{"type": "ARC"}, {"type": "ARC"}]
    arcs = [{"type": "ARC", "radius": radius, "fit_error_px": .05,
             "radius_binding": {"record_id": record}} for radius, record in ((12., "r1"), (24., "r2"))]
    monkeypatch.setattr("contour_agent.annotated_arc_resegmentation.resegment_annotated_arcs",
                        lambda *args, **kwargs: {"entities": deepcopy(arcs), "record_ids": ["r1", "r2"]})
    replacement = _replacement_chain(np.zeros((4, 2)), "split_chain_at_source_features", .2, old,
                                     radius_targets=[{}, {}])
    assert [row["radius"] for row in replacement] == [12., 24.]
    assert all(row["source_partition_evidence"] for row in replacement)


def _candidate(name, evidence):
    return {"id": name, "graph": {"source_evidence": {"topology_edit": evidence}}}


def test_preflight_prefers_real_exact_radius_repair_to_free_refit_without_waiving_gates():
    free = _candidate("free", {"action": "refit_chain_as_annotated_arc",
        "record_id": "r1", "radius_binding_applied": False,
        "radius_binding_status": "unresolved_fixed_radius_fit_failed"})
    constructed = _candidate("constructed", {"action": "insert_annotated_fillet",
        "radius_binding_applied": True, "bound_record_ids": ["r2"]})
    unsupported = _candidate("unsupported", {"action": "insert_annotated_fillet",
        "radius_binding_applied": True, "bound_record_ids": ["r3", "r4"]})
    generic = _candidate("generic", {"action": "merge_chain_best_fit"})
    rows = [free, unsupported, generic, constructed]
    validations = {row["id"]: {"passed": row is not unsupported} for row in rows}
    original = deepcopy(rows)
    assert [row["id"] for row in preflight_order(rows, validations)] == [
        "constructed", "free", "generic", "unsupported"]
    assert rows == original
    assert "verified_radius_record_ids" not in constructed


def test_radius_partition_remains_available_across_arbitrary_closed_cycle_seam():
    entities = [{"id": f"g{i}", "type": "ARC", "radius": 20. + i} for i in range(6)]
    inventory = [{"record_id": "r0", "kind": "radius", "nominal": 20.},
                 {"record_id": "r5", "kind": "radius", "nominal": 25.}]
    graph = {"units": "mm", "entities": entities, "annotation_support": [
        {"record_id": record, "kind": "radius", "status": "candidate_supported",
         "candidate_entity_id": entity, "arrowhead_verified": True}
        for record, entity in (("r0", "g0"), ("r5", "g5"))]}
    # Every cyclic rotation names the same physical candidate. Neither source
    # features nor their type hypotheses should depend on JSON array origin.
    for offset in range(len(entities)):
        graph["entities"] = entities[offset:] + entities[:offset]
        partitions = [row for row in propose_annotation_arc_edits(graph, inventory)
                      if row["action"] == "split_chain_at_source_features"]
        assert len(partitions) == 1
        assert partitions[0]["entity_ids"] == ["g5", "g0"]
