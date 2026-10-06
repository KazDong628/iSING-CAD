"""Source-supported topology edits: numeric features, lineage and isolation."""
import copy
import hashlib
import math

import cv2
import numpy as np
import pytest

from contour_agent.topology_candidates import _affines, _base_source_entities
from contour_agent.topology_editing import (
    _annotated_line_fillet, _apply_one, _split_source_chain, execute_topology_edits,
    propose_annotation_arc_edits, _replacement_chain, _radius_edit_evidence,
)
from contour_agent.vectorize import _sample_entities


def _corner_parts(radius=20.):
    return [
        {"type": "LINE", "start": [20., 20.], "end": [100., 20.]},
        {"type": "ARC", "start": [100., 20.], "end": [120., 40.],
         "center": [100., 40.], "radius": radius, "clockwise": False},
        {"type": "LINE", "start": [120., 40.], "end": [120., 110.]},
    ]


def _binding():
    return {"record_id": "r001", "nominal": 20., "arrowhead_verified": True,
            "target_source_px": [114.1421356237, 25.8578643763]}


def _fixture(tmp_path, *, split_top=False):
    if split_top:
        points = [[20, 20], [50, 20], [80, 20], [120, 20], [120, 60], [120, 110], [20, 110]]
        source = [{"type": "LINE", "start": a, "end": b}
                  for a, b in zip(points, points[1:] + points[:1])]
    else:
        points = [[20, 20], [120, 20], [120, 110], [20, 110]]
        source = [*_corner_parts(), {"type": "LINE", "start": [120, 110], "end": [20, 110]},
                  {"type": "LINE", "start": [20, 110], "end": [20, 20]}]
    raw = _sample_entities(source, max_step_px=.5)[0]
    image = np.full((150, 150, 3), 255, np.uint8)
    cv2.polylines(image, [np.rint(raw).astype(np.int32)], True, (0, 0, 0), 2)
    path = tmp_path / "source.png"
    cv2.imencode(".png", image)[1].tofile(str(path))
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    graph = {
        "candidate_id": "base", "source_sha256": source_hash, "units": "mm",
        "source_grid_pitch_px": 1., "proposal_tolerance_px": .5,
        "coordinate_system": {"units": "mm", "origin_source_px": [0., 0.]},
        "nodes": [{"id": f"v{i:03d}", "source_px": p, "x": p[0], "y": -p[1]}
                  for i, p in enumerate(points)],
        "entities": [{"id": f"g{i:03d}", "type": "LINE", "start": [a[0], -a[1]],
                      "end": [b[0], -b[1]], "start_node": f"v{i:03d}",
                      "end_node": f"v{(i + 1) % len(points):03d}"}
                     for i, (a, b) in enumerate(zip(points, points[1:] + points[:1]))],
        "annotation_support": [],
    }
    baseline = {"extraction": {"raw_polyline_px": raw.tolist()},
                "coordinate_system": graph["coordinate_system"], "scale": {"pixels_per_mm": 1.}}
    candidate = {"id": "base", "graph": graph}
    bundle = {"source_sha256": source_hash, "base_graph_sha256": "baseline-hash", "annotation_inventory": []}
    return path, graph, baseline, candidate, bundle


def _tangent(primitive, endpoint):
    delta = np.array(primitive["end"]) - primitive["start"]
    if primitive["type"] == "ARC":
        radial = np.array(primitive[endpoint]) - primitive["center"]
        delta = np.array([-radial[1], radial[0]]) * (-1 if primitive["clockwise"] else 1)
    return delta / np.linalg.norm(delta)


def test_exact_radius_fillet_moves_internal_joints_and_preserves_tangency():
    source = _sample_entities(_corner_parts(), max_step_px=.5)[0]
    parts = _annotated_line_fillet(source, 20., .5, _binding())
    assert [part["type"] for part in parts] == ["LINE", "ARC", "LINE"]
    assert parts[1]["radius"] == 20.
    assert parts[1]["radius_binding_status"] == "applied"
    assert parts[0]["start"] == [20., 20.] and parts[-1]["end"] == [120., 110.]
    for first, second in zip(parts, parts[1:]):
        assert first["end"] == pytest.approx(second["start"])
        assert float(_tangent(first, "end") @ _tangent(second, "start")) > 1 - 1e-12


@pytest.mark.parametrize("radius,target,directed", [
    (3., None, True), (20., [40., 20.], True), (20., None, False),
])
def test_fillet_cannot_invent_radius_or_use_target_on_neighboring_straight(radius, target, directed):
    source = _sample_entities(_corner_parts(), max_step_px=.5)[0]
    binding = _binding()
    binding["arrowhead_verified"] = directed
    if target is not None:
        binding["target_source_px"] = target
    with pytest.raises(ValueError):
        _annotated_line_fillet(source, radius, .5, binding)


def test_closed_profile_can_gain_an_evidence_bound_small_feature(tmp_path):
    path, graph, baseline, candidate, bundle = _fixture(tmp_path)
    graph["annotation_support"] = [{"record_id": "r001", "kind": "radius", "status": "candidate_supported",
        "candidate_entity_id": "g000", "arrowhead_verified": True, "target_gap_px": 5.86,
        "source_evidence": {"target_source_px": _binding()["target_source_px"]}}]
    inventory = [{"record_id": "r001", "kind": "radius", "nominal": 20., "text": "R20"}]
    original = copy.deepcopy(graph)
    entities, quality, execution = _apply_one(graph, baseline,
        {"action": "insert_annotated_fillet", "entity_ids": ["g000", "g001"], "record_id": "r001"}, inventory)
    assert len(entities) == 5  # Was 4: restoring a feature may legitimately add an object.
    assert quality["sampled_topology_valid"]
    assert execution["feature_restoration_validated"] and execution["radius_binding_applied"]
    assert execution["net_entity_reduction"] == -1
    assert graph == original
    bundle["annotation_inventory"] = inventory
    rows, _ = execute_topology_edits(path, {"records": []}, baseline, candidate, bundle,
        [{"action": "insert_annotated_fillet", "entity_ids": ["g000", "g001"], "record_id": "r001"}],
        tmp_path / "fillet")
    assert len(rows) == 1
    constructed = next(e for e in rows[0]["graph"]["entities"] if e["type"] == "ARC")
    assert constructed["radius"] == pytest.approx(20.)
    assert constructed["radius_binding"]["record_id"] == "r001"
    assert constructed["radius_constructed"] is True
    assert constructed["radius_binding_status"] == "constructed_unverified"
    assert constructed["dimension_bound"] is False


def test_source_split_finds_multiple_primitives_without_model_coordinates():
    source = _sample_entities(_corner_parts(), max_step_px=.5)[0]
    pieces = _split_source_chain(source, .5)
    assert [p["type"] for p in pieces] == ["LINE", "ARC", "LINE"]
    assert pieces[0]["start"] == source[0].tolist()
    assert pieces[-1]["end"] == source[-1].tolist()
    assert all(a["end"] == b["start"] for a, b in zip(pieces, pieces[1:]))
    with pytest.raises(ValueError, match="no_source_feature_requires_split"):
        _split_source_chain(np.array([[0., 0.], [20., 0.], [40., 0.]]), .5)


def test_stable_lineage_survives_display_id_rotation_and_failed_sibling_edit(tmp_path):
    path, graph, baseline, candidate, bundle = _fixture(tmp_path, split_top=True)
    transform, _, determinant = _affines(graph, baseline)
    original_source = _base_source_entities(graph, transform, determinant)
    operations = [
        {"action": "merge_chain_as_line", "entity_ids": ["g003", "g004"], "record_id": None},
        {"action": "merge_chain_as_line", "entity_ids": ["g000", "g002"], "record_id": None},
        {"action": "merge_chain_as_line", "entity_ids": ["g000", "g001", "g002"], "record_id": None},
    ]
    rows, audit = execute_topology_edits(path, {"records": []}, baseline, candidate, bundle, operations, tmp_path / "edits")
    by_id = {row["id"]: row for row in rows}
    assert audit["operations"][1]["status"] == "rejected"
    combined = by_id["cand-edit-all"]["graph"]
    assert len(combined["entities"]) == 4
    assert audit["operations"][-1]["operation"]["included_operation_indices"] == [1, 3]
    rotated = by_id["cand-edit-01-line"]["graph"]
    mapping = rotated["entity_identity"]["parent_to_current"]
    assert mapping["g003"] == ["g000"] and mapping["g004"] == ["g000"]
    untouched = next(e for e in rotated["entities"] if e["parent_entity_ids"] == ["g000"])
    assert untouched["id"] != "g000"
    assert untouched["stable_id"] == original_source[0]["stable_id"]
    merged = rotated["entities"][0]
    assert set(merged["parent_stable_ids"]) == {original_source[3]["stable_id"], original_source[4]["stable_id"]}


def test_single_entity_retype_does_not_require_artificial_extra_segment(tmp_path):
    _, graph, baseline, _, _ = _fixture(tmp_path, split_top=True)
    rows, quality, execution = _apply_one(graph, baseline,
        {"action": "refit_entity_as_line", "entity_ids": ["g000"], "record_id": None}, [])
    assert rows[0]["type"] == "LINE" and quality["sampled_topology_valid"]
    assert execution["removed_entity_count"] == execution["replacement_count"] == 1
    with pytest.raises(ValueError, match="single_entity_refit_requires_one_entity"):
        _apply_one(graph, baseline, {"action": "refit_entity_as_line", "entity_ids": ["g000", "g001"]}, [])


def test_local_proposals_reserve_fillet_and_unlabelled_platform_repair(tmp_path):
    _, graph, baseline, _, _ = _fixture(tmp_path)
    graph["annotation_support"] = [{"record_id": "r001", "kind": "radius", "status": "candidate_supported",
        "candidate_entity_id": "g000", "arrowhead_verified": True, "target_gap_px": 5.86,
        "source_evidence": {"target_source_px": _binding()["target_source_px"]}}]
    bottom = graph["entities"][2]
    bottom.update(type="ARC", center=[70., 890.], radius=float(np.hypot(50., 1000.)), clockwise=True)
    inventory = [{"record_id": "r001", "kind": "radius", "nominal": 20., "text": "R20"}]
    operations = propose_annotation_arc_edits(graph, inventory, limit=3)
    assert [op["action"] for op in operations] == ["insert_annotated_fillet", "refit_entity_as_line", "refit_chain_as_annotated_arc"]
    assert operations[0]["entity_ids"] == ["g000", "g001"]
    assert operations[1]["entity_ids"] == ["g002"]
    replacement, quality, _ = _apply_one(graph, baseline, operations[1], inventory)
    assert replacement[0]["type"] == "LINE" and quality["sampled_topology_valid"]
    # The same shallow geometry becomes protected when a source radius label
    # reaches it. A simplification hypothesis must not outrank that evidence.
    graph["annotation_support"].append({"record_id": "r002", "kind": "radius", "status": "candidate_supported", "candidate_entity_id": "g002"})
    assert not any(op["action"] == "refit_entity_as_line" for op in propose_annotation_arc_edits(graph, inventory, limit=4))


def test_numeric_radius_binding_cannot_be_dropped_by_later_simplification():
    arc = copy.deepcopy(_corner_parts()[1])
    arc.update(stable_id="already-bound", parent_stable_ids=["already-bound"], parent_entity_ids=["g001"], radius_binding=_binding())
    points = _sample_entities([arc], max_step_px=.5)[0]
    # Even when the pixel budget permits a line chord, an already constructed
    # R20 is a constraint that a later type edit must preserve.
    with pytest.raises(ValueError, match="edit_would_discard_bound_radius"):
        _replacement_chain(points, "refit_entity_as_line", 8., [arc])


def test_junction_target_allows_explicit_adjacent_hypothesis_without_rewriting_ids():
    graph = {"units": "mm", "source_grid_pitch_px": 1., "entities": [
        {"id": "g000", "type": "LINE", "start": [0., 0.], "end": [10., 0.]},
        {"id": "g001", "type": "ARC", "start": [10., 0.], "end": [15., -5.],
         "center": [10., -5.], "radius": 5., "clockwise": True},
        {"id": "g002", "type": "LINE", "start": [15., -5.], "end": [15., -30.]},
        {"id": "g003", "type": "LINE", "start": [15., -30.], "end": [0., 0.]},
    ], "annotation_support": [{"record_id": "r001", "kind": "radius", "status": "candidate_supported",
        "candidate_entity_id": "g001", "arrowhead_verified": True, "target_gap_px": .01,
        "source_evidence": {"target_source_px": [10.5, .02]}}]}
    inventory = [{"record_id": "r001", "kind": "radius", "nominal": 5., "text": "R5"}]
    transform = lambda points: np.asarray(points) * [1., -1.]
    operation = {"action": "refit_chain_as_annotated_arc", "entity_ids": ["g000"], "record_id": "r001"}
    original = copy.deepcopy(graph)
    _, radius, evidence = _radius_edit_evidence(graph, operation, inventory, ["g000"], transform)
    assert radius == 5.
    assert evidence["target_match_method"] == "bounded_adjacent_source_target_hypothesis"
    assert evidence["target_match_entity_ids"] == ["g000"]
    assert evidence["target_gap_px"] <= 2.25
    assert graph == original and operation["entity_ids"] == ["g000"]
    # Merely being adjacent is insufficient: the far end of the other neighbor
    # must not be granted the same annotation because the model selected it.
    with pytest.raises(ValueError, match="radius_annotation_not_supported_by_named_chain"):
        _radius_edit_evidence(graph, operation, inventory, ["g002"], transform)
    graph["annotation_support"][0]["source_evidence"] = {}
    with pytest.raises(ValueError, match="radius_annotation_not_supported_by_named_chain"):
        _radius_edit_evidence(graph, operation, inventory, ["g000"], transform)


def _short_corner_fixture(tmp_path):
    path, graph, baseline, candidate, bundle = _fixture(tmp_path)
    points = [[20.,20.],[100.,20.],[120.,40.],[120.,110.],[20.,110.]]
    graph["nodes"]=[{"id":f"v{i:03d}","source_px":p,"x":p[0],"y":-p[1]} for i,p in enumerate(points)]
    graph["entities"]=[{"id":f"g{i:03d}","type":"LINE","start":[a[0],-a[1]],"end":[b[0],-b[1]],
                        "start_node":f"v{i:03d}","end_node":f"v{(i+1)%len(points):03d}"}
                       for i,(a,b) in enumerate(zip(points,points[1:]+points[:1]))]
    graph["annotation_support"]=[{"record_id":"r001","kind":"radius","status":"candidate_supported",
        "candidate_entity_id":"g001","arrowhead_verified":True,"target_gap_px":5.86,
        "source_evidence":{"target_source_px":_binding()["target_source_px"]}}]
    inventory=[{"record_id":"r001","kind":"radius","nominal":20.,"text":"R20"}]
    return graph, baseline, inventory


@pytest.mark.parametrize("ids",[["g000","g001"],["g001","g002"],["g001"]])
def test_fillet_completes_both_supports_around_short_corner_without_changing_source(tmp_path,ids):
    graph,baseline,inventory=_short_corner_fixture(tmp_path)
    original=copy.deepcopy(graph)
    operation={"action":"insert_annotated_fillet","entity_ids":ids,"record_id":"r001"}
    entities,quality,detail=_apply_one(graph,baseline,operation,inventory)
    assert detail["entity_ids"]==["g000","g001","g002"]
    assert detail["requested_entity_ids"]==ids and detail["support_scope_expanded"]
    assert detail["removed_entity_count"]==detail["replacement_count"]==3
    assert [e["type"] for e in entities[:3]]==["LINE","ARC","LINE"]
    assert entities[1]["radius"]==pytest.approx(20.,abs=1e-12) and detail["radius_binding_applied"]
    assert entities[1]["radius_binding"]["nominal"]==20.
    assert entities[0]["start"]==pytest.approx([20.,20.]) and entities[2]["end"]==pytest.approx([120.,110.])
    assert quality["sampled_topology_valid"] and graph==original and operation["entity_ids"]==ids
    proposals=propose_annotation_arc_edits(graph,inventory)
    assert proposals[0]["action"]=="insert_annotated_fillet"
    assert proposals[0]["entity_ids"]==["g000","g001","g002"]


def test_fillet_scope_does_not_expand_without_verified_source_arrow(tmp_path):
    from contour_agent.topology_editing import _complete_fillet_support_scope
    graph,baseline,inventory=_short_corner_fixture(tmp_path)
    graph["annotation_support"][0]["arrowhead_verified"]=False
    operation={"action":"insert_annotated_fillet","entity_ids":["g000","g001"],"record_id":"r001"}
    assert _complete_fillet_support_scope(graph,inventory,operation)==operation["entity_ids"]
    with pytest.raises(ValueError):_apply_one(graph,baseline,operation,inventory)


def test_combined_edits_detect_overlap_in_completed_fillet_support(tmp_path):
    from contour_agent.topology_editing import _apply_combined
    graph,baseline,inventory=_short_corner_fixture(tmp_path)
    operations=[{"action":"insert_annotated_fillet","entity_ids":["g000","g001"],"record_id":"r001"},
                {"action":"refit_entity_as_line","entity_ids":["g002"],"record_id":None}]
    with pytest.raises(ValueError,match="combined_edit_chains_overlap"):
        _apply_combined(graph,baseline,operations,inventory)


def test_source_polyline_distance_is_independent_of_collinear_vertex_density():
    from contour_agent.topology_editing import _source_polyline_distance
    sparse=np.array([[0.,0.],[100.,0.],[100.,50.]])
    dense=np.vstack([np.column_stack([np.linspace(0,100,201),np.zeros(201)]),
                     np.column_stack([np.full(100,100.),np.linspace(.5,50.,100)])])
    samples=np.array([[50.,0.],[50.,2.],[100.,25.],[103.,25.],[120.,70.]])
    expected=[0.,2.,0.,3.,math.sqrt(800.)]
    assert _source_polyline_distance(samples,sparse)==pytest.approx(expected,abs=1e-12)
    assert _source_polyline_distance(samples,dense)==pytest.approx(expected,abs=1e-12)


def test_source_path_localization_does_not_penalize_long_sparse_straight_edges():
    from contour_agent.topology_editing import _ring_path
    ring=np.array([[0.,0.],[100.,0.],[100.,50.],[0.,50.],[0.,0.]])
    samples=np.vstack([np.column_stack([np.linspace(0,100,201),np.zeros(201)]),
                       np.column_stack([np.full(100,100.),np.linspace(.5,50.,100)])])
    selected,mismatch=_ring_path(ring,ring[0],ring[2],samples)
    assert selected==pytest.approx(ring[:3])
    assert mismatch==pytest.approx(.15,abs=1e-12)


def test_sparse_straight_supports_do_not_create_artificial_holes_for_fillet():
    from contour_agent.topology_editing import _refined_straight_fillet
    parts=_corner_parts();arc=_sample_entities(parts[1:2],max_step_px=.5)[0]
    sparse=np.vstack([parts[0]["start"],arc,parts[-1]["end"]])
    result=_refined_straight_fillet(sparse,20.,.5,_binding())
    assert result[1]["radius"]==20.
    assert result[1]["source_refinement"]["maximum_reverse_error_px"]<.5
    assert result[0]["start"]==parts[0]["start"] and result[-1]["end"]==parts[-1]["end"]
    assert result[1]["source_refinement"]["outer_endpoints_fixed"]


def test_fillet_refinement_cannot_accept_optimizer_nonconvergence(monkeypatch):
    import contour_agent.topology_editing as editing
    actual=editing.minimize
    def exhausted(*args,**kwargs):
        fit=actual(*args,**kwargs);fit.success=False
        return fit
    monkeypatch.setattr(editing,"minimize",exhausted)
    source=_sample_entities(_corner_parts(),max_step_px=.5)[0]
    with pytest.raises(ValueError,match="source_does_not_support_exact_annotated_fillet"):
        editing._refined_straight_fillet(source,20.,.5,_binding())
