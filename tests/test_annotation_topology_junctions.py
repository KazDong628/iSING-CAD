"""Annotation targets at joints are hypotheses until source geometry verifies them."""

import json

import httpx
import numpy as np
import pytest
from PIL import Image

from contour_agent.config import Settings
from contour_agent.topology_candidates import _entity_annotation_support
from contour_agent.topology_edit_provider import TopologyEditProvider, _rank_radius_detail_records
from contour_agent.topology_editing import _radius_edit_evidence, propose_annotation_arc_edits


def _joint_graph():
    return {"units": "mm", "source_grid_pitch_px": 1., "entities": [
        {"id": "g000", "type": "LINE", "start": [0., 0.], "end": [10., 0.]},
        {"id": "g001", "type": "ARC", "start": [10., 0.], "end": [15., -5.],
         "center": [10., -5.], "radius": 5., "clockwise": True},
        {"id": "g002", "type": "LINE", "start": [15., -5.], "end": [15., -30.]},
        {"id": "g003", "type": "LINE", "start": [15., -30.], "end": [0., 0.]},
    ]}


def test_directed_joint_label_keeps_adjacent_arc_as_source_hypothesis():
    source_entities = [
        {"type": "LINE", "start": [0., 0.], "end": [10., 0.]},
        {"type": "ARC", "start": [10., 0.], "end": [15., 5.],
         "center": [10., 5.], "radius": 5., "clockwise": False},
        {"type": "LINE", "start": [15., 5.], "end": [15., 30.]},
        {"type": "LINE", "start": [15., 30.], "end": [0., 0.]},
    ]
    inventory = [{"record_id": "r001", "kind": "radius", "nominal": 5.,
                  "leader": {"target_source_px": [9.5, 0.], "arrowhead_verified": True}}]
    support, _ = _entity_annotation_support(source_entities, inventory, 1.)
    assert support[0]["candidate_entity_id"] == "g000"
    assert support[0]["adjacent_target_hypotheses"] == [{
        "entity_id": "g001", "entity_type": "ARC", "target_gap_px": pytest.approx(.5, abs=.1)}]

    graph = _joint_graph()
    graph["annotation_support"] = support
    transform = lambda points: np.asarray(points, float) * [1., -1.]
    with pytest.raises(ValueError, match="radius_annotation_protects_arc_chain"):
        _radius_edit_evidence(graph, {"action": "refit_entity_as_line", "record_id": None},
                              inventory, ["g001"], transform)
    # An undirected line or a target far from the arc cannot protect it.
    graph["annotation_support"][0]["arrowhead_verified"] = False
    assert _radius_edit_evidence(graph, {"action": "refit_entity_as_line"},
                                 inventory, ["g001"], transform) == (False, None, None)
    graph["annotation_support"][0]["arrowhead_verified"] = True
    graph["annotation_support"][0]["source_evidence"]["target_source_px"] = [2., 0.]
    assert _radius_edit_evidence(graph, {"action": "refit_entity_as_line"},
                                 inventory, ["g001"], transform) == (False, None, None)


def test_two_labels_on_one_coarse_line_offer_local_edits_and_two_detail_panels():
    graph = {"units": "mm", "source_grid_pitch_px": 1.,
             "nodes": [{"id": f"v{i:03d}", "source_px": point}
                       for i, point in enumerate(([0., 0.], [100., 0.], [100., 30.], [0., 30.]))],
             "entities": [
                 {"id": "g000", "type": "LINE", "start_node": "v000", "end_node": "v001"},
                 {"id": "g001", "type": "LINE", "start_node": "v001", "end_node": "v002"},
                 {"id": "g002", "type": "LINE", "start_node": "v002", "end_node": "v003"},
                 {"id": "g003", "type": "LINE", "start_node": "v003", "end_node": "v000"},
             ]}
    records = [{"record_id": "r001", "kind": "radius", "nominal": 3., "text": "R3",
                "leader": {"target_source_px": [2., 0.], "arrowhead_verified": True}},
               {"record_id": "r002", "kind": "radius", "nominal": 5., "text": "R5",
                "leader": {"target_source_px": [98., 0.], "arrowhead_verified": True}}]
    graph["annotation_support"] = [{"record_id": row["record_id"], "kind": "radius",
        "status": "candidate_supported", "candidate_entity_id": "g000",
        "arrowhead_verified": True, "target_gap_px": 0.,
        "source_evidence": {"target_source_px": row["leader"]["target_source_px"]}}
        for row in records]
    operations = propose_annotation_arc_edits(graph, records, limit=4)
    assert any(row["action"] == "insert_annotated_fillet" for row in operations)
    assert not any(row["action"] == "refit_chain_as_annotated_arc" and row["entity_ids"] == ["g000"]
                   for row in operations)
    details = _rank_radius_detail_records(records, graph, limit=3)
    assert {row[1]["record_id"] for row in details} == {"r001", "r002"}


def test_online_editor_sees_joint_hypothesis_without_receiving_target_coordinates(monkeypatch, tmp_path):
    source, overlay = tmp_path / "source.png", tmp_path / "overlay.png"
    Image.new("RGB", (64, 48), "white").save(source)
    Image.new("RGB", (64, 48), "white").save(overlay)
    graph = _joint_graph()
    graph["annotation_support"] = [{"record_id": "r001", "kind": "radius", "nominal": 5.,
        "status": "candidate_supported", "candidate_entity_id": "g000", "candidate_entity_type": "LINE",
        "arrowhead_verified": True, "target_gap_px": 0.,
        "adjacent_target_hypotheses": [{"entity_id": "g001", "entity_type": "ARC", "target_gap_px": .5}],
        "source_evidence": {"target_source_px": [9.5, 0.]}}]
    captured = []
    original = httpx.AsyncClient

    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps({"observation": "接点附近的半径归属仍需核对。", "operations": [],
                                   "confidence": "abstain"}, ensure_ascii=False)}}]})

    monkeypatch.setattr("contour_agent.topology_edit_provider.httpx.AsyncClient",
                        lambda *args, **kwargs: original(*args, **kwargs, transport=httpx.MockTransport(handler)))
    candidate = {"id": "cand-base", "graph": graph}
    record = {"record_id": "r001", "kind": "radius", "nominal": 5., "text": "R5",
              "source_box": [[1, 1], [2, 2]],
              "leader": {"target_source_px": [9.5, 0.], "arrowhead_verified": True}}
    receipt = TopologyEditProvider(Settings(api_key="test-only-key")).propose(
        source, overlay, candidate, [record])
    packet = json.loads(captured[0]["messages"][1]["content"][0]["text"])
    assert receipt["status"] == "succeeded"
    assert packet["protected_radius_entities"] == ["g000", "g001"]
    assert packet["entity_annotation_support"][0]["adjacent_target_hypotheses"][0]["entity_id"] == "g001"
    assert "target_source_px" not in json.dumps(packet)
