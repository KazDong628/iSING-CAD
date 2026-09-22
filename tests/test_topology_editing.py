import copy
import hashlib
import json

import cv2
import httpx
import numpy as np
from PIL import Image
import pytest

from contour_agent.config import Settings
from contour_agent.topology import build_topology
from contour_agent.topology_edit_provider import (
    TopologyEditProvider,
    TopologyEvaluationProvider,
    _rank_radius_detail_records,
    validate_edit_response,
    validate_evaluation_response,
)
from contour_agent.topology_editing import execute_topology_edits
from contour_agent.topology_editing import (
    _fit_replacement,
    _radius_edit_evidence,
    propose_annotation_arc_edits,
)
from contour_agent.vision_provider import _InspectionError


def _graph_ids(count=7):
    return {"entities": [{"id": f"g{i:03d}", "type": "LINE"} for i in range(count)]}


def _edit_payload(entity_ids=("g000", "g001")):
    return {
        "observation": "相邻短线段共线，可由本地内核合并。",
        "operations": [{
            "action": "merge_chain_as_line",
            "entity_ids": list(entity_ids),
            "record_id": "r001",
            "evidence_tags": ["micro_segment", "collinear_support"],
        }],
        "confidence": "high",
    }


def test_edit_and_evaluation_schemas_only_accept_whitelisted_ids_and_decisions():
    graph = _graph_ids()
    result = validate_edit_response(json.dumps(_edit_payload()), graph, {"r001"})
    assert result["operations"][0]["entity_ids"] == ["g000", "g001"]

    for ids in (("g000", "g002"), ("g000", "unknown")):
        with pytest.raises(_InspectionError):
            validate_edit_response(json.dumps(_edit_payload(ids)), graph, {"r001"})
    extra = _edit_payload()
    extra["operations"][0]["start"] = [1, 2]
    with pytest.raises(_InspectionError):
        validate_edit_response(json.dumps(extra), graph, {"r001"})
    with pytest.raises(_InspectionError):
        validate_edit_response(json.dumps({"observation": "无", "operations": [], "confidence": "high"}), graph, set())

    annotated = _edit_payload(("g003",))
    annotated["operations"][0].update(action="refit_chain_as_annotated_arc", record_id="r001",
                                       evidence_tags=["annotation_target", "cocircular_support"])
    assert validate_edit_response(json.dumps(annotated), graph, {"r001"})["operations"][0]["entity_ids"] == ["g003"]
    annotated["operations"][0]["record_id"] = None
    with pytest.raises(_InspectionError):
        validate_edit_response(json.dumps(annotated), graph, {"r001"})

    accepted = validate_evaluation_response(json.dumps({
        "candidate_id": "cand-edit-01-line", "observation": "边界连续且减少碎线。",
        "decision": "accept_edit", "evidence_tags": ["continuity", "primitive_count"],
        "confidence": "high",
    }), {"cand-base", "cand-edit-01-line"}, "cand-base")
    assert accepted["decision"] == "accept_edit"
    with pytest.raises(_InspectionError):
        validate_evaluation_response(json.dumps({
            "candidate_id": "unknown", "observation": "无", "decision": "accept_edit",
            "evidence_tags": [], "confidence": "high",
        }), {"cand-base", "cand-edit-01-line"}, "cand-base")
    with pytest.raises(_InspectionError):
        validate_evaluation_response(json.dumps({
            "candidate_id": "cand-edit-01-line", "observation": "无", "decision": "preserve_base",
            "evidence_tags": [], "confidence": "high",
        }), {"cand-base", "cand-edit-01-line"}, "cand-base")


def _local_fixture(tmp_path):
    # Top is split into three collinear objects and right into two. The source
    # image contains only the rectangular material boundary.
    points = [[20, 20], [70, 20], [120, 20], [200, 20], [200, 70], [200, 120], [20, 120]]
    image = np.full((160, 240, 3), 255, np.uint8)
    cv2.polylines(image, [np.asarray(points, np.int32)], True, (0, 0, 0), 2)
    image_path = tmp_path / "source.png"
    assert cv2.imencode(".png", image)[0]
    cv2.imencode(".png", image)[1].tofile(str(image_path))
    raw = [*points, points[0]]
    baseline = {
        "extraction": {"raw_polyline_px": raw, "polyline_px": raw, "model": {"size": 240}},
        "coordinate_system": {"units": "pixel", "origin_source_px": [0.0, 0.0]},
        "scale": {"status": "unresolved", "pixels_per_mm": None},
    }
    graph = build_topology(image_path, {"records": []}, baseline, tmp_path / "base")
    graph = copy.deepcopy(graph)
    graph["nodes"] = [
        {"id": f"v{i:03d}", "x": float(x), "y": float(-y), "source_px": [float(x), float(y)]}
        for i, (x, y) in enumerate(points)
    ]
    graph["entities"] = [{
        "id": f"g{i:03d}", "type": "LINE", "start_node": f"v{i:03d}",
        "end_node": f"v{(i + 1) % len(points):03d}",
        "start": [float(points[i][0]), float(-points[i][1])],
        "end": [float(points[(i + 1) % len(points)][0]), float(-points[(i + 1) % len(points)][1])],
        "parameter_source": "test_oversegmentation", "dimension_bound": False,
    } for i in range(len(points))]
    graph["relations"] = []
    candidate = {"id": "cand-base", "graph": graph}
    bundle = {
        "source_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "base_graph_sha256": "parent-source-topology-hash",
        "annotation_inventory": [], "candidates": [candidate],
    }
    return image_path, baseline, candidate, bundle


def test_local_executor_refits_single_and_combined_edits_without_mutating_base(tmp_path):
    image_path, baseline, candidate, bundle = _local_fixture(tmp_path)
    original = copy.deepcopy(candidate)
    operations = [
        {"action": "merge_chain_as_line", "entity_ids": ["g000", "g001", "g002"],
         "record_id": None, "evidence_tags": ["collinear_support"]},
        {"action": "merge_chain_as_line", "entity_ids": ["g003", "g004"],
         "record_id": None, "evidence_tags": ["collinear_support"]},
    ]

    rows, audit = execute_topology_edits(
        image_path, {"records": []}, baseline, candidate, bundle, operations, tmp_path / "edits")

    by_id = {row["id"]: row for row in rows}
    assert set(by_id) == {"cand-edit-01-line", "cand-edit-02-line", "cand-edit-all"}
    combined = by_id["cand-edit-all"]
    assert combined["entity_counts"] == {"total": 4, "LINE": 4, "ARC": 0}
    assert combined["ground_truth_used"] is False
    assert combined["graph"]["source_evidence"]["topology_edit"]["net_entity_reduction"] == 3
    assert combined["graph"]["validation"]["ordered_entity_cycle"] is True
    assert all(entity["type"] in {"LINE", "ARC"} for entity in combined["graph"]["entities"])
    assert all(entity["end"] == pytest.approx(combined["graph"]["entities"][(i + 1) % 4]["start"])
               for i, entity in enumerate(combined["graph"]["entities"]))
    assert Image.open(combined["overlay_path"]).size == (240, 160)
    assert audit["accepted_candidates"] == 3 and audit["ground_truth_used"] is False
    assert candidate == original


def test_local_executor_rejects_bad_chain_without_mutating_base(tmp_path):
    image_path, baseline, candidate, bundle = _local_fixture(tmp_path)
    original = copy.deepcopy(candidate)
    rows, audit = execute_topology_edits(image_path, {"records": []}, baseline, candidate, bundle, [{
        "action": "merge_chain_as_line", "entity_ids": ["g000", "g002"],
        "record_id": None, "evidence_tags": [],
    }], tmp_path / "edits")
    assert rows == []
    assert audit["operations"][0]["reason"] == "edit_entities_not_ordered_consecutive_chain"
    assert candidate == original


def test_radius_annotation_protects_chain_from_line_and_resolves_fixed_radius():
    graph = {
        "units": "mm", "source_grid_pitch_px": 4.0,
        "annotation_support": [{"record_id": "r022", "kind": "radius", "nominal": 36.0,
                                "status": "candidate_supported", "candidate_entity_id": "g023",
                                "target_gap_px": .8, "arrowhead_verified": False}],
    }
    inventory = [{"record_id": "r022", "text": "R36", "kind": "radius", "nominal": 36.0}]
    transform = lambda points: np.asarray(points, float) * [4., -4.] + [10., 20.]
    with pytest.raises(ValueError, match="radius_annotation_protects_arc_chain"):
        _radius_edit_evidence(graph, {"action": "merge_chain_as_line", "record_id": None},
                              inventory, ["g022", "g023"], transform)
    force_arc, radius_px, binding = _radius_edit_evidence(
        graph, {"action": "refit_chain_as_annotated_arc", "record_id": "r022"},
        inventory, ["g023"], transform)
    assert force_arc is True and radius_px == pytest.approx(144.)
    assert binding["nominal"] == 36.0 and binding["target_gap_px"] == pytest.approx(.8)


def test_annotated_arc_keeps_arc_semantics_when_noisy_boundary_rejects_exact_radius():
    theta = np.linspace(0.0, 0.8, 32)
    points = np.column_stack([25.0 * np.cos(theta), 25.0 * np.sin(theta)])
    binding = {"record_id": "r022", "text": "R36", "nominal": 36.0}

    # An 8 px declared radius cannot span this chord.  The annotation still
    # proves the primitive class, so the executor must retain a source-fitted
    # ARC while leaving the unresolved numeric radius unbound.
    replacement = _fit_replacement(
        points, "refit_chain_as_annotated_arc", 0.5,
        annotated_radius_px=8.0, radius_binding=binding,
    )

    assert replacement["type"] == "ARC"
    assert replacement["radius"] == pytest.approx(25.0, rel=0.02)
    assert replacement["radius_annotation_evidence"] == binding
    assert "radius_binding" not in replacement


def _provider_candidate(candidate_id, overlay, *, count=5):
    entities = [{"id": f"g{i:03d}", "type": "LINE", "start": [i, 0], "end": [i + 1, 0]}
                for i in range(count)]
    return {
        "id": candidate_id, "overlay_path": str(overlay), "ground_truth_used": False,
        "graph": {"ground_truth_used": False, "entities": entities, "relations": [],
                  "annotation_support": [], "validation": {"closed": True, "connected": True,
                  "simple": True, "ordered_entity_cycle": True}},
        "entity_counts": {"total": count, "LINE": count, "ARC": 0},
        "source_stroke_support": {"edge_supported_fraction": .95},
        "annotation_support": {"compatibility_fraction": .8},
        "unsupported_primitive_count": min(1, count), "binding_candidate_ids": [],
        "ground_truth_secret": "NEVER-SEND-GT",
    }


def _patch_client(monkeypatch, handler):
    original = httpx.AsyncClient

    def factory(*args, **kwargs):
        return original(*args, **kwargs, transport=httpx.MockTransport(handler))

    monkeypatch.setattr("contour_agent.topology_edit_provider.httpx.AsyncClient", factory)


def _chat_response(value):
    return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
        "content": json.dumps(value, ensure_ascii=False)}}]})


def test_editor_and_independent_evaluator_send_images_but_no_geometry_or_gt(monkeypatch, tmp_path):
    source = tmp_path / "source.png"
    base_overlay = tmp_path / "base.png"
    edit_overlay = tmp_path / "edit.png"
    for path in (source, base_overlay, edit_overlay):
        Image.new("RGB", (96, 64), "white").save(path)
    base = _provider_candidate("cand-base", base_overlay, count=5)
    edited = _provider_candidate("cand-edit-01-line", edit_overlay, count=4)
    payloads = []

    def handler(request):
        payload = json.loads(request.content)
        payloads.append(payload)
        if len(payloads) == 1:
            return _chat_response(_edit_payload())
        return _chat_response({
            "candidate_id": "cand-edit-01-line", "observation": "编辑候选连续且少一个冗余对象。",
            "decision": "accept_edit", "evidence_tags": ["continuity", "primitive_count"],
            "confidence": "high",
        })

    _patch_client(monkeypatch, handler)
    settings = Settings(api_key="private-test-key")
    edit_receipt = TopologyEditProvider(settings).propose(
        source, base_overlay, base,
        [{"record_id": "r001", "text": "R40", "kind": "radius", "nominal": 40,
          "leader_status": "directed_arrow_candidate"}],
    )
    evaluation_receipt = TopologyEvaluationProvider(settings).select(
        source, [base, edited], "cand-base")

    assert edit_receipt["status"] == "succeeded" and edit_receipt["schema_success"]
    assert evaluation_receipt["status"] == "succeeded" and evaluation_receipt["schema_success"]
    assert evaluation_receipt["selected_candidate_id"] == "cand-edit-01-line"
    assert edit_receipt["coordinates_sent"] is False and evaluation_receipt["coordinates_sent"] is False
    editor_wire, evaluator_wire = (json.dumps(row, ensure_ascii=False) for row in payloads)
    assert editor_wire.count('"type": "image_url"') == 2
    assert evaluator_wire.count('"type": "image_url"') == 3
    for wire in (editor_wire, evaluator_wire):
        assert "NEVER-SEND-GT" not in wire and '"start"' not in wire and '"end"' not in wire
    assert "private-test-key" not in json.dumps([edit_receipt, evaluation_receipt])


def test_editor_sends_radius_detail_and_protected_entity_semantics(monkeypatch, tmp_path):
    source, overlay = tmp_path / "source.png", tmp_path / "overlay.png"
    Image.new("RGB", (240, 160), "white").save(source)
    Image.new("RGB", (240, 160), "white").save(overlay)
    candidate = _provider_candidate("cand-base", overlay, count=5)
    candidate["graph"]["annotation_support"] = [{
        "record_id": "r022", "text": "R36", "kind": "radius", "nominal": 36.0,
        "candidate_entity_id": "g001", "candidate_entity_type": "LINE",
        "target_gap_px": 1.0, "arrowhead_verified": True, "status": "candidate_supported",
    }]
    payloads = []

    def handler(request):
        payloads.append(json.loads(request.content))
        return _chat_response({"observation": "R36引线指向g001，应恢复为圆弧。", "operations": [{
            "action": "refit_chain_as_annotated_arc", "entity_ids": ["g001"],
            "record_id": "r022", "evidence_tags": ["annotation_target", "source_boundary"],
        }], "confidence": "high"})

    _patch_client(monkeypatch, handler)
    receipt = TopologyEditProvider(Settings(api_key="private-test-key")).propose(
        source, overlay, candidate, [{
            "record_id": "r022", "text": "R36", "kind": "radius", "nominal": 36.0,
            "source_box": [[80, 70], [110, 70], [110, 95], [80, 95]],
            "leader_status": "directed_arrow_candidate",
            "leader": {"target_source_px": [145, 82], "arrowhead_verified": True},
        }])
    wire = json.dumps(payloads[0], ensure_ascii=False)
    packet = json.loads(payloads[0]["messages"][1]["content"][0]["text"])
    assert receipt["status"] == "succeeded"
    assert wire.count('"type": "image_url"') == 3
    assert packet["protected_radius_entities"] == ["g001"]
    assert receipt["operations"][0]["action"] == "refit_chain_as_annotated_arc"


def test_radius_detail_ranking_keeps_unique_r36_target_ahead_of_ambiguous_ocr_order():
    inventory = [{"record_id": f"r{i:03d}", "kind": "radius", "nominal": nominal,
                  "leader": {"target_source_px": [10 + i, 20]}, "source_box": [[0, 0], [2, 2]]}
                 for i, nominal in enumerate((3, 5, 60, 3, 40, 36, 5, 40), 1)]
    targets = ("g017", "g017", "g018", "g009", "g009", "g023", "g022", "g027")
    types = {"g017": "ARC", "g018": "ARC", "g009": "LINE", "g023": "ARC",
             "g022": "ARC", "g027": "LINE"}
    radii = {"g017": 712.0, "g018": 993.0, "g023": 111.4, "g022": 651.0}
    graph = {
        "units": "mm",
        "entities": [{"id": entity_id, "type": kind, "radius": radii.get(entity_id)}
                     for entity_id, kind in types.items()],
        "annotation_support": [{
            "record_id": record["record_id"], "kind": "radius", "status": "candidate_supported",
            "candidate_entity_id": entity_id, "target_gap_px": 0.8,
            "arrowhead_verified": record["record_id"] in {"r004", "r008"},
        } for record, entity_id in zip(inventory, targets)],
    }

    selected = _rank_radius_detail_records(inventory, graph, limit=6)
    selected_ids = [record["record_id"] for _, record, _ in selected]

    assert "r006" in selected_ids  # R36 -> g023 survives the bounded image budget.
    assert len({entity_id for _, _, entity_id in selected if entity_id}) == len(selected)

    local_operations = propose_annotation_arc_edits(graph, inventory, limit=4)
    assert any(row["record_id"] == "r006" and row["entity_ids"] == ["g023"]
               for row in local_operations)
    assert all(row["action"] == "refit_chain_as_annotated_arc" for row in local_operations)


def test_editor_without_key_abstains_before_reading_missing_images(tmp_path):
    missing = tmp_path / "missing.png"
    candidate = _provider_candidate("cand-base", missing)
    receipt = TopologyEditProvider(Settings(api_key="")).propose(missing, missing, candidate, [])
    assert receipt["status"] == "not_configured"
    assert receipt["network_requests"] == 0 and receipt["operations"] == []
