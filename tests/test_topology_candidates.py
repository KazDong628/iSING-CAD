"""Bounded source-only topology candidate generation tests."""
import copy
import hashlib
import json
import math

import cv2
import numpy as np
import pytest

from contour_agent.topology import build_topology
from contour_agent.topology_candidates import generate_topology_candidates, materialize_selected_candidate


def _fixture(tmp_path):
    image = np.full((280, 380, 3), 255, np.uint8)
    # One material boundary with two small source-observed shoulders.  Their
    # survival at different tolerances creates useful competing complexities.
    boundary = np.asarray([[30, 70], [145, 70], [154, 58], [164, 70], [280, 70],
                           [315, 82], [335, 110], [340, 200], [322, 230], [292, 244],
                           [190, 244], [180, 256], [170, 244], [52, 244], [30, 220]], np.int32)
    cv2.polylines(image, [boundary], True, (0, 0, 0), 2, cv2.LINE_AA)
    # Source annotation candidate: OCR box is declared below; only its leader
    # stroke is required here, not synthetic text content.
    cv2.line(image, (115, 35), (154, 59), (0, 0, 0), 2, cv2.LINE_AA)
    path = tmp_path/"source.png"
    cv2.imencode(".png", image)[1].tofile(str(path))
    raw = np.vstack([boundary, boundary[0]]).astype(float).tolist()
    model = {"extraction": {"raw_polyline_px": raw, "polyline_px": raw,
                             "model": {"size": 190,
                                       "training_provenance": {"reference_path": "must-not-leak"}}},
             "coordinate_system": {"units": "pixel", "origin_source_px": [0., 0.]},
             "scale": {"status": "unresolved", "pixels_per_mm": None}}
    document = {"records": [{"text": "R12", "box": [[72, 23], [112, 23], [112, 40], [72, 40]]}]}
    base = build_topology(path, document, model, tmp_path/"base")
    return path, document, model, base


def test_candidates_are_bounded_source_only_shared_node_line_arc_graphs(tmp_path):
    path, document, model, base = _fixture(tmp_path)
    bundle = generate_topology_candidates(path, document, model, base, tmp_path/"candidates",
                                          max_candidates=4)
    assert bundle["schema_version"] == "source-topology-candidate-set-v1"
    assert bundle["ground_truth_used"] is False
    assert bundle["candidate_count"] == 4
    assert bundle["candidates"][0]["strategy"]["name"] == "base_topology"
    assert bundle["candidates"][0]["entity_counts"]["total"] == len(base["entities"])
    assert bundle["generation"]["attempted_strategies"] == 4
    assert len(bundle["candidates"]) <= 5
    for candidate in bundle["candidates"]:
        graph = candidate["topology"]
        assert graph == candidate["graph"]
        assert graph["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert graph["ground_truth_used"] is False
        assert set(candidate["entity_counts"]) == {"total", "LINE", "ARC"}
        assert 0 <= candidate["annotation_coverage"] <= 1
        assert candidate["unsupported_primitive_count"] >= 0
        assert isinstance(candidate["binding_candidate_ids"], list)
        assert isinstance(candidate["evidence_ids"], list)
        assert candidate["relation_ids"] == [row["id"] for row in graph["relations"]]
        assert {entity["type"] for entity in graph["entities"]} <= {"LINE", "ARC"}
        for index, entity in enumerate(graph["entities"]):
            following = graph["entities"][(index+1) % len(graph["entities"])]
            assert entity["end_node"] == following["start_node"]
            assert entity["end"] == pytest.approx(following["start"])
        assert (tmp_path/"candidates"/candidate["overlay_path"]).is_file()
        residual = candidate["source_residual"]["source_boundary_deviation_px"]
        assert math.isfinite(residual["conservative_upper_bound_px"])
        assert candidate["planner_signals"]["reference_accuracy_measured"] is False
    saved = json.loads((tmp_path/"candidates/topology-candidates.json").read_text(encoding="utf-8"))
    assert saved == bundle
    assert "must-not-leak" not in json.dumps(saved)


def test_materialize_requires_unchanged_bundle_member_and_copies_overlay(tmp_path):
    path, document, model, base = _fixture(tmp_path)
    output = tmp_path/"candidates"
    bundle = generate_topology_candidates(path, document, model, base, output, max_candidates=3)
    selected = bundle["candidates"][1]
    result = materialize_selected_candidate(selected, bundle, output)
    assert result["status"] == "materialized" and result["ground_truth_used"] is False
    assert json.loads((output/"topology.json").read_text(encoding="utf-8")) == selected["graph"]
    assert (output/"topology-overlay.png").read_bytes() == (output/selected["overlay_path"]).read_bytes()
    changed = json.loads(json.dumps(selected))
    changed["graph"]["entities"][0]["start"][0] += 1
    with pytest.raises(ValueError, match="unchanged member"):
        materialize_selected_candidate(changed, bundle, output)


def test_new_candidates_refit_original_mask_not_an_already_approximated_cad(tmp_path,monkeypatch):
    from contour_agent import topology_candidates as module
    path,document,model,base=_fixture(tmp_path)
    expected=np.asarray(model["extraction"]["raw_polyline_px"])
    original=module.fit_polyline
    observations=[]
    def measured(points,*args,**kwargs):
        observations.append(np.asarray(points).copy())
        return original(points,*args,**kwargs)
    monkeypatch.setattr(module,"fit_polyline",measured)
    bundle=generate_topology_candidates(path,document,model,base,max_candidates=3)
    assert observations and all(np.array_equal(points,expected) for points in observations)
    assert bundle["annotation_inventory_summary"]["boundary_observation_source"]=="initial_extraction_raw_polyline_px"
    # The retained candidate is still the original CAD rollback, not a silent
    # overwrite of baseline geometry with a different segmentation fit.
    first=bundle["candidates"][0]["graph"]["entities"]
    assert len(first)==len(base["entities"])
    for old,new in zip(base["entities"],first):
        assert old["type"]==new["type"]
        assert old["start"]==pytest.approx(new["start"])


def test_invalid_raw_observation_cannot_silently_fall_back_to_cad_samples(tmp_path):
    path,document,model,base=_fixture(tmp_path)
    model["extraction"]["raw_polyline_px"]=[[0,0],[20,20],[0,20],[20,0],[0,0]]
    with pytest.raises(ValueError,match="immutable source mask"):
        generate_topology_candidates(path,document,model,base,max_candidates=3)


def test_constructed_radius_roundtrip_preserves_nominal_only_after_incidence_check():
    from contour_agent.topology_candidates import _to_graph
    center=np.array([1000.,1000.])
    a=center+3*np.array([math.cos(.27),math.sin(.27)])
    b=center+3*np.array([math.cos(1.3),math.sin(1.3)])
    tip=center+np.array([6.,6.])
    source=[{"type":"ARC","start":a.tolist(),"end":b.tolist(),"center":center.tolist(),
             "radius":3.,"clockwise":False,"radius_binding":{"record_id":"r000","nominal":3.}},
            {"type":"LINE","start":b.tolist(),"end":tip.tolist()},
            {"type":"LINE","start":tip.tolist(),"end":a.tolist()}]
    def convert():
        return _to_graph(source,lambda points:points,1.,{"units":"mm"},"test","source","parent",1.,
                         {"sampled_topology_valid":True},[],{"source_stroke_support":{}},{})
    result=convert()["entities"][0]
    assert result["radius"]==3.
    assert result["dimension_bound"] is False  # Construction does not prove association.
    source[1]["start"][0]+=.1
    with pytest.raises(ValueError,match="incidence_failed"):
        convert()


def test_source_hash_gt_guard_and_candidate_bound_are_enforced(tmp_path):
    path, document, model, base = _fixture(tmp_path)
    changed = json.loads(json.dumps(base))
    changed["source_sha256"] = "0"*64
    with pytest.raises(ValueError, match="source hash"):
        generate_topology_candidates(path, document, model, changed, max_candidates=3)
    changed = json.loads(json.dumps(base))
    changed["ground_truth_used"] = True
    with pytest.raises(ValueError, match="ground_truth_used=false"):
        generate_topology_candidates(path, document, model, changed, max_candidates=3)
    for value in (2, 6, True):
        with pytest.raises(ValueError, match="3 to 5"):
            generate_topology_candidates(path, document, model, base, max_candidates=value)


def test_candidate_rechecks_inherited_radius_segments_without_trusting_old_verdict(tmp_path, monkeypatch):
    from contour_agent import topology_candidates as module

    path, _, model, _ = _fixture(tmp_path)
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    cv2.line(image, (140, 210), (180, 255), (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imencode(".png", image)[1].tofile(str(path))
    document = {"records": [
        {"text": "R12", "box": [[72, 23], [112, 23], [112, 40], [72, 40]]},
        {"text": "R8", "box": [[106, 187], [145, 187], [145, 218], [106, 218]]},
    ]}
    base = build_topology(path, document, model, tmp_path / "base-with-two-labels")
    segments = {"r000": [[115., 35.], [154., 59.]],
                "r001": [[140., 210.], [180., 255.]]}
    attempts = []

    # The transfer contract is under test here. The independent source-pixel
    # verifier has its own tests; this stub accepts only the two drawn strokes.
    def recheck(gray, record, segment, boundary, band, contours, *, verifier):
        attempts.append(record["id"])
        expected = np.asarray(segments[record["id"]])
        if not np.allclose(segment, expected) or gray[int(expected[1, 1]), int(expected[1, 0])] >= 200:
            return None
        return {"score": 1., "segment_px": expected.tolist(), "arrowhead_verified": True,
                "arrowhead": {"tip_px": expected[1].tolist(), "verified": True}}

    monkeypatch.setattr(module, "_leaders", lambda *args: [])
    monkeypatch.setattr(module, "native_radius_leader_segments", lambda *args: [])
    monkeypatch.setattr(module, "verify_source_hough_leader", recheck)

    stale = copy.deepcopy(base)
    stale["annotation_support"] = [
        {"record_id": record_id, "kind": "radius", "arrowhead_verified": True,
         "source_evidence": {"segment_px": [[5., 5.], [8., 8.]]}}
        for record_id in segments]
    refreshed = copy.deepcopy(base)
    refreshed["annotation_support"] = [
        {"record_id": record_id, "kind": "radius", "arrowhead_verified": False,
         "source_evidence": {"segment_px": segment}}
        for record_id, segment in segments.items()]
    checkpoint = copy.deepcopy(stale)
    checkpoint["radius_source_segment_hypotheses"] = [
        {"record_id": record_id, "kind": "radius", "arrowhead_verified": True,
         "candidate_entity_id": "old-target",
         "source_evidence": {"segment_px": segment, "arrowhead": {"verified": True}}}
        for record_id, segment in segments.items()]

    old_bundle = generate_topology_candidates(path, document, model, stale, max_candidates=3)
    new_bundle = generate_topology_candidates(path, document, model, refreshed, max_candidates=3)
    checkpoint_bundle = generate_topology_candidates(path, document, model, checkpoint, max_candidates=3)
    old_graph = old_bundle["candidates"][0]["graph"]
    new_graph = new_bundle["candidates"][0]["graph"]
    checkpoint_graph = checkpoint_bundle["candidates"][0]["graph"]
    assert old_graph["entities"] == new_graph["entities"]
    assert checkpoint_graph["entities"] == old_graph["entities"]
    assert all(row["status"] == "no_source_leader_candidate" for row in old_graph["annotation_support"])
    assert all(row["arrowhead_verified"] is True for row in new_graph["annotation_support"])
    assert all(row["arrowhead_verified"] is True for row in checkpoint_graph["annotation_support"])
    assert checkpoint_graph["radius_source_segment_hypotheses"] == [
        {"record_id": record_id, "kind": "radius", "source_evidence": {"segment_px": segment}}
        for record_id, segment in segments.items()]
    assert {row["record_id"] for row in new_bundle["annotation_inventory"]
            if row["reverified_inherited_source_segment_count"] == 1} == set(segments)
    assert set(attempts) == set(segments)
