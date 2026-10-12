"""Derived dimensions need finite source attachments and explicit uncertainty."""
from copy import deepcopy
import json

import numpy as np
from PIL import Image, ImageDraw
import pytest

from contour_agent.binding_provider import bounded_inventory
from contour_agent.constraint_binding import (
    _derived_extension_supports,
    _derived_station_groups,
    _length_candidates,
    _merge_derived_station_groups,
    analyze_constraint_bindings,
)
from contour_agent.ocr import canonical_records
from contour_agent.parametric_solver import solve_parametric
from test_parametric_solver import rectangle


def stroke(lo, hi, cross=100., thickness=2.):
    return {"lo": float(lo), "hi": float(hi), "cross": float(cross),
            "thickness": float(thickness), "span": float(hi-lo+1)}


def level_location(identifier="g0", *, interval=(80., 380.), station=100.):
    return {"id": "level_"+identifier+"_y", "x": 30., "y": 20.,
            "source_px": [230., station],
            "anchor": {"type": "support_level", "entity_id": identifier, "axis": "y"},
            "evidence": {"source_attachment_interval_px": list(interval),
                         "observed_source_level_px": station,
                         "nominal_used": False, "coordinates_locked": False}}


def extremum_location(*, observed=(150., 100.), uncertainty=1.):
    return {"id": "extremum_arc_y_maximum", "x": 25., "y": 20.,
            "source_px": [150., 100.],
            "anchor": {"type": "curve_extremum", "entity_id": "arc", "axis": "y", "sense": "maximum"},
            "evidence": {"observed_source_extremum_px": list(observed),
                         "source_position_uncertainty_px": uncertainty,
                         "section_interpretation_not_resolved": True,
                         "nominal_used": False, "coordinates_locked": False}}


def test_level_attachment_uses_nearest_finite_face_point_not_midpoint_or_infinite_extension():
    location = level_location()
    observed = _derived_station_groups([location], [stroke(25., 90.)], 1, 100., 30., 4.)
    assert len(observed) == 1
    option, evidence = observed[0]
    assert option[2] is location
    assert evidence["observed_attachment_px"] == [80., 100.]
    assert evidence["station_interpretation_verified"]
    assert not evidence["coordinate_equality_enforced"]
    # An infinite extension of the boundary LINE would meet x=30; its finite
    # support starts at x=80. A shaft isolated at x=30 cannot attach to that line.
    assert _derived_station_groups([location], [stroke(20., 40.)], 1, 100., 30., 4.) == []


def test_two_observed_collinear_strokes_recover_a_drafting_gap_with_recorded_evidence():
    segments = [stroke(20., 70.), stroke(80., 380.)]
    support = _derived_extension_supports(segments, 100., 30., 80., 4.)
    assert len(support) == 1
    assert support[0]["observed_gap_px"] == 10.
    assert support[0]["method"] == "two_collinear_source_strokes_with_observed_drafting_gap"
    assert support[0]["observed_segments"] == segments
    assert support[0]["nominal_used"] is False
    groups = _derived_station_groups([level_location()], segments, 1, 100., 30., 4.)
    assert len(groups) == 1 and groups[0][1]["station_interpretation_verified"]


@pytest.mark.parametrize("segments,cross,target", [
    ([stroke(20., 45.), stroke(80., 380.)], 30., 80.),  # Too large an absolute gap.
    ([stroke(30., 45.), stroke(55., 380.)], 30., 55.),  # Gap dominates measured route.
    ([stroke(20., 70.), stroke(80., 380., cross=103.)], 30., 80.),  # Parallel, not collinear.
    ([stroke(35., 70.), stroke(80., 380.)], 20., 80.),  # Neither stroke reaches shaft.
    ([stroke(20., 70.), stroke(80., 100.)], 30., 130.),  # Neither reaches finite attachment.
])
def test_disconnected_or_misaligned_source_strokes_are_not_fabricated_into_extensions(segments, cross, target):
    assert _derived_extension_supports(segments, 100., cross, target, 4.) == []


def test_competing_drafting_gap_bridges_remain_distinct_alternatives():
    segments = [stroke(20., 70.), stroke(80., 180., cross=100.), stroke(82., 180., cross=101.)]
    result = _derived_extension_supports(segments, 100., 30., 120., 4.)
    assert len(result) == 2
    assert {value["observed_gap_px"] for value in result} == {10., 12.}


@pytest.mark.parametrize("observed,segments,verified", [
    ((150., 100.), [stroke(20., 160.)], True),
    ((150., 102.), [stroke(20., 160.)], False),  # Only fits proposal band, outside occupied ink.
    ((150., 100.), [stroke(20., 140.), stroke(150., 200.)], False),  # Gapped inference.
    ((163., 100.), [stroke(20., 160.)], False),  # Outside finite observed stroke, inside coarse band.
])
def test_extremum_measurement_requires_direct_observed_station_not_nearby_curve(observed, segments, verified):
    groups = _derived_station_groups([extremum_location(observed=observed)], segments, 1, 100., 30., 4., section_lines=[])
    assert len(groups) == 1
    assert groups[0][1]["station_interpretation_verified"] is verified
    assert groups[0][1]["construction"]["section_interpretation_not_resolved"]


@pytest.mark.parametrize("sections,expected_count", [(None, 0), ([stroke(70., 140., cross=150.)], 1)])
def test_extremum_is_not_semantically_admitted_without_checking_competing_section(sections, expected_count):
    groups = _derived_station_groups([extremum_location()], [stroke(20., 160.)], 1, 100., 30., 4., section_lines=sections)
    assert len(groups) == 1
    assert groups[0][1]["station_interpretation_verified"] is False
    assert groups[0][1]["competing_source_section_count"] == expected_count
    assert groups[0][1]["competing_source_section_check_performed"] is (sections is not None)


def test_source_level_replaces_its_arbitrary_endpoint_group_but_preserves_other_supports():
    graph = rectangle(width=60., height=40.)
    node = graph["nodes"][0]
    endpoint_group = ((0., 50., node, stroke(20., 100.)), {
        "support_kind": "source_ink_station", "member_nodes": ["v0", "v1"], "source_supported_entities": ["g0"]})
    other_node = {"id": "other_node", "source_px": [90., 100.]}
    unrelated = ((0., 60., other_node, stroke(20., 100.)), {
        "support_kind": "source_ink_station", "member_nodes": ["other_node"], "source_supported_entities": ["other"]})
    derived = _derived_station_groups([level_location()], [stroke(20., 100.)], 1, 100., 30., 4.)
    merged = _merge_derived_station_groups([endpoint_group, unrelated], derived, graph)
    assert len(merged) == 2
    assert all(item[0][2].get("id") != "v0" for item in merged)
    assert any(item[0][2]["id"] == "other_node" for item in merged)
    level_group = next(item for item in merged if item[0][2].get("anchor"))
    assert level_group[1]["equivalent_source_endpoint_nodes"] == ["v0", "v1"]


def gapped_dimension_source(tmp_path, *, nominal=40.):
    graph = rectangle(width=60., height=40.)
    transform = lambda points: np.asarray(points, float)*[5., -5.]+[80., 330.]
    for node in graph["nodes"]:
        node["source_px"] = transform([[node["x"], node["y"]]])[0].tolist()
    graph.update(proposal_tolerance_px=4., relations=[])
    graph["coordinate_system"] = {"units": "mm", "origin_source_px": [80., 330.]}
    image = Image.new("L", (460, 400), "white"); draw = ImageDraw.Draw(image)
    for entity in graph["entities"]:
        draw.line([tuple(value) for value in transform([entity["start"], entity["end"]])], fill="black", width=2)
    draw.line((30, 130, 30, 330), fill="black", width=1)
    for y in (130, 330):
        draw.line((20, y, 70, y), fill="black", width=2)
        sign = 1 if y == 130 else -1
        draw.polygon([(30, y), (26, y+sign*18), (34, y+sign*18)], fill="black")
    document = {"records": [{"text": str(nominal), "box": [[5, 212], [22, 212], [22, 241], [5, 241]]}]}
    path = tmp_path/"gapped-source.png"; image.save(path)
    model = {"scale": {"pixels_per_mm": 5.}, "coordinate_system": graph["coordinate_system"]}
    return path, np.asarray(image), document, graph, model


def test_raster_to_candidate_keeps_source_levels_as_symbols_and_drafting_gaps_as_evidence(tmp_path):
    _, gray, document, graph, _ = gapped_dimension_source(tmp_path)
    records = canonical_records(document)
    candidates = _length_candidates(gray, records, graph, 4., 5.)
    assert len(candidates) == 1, candidates
    candidate = candidates[0]
    assert candidate["local_reliable"] and candidate["nodes"] == []
    assert candidate["anchors"] == [
        {"type": "support_level", "entity_id": "g0", "axis": "y"},
        {"type": "support_level", "entity_id": "g2", "axis": "y"}]
    assert candidate["entities"] == ["g0", "g2"]
    assert all(value["extension"]["observed_gap_px"] == pytest.approx(10.)
               for value in candidate["evidence"]["observed_station_groups"])
    assert all(value["station_interpretation_verified"] for value in candidate["evidence"]["observed_station_groups"])
    records[0]["parsed"]["nominal"] = 999.
    mismatched = _length_candidates(gray, records, graph, 4., 5.)
    assert len(mismatched) == 1 and mismatched[0]["anchors"] == candidate["anchors"]
    assert not mismatched[0]["local_reliable"]


def test_tilted_fitted_face_does_not_turn_a_source_level_dimension_into_arbitrary_vertex_distance(tmp_path):
    _, gray, document, graph, _ = gapped_dimension_source(tmp_path)
    # The source horizontal face is unchanged. Only its approximate CAD proposal
    # is tilted, exactly the condition in which endpoint-only binding is wrong.
    graph["nodes"][1]["y"] = .3
    graph["nodes"][1]["source_px"][1] -= 1.5
    graph["entities"][0]["end"][1] = .3
    graph["entities"][1]["start"][1] = .3
    candidates = _length_candidates(gray, canonical_records(document), graph, 4., 5.)
    assert len(candidates) == 1 and candidates[0]["local_reliable"]
    assert candidates[0]["nodes"] == []
    assert [value["type"] for value in candidates[0]["anchors"]] == ["support_level", "support_level"]


def test_new_anchor_provider_packet_preserves_semantic_uncertainty_without_literal_coordinates():
    groups = _derived_station_groups([extremum_location(observed=(150., 102.))], [stroke(20., 160.)], 1, 100., 30., 4.)
    anchor = deepcopy(groups[0][1]["anchor"]); anchor["private_coordinates"] = [99., 999.]
    candidate = {"id": "c0", "record_id": "r0", "kind": "distance_y", "entities": ["arc"], "nodes": [],
                 "anchors": [anchor, {"type": "node", "node_id": "v0"}],
                 "evidence": {"observed_station_groups": [groups[0][1]]}}
    packet = bounded_inventory({"units": "mm", "records": [{"id": "r0", "text": "20"}], "candidates": [candidate], "relations": []})
    assert packet["candidates"][0]["anchors"][0] == {key: value for key, value in anchor.items() if key != "private_coordinates"}
    assert packet["candidates"][0]["evidence"]["observed_station_groups"][0]["station_interpretation_verified"] is False
    assert "private_coordinates" not in json.dumps(packet)


def test_selected_source_level_dimension_reaches_joint_solver_without_new_contour_objects(tmp_path):
    path, _, document, graph, model = gapped_dimension_source(tmp_path)
    class Selector:
        def select(self, image, topology, inventory):
            candidate = next(row for row in inventory["candidates"] if row.get("anchors"))
            self.packet = bounded_inventory(inventory)
            return {"status": "succeeded", "network_requests": 1, "http_success": True, "schema_success": True,
                    "bindings": [{"record_id": candidate["record_id"], "candidate_id": candidate["id"], "observed_text": "40"}],
                    "relations": [], "input_record_ids": [row["id"] for row in inventory["records"]],
                    "input_candidate_ids": [row["id"] for row in inventory["candidates"]],
                    "input_relation_ids": [row["id"] for row in inventory["relations"]]}
    provider = Selector()
    result = analyze_constraint_bindings(path, document, model, graph, tmp_path/"bindings", provider=provider, use_api=True)
    assert result["counts"]["api_accepted"] == 1, result
    measurement = next(value for value in result["constraints"] if value.get("record_id"))
    assert measurement["nodes"] == [] and measurement["value"] == 40.
    assert measurement["required"] and measurement["enforcement"] == "exact"
    solved = solve_parametric(graph, result["constraints"])
    assert solved["accepted"], solved
    assert solved["strict_measurement_contract"]["satisfied"]
    assert len(solved["entities"]) == len(graph["entities"])
    assert len(solved["nodes"]) == len(graph["nodes"])
