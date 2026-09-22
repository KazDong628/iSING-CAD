import math

import ezdxf
import pytest

from contour_agent.dxf_comparison import audit_dxf, compare_dxf_entities, _correspondence
from contour_agent.evaluation import _curves


def rectangle(path, *, units=4, split=False, reverse=False, shift=(0, 0), width=10):
    document = ezdxf.new(); document.units = units
    points = [(0, 0), (width, 0), (width, 5), (0, 5)]
    segments = list(zip(points, points[1:] + points[:1]))
    if reverse: segments = list(reversed(segments))
    for start, end in segments:
        start = (start[0]+shift[0], start[1]+shift[1]); end = (end[0]+shift[0], end[1]+shift[1])
        midpoint = ((start[0]+end[0])/2, (start[1]+end[1])/2)
        for a, b in ((start, midpoint), (midpoint, end)) if split else ((start, end),):
            document.modelspace().add_line(a, b, dxfattribs={"layer": "MAIN"})
    document.saveas(path)


def test_raw_objects_and_filtered_primitives_are_separate(tmp_path):
    path = tmp_path / "polyline.dxf"
    document = ezdxf.new(); document.units = 4
    model = document.modelspace()
    model.add_lwpolyline([(0,0), (5,0), (5,5), (0,5)], close=True, dxfattribs={"layer":"MAIN"})
    model.add_line((0,0), (99,0), dxfattribs={"layer":"CONSTRUCTION"})
    model.add_text("only annotation", dxfattribs={"layer":"NOTES"})
    model.add_arc((0,0), 2, 0, 90, dxfattribs={"layer":"GT_CLOSURE"})
    document.saveas(path)
    result = audit_dxf(path)
    assert result["raw_modelspace"]["count"] == 4
    assert result["raw_modelspace"]["types"] == {"LWPOLYLINE":1,"LINE":1,"TEXT":1,"ARC":1}
    profile = result["filtered_profile"]
    assert profile["count"] == 5 and profile["types"] == {"LINE":4,"ARC":1}
    assert profile["excluded"] == {"layer:CONSTRUCTION":1, "type:TEXT":1}
    assert profile["assumed_layers"] == ["GT_CLOSURE"]
    assert profile["entities"][-1]["sweep_deg"] == pytest.approx(90)
    assert len(result["raw_modelspace"]["entities"][0]["vertices_xy_bulge"]) == 4


def test_shuffled_entities_match_geometry_not_order(tmp_path):
    prediction, reference = tmp_path / "p.dxf", tmp_path / "r.dxf"
    rectangle(prediction, reverse=True, shift=(100, -30))
    rectangle(reference)
    result = compare_dxf_entities(prediction, reference)
    assert result["physical_score"]["reference_within_0_1mm"]
    assert result["entity_correspondence"]["counts"] == {"unique_full_parameter_agreement":4}
    assert any(row["prediction_id"][1:] != row["reference_id"][1:] for row in result["entity_correspondence"]["rows"])
    assert not result["engineering_verified"]
    assert result["alignment"]["transform"]["scale"] == 1


def test_split_segments_are_fragments_not_forced_parameter_matches(tmp_path):
    prediction, reference = tmp_path / "p.dxf", tmp_path / "r.dxf"
    rectangle(prediction, split=True); rectangle(reference)
    result = compare_dxf_entities(prediction, reference)
    assert result["physical_score"]["reference_within_0_1mm"]  # Same geometric contour, different primitive decomposition.
    correspondence = result["entity_correspondence"]
    assert correspondence["counts"] == {"possible_fragment_coverage":8}
    assert len(correspondence["one_to_many_fragment_candidates"]) == 4
    assert all(row["parameter_errors"] is None and row["reference_id"] is None for row in correspondence["rows"])


def test_duplicate_candidates_remain_ambiguous(tmp_path):
    prediction, reference = tmp_path / "p.dxf", tmp_path / "r.dxf"
    rectangle(prediction); rectangle(reference)
    document = ezdxf.readfile(reference)
    for entity in list(document.modelspace()): document.modelspace().add_entity(entity.copy())
    document.saveas(reference)
    result = compare_dxf_entities(prediction, reference)
    assert result["entity_correspondence"]["counts"] == {"ambiguous_multiple_full_candidates":4}
    assert not result["physical_score"]["reference_within_0_1mm"]


@pytest.mark.parametrize("prediction_units,reference_units", [(0,4),(4,0),(0,0)])
def test_unknown_units_have_no_mm_parameter_error(tmp_path, prediction_units, reference_units):
    prediction, reference = tmp_path / "p.dxf", tmp_path / "r.dxf"
    rectangle(prediction, units=prediction_units); rectangle(reference, units=reference_units)
    result = compare_dxf_entities(prediction, reference)
    assert not result["physical_units_available"]
    assert not result["physical_score"]["reference_within_0_1mm"]
    assert "registered_metrics" not in result["physical_score"]
    assert result["entity_correspondence"]["rows"] == []
    assert result["alignment"]["coordinate_unit"] == "dimensionless"
    assert result["alignment"]["separate_display_scales"]
    for name in ("prediction_info", "reference_info"):
        info = result["physical_score"].get(name)
        if info and not info["source_units"]:
            assert info["unit_assumption"] is None and info["millimetre_conversion_factor"] is None
    if not prediction_units:
        assert "endpoint_tolerance_mm" not in result["physical_score"]["candidate_validation"]
        assert result["physical_score"]["candidate_validation"]["coordinate_unit"] == "drawing_units_unspecified"
    for side, units in (("prediction",prediction_units),("reference",reference_units)):
        assert result[side]["filtered_profile"]["unit_assumption"] is None
        if not units: assert result[side]["filtered_profile"]["coordinate_unit"] == "drawing_units_unspecified"


def test_endpoint_connections_tangency_and_corner_intent_are_distinct(tmp_path):
    path = tmp_path / "joint.dxf"
    document = ezdxf.new(); document.units = 4
    document.modelspace().add_line((0,-2), (0,0))
    document.modelspace().add_arc((1,0), 1, 0, 180)
    document.modelspace().add_line((2,0), (3,0))
    document.saveas(path)
    result = audit_dxf(path)
    nodes = result["connections"]["nodes"]
    smooth = next(node for node in nodes if math.dist(node["point"], [0,0]) < 1e-8)
    corner = next(node for node in nodes if math.dist(node["point"], [2,0]) < 1e-8)
    assert smooth["joint_type"] == "ARC-LINE"
    assert smooth["deviation_from_tangent_continuity_deg"] == pytest.approx(0, abs=1e-5)
    assert corner["deviation_from_tangent_continuity_deg"] == pytest.approx(90)
    assert not corner["intended_tangency_known"]
    assert not result["connections"]["closed"]
    assert all(node["nearest_other_endpoint_distance"] is not None for node in nodes if node["degree"] == 1)


def test_declared_inch_units_converted_but_raw_parameters_retained(tmp_path):
    path = tmp_path / "inch.dxf"
    document = ezdxf.new(); document.units = 1
    document.modelspace().add_arc((1,2), 1, 0, .5)
    document.saveas(path)
    result = audit_dxf(path)
    assert result["raw_modelspace"]["entities"][0]["radius"] == 1
    profile = result["filtered_profile"]
    assert profile["entities"][0]["radius"] == pytest.approx(25.4)
    assert profile["entities"][0]["center"] == pytest.approx([25.4, 50.8])
    assert profile["arc_diagnostics"]["sweep_below_1_degree"] == 1
    assert profile["coordinate_unit"] == "mm"


def test_error_localization_preserves_full_score_and_marks_closure(tmp_path):
    prediction, reference = tmp_path / "p.dxf", tmp_path / "r.dxf"
    rectangle(prediction, width=8); rectangle(reference, width=10)
    document = ezdxf.readfile(reference)
    list(document.modelspace())[1].dxf.layer = "GT_CLOSURE"
    document.saveas(reference)
    result = compare_dxf_entities(prediction, reference)
    localization = result["error_localization"]
    assert localization["reference_directed_scope"]["reference_assumptions"]["max_error_mm"] > .1
    assert localization["reference_directed_scope"]["reference_core"]["max_error_mm"] > .1
    assert len(localization["per_reference_curve"]) == 4
    worst = max([row["max_error_mm"] for row in localization["per_reference_curve"] + localization["per_prediction_curve"]])
    assert worst == pytest.approx(result["physical_score"]["registered_metrics"]["max_error_mm"])
    assert not result["physical_score"]["reference_within_0_1mm"]


def test_complementary_semicircles_are_not_matched_by_reversed_endpoints(tmp_path):
    paths = [tmp_path / "upper.dxf", tmp_path / "lower.dxf"]
    for path, start, end in ((paths[0],0,180), (paths[1],180,360)):
        document = ezdxf.new(); document.units = 4
        document.modelspace().add_arc((0,0), 10, start, end)
        document.saveas(path)
    upper, _ = _curves(paths[0]); lower, _ = _curves(paths[1])
    result = _correspondence(upper, lower)
    assert result["counts"] == {"unmatched_geometry":1}
    assert result["rows"][0]["parameter_errors"] is None
    assert result["rows"][0]["nearest_candidates"][0]["sampled_symmetric_max_mm"] > 10


def test_chord_is_not_a_fragment_of_arc_merely_because_endpoints_touch(tmp_path):
    arc=tmp_path/"arc.dxf";line=tmp_path/"line.dxf"
    document=ezdxf.new();document.units=4
    document.modelspace().add_arc((0,0),10,0,180);document.saveas(arc)
    document=ezdxf.new();document.units=4
    document.modelspace().add_line((-10,0),(10,0));document.saveas(line)
    p,_=_curves(line);r,_=_curves(arc)
    result=_correspondence(p,r)
    assert result["counts"]=={"unmatched_geometry":1}
    assert result["rows"][0]["nearest_candidates"][0]["sampled_prediction_to_reference_max_mm"]>9
