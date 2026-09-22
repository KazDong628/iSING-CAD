"""Geometry acceptance tests, with no runtime or test dependence on held-out GT."""

import builtins
import copy
import json
import math
from pathlib import Path

import ezdxf
import pytest

from contour_agent.geometry import solve_profile, template_schema, validate_entities


def defaults():
    return {field["id"]: field["default"] for field in template_schema()["parameters"]}


def test_calibration_topology_and_reverse_dimensions(tmp_path):
    result = solve_profile(defaults(), tmp_path)
    assert result["template_id"] == "solid293-v1"
    assert result["calibration_case"] == "solid-arrow-ping__img_000293"
    assert len(result["entities"]) == 21
    assert sum(e["type"] == "ARC" for e in result["entities"]) == 11
    assert result["validation"]["passed"]
    assert result["validation"]["max_gap_mm"] < 1e-6
    assert result["validation"]["max_tangent_error_deg"] < 1e-5
    assert all(dimension["passed"] for dimension in result["validation"]["dimensions"])
    assert result["validation"]["dxf_readback"]["passed"]
    assert result["validation"]["dxf_readback"]["max_roundtrip_error_mm"] < 1e-6
    doc = ezdxf.readfile(tmp_path / "drawing.dxf")
    assert len(list(doc.modelspace())) == 21
    assert doc.header["$INSUNITS"] == 4
    assert (tmp_path / "preview.svg").read_text(encoding="utf-8").count("<path") == 1
    assert json.loads((tmp_path / "model.json").read_text(encoding="utf-8"))["parameters"] == defaults()


@pytest.mark.parametrize("parameter", list(defaults()))
def test_each_dimension_changes_geometry_and_keeps_constraints(parameter, tmp_path):
    original = solve_profile(defaults(), tmp_path / "original")
    changed = defaults()
    changed[parameter] += 1
    rebuilt = solve_profile(changed, tmp_path / "changed")
    assert rebuilt["validation"]["passed"], rebuilt["validation"]["issues"]
    assert rebuilt["validation"]["dxf_readback"]["passed"]
    difference = max(math.dist(a[key], b[key]) for a, b in zip(original["entities"], rebuilt["entities"])
                     for key in ("start", "end"))
    assert difference > 1e-4, f"{parameter} did not drive the contour"
    dimension = next(d for d in rebuilt["validation"]["dimensions"] if d["id"] == parameter)
    assert dimension["actual"] == pytest.approx(changed[parameter], abs=1e-6)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -1, True, "188"])
def test_invalid_values_are_rejected_without_artifacts(bad_value, tmp_path):
    p = defaults()
    p["d_left"] = bad_value
    with pytest.raises(ValueError):
        solve_profile(p, tmp_path / "should_not_exist")
    assert not (tmp_path / "should_not_exist").exists()


def test_missing_dimensions_never_silently_use_calibration_defaults(tmp_path):
    p = defaults()
    del p["left_height"]
    with pytest.raises(ValueError, match="Missing required dimensions: left_height"):
        solve_profile(p, tmp_path)


def test_unknown_dimensions_and_radial_order_rejected(tmp_path):
    p = defaults()
    p["made_up"] = 42
    with pytest.raises(ValueError, match="Unknown dimensions"):
        solve_profile(p, tmp_path)
    p = defaults()
    p["d_shoulder"] = 180
    with pytest.raises(ValueError, match="Radial stations"):
        solve_profile(p, tmp_path)


def test_impossible_template_branch_is_actionable(tmp_path):
    p = defaults()
    p["r_upper_left"] = 900
    with pytest.raises(ValueError, match="reversed/zero straight segments"):
        solve_profile(p, tmp_path)


def test_validator_catches_broken_circle_and_join(tmp_path):
    entities = solve_profile(defaults(), tmp_path)["entities"]
    broken = copy.deepcopy(entities)
    broken[4]["center"][0] += 0.1
    report = validate_entities(broken, defaults())
    assert not report["passed"]
    assert report["max_radial_error_mm"] > 1e-3
    assert report["max_tangent_error_deg"] > 1e-5
    broken = copy.deepcopy(entities)
    broken[3]["end"][0] += 0.01
    assert not validate_entities(broken, defaults())["passed"]


def test_runtime_has_no_dataset_access(tmp_path, monkeypatch):
    original_open = builtins.open

    def no_dataset(file, *args, **kwargs):
        if isinstance(file, (str, Path)):
            assert "__dataset" not in str(file), "Runtime accessed calibration or ground truth files"
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", no_dataset)
    assert solve_profile(defaults(), tmp_path)["validation"]["passed"]


def test_assumptions_are_required_and_schema_is_not_mutable():
    schema = template_schema()
    assert len(schema["shape_priors"]["upper_join_directions_deg"]) == 3
    assert len(schema["shape_priors"]["lower_join_directions_deg"]) == 4
    assert all(assumption["required"] for assumption in schema["assumptions"])
    estimated = [p["id"] for p in schema["parameters"] if p["source_kind"] == "estimated"]
    assert estimated == ["r_transition"]
    schema["parameters"][0]["default"] = -999
    assert template_schema()["parameters"][0]["default"] == 188
