import copy
import hashlib

import cv2
import numpy as np
import pytest

from contour_agent.profile_scope import simplify_terminal_scope, export_terminal_scope, source_annotation_identity


def fixture():
    def line(i, a, b):
        return dict(id=f"g{i}", type="LINE", start=a, end=b,
                    start_node=f"v{i}", end_node=f"v{(i+1)%6}")
    entities = [line(0, [10., 0.], [10., 8.]),
        dict(id="g1", type="ARC", start=[10., 8.], end=[8., 10.], center=[8., 8.],
             radius=2., clockwise=False, start_node="v1", end_node="v2"),
        line(2, [8., 10.], [0., 10.]), line(3, [0., 10.], [0., 2.]),
        dict(id="g4", type="ARC", start=[0., 2.], end=[2., 0.], center=[2., 2.],
             radius=2., clockwise=False, start_node="v4", end_node="v5"),
        line(5, [2., 0.], [10., 0.])]
    constraints = [dict(id="radius-kept", kind="radius", record_id="r001", entities=["g4"],
                         nodes=[], value=2., required=True, enforcement="exact", source="ocr_api_binding"),
        dict(id="height-kept", kind="distance_y", record_id="r003", entities=["g2", "g5"],
             nodes=[], value=10., required=True, enforcement="exact", source="ocr_api_binding",
             anchors=[dict(type="support_level", entity_id=e, axis="y") for e in ("g5", "g2")]),
        dict(id="corner-removed", kind="tangent", record_id=None, entities=["g0", "g1"],
             nodes=["v1"], value=None, source="source_geometry")]
    doc = dict(records=[dict(text=t) for t in ("R2", "R2", "R16", "10")])
    policy = dict(authority="explicit_user_request", mode="axis_line", entity_ids=["g0", "g1"],
                  deferred_record_ids=["r000", "r002"], axis="auto", max_endpoint_adjustment_mm=3.,
                  source_annotation_sha256=source_annotation_identity(doc))
    return entities, constraints, doc, policy


def test_declared_terminal_line_preserves_other_equal_radius_and_dimensions():
    e, c, doc, p = fixture()
    originals = copy.deepcopy((e, c, doc, p))
    projected, kept, receipt = simplify_terminal_scope(e, c, doc, p)
    assert projected[0]["type"] == "LINE"
    assert projected[0]["start"] == [10., 0.] and projected[0]["end"] == [10., 10.]
    assert next(x for x in projected if x["id"] == "g4") == e[4]
    assert {x["id"] for x in kept} == {"radius-kept", "height-kept"}
    assert receipt["exact_active_radius_count"] == receipt["active_radius_count"] == 1
    assert receipt["full_source_radius_count"] == 3 and not receipt["full_source_annotation_target_satisfied"]
    assert all(row["passed"] for row in receipt["retained_source_checks"].values())
    assert (e, c, doc, p) == originals


def test_terminal_chain_can_cross_ring_start_and_support_horizontal_axis():
    e, c, doc, p = fixture()
    for row in e:
        for key in ("start", "end", "center"):
            if key in row: row[key] = row[key][::-1]
        if row["type"] == "ARC": row["clockwise"] = not row["clockwise"]
    projected, _, receipt = simplify_terminal_scope(e[1:]+e[:1], [], doc, p)
    assert receipt["axis"] == "horizontal"
    assert projected[0]["start"][1] == projected[0]["end"][1] == 10.


@pytest.mark.parametrize("change,reason", [
    ({"authority": "model_guess"}, "explicit_user_scope"),
    ({"entity_ids": ["g0", "g4"]}, "contiguous"),
    ({"entity_ids": ["g0", "missing"]}, "entity_ids"),
    ({"max_endpoint_adjustment_mm": 1.}, "region_budget"),
    ({"deferred_record_ids": ["r003"]}, "radius_record_ids"),
])
def test_scope_cannot_be_inferred_or_silently_widened(change, reason):
    e, c, doc, p = fixture(); p.update(change)
    with pytest.raises(ValueError, match=reason): simplify_terminal_scope(e, c, doc, p)


def test_undeclared_annotation_on_removed_primitive_stops_projection():
    e, c, doc, p = fixture()
    c.append(dict(id="do-not-drop", kind="radius", record_id="r001", entities=["g1"], nodes=[], value=2.))
    with pytest.raises(ValueError, match="undeclared_source_constraint"):
        simplify_terminal_scope(e, c, doc, p)


def test_retained_dimension_regression_stops_projection_instead_of_loosening_gate():
    e, c, doc, p = fixture()
    c = c[:1]
    c.append(dict(id="terminal-distance", kind="distance", record_id="r003", entities=[],
                  nodes=["v0", "v2"], value=164.**.5, required=True,
                  source="ocr_api_binding", enforcement="exact", nominal_source="source_ocr"))
    with pytest.raises(ValueError, match="retained_source_equations_failed"):
        simplify_terminal_scope(e, c, doc, p)


def test_unsupported_equation_is_not_silently_treated_as_certified():
    e, c, doc, p = fixture()
    c.append(dict(id="unknown", kind="unsupported", entities=["g0"], nodes=[]))
    with pytest.raises(ValueError, match="unsupported_retained"):
        simplify_terminal_scope(e, c, doc, p)


def test_changed_or_reordered_source_annotations_require_a_new_scope_policy():
    e, c, doc, p = fixture()
    doc["records"] = doc["records"][1:] + doc["records"][:1]
    with pytest.raises(ValueError, match="annotation_identity"):
        simplify_terminal_scope(e, c, doc, p)


def test_real_export_readback_is_separate_and_source_identity_guarded(tmp_path):
    e, c, doc, p = fixture()
    image = tmp_path / "source.png"
    cv2.imencode(".png", np.full((60, 60, 3), 255, dtype=np.uint8))[1].tofile(str(image))
    p["source_image_sha256"] = hashlib.sha256(image.read_bytes()).hexdigest()
    baseline = dict(entities=e, coordinate_system=dict(units="mm", origin_source_px=[0., 50.]),
        scale=dict(pixels_per_mm=4.), extraction={}, polyline_px=[[40,50],[40,18],[32,10],[0,10],[0,42],[8,50]],
        complete_material_exterior=True, validation={}, source_input_identity=dict(source_image_sha256=p["source_image_sha256"]))
    original = copy.deepcopy(baseline)
    out = tmp_path / "scoped"
    model, receipt = export_terminal_scope(image, doc, baseline, {"constraints":c}, p, out)
    assert receipt["axis_exact_in_model_and_native_dxf"]
    assert all(check["passed"] for check in receipt["retained_native_checks"].values())
    assert model["completion_class"] == "user_scoped_partial_profile"
    assert receipt["new_online_api_calls"] == 0 and not receipt["new_optimization_performed"]
    assert baseline == original
    with pytest.raises(ValueError, match="fresh_output"):
        export_terminal_scope(image, doc, baseline, {"constraints":c}, p, out)
    p["source_image_sha256"] = "0"*64
    with pytest.raises(ValueError, match="source_identity"):
        export_terminal_scope(image, doc, baseline, {"constraints":c}, p, tmp_path / "wrong")
