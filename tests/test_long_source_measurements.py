"""Remote dimensions and common centers are source observations, not fit locks."""
from copy import deepcopy

import cv2
import numpy as np
import pytest

from contour_agent.dimension_evidence import _linear_witnesses, binding_linear_witnesses
from contour_agent.measurement_anchors import (
    source_curve_center_locations, validate_derived_anchor, derived_anchor_equalities,
    derived_anchor_point,
)
from contour_agent.parametric_solver import _strict_measurement_contract
from contour_agent.source_measurement_tracks import radial_shaft_track
from test_source_curve_center_measurements import source_fixture


def long_dimension(missing_arrow=False):
    image = np.full((450, 1400), 255, np.uint8)
    cv2.line(image, (80, 340), (1300, 340), 0, 1)
    cv2.line(image, (100, 190), (100, 400), 0, 1)
    # The actual station is a finite short extension at a remote center.
    cv2.line(image, (1180, 315), (1180, 365), 0, 1)
    cv2.line(image, (1280, 150), (1280, 400), 0, 1)
    cv2.fillConvexPoly(image, np.array([[100, 340], [125, 334], [125, 346]]), 0)
    if not missing_arrow:
        cv2.fillConvexPoly(image, np.array([[1180, 340], [1155, 334], [1155, 346]]), 0)
    records = [{"id": "r0", "text": "270", "parsed": {"kind": "length", "nominal": 270.},
                "box": [[570, 290], [650, 290], [650, 335], [570, 335]]}]
    return image, records


def test_remote_short_extension_requires_both_arrows_not_label_height_or_nominal():
    image, records = long_dimension()
    old = _linear_witnesses(image, records, split_chains=True)
    assert old[0]["line"]["hi"] == 1280.
    new = binding_linear_witnesses(image, records)
    assert len(new) == 1
    assert (new[0]["line"]["lo"], new[0]["line"]["hi"]) == (100., 1180.)
    assert new[0]["binding_evidence"]["verified"]
    assert not new[0]["binding_evidence"]["scale_estimation_eligible"]
    records[0]["parsed"]["nominal"] = 999.
    assert binding_linear_witnesses(image, records)[0]["line"] == new[0]["line"]
    image, records = long_dimension(missing_arrow=True)
    assert not any(row.get("binding_evidence", {}).get("verified")
                   for row in binding_linear_witnesses(image, records))


def test_two_unique_radius_shafts_declare_symbolic_common_center_with_both_equalities():
    gray, graph, observations, center = source_fixture()
    graph["entities"].append({**deepcopy(graph["entities"][0]), "id": "g1", "center": [6., 1.],
        "start": [16., 1.], "end": [6., 11.]})
    observations[1]["target_candidates"] = [{"entity_id": "g1", "entity_type": "ARC"}]
    rows = source_curve_center_locations(gray, graph, observations, "y")
    assert len(rows) == 1
    anchor = rows[0]["anchor"]
    assert anchor == {"type": "common_curve_center", "entity_id": "g0", "other_entity_id": "g1", "axis": "y"}
    entities = {entity["id"]: entity for entity in graph["entities"]}
    assert validate_derived_anchor(anchor, entities, "y")[1] == {"g0", "g1"}
    np.testing.assert_allclose(rows[0]["source_px"], center)
    np.testing.assert_array_equal(derived_anchor_equalities(anchor, entities), [-1., -1.])
    with pytest.raises(ValueError, match="do not coincide"):
        derived_anchor_point(anchor, entities)
    entities["g1"]["center"] = [5., 0.]
    np.testing.assert_array_equal(derived_anchor_point(anchor, entities), [5., 0.])
    # Native certification checks both center coordinates even if the measured
    # ordinate remains exact. An x-only concentricity failure is not hidden.
    row = {"id": "m", "record_id": "r", "kind": "distance_y", "nodes": [],
        "entities": ["g0", "g1"], "value": 5., "source": "ocr_local_binding",
        "anchors": [anchor, {"type": "node", "node_id": "n"}]}
    assert _strict_measurement_contract({"n": [0., 5.]}, list(entities.values()), [row])["satisfied"]
    entities["g1"]["center"][0] += .01
    assert not _strict_measurement_contract({"n": [0., 5.]}, list(entities.values()), [row])["satisfied"]


def test_long_shaft_remeasurement_corrects_short_direction_without_reading_nominals():
    image = np.full((650, 650), 255, np.uint8)
    tip = np.array([100., 80.]); tail = np.array([450., 530.])
    cv2.line(image, tuple(tip.astype(int)), tuple(tail.astype(int)), 0, 2)
    actual = (tip-tail)/np.linalg.norm(tip-tail)
    angle = np.radians(1.)
    noisy = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])@actual
    observation = {"record_id": "r0", "nominal": 180., "arrowhead": {
        "tip_px": tip.tolist(), "direction_px": noisy.tolist()}}
    measured = radial_shaft_track(image, observation)
    assert measured and measured["evidence"]["verified"]
    assert measured["evidence"]["short_direction_change_degrees"] == pytest.approx(1., abs=.1)
    observation["nominal"] = 1.
    assert radial_shaft_track(image, observation) == measured
    cv2.rectangle(image, (220, 230), (330, 370), 255, -1)
    truncated = radial_shaft_track(image, observation)
    # The prefix may still measure a direction. It cannot include the remote
    # post-gap segment, which the separate center proof must not extrapolate.
    assert truncated is None or truncated["evidence"]["observed_span_px"] < 180.


def test_common_center_schema_rejects_coordinate_injection_and_unknown_second_arc():
    _, graph, _, _ = source_fixture()
    entities = {entity["id"]: entity for entity in graph["entities"]}
    anchor = {"type": "common_curve_center", "entity_id": "g0", "other_entity_id": "missing", "axis": "y"}
    with pytest.raises(ValueError, match="unknown"):
        validate_derived_anchor(anchor, entities, "y")
    with pytest.raises(ValueError, match="literal coordinates"):
        validate_derived_anchor({**anchor, "x": 100.}, entities, "y")


def test_only_owned_ocr_occlusion_can_interrupt_an_observed_radial_tail():
    gray, graph, observations, _ = source_fixture()
    cv2.rectangle(gray, (168, 238), (192, 276), 255, -1)
    records = [{"id": observations[0]["record_id"], "box": [[168, 238], [192, 238], [192, 276], [168, 276]]}]
    assert source_curve_center_locations(gray, graph, observations, "y") == []
    restored = source_curve_center_locations(gray, graph, observations, "y", records=records)
    assert restored
    support = restored[0]["evidence"]["tail_support"][0]
    assert support["visible_tail_fraction"] < 1.
    assert support["own_ocr_occlusion_is_not_ink_support"]
    assert support["long_shaft_remeasurement"]["nominal_used"] is False
    records[0]["id"] = "some_other_record"
    assert source_curve_center_locations(gray, graph, observations, "y", records=records) == []


def test_provider_retains_both_common_center_references_without_extra_coordinates():
    from contour_agent.binding_provider import bounded_inventory
    anchor = {"type": "common_curve_center", "entity_id": "g0", "other_entity_id": "g1", "axis": "y"}
    inventory = {"units": "mm", "records": [{"id": "r0", "text": "10", "parsed": {"kind": "length", "nominal": 10.}}],
        "candidates": [{"id": "c0", "record_id": "r0", "kind": "distance_y", "value": 10., "entities": ["g0", "g1"],
            "nodes": [], "anchors": [{**anchor, "literal_pixel": [240., 360.]}, {"type": "node", "node_id": "v0"}],
            "evidence": {}, "local_reliable": True}], "relations": []}
    packet = bounded_inventory(inventory)
    assert packet["candidates"][0]["anchors"][0] == anchor


def test_joint_solver_enforces_common_center_as_well_as_dimension_and_native_readback(tmp_path):
    import ezdxf
    from test_measurement_anchor_binding import native_document
    from contour_agent.parametric_solver import solve_parametric, native_measurement_checks
    points = np.array([[10.02, 0.], [0., 10.], [-10., 0.], [0., -10.]])
    nodes = [{"id": "v"+str(i), "x": float(p[0]), "y": float(p[1]), "source_px": p.tolist()}
             for i, p in enumerate(points)]
    entities = []
    for i, a in enumerate(points):
        b = points[(i+1) % 4]; chord = b-a
        normal = np.array([-chord[1], chord[0]])/np.linalg.norm(chord)
        center = (a+b)/2+np.sqrt(100.-np.dot(chord, chord)/4)*normal
        entities.append({"id": "g"+str(i), "type": "ARC", "start": a.tolist(), "end": b.tolist(),
            "start_node": "v"+str(i), "end_node": "v"+str((i+1) % 4),
            "center": center.tolist(), "radius": 10., "clockwise": False})
    graph = {"units": "mm", "nodes": nodes, "entities": entities}
    constraints = [{"id": "r"+str(i), "record_id": "ocr"+str(i), "source": "ocr_local_binding",
        "kind": "radius", "entities": ["g"+str(i)], "nodes": [], "value": 10.} for i in range(4)]
    constraints.append({"id": "m", "record_id": "ocr10", "source": "ocr_local_binding",
        "kind": "distance_x", "entities": ["g0", "g2"], "nodes": [], "value": 10.,
        "anchors": [{"type": "common_curve_center", "entity_id": "g0", "other_entity_id": "g2", "axis": "x"},
                    {"type": "node", "node_id": "v0"}]})
    solved = solve_parametric(graph, constraints)
    assert solved["accepted"], solved.get("diagnostics")
    indexed = {entity["id"]: entity for entity in solved["entities"]}
    np.testing.assert_allclose(indexed["g0"]["center"], indexed["g2"]["center"], atol=1e-7)
    assert solved["strict_measurement_contract"]["satisfied"]
    assert solved["strict_radius_contract"]["satisfied"]
    path = tmp_path/"circle.dxf"
    native_document({"entities": solved["entities"]}).saveas(path)
    exported = ezdxf.readfile(path)
    assert native_measurement_checks(solved["entities"], [constraints[-1]], dxf_document=exported)["passed"]
    arc = list(exported.modelspace())[2]
    old = arc.dxf.center
    arc.dxf.center = (old.x, old.y+.001, old.z)
    assert not native_measurement_checks(solved["entities"], [constraints[-1]], dxf_document=exported)["passed"]
    # Moving only the second center must invalidate the output certificate.
    changed = deepcopy(solved["entities"])
    changed[2]["center"][1] += .001
    decoded = {node["id"]: [node["x"], node["y"]] for node in solved["nodes"]}
    assert not _strict_measurement_contract(decoded, changed, [constraints[-1]])["satisfied"]
