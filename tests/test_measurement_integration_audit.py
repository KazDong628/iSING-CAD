"""Independent integration audit: native DXF, symbolic sections and job scope.

All geometry is synthetic analytic input, never evaluation reference data.
"""
import copy
import math

import ezdxf
import pytest

from contour_agent.config import Settings
from contour_agent.dimension_policy import source_dimension_interval
from contour_agent.execution_control import ExecutionInterrupted, execution_checkpoint
from contour_agent.parametric_solver import native_measurement_checks, solve_parametric
from contour_agent.service import AgentService


def _sector(*, outer=180., inner=160., inner_start=90., inner_end=0.):
    def point(radius, angle):
        return [radius*math.cos(math.radians(angle)), radius*math.sin(math.radians(angle))]
    points = [point(outer, 0.), point(outer, 90.), point(inner, inner_start), point(inner, inner_end)]
    entities = []
    for i in range(4):
        entity = dict(id=f"g{i}", type="ARC" if i % 2 == 0 else "LINE",
            start_node=f"v{i}", end_node=f"v{(i+1)%4}", start=points[i], end=points[(i+1)%4])
        if i % 2 == 0:
            entity.update(center=[0., 0.], radius=outer if i == 0 else inner, clockwise=i == 2)
        entities.append(entity)
    return dict(units="mm", entities=entities,
        nodes=[dict(id=f"v{i}", x=p[0], y=p[1], source_px=p) for i, p in enumerate(points)])


def _interval_gap():
    _, metadata = source_dimension_interval(dict(kind="length", nominal=20., lower_deviation=0., upper_deviation=3.))
    return dict(id="thickness", record_id="rthickness", kind="radial_gap", entities=["g0", "g2"], nodes=[],
        value=20., source="ocr_api_binding", required=True, nominal_source="source_ocr",
        enforcement="source_tolerance_interval", source_tolerance=metadata)


def _native_document(tmp_path, entities):
    document = ezdxf.new()
    msp = document.modelspace()
    for entity in entities:
        if entity["type"] == "LINE":
            msp.add_line(entity["start"], entity["end"])
        else:
            center = entity["center"]
            angles = [math.degrees(math.atan2(entity[k][1]-center[1], entity[k][0]-center[0])) % 360.
                      for k in ("start", "end")]
            if entity["clockwise"]:
                angles.reverse()
            msp.add_arc(center, entity["radius"], *angles)
    path = tmp_path/"synthetic.dxf"
    document.saveas(path)
    return ezdxf.readfile(path)


def test_native_radial_interval_accepts_neither_nominal_nor_midpoint_value(tmp_path):
    graph = _sector(inner=159.)  # Gap 21 is inside [20, 23], distinct from midpoint 21.5.
    row = _interval_gap()
    original = copy.deepcopy((graph, row))
    audit = native_measurement_checks(graph["entities"], [row],
        dxf_document=_native_document(tmp_path, graph["entities"]))
    assert audit["passed"] and audit["dxf_readback_performed"] and audit["native_mapping_verified"]
    check = audit["checks"][0]
    assert check["model"]["actual"] == check["dxf"]["actual"] == 21.
    assert check["dxf"]["source_tolerance_interval"]["passed"]
    assert (graph, row) == original


@pytest.mark.parametrize("change", ["radius", "center", "missing_entity"])
def test_actual_native_dxf_tampering_cannot_be_hidden_by_model_accepted_flag(tmp_path, change):
    graph = _sector()
    document = _native_document(tmp_path, graph["entities"])
    arc = list(document.modelspace().query("ARC"))[0]
    if change == "radius":
        arc.dxf.radius += .01
    elif change == "center":
        arc.dxf.center = (.01, 0., 0.)
    else:
        document.modelspace().delete_entity(arc)
    row = {**_interval_gap(), "accepted": True, "passed": True}
    audit = native_measurement_checks(graph["entities"], [row], dxf_document=document)
    assert not audit["passed"]
    assert audit["dxf_readback_performed"]
    assert audit["issues"]


def test_identical_radius_difference_needs_one_common_finite_angular_section(tmp_path):
    # The two source ARCs have a common center and R180-R160=20, but outer is
    # 0..90 degrees while inner is 120..150. A full-circle shortcut is invalid.
    graph = _sector(inner_start=150., inner_end=120.)
    audit = native_measurement_checks(graph["entities"], [_interval_gap()],
        dxf_document=_native_document(tmp_path, graph["entities"]))
    assert not audit["passed"]
    assert audit["checks"][0]["model"]["actual"] == 20.
    assert audit["checks"][0]["model"]["passed"] is False


def test_valid_offset_circle_geometry_cannot_certify_a_radial_thickness(tmp_path):
    graph = _sector()
    inner = graph["entities"][2]
    inner["center"][0] += .5
    for key in ("start", "end"):
        inner[key][0] += .5
    graph["entities"][1]["end"] = list(inner["start"])
    graph["entities"][3]["start"] = list(inner["end"])
    for index, key in ((2, "start"), (3, "end")):
        graph["nodes"][index].update(x=inner[key][0], y=inner[key][1])
    audit = native_measurement_checks(graph["entities"], [_interval_gap()],
        dxf_document=_native_document(tmp_path, graph["entities"]))
    assert audit["native_mapping_verified"]  # A valid exported curve, not malformed radius/endpoints.
    assert not audit["passed"]
    check = audit["checks"][0]["dxf"]
    assert check["actual"] == 20.
    assert check["intersection_conditions"][0]["common_center_residual_mm"] == [-.5, 0.]


def test_interval_cannot_reintroduce_the_old_midpoint_as_an_equality():
    graph = _sector()
    row = _interval_gap()
    row["value"] = 21.5
    with pytest.raises(ValueError, match="keeps its nominal"):
        solve_parametric(graph, [row])


def _loosened_interval(row):
    loose = copy.deepcopy(row)
    _, loose["source_tolerance"] = source_dimension_interval(
        dict(kind="length", nominal=20., lower_deviation=-20., upper_deviation=979.))
    return loose


def test_passed_solver_receipt_cannot_claim_it_preserved_a_different_source_interval():
    from contour_agent.parametric_pipeline import _constraint_receipts_complete
    graph = _sector()
    required = _interval_gap()
    loose = {**_loosened_interval(required), "passed": True}
    assert not _constraint_receipts_complete({"constraints": [required]}, {"constraints": [loose]}, graph)


def test_structural_edit_certificate_cannot_silently_loosen_a_known_source_interval():
    from contour_agent.source_topology_initial_guess import required_constraints_preserved
    graph = _sector()
    required = _interval_gap()
    audit = required_constraints_preserved(graph, copy.deepcopy(graph), [required], [_loosened_interval(required)])
    assert not audit["passed"]
    assert audit["failures"]


def _section_constraint(*, branch="positive", offset=100.):
    def anchor(entity):
        return dict(type="section_curve_intersection", entity_id=entity, axis="y", branch=branch,
            station=dict(type="source_offset_from_line_intersection", entities=["g1", "g3"],
                axis="x", direction=1, source_record_id="rstation", source_offset=offset, reference=True))
    return dict(id="section", record_id="rsection", kind="distance_y", entities=["g0", "g1", "g2", "g3"],
        nodes=[], anchors=[anchor("g2"), anchor("g0")], value=math.sqrt(180.**2-100.**2)-math.sqrt(160.**2-100.**2),
        source="ocr_api_binding", required=True, enforcement="exact", nominal_source="source_ocr")


def test_symbolic_section_and_exact_radii_reach_joint_solver_and_real_dxf_readback(tmp_path):
    graph = _sector()
    section = _section_constraint()
    radii = [dict(id=f"R{i}", record_id=f"rR{i}", kind="radius", entities=[f"g{i}"], nodes=[],
                  value=value, source="ocr_api_binding") for i, value in ((0, 180.), (2, 160.))]
    solution = solve_parametric(graph, [*radii, section])
    assert solution["accepted"], solution.get("validation")
    assert solution["constraints"][-1]["absolute_residual"] <= 1e-7
    assert [r["actual"] for r in solution["constraints"][:2]] == [180., 160.]
    audit = native_measurement_checks(solution["entities"], [section],
        dxf_document=_native_document(tmp_path, solution["entities"]))
    assert audit["passed"] and audit["dxf_readback_performed"]
    assert len(audit["checks"][0]["dxf"]["derived_anchor_points"]) == 2


@pytest.mark.parametrize("defect", ["opposite_circle_branch", "changed_source_station", "literal_coordinate"])
def test_section_native_certificate_never_uses_full_circle_or_stale_anchor(defect):
    graph = _sector()
    section = _section_constraint()
    if defect == "opposite_circle_branch":
        for anchor in section["anchors"]:
            anchor["branch"] = "negative"
    elif defect == "changed_source_station":
        for anchor in section["anchors"]:
            anchor["station"]["source_offset"] = 110.
    else:
        section["anchors"][0]["x"] = 100.
    assert not native_measurement_checks(graph["entities"], [section])["passed"]


@pytest.mark.parametrize("stop", ["cancel", "superseded_revision"])
def test_service_worker_installs_and_resets_job_cancellation_scope_preserving_frozen_core(tmp_path, monkeypatch, stop):
    service = AgentService(Settings(runtime_root=tmp_path, api_key=""))
    job = dict(id="a"*32, status="solving", solve_revision=0, events=[],
               updated_at="2026-10-13T00:00:00+00:00")
    service.store.save(job)
    core = tmp_path/"drawing.dxf"
    core.write_bytes(b"previous-valid-core")
    observed = []
    def work(job_id):
        if stop == "cancel":
            service.cancel(job_id)
        else:
            updated = service.store.get(job_id)
            updated["solve_revision"] = 1
            service.store.save(updated)
        try:
            execution_checkpoint("synthetic.next_cpu_operation")
        except ExecutionInterrupted as error:
            observed.append(error.reason)
            return
        core.write_bytes(b"forbidden-late-publication")
    monkeypatch.setattr(service, "_automatic_work", work)
    try:
        service._automatic(job["id"])
        assert observed == ["cancelled"]
        assert core.read_bytes() == b"previous-valid-core"
        assert service.store.get(job["id"])["status"] == ("cancelled" if stop == "cancel" else "solving")
        execution_checkpoint("outside_worker_scope")  # Context must not leak to the next task.
    finally:
        service.close()
