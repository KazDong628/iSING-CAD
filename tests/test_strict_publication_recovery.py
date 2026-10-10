"""Recovery uses current synthetic CORE, never an API or reference drawing."""
from copy import deepcopy
import hashlib
import json
import shutil

import ezdxf
import pytest

from contour_agent.parametric_pipeline import CORE
from contour_agent.radius_contract import annotation_radius_contract, exact_radius_checks
from contour_agent.reconstruction_contract import reconstruction_contract
from contour_agent.relation_contract import relation_checks
from contour_agent.service import AgentService, _parameterization_needs_review
from test_parametric_pipeline import prepared, recover, seed_pending_job


def _save_model(output, model):
    for name, value in (("model.json", model), ("validation.json", model["validation"])):
        (output/name).write_text(json.dumps(value), encoding="utf8")


def _certified_core(tmp_path):
    _, model, output = prepared(tmp_path)
    constraints = [
        dict(id="r", kind="radius", entities=["g002"], nodes=[], value=5., record_id="r001"),
        dict(id="t0", kind="tangent", entities=["g001", "g002"], nodes=["v002"], value=0.),
        dict(id="t1", kind="tangent", entities=["g002", "g003"], nodes=["v003"], value=0.),
        *[dict(id=f"d{i}", kind="vertical" if i == 0 else "horizontal", entities=[f"g{i:03d}"], nodes=[])
          for i in (0, 1, 3)],
    ]
    coverage = dict(all_radius_records_resolved=True, all_confirmed_arrows_bound=True,
                    bound_mappings=[dict(record_id="r001", entity_id="g002", nominal=5.)])
    stage = dict(accepted=True, constraint_subset_accepted=True, status="completed",
                 constraints=constraints, solver_constraint_checks=[{**row, "passed": True} for row in constraints],
                 binding_counts=dict(recognized_dimensions=3, bound_source_records=3, unbound_dimensions=0),
                 radius_binding_coverage=coverage, underconstrained=False,
                 solver=dict(accepted=True, validation=dict(constraint_subset_satisfied=True),
                             diagnostics=dict(remaining_shape_dof=0)))
    native = ezdxf.readfile(output/"drawing.dxf")
    radius = annotation_radius_contract(stage, dict(accepted=True, entities=model["entities"]))
    radius.update(current_dxf_verified=True, publication_status="committed")
    stage["annotation_radius_contract"] = deepcopy(radius)
    model["validation"].update(
        exact_radius_validation=exact_radius_checks(model["entities"], constraints, dxf_document=native),
        strict_relation_validation=relation_checks(model["entities"], constraints, dxf_document=native),
        annotation_radius_contract=radius)
    model["validation"]["reconstruction_contract"] = reconstruction_contract(model["entities"], stage, model["validation"])
    assert model["validation"]["reconstruction_contract"]["satisfied"]
    model["parameterization"] = stage
    _save_model(output, model)
    return model, output


def _recover(tmp_path, monkeypatch, model, output):
    seed_pending_job(tmp_path, model, output)
    service = recover(tmp_path, monkeypatch)
    try:
        return service.store.get("persisted")
    finally:
        service.close()


@pytest.mark.parametrize("missing", ["strict_relation_validation", "reconstruction_contract", "both"])
def test_legacy_accepted_export_stays_downloadable_but_cannot_be_recognized_as_complete(tmp_path, monkeypatch, missing):
    model, output = _certified_core(tmp_path)
    for field in ("strict_relation_validation", "reconstruction_contract"):
        if missing in (field, "both"):
            model["validation"].pop(field)
    _save_model(output, model)
    before = {name: (output/name).read_bytes() for name in CORE}
    job = _recover(tmp_path, monkeypatch, model, output)
    assert job["status"] == "needs_review"
    assert not job["parameterization"]["accepted"]
    assert job["artifacts"]["dxf"] and job["validation"]["passed"]
    assert job["automatic_completion"]  # The artifact is still a valid closed draft.
    assert before == {name: (output/name).read_bytes() for name in CORE}
    assert not job["parameterization"]["recovery_certificate"]["legacy_export_automatically_certified"]


def test_complete_current_native_certificate_recovers_without_changing_core(tmp_path, monkeypatch):
    model, output = _certified_core(tmp_path)
    before = {name: (output/name).read_bytes() for name in CORE}
    job = _recover(tmp_path, monkeypatch, model, output)
    assert job["status"] == "completed" and job["parameterization"]["accepted"]
    assert job["parameterization"]["recovery_certificate"]["strict_relation_validation"]["satisfied_count"] == 2
    assert not _parameterization_needs_review(job)
    assert before == {name: (output/name).read_bytes() for name in CORE}


@pytest.mark.parametrize("defect", ["unknown_joint", "remaining_shape_dof"])
def test_cached_complete_flag_cannot_hide_unknown_join_or_remaining_dof(tmp_path, monkeypatch, defect):
    model, output = _certified_core(tmp_path)
    if defect == "unknown_joint":
        model["parameterization"]["constraints"] = [c for c in model["parameterization"]["constraints"] if c["id"] != "t1"]
        model["validation"]["strict_relation_validation"] = relation_checks(
            model["entities"], model["parameterization"]["constraints"],
            dxf_document=ezdxf.readfile(output/"drawing.dxf"))
    else:
        model["parameterization"]["underconstrained"] = True
        model["parameterization"]["solver"]["diagnostics"]["remaining_shape_dof"] = 4
    _save_model(output, model)
    job = _recover(tmp_path, monkeypatch, model, output)
    assert job["status"] == "needs_review" and not job["parameterization"]["accepted"]
    audit = job["parameterization"]["recovery_certificate"]
    assert audit["constraint_subset_verified"]
    assert not audit["reconstruction_contract"]["satisfied"]
    assert audit["reconstruction_contract"]["unresolved_joint_count"] == (1 if defect == "unknown_joint" else 0)
    assert audit["reconstruction_contract"]["remaining_shape_dof"] == (4 if defect == "remaining_shape_dof" else 0)


def test_native_geometry_is_recomputed_not_trusted_from_old_strict_flags(tmp_path, monkeypatch):
    model, output = _certified_core(tmp_path)
    # Coherent model + DXF with a very slightly sloping top line. Cached strict
    # success describes an older shape; ordinary readback still matches exactly.
    model["entities"][0]["end"][1] += 1e-5
    model["entities"][1]["start"][1] += 1e-5
    native = ezdxf.readfile(output/"drawing.dxf")
    primitives = list(native.modelspace())
    primitives[0].dxf.end = model["entities"][0]["end"]
    primitives[1].dxf.start = model["entities"][1]["start"]
    native.saveas(output/"drawing.dxf")
    _save_model(output, model)
    job = _recover(tmp_path, monkeypatch, model, output)
    audit = job["parameterization"]["recovery_certificate"]
    assert job["status"] == "needs_review" and not job["parameterization"]["accepted"]
    assert not audit["constraint_subset_verified"]
    assert not audit["strict_relation_validation"]["passed"]
    assert audit["strict_relation_validation"]["checks"][0]["dxf"]["angle_residual_deg"] > 1e-7


def test_newer_success_sidecar_cannot_certify_old_current_core(tmp_path, monkeypatch):
    model, output = _certified_core(tmp_path)
    newer = deepcopy(model["parameterization"])
    newer["provider"] = dict(status="succeeded", network_requests=1, http_success=True, schema_success=True)
    (output/"parametric-stage.json").write_text(json.dumps(newer), encoding="utf8")
    (output/"reconstruction-contract.json").write_text(json.dumps(model["validation"]["reconstruction_contract"]), encoding="utf8")
    model["validation"].pop("strict_relation_validation")
    model["validation"].pop("reconstruction_contract")
    _save_model(output, model)
    job = _recover(tmp_path, monkeypatch, model, output)
    assert job["status"] == "needs_review" and not job["parameterization"]["accepted"]
    assert job["parameterization"]["provider"]["http_success"] is True
    assert not job["parameterization"]["recovery_certificate"]["complete"]


@pytest.mark.parametrize("legacy", [False, True])
def test_rollback_checks_last_valid_core_and_keeps_rollback_manifest(tmp_path, legacy):
    model, output = _certified_core(tmp_path)
    if legacy:
        model["validation"].pop("strict_relation_validation")
        model["validation"].pop("reconstruction_contract")
        _save_model(output, model)
    hashes = {name: hashlib.sha256((output/name).read_bytes()).hexdigest() for name in CORE}
    for name in CORE:
        shutil.copyfile(output/name, output/("last-valid-"+name))
    (output/"preview.svg").write_text("interrupted replacement", encoding="utf8")
    job = dict(parameterization=dict(publication=dict(status="pending", candidate_sha256={name: "wrong" for name in CORE},
                   rollback_prefix="last-valid-", rollback_sha256=hashes)), issues=[])
    assert AgentService._recover_automatic_geometry(job, output)
    assert job["parameterization"]["accepted"] is (not legacy)
    assert job["parameterization"]["publication"]["status"] == "rolled_back"
    assert {name: hashlib.sha256((output/name).read_bytes()).hexdigest() for name in CORE} == hashes
    if legacy:
        assert job["status"] == "needs_review"


def test_review_helper_does_not_accept_an_old_radius_only_success():
    assert _parameterization_needs_review(dict(parameterization=dict(accepted=True), validation=dict(passed=True)))
