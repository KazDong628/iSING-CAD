"""Publication invariants use synthetic geometry, never a provider or GT."""
from copy import deepcopy
import hashlib
import json

import ezdxf
import pytest

from contour_agent.parametric_pipeline import CORE, refine_parametric, _os_error_diagnostic
from test_parametric_pipeline import prepared, supported_graph, seed_pending_job, recover
from test_radius_contract import equation


def setup_radius_pipeline(tmp_path, monkeypatch, *, nominal=5., source_passed=True,
                          complete=True, solver_checks=True, progress=None):
    image, baseline, output = prepared(tmp_path)
    graph = supported_graph(image, baseline)
    monkeypatch.setattr("contour_agent.topology.build_topology", lambda *a, **k: graph)
    monkeypatch.setattr("contour_agent.parametric_pipeline._plan_topology", lambda *a, **k: (graph, {
        "selected_candidate_id": "source", "selection_source": "local_evaluator", "local_evaluation": {}}))
    monkeypatch.setattr("contour_agent.parametric_pipeline._solved_source_validation", lambda *a, **k: {
        "passed": source_passed, "reasons": [] if source_passed else ["solved_source_stroke_support_degraded"]})
    coverage = {"all_radius_records_resolved": complete, "all_confirmed_arrows_bound": True,
                "unknown_arrow_records": [] if complete else ["r002"],
                "bound_mappings": [{"record_id": "r001", "entity_id": "g002", "nominal": nominal}]}
    # Integration bindings follow the complete solver protocol. The minimal
    # equation() helper intentionally only covers standalone radius audits.
    constraints = [{**equation(nominal), "id": "radius_r001", "source": "ocr_local_binding", "nodes": []}]
    monkeypatch.setattr("contour_agent.constraint_binding.analyze_constraint_bindings", lambda *a, **k: {
        "constraints": constraints, "radius_binding_coverage": coverage,
        "provider": {"status": "disabled", "network_requests": 0}})
    solved = deepcopy(graph["entities"])
    ratio = nominal/5.
    for entity in solved:
        for key in ("start", "end", "center"):
            if key in entity:
                entity[key] = [v*ratio for v in entity[key]]
        if entity["type"] == "ARC":
            entity["radius"] = nominal
    monkeypatch.setattr("contour_agent.parametric_solver.solve_parametric", lambda *a, **k: {
        "accepted": True, "status": "accepted", "entities": solved,
        "constraints": constraints if solver_checks else [], "validation": {"passed": True}})
    model, stage = refine_parametric(image, {}, baseline, output, progress=progress)
    return model, stage, output


def test_rejected_exact_candidate_cannot_certify_retained_dxf(tmp_path, monkeypatch):
    model, stage, output = setup_radius_pipeline(tmp_path, monkeypatch, nominal=6., source_passed=False)
    actual = list(ezdxf.readfile(output/"drawing.dxf").modelspace().query("ARC"))[0].dxf.radius
    assert actual == 5.  # The valid earlier draft, not the rejected exact-R6 candidate.
    assert not stage["accepted"] and model["validation"]["passed"]
    contract = json.loads((output/"radius-contract.json").read_text(encoding="utf8"))
    assert contract["satisfied"] is False
    assert contract["all_annotated_radii_verified"] is False


def test_solver_missing_check_cannot_replace_required_dxf_audit_with_empty_success(tmp_path, monkeypatch):
    _, stage, output = setup_radius_pipeline(tmp_path, monkeypatch, solver_checks=False)
    contract = json.loads((output/"radius-contract.json").read_text(encoding="utf8"))
    if stage["constraint_subset_accepted"]:
        # Export may independently recover the complete admitted obligations.
        checks = contract["exact_radius_validation"]
        assert checks["dxf_readback_performed"] is True
        assert checks["required_count"] == 1
        assert checks["checks"][0]["dxf_radius"] == checks["checks"][0]["nominal"] == 5.
    else:
        assert contract["satisfied"] is False


@pytest.mark.parametrize("complete", [True,False])
def test_restart_preserves_unresolved_radius_review_status(tmp_path, monkeypatch, complete):
    model, stage, output = setup_radius_pipeline(tmp_path, monkeypatch, complete=complete)
    assert stage["constraint_subset_accepted"] and not stage["accepted"]
    seed_pending_job(tmp_path, model, output)
    service = recover(tmp_path, monkeypatch)
    try:
        job = service.store.get("persisted")
        assert job["validation"]["passed"] and job["automatic_completion"]
        assert not job["parameterization"]["accepted"]
        assert job["parameterization"]["status"] == ("completed_with_unresolved_attributes" if complete else
                                                       "completed_with_unresolved_radii")
        audit = job["parameterization"]["recovery_certificate"]
        assert audit["numerical_constraint_validation"]["passed"]
        assert audit["radius_subset_verified"]
        assert audit["annotated_radii_verified"] is complete
        assert job["status"] == "needs_review"
    finally:
        service.executor.shutdown(wait=True)


@pytest.mark.parametrize("exception_type", [RuntimeError, InterruptedError])
def test_publication_tail_failure_provenance_matches_actual_rolled_back_core(tmp_path, monkeypatch, exception_type):
    """A failed progress callback must not leave the withdrawn R6 DXF certified."""
    output = tmp_path/"runtime/jobs/persisted/automatic-001"
    candidate = {}
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()

    def fail_after_core_commit(name, message):
        if name == "parametric_export":
            # Exercise the actual disk transition before failure, rather than
            # asserting only a mocked stage flag. The candidate really is R6.
            candidate["hashes"] = {name: digest(output/name) for name in CORE}
            candidate["radius"] = list(ezdxf.readfile(output/"drawing.dxf").modelspace().query("ARC"))[0].dxf.radius
            raise exception_type("synthetic_publication_tail_failure")

    if exception_type is InterruptedError:
        with pytest.raises(InterruptedError, match="synthetic_publication_tail_failure"):
            setup_radius_pipeline(tmp_path, monkeypatch, nominal=6., progress=fail_after_core_commit)
    else:
        _, stage, _ = setup_radius_pipeline(tmp_path, monkeypatch, nominal=6., progress=fail_after_core_commit)
        assert stage["status"] == "failed" and not stage["accepted"]

    assert candidate["radius"] == 6.
    actual = {name: digest(output/name) for name in CORE}
    restored = {name: digest(output/("last-valid-"+name)) for name in CORE}
    assert actual == restored
    assert actual["drawing.dxf"] != candidate["hashes"]["drawing.dxf"]
    assert list(ezdxf.readfile(output/"drawing.dxf").modelspace().query("ARC"))[0].dxf.radius == 5.
    saved = json.loads((output/"model.json").read_text(encoding="utf8"))
    validation = json.loads((output/"validation.json").read_text(encoding="utf8"))
    assert saved["validation"] == validation and validation["passed"]
    assert saved["parameterization"]["topology_exported"] and not saved["parameterization"]["accepted"]
    contract = json.loads((output/"radius-contract.json").read_text(encoding="utf8"))
    assert not contract["satisfied"] and not contract["current_dxf_verified"]

    provenance = json.loads((output/"workflow-provenance.json").read_text(encoding="utf8"))
    assert provenance["prediction_dxf_sha256"] == actual["drawing.dxf"]
    assert provenance["published_core_sha256"] == actual
    assert provenance["published_core_manifest_match"] == "rollback"
    assert provenance["publication_integrity_verified"] is True
    assert provenance["published_artifact_kind"] == "topology"
    assert provenance["parameterization_accepted"] is False
    assert provenance["constraint_subset_accepted"] is False
    current_contract=json.loads((output/"reconstruction-contract.json").read_text(encoding="utf8"))
    assert current_contract["satisfied"] is False
    assert current_contract["prediction_dxf_sha256"] == digest(output/"drawing.dxf")
    assert current_contract["status"] == "not_certified"
    assert provenance["strict_radius_contract"]["satisfied"] is False


@pytest.mark.parametrize("complete", [True, False])
def test_current_published_provenance_distinguishes_subset_from_all_radii(tmp_path, monkeypatch, complete):
    _, stage, output = setup_radius_pipeline(tmp_path, monkeypatch, complete=complete)
    provenance = json.loads((output/"workflow-provenance.json").read_text(encoding="utf8"))
    assert stage["accepted"] is False
    assert provenance["publication_integrity_verified"] is True
    assert provenance["published_core_manifest_match"] == "candidate"
    assert provenance["published_artifact_kind"] == "parametric"
    assert provenance["constraint_subset_accepted"] is True
    assert provenance["parameterization_accepted"] is False
    assert provenance["strict_radius_contract"]["satisfied"] is complete
    assert provenance["prediction_dxf_sha256"] == hashlib.sha256((output/"drawing.dxf").read_bytes()).hexdigest()


def test_os_error_diagnostic_excludes_error_text_and_unrelated_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    image, output = tmp_path/"inputs/source.png", tmp_path/"output"
    known = PermissionError(13, "secret-provider-value", str(output/"binding-candidates.json"))
    known.winerror = 32
    diagnostic = _os_error_diagnostic(known, image, output)
    assert diagnostic == {"kind":"os_error", "errno":13, "winerror":32,
                          "known_file_locations":[{"attribute":"filename", "role":"output_artifact",
                                                   "workspace_relative_path":"output/binding-candidates.json"}]}
    unrelated = PermissionError(13, "secret-provider-value", str(tmp_path/"private/secret.env"))
    redacted = _os_error_diagnostic(unrelated, image, output)
    assert redacted["known_file_locations"] == []
    assert "secret" not in json.dumps([diagnostic, redacted])
