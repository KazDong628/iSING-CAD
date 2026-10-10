"""A source-witnessed finite OCR line must reach the bounded edit preflight."""
from copy import deepcopy

from contour_agent.parametric_pipeline import (
    _ordered_topology_operations,
    _topology_edit_stage,
)


def _angle_graph(*, verified=True, span=42.):
    return {
        "source_grid_pitch_px": 1.5,
        "entities": [{"id": "g010", "type": "ARC", "end_node": "v011"},
                     {"id": "g011", "type": "ARC", "start_node": "v011"},
                     {"id": "g012", "type": "LINE"}],
        "nodes": [{"id": "v011", "source_px": [55., 105.]}],
        "annotation_support": [
            {"candidate_entity_id": entity_id, "kind": "radius",
             "status": "candidate_supported", "arrowhead_verified": True}
            for entity_id in ("g010", "g011")],
        "angle_source_observations": [{
            "record_id": "r032", "nominal": 24., "reference_axis": "vertical",
            "verified": verified,
            "source_line": {"start_px": [10., 10.], "end_px": [100., 200.]},
            "target_candidates": [{"entity_id": "g010", "entity_type": "ARC",
                                   "whole_line_supported": False, "supported_span_px": span}],
        }],
    }


def _provider_operations():
    return [{"action": "refit_entity_as_line", "entity_ids": [f"g{i:03d}"],
             "record_id": None} for i in range(5)]


def _line_operation():
    return {"action": "restore_annotated_line_support", "entity_ids": ["g010", "g011"],
            "record_id": "r032"}


def test_five_provider_edits_leave_one_trial_for_unmet_verified_line():
    provider = _provider_operations()
    line = _line_operation()
    operations, skipped = _ordered_topology_operations(
        provider, [line], _angle_graph(), [{"record_id": "r032", "kind": "angle",
                                             "nominal": 24.}],
        {"satisfied_record_ids": []})
    assert operations == [*provider[:4], line]
    assert skipped == []


def test_unwitnessed_or_satisfied_line_does_not_displace_provider_edit():
    provider = _provider_operations()
    inventory = [{"record_id": "r032", "kind": "angle", "nominal": 24.}]
    for graph, feedback in [
        (_angle_graph(verified=False), {"satisfied_record_ids": []}),
        (_angle_graph(span=3.), {"satisfied_record_ids": []}),
        (_angle_graph(), {"satisfied_record_ids": ["r032"]}),
    ]:
        operations, skipped = _ordered_topology_operations(
            provider, [_line_operation()], graph, inventory, feedback)
        assert operations == provider
        assert skipped == []
    single = {**_line_operation(), "entity_ids": ["g010"]}
    operations, _ = _ordered_topology_operations(
        provider, [single], _angle_graph(), inventory, {"satisfied_record_ids": []})
    assert operations == provider


def test_reserved_line_enters_preflight_and_online_base_verdict_cannot_erase_it(
        tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    source.write_bytes(b"synthetic source")
    base_graph = _angle_graph()
    base_graph["entities"] = [
        {"id": f"g{i:03d}", "type": "ARC" if i in (10, 11) else "LINE",
         **({"end_node": "v011"} if i == 10 else
            {"start_node": "v011"} if i == 11 else {})}
        for i in range(19)]
    base = {"id": "base", "graph": base_graph, "overlay_path": str(source)}
    bundle = {"annotation_inventory": [{"record_id": "r032", "kind": "angle",
                                        "nominal": 24.}], "candidates": [base]}
    feedback = {"satisfied_record_ids": [], "bound_record_ids": [], "issue_count": 8,
                "source_validation": {"passed": True},
                "topology_source_validation": {"passed": True}}
    provider = _provider_operations()
    line = _line_operation()
    attempted = []
    preflighted = []

    class Editor:
        def propose(self, *_args, **_kwargs):
            return {"schema_success": True, "operations": deepcopy(provider)}

    class Evaluator:
        def select(self, *_args, **_kwargs):
            return {"schema_success": True, "selected_candidate_id": "base",
                    "decision": "preserve_base"}

    def execute(_image, _document, _baseline, _base, _bundle, operations, _output):
        attempted.extend(deepcopy(operations))
        graph = deepcopy(base_graph)
        graph["entities"].append({"id": "g019", "type": "LINE",
                                  "angle_support_evidence": {
                                      "record_id": "r032",
                                      "requires_angle_binding_and_solve": True,
                                      "ground_truth_used": False,
                                      "source_interval_endpoints_px": [[10., 10.], [50., 50.]]}})
        graph["source_evidence"] = {"topology_edit": {
            **line, "resegmentation_applied": True, "net_entity_reduction": -1,
            "angle_support_record_ids": ["r032"]}}
        candidate = {"id": "cand-edit-05-annotated-line", "graph": graph}
        audit = {"operations": [{"operation_index": 5, "status": "accepted_as_candidate",
                                 "candidate_id": candidate["id"], "operation": line}]}
        return [candidate], audit

    def verify(candidate, *, source_validation):
        preflighted.append(candidate["id"])
        assert source_validation["passed"]
        return {"solver_status": "accepted", "solver_accepted": True,
                "constraint_count": 1, "source_validation": {"passed": True},
                "topology_source_validation": {"passed": True},
                "satisfied_record_ids": ["r032"], "issue_count": 7}

    def evaluate(pool, **_kwargs):
        return {"admissible_candidate_ids": [row["id"] for row in pool],
                "evaluated": [{"candidate_id": row["id"], "admissible": True,
                               "score": .86 if row["id"] == "base" else .85,
                               "metrics": {"entity_count": 19 if row["id"] == "base" else 20,
                                           "source_boundary_support": .83,
                                           "unsupported_primitive_count": 4}}
                              for row in pool]}

    monkeypatch.setattr("contour_agent.topology_editing.propose_annotation_arc_edits",
                        lambda *_args, **_kwargs: [line])
    monkeypatch.setattr("contour_agent.topology_editing.execute_topology_edits", execute)
    monkeypatch.setattr("contour_agent.parametric_pipeline._topology_source_validation",
                        lambda *_args: {"passed": True})
    monkeypatch.setattr("contour_agent.parametric_pipeline.constraint_regression",
                        lambda *_args: {"passed": True, "reasons": []})
    monkeypatch.setattr("contour_agent.planning_provider.evaluate_candidates", evaluate)

    selected, receipt = _topology_edit_stage(
        source, {}, {}, base, bundle, tmp_path,
        editor_provider=Editor(), evaluator_provider=Evaluator(), use_api=True,
        feedback=feedback, verifier=verify)
    assert attempted == [*provider[:4], line]
    assert preflighted == ["cand-edit-05-annotated-line"]
    assert receipt["preflight_queue"][0]["status"] == "passed"
    assert receipt["evaluator"]["selected_candidate_id"] == "base"
    assert selected["id"] == "cand-edit-05-annotated-line"
    assert receipt["acceptance_gate"]["selection_source"] == "source_verified_required_ocr_topology"
