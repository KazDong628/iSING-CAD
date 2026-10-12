"""A witnessed missing LINE must reach the existing solver before optional work."""
from copy import deepcopy

import pytest

import contour_agent.parametric_pipeline as pipeline
from contour_agent.reconstruction_feedback import geometry_fingerprint
from contour_agent.topology_search import operation_fingerprint


def _fixture():
    graph = {"candidate_id": "base", "units": "mm", "source_grid_pitch_px": 2.,
        "entities": [
            {"id": "g000", "type": "ARC", "start": [0., 0.], "end": [10., 10.],
             "center": [0., 10.], "radius": 10., "clockwise": False},
            {"id": "g001", "type": "LINE", "start": [10., 10.], "end": [0., 0.]}],
        "angle_source_observations": [{"record_id": "a24", "nominal": 24.,
            "reference_axis": "vertical", "verified": True,
            "source_line": {"start_px": [0., 0.], "end_px": [8., 20.]},
            "target_candidates": [{"entity_id": "g000", "supported_span_px": 20.,
                                   "whole_line_supported": False}]}]}
    return {"id": "base", "graph": graph}, [{"record_id": "a24", "kind": "angle", "nominal": 24.}]


def test_line_priority_requires_unmet_finite_source_evidence_and_fresh_parent_operation():
    base, inventory = _fixture()
    feedback = {"satisfied_record_ids": []}
    operations = pipeline._pending_source_line_repairs(base, inventory, feedback, {})
    assert len(operations) == 1 and operations[0]["record_id"] == "a24"
    assert pipeline._pending_source_line_repairs(base, inventory,
        {"satisfied_record_ids": ["a24"]}, {}) == []
    key = operation_fingerprint(geometry_fingerprint(base["graph"]), operations[0])
    assert pipeline._pending_source_line_repairs(base, inventory, feedback, {key: {}}) == []
    other = deepcopy(base)
    other["graph"]["entities"][0]["radius"] = 11.
    assert len(pipeline._pending_source_line_repairs(other, inventory, feedback, {key: {}})) == 1
    for change in ({"verified": False}, {"target_candidates": [{"entity_id": "g000",
                            "supported_span_px": 7., "whole_line_supported": False}]}):
        invalid = deepcopy(base)
        invalid["graph"]["angle_source_observations"][0].update(change)
        assert pipeline._pending_source_line_repairs(invalid, inventory, feedback, {}) == []


def test_line_priority_reads_fresh_edit_observations_without_mutating_certified_parent():
    base, inventory = _fixture()
    fresh = base["graph"].pop("angle_source_observations")
    original = deepcopy(base["graph"])
    base["edit_source_observations"] = {"angle_source_observations": fresh}
    assert pipeline._pending_source_line_repairs(base, inventory, {"satisfied_record_ids": []}, {})
    assert base["graph"] == original


def test_solved_larger_angle_is_filtered_before_selecting_next_line_obligation():
    base,inventory=_fixture()
    solved=deepcopy(base['graph']['angle_source_observations'][0])
    solved.update(record_id='a18',nominal=18.)
    solved['source_line']['end_px']=[16.,40.]
    solved['target_candidates'][0]['supported_span_px']=40.
    base['graph']['angle_source_observations'].insert(0,solved)
    inventory.insert(0,{'record_id':'a18','kind':'angle','nominal':18.})
    original=deepcopy(base)
    operations=pipeline._pending_source_line_repairs(base,inventory,
        {'satisfied_record_ids':['a18']},{})
    assert len(operations)==1 and operations[0]['record_id']=='a24'
    assert base==original


def test_visited_first_scope_does_not_starve_next_source_owned_scope():
    base,inventory=_fixture()
    another=deepcopy(base['graph']['entities'][0]);another['id']='g002'
    base['graph']['entities'].insert(1,another)
    observation=base['graph']['angle_source_observations'][0]
    another_target=deepcopy(observation['target_candidates'][0]);another_target['entity_id']='g002'
    observation['target_candidates'].append(another_target)
    feedback={'satisfied_record_ids':[]}
    first=pipeline._pending_source_line_repairs(base,inventory,feedback,{})[0]
    assert first['entity_ids']==['g000']
    key=operation_fingerprint(geometry_fingerprint(base['graph']),first)
    second=pipeline._pending_source_line_repairs(base,inventory,feedback,{key:{}})
    assert len(second)==1 and second[0]['entity_ids']==['g002']


def _mock_solver(monkeypatch, clock, calls):
    monkeypatch.setattr(pipeline, "_topology_source_validation", lambda *args: {"passed": True})
    monkeypatch.setattr(pipeline, "_preflight_input_identity", lambda *args: {"reference_geometry_used": False})
    monkeypatch.setattr(pipeline, "_solved_source_validation", lambda *args: {"passed": True})
    monkeypatch.setattr("contour_agent.source_support_diagnostics.source_support_diagnostics",
                        lambda *args: {"status": "measured"})
    def bindings(image, document, baseline, graph, output, **kwargs):
        calls.append("binding:" + graph["candidate_id"])
        clock[0] += 100.
        return {"constraints": [], "angle_source_observations": graph.get("angle_source_observations", [])}
    monkeypatch.setattr("contour_agent.constraint_binding.analyze_constraint_bindings", bindings)
    monkeypatch.setattr("contour_agent.parametric_solver.solve_parametric", lambda graph, constraints, **kwargs:
        {"status": "accepted", "accepted": True, "entities": graph["entities"], "constraints": [],
         "diagnostics": {"remaining_shape_dof": 4}})
    monkeypatch.setattr("contour_agent.topology_search.time.monotonic", lambda: clock[0])


def test_missing_line_reaches_shared_preflight_before_bootstrap_and_generic_seeds(tmp_path, monkeypatch):
    base, inventory = _fixture()
    alternate = deepcopy(base)
    alternate["id"] = alternate["graph"]["candidate_id"] = "alternate"
    alternate["graph"]["entities"][0]["radius"] = 11.
    clock, calls = [0.], []
    _mock_solver(monkeypatch, clock, calls)
    monkeypatch.setattr(pipeline, "_source_joint_bootstrap", lambda *args, **kwargs:
                        pytest.fail("Optional bootstrap consumed the mandatory LINE's budget"))
    def stage(image, document, baseline, parent, bundle, output, **kwargs):
        assert kwargs["source_line_preflight_first"] is True
        assert calls == ["binding:base"]
        child = deepcopy(parent)
        child["id"] = child["graph"]["candidate_id"] = "repaired"
        child["graph"]["entities"][0]["radius"] = 12.
        report = kwargs["verifier"](child)
        assert report["solver_status"] != "preflight_not_run"
        # Scheduling alone cannot adopt this uncertified child.
        return parent, {"base_candidate_id": parent["id"], "final_candidate_id": parent["id"],
            "acceptance_gate": {"accepted": False}, "execution": {"operations": []},
            "editor": {}, "evaluator": {}, "trusted_candidate_ids": [parent["id"]]}
    monkeypatch.setattr(pipeline, "_topology_edit_stage", stage)
    final, report = pipeline._topology_edit_loop("unused", {}, {}, base,
        {"annotation_inventory": inventory, "candidates": [base, alternate]}, tmp_path,
        max_rounds=1, max_seconds=300., initial_seed_ids=("alternate",))
    assert calls == ["binding:base", "binding:repaired"]
    assert final["id"] == "base"
    assert report["initial_seed_audit"]["source_line_priority"]["status"] == "reserved"
    assert report["initial_seed_audit"]["attempts"][0]["reason"] == "source_line_numeric_preflight_prioritized"
    assert report["budget"]["max_seconds"] == 300.
    assert report["budget"]["preflights_used"] == 2


def test_priority_stage_preflights_only_line_before_optional_editor_and_radius_edits(tmp_path, monkeypatch):
    base, inventory = _fixture()
    base["overlay_path"] = str(tmp_path / "overlay.png")
    (tmp_path / "overlay.png").write_bytes(b"source overlay")
    operation = pipeline._pending_source_line_repairs(base, inventory, {"satisfied_record_ids": []}, {})[0]
    optional = {"action": "refit_chain_as_annotated_arc", "entity_ids": ["g000"], "record_id": "r99"}
    child = deepcopy(base)
    child["id"] = child["graph"]["candidate_id"] = "repair"
    child["graph"]["entities"][0]["radius"] = 11.
    child["graph"]["source_evidence"] = {"topology_edit": {
        "action": "restore_annotated_line_support", "resegmentation_applied": True,
        "angle_support_record_ids": ["a24"], "record_id": "a24"}}
    monkeypatch.setattr("contour_agent.topology_editing.propose_annotation_arc_edits", lambda *args, **kwargs:
                        [operation, optional])
    calls = []
    def execute(image, document, baseline, parent, bundle, operations, output):
        calls.append("kernel")
        assert operations == [operation]
        return [child], {"operations": [{"operation": operation, "candidate_id": "repair",
                                       "status": "accepted_as_candidate"}]}
    monkeypatch.setattr("contour_agent.topology_editing.execute_topology_edits", execute)
    monkeypatch.setattr(pipeline, "_topology_source_validation", lambda *args: {"passed": True})
    monkeypatch.setattr("contour_agent.planning_provider.evaluate_candidates", lambda rows, **kwargs:
        {"admissible_candidate_ids": ["base"], "evaluated": [{"candidate_id": "base", "admissible": True,
          "score": 1., "metrics": {"entity_count": 2, "source_boundary_support": 1., "unsupported_primitive_count": 0}}]})
    class Editor:
        def propose(self, *args, **kwargs):
            pytest.fail("Optional provider ran before the mandatory source LINE was preflighted")
    def verifier(candidate, *, source_validation):
        calls.append("preflight")
        assert source_validation == {"passed": True}
        return {"solver_status": "candidate_rejected", "solver_accepted": False,
            "binding_status": "local", "constraint_count": 0, "issue_count": 1,
            "topology_source_validation": source_validation, "source_validation": {"passed": False}}
    final, report = pipeline._topology_edit_stage("unused", {}, {}, base,
        {"annotation_inventory": inventory, "candidates": [base]}, tmp_path,
        use_api=True, editor_provider=Editor(), verifier=verifier,
        provider_guard=lambda role: pytest.fail("Skipped provider must not consume request budget"),
        feedback={"satisfied_record_ids": [], "issue_count": 0}, source_line_preflight_first=True)
    assert calls == ["kernel", "preflight"]
    assert report["editor"]["network_requests"] == 0
    assert report["source_line_preflight_prioritized"] is True
    assert report["editor"]["reason"] == "source_line_numeric_preflight_prioritized"
    assert report["exploratory_candidate_ids"] == ["repair"]
    assert final["id"] == "base"


def test_priority_stage_filters_visited_scope_before_reserving_one_preflight(tmp_path,monkeypatch):
    base,inventory=_fixture()
    other=deepcopy(base['graph']['entities'][0]);other['id']='g002'
    base['graph']['entities'].insert(1,other)
    target=deepcopy(base['graph']['angle_source_observations'][0]['target_candidates'][0])
    target['entity_id']='g002';base['graph']['angle_source_observations'][0]['target_candidates'].append(target)
    first={'action':'restore_annotated_line_support','entity_ids':['g000'],'record_id':'a24',
           'evidence_tags':['annotation_target','collinear_support','source_boundary']}
    fresh={**first,'entity_ids':['g002']}
    # Exercise the real generic proposer: its top-1 LINE reservation exposes
    # only the visited scope. The execution stage must use the complete source
    # scope list already available to the scheduler.
    from contour_agent.topology_editing import propose_annotation_arc_edits
    generic=propose_annotation_arc_edits(base['graph'],inventory,limit=4)
    assert generic[0]['entity_ids']==['g000']
    assert all(row['entity_ids']!=['g002'] for row in generic)
    seen=[]
    def execute(*args):
        seen.extend(args[5]);return [],{'operations':[]}
    monkeypatch.setattr('contour_agent.topology_editing.execute_topology_edits',execute)
    monkeypatch.setattr('contour_agent.planning_provider.evaluate_candidates',lambda *args,**kwargs:
        {'admissible_candidate_ids':['base'],'evaluated':[]})
    pipeline._topology_edit_stage('unused',{}, {},base,{'annotation_inventory':inventory},tmp_path,
        verifier=lambda *args:{},feedback={'satisfied_record_ids':[]},source_line_preflight_first=True,
        operation_filter=lambda row:row['entity_ids']!=['g000'])
    assert seen==[fresh]


def test_stateful_reservation_reaches_kernel_once_and_does_not_consume_unselected_scope(tmp_path, monkeypatch):
    from contour_agent.topology_search import SearchBudget
    base, inventory = _fixture()
    other = deepcopy(base['graph']['entities'][0]); other['id'] = 'g002'
    base['graph']['entities'].insert(1, other)
    target = deepcopy(base['graph']['angle_source_observations'][0]['target_candidates'][0])
    target['entity_id'] = 'g002'
    base['graph']['angle_source_observations'][0]['target_candidates'].append(target)
    budget = SearchBudget(max_seconds=300.)
    parent_hash = geometry_fingerprint(base['graph'])
    calls, executed = [], []
    def reserve(operation):
        calls.append(tuple(operation['entity_ids']))
        return budget.visit_operation(parent_hash, operation)[0]
    def execute(*args):
        executed.extend(args[5]); return [], {'operations': []}
    monkeypatch.setattr('contour_agent.topology_editing.execute_topology_edits', execute)
    monkeypatch.setattr('contour_agent.planning_provider.evaluate_candidates', lambda *args, **kwargs:
        {'admissible_candidate_ids': ['base'], 'evaluated': []})
    def stage():
        return pipeline._topology_edit_stage('unused', {}, {}, base, {'annotation_inventory': inventory}, tmp_path,
            verifier=lambda *args: {}, feedback={'satisfied_record_ids': []},
            source_line_preflight_first=True, operation_filter=reserve)
    stage()
    assert calls == [('g000',)]
    assert [row['entity_ids'] for row in executed] == [['g000']]
    assert len(budget.visited_operations) == 1
    # Next stage skips the actual prior reservation and reaches the next
    # scope. Within that stage the new reservation must remain executable.
    calls.clear(); executed.clear()
    stage()
    assert calls == [('g000',), ('g002',)]
    assert [row['entity_ids'] for row in executed] == [['g002']]
    assert len(budget.visited_operations) == 2


def _locked_feedback(target="source-arc", value=176., kind="radius", axis=None):
    return {"bound_record_ids": ["r176"], "constraint_count": 1,
        "solver_status": "accepted", "solver_accepted": True,
        "source_validation": {"passed": True}, "post_solve_source_accepted": True,
        "source_binding_locks": [{"record_id": "r176", "kind": kind, "value": value,
            "angle_mode": None, "reference_axis": axis, "strict_source_nominal": True,
            "target_stable_ids": [target], "target_lineage_available": True,
            "fresh_binding_and_solve_verified": True}]}


def test_binding_lock_accepts_freshly_verified_descendant_after_split_but_not_other_object():
    before = _locked_feedback()
    descendant = _locked_feedback("child-arc")
    descendant["entity_ancestry"] = {"child-arc": ["child-arc", "source-arc"]}
    assert pipeline._source_locked_constraint_regression(before, descendant)["passed"] is True
    unrelated = deepcopy(descendant)
    unrelated["entity_ancestry"] = {"child-arc": ["child-arc", "different-source-arc"]}
    gate = pipeline._source_locked_constraint_regression(before, unrelated)
    assert gate["passed"] is False
    assert gate["source_binding_lock_failures"] == [{"record_id": "r176",
        "reason": "verified_source_binding_target_lineage_lost"}]


@pytest.mark.parametrize("alteration", [{"value": 140.555}, {"strict_source_nominal": False},
    {"reference_axis": "horizontal"}, {"fresh_binding_and_solve_verified": False}])
def test_binding_lock_rejects_changed_semantics_despite_preserved_record_and_counts(alteration):
    before = _locked_feedback()
    after = deepcopy(before)
    after["source_binding_locks"][0].update(alteration)
    assert pipeline.constraint_regression(before, after)["passed"] is True
    gate = pipeline._source_locked_constraint_regression(before, after)
    assert gate["passed"] is False
    assert "verified_source_nominal_changed_or_unsolved" in gate["reasons"]


def test_source_binding_locks_are_built_only_from_the_source_and_current_accepted_solve():
    graph = {"entities": [{"id": "g019", "stable_id": "source-arc", "type": "ARC", "radius": 140.555}]}
    row = {"id": "k176", "record_id": "r176", "kind": "radius", "entities": ["g019"],
           "value": 176., "required": True, "source": "ocr_local_binding", "enforcement": "exact",
           "nominal_source": "source_ocr"}
    bindings = {"constraints": [row]}
    solution = {"accepted": True, "constraints": [{**row, "passed": True}]}
    feedback = {"solver_accepted": True, "post_solve_source_accepted": True,
                "verified_radius_record_ids": ["r176"]}
    locks = pipeline._source_binding_locks(graph, bindings, solution, feedback)
    assert locks[0]["value"] == 176.  # Never freeze the initial fitted radius.
    assert locks[0]["target_stable_ids"] == ["source-arc"]
    assert pipeline._source_binding_locks(graph, bindings, {**solution, "accepted": False}, feedback) == []
    assert pipeline._source_binding_locks(graph, bindings, solution,
        {**feedback, "post_solve_source_accepted": False}) == []
    assert pipeline._source_binding_locks(graph, bindings, solution,
        {**feedback, "verified_radius_record_ids": []}) == []
