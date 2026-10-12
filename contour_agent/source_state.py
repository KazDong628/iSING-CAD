"""Resume a matched source geometry/binding/solve tuple, never a reference CAD."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")


def resume_source_state(image, document, baseline, graph, bindings, solution, inventory, output):
    from .parametric_pipeline import _constraint_receipts_complete, _solved_source_validation
    from .parametric_solver import native_measurement_checks, native_source_nominal_angle_checks
    from .radius_contract import exact_radius_checks
    from .relation_contract import relation_checks
    from .source_radius_hypotheses import graph_from_solution

    if (graph.get("source_sha256") != hashlib.sha256(Path(image).read_bytes()).hexdigest() or
            graph.get("ground_truth_used") is not False):
        raise ValueError("resume_source_provenance_mismatch")
    if solution.get("accepted") is not True or not _constraint_receipts_complete(bindings, solution, graph):
        raise ValueError("resume_requires_matching_certified_source_constraints")
    entities = solution["entities"]
    checks = [exact_radius_checks(entities, bindings["constraints"]),
              relation_checks(entities, bindings["constraints"]),
              native_measurement_checks(entities, bindings["constraints"]),
              native_source_nominal_angle_checks(entities, bindings["constraints"])]
    if any(check.get("passed") is not True for check in checks):
        raise ValueError("resume_native_source_equations_failed")
    validation = _solved_source_validation(image, document, baseline, graph, entities)
    if validation.get("passed") is not True:
        raise ValueError("resume_original_source_validation_failed")
    observed = graph_from_solution(graph, baseline, solution)
    # Flatten an older resume into its original domain instead of accumulating
    # shifted displacement boxes on each replay.
    from .source_topology_initial_guess import line_seed_problem_graphs
    domain, parent = line_seed_problem_graphs(graph)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    files = {"domain.json": domain, "parent.json": parent, "bindings.json": bindings,
             "solution.json": solution, "inventory.json": inventory}
    for name, value in files.items():
        _write(output/name, value)
    observed.setdefault("source_evidence", {}).pop("source_line_initial_guess", None)
    observed["source_evidence"]["proposal_stroke_support"] = copy.deepcopy(validation["after"])
    observed["source_evidence"]["baseline_stroke_support"] = copy.deepcopy(validation["before"])
    proof = {"method": "matched_certified_source_state_v1", "ground_truth_used": False,
             "artifact_directory": str(output.resolve()),
             "sha256": {name: hashlib.sha256((output/name).read_bytes()).hexdigest() for name in files},
             "source_sha256": observed.get("source_sha256"),
             "original_domain_recentered": False, "fresh_binding_required": True}
    observed["source_evidence"]["source_state_resume"] = proof
    _write(output/'observed-topology.json',observed)
    _write(output/"resume.json", {**proof, "source_validation": validation,
                                "native_equations_passed": True})
    # A rejected next attempt must not erase the already certified source
    # subset. This checkpoint is re-admitted from current source evidence by
    # the normal publication guard, never trusted just because it is saved.
    from .reconstruction_feedback import reconstruction_feedback
    feedback=reconstruction_feedback(observed,bindings,solution)
    feedback['source_validation']=copy.deepcopy(validation)
    for name,value in [('constraint-bindings.json',bindings),('parametric-solve.json',solution),
                       ('reconstruction-feedback.json',feedback)]:
        _write(output/name,value)
    return observed


def resumed_files(graph):
    proof = (graph.get("source_evidence") or {}).get("source_state_resume")
    if not proof:
        return None
    if (proof.get("method") != "matched_certified_source_state_v1" or
            proof.get("ground_truth_used") is not False or
            proof.get("source_sha256") != graph.get("source_sha256")):
        raise ValueError("invalid_source_state_resume")
    result = {}
    for name in ("domain.json", "parent.json", "bindings.json", "solution.json", "inventory.json"):
        raw = (Path(proof["artifact_directory"])/name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != proof["sha256"].get(name):
            raise ValueError("source_state_resume_snapshot_changed")
        result[name] = json.loads(raw)
    identity = lambda e: tuple(e.get(k) for k in ("id", "type", "start_node", "end_node", "clockwise"))
    domain = result["domain.json"]
    if ([identity(e) for e in domain["entities"]] != [identity(e) for e in graph["entities"]] or
            domain.get("units") != graph.get("units") or domain.get("source_sha256") != graph.get("source_sha256")):
        raise ValueError("source_state_resume_identity_changed")
    return result


def retain_checkpoint_for_fresh_source_recheck(model, graph, *, frozen_replay):
    """New arrow proposals require rechecking, not erasure of a certified subset."""
    from .reconstruction_priority import mask_is_hard
    return bool(frozen_replay and not mask_is_hard(model) and resumed_files(graph) is not None)


def revalidate_checkpoint_binding_subset(image, document, baseline, trusted, bindings, inventory, output):
    """Fresh source proof distinguishes nondetection from positive contradiction."""
    from .source_radius_hypotheses import (graph_from_solution, carried_binding_certificate,
                                          merge_carried_bindings)
    from .constraint_binding import radius_binding_coverage
    from .parametric_pipeline import _constraint_signatures, _canonical_sha256
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    original=trusted['bindings']['constraints']
    graph=trusted['graph']
    required_digest=_canonical_sha256(sorted(_constraint_signatures(graph,original)))
    receipt={'status':'rejected','passed':False,'ground_truth_used':False,
        'previous_constraint_count':len(original),'previous_typed_constraints_sha256':required_digest,
        'input_identity_verified_against_checkpoint':False,'current_API_reconfirmed':False}
    try:
        observed=graph_from_solution(graph,baseline,trusted['solution'])
        carry=carried_binding_certificate(image,document,baseline,graph,observed,
            trusted['inventory'],original,inventory,bindings,trusted['solution'])
        receipt['carried_binding_certificate']=carry
        if carry.get('passed') is True:
            updated,fresh=merge_carried_bindings(trusted['inventory'],inventory,bindings,carry)
            if not _constraint_signatures(graph,original)<=_constraint_signatures(graph,updated['constraints']):
                raise ValueError('checkpoint_source_recheck_lost_known_typed_obligation')
            bound={r.get('record_id') for r in updated['constraints']}-{None}
            counts=updated.setdefault('counts',{})
            counts.update(constraints=len(updated['constraints']),bound_source_records=len(bound),
                bound_angle_records=sum(r['kind']=='angle' for r in updated['constraints']),
                carried_source_constraints=len(carry['constraints']))
            if 'recognized_dimensions' in counts:counts['unbound_dimensions']=counts['recognized_dimensions']-len(bound)
            updated['radius_binding_coverage']=radius_binding_coverage(fresh,graph,updated['constraints'],updated['bindings'])
            updated['inventory_artifact']=str(output/'binding-candidates.json')
            receipt.update(status='known_source_subset_revalidated',passed=True)
            bindings,inventory=updated,fresh
    except (ValueError,KeyError,TypeError,ArithmeticError,OSError) as error:
        receipt['error_type']=type(error).__name__
    _write(output/'checkpoint-known-source-revalidation.json',receipt)
    if receipt['passed']:
        _write(output/'constraint-bindings.json',bindings)
        _write(output/'binding-candidates.json',inventory)
    return bindings,inventory,receipt
