"""Source-design contact obligations followed by an unchanged joint problem.

Unlike a derivative test at an approximate vertex, source review can request
moving the contact. Model consensus is a design hypothesis, not proof of source
accuracy. Publication still requires all prior equations, source gates and
independent native DXF readback. Reference CAD is never consumed here.
"""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path

import numpy as np

from .joint_review_provider import MAX_JOINTS, validate_joint_review


def joint_review_inventory(image, document, baseline, graph, bindings, solution):
    from .ocr import canonical_records
    from .source_radius_hypotheses import graph_from_solution
    from .source_topology_completion import unconfirmed_curved_joins
    from .topology_candidates import _affines
    observed = graph_from_solution(graph, baseline, solution)
    transform, _, _ = _affines(graph, baseline)
    entities = {e["id"]: e for e in observed["entities"]}
    owned = {}
    for c in bindings["constraints"]:
        if c.get("record_id"):
            for eid in c.get("entities", []):
                owned.setdefault(eid, []).append(c["record_id"])
    rows = []
    for ids in unconfirmed_curved_joins(observed, bindings):
        a, b = [entities[e] for e in ids]
        shared = set((a["start_node"], a["end_node"])) & set((b["start_node"], b["end_node"]))
        if len(shared) != 1 or a["end_node"] != b["start_node"]:
            continue
        # A source-bound radius is necessary to identify a design contact.
        # An arbitrary free fitted ARC does not acquire a G1 obligation here.
        radius_records = [c["record_id"] for c in bindings["constraints"]
                          if c["kind"] == "radius" and any(e in ids for e in c["entities"])
                          and c.get("source_arrow_verified") is True]
        if not radius_records:
            continue
        source_point = transform([a["end"]])[0].tolist()
        rows.append(dict(joint_id=f"j{len(rows):03d}", entities=ids, node_id=a["end_node"],
                         object_types=[a["type"], b["type"]], source_joint_px=source_point,
                         search_half_extent_px=140.,
                         source_record_ids=sorted(set(owned.get(ids[0], [])+owned.get(ids[1], []))),
                         source_radius_record_ids=sorted(set(radius_records))))
        if len(rows)==MAX_JOINTS:break
    records = [dict(id=r["id"], text=r["text"], box=r["box"]) for r in canonical_records(document)]
    return rows, records


def analytic_fillet_seed(entities, constraints):
    """Initialize two-LINE fillets from source constants, never lock this seed."""
    result = copy.deepcopy(entities)
    radii = {c["entities"][0]: float(c["value"]) for c in constraints
             if c["kind"] == "radius" and len(c["entities"]) == 1}
    tangents = {frozenset(c["entities"]) for c in constraints if c["kind"] == "tangent"}
    directions = {c["entities"][0] for c in constraints if c["kind"] in {"angle", "horizontal", "vertical"}
                  and len(c["entities"]) == 1}
    changed = []
    for i, arc in enumerate(result):
        before, after = result[(i-1) % len(result)], result[(i+1) % len(result)]
        if (arc["type"] != "ARC" or arc["id"] not in radii or before["type"] != "LINE" or
                after["type"] != "LINE" or not {before["id"], after["id"]} <= directions or
                any(frozenset([arc["id"], e["id"]]) not in tangents for e in (before, after))):
            continue
        a, p, q, b = [np.asarray(x, float) for x in (before["start"], before["end"], after["start"], after["end"])]
        u, v = p-a, b-q
        try:
            ts = np.linalg.solve(np.column_stack((u, -v)), q-a)
        except np.linalg.LinAlgError:
            continue
        corner = a+ts[0]*u
        u /= np.linalg.norm(u); v /= np.linalg.norm(v)
        turn = math.atan2(float(u[0]*v[1]-u[1]*v[0]), float(u@v))
        if not math.radians(5.) < abs(turn) < math.radians(160.) or (turn < 0) != arc["clockwise"]:
            continue
        radius = radii[arc["id"]]; trim = radius*math.tan(abs(turn)/2.)
        first, last = corner-trim*u, corner+trim*v
        if float((first-a)@u) <= 1e-6 or float((b-last)@v) <= 1e-6:
            continue
        center = first+np.array([-u[1], u[0]])*math.copysign(radius, turn)
        before["end"] = first.tolist(); after["start"] = last.tolist()
        arc.update(start=first.tolist(), end=last.tolist(), center=center.tolist(), radius=radius)
        changed.append(arc["id"])
    return result, dict(method="source_two_line_exact_fillet_initialization_v1", entities=changed,
                        initial_guess_only=True, endpoints_fixed_in_solver=False, ground_truth_used=False)


def complete_source_joints(image, document, baseline, graph, bindings, solution, inventory, output, *,
                           provider=None, use_api=False, budget_seconds=120., refiner=None,
                           source_validator=None, exporter=None):
    from .parametric_pipeline import (_write, _constraint_receipts_complete, _final_source_multistart,
                                      _solved_source_validation, export_parametric)
    from .parametric_solver import solve_parametric
    refiner = refiner or _final_source_multistart
    validate = source_validator or _solved_source_validation
    exporter = exporter or export_parametric
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    receipt = dict(schema_version="source-design-joint-completion-v1", accepted=False, reviews=[],
                   obligations=[], ground_truth_used=False, acceptance_thresholds_changed=False,
                   optimization_domain_recentered=False, maximum_network_requests=2,
                   maximum_solver_seconds=min(120., max(0., float(budget_seconds))),
                   model_consensus_is_source_accuracy_proof=False, publication_approved=False)
    def finish(reason, state=None):
        receipt["reason"] = reason; _write(output/"completion.json", receipt)
        return (*(state or (bindings, solution, inventory)), receipt)
    if (solution.get("accepted") is not True or
            not _constraint_receipts_complete(bindings, solution, graph)):
        return finish("previous_subset_not_certified")
    if not use_api or provider is None:
        return finish("online_source_design_review_unavailable")
    if graph.get("source_sha256") != hashlib.sha256(Path(image).read_bytes()).hexdigest():
        return finish("source_image_identity_changed")
    rows, records = joint_review_inventory(image, document, baseline, graph, bindings, solution)
    receipt["obligations"] = copy.deepcopy(rows)
    if not rows:
        return finish("no_eligible_unconfirmed_source_contacts")
    try:
        _write(output/"completion.json", receipt)
        for view in (0, 1):
            try:
                review = provider.review(image, rows, records, view=view, timeout=90.)
                safe = {k: review[k] for k in ("status", "http_success", "schema_success", "http_status",
                    "network_requests", "model", "protocol", "crops", "image_sent", "ground_truth_sent",
                    "cad_values_sent", "elapsed_seconds") if k in review}
                if not (review.get("http_success") is True and review.get("schema_success") is True):
                    raise ValueError("joint_review_not_succeeded")
                safe["joints"] = validate_joint_review({"joints": review.get("joints")},
                    {r["joint_id"] for r in rows}, {r["id"] for r in records})
            except InterruptedError:
                raise
            except Exception as error:
                safe = dict(status="failed", failure_type=type(error).__name__,
                            http_success=getattr(error, "http_success", None), schema_success=False,
                            network_requests=getattr(error, "network_requests", None))
            receipt["reviews"].append(safe); _write(output/"completion.json", receipt)
        if any(r.get("schema_success") is not True for r in receipt["reviews"]):
            return finish("source_review_incomplete_previous_artifacts_retained")
        opinions = [{r["joint_id"]: r for r in review["joints"]} for review in receipt["reviews"]]
        fresh, inv = copy.deepcopy(bindings), copy.deepcopy(inventory)
        added = []
        for row in rows:
            a, b = [op[row["joint_id"]] for op in opinions]
            support = set(a["source_record_ids"]) & set(b["source_record_ids"])
            eligible = (a["relationship"] == b["relationship"] == "tangent" and
                        bool(support & set(row["source_radius_record_ids"])))
            row["review_status"] = "source_design_tangent_consensus" if eligible else "unresolved_design_relationship"
            if not eligible:
                continue
            cid = "kdesign_"+row["node_id"]
            if any(c["id"] == cid for c in fresh["constraints"]):
                raise ValueError("source_joint_constraint_identity_collision")
            proof = dict(method="two_view_original_image_design_contact_v1", verified=True,
                evidence_class="source_design_review", independent_source_tangent_measurement=False,
                source_record_ids=sorted(support), source_image_sha256=graph["source_sha256"],
                original_image_review_artifact=str((output/"completion.json").resolve()),
                approximate_contact_is_not_design_evidence=True, ground_truth_used=False)
            c = dict(id=cid, kind="tangent", record_id=None, entities=row["entities"], nodes=[row["node_id"]],
                     value=0., source="source_geometry", required=True, enforcement="exact",
                     evidence_class="source_design_review", admission_method="online_source_design_consensus")
            fresh["constraints"].append(c); added.append(cid)
            fresh.setdefault("bindings", []).append(dict(relation_id=cid[1:], accepted=True,
                source="source_design_review", evidence=copy.deepcopy(proof),
                reason="original_image_design_consensus_pending_joint_solve"))
            inv.setdefault("relations", []).append(dict(id=cid[1:], type="tangent", entities=row["entities"],
                nodes=[row["node_id"]], source="source_design_review", required=True,
                local_reliable=True, evidence=proof))
        receipt["obligations"] = rows; receipt["added_constraint_ids"] = added
        if not added:
            return finish("no_new_source_design_contact")
        fresh.setdefault("counts", {})["constraints"] = len(fresh["constraints"])
        fresh["source_joint_review_artifact"] = str((output/"completion.json").resolve())
        _write(output/"constraint-bindings-proposed.json", fresh)
        seed, receipt["initialization"] = analytic_fillet_seed(solution["entities"], fresh["constraints"])
        _write(output/"completion.json", receipt)
        candidate = refiner(image, document, baseline, graph, fresh["constraints"], output/"joint-solve",
            solver=solve_parametric, budget_seconds=receipt["maximum_solver_seconds"], initial_seed_entities=seed)
        _write(output/"parametric-solution.json", candidate)
        receipt["solver_status"] = candidate.get("status")
        if (candidate.get("accepted") is not True or
                not _constraint_receipts_complete(fresh, candidate, graph)):
            return finish("enlarged_joint_problem_not_certified")
        check = validate(image, document, baseline, graph, candidate["entities"])
        receipt["original_source_validation"] = check
        if check.get("passed") is not True:
            return finish("original_source_gate_failed")
        old_dof = (solution.get("diagnostics") or {}).get("remaining_shape_dof")
        new_dof = (candidate.get("diagnostics") or {}).get("remaining_shape_dof")
        receipt.update(shape_dof_before=old_dof, shape_dof_after=new_dof)
        if isinstance(old_dof, (int, float)) and isinstance(new_dof, (int, float)) and new_dof > old_dof:
            return finish("shape_freedoms_increased")
        exported = exporter(image, baseline, {**candidate, "constraints": fresh["constraints"]}, output/"native-export")
        required = ("exact_radius_validation", "strict_relation_validation", "native_measurement_validation",
                    "native_source_angle_validation", "native_axis_relation_validation")
        receipt["native_checks"] = {k: exported["validation"].get(k, {}) for k in required}
        if any(v.get("passed") is not True or v.get("dxf_readback_performed") is not True
               for v in receipt["native_checks"].values()):
            return finish("native_equations_not_certified")
        fresh["inventory_artifact"] = str(output/"binding-candidates.json")
        _write(output/"binding-candidates.json", inv); _write(output/"constraint-bindings.json", fresh)
        receipt.update(accepted=True, all_prior_equations_retained=True)
        return finish("source_design_contacts_jointly_certified", (fresh, candidate, inv))
    except InterruptedError:
        receipt["status"] = "interrupted"; _write(output/"completion.json", receipt)
        raise
    except Exception as error:
        receipt["failure_type"] = type(error).__name__
        return finish("joint_completion_failed_previous_artifacts_retained")
