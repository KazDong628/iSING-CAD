"""Source-owned topology proposals are initial guesses, never certificates."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np


def line_seed_admission(graph, source_validation):
    """Admit one finite annotated LINE to a bounded solve, not publication."""
    result={"eligible":False,"reason":"not_a_source_owned_line_initial_guess",
            "publication_approved":False,"ground_truth_used":False}
    source_first = ((source_validation.get("priority_resolution") or {}).get(
        "mask_fidelity_required_for_source_publication") is False and source_validation.get("passed") is True)
    if not source_first and (source_validation.get("passed") is not False or
            set(source_validation.get("reasons") or [])!={"original_oracle_mask_budget_exceeded"}):
        return result
    edit=(graph.get("source_evidence") or {}).get("topology_edit") or {}
    rid=edit.get("record_id")
    if (edit.get("action")!="restore_annotated_line_support" or
            edit.get("resegmentation_applied") is not True or
            edit.get("ground_truth_used") is not False or
            not isinstance(rid,str) or rid not in (edit.get("angle_support_record_ids") or []) or
            edit.get("net_entity_reduction")!=-1):
        return result
    try:
        grid=float(graph["source_grid_pitch_px"])
        matches=[]
        for entity in graph["entities"]:
            proof=entity.get("angle_support_evidence") or {}
            if proof.get("record_id")!=rid:continue
            ends=np.asarray(proof.get("source_interval_endpoints_px"),float)
            nominal=proof.get("nominal")
            if (entity.get("type")!="LINE" or
                    proof.get("requires_angle_binding_and_solve") is not True or
                    proof.get("ground_truth_used") is not False or
                    proof.get("method") not in {"locally_verified_angle_ink_interval_topology_restore",
                                               "verified_two_radius_joint_line_topology_restore"} or
                    proof.get("reference_axis") not in {"horizontal","vertical"} or
                    type(nominal) not in (int,float) or not math.isfinite(nominal) or not 0<nominal<90 or
                    not math.isfinite(grid) or grid<=0 or ends.shape!=(2,2) or not np.isfinite(ends).all() or
                    float(np.linalg.norm(ends[1]-ends[0]))<4*grid):
                return result
            if proof.get("method")=="verified_two_radius_joint_line_topology_restore":
                contact=entity.get("source_contact_construction") or {}
                radii=contact.get("radius_record_ids")
                if (contact.get("method")!="source_nominal_exact_arc_line_arc_v1" or
                        contact.get("angle_record_id")!=rid or
                        contact.get("ground_truth_used") is not False or
                        not isinstance(radii,list) or len(radii)!=2 or len(set(radii))!=2 or
                        not all(isinstance(value,str) and value for value in radii) or
                        any(contact.get(key) is not True for key in
                            ("radius_exact","angle_exact","contacts_exact","outer_endpoints_fixed",
                             "source_line_support_checked_during_search"))):
                    return result
            matches.append({"entity_id":entity["id"],"record_id":rid,
                            "value":nominal,"reference_axis":proof["reference_axis"]})
        if len(matches)!=1:return result
    except (ValueError,TypeError,KeyError):
        return result
    return {**result,"eligible":True,"reason":("source_design_candidate_requires_joint_certification"
            if source_first else "only_initial_mask_fit_failed"),**matches[0]}


def _unique_retained_finite_line(parent, descendants):
    """Identify a trimmed support by its fixed outer endpoint and direction.

    LINE/ARC/LINE edits have two parents and a shared replacement chain. The
    chain alone is ambiguous; a unique collinear finite descendant retaining
    an original endpoint identifies the old support without using a label's
    value or the prediction's proximity to reference geometry.
    """
    try:
        endpoints=np.asarray([parent['start'],parent['end']],float)
        delta=endpoints[1]-endpoints[0];length=float(np.linalg.norm(delta))
        if endpoints.shape!=(2,2) or not np.isfinite(endpoints).all() or length<=1e-7:return None
        direction=delta/length;matches=[]
        for row in descendants:
            points=np.asarray([row['start'],row['end']],float)
            if points.shape!=(2,2) or not np.isfinite(points).all():continue
            relative=points-endpoints[0];along=relative@direction
            cross=relative[:,0]*direction[1]-relative[:,1]*direction[0]
            if (np.max(np.abs(cross))<=1e-7 and
                    float(along[1]-along[0])>1e-7 and
                    min(along)>=-1e-7 and max(along)<=length+1e-7 and
                    (np.linalg.norm(points[0]-endpoints[0])<=1e-7 or
                     np.linalg.norm(points[1]-endpoints[1])<=1e-7)):
                matches.append(row['id'])
        return matches[0] if len(matches)==1 else None
    except (ValueError,TypeError,KeyError):return None


def required_constraints_preserved(parent_graph, graph, required, current):
    """Fresh ownership must retain every previous typed source obligation."""
    old={e["id"]:e for e in parent_graph.get("entities",[])}
    new={e["id"]:e for e in graph.get("entities",[])}
    lineage=(graph.get("entity_identity") or {}).get("parent_to_current") or {}
    entity_mapping={}
    source_radius_mapping={}
    finite_line_mapping={}
    node_mapping={}
    for eid,entity in old.items():
        descendants=lineage.get(eid,[]) if lineage else [eid]
        same_type=[child for child in descendants if child in new and new[child].get("type")==entity.get("type")]
        if len(same_type)==1:entity_mapping[eid]=same_type[0]
        elif entity.get('type')=='LINE':
            child=_unique_retained_finite_line(entity,[new[e] for e in same_type])
            if child:
                entity_mapping[eid]=child
                finite_line_mapping[eid]=dict(entity_id=child,
                    method='unique_collinear_finite_descendant_with_retained_outer_endpoint')
        elif entity.get("type")=="ARC":
            previous=[r for r in required if r.get("kind")=="radius" and r.get("entities")==[eid]]
            if len(previous)==1:
                radius=previous[0]
                fresh=[r for r in current if r.get("kind")=="radius" and
                    r.get("record_id")==radius.get("record_id") and r.get("value")==radius.get("value") and
                    r.get("source_arrow_verified") is True and r.get("required") is True and
                    r.get("enforcement")=="exact" and r.get("nominal_source")=="source_ocr" and
                    r.get("source") in {"ocr_local_binding","ocr_api_binding"} and
                    len(r.get("entities") or [])==1]
                if len(fresh)==1 and fresh[0]["entities"][0] in same_type:
                    entity_mapping[eid]=fresh[0]["entities"][0]
                    source_radius_mapping[eid]=dict(entity_id=entity_mapping[eid],
                        record_id=radius.get("record_id"),method="fresh_independent_source_arrow_ownership")
        if descendants and all(child in new for child in descendants):
            contacts=(("start",entity_mapping[eid]),("end",entity_mapping[eid])) if eid in source_radius_mapping or eid in finite_line_mapping else \
                (("start",descendants[0]),("end",descendants[-1]))
            for endpoint,child in contacts:
                if eid in finite_line_mapping and math.dist(entity[endpoint],new[child][endpoint])>1e-7:
                    # A trimmed corner is a new contact, not the old virtual
                    # intersection. Endpoint dimensions need fresh anchors.
                    continue
                node=entity.get(endpoint+"_node");mapped=new[child].get(endpoint+"_node")
                if node is not None and mapped is not None:
                    node_mapping[node]=mapped if node not in node_mapping or node_mapping[node]==mapped else None
    def remap(value,key=None):
        if isinstance(value,list):return [remap(item,key) for item in value]
        if isinstance(value,dict):return {name:remap(item,name) for name,item in value.items()}
        if key in {"entity_id","entities"}:return entity_mapping.get(value,"unmapped")
        if key in {"node_id","nodes"}:return node_mapping.get(value,"unmapped")
        return value
    def trusted_source(row):
        return row.get("source") in {"ocr_local_binding","ocr_api_binding","source_geometry"}
    failures=[];replacements=[]
    for row in required:
        targets=[]
        for eid in row.get("entities",[]):
            descendants=lineage.get(eid,[]) if lineage else [eid]
            same_type=([entity_mapping[eid]] if eid in source_radius_mapping or eid in finite_line_mapping else
                [child for child in descendants if child in new and new[child].get("type")==old.get(eid,{}).get("type")])
            if len(same_type)!=1:
                targets=[];break
            targets.append(same_type[0])
        semantic=("kind","record_id","value","reference_axis","angle_mode","source_station_relation",
                  "required","enforcement","nominal_source","tolerance","source_dimension_semantics")
        def same_objects(fresh):
            # Typed anchors, not this membership list, define the signed
            # measurement order. Traversal renumbering may reorder the list
            # while preserving both anchors and their direction exactly.
            if row.get("kind") in {"distance", "distance_x", "distance_y"} and row.get("anchors"):
                members=fresh.get("entities") or []
                return len(members)==len(targets) and set(members)==set(targets)
            return fresh.get("entities")==targets
        mapped_nodes=remap(row.get("nodes",[]),"nodes")
        if row.get("kind")=="tangent" and len(targets)==2 and len(row.get("nodes") or [])==1:
            # Map this pair's incident endpoints, rather than a global node
            # map that can be ambiguous at a newly split neighboring contact.
            node=row['nodes'][0];incident=[]
            for eid,target in zip(row['entities'],targets):
                sides=[side for side in ('start','end') if old[eid].get(side+'_node')==node]
                if len(sides)==1:incident.append(new[target].get(sides[0]+'_node'))
            if len(incident)==2 and incident[0] is not None and incident[0]==incident[1]:mapped_nodes=[incident[0]]
        matches=[fresh for fresh in current if targets and same_objects(fresh) and
                 all(fresh.get(k)==row.get(k) for k in semantic) and
                 (not trusted_source(row) or trusted_source(fresh)) and
                 (not row.get("anchors") or fresh.get("anchors")==remap(row["anchors"])) and
                 (not row.get("nodes") or fresh.get("nodes")==mapped_nodes)]
        if not matches:
            from .source_relation_replacement import inferred_contact_path
            replacement=inferred_contact_path(parent_graph,graph,row,entity_mapping,current)
            if replacement:
                replacements.append(replacement)
                continue
            failures.append({"constraint_id":row.get("id"),"record_id":row.get("record_id"),
                             "kind":row.get("kind"),"reason":"old_typed_source_obligation_not_freshly_retained"})
    return {"passed":not failures,"required_count":len(required),"failures":failures,
            "inferred_relation_replacements":replacements,
            "source_arrow_radius_identity_mapping":source_radius_mapping,
            "retained_finite_line_identity_mapping":finite_line_mapping}


def line_seed_problem_graphs(graph):
    """Recover the immutable optimization domain and original ink comparator."""
    proof=(graph.get("source_evidence") or {}).get("source_line_initial_guess") or {}
    if not proof:
        from .source_state import resumed_files
        resumed=resumed_files(graph)
        return (resumed["domain.json"],resumed["parent.json"]) if resumed else (graph,graph)
    directory=Path(proof["artifact_directory"])
    def read(name,digest):
        raw=(directory/name).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError("line_initial_guess_snapshot_changed")
        return json.loads(raw)
    domain=read("initial-topology.json",proof["original_domain_sha256"])
    parent=read("parent-topology.json",proof["original_parent_sha256"])
    identity=lambda e:tuple(e.get(k) for k in ("id","type","start_node","end_node","clockwise"))
    if ([identity(e) for e in domain["entities"]]!=[identity(e) for e in graph["entities"]] or
            domain.get("source_sha256")!=graph.get("source_sha256") or domain.get("units")!=graph.get("units")):
        raise ValueError("line_initial_guess_domain_identity_changed")
    for entity in parent["entities"]:entity.pop("radius_binding",None)
    return domain,parent


def retain_line_seed_bindings(image_path,document,baseline,graph,bindings,inventory,output,*,allow_partial_resume=False):
    """Recheck established seed obligations before further binding/optimization."""
    proof=(graph.get('source_evidence') or {}).get('source_line_initial_guess') or {}
    if not proof:
        from .source_state import resumed_files
        resumed=resumed_files(graph)
        if not resumed:return bindings,inventory,{'status':'not_applicable'}
        from .source_radius_hypotheses import carried_binding_certificate,merge_carried_bindings
        carry=carried_binding_certificate(image_path,document,baseline,resumed['domain.json'],graph,
            resumed['inventory.json'],resumed['bindings.json']['constraints'],inventory,bindings,resumed['solution.json'])
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        (output/'resumed-known-source-certificate.json').write_text(json.dumps(carry,ensure_ascii=False,indent=2),encoding='utf8')
        partial=None
        if carry.get('passed') is not True:
            known=resumed['bindings.json']['constraints']
            checks=carry.get('checks',[])
            # A stale/mismatched input or aborted certificate is not a valid
            # partial audit. Only a complete per-obligation recheck permits
            # continuing the ordinary reconstruction, never checkpoint reuse.
            if (not allow_partial_resume or carry.get('reason') is not None or
                    len(checks)!=len(known) or
                    {r.get('constraint_id') for r in checks}!={r['id'] for r in known}):
                raise ValueError('resumed_known_source_obligation_not_retained')
            verified_ids={r['constraint_id'] for r in checks if r.get('passed') is True}
            verified=[r for r in known if r['id'] in verified_ids]
            partial={'status':'resumed_source_subset_retained_pending_repair',
                'all_previous_obligations_recertified':False,
                'previous_constraint_count':len(known),'verified_previous_constraint_count':len(verified),
                'pending_previous_constraints':[copy.deepcopy(r) for r in known if r['id'] not in verified_ids],
                'failed_source_checks':[copy.deepcopy(r) for r in checks if r.get('passed') is not True],
                'complete_checkpoint_reuse_permitted':False,'publication_approved':False,
                'current_API_reconfirmed':False,'ground_truth_used':False}
            # Produce a genuine certificate for the verified subset; never
            # change passed=False on the original complete certificate.
            carry=carried_binding_certificate(image_path,document,baseline,resumed['domain.json'],graph,
                resumed['inventory.json'],verified,inventory,bindings,resumed['solution.json'])
            if carry.get('passed') is not True:raise ValueError('resumed_verified_subset_recheck_failed')
            (output/'resumed-verified-subset-certificate.json').write_text(
                json.dumps(carry,ensure_ascii=False,indent=2),encoding='utf8')
            (output/'resumed-pending-source-repair.json').write_text(
                json.dumps(partial,ensure_ascii=False,indent=2),encoding='utf8')
        updated,inventory=merge_carried_bindings(resumed['inventory.json'],inventory,bindings,carry)
        from .constraint_binding import radius_binding_coverage
        bound={row.get('record_id') for row in updated['constraints']}-{None}
        counts=updated.setdefault('counts',{})
        counts.update(constraints=len(updated['constraints']),bound_source_records=len(bound),
            bound_angle_records=sum(row['kind']=='angle' for row in updated['constraints']),
            carried_source_constraints=len(carry['constraints']))
        if 'recognized_dimensions' in counts:
            counts['unbound_dimensions']=counts['recognized_dimensions']-len(bound)
        updated['radius_binding_coverage']=radius_binding_coverage(inventory,graph,updated['constraints'],updated['bindings'])
        inventory['carried_resumed_source_binding_certificate']=carry
        updated['inventory_artifact']=str(output/'binding-candidates.json')
        if partial is not None:
            updated['pending_resumed_source_obligations']=copy.deepcopy(partial)
        for name,value in [('binding-candidates.json',inventory),('constraint-bindings.json',updated)]:
            (output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
        if partial is not None:return updated,inventory,{**partial,'carried_count':len(carry['constraints'])}
        return updated,inventory,{'status':'resumed_source_bindings_retained','current_API_reconfirmed':False,
                                 'carried_count':len(carry['constraints'])}
    from .source_radius_hypotheses import carried_binding_certificate,merge_carried_bindings
    from .constraint_binding import radius_binding_coverage
    from .parametric_pipeline import _constraint_signatures
    directory=Path(proof['artifact_directory']);output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    def read(name,digest):
        raw=(directory/name).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=digest:
            raise ValueError('line_seed_binding_certificate_changed')
        return json.loads(raw)
    domain,_=line_seed_problem_graphs(graph)
    original_inventory=read('initial-binding/binding-candidates.json',proof['original_inventory_sha256'])
    known=read('initial-binding/constraint-bindings.json',proof['original_bindings_sha256'])['constraints']
    solution=read('solve/parametric-solve.json',proof['seed_solution_sha256'])
    carry=carried_binding_certificate(image_path,document,baseline,domain,graph,
        original_inventory,known,inventory,bindings,solution)
    (output/'line-seed-binding-certificate.json').write_text(json.dumps(carry,ensure_ascii=False,indent=2),encoding='utf8')
    if carry.get('passed') is not True:
        raise ValueError('line_seed_known_source_obligation_not_retained')
    updated,inventory=merge_carried_bindings(original_inventory,inventory,bindings,carry)
    if not _constraint_signatures(domain,known)<=_constraint_signatures(graph,updated['constraints']):
        raise ValueError('line_seed_typed_obligation_regression')
    bound={row.get('record_id') for row in updated['constraints']}-{None}
    counts=updated.setdefault('counts',{})
    counts.update(constraints=len(updated['constraints']),bound_source_records=len(bound),
        bound_angle_records=sum(row['kind']=='angle' for row in updated['constraints']),
        carried_source_constraints=len(carry['constraints']))
    if 'recognized_dimensions' in counts:counts['unbound_dimensions']=counts['recognized_dimensions']-len(bound)
    updated['radius_binding_coverage']=radius_binding_coverage(inventory,graph,updated['constraints'],updated['bindings'])
    inventory['carried_line_seed_binding_certificate']=carry
    updated['inventory_artifact']=str(output/'binding-candidates.json')
    for name,value in [('binding-candidates.json',inventory),('constraint-bindings.json',updated)]:
        (output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
    return updated,inventory,{'status':'source_bindings_retained','current_API_reconfirmed':False,
                             'carried_count':len(carry['constraints'])}


def repair_source_owned_line_seed(image_path, document, baseline, candidate, output, *,
                                  parent_graph, parent_constraints, source_validation,
                                  budget_seconds=240.):
    """Solve an unsafe initial fit; return only a fully source-valid new seed.

    Fresh formal binding is still required by the caller. The original graph
    defines the optimization domain throughout; a soft RMS iterate can never
    be returned unless the strict solver and both unchanged ink gates pass.
    """
    from .constraint_binding import analyze_constraint_bindings
    from .parametric_pipeline import (_solver_source_observation,_solved_source_validation,
                                      _topology_source_validation)
    from .parametric_solver import solve_parametric, _strict_ocr_axis_angle, _strict_source_nominal
    from .source_radius_hypotheses import graph_from_solution
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    graph=copy.deepcopy(candidate["graph"])
    admission=line_seed_admission(graph,source_validation)
    started=time.monotonic();deadline=started+min(240.,max(0.,float(budget_seconds)))
    receipt={"schema_version":"source-line-initial-guess-v1","status":"not_applicable",
             "admission":admission,"accepted":False,"publication_approved":False,
             "formal_binding_required":True,"ground_truth_used":False,
             "original_initial_source_validation":copy.deepcopy(source_validation),
             "maximum_wall_seconds":min(240.,max(0.,float(budget_seconds))),
             "original_optimization_domain":"input_candidate_graph",
             "original_mask_and_source_thresholds_unchanged":True}
    def write(name,value):
        (output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf8")
    def finish(reason,status="rejected",updated=None):
        receipt.update(status=status,reason=reason,accepted=updated is not None,
                       elapsed_seconds=time.monotonic()-started)
        write("source-line-initial-guess.json",receipt)
        return updated,receipt
    if not admission["eligible"]:return finish(admission["reason"],"not_applicable")
    if not parent_constraints or not parent_graph:
        return finish("certified_parent_source_obligations_unavailable")
    if deadline<=time.monotonic():return finish("shared_topology_budget_exhausted")
    receipt["status"]="running";write("source-line-initial-guess.json",receipt)
    write("initial-topology.json",graph)
    write("parent-topology.json",parent_graph)
    try:
        bindings=analyze_constraint_bindings(image_path,document,baseline,graph,output/"initial-binding",use_api=False)
        current=bindings.get("constraints") or []
        fresh_angle=[row for row in current if row.get("record_id")==admission["record_id"] and
            row.get("kind")=="angle" and row.get("entities")==[admission["entity_id"]] and
            row.get("value")==admission["value"] and row.get("reference_axis")==admission["reference_axis"] and
            (_strict_source_nominal(row) or _strict_ocr_axis_angle(row))]
        if len(fresh_angle)!=1:return finish("new_line_angle_not_independently_strictly_bound")
        locks=required_constraints_preserved(parent_graph,graph,parent_constraints,current)
        receipt["known_source_obligations"]=locks
        if not locks["passed"]:return finish("previous_source_obligation_not_preserved")
        remaining=deadline-time.monotonic()
        if remaining<=0:return finish("shared_topology_budget_exhausted_after_binding")
        observation=_solver_source_observation(baseline,graph)
        from .reconstruction_priority import mask_is_hard
        if not observation or (mask_is_hard(baseline) and not observation.get("boundary_error_budget")):
            return finish("original_source_budget_unavailable")
        if observation.get("boundary_error_budget"):
            observation["boundary_error_budget"].update(max_wall_seconds=min(120.,remaining),
                                                        max_objective_evaluations=30000)
        if observation.get("optimization_budget"):
            observation["optimization_budget"].update(max_wall_seconds=min(120.,remaining),max_objective_evaluations=30000)
        if not mask_is_hard(baseline):
            from .source_stroke_prior import source_stroke_observation
            observation['image_stroke_prior'] = source_stroke_observation(image_path,document,baseline,graph)
            receipt['original_image_observation_supplied_to_solver'] = True
        solution=solve_parametric(graph,current,output_dir=output/"solve",source_observation=observation,
                                  source_warm_start=True)
        receipt["solver_status"]=solution.get("status")
        if solution.get("accepted") is not True:return finish("strict_original_problem_solve_failed")
        checks={row.get("id"):row for row in solution.get("constraints") or []}
        if any(checks.get(row.get("id"),{}).get("passed") is not True for row in current):
            return finish("strict_constraint_receipts_incomplete")
        if time.monotonic()>=deadline:return finish("shared_topology_budget_exhausted_after_solve")
        own=_solved_source_validation(image_path,document,baseline,graph,solution["entities"])
        parent=_solved_source_validation(image_path,document,baseline,parent_graph,solution["entities"])
        # Parent entity IDs can change after a split. The old typed radius locks
        # were freshly mapped above, so evaluate its ink/raw-mask comparator
        # without misusing its old display IDs as construction certificates.
        if parent.get("constructed_radius_preservation"):
            comparator=copy.deepcopy(parent_graph)
            for entity in comparator["entities"]:entity.pop("radius_binding",None)
            parent=_solved_source_validation(image_path,document,baseline,comparator,solution["entities"])
        write("initial-candidate-solved-source-validation.json",own)
        write("parent-solved-source-validation.json",parent)
        receipt["original_candidate_source_passed"]=own.get("passed") is True
        receipt["original_parent_source_passed"]=parent.get("passed") is True
        if not own.get("passed") or not parent.get("passed"):
            return finish("original_post_solve_source_gate_failed")
        solved=graph_from_solution(graph,baseline,solution)
        solved["source_evidence"].pop("source_radius_hypothesis",None)
        solved["source_evidence"]["proposal_stroke_support"]=copy.deepcopy(parent["after"])
        solved["source_evidence"]["baseline_stroke_support"]=copy.deepcopy(parent["before"])
        solved["source_evidence"]["source_line_initial_guess"]={
            "record_id":admission["record_id"],"initial_fit_was_not_a_certificate":True,
            "strict_source_solve_passed":True,"fresh_formal_binding_required":True,
            "artifact_directory":str(output.resolve()),
            "original_domain_sha256":hashlib.sha256((output/"initial-topology.json").read_bytes()).hexdigest(),
            "original_parent_sha256":hashlib.sha256((output/"parent-topology.json").read_bytes()).hexdigest(),
            "original_inventory_sha256":hashlib.sha256((output/"initial-binding/binding-candidates.json").read_bytes()).hexdigest(),
            "original_bindings_sha256":hashlib.sha256((output/"initial-binding/constraint-bindings.json").read_bytes()).hexdigest(),
            "seed_solution_sha256":hashlib.sha256((output/"solve/parametric-solve.json").read_bytes()).hexdigest(),
            "ground_truth_used":False}
        validation=_topology_source_validation(image_path,baseline,solved)
        write("materialized-source-validation.json",validation)
        if validation.get("passed") is not True:return finish("materialized_source_validation_failed")
        write("solved-topology.json",solved)
        updated=copy.deepcopy(candidate);updated["graph"]=solved
        updated["source_stroke_support"]=copy.deepcopy(parent["after"])
        updated["source_initial_guess_refinement"]={"artifact":str(output),"formal_binding_required":True}
        return finish("source_valid_initial_solution_requires_fresh_binding","source_valid_seed",updated)
    except InterruptedError:
        raise
    except (ValueError,TypeError,KeyError,OSError,ArithmeticError) as error:
        receipt["failure_type"]=type(error).__name__
        return finish("source_line_initial_guess_failed")
