"""Source topology -> sourced bindings -> numerical constraints -> CAD export.

The initial pixel trace is preserved. A parametric correction has its own
acceptance criteria; its displacement from the mask is diagnostic, not a claim
that the old pixel-fidelity gate passed. No reference geometry is consumed.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import shutil

import cv2
import ezdxf
import numpy as np
from shapely.geometry import Polygon

from .automatic import _verify_dxf_readback
from .vectorize import _sample_entities, assess_fit_quality


CORE = ("drawing.dxf", "preview.svg", "overlay.png", "model.json", "validation.json", "curve-fit.json")


def _write(path, value):
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf8")
    temporary.replace(path)


def _export_chain(image_path, baseline, entities, output_dir, *, topology=False, constraint_validation=None, source_validation=None):
    """Export geometry under an explicit source-topology or constraint basis."""
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    entities=copy.deepcopy(entities)
    if not 2 <= len(entities) <= 1000:
        raise ValueError("Unsupported parametric entity count")
    units=baseline["coordinate_system"]["units"]
    scale=float(baseline["scale"]["pixels_per_mm"]) if units=="mm" else 1.
    origin=baseline["coordinate_system"]["origin_source_px"]
    points,_,_= _sample_entities(entities,max_step_px=.15 if units=="mm" else 1.)
    polygon=Polygon(points)
    gaps=[math.dist(e["end"],entities[(i+1)%len(entities)]["start"]) for i,e in enumerate(entities)]
    if not polygon.is_valid or polygon.area<=0 or max(gaps)>1e-7:
        raise ValueError("Parametric candidate is not a valid closed contour")
    document=ezdxf.new("R2010");document.units=4 if units=="mm" else 0
    document.layers.new("PARAMETRIC_MAIN_PROFILE",dxfattribs={"color":3})
    commands=[]
    for index,entity in enumerate(entities):
        if index==0:commands.append(f'M {entity["start"][0]} {-entity["start"][1]}')
        attributes={"layer":"PARAMETRIC_MAIN_PROFILE"}
        if entity["type"]=="LINE":
            document.modelspace().add_line(entity["start"],entity["end"],dxfattribs=attributes)
            commands.append(f'L {entity["end"][0]} {-entity["end"][1]}')
        elif entity["type"]=="ARC":
            center=entity["center"];radius=entity["radius"]
            a=math.atan2(entity["start"][1]-center[1],entity["start"][0]-center[0])
            b=math.atan2(entity["end"][1]-center[1],entity["end"][0]-center[0])
            clockwise=entity["clockwise"]
            sweep=(a-b)%(2*math.pi) if clockwise else (b-a)%(2*math.pi)
            document.modelspace().add_arc(center,radius,math.degrees(b if clockwise else a)%360,
                                         math.degrees(a if clockwise else b)%360,dxfattribs=attributes)
            commands.append(f'A {radius} {radius} 0 {int(sweep>math.pi)} {int(clockwise)} {entity["end"][0]} {-entity["end"][1]}')
        else:raise ValueError("Unsupported parametric primitive")
    document.saveas(output/"drawing.dxf")
    readback=_verify_dxf_readback(ezdxf.readfile(output/"drawing.dxf"),entities,expected_units=document.units)
    if not readback["passed"]:raise ValueError("Parametric DXF readback failed")
    x0,y0,x1,y1=polygon.bounds;pad=max(x1-x0,y1-y0)*.05
    svg=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0-pad} {-y1-pad} {x1-x0+2*pad} {y1-y0+2*pad}"><title>Source-bound parametric contour; partial constraints only</title><path d="{" ".join(commands)} Z" fill="#dbe9e2" stroke="#087f72" stroke-width="1.6" vector-effect="non-scaling-stroke"/></svg>'
    (output/"preview.svg").write_text(svg,encoding="utf8")
    pixel_points=np.column_stack((points[:,0]*scale+origin[0],origin[1]-points[:,1]*scale))
    source=cv2.imdecode(np.fromfile(str(image_path),np.uint8),cv2.IMREAD_COLOR)
    if source is None:raise ValueError("Source image unavailable during parametric export")
    display_scale=min(1.,2200/max(source.shape[:2]))
    display=cv2.resize(source,None,fx=display_scale,fy=display_scale)
    cv2.polylines(display,[np.rint(pixel_points*display_scale).astype(np.int32)],True,(0,0,220),2)
    cv2.imencode(".png",display)[1].tofile(str(output/"overlay.png"))
    pixel_entities=[]
    def px(point):return [point[0]*scale+origin[0],origin[1]-point[1]*scale]
    for entity in entities:
        mapped={**entity,"start":px(entity["start"]),"end":px(entity["end"])}
        if entity["type"]=="ARC":mapped.update(center=px(entity["center"]),radius=entity["radius"]*scale,clockwise=not entity["clockwise"])
        pixel_entities.append(mapped)
    raw=baseline["extraction"].get("raw_polyline_px") or baseline["polyline_px"]
    deviation=assess_fit_quality(raw,pixel_entities)
    deviation.update(method="source-topology-mask-displacement-diagnostic-v1" if topology else "parametric-mask-displacement-diagnostic-v1",passed=None,
                     used_as_acceptance_gate=False,fallback_used=False,
                     scope="Mask displacement after source-bound constraints; the initial pixel-fidelity gate is preserved separately, not relaxed or relabelled.")
    validation={"passed":True,"entity_count":len(entities),"scaled_mm":units=="mm",
                "material_connectivity":baseline.get("validation",{}).get("material_connectivity"),
                "complete_material_exterior":baseline.get("complete_material_exterior",True),
                "max_gap_mm":max(gaps) if units=="mm" else None,"self_intersection":False,
                "dxf_readback":readback,"dimensions_verified":False,"reference_verified":False,
                "engineering_certified":False,"radius_bindings":[],"curve_fit":deviation,
                "constraint_validation":constraint_validation or {"status":"not_run","constraint_subset_satisfied":False},
                "source_topology_validation":source_validation,
                "validation_basis":"source_stroke_supported_topology_and_closed_geometry" if topology else "accepted_numerical_constraint_subset_and_closed_geometry",
                "source_mask_fidelity_used_as_acceptance_gate":False,
                "issues":[],"meaning":"Partial source-bound parametric draft; geometry and enforced constraints are checked separately from reference accuracy."}
    model=copy.deepcopy(baseline)
    model.update(entities=entities,validation=validation,curve_fit=deviation,
                 initial_curve_fit=baseline.get("curve_fit"),automatic_completion=baseline.get("complete_material_exterior",True),
                 algorithm_version="source-topology-draft-v1" if topology else "source-topology-bound-parametric-v1",
                 completion_class=("source_topology_draft" if topology else "partial_parametric_draft") if baseline.get("complete_material_exterior",True) else "incomplete_material_exterior_draft",
                 fitted_polyline_px=pixel_points.tolist(),
                 bounds={"min_x":x0,"min_y":y0,"max_x":x1,"max_y":y1},
                 scope="Approximate source topology plus explicitly sourced numerical constraints; unbound dimensions remain unverified.")
    _write(output/"model.json",model);_write(output/"validation.json",validation);_write(output/"curve-fit.json",deviation)
    return model


def export_parametric(image_path, baseline, solution, output_dir):
    """Only a solver-accepted constraint subset uses the parametric basis."""
    if solution.get("accepted") is not True:
        raise ValueError("Only an accepted numerical solution may be exported")
    if solution.get("units",baseline["coordinate_system"]["units"])!=baseline["coordinate_system"]["units"]:
        raise ValueError("Solution and source units differ")
    return _export_chain(image_path,baseline,solution["entities"],output_dir,constraint_validation=solution.get("validation"))


def _topology_source_validation(image_path, baseline, graph):
    evidence=graph.get("source_evidence") or {}
    before=evidence.get("baseline_stroke_support") or {}
    after=evidence.get("proposal_stroke_support") or {}
    reasons=[]
    if graph.get("source_sha256")!=hashlib.sha256(Path(image_path).read_bytes()).hexdigest():reasons.append("source_hash_mismatch")
    if graph.get("ground_truth_used") is not False:reasons.append("source_provenance_not_declared")
    if graph.get("units")!=baseline["coordinate_system"]["units"]:reasons.append("units_mismatch")
    coordinate_check={"passed":False,"tolerance_source_px":1e-5,"maximum_source_mapping_error_px":None}
    try:
        base_system=baseline["coordinate_system"]
        graph_system=graph["coordinate_system"]
        if base_system.get("units") not in {"mm","pixel"}:reasons.append("units_mismatch")
        if graph_system.get("units")!=base_system["units"] or graph.get("units")!=graph_system.get("units"):
            reasons.append("coordinate_units_mismatch")
        origin=np.asarray(base_system["origin_source_px"],float)
        graph_origin=np.asarray(graph_system["origin_source_px"],float)
        if origin.shape!=(2,) or graph_origin.shape!=(2,) or not np.isfinite(origin).all() or not np.isfinite(graph_origin).all():
            reasons.append("invalid_coordinate_origin")
        elif not np.allclose(origin,graph_origin,rtol=0,atol=1e-5):
            reasons.append("coordinate_origin_mismatch")
        if graph_system.get("x") not in {None,"image right"} or graph_system.get("y") not in {None,"image up"}:
            reasons.append("coordinate_axis_mismatch")
        scale=1.
        if base_system["units"]=="mm":
            scale=float(baseline["scale"]["pixels_per_mm"])
            if baseline["scale"].get("status")!="resolved" or not math.isfinite(scale) or scale<=0:
                reasons.append("invalid_source_scale")
        nodes=graph["nodes"]
        if not isinstance(nodes,list) or not nodes:
            reasons.append("missing_source_nodes")
        else:
            xy=np.asarray([[node["x"],node["y"]] for node in nodes],float)
            pixels=np.asarray([node["source_px"] for node in nodes],float)
            if xy.shape!=pixels.shape or xy.shape!=(len(nodes),2) or not np.isfinite(xy).all() or not np.isfinite(pixels).all():
                reasons.append("invalid_source_node_coordinates")
            else:
                expected=xy*[scale,-scale]+origin
                maximum=float(np.max(np.abs(expected-pixels)))
                coordinate_check["maximum_source_mapping_error_px"]=maximum if math.isfinite(maximum) else None
                if not math.isfinite(maximum) or maximum>1e-5:reasons.append("node_source_mapping_mismatch")
            indexed={node["id"]:node for node in nodes}
            for entity in graph["entities"]:
                for endpoint,node_key in (("start","start_node"),("end","end_node")):
                    node=indexed[entity[node_key]]
                    if not np.allclose(entity[endpoint],[node["x"],node["y"]],rtol=0,atol=1e-7):
                        reasons.append("entity_node_coordinate_mismatch")
                        break
        coordinate_check["passed"]=not any(reason in reasons for reason in (
            "units_mismatch","coordinate_units_mismatch","invalid_coordinate_origin","coordinate_origin_mismatch",
            "coordinate_axis_mismatch","invalid_source_scale","missing_source_nodes","invalid_source_node_coordinates",
            "node_source_mapping_mismatch","entity_node_coordinate_mismatch"))
    except (KeyError,TypeError,ValueError,IndexError):
        reasons.append("invalid_source_coordinate_mapping")
    if not all((graph.get("validation") or {}).get(k) is True for k in ("closed","connected","simple","ordered_entity_cycle")):
        reasons.append("topology_validation_failed")
    try:
        metrics=[float(before["edge_supported_fraction"]),float(after["edge_supported_fraction"]),
                 float(before["p90_edge_distance_px"]),float(after["p90_edge_distance_px"]),float(graph["source_grid_pitch_px"])]
        if not all(math.isfinite(n) for n in metrics) or not 0<=metrics[0]<=1 or not 0<metrics[1]<=1 or min(metrics[2:])<0:
            reasons.append("invalid_source_support_metrics")
        # Same declared source-stroke comparison as topology candidate selection;
        # no reliance on the previous strict mask-fit acceptance flag.
        elif metrics[1]<metrics[0]-.035 or metrics[3]>metrics[2]+metrics[4]:
            reasons.append("source_stroke_support_degraded")
    except (KeyError,TypeError,ValueError):reasons.append("missing_source_support_metrics")
    return {"passed":not reasons,"reasons":list(dict.fromkeys(reasons)),"source_evidence":evidence,"source_sha256":graph.get("source_sha256"),
            "coordinate_mapping":coordinate_check,
            "acceptance_basis":"source_hash_stroke_support_closed_simple_geometry_and_DXF_readback",
            "dimensions_verified":False,"reference_verified":False,"mask_fidelity_gate_used":False}


def _topology_edit_stage(image_path, document, baseline, selected, bundle, output, *,
                         editor_provider=None, evaluator_provider=None, use_api=False):
    """Propose, execute and independently evaluate bounded local topology edits."""
    from .planning_provider import evaluate_candidates
    from .topology_editing import execute_topology_edits, propose_annotation_arc_edits

    editor_receipt={"status":"disabled" if not use_api else "not_configured",
                    "reason":"topology_edit_disabled" if not use_api else "topology_edit_provider_missing",
                    "schema_success":False,"network_requests":0,"operations":[],"confidence":"abstain",
                    "ground_truth_sent":False,"coordinates_sent":False}
    evaluator_receipt={"status":"disabled" if not use_api else "not_configured",
                       "reason":"topology_evaluation_disabled" if not use_api else "topology_evaluator_missing",
                       "schema_success":False,"network_requests":0,"selected_candidate_id":None,
                       "decision":"abstain","confidence":"abstain","ground_truth_sent":False,
                       "coordinates_sent":False}
    execution={"schema_version":"local-topology-edit-execution-v1","base_candidate_id":selected.get("id"),
               "proposed":0,"accepted_candidates":0,"operations":[],"ground_truth_used":False}
    edited=[]
    overlay=Path(selected.get("overlay_path") or output/"topology-overlay.png")
    if not overlay.is_absolute():overlay=output/overlay
    if use_api and editor_provider is not None and overlay.is_file():
        try:
            editor_receipt=editor_provider.propose(image_path,overlay,selected,bundle.get("annotation_inventory",[]))
        except Exception:
            editor_receipt.update(status="failed",reason="topology_edit_provider_error",
                                  schema_success=False,network_requests=None)
    agent_operations=(editor_receipt.get("operations",[])
                      if editor_receipt.get("schema_success") is True else [])
    local_annotation_operations=propose_annotation_arc_edits(
        selected.get("graph") or {},bundle.get("annotation_inventory",[]),limit=4)
    operations=[];occupied=set();seen=set()
    for operation in [*local_annotation_operations,*agent_operations]:
        ids=tuple(operation.get("entity_ids") or [])
        signature=(operation.get("action"),ids,operation.get("record_id"))
        if signature in seen or occupied.intersection(ids):continue
        operations.append(operation);seen.add(signature);occupied.update(ids)
        if len(operations)>=5:break
    if operations:
        try:
            edited,execution=execute_topology_edits(image_path,document,baseline,selected,bundle,operations,output)
        except Exception:
            execution.update(proposed=len(operations),status="failed",reason="local_topology_edit_execution_error")
    pool=[selected,*edited]
    local=evaluate_candidates(pool,max_candidates=5)
    admissible=set(local.get("admissible_candidate_ids",[]))
    if edited and use_api and evaluator_provider is not None:
        try:
            evaluator_receipt=evaluator_provider.select(
                image_path,pool,selected["id"],bundle.get("annotation_inventory",[]))
        except Exception:
            evaluator_receipt.update(status="failed",reason="topology_evaluation_provider_error",
                                     schema_success=False,network_requests=None)
    proposed_id=(evaluator_receipt.get("selected_candidate_id")
                 if evaluator_receipt.get("schema_success") is True else None)
    by_id={row["id"]:row for row in pool}
    evaluated={row["candidate_id"]:row for row in local.get("evaluated",[]) if row.get("admissible")}
    gate={"accepted":False,"reason":"evaluator_did_not_select_an_edit",
          "proposed_candidate_id":proposed_id,"base_candidate_id":selected.get("id")}
    final=selected
    if proposed_id==selected.get("id") and proposed_id in admissible:
        gate.update(reason="evaluator_preserved_base")
    elif proposed_id in admissible and proposed_id in by_id:
        base_eval=evaluated.get(selected.get("id"),{});edit_eval=evaluated.get(proposed_id,{})
        base_metrics=base_eval.get("metrics",{});edit_metrics=edit_eval.get("metrics",{})
        edit_execution=((by_id[proposed_id].get("graph") or {}).get("source_evidence") or {}).get("topology_edit") or {}
        edit_actions=[edit_execution.get("action")]
        edit_actions.extend(row.get("action") for row in edit_execution.get("operations",[]) if isinstance(row,dict))
        annotation_guided="refit_chain_as_annotated_arc" in edit_actions
        count_improved=edit_metrics.get("entity_count",10**9) < base_metrics.get("entity_count",0)
        count_preserved_for_annotation=bool(
            annotation_guided and edit_metrics.get("entity_count",10**9) <= base_metrics.get("entity_count",0))
        accepted=bool(
            edit_eval.get("score") is not None and base_eval.get("score") is not None and
            edit_eval["score"] >= base_eval["score"]-.02 and
            edit_metrics.get("source_boundary_support",0.) >= base_metrics.get("source_boundary_support",0.)-.02 and
            edit_metrics.get("unsupported_primitive_count",10**9) <= base_metrics.get("unsupported_primitive_count",0)+1 and
            (count_improved or count_preserved_for_annotation)
        )
        gate={"accepted":accepted,
              "reason":None if accepted else "edited_candidate_failed_local_improvement_gate",
              "proposed_candidate_id":proposed_id,"base_candidate_id":selected.get("id"),
              "minimum_local_score":round(float(base_eval.get("score",0.))-.02,9),
              "minimum_source_boundary_support":max(0.,float(base_metrics.get("source_boundary_support",0.))-.02),
              "maximum_unsupported_primitive_count":int(base_metrics.get("unsupported_primitive_count",0))+1,
              "must_reduce_entity_count":not annotation_guided,
              "annotation_guided_primitive_correction":annotation_guided}
        if accepted:final=by_id[proposed_id]
    record={"schema_version":"multimodal-local-topology-edit-stage-v1","status":"completed",
            "base_candidate_id":selected.get("id"),"final_candidate_id":final.get("id"),
            "editor":editor_receipt,"execution":execution,"local_evaluation":local,
            "evaluator":evaluator_receipt,"acceptance_gate":gate,
            "local_annotation_operations":local_annotation_operations,
            "executed_operation_count":len(operations),
            "edited_candidate_ids":[row["id"] for row in edited],"ground_truth_used":False,
            "scope":"Agent proposes entity-ID edits; local geometry executes and validates; a separate evaluator selects; GT is unavailable."}
    _write(output/"topology-edit-proposals.json",record)
    if edited:
        bundle["candidates"].extend(edited)
        bundle["candidate_count"]=len(bundle["candidates"])
        bundle.setdefault("generation",{})["local_edit_candidate_count"]=len(edited)
        _write(output/"topology-candidates.json",bundle)
    return final,record


def _plan_topology(image_path, document, baseline, base_graph, output_dir, *, planner_provider=None,
                   editor_provider=None, evaluator_provider=None, use_api=False):
    """Choose one bounded source-only topology candidate and persist the audit.

    Annotation evidence controls whether simplification is allowed.  With no
    detected source leader at all, the existing topology is retained instead
    of treating a low primitive count as evidence.  The online planner can only
    select candidates that passed the deterministic local evaluator.
    """
    from .planning_provider import evaluate_candidates
    from .topology_candidates import generate_topology_candidates, materialize_selected_candidate

    output=Path(output_dir)
    bundle=generate_topology_candidates(image_path,document,baseline,base_graph,output,max_candidates=5)
    candidates=bundle["candidates"]
    by_id={row["id"]:row for row in candidates}
    local=evaluate_candidates(candidates,max_candidates=5)
    admissible=set(local["admissible_candidate_ids"])
    base_id=candidates[0]["id"]
    leader_rows=[row for row in bundle.get("annotation_inventory",[]) if isinstance(row.get("leader"),dict)]
    annotation_evidence=bool(leader_rows)
    receipt={"status":"disabled" if not use_api else "not_configured",
             "reason":"online_planning_disabled" if not use_api else "planning_provider_missing",
             "protocol":"chat-source-topology-planning-v1","network_requests":0,
             "http_success":False,"schema_success":False,"image_sent":False,
             "ground_truth_sent":False,"coordinates_sent":False,"dimensions_generated":False,
             "selected_candidate_id":None,"relation_ids":[],"binding_candidate_ids":[],
             "observed_evidence_ids":[],"confidence":"abstain",
             "rationale_code":"insufficient_source_evidence","local_evaluation":local}
    if use_api and not annotation_evidence:
        receipt.update(status="skipped",reason="no_source_annotation_leader_evidence")
    elif use_api and planner_provider is not None:
        try:
            receipt=planner_provider.select(image_path,candidates)
        except Exception:
            # Provider exceptions are not serialized because they can contain
            # headers.  Source-only candidates and local evaluation survive.
            receipt.update(status="failed",reason="planning_provider_error",
                           network_requests=None,http_success=None,schema_success=False)

    online_id=receipt.get("selected_candidate_id") if receipt.get("schema_success") is True else None
    local_id=local.get("recommended_candidate_id")
    evaluations={row["candidate_id"]:row for row in local.get("evaluated",[]) if row.get("admissible")}
    online_gate={"accepted":False,"reason":"online_selection_unavailable"}
    if online_id in admissible and local_id in admissible:
        proposed=evaluations[online_id];recommended=evaluations[local_id]
        pm,rm=proposed["metrics"],recommended["metrics"]
        maximum_entities=max(rm["entity_count"]+2,int(math.ceil(1.15*rm["entity_count"])))
        accepted=(pm["entity_count"]<=maximum_entities and
                  pm["unsupported_primitive_count"]<=rm["unsupported_primitive_count"]+2 and
                  proposed["score"]>=recommended["score"]-.02)
        online_gate={"accepted":accepted,
                     "reason":None if accepted else "online_candidate_exceeds_local_compactness_budget",
                     "candidate_id":online_id,"local_recommended_candidate_id":local_id,
                     "maximum_entity_count":maximum_entities,
                     "maximum_unsupported_primitive_count":rm["unsupported_primitive_count"]+2,
                     "minimum_local_score":round(recommended["score"]-.02,9)}
    if not annotation_evidence:
        # The candidate generator recomputes diagnostic stroke scores.  Those
        # scores are not a reason to replace the already validated base graph
        # when no annotation points to a different object decomposition.
        selected_id=base_id
        source="base_topology_no_annotation_leader_evidence"
        selected=by_id[selected_id]
        _write(output/"topology.json",base_graph)
        materialization={"status":"preserved_base_topology","candidate_id":selected_id,
                         "topology_path":str(output/"topology.json"),
                         "overlay_path":str(output/"topology-overlay.png") if (output/"topology-overlay.png").is_file() else None,
                         "ground_truth_used":False,"reference_accuracy_verified":False}
        selected_graph=base_graph
    elif online_id in admissible and online_gate["accepted"]:
        selected_id=online_id
        source="online_planning_agent"
    else:
        selected_id=local_id
        source=("local_evaluator_after_online_rejection" if online_id in admissible else
                "local_evaluator_after_online_abstention") if use_api else "local_evaluator"
    if annotation_evidence and selected_id not in admissible:
        # The preserved base may itself have failed a new local hard gate.  Do
        # not publish another candidate merely because it has fewer objects.
        raise ValueError("No admissible source topology candidate")
    edit_stage=None
    if annotation_evidence:
        selected=by_id[selected_id]
        selected,edit_stage=_topology_edit_stage(image_path,document,baseline,selected,bundle,output,
                                                  editor_provider=editor_provider,
                                                  evaluator_provider=evaluator_provider,use_api=use_api)
        selected_id=selected["id"]
        if edit_stage.get("acceptance_gate",{}).get("accepted"):
            source="multimodal_local_topology_edit"
        materialization=materialize_selected_candidate(selected,bundle,output)
        selected_graph=selected["graph"]
    elif use_api and editor_provider is not None:
        # Even without a detected leader, a multimodal editor may diagnose a
        # visible micro-segment.  The edit still needs local execution and an
        # independent evaluator; abstention preserves the exact base graph.
        selected=by_id[base_id]
        edited_selected,edit_stage=_topology_edit_stage(image_path,document,baseline,selected,bundle,output,
                                                         editor_provider=editor_provider,
                                                         evaluator_provider=evaluator_provider,use_api=use_api)
        if edit_stage.get("acceptance_gate",{}).get("accepted"):
            selected=edited_selected;selected_id=selected["id"]
            source="multimodal_local_topology_edit_without_annotation_leader"
            materialization=materialize_selected_candidate(selected,bundle,output)
            selected_graph=selected["graph"]
    plan={"schema_version":"source-topology-plan-v1","status":"selected",
          "selected_candidate_id":selected_id,"selection_source":source,
          "annotation_evidence_available":annotation_evidence,
          "source_annotation_leader_count":len(leader_rows),
          "candidate_count":len(candidates),"local_evaluation":local,
           "provider":receipt,"online_selection_gate":online_gate,"topology_editing":edit_stage,
          "selected_entity_counts":selected.get("entity_counts",{}),
          "selected_annotation_coverage":selected.get("annotation_coverage"),
          "selected_unsupported_primitive_count":selected.get("unsupported_primitive_count"),
          "materialization":materialization,"ground_truth_used":False,
          "dimensions_solved":False,"reference_accuracy_verified":False,
          "scope":"Source-only topology planning. Candidate selection does not certify dimensions or GT accuracy."}
    _write(output/"topology-plan.json",plan)
    return selected_graph,plan


def refine_parametric(image_path, document, baseline, output_dir, *, provider=None, planner_provider=None,
                      editor_provider=None, evaluator_provider=None, use_api=False, progress=None):
    from .topology import build_topology
    from .constraint_binding import analyze_constraint_bindings
    from .parametric_solver import solve_parametric
    output=Path(output_dir)
    current=baseline
    emit=progress or (lambda stage,message:None)
    stage={"status":"running","accepted":False,"ground_truth_used":False,"all_dimensions_verified":False,
           "geometry_updated_by_api":False,"dimensions_updated_by_api":False,
           "provider":{"status":"not_invoked","network_requests":0},"stages":[]}
    def checkpoint(name,message):
        stage["stage"]=name;stage["stages"].append(name)
        _write(output/"parametric-stage.json",stage)
        emit(name,message)
    def publish(candidate_dir,updated,*,kind,api_geometry=False,api_dimensions=False):
        """Rollback targets the last complete draft; baseline-* stays immutable."""
        is_parametric=kind=="parametric"
        published_stage={**copy.deepcopy(stage),"status":"completed" if is_parametric else "source_topology_exported",
                         "accepted":is_parametric,"topology_exported":stage.get("topology_exported",False) or not is_parametric,
                         "geometry_updated_by_api":bool(is_parametric and api_geometry),
                         "dimensions_updated_by_api":bool(is_parametric and api_dimensions),
                         "stage":kind+"_export","stages":[*stage["stages"],kind+"_publish",kind+"_export"]}
        published_stage.pop("publication",None)
        updated["parameterization"]=published_stage
        updated["parameterization_snapshot_scope"]="Export-time geometry metadata; parametric-stage.json and constraint-bindings.json contain the latest stage/online receipts."
        _write(candidate_dir/"model.json",updated)
        def hashes(directory,prefix=""):
            return {name:hashlib.sha256((directory/(prefix+name)).read_bytes()).hexdigest() for name in CORE}
        for name in CORE:shutil.copyfile(output/name,output/("last-valid-"+name))
        stage["publication"]={"status":"pending","kind":kind,"candidate_sha256":hashes(candidate_dir),
                              "rollback_prefix":"last-valid-","rollback_sha256":hashes(output,"last-valid-"),
                              "baseline_sha256":hashes(output,"baseline-")}
        checkpoint(kind+"_publish","新草稿已独立验证；准备发布成套产物，并保留上一套有效草稿。")
        try:
            for name in CORE:
                temp=output/(name+".next")
                shutil.copyfile(candidate_dir/name,temp);temp.replace(output/name)
            stage.update(status="completed" if is_parametric else "running",accepted=is_parametric,
                         topology_exported=published_stage["topology_exported"],
                         geometry_updated_by_api=published_stage["geometry_updated_by_api"],
                         dimensions_updated_by_api=published_stage["dimensions_updated_by_api"])
            stage["publication"]["status"]="committed"
            checkpoint(kind+"_export",f"已发布 {len(updated['entities'])} 个图元；"+("仅已绑定约束子集通过求解。" if is_parametric else "已保留源笔画支持的轮廓修正，尺寸尚未求解。"))
        except Exception:
            for name in CORE:shutil.copyfile(output/("last-valid-"+name),output/name)
            stage["publication"]["status"]="rolled_back"
            stage.update(status="interrupted",accepted=False,topology_exported=bool(current.get("parameterization",{}).get("topology_exported")),
                         geometry_updated_by_api=False,dimensions_updated_by_api=False)
            _write(output/"parametric-stage.json",stage)
            raise
        updated["parameterization"]=copy.deepcopy(stage)
        return updated
    try:
        for name in CORE:
            if (output/name).is_file() and not (output/("baseline-"+name)).exists():
                shutil.copyfile(output/name,output/("baseline-"+name))
        checkpoint("topology","从近似边界提取图元连接图，并用原图笔画检查局部误分割。")
        base_graph=build_topology(image_path,document,baseline,output)
        checkpoint("topology_planning","生成多个 LINE/ARC 拓扑候选，按标注引线覆盖和源边界证据规划对象数量。")
        graph,topology_plan=_plan_topology(image_path,document,baseline,base_graph,output,
                                           planner_provider=planner_provider,editor_provider=editor_provider,
                                           evaluator_provider=evaluator_provider,use_api=use_api)
        stage["topology_planning"]={key:value for key,value in topology_plan.items()
                                     if key not in {"local_evaluation"}}
        stage["topology_planning"]["local_evaluation"]=topology_plan["local_evaluation"]
        stage["topology"]={"entity_count":len(graph["entities"]),"node_count":len(graph["nodes"]),
                           "units":graph["units"],"status":graph.get("status"),
                           "initial_entity_count":len(baseline["entities"]),
                           "base_topology_entity_count":len(base_graph["entities"]),
                           "selected_candidate_id":topology_plan["selected_candidate_id"],
                           "selection_source":topology_plan["selection_source"]}
        source_validation=_topology_source_validation(image_path,baseline,graph)
        stage["topology_source_validation"]=source_validation
        if not source_validation["passed"]:
            stage.update(status="candidate_rejected",accepted=False,reason="source_topology_validation_failed")
            checkpoint("topology_rejected","源拓扑候选的来源、坐标或笔画支持校验未通过；保留上一套有效草稿，不送在线绑定或求解。")
            return current,stage
        checkpoint("topology_validate","独立核验源笔画支持、闭合拓扑与DXF回读，不把像素拟合或尺寸求解结果混为一项。")
        topology_dir=output/"source-topology-export"
        topology_model=_export_chain(image_path,baseline,graph["entities"],topology_dir,
                                     topology=True,source_validation=source_validation)
        current=publish(topology_dir,topology_model,kind="topology")
        stage["provider"]={"status":"pending" if use_api else "disabled","network_requests":None if use_api else 0,
                           "http_success":None if use_api else False,"schema_success":False}
        checkpoint("constraint_binding","关联原图标注与候选图元；数值只取自标注，歧义保留为未绑定。")
        bindings=analyze_constraint_bindings(image_path,document,baseline,graph,output,provider=provider,use_api=use_api)
        stage["provider"]=bindings.get("provider",{})
        stage["binding_counts"]=bindings.get("counts",{})
        stage["constraints"]=bindings.get("constraints",[])
        stage["issues"]=bindings.get("issues",[])
        checkpoint("parametric_solve","联合求解图元尺寸与连接关系，逐项核验已绑定约束。")
        solution=solve_parametric(graph,stage["constraints"],output_dir=output)
        _write(output/"parametric-solution.json",solution)
        stage["solver"]={key:value for key,value in solution.items() if key not in {"entities","nodes","candidate_entities","candidate_nodes","constraints"}}
        if solution.get("accepted") is not True:
            stage.update(status="candidate_rejected",accepted=False,reason=solution.get("status","solver_rejected"))
            checkpoint("parametric_retained","参数候选未通过检查，保留最后一套有效轮廓和本次约束诊断。")
            return ({**current,"parameterization":copy.deepcopy(stage)} if stage.get("topology_exported") else current),stage
        candidate_dir=output/"parametric-export"
        checkpoint("parametric_validate","数值候选已产生；独立验证导出拓扑、单位与DXF回读后再发布。")
        updated=export_parametric(image_path,baseline,solution,candidate_dir)
        stage.update(underconstrained=solution.get("underconstrained",True))
        selected=bindings.get("bindings",[]) if bindings.get("provider",{}).get("schema_success") is True else []
        api_dimensions=any(row.get("accepted") is True and row.get("source")=="ocr_api_binding" for row in selected)
        api_relations=any(row.get("accepted") is True and row.get("relation_id") for row in selected)
        updated=publish(candidate_dir,updated,kind="parametric",api_geometry=api_dimensions or api_relations,
                        api_dimensions=api_dimensions)
        return updated,stage
    except InterruptedError:
        raise
    except Exception as error:
        if stage.get("provider",{}).get("status")=="pending":
            stage["provider"].update(status="interrupted",network_requests=None,http_success=None,schema_success=False)
        stage.update(status="failed",accepted=False,error_type=type(error).__name__,
                     geometry_updated_by_api=False,dimensions_updated_by_api=False,
                     error_code="parametric_"+stage.get("stage","initialization")+"_failed",
                     reason="parametric_stage_failed_last_valid_preserved")
        safe_export_errors={"Unsupported parametric entity count":"entity_count_outside_export_limit",
                            "Parametric candidate is not a valid closed contour":"invalid_export_topology",
                            "Unsupported parametric primitive":"unsupported_export_primitive",
                            "Parametric DXF readback failed":"dxf_readback_failed",
                            "Source image unavailable during parametric export":"source_image_unavailable",
                            "Solution and source units differ":"solution_units_mismatch"}
        if isinstance(error,ValueError) and str(error) in safe_export_errors:
            stage["error_code"]=safe_export_errors[str(error)]
        _write(output/"parametric-stage.json",stage)
        emit("parametric_unavailable","参数化阶段未完成，已保留最后一套有效草稿及阶段证据。")
        return ({**current,"parameterization":copy.deepcopy(stage)} if stage.get("topology_exported") else current),stage
