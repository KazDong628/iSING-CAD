"""Source topology -> sourced bindings -> numerical constraints -> CAD export.

The initial pixel trace is preserved. Parametric corrections must pass source
and dimensional checks, including the original oracle-mask budget when used.
Reference CAD geometry is never consumed by this module.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import inspect
from pathlib import Path
import shutil

import cv2
import ezdxf
import numpy as np
from shapely.geometry import Polygon

from .automatic import _verify_dxf_readback
from .vectorize import _sample_entities, assess_fit_quality
from .reconstruction_feedback import (reconstruction_feedback, geometry_fingerprint,
                                       constraint_regression, primitive_diagnostics,
                                       source_failure_feedback, source_mask_interval_diagnostics)
from .radius_contract import exact_radius_checks, annotation_radius_contract
from .topology import stroke_support_fraction


CORE = ("drawing.dxf", "preview.svg", "overlay.png", "model.json", "validation.json", "curve-fit.json")


def _write(path, value):
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf8")
    temporary.replace(path)


def _os_error_diagnostic(error, image_path, output):
    """Persist I/O facts without copying exception text or arbitrary paths."""
    if not isinstance(error, OSError):
        return None
    diagnostic={"kind":"os_error", "errno":error.errno,
                "winerror":getattr(error,"winerror",None)}
    locations=[]
    workspace=Path.cwd().resolve()
    for attribute in ("filename", "filename2"):
        name=getattr(error,attribute,None)
        if not isinstance(name,(str,bytes)):
            continue
        try:
            path=Path(name).resolve()
            if path==Path(image_path).resolve():
                role="source_image"
            elif path.is_relative_to(Path(output).resolve()):
                role="output_artifact"
            else:
                continue
            locations.append({"attribute":attribute,"role":role,
                              "workspace_relative_path":path.relative_to(workspace).as_posix()})
        except (OSError,ValueError,TypeError):
            continue
    diagnostic["known_file_locations"]=locations
    return diagnostic


def _write_workflow_provenance(output, stage, context):
    """Describe current CORE bytes, separately from the latest attempted stage.

    This deliberately does not emit progress: rollback must be able to repair
    its receipt even when a progress callback was the cause of the rollback.
    """
    output=Path(output)
    hashes={name:hashlib.sha256((output/name).read_bytes()).hexdigest()
            for name in CORE if (output/name).is_file()}
    publication=stage.get("publication",{})
    manifests={key:publication.get(key+"_sha256",{}) for key in ("candidate","rollback","baseline")}
    if not manifests["baseline"]:
        manifests["baseline"]={name:hashlib.sha256((output/("baseline-"+name)).read_bytes()).hexdigest()
                               for name in CORE if (output/("baseline-"+name)).is_file()}
    matched=next((key for key,values in manifests.items()
                  if set(hashes)==set(CORE) and hashes==values),None)
    def read_json(name):
        try:
            value=json.loads((output/name).read_text(encoding="utf8"))
            return value if isinstance(value,dict) else {}
        except (OSError,ValueError):
            return {}
    model=read_json("model.json")
    validation=read_json("validation.json")
    metadata_consistent=bool(model and validation and model.get("validation")==validation)
    integrity=bool(matched and metadata_consistent)
    version=model.get("algorithm_version")
    kind=("parametric" if version=="source-topology-bound-parametric-v1" else
          "topology" if version=="source-topology-draft-v1" else "initial_pixel_fit_draft")
    actual_stage=model.get("parameterization") or {}
    contract=copy.deepcopy(validation.get("annotation_radius_contract") or
                           stage.get("annotation_radius_contract"))
    if contract:
        exact=validation.get("exact_radius_validation") or {}
        current_verified=bool(integrity and kind=="parametric" and
                              validation.get("annotation_radius_contract",{}).get("current_dxf_verified") and
                              exact.get("passed") and exact.get("dxf_readback_performed"))
        contract.update(current_dxf_verified=current_verified,
                        satisfied=bool(current_verified and contract.get("satisfied")),
                        all_annotated_radii_verified=bool(current_verified and contract.get("satisfied")))
    accepted=bool(integrity and actual_stage.get("accepted") and
                  contract and contract.get("satisfied"))
    provenance={"schema_version":"cad-reconstruction-provenance-v1",**context,
                "reference_dxf_used_as_prediction_geometry":False,"reference_geometry_sent_to_provider":False,
                "initial_geometry":"local_mask_to_LINE_ARC_fitting",
                "topology_selection_source":stage.get("topology",{}).get("selection_source"),
                "source_arrow_localization":stage.get("radius_target_provider",{}),
                "binding_provider":stage.get("provider",{}),
                "solver_requested":"parametric_solve" in stage.get("stages",[]),
                "solver_executed":bool(stage.get("solver")),
                "solver_status":stage.get("solver",{}).get("status"),
                "strict_radius_contract":contract,
                "published_artifact_kind":kind if integrity else "unverified_core",
                "published_core_sha256":hashes,"published_core_manifest_match":matched,
                "publication_integrity_verified":integrity,"published_metadata_consistent":metadata_consistent,
                "attempted_publication_kind":publication.get("kind"),
                "attempted_publication_status":publication.get("status"),
                "parameterization_accepted":accepted,
                "constraint_subset_accepted":bool(integrity and actual_stage.get("constraint_subset_accepted") and
                                                    contract and contract.get("current_dxf_verified")),
                "all_dimensions_verified":False,"reference_verified":False,
                "prediction_dxf_sha256":hashes.get("drawing.dxf")}
    _write(output/"workflow-provenance.json",provenance)
    return provenance


def _export_chain(image_path, baseline, entities, output_dir, *, topology=False, constraint_validation=None, source_validation=None):
    """Export geometry under an explicit source-topology or constraint basis."""
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    entities=copy.deepcopy(entities)
    if topology:
        for entity in entities:
            if entity.get("radius_binding"):
                entity.update(dimension_bound=False,radius_binding_status="constructed_unverified")
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
                "source_mask_fidelity_used_as_acceptance_gate":bool((source_validation or {}).get("oracle_mask_validation",{}).get("applicable")),
                "issues":[],"meaning":"Partial source-bound parametric draft; geometry and enforced constraints are checked separately from reference accuracy."}
    validation["primitive_diagnostics"]=primitive_diagnostics(entities)
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
    entities=copy.deepcopy(solution["entities"])
    exact=exact_radius_checks(entities,solution.get("constraints",[]))
    if not exact["passed"]:
        raise ValueError("Annotated ARC radius must exactly equal its source value")
    for entity in entities:
        if entity.get("radius_binding"):
            verified=any(row.get("entity_id")==entity.get("id") and
                         row.get("record_id")==entity["radius_binding"].get("record_id") and row.get("passed") is True
                         for row in exact["checks"])
            entity.update(dimension_bound=verified,radius_binding_status="verified_constraint_subset" if verified else "constructed_unverified")
    model=_export_chain(image_path,baseline,entities,output_dir,constraint_validation=solution.get("validation"))
    readback=exact_radius_checks(entities,solution.get("constraints",[]),
                                dxf_document=ezdxf.readfile(Path(output_dir)/"drawing.dxf"))
    if not readback["passed"]:
        raise ValueError("Exported ARC radius differs from its source value")
    model["validation"]["exact_radius_validation"]=readback
    _write(Path(output_dir)/"validation.json",model["validation"])
    _write(Path(output_dir)/"model.json",model)
    return model


def _oracle_mask_validation(baseline, entities):
    """Compare every oracle-stage result with the immutable initial mask.

    This is a representation check against the explicitly supplied raster
    input, not a reference-DXF comparison. Never reset its budget to a newer
    fitted topology or a solver proposal.
    """
    if baseline.get("oracle_mask_conditioned") is not True:
        return {"applicable":False,"passed":True}
    try:
        budget=float(baseline["curve_fit"]["total_deviation_budget_px"])
        if not math.isfinite(budget) or budget<=0:raise ValueError("invalid initial mask budget")
        ring=baseline["extraction"]["raw_polyline_px"]
        system=baseline["coordinate_system"]
        scale=float(baseline["scale"]["pixels_per_mm"]) if system["units"]=="mm" else 1.
        origin=np.asarray(system["origin_source_px"],float)
        source_entities=[]
        for item in entities:
            source={"type":item["type"]}
            for key in ("start","end","center"):
                if key in item:source[key]=(np.asarray(item[key],float)*[scale,-scale]+origin).tolist()
            if item["type"]=="ARC":source.update(radius=float(item["radius"])*scale,clockwise=not bool(item["clockwise"]))
            source_entities.append(source)
        quality=assess_fit_quality(ring,source_entities)
        deviation=float(quality["source_boundary_deviation_px"]["conservative_upper_bound_px"])
        passed=bool(quality["sampled_topology_valid"] and math.isfinite(deviation) and deviation<=budget)
        return {"applicable":True,"passed":passed,"original_deviation_budget_px":budget,
                "conservative_max_deviation_px":deviation,"quality":quality,
                "observation":"initial_extraction_raw_polyline_px","budget_source":"initial_curve_fit_total_deviation_budget_px",
                "reason":None if passed else "original_oracle_mask_budget_exceeded",
                "reference_dxf_read":False,"reference_accuracy_verified":False}
    except (KeyError,TypeError,ValueError,IndexError,ArithmeticError):
        return {"applicable":True,"passed":False,"reason":"original_oracle_mask_observation_unavailable",
                "reference_dxf_read":False,"reference_accuracy_verified":False}


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
        metrics=[stroke_support_fraction(before,against=after),stroke_support_fraction(after,against=before),
                 float(before["p90_edge_distance_px"]),float(after["p90_edge_distance_px"]),float(graph["source_grid_pitch_px"])]
        if not all(math.isfinite(n) for n in metrics) or not 0<=metrics[0]<=1 or not 0<metrics[1]<=1 or min(metrics[2:])<0:
            reasons.append("invalid_source_support_metrics")
        # Same declared source-stroke comparison as topology candidate selection;
        # no reliance on the previous strict mask-fit acceptance flag.
        elif metrics[1]<metrics[0]-.035 or metrics[3]>metrics[2]+metrics[4]:
            reasons.append("source_stroke_support_degraded")
    except (KeyError,TypeError,ValueError):reasons.append("missing_source_support_metrics")
    mask_validation=_oracle_mask_validation(baseline,graph.get("entities",[]))
    if not mask_validation["passed"]:reasons.append(mask_validation["reason"])
    return {"passed":not reasons,"reasons":list(dict.fromkeys(reasons)),"source_evidence":evidence,"source_sha256":graph.get("source_sha256"),
            "coordinate_mapping":coordinate_check,
            "oracle_mask_validation":mask_validation,
            "acceptance_basis":"source_hash_stroke_support_closed_simple_geometry_and_DXF_readback",
            "dimensions_verified":False,"reference_verified":False,"mask_fidelity_gate_used":mask_validation["applicable"]}


def _solved_source_validation(image_path, document, baseline, graph, entities):
    """Check source ink again after solving, at the original fixed stroke gate."""
    from .topology import _StrokeEvidence
    from .ocr import canonical_records
    image=cv2.imdecode(np.fromfile(str(image_path),np.uint8),cv2.IMREAD_GRAYSCALE)
    if image is None:return {"passed":False,"reasons":["source_image_unavailable"]}
    grid=float(graph.get("source_grid_pitch_px") or 1.)
    evidence=_StrokeEvidence(image,canonical_records(document),grid)
    scale=float(baseline["scale"]["pixels_per_mm"]) if graph["units"]=="mm" else 1.
    origin=np.asarray(baseline["coordinate_system"]["origin_source_px"],float)
    def measure(chain):
        points,_,_=_sample_entities(chain,max_step_px=max(.25,grid/scale/2))
        return evidence.summarize(points*[scale,-scale]+origin)
    before,after=measure(graph["entities"]),measure(entities)
    passed=(stroke_support_fraction(after,against=before)>=stroke_support_fraction(before,against=after)-.035 and
            after["p90_edge_distance_px"]<=before["p90_edge_distance_px"]+grid)
    reasons=[] if passed else ["solved_source_stroke_support_degraded"]
    by_id={row.get("id"):row for row in entities}
    constructed=[]
    for entity in graph["entities"]:
        binding=entity.get("radius_binding")
        if entity.get("type")=="ARC" and binding and isinstance(binding.get("nominal"),(int,float)):
            current=by_id.get(entity["id"],{})
            nominal=float(binding["nominal"])
            radius=current.get("radius")
            valid_radius=(type(radius) in (int,float) and math.isfinite(radius) and radius>0)
            delta=abs(float(radius)-nominal) if valid_radius and math.isfinite(nominal) else None
            constructed.append({"entity_id":entity["id"],"record_id":binding.get("record_id"),"absolute_residual":delta,
                                "nominal":nominal if math.isfinite(nominal) else None,
                                "actual_radius":float(radius) if valid_radius else None,
                                "tolerance":0.,"passed":delta==0.,"binding_verified":False})
            if delta!=0.:reasons.append("constructed_annotation_radius_changed_after_solve")
    mask_validation=_oracle_mask_validation(baseline,entities)
    if not mask_validation["passed"]:reasons.append(mask_validation["reason"])
    return {"passed":not reasons,"reasons":reasons,"constructed_radius_preservation":constructed,
            "oracle_mask_validation":mask_validation,
            "before":before,"after":after,"maximum_support_drop":.035,"maximum_p90_increase_px":grid,
            "scope":("Source-stroke comparison and original oracle raster representation budget after numerical solving; no reference DXF comparison."
                     if mask_validation["applicable"] else "Source-stroke comparison after numerical solving; not GT or a relaxed mask-fidelity gate."),
            "reference_verified":False}


def _topology_edit_stage(image_path, document, baseline, selected, bundle, output, *,
                         editor_provider=None, evaluator_provider=None, use_api=False,
                         round_index=1, feedback=None, verifier=None, operation_filter=None,
                         candidate_namespace=None, provider_guard=None):
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
            options={"feedback":feedback} if "feedback" in inspect.signature(editor_provider.propose).parameters else {}
            remaining=provider_guard("editor") if provider_guard else 600.
            if remaining is not None:
                if "request_timeout_seconds" in inspect.signature(editor_provider.propose).parameters:
                    options["request_timeout_seconds"]=remaining
                editor_receipt=editor_provider.propose(image_path,overlay,selected,bundle.get("annotation_inventory",[]),**options)
            else:
                editor_receipt.update(status="skipped",reason="topology_search_provider_budget_exhausted")
        except InterruptedError:
            raise
        except Exception:
            editor_receipt.update(status="failed",reason="topology_edit_provider_error",
                                  schema_success=False,network_requests=None)
    agent_operations=(editor_receipt.get("operations",[])
                      if editor_receipt.get("schema_success") is True else [])
    unresolved=((feedback or {}).get("radius_binding_coverage") or {}).get("unresolved",[])
    local_annotation_operations=propose_annotation_arc_edits(
        selected.get("graph") or {},bundle.get("annotation_inventory",[]),limit=4,
        unresolved_radius_record_ids={row["record_id"] for row in unresolved
                                      if isinstance(row,dict) and isinstance(row.get("record_id"),str)})
    operations=[];seen=set();skipped_visited=[]
    # Explicit agent edits get the bounded operation budget before deterministic
    # hints; a failed generic R-refit must not starve a requested fillet/split.
    for operation in [*agent_operations,*local_annotation_operations]:
        ids=tuple(operation.get("entity_ids") or [])
        signature=(operation.get("action"),ids,operation.get("record_id"))
        if signature in seen:continue
        if operation_filter is not None and not operation_filter(operation):
            skipped_visited.append({"operation":operation,"status":"skipped","reason":"visited_parent_operation"})
            seen.add(signature)
            continue
        operations.append(operation);seen.add(signature)
        if len(operations)>=5:break
    if operations:
        try:
            edited,execution=execute_topology_edits(image_path,document,baseline,selected,bundle,operations,output)
        except InterruptedError:
            raise
        except Exception:
            execution.update(proposed=len(operations),status="failed",reason="local_topology_edit_execution_error")
    execution.setdefault("operations",[]).extend(skipped_visited)
    if round_index>1 or candidate_namespace:
        prefix=candidate_namespace or f"r{round_index:02d}"
        mapping={row["id"]:f"{prefix}-{row['id']}" for row in edited}
        for row in edited:
            row["id"]=mapping[row["id"]]
            row["graph"]["candidate_id"]=row["id"]
            if row["graph"].get("entity_identity"):
                row["graph"]["entity_identity"]["display_id_scope"]=row["id"]
        for row in execution.get("operations",[]):
            if row.get("candidate_id") in mapping:row["candidate_id"]=mapping[row["candidate_id"]]
    if verifier:
        for row in edited:
            row["constraint_feedback"]=verifier(row)
            row["constraint_regression_gate"]=constraint_regression(feedback or {},row["constraint_feedback"])
        by_candidate={row["id"]:row for row in edited}
        for operation in execution.get("operations",[]):
            row=by_candidate.get(operation.get("candidate_id"))
            if row is None:continue
            report=row["constraint_feedback"]
            operation["preflight_status"]="passed" if row["constraint_regression_gate"]["passed"] else "rejected"
            operation["preflight_reasons"]=row["constraint_regression_gate"]["reasons"]
            operation["source_validation"]=report.get("source_validation",{})
            operation["local_source_diagnostics"]=report.get("local_source_diagnostics",{})
    pool=[selected,*[row for row in edited if row.get("constraint_regression_gate",{}).get("passed",True)]]
    local=evaluate_candidates(pool,max_candidates=5)
    admissible=set(local.get("admissible_candidate_ids",[]))
    if len(pool)<=1:
        evaluator_receipt.update(status="skipped",reason="no_valid_edit_candidate",network_requests=0)
    if len(pool)>1 and use_api and evaluator_provider is not None:
        try:
            remaining=provider_guard("evaluator") if provider_guard else 600.
            if remaining is not None:
                options={"request_timeout_seconds":remaining} if "request_timeout_seconds" in inspect.signature(evaluator_provider.select).parameters else {}
                evaluator_receipt=evaluator_provider.select(
                    image_path,pool,selected["id"],bundle.get("annotation_inventory",[]),**options)
            else:
                evaluator_receipt.update(status="skipped",reason="topology_search_provider_budget_exhausted")
        except InterruptedError:
            raise
        except Exception:
            evaluator_receipt.update(status="failed",reason="topology_evaluation_provider_error",
                                     schema_success=False,network_requests=None)
    proposed_id=(evaluator_receipt.get("selected_candidate_id")
                 if evaluator_receipt.get("schema_success") is True else None)
    by_id={row["id"]:row for row in pool}
    evaluated={row["candidate_id"]:row for row in local.get("evaluated",[]) if row.get("admissible")}
    if proposed_id is None and evaluator_receipt.get("schema_success") is not True:
        # Offline/API-failure fallback requires strict local improvement, not
        # merely the most plausible model-selected hypothesis.
        base_eval=evaluated.get(selected.get("id"),{})
        eligible=[row for row in edited if row["id"] in admissible and
                  row.get("constraint_regression_gate",{}).get("passed",True) and
                  evaluated[row["id"]].get("score",0)>=base_eval.get("score",1) and
                  (len(row["graph"]["entities"])<len(selected["graph"]["entities"]) or
                   row.get("constraint_feedback",{}).get("issue_count",10**9)<(feedback or {}).get("issue_count",0))]
        if eligible:
            proposed_id=max(eligible,key=lambda row:evaluated[row["id"]]["score"])["id"]
    gate={"accepted":False,"reason":"evaluator_did_not_select_an_edit",
          "proposed_candidate_id":proposed_id,"base_candidate_id":selected.get("id")}
    final=selected
    branch_gates={}
    for candidate_id in admissible:
        if candidate_id==selected.get("id") or candidate_id not in by_id:continue
        base_eval=evaluated.get(selected.get("id"),{})
        edit_eval=evaluated.get(candidate_id,{})
        base_metrics=base_eval.get("metrics",{});edit_metrics=edit_eval.get("metrics",{})
        edit_execution=((by_id[candidate_id].get("graph") or {}).get("source_evidence") or {}).get("topology_edit") or {}
        edit_actions=[edit_execution.get("action")]
        edit_actions.extend(row.get("action") for row in edit_execution.get("operations",[]) if isinstance(row,dict))
        validated_partition=bool(edit_execution.get("resegmentation_applied") and
                                 edit_execution.get("radius_binding_applied") and
                                 len(edit_execution.get("bound_record_ids") or [])>=2)
        annotation_guided=bool(set(edit_actions)&{"refit_chain_as_annotated_arc","insert_annotated_fillet"} or
                               validated_partition)
        count_improved=edit_metrics.get("entity_count",10**9) < base_metrics.get("entity_count",0)
        count_preserved_for_annotation=bool(
            annotation_guided and edit_metrics.get("entity_count",10**9) <= base_metrics.get("entity_count",0))
        growth_operations=[row for row in (edit_execution.get("operations") or [edit_execution])
                           if isinstance(row,dict) and row.get("net_entity_reduction",0)<0]
        feature_restored=bool(growth_operations) and all(
            row.get("radius_binding_applied") and (row.get("feature_restoration_validated") or
                (row.get("resegmentation_applied") and len(row.get("bound_record_ids") or [])>=2))
            for row in growth_operations)
        type_corrected=bool("refit_entity_as_line" in edit_actions and
                            edit_metrics.get("source_boundary_support",0)>=base_metrics.get("source_boundary_support",0) and
                            edit_metrics.get("entity_count")==base_metrics.get("entity_count"))
        regression=by_id[candidate_id].get("constraint_regression_gate",{"passed":True})
        accepted=bool(
            edit_eval.get("score") is not None and base_eval.get("score") is not None and
            edit_eval["score"] >= base_eval["score"]-.02 and
            edit_metrics.get("source_boundary_support",0.) >= base_metrics.get("source_boundary_support",0.)-.02 and
            edit_metrics.get("unsupported_primitive_count",10**9) <= base_metrics.get("unsupported_primitive_count",0)+1 and
            (count_improved or count_preserved_for_annotation or feature_restored or type_corrected) and
            regression.get("passed") is True
        )
        branch_gates[candidate_id]={"accepted":accepted,
              "reason":None if accepted else "edited_candidate_failed_local_improvement_gate",
              "proposed_candidate_id":candidate_id,"base_candidate_id":selected.get("id"),
              "minimum_local_score":round(float(base_eval.get("score",0.))-.02,9),
              "minimum_source_boundary_support":max(0.,float(base_metrics.get("source_boundary_support",0.))-.02),
              "maximum_unsupported_primitive_count":int(base_metrics.get("unsupported_primitive_count",0))+1,
              "must_reduce_entity_count":not (annotation_guided or feature_restored or type_corrected),
              "annotation_guided_primitive_correction":annotation_guided,
              "source_validated_feature_restoration":feature_restored,
              "constraint_regression":regression,
              "selection_source":"online_evaluator" if evaluator_receipt.get("schema_success") else "strict_local_improvement"}
    if proposed_id==selected.get("id") and proposed_id in admissible:
        gate.update(reason="evaluator_preserved_base")
    elif proposed_id in branch_gates:
        gate=branch_gates[proposed_id]
        if gate["accepted"]:final=by_id[proposed_id]
    from .topology_search import preflight_admissible
    trusted=[row for row in pool if (row["id"]==selected.get("id") or branch_gates.get(row["id"],{}).get("accepted")) and
             preflight_admissible(row.get("constraint_feedback") or (feedback if row["id"]==selected.get("id") else {}) or {})]
    exploratory=[]
    for row in edited:
        report=row.get("constraint_feedback") or {}
        if (report.get("topology_source_validation") or {}).get("passed") is True and not preflight_admissible(report) and report.get("binding_status")!="not_run":
            row["diagnostic_only"]=True
            row["formal_binding_coverage_for_ranking"]=0
            exploratory.append(row)
    record={"schema_version":"multimodal-local-topology-edit-stage-v1","status":"completed",
            "base_candidate_id":selected.get("id"),"final_candidate_id":final.get("id"),
            "editor":editor_receipt,"execution":execution,"local_evaluation":local,
            "evaluator":evaluator_receipt,"acceptance_gate":gate,
            "local_annotation_operations":local_annotation_operations,
            "executed_operation_count":len(operations),
             "edited_candidate_ids":[row["id"] for row in edited],"ground_truth_used":False,
             "trusted_candidate_ids":[row["id"] for row in trusted],"branch_acceptance_gates":branch_gates,
             "exploratory_candidate_ids":[row["id"] for row in exploratory],
             "exploration_scope":"Original kernel source/mask/ink gates passed, later numerical preflight failed; never publish directly.",
             "visited_operations_skipped":len(skipped_visited),
            "scope":"Agent proposes entity-ID edits; local geometry executes and validates; a separate evaluator selects; GT is unavailable."}
    _write(output/"topology-edit-proposals.json",record)
    if edited:
        bundle["candidates"].extend(edited)
        bundle["candidate_count"]=len(bundle["candidates"])
        bundle.setdefault("generation",{})["local_edit_candidate_count"]=len(edited)
        _write(output/"topology-candidates.json",bundle)
    return final,record


def _solver_source_observation(baseline, graph):
    """Map the immutable input mask to design units without opening any file."""
    raw=(baseline.get("extraction") or {}).get("raw_polyline_px")
    if raw is None:return None
    from .topology_candidates import _affines
    from .vectorize import _closed_ring
    points=_closed_ring(raw)
    if len(points)>30000 or not np.isfinite(points).all():
        raise ValueError("invalid_solver_mask_observation")
    _,source_to_design,_=_affines(graph,baseline)
    points=source_to_design(points)
    if not np.isfinite(points).all():raise ValueError("invalid_solver_mask_mapping")
    observation={"units":graph.get("units"),"points":points.tolist(),
             "provenance":"input_mask_boundary",
             "oracle_mask_conditioned":baseline.get("oracle_mask_conditioned") is True,
             "reference_dxf_read":False}
    budget=(baseline.get("curve_fit") or {}).get("total_deviation_budget_px")
    ppm=float((baseline.get("scale") or {}).get("pixels_per_mm") or 1.) if graph.get("units")=="mm" else 1.
    if type(budget) in (int,float) and math.isfinite(budget) and budget>0 and math.isfinite(ppm) and ppm>0:
        observation["boundary_error_budget"]={"units":graph.get("units"),"maximum_deviation":float(budget)/ppm,
            "sampling_step":.5/ppm,"source":"initial_curve_fit_total_deviation_budget_px"}
    return observation


def _solve_with_source_observation(solver, graph, constraints, baseline, output, *, budget_seconds=None):
    # Explicit signature compatibility keeps external solver adapters usable;
    # the production solver always declares the observation argument.
    options={"output_dir":output}
    if "source_observation" in inspect.signature(solver).parameters:
        options["source_observation"]=_solver_source_observation(baseline,graph)
        observation=options["source_observation"]
        if budget_seconds is not None and observation and "boundary_error_budget" in observation:
            observation["boundary_error_budget"]["max_wall_seconds"]=min(120.,float(budget_seconds))
    return solver(graph,constraints,**options)


def _topology_edit_loop(image_path, document, baseline, selected, bundle, output, *,
                        editor_provider=None, evaluator_provider=None, use_api=False,
                        progress=None, max_rounds=3, beam_width=3, max_preflights=18,
                        max_provider_calls=12, max_seconds=600.):
    """Bounded beam of source-certified edits; no GT-driven candidate selection."""
    from .constraint_binding import analyze_constraint_bindings
    from .parametric_solver import solve_parametric
    from .topology_search import SearchBudget, preflight_admissible, retain_distinct
    if isinstance(max_rounds,bool) or not 1<=max_rounds<=3:
        raise ValueError("topology_iteration_budget_must_be_1_to_3")
    if type(beam_width) is not int or not 1<=beam_width<=3:
        raise ValueError("topology_beam_width_must_be_1_to_3")
    output=Path(output);root=output/"topology-iterations";root.mkdir(parents=True,exist_ok=True)
    emit=progress or (lambda stage,message:None)
    budget=SearchBudget(max_preflights=max_preflights,max_provider_calls=max_provider_calls,max_seconds=max_seconds)
    record={"schema_version":"bounded-topology-iterations-v2","status":"running","max_rounds":max_rounds,
            "beam_width":beam_width,"ranking":"verified_radius_records, satisfied_independent_records, satisfied_structural_constraints, remaining_shape_dof, source_error",
            "rounds":[],"stop_reason":None,"ground_truth_used":False,
            "scope":"Source-only bounded beam; numerical construction is not fresh binding, and model verdicts do not certify reference accuracy."}
    cache={};geometries={};seen_selected={geometry_fingerprint(selected["graph"])}
    def persist():
        record["budget"]=budget.summary()
        record["visited_operations"]=list(budget.visited_operations.values())
        record["visited_geometries"]=list(geometries.values())
        _write(output/"topology-iterations.json",record)
    def verify(candidate):
        graph=candidate["graph"];geometry_key=geometry_fingerprint(graph)
        # Geometry aliases can carry different current IDs/annotation targets;
        # never reuse a stale binding-ID map solely because coordinates match.
        semantics={"geometry":geometry_key,"entity_ids":[row.get("id") for row in graph.get("entities",[])],
                   "annotations":graph.get("annotation_support",[])}
        key=hashlib.sha256(json.dumps(semantics,sort_keys=True,allow_nan=False).encode()).hexdigest()
        if key in cache:
            result=copy.deepcopy(cache[key]);result["candidate_id"]=candidate["id"]
            return result
        if not budget.reserve_preflight():
            result=reconstruction_feedback(graph)
            result.update(solver_status="preflight_failed",constraint_count=0,binding_status="not_run",
                          source_validation={"passed":False,"reasons":["topology_search_preflight_budget_exhausted"]})
            return result
        persist()
        directory=root/"constraint-preflight"/key[:16];directory.mkdir(parents=True,exist_ok=True)
        source_validation=_topology_source_validation(image_path,baseline,graph)
        _write(directory/"source-validation.json",source_validation)
        if not source_validation["passed"]:
            result=reconstruction_feedback(graph)
            result.update(solver_status="source_validation_failed",constraint_count=0,
                          binding_status="not_run",source_validation=source_validation)
            try:
                from .source_support_diagnostics import source_support_diagnostics
                diagnostic=source_support_diagnostics(image_path,document,baseline,graph)
                _write(directory/"source-support-diagnostics.json",diagnostic)
                result["local_source_diagnostics"]=diagnostic
            except (OSError,KeyError,TypeError,ValueError):
                result["local_source_diagnostics"]={"status":"unavailable"}
        else:
            try:
                bindings=analyze_constraint_bindings(image_path,document,baseline,graph,directory,use_api=False)
                if budget.remaining_seconds()<=0:
                    raise TimeoutError("topology_search_time_budget_exhausted")
                solution=_solve_with_source_observation(solve_parametric,graph,bindings.get("constraints",[]),baseline,directory,
                                                         budget_seconds=budget.remaining_seconds())
                result=reconstruction_feedback(graph,bindings,solution)
                diagnostic_entities=solution.get("entities") if solution.get("accepted") else solution.get("candidate_entities")
                if diagnostic_entities:
                    validation=_solved_source_validation(image_path,document,baseline,graph,diagnostic_entities)
                    _write(directory/"solved-source-validation.json",validation)
                    mask_diagnostics=None
                    if not validation["passed"]:
                        try:
                            mask_diagnostics=source_mask_interval_diagnostics(baseline,graph,diagnostic_entities)
                            mask_diagnostics["geometry_stage"]="solver_candidate" if solution.get("accepted") else "rejected_solver_candidate"
                            _write(directory/"source-mask-interval-diagnostics.json",mask_diagnostics)
                        except (KeyError,TypeError,ValueError,ArithmeticError):
                            mask_diagnostics={"status":"unavailable"}
                        from .source_support_diagnostics import source_support_diagnostics
                        solved_graph={**graph,"entities":diagnostic_entities}
                        diagnostic=source_support_diagnostics(image_path,document,baseline,solved_graph)
                        diagnostic["geometry_stage"]="solver_candidate" if solution.get("accepted") else "rejected_solver_candidate"
                        _write(directory/"source-support-diagnostics.json",diagnostic)
                        result["local_source_diagnostics"]=diagnostic
                    source_failure_feedback(result,graph,validation,mask_diagnostics)
            except InterruptedError:
                raise
            except Exception:
                result=reconstruction_feedback(graph)
                result.update(solver_status="preflight_failed",constraint_count=0,binding_status="not_run",
                              source_validation={"passed":False,"reasons":["source_constraint_preflight_failed"]})
        result["topology_source_validation"]=source_validation
        result["preflight_artifact"]=str(directory)
        if result.get("solver_accepted") is True and result.get("post_solve_source_accepted") is True:
            _write(directory/"topology.json",graph)
            _write(directory/"preflight-input-identity.json",_preflight_input_identity(image_path,document,baseline,graph))
        _write(directory/"reconstruction-feedback.json",result)
        cache[key]=copy.deepcopy(result)
        geometries.setdefault(geometry_key,{"geometry_sha256":geometry_key,"candidate_ids":[]})["candidate_ids"].append(candidate["id"])
        geometries[geometry_key].update(solver_status=result.get("solver_status"),
            source_passed=(result.get("source_validation") or source_validation).get("passed"),
            failure_reasons=(result.get("source_validation") or {}).get("reasons",[]),
            remaining_shape_dof=result.get("remaining_shape_dof"),issue_count=result.get("issue_count"))
        persist()
        return result
    selected["constraint_feedback"]=verify(selected)
    beam=[selected]
    for number in range(1,max_rounds+1):
        previous_selected=selected
        previous_beam={geometry_fingerprint(row["graph"]) for row in beam}
        branches=[];next_pool=[];exploration_pool=[];fresh_operations=False;retryable_failure=False
        persist()
        emit("topology_edit_iteration",f"局部拓扑第 {number}/{max_rounds} 轮：遍历最多 {len(beam)} 个候选，结合独立标注绑定、求解与源边界证据。")
        for branch_index,parent in enumerate(beam,1):
            if budget.remaining_seconds()<=0:
                record["stop_reason"]="topology_search_time_budget_exhausted";break
            feedback=copy.deepcopy(verify(parent));feedback["round"]=number
            if parent.get("diagnostic_only") and (feedback.get("topology_source_validation") or {}).get("passed") is True:
                exploration_pool.append(parent)
            feedback["previous_operations"]=[{**row.get("operation",{}),"status":row.get("preflight_status",row.get("status")),
                                          "reason":(row.get("preflight_reasons") or [row.get("reason")])[0],
                                          "source_validation":row.get("source_validation",{}),
                                          "local_source_diagnostics":row.get("local_source_diagnostics",{})}
                                            for previous in record["rounds"]
                                            for row in previous["execution"].get("operations",[]) if isinstance(row.get("operation"),dict)][-15:]
            feedback["previous_rounds"]=[{"round":row["round"],"acceptance_gate":row["acceptance_gate"],
                                      "execution":row["execution"]} for row in record["rounds"]]
            if record["rounds"]:
                previous=record["rounds"][-1]
                feedback["previous_provider_failures"]=[name for name in ("editor","evaluator")
                if previous.get(name,{}).get("status")=="failed" and
                   previous.get(name,{}).get("schema_success") is not True]
            parent_hash=geometry_fingerprint(parent["graph"])
            known_operations=set(budget.visited_operations)
            def operation_filter(operation):
                allowed,_=budget.visit_operation(parent_hash,operation);persist()
                return allowed
            def provider_guard(role):
                remaining=budget.reserve_provider(role,parent_hash);persist()
                return remaining
            directory=root/f"round-{number:02d}"
            if len(beam)>1:directory=directory/f"branch-{branch_index:02d}"
            directory.mkdir(parents=True,exist_ok=True)
            proposed,step=_topology_edit_stage(image_path,document,baseline,parent,bundle,directory,
                editor_provider=editor_provider,evaluator_provider=evaluator_provider,use_api=use_api,
                round_index=number,feedback=feedback,verifier=verify,operation_filter=operation_filter,
                candidate_namespace=f"r{number:02d}-b{branch_index:02d}",provider_guard=provider_guard)
            step.update(round=number,branch=branch_index,parent_geometry_sha256=parent_hash,feedback=feedback)
            if step["acceptance_gate"].get("accepted") and geometry_fingerprint(proposed["graph"])==parent_hash:
                step["acceptance_gate"].update(accepted=False,reason="repeated_geometry")
                step["final_candidate_id"]=parent["id"]
            for operation in step.get("execution",{}).get("operations",[]):
                value=operation.get("operation")
                if not isinstance(value,dict):continue
                _,key=budget.visit_operation(parent_hash,value)
                operation["parent_operation_sha256"]=key
                budget.visited_operations[key].update(status=operation.get("preflight_status",operation.get("status")),
                    reasons=operation.get("preflight_reasons") or [operation.get("reason")],
                    candidate_id=operation.get("candidate_id"))
            for operation in step.get("editor",{}).get("operations",[]):
                if isinstance(operation,dict):budget.visit_operation(parent_hash,operation)
            fresh_operations |= bool(set(budget.visited_operations)-known_operations)
            retryable_failure |= any(step.get(role,{}).get("status")=="failed" and
                                     step.get(role,{}).get("schema_success") is not True for role in ("editor","evaluator"))
            by_id={row["id"]:row for row in bundle.get("candidates",[])}
            by_id.update({parent["id"]:parent,proposed["id"]:proposed})
            trusted=step.get("trusted_candidate_ids")
            if isinstance(trusted,list):
                options=[by_id[name] for name in trusted if name in by_id]
            else:
                options=[proposed] if step["acceptance_gate"].get("accepted") else [parent]
            for row in options:
                row["constraint_feedback"]=verify(row)
                if (row["constraint_feedback"].get("topology_source_validation") or {}).get("passed") is True and preflight_admissible(row["constraint_feedback"]):
                    row.pop("diagnostic_only",None)
                    next_pool.append(row)
            for name in step.get("exploratory_candidate_ids",[]):
                if name not in by_id:continue
                row=by_id[name];row["constraint_feedback"]=verify(row)
                if (row["constraint_feedback"].get("topology_source_validation") or {}).get("passed") is True:
                    row["diagnostic_only"]=True;row["formal_binding_coverage_for_ranking"]=0
                    exploration_pool.append(row)
            branches.append(step)
            persist()
            if budget.preflights>=budget.max_preflights:
                record["stop_reason"]="topology_search_preflight_budget_exhausted";break
            if use_api and budget.provider_calls>=budget.max_provider_calls:
                record["stop_reason"]="topology_search_provider_budget_exhausted";break
        if not branches:break
        preferred=next((row["final_candidate_id"] for row in branches if row["acceptance_gate"].get("accepted")),selected["id"])
        credible=retain_distinct(next_pool,width=beam_width,preferred_id=preferred)
        selected=credible[0] if credible else previous_selected
        # One publishable/fallback parent plus at most two kernel-safe diagnostic
        # branches lets a two-region repair span rounds without publishing an
        # intermediate failed numerical candidate or claiming its R coverage.
        exploration=retain_distinct(exploration_pool,width=min(2,beam_width),preferred_id=None)
        beam=[selected];beam_hashes={geometry_fingerprint(selected["graph"])}
        for row in [*exploration,*credible[1:]]:
            key=geometry_fingerprint(row["graph"])
            if key in beam_hashes:continue
            beam.append(row);beam_hashes.add(key)
            if len(beam)>=beam_width:break
        beam=beam[:beam_width]
        signature=geometry_fingerprint(selected["graph"])
        changed=signature!=geometry_fingerprint(previous_selected["graph"])
        step=copy.deepcopy(branches[0])
        step["branches"]=branches
        step["execution"]={**step.get("execution",{}),"operations":[operation for branch in branches for operation in branch.get("execution",{}).get("operations",[])]}
        step["beam_candidate_ids"]=[row["id"] for row in beam if not row.get("diagnostic_only") and preflight_admissible(row.get("constraint_feedback") or {})]
        step["exploratory_candidate_ids"]=[row["id"] for row in beam if row.get("diagnostic_only")]
        step["acceptance_gate"]={**step["acceptance_gate"],"accepted":changed,
                                 "base_candidate_id":previous_selected["id"],"proposed_candidate_id":selected["id"]}
        step["final_candidate_id"]=selected["id"]
        if changed:
            if signature in seen_selected:
                step["acceptance_gate"].update(accepted=False,reason="repeated_geometry")
                selected=previous_selected;record["stop_reason"]="repeated_geometry"
            else:seen_selected.add(signature)
        elif any(branch["acceptance_gate"].get("reason")=="repeated_geometry" for branch in branches) and not changed:
            step["acceptance_gate"].update(accepted=False,reason="repeated_geometry")
            record["stop_reason"]="repeated_geometry"
        frontier_changed={geometry_fingerprint(row["graph"]) for row in beam}!=previous_beam
        if not changed and not frontier_changed and not record["stop_reason"]:
            can_revise=use_api and (fresh_operations or retryable_failure) and number<max_rounds
            if retryable_failure and can_revise:
                step["retry_reason"]="provider_failure_within_existing_round_budget"
            if not can_revise:record["stop_reason"]="no_accepted_improvement"
        record["rounds"].append(step)
        record["final_candidate_id"]=selected["id"]
        record["final_feedback"]=verify(selected)
        record["beam_candidate_ids"]=[row["id"] for row in beam if not row.get("diagnostic_only") and preflight_admissible(row.get("constraint_feedback") or {})]
        record["exploratory_candidate_ids"]=[row["id"] for row in beam if row.get("diagnostic_only")]
        record["diagnostic_seed_candidate_ids"]=[row["id"] for row in beam if not preflight_admissible(row.get("constraint_feedback") or {})]
        persist()
        _write(output/"topology-candidates.json",bundle)
        if record["stop_reason"]:break
    record["final_source_rechecks"]=[{"candidate_id":row["id"],
        "validation":_topology_source_validation(image_path,baseline,row["graph"])} for row in beam]
    passing={row["candidate_id"] for row in record["final_source_rechecks"] if row["validation"].get("passed") is True}
    if selected["id"] not in passing:
        selected=next((row for row in beam if row["id"] in passing),previous_selected if record["rounds"] else selected)
        record["stop_reason"]="final_source_recheck_failed"
    record["final_candidate_id"]=selected["id"]
    record.update(status="completed",stop_reason=record["stop_reason"] or "round_budget_exhausted")
    persist()
    # Preserve old receipt consumers while adding the full per-round audit.
    summary={**(record["rounds"][-1] if record["rounds"] else {}),"rounds":record["rounds"],"max_rounds":max_rounds,
             "stop_reason":record["stop_reason"],"final_candidate_id":selected["id"],
             "beam_width":beam_width,"budget":record["budget"],"visited_operations":record["visited_operations"],
             "visited_geometries":record["visited_geometries"],"final_source_rechecks":record["final_source_rechecks"],
             "accepted_round_count":sum(row["acceptance_gate"].get("accepted") is True for row in record["rounds"])}
    _write(output/"topology-edit-proposals.json",summary)
    return selected,summary


def _source_valid_topology_choice(image_path, baseline, selected, selected_graph, candidates,
                                  local_evaluation, base_graph, *, allow_alternatives=True):
    """Recheck ranked hypotheses at the unchanged publication gate.

    Ranking is not publication approval. A rejected simplification must not
    prevent the last source-supported topology from reaching binding/solving.
    Every attempted graph retains its exact validation receipt.
    """
    by_id={row["id"]:row for row in candidates}
    base=candidates[0]
    choices=[(selected,selected_graph,"selected_candidate")]
    if allow_alternatives:
        ranked=sorted((row for row in local_evaluation.get("evaluated",[]) if row.get("admissible")),
                      key=lambda row:float(row.get("score") or 0.),reverse=True)
        choices.extend((by_id[row["candidate_id"]],by_id[row["candidate_id"]]["graph"],"ranked_candidate")
                       for row in ranked if row["candidate_id"] in by_id and
                       row["candidate_id"] not in {selected["id"],base["id"]})
    # Even without annotation leaders an online editor may have proposed a
    # different graph. Its rejection must still preserve the exact base;
    # allow_alternatives controls generated candidates, not this rollback.
    if selected_graph != base_graph:
        choices.append((base,base_graph,"preserved_base_topology"))
    attempts=[]
    for candidate,graph,kind in choices:
        validation=_topology_source_validation(image_path,baseline,graph)
        attempts.append({"candidate_id":candidate["id"],"graph_source":kind,"validation":validation})
        if validation["passed"]:
            return candidate,graph,{"status":"retained_selection" if len(attempts)==1 else "fallback_selected",
                                    "original_selected_candidate_id":selected["id"],
                                    "selected_candidate_id":candidate["id"],"selected_graph_source":kind,
                                    "attempts":attempts,"thresholds_unchanged":True}
    return selected,selected_graph,{"status":"no_source_valid_topology",
                                   "original_selected_candidate_id":selected["id"],
                                   "selected_candidate_id":None,"attempts":attempts,"thresholds_unchanged":True}


def _plan_topology(image_path, document, baseline, base_graph, output_dir, *, planner_provider=None,
                   editor_provider=None, evaluator_provider=None, use_api=False, progress=None):
    """Choose one bounded source-only topology candidate and persist the audit.

    Annotation evidence controls whether simplification is allowed.  With no
    detected source leader at all, the existing topology is retained instead
    of treating a low primitive count as evidence.  The online planner can only
    select candidates that passed the deterministic local evaluator.
    """
    from .planning_provider import evaluate_candidates
    from .topology_candidates import generate_topology_candidates, materialize_selected_candidate

    output=Path(output_dir)
    base_overlay=(output/"topology-overlay.png").read_bytes() if (output/"topology-overlay.png").is_file() else None
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
        except InterruptedError:
            raise
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
    if annotation_evidence:
        selected=by_id[selected_id]
        selected_graph=selected["graph"]
    selected,selected_graph,before_edit_selection=_source_valid_topology_choice(
        image_path,baseline,selected,selected_graph,bundle["candidates"],local,base_graph,
        allow_alternatives=annotation_evidence)
    source_valid_before_edit=before_edit_selection["status"]!="no_source_valid_topology"
    if before_edit_selection["status"]=="fallback_selected":
        selected_id=selected["id"]
        source="source_validation_fallback_before_edit"

    def preserve_base_materialization():
        _write(output/"topology.json",base_graph)
        if base_overlay is not None:(output/"topology-overlay.png").write_bytes(base_overlay)
        return {"status":"preserved_base_topology","candidate_id":selected["id"],
                "topology_path":str(output/"topology.json"),
                "overlay_path":str(output/"topology-overlay.png") if base_overlay is not None else None,
                "ground_truth_used":False,"reference_accuracy_verified":False}

    edit_stage=None
    if source_valid_before_edit:
        # A preserved original graph can differ from the generated candidate
        # wrapper with the same display ID. Bind/edit the graph that actually
        # passed validation, with its original overlay. Do not overwrite the
        # immutable generated bundle member merely to make its hash agree.
        preserved_base_before_edit=(selected_graph==base_graph and
                                    (before_edit_selection.get("selected_graph_source")=="preserved_base_topology"
                                     or not annotation_evidence))
        if selected.get("graph")!=selected_graph or preserved_base_before_edit:
            selected=copy.deepcopy(selected)
            selected["graph"]=copy.deepcopy(selected_graph)
            selected["entity_counts"]={"total":len(selected_graph["entities"]),
                                       "LINE":sum(row["type"]=="LINE" for row in selected_graph["entities"]),
                                       "ARC":sum(row["type"]=="ARC" for row in selected_graph["entities"])}
            if preserved_base_before_edit:
                materialization=preserve_base_materialization()
                selected["overlay_path"]=materialization["overlay_path"]
        if annotation_evidence or (use_api and editor_provider is not None):
            selected,edit_stage=_topology_edit_loop(image_path,document,baseline,selected,bundle,output,
                editor_provider=editor_provider,evaluator_provider=evaluator_provider,use_api=use_api,progress=progress)
            selected_id=selected["id"]
            selected_graph=selected["graph"]
            if edit_stage.get("accepted_round_count",0):
                source=("multimodal_local_topology_edit" if annotation_evidence else
                        "multimodal_local_topology_edit_without_annotation_leader")
        # The exact preserved graph is not necessarily a generated bundle
        # member. Only newly registered/unchanged generated candidates pass
        # through materialize_selected_candidate's membership/hash guard.
        if preserved_base_before_edit and not (edit_stage or {}).get("accepted_round_count",0):
            materialization=preserve_base_materialization()
        elif annotation_evidence or (edit_stage or {}).get("accepted_round_count",0):
            materialization=materialize_selected_candidate(selected,bundle,output)
    else:
        edit_stage={"status":"skipped","reason":"no_source_valid_topology", "accepted_round_count":0,
                    "ground_truth_used":False}
        materialization={"status":"not_materialized","reason":"no_source_valid_topology",
                         "candidate_id":selected["id"],"ground_truth_used":False,"reference_accuracy_verified":False}
    selected,selected_graph,source_selection=_source_valid_topology_choice(
        image_path,baseline,selected,selected_graph,bundle["candidates"],local,base_graph,
        allow_alternatives=annotation_evidence)
    if source_selection["status"]=="fallback_selected":
        selected_id=selected["id"]
        source="source_validation_fallback"
        if source_selection["selected_graph_source"]=="preserved_base_topology":
            materialization=preserve_base_materialization()
        else:
            materialization=materialize_selected_candidate(selected,bundle,output)
    plan={"schema_version":"source-topology-plan-v1","status":"selected" if source_selection["status"]!="no_source_valid_topology" else "no_source_valid_topology",
          "selected_candidate_id":selected_id,"selection_source":source,
          "annotation_evidence_available":annotation_evidence,
          "source_annotation_leader_count":len(leader_rows),
          "candidate_count":len(candidates),"local_evaluation":local,
           "provider":receipt,"online_selection_gate":online_gate,"topology_editing":edit_stage,
           "source_validation_before_edit":before_edit_selection,"source_validation_selection":source_selection,
          "selected_entity_counts":({"total":len(selected_graph["entities"]),
                                     "LINE":sum(row["type"]=="LINE" for row in selected_graph["entities"]),
                                     "ARC":sum(row["type"]=="ARC" for row in selected_graph["entities"])}
                                    if source_selection.get("selected_graph_source")=="preserved_base_topology"
                                    else selected.get("entity_counts",{})),
          "selected_annotation_coverage":selected.get("annotation_coverage"),
          "selected_unsupported_primitive_count":selected.get("unsupported_primitive_count"),
          "selected_preflight_artifact":(selected.get("constraint_feedback") or {}).get("preflight_artifact"),
          "materialization":materialization,"ground_truth_used":False,
          "dimensions_solved":False,"reference_accuracy_verified":False,
          "scope":"Source-only topology planning. Candidate selection does not certify dimensions or GT accuracy."}
    _write(output/"topology-plan.json",plan)
    return selected_graph,plan


def _canonical_sha256(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,
                                    separators=(",",":")).encode()).hexdigest()


def _preflight_input_identity(image_path, document, baseline, graph):
    return {"schema_version":"trusted-preflight-input-v1",
            "source_image_sha256":hashlib.sha256(Path(image_path).read_bytes()).hexdigest(),
            "source_ocr_sha256":_canonical_sha256(document),
            "immutable_baseline_sha256":_canonical_sha256(baseline),
            "topology_sha256":_canonical_sha256(graph),"reference_geometry_used":False}


def _binding_evidence_sha256(inventory):
    # Exclude paths, packet truncation and provider choices. These are the
    # actual independently measured source observations and candidate objects.
    keys=("units","source_image_sha256","proposal_tolerance_px","all_records","all_candidates",
          "radius_source_observations","relations","source_arrow_ownership")
    if not isinstance(inventory,dict) or any(key not in inventory for key in ("all_records","all_candidates")):
        return None
    return _canonical_sha256({key:inventory.get(key) for key in keys})


def _constraint_signature(row):
    return _canonical_sha256({key:row.get(key) for key in ("kind","record_id","entities","nodes","value","angle_mode")})


def _constraint_signatures(graph, rows):
    # Use the solver's own graph-aware input normalization. A LINE axis may
    # legally expand nodes=[] to its two endpoints; a tangent may expand to its
    # unique shared joint. Wrong nodes and different angle modes stay distinct.
    from .parametric_solver import _validate_inputs
    return {_constraint_signature(row) for row in _validate_inputs(graph,rows)[2]}


def _constraint_receipts_complete(bindings, solution, graph):
    keys=("kind","entities","nodes","value")
    required=bindings.get("constraints",[])
    if any(any(key not in row for key in keys) for row in required):return False
    passed=[row for row in solution.get("constraints",[])
            if row.get("passed") is True and all(key in row for key in keys)]
    try:return _constraint_signatures(graph,required)<=_constraint_signatures(graph,passed)
    except (ValueError,KeyError,TypeError,IndexError):return False


def _certify_preflight_checkpoint(image_path, document, baseline, graph, artifact_dir, output_dir):
    """Revalidate one whole checkpoint; never splice old constraints into a new graph.

    Legacy receipts without an input manifest are usable only after fresh local
    source admission of every constraint, the unchanged source gates, numerical
    receipts and a new DXF export/readback. Construction metadata is insufficient.
    """
    from .constraint_binding import analyze_constraint_bindings, radius_binding_coverage
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    receipt={"status":"unavailable","reason":"no_trusted_preflight","reference_geometry_used":False}
    if not artifact_dir:
        _write(output/"certificate.json",receipt);return None,receipt
    directory=Path(artifact_dir)
    try:
        def read(path):
            value=json.loads(path.read_text(encoding="utf8"))
            if not isinstance(value,dict):raise ValueError("invalid_checkpoint_object")
            return value
        feedback=read(directory/"reconstruction-feedback.json")
        bindings=read(directory/"constraint-bindings.json")
        solution=read(directory/"parametric-solve.json")
        identity=_preflight_input_identity(image_path,document,baseline,graph)
        manifest=directory/"preflight-input-identity.json"
        if manifest.is_file() and read(manifest)!=identity:
            raise ValueError("checkpoint_input_identity_mismatch")
        if feedback.get("geometry_sha256")!=geometry_fingerprint(graph):
            raise ValueError("checkpoint_geometry_mismatch")
        if (solution.get("accepted") is not True or feedback.get("solver_accepted") is not True or
                (feedback.get("source_validation") or {}).get("passed") is not True):
            raise ValueError("checkpoint_not_source_and_solver_admitted")
        if [row.get("id") for row in solution.get("entities",[])]!=[row.get("id") for row in graph["entities"]]:
            raise ValueError("checkpoint_current_object_ids_mismatch")
        required=_constraint_signatures(graph,bindings.get("constraints",[]))
        if not required or not _constraint_receipts_complete(bindings,solution,graph):
            raise ValueError("checkpoint_constraint_receipts_incomplete")
        recheck_dir=output/"source-recheck"
        recheck=analyze_constraint_bindings(image_path,document,baseline,graph,recheck_dir,use_api=False)
        current=_constraint_signatures(graph,recheck.get("constraints",[]))
        if not required<=current:
            raise ValueError("checkpoint_independent_source_evidence_no_longer_admits_constraints")
        inventory=read(recheck_dir/"binding-candidates.json")
        # Coverage is reconstructed from the present source inventory and the
        # checkpoint's own admitted subset. Freshly discovered obligations are
        # retained; fresh constraints are never injected into the saved solve.
        bindings=copy.deepcopy(bindings)
        bindings["radius_binding_coverage"]=radius_binding_coverage(inventory,graph,bindings["constraints"],bindings.get("bindings",[]))
        source=_topology_source_validation(image_path,baseline,graph)
        solved=_solved_source_validation(image_path,document,baseline,graph,solution["entities"])
        if source.get("passed") is not True or solved.get("passed") is not True:
            raise ValueError("checkpoint_current_source_gate_failed")
        export_solution={**solution,"constraints":copy.deepcopy(bindings["constraints"])}
        candidate_dir=output/"export"
        model=export_parametric(image_path,baseline,export_solution,candidate_dir)
        exact=model["validation"]["exact_radius_validation"]
        if not exact.get("passed") or not exact.get("dxf_readback_performed"):
            raise ValueError("checkpoint_exact_radius_readback_failed")
        model["validation"]["solved_source_validation"]=copy.deepcopy(solved)
        model["validation"]["source_mask_fidelity_used_as_acceptance_gate"]=bool(solved.get("oracle_mask_validation",{}).get("applicable"))
        contract=annotation_radius_contract(bindings,solution)
        contract.update(candidate_satisfied=contract["satisfied"],satisfied=False,
                        all_annotated_radii_verified=False,current_dxf_verified=False,
                        publication_status="candidate_only",exact_radius_validation=copy.deepcopy(exact))
        fresh_feedback=reconstruction_feedback(graph,bindings,solution)
        fresh_feedback["source_validation"]=solved
        fresh_feedback["post_solve_source_accepted"]=True
        receipt={"status":"certified","input_identity":identity,
                 "source_evidence_sha256":_binding_evidence_sha256(inventory),
                 "legacy_receipt_revalidated":not manifest.is_file(),
                 "source_and_mask_passed":True,"constraint_receipts_verified":True,
                 "native_radius_and_dxf_readback_verified":True,"reference_geometry_used":False,
                 "verified_radius_record_ids":fresh_feedback["verified_radius_record_ids"]}
        for name,value in (("topology.json",graph),("constraint-bindings.json",bindings),
                           ("parametric-solution.json",solution),("reconstruction-feedback.json",fresh_feedback),
                           ("source-validation.json",solved),("certificate.json",receipt)):
            _write(output/name,value)
        return {"graph":graph,"inventory":inventory,"binding_topology_artifact":recheck_dir/"binding-topology.png",
                "bindings":bindings,"solution":solution,"feedback":fresh_feedback,"source_validation":solved,
                "contract":contract,"model":model,"export_directory":candidate_dir,"certificate":receipt},receipt
    except InterruptedError:
        raise
    except (OSError,ValueError,KeyError,TypeError,ArithmeticError) as error:
        safe={"checkpoint_input_identity_mismatch","checkpoint_geometry_mismatch","checkpoint_not_source_and_solver_admitted",
              "checkpoint_current_object_ids_mismatch","checkpoint_constraint_receipts_incomplete",
              "checkpoint_independent_source_evidence_no_longer_admits_constraints","checkpoint_current_source_gate_failed",
              "checkpoint_exact_radius_readback_failed"}
        receipt.update(status="rejected",reason=str(error) if str(error) in safe else "checkpoint_certification_failed",
                       error_type=type(error).__name__)
        _write(output/"certificate.json",receipt)
        return None,receipt


def _final_checkpoint_replacement_gate(trusted, attempt, *, evidence_sha256):
    """Same source evidence plus non-regressing certified coverage, never a merge."""
    same=evidence_sha256 is not None and evidence_sha256==trusted["certificate"].get("source_evidence_sha256")
    reasons=[]
    if not same:reasons.append("independent_source_evidence_changed")
    if attempt["solution"].get("accepted") is not True:reasons.append("final_numerical_solution_not_accepted")
    if attempt["source_validation"].get("passed") is not True:reasons.extend(attempt["source_validation"].get("reasons") or ["final_source_gate_failed"])
    exact=attempt.get("model",{}).get("validation",{}).get("exact_radius_validation") or {}
    if not exact.get("passed") or not exact.get("dxf_readback_performed"):
        reasons.append("final_native_radius_and_dxf_readback_not_verified")
    graph=trusted["graph"]
    if not _constraint_receipts_complete(attempt["bindings"],attempt["solution"],graph):
        reasons.append("final_constraint_receipts_incomplete")
    try:
        old_constraints=_constraint_signatures(graph,trusted["bindings"].get("constraints",[]))
        new_constraints=_constraint_signatures(graph,attempt["bindings"].get("constraints",[]))
        if not old_constraints<=new_constraints:reasons.append("previously_verified_constraint_set_lost")
    except (ValueError,KeyError,TypeError,IndexError):reasons.append("final_constraint_references_not_validated")
    regression=constraint_regression(trusted["feedback"],attempt["feedback"])
    if not regression["passed"]:reasons.extend(regression["reasons"])
    old=set(trusted["feedback"].get("verified_radius_record_ids") or [])
    new=set(attempt["feedback"].get("verified_radius_record_ids") or [])
    if not old<=new:reasons.append("previously_verified_radius_coverage_lost")
    a,b=trusted["feedback"].get("remaining_shape_dof"),attempt["feedback"].get("remaining_shape_dof")
    if type(a) in (int,float) and type(b) in (int,float) and b>a:reasons.append("remaining_shape_dof_increased")
    return {"replace_checkpoint":not reasons,"retain_checkpoint":same and bool(reasons),
            "reasons":list(dict.fromkeys(reasons)),"same_independent_source_evidence":same,
            "constraint_regression":regression,"previous_verified_radius_records":sorted(old),
            "attempt_verified_radius_records":sorted(new),"reference_geometry_used":False}


def refine_parametric(image_path, document, baseline, output_dir, *, provider=None, planner_provider=None,
                      editor_provider=None, evaluator_provider=None, use_api=False, progress=None,
                      frozen_topology=None, trusted_preflight_dir=None):
    from .topology import build_topology
    from .constraint_binding import analyze_constraint_bindings
    from .parametric_solver import solve_parametric
    output=Path(output_dir)
    current=baseline
    emit=progress or (lambda stage,message:None)
    oracle_mask=baseline.get("oracle_mask_conditioned") is True
    input_ocr_sha256=hashlib.sha256(json.dumps(document,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    stage={"status":"running","accepted":False,"ground_truth_used":oracle_mask,
           "ground_truth_use":"raster_mask_input_only" if oracle_mask else "none",
           "oracle_mask_conditioned":oracle_mask,"all_dimensions_verified":False,
           "geometry_updated_by_api":False,"dimensions_updated_by_api":False,
           "provider":{"status":"not_invoked","network_requests":0},"stages":[]}
    provenance_context={"input_source_image_sha256":None,
                        "input_ocr_canonical_sha256":input_ocr_sha256,
                        "mask_input":"gt_raster_development_input" if oracle_mask else "segmentation_or_reviewed_material_mask"}
    def checkpoint(name,message):
        stage["stage"]=name;stage["stages"].append(name)
        _write(output/"parametric-stage.json",stage)
        _write_workflow_provenance(output,stage,provenance_context)
        emit(name,message)
    def publish(candidate_dir,updated,*,kind,api_geometry=False,api_dimensions=False):
        """Rollback targets the last complete draft; baseline-* stays immutable."""
        is_parametric=kind=="parametric"
        contract=copy.deepcopy(stage.get("annotation_radius_contract") or {})
        if is_parametric:
            complete_radii=bool(contract.get("candidate_satisfied") and
                contract.get("exact_radius_validation",{}).get("passed") and
                contract.get("exact_radius_validation",{}).get("dxf_readback_performed"))
            contract.update(satisfied=complete_radii,all_annotated_radii_verified=complete_radii,
                            current_dxf_verified=True,publication_status="committed")
            updated["validation"]["annotation_radius_contract"]=copy.deepcopy(contract)
            updated["validation"]["all_annotated_radii_verified"]=complete_radii
            _write(candidate_dir/"validation.json",updated["validation"])
        else:complete_radii=False
        parametric_status="completed" if complete_radii else "completed_with_unresolved_radii"
        published_stage={**copy.deepcopy(stage),"status":parametric_status if is_parametric else "source_topology_exported",
                         "accepted":is_parametric and complete_radii,"constraint_subset_accepted":is_parametric,
                         "topology_exported":stage.get("topology_exported",False) or not is_parametric,
                         "geometry_updated_by_api":bool(is_parametric and api_geometry),
                         "dimensions_updated_by_api":bool(is_parametric and api_dimensions),
                         "stage":kind+"_export","stages":[*stage["stages"],kind+"_publish",kind+"_export"]}
        published_stage.pop("publication",None)
        if is_parametric:published_stage["annotation_radius_contract"]=copy.deepcopy(contract)
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
            stage.update(status=parametric_status if is_parametric else "running",accepted=is_parametric and complete_radii,
                         constraint_subset_accepted=is_parametric,
                         topology_exported=published_stage["topology_exported"],
                         geometry_updated_by_api=published_stage["geometry_updated_by_api"],
                         dimensions_updated_by_api=published_stage["dimensions_updated_by_api"])
            stage["publication"]["status"]="committed"
            if is_parametric:
                stage["annotation_radius_contract"]=contract
                _write(output/"radius-contract.json",contract)
            checkpoint(kind+"_export",f"已发布 {len(updated['entities'])} 个图元；"+(
                ("已绑定半径经求解和DXF回读精确核验，其他尺寸与GT精度单独检查。" if complete_radii else
                 "已绑定约束子集通过求解，但仍有半径箭头或对应对象未核实，保留为未完成尺寸验证的草稿。")
                if is_parametric else "已保留源笔画支持的轮廓修正，尺寸尚未求解。"))
        except Exception:
            for name in CORE:shutil.copyfile(output/("last-valid-"+name),output/name)
            stage["publication"]["status"]="rolled_back"
            if is_parametric:
                contract.update(satisfied=False,all_annotated_radii_verified=False,
                                current_dxf_verified=False,publication_status="rolled_back")
                stage["annotation_radius_contract"]=contract
                _write(output/"radius-contract.json",contract)
            stage.update(status="interrupted",accepted=False,topology_exported=bool(current.get("parameterization",{}).get("topology_exported")),
                         constraint_subset_accepted=bool(current.get("parameterization",{}).get("constraint_subset_accepted")),
                         geometry_updated_by_api=False,dimensions_updated_by_api=False)
            _write(output/"parametric-stage.json",stage)
            _write_workflow_provenance(output,stage,provenance_context)
            raise
        updated["parameterization"]=copy.deepcopy(stage)
        return updated
    try:
        for name in CORE:
            if (output/name).is_file() and not (output/("baseline-"+name)).exists():
                shutil.copyfile(output/name,output/("baseline-"+name))
        provenance_context["input_source_image_sha256"]=hashlib.sha256(Path(image_path).read_bytes()).hexdigest()
        # Proposals are source-image pixels only. They augment detection and
        # must pass the same local ink/arrow checks before becoming bindings.
        if frozen_topology is None and use_api and editor_provider is not None and hasattr(editor_provider,"settings"):
            checkpoint("radius_target_localization","在线定位半径引线与箭尖；随后由原图像素复核，不读取参考CAD坐标。")
            from .radius_target_provider import RadiusTargetProvider, locate_radius_targets
            document,receipt=locate_radius_targets(image_path,document,baseline,output,
                                                   RadiusTargetProvider(editor_provider.settings),progress=checkpoint)
            stage["radius_target_provider"]={key:value for key,value in receipt.items() if key!="proposals"}
            checkpoint("radius_target_localization_finished","半径箭头提案已保存；未检测或歧义不会被视为没有标注。")
        checkpoint("topology","从近似边界提取图元连接图，并用原图笔画检查局部误分割。")
        if frozen_topology is None:
            base_graph=build_topology(image_path,document,baseline,output)
            checkpoint("topology_planning","生成多个 LINE/ARC 拓扑候选，按标注引线覆盖和源边界证据规划对象数量。")
            graph,topology_plan=_plan_topology(image_path,document,baseline,base_graph,output,
                                               planner_provider=planner_provider,editor_provider=editor_provider,
                                               evaluator_provider=evaluator_provider,use_api=use_api,progress=checkpoint)
        else:
            # Explicit final-stage replay of one frozen source candidate. This
            # does not rerun or claim a new upstream online reconstruction.
            graph=copy.deepcopy(frozen_topology);base_graph=graph
            topology_plan={"selected_candidate_id":graph.get("candidate_id"),
                "selection_source":"frozen_topology_final_stage_replay","local_evaluation":{},
                "selected_preflight_artifact":str(trusted_preflight_dir) if trusted_preflight_dir else None,
                "upstream_reconstruction_rerun":False,"ground_truth_used":False}
            _write(output/"topology.json",graph);_write(output/"topology-plan.json",topology_plan)
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
        trusted,certificate=_certify_preflight_checkpoint(image_path,document,baseline,graph,
            trusted_preflight_dir or topology_plan.get("selected_preflight_artifact"),output/"trusted-preflight")
        stage["trusted_preflight_checkpoint"]=certificate
        if trusted is not None:
            def adopt(prepared,selection):
                bound=copy.deepcopy(prepared["bindings"]);sol=prepared["solution"]
                _write(output/"binding-candidates.json",prepared["inventory"])
                bound["inventory_artifact"]=str(output/"binding-candidates.json")
                diagram=prepared.get("binding_topology_artifact")
                if diagram and Path(diagram).is_file():
                    shutil.copyfile(diagram,output/"binding-topology.png")
                    bound["topology_artifact"]=str(output/"binding-topology.png")
                stage.update(binding_counts=bound.get("counts",{}),constraints=bound.get("constraints",[]),
                    issues=bound.get("issues",[]),radius_binding_coverage=bound.get("radius_binding_coverage"),
                    reconstruction_feedback=prepared["feedback"],solved_source_validation=prepared["source_validation"],
                    annotation_radius_contract=copy.deepcopy(prepared["contract"]),
                    solver={key:value for key,value in sol.items() if key not in {"entities","nodes","candidate_entities","candidate_nodes","constraints"}},
                    selected_binding_provider=bound.get("provider",{}),binding_selection=selection,
                    underconstrained=sol.get("underconstrained",True))
                for name,value in (("constraint-bindings.json",bound),("parametric-solution.json",sol),
                                   ("reconstruction-feedback.json",prepared["feedback"]),
                                   ("radius-contract.json",stage["annotation_radius_contract"])):
                    _write(output/name,value)
            adopt(trusted,{"status":"certified_preflight_baseline"})
            stage["provider"]=trusted["bindings"].get("provider",{})
            current=publish(trusted["export_directory"],trusted["model"],kind="parametric")
            # The provider attempt is isolated from the already complete,
            # certified artifact set. Dropping a sent label is not a new source
            # observation and must not silently replace this checkpoint.
            stage["provider"]={"status":"pending" if use_api else "disabled","network_requests":None if use_api else 0,
                               "http_success":None if use_api else False,"schema_success":False}
            checkpoint("constraint_binding","可信预检已成套发布；在线重绑定作为独立候选，覆盖不得回退。")
            attempt_dir=output/"final-binding-attempt";attempt_dir.mkdir(parents=True,exist_ok=True)
            bindings=analyze_constraint_bindings(image_path,document,baseline,graph,attempt_dir,provider=provider,use_api=use_api)
            stage["provider"]=bindings.get("provider",{})
            checkpoint("parametric_solve","独立求解在线重绑定候选，并比较当前证据与已核实约束。")
            solution=_solve_with_source_observation(solve_parametric,graph,bindings.get("constraints",[]),baseline,attempt_dir)
            _write(attempt_dir/"parametric-solution.json",solution)
            attempted_entities=solution.get("entities") if solution.get("accepted") else solution.get("candidate_entities")
            source_check=(_solved_source_validation(image_path,document,baseline,graph,attempted_entities)
                          if attempted_entities else {"passed":False,"reasons":["final_numerical_candidate_unavailable"]})
            feedback=reconstruction_feedback(graph,bindings,solution)
            source_failure_feedback(feedback,graph,source_check)
            _write(attempt_dir/"reconstruction-feedback.json",feedback)
            _write(attempt_dir/"solved-source-validation.json",source_check)
            contract=annotation_radius_contract(bindings,solution)
            contract.update(candidate_satisfied=contract["satisfied"],satisfied=False,
                            all_annotated_radii_verified=False,current_dxf_verified=False,
                            publication_status="candidate_only")
            attempt={"bindings":bindings,"solution":solution,"feedback":feedback,
                     "source_validation":source_check,"contract":contract}
            if solution.get("accepted") is True and source_check.get("passed") is True:
                candidate_dir=attempt_dir/"parametric-export"
                updated=export_parametric(image_path,baseline,
                    {**solution,"constraints":copy.deepcopy(bindings.get("constraints",[]))},candidate_dir)
                updated["validation"]["solved_source_validation"]=copy.deepcopy(source_check)
                updated["validation"]["source_mask_fidelity_used_as_acceptance_gate"]=bool(source_check.get("oracle_mask_validation",{}).get("applicable"))
                contract["exact_radius_validation"]=copy.deepcopy(updated["validation"]["exact_radius_validation"])
                attempt.update(model=updated,export_directory=candidate_dir)
            _write(attempt_dir/"radius-contract.json",contract)
            try:
                inventory=json.loads((attempt_dir/"binding-candidates.json").read_text(encoding="utf8"))
            except (OSError,ValueError):inventory={}
            attempt.update(inventory=inventory,binding_topology_artifact=attempt_dir/"binding-topology.png")
            guard=_final_checkpoint_replacement_gate(trusted,attempt,evidence_sha256=_binding_evidence_sha256(inventory))
            # A different independent observation invalidates reuse rather
            # than authorizing a stale source obligation or an old target.
            if _preflight_input_identity(image_path,document,baseline,graph)!=certificate["input_identity"]:
                guard.update(replace_checkpoint=False,retain_checkpoint=False)
                guard["reasons"].append("checkpoint_inputs_changed_during_final_attempt")
            stage["final_binding_attempt"]={"directory":str(attempt_dir),"provider":stage["provider"],
                "feedback":feedback,"source_validation":source_check,"radius_contract":contract,"replacement_gate":guard}
            _write(output/"binding-publication-guard.json",stage["final_binding_attempt"])
            if guard["replace_checkpoint"]:
                adopt(attempt,{"status":"final_online_candidate_selected","replacement_gate":guard})
                decisions=bindings.get("bindings",[]) if stage["provider"].get("schema_success") is True else []
                api_dimensions=any(row.get("accepted") is True and row.get("source")=="ocr_api_binding" for row in decisions)
                api_relations=any(row.get("accepted") is True and row.get("relation_id") and row.get("admission_method")=="api_and_source" for row in decisions)
                current=publish(attempt["export_directory"],attempt["model"],kind="parametric",
                                api_geometry=api_dimensions or api_relations,api_dimensions=api_dimensions)
            elif guard["retain_checkpoint"]:
                adopt(trusted,{"status":"certified_preflight_retained","replacement_gate":guard})
                current=publish(trusted["export_directory"],trusted["model"],kind="parametric")
            else:
                # The source-only draft has no asserted dimensions, so it is
                # the safe fallback when new independent evidence contradicts
                # the dimensional checkpoint. No old bindings are merged.
                stage["trusted_preflight_checkpoint"]["status"]="revoked_by_changed_source_evidence"
                stage["revoked_checkpoint_diagnostics"]={"certificate":copy.deepcopy(certificate),
                    "radius_contract":copy.deepcopy(stage["annotation_radius_contract"]),
                    "feedback":copy.deepcopy(trusted["feedback"]),"current_dxf_verified":False}
                revoked_bindings=copy.deepcopy(bindings)
                revoked_bindings.update(constraints=[],bindings=[],counts={"constraints":0})
                _write(output/"binding-candidates.json",inventory)
                revoked_bindings["inventory_artifact"]=str(output/"binding-candidates.json")
                from .constraint_binding import radius_binding_coverage
                revoked_bindings["radius_binding_coverage"]=radius_binding_coverage(inventory,graph,[],[])
                revoked_solution={"accepted":False,"status":"revoked_by_changed_source_evidence",
                    "entities":copy.deepcopy(graph["entities"]),"constraints":[],"underconstrained":True}
                revoked_feedback=reconstruction_feedback(graph,revoked_bindings,revoked_solution)
                revoked_contract=annotation_radius_contract(revoked_bindings,revoked_solution)
                revoked_contract.update(satisfied=False,candidate_satisfied=False,all_annotated_radii_verified=False,
                    current_dxf_verified=False,publication_status="revoked")
                stage.update(constraints=[],binding_counts={"constraints":0},
                    radius_binding_coverage=revoked_bindings["radius_binding_coverage"],
                    reconstruction_feedback=revoked_feedback,annotation_radius_contract=revoked_contract,
                    solver={"status":"not_applied_to_current_source_topology","accepted":False},
                    selected_binding_provider={"status":"revoked","network_requests":0},
                    accepted=False,constraint_subset_accepted=False,geometry_updated_by_api=False,
                    dimensions_updated_by_api=False,underconstrained=True)
                for name,value in (("constraint-bindings.json",revoked_bindings),
                    ("parametric-solution.json",revoked_solution),("reconstruction-feedback.json",revoked_feedback),
                    ("radius-contract.json",revoked_contract)):_write(output/name,value)
                current=publish(topology_dir,topology_model,kind="topology")
                stage.update(status="candidate_rejected",accepted=False,constraint_subset_accepted=False,
                             reason="final_independent_evidence_changed_checkpoint_not_reused")
                checkpoint("parametric_retained","独立源证据发生变化，撤销尺寸检查点，保留源轮廓草稿与双方诊断。")
            return current,stage
        stage["provider"]={"status":"pending" if use_api else "disabled","network_requests":None if use_api else 0,
                           "http_success":None if use_api else False,"schema_success":False}
        checkpoint("constraint_binding","关联原图标注与候选图元；数值只取自标注，歧义保留为未绑定。")
        bindings=analyze_constraint_bindings(image_path,document,baseline,graph,output,provider=provider,use_api=use_api)
        stage["provider"]=bindings.get("provider",{})
        stage["binding_counts"]=bindings.get("counts",{})
        stage["constraints"]=bindings.get("constraints",[])
        stage["issues"]=bindings.get("issues",[])
        stage["radius_binding_coverage"]=bindings.get("radius_binding_coverage")
        checkpoint("parametric_solve","联合求解图元尺寸与连接关系，逐项核验已绑定约束。")
        solution=_solve_with_source_observation(solve_parametric,graph,stage["constraints"],baseline,output)
        _write(output/"parametric-solution.json",solution)
        feedback=reconstruction_feedback(graph,bindings,solution)
        _write(output/"reconstruction-feedback.json",feedback)
        stage["reconstruction_feedback"]=feedback
        stage["solver"]={key:value for key,value in solution.items() if key not in {"entities","nodes","candidate_entities","candidate_nodes","constraints"}}
        stage["annotation_radius_contract"]=annotation_radius_contract(bindings,solution)
        stage["annotation_radius_contract"].update(
            candidate_satisfied=stage["annotation_radius_contract"]["satisfied"],
            satisfied=False,all_annotated_radii_verified=False,current_dxf_verified=False,
            publication_status="candidate_only")
        _write(output/"radius-contract.json",stage["annotation_radius_contract"])
        if solution.get("accepted") is not True:
            stage.update(status="candidate_rejected",accepted=False,reason=solution.get("status","solver_rejected"))
            checkpoint("parametric_retained","参数候选未通过检查，保留最后一套有效轮廓和本次约束诊断。")
            return ({**current,"parameterization":copy.deepcopy(stage)} if stage.get("topology_exported") else current),stage
        candidate_dir=output/"parametric-export"
        checkpoint("parametric_validate","数值候选已产生；独立验证导出拓扑、单位与DXF回读后再发布。")
        source_check=_solved_source_validation(image_path,document,baseline,graph,solution["entities"])
        stage["solved_source_validation"]=source_check
        if not source_check["passed"]:
            failure_reason=(source_check.get("reasons") or ["solved_source_stroke_support_degraded"])[0]
            stage.update(status="candidate_rejected",accepted=False,reason=failure_reason)
            message=("求解后的边界超出原始掩膜表示误差预算；保留上一套有效轮廓。"
                     if failure_reason.startswith("original_oracle_mask_") else
                     "求解后的边界偏离原图笔画；保留上一套有效轮廓，约束残差与源图检查分别记录。")
            checkpoint("parametric_retained",message)
            return {**current,"parameterization":copy.deepcopy(stage)},stage
        # The authoritative obligations come from admitted source bindings,
        # not a solver's potentially incomplete list of residual receipts.
        export_solution={**solution,"constraints":copy.deepcopy(stage["constraints"])}
        updated=export_parametric(image_path,baseline,export_solution,candidate_dir)
        updated["validation"]["solved_source_validation"]=copy.deepcopy(source_check)
        updated["validation"]["source_mask_fidelity_used_as_acceptance_gate"]=bool(
            source_check.get("oracle_mask_validation",{}).get("applicable"))
        stage["annotation_radius_contract"]["exact_radius_validation"]=copy.deepcopy(updated["validation"]["exact_radius_validation"])
        _write(output/"radius-contract.json",stage["annotation_radius_contract"])
        updated["validation"]["annotation_radius_contract"]=copy.deepcopy(stage["annotation_radius_contract"])
        updated["validation"]["all_annotated_radii_verified"]=stage["annotation_radius_contract"]["satisfied"]
        if not stage["annotation_radius_contract"]["candidate_satisfied"]:
            updated["completion_class"]="partial_parametric_draft_unresolved_radii"
            updated.setdefault("issues",[]).append("半径标注覆盖尚未完成：未知箭头不能作为无标注豁免；已解半径与未解标注分开记录。")
        _write(candidate_dir/"validation.json",updated["validation"])
        stage.update(underconstrained=solution.get("underconstrained",True))
        selected=bindings.get("bindings",[]) if bindings.get("provider",{}).get("schema_success") is True else []
        api_dimensions=any(row.get("accepted") is True and row.get("source")=="ocr_api_binding" for row in selected)
        api_relations=any(row.get("accepted") is True and row.get("relation_id") and
                          row.get("admission_method","api_and_source")=="api_and_source" for row in selected)
        updated=publish(candidate_dir,updated,kind="parametric",api_geometry=api_dimensions or api_relations,
                        api_dimensions=api_dimensions)
        return updated,stage
    except InterruptedError as error:
        stage.update(status="interrupted",accepted=False,
                     error_type=type(error).__name__,reason="parametric_stage_interrupted_last_valid_preserved")
        diagnostic=_os_error_diagnostic(error,image_path,output)
        if diagnostic:stage["error_diagnostic"]=diagnostic
        _write(output/"parametric-stage.json",stage)
        _write_workflow_provenance(output,stage,provenance_context)
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
        diagnostic=_os_error_diagnostic(error,image_path,output)
        if diagnostic:stage["error_diagnostic"]=diagnostic
        _write(output/"parametric-stage.json",stage)
        _write_workflow_provenance(output,stage,provenance_context)
        emit("parametric_unavailable","参数化阶段未完成，已保留最后一套有效草稿及阶段证据。")
        return ({**current,"parameterization":copy.deepcopy(stage)} if stage.get("topology_exported") else current),stage
