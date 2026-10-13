"""Source topology -> sourced bindings -> numerical constraints -> CAD export.

The initial pixel trace is preserved. Parametric corrections must pass source
and dimensional checks. Explicit source-first mode retains mask failures as
diagnostics; legacy mode also requires the original oracle-mask budget.
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
import time

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
from .relation_contract import relation_checks
from .reconstruction_contract import reconstruction_contract
from .topology import stroke_support_fraction
from .source_ocr_review import canonical_document_sha256
from .reconstruction_priority import mask_is_hard


CORE = ("drawing.dxf", "preview.svg", "overlay.png", "model.json", "validation.json", "curve-fit.json")


def source_input_identity(image_path, document):
    """Bind an exported model to the exact source OCR accepted at stage entry.

    This precedes transient arrow-localization proposals. The whole effective
    OCR document is covered, including host-controlled admission metadata.
    """
    return {"schema_version": "source-input-identity-v1",
            "source_image_sha256": hashlib.sha256(Path(image_path).read_bytes()).hexdigest(),
            "effective_ocr_sha256": canonical_document_sha256(document)}


def _write(path, value):
    # A final artifact can fit Windows' path limit while ``name.json.tmp``
    # does not. Use a shorter sibling for the atomic write. The deterministic
    # per-destination name retains the existing single-writer semantics.
    temporary=path.with_name("."+hashlib.sha256(path.name.encode("utf8")).hexdigest()[:12]+".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf8")
    temporary.replace(path)


def _record_published_source_validation(model, source_check, baseline):
    validation=model.setdefault('validation',{})
    validation['solved_source_validation']=copy.deepcopy(source_check)
    validation['source_mask_fidelity_used_as_acceptance_gate']=bool(
        source_check.get('oracle_mask_validation',{}).get('applicable') and mask_is_hard(baseline))


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
    from .binding_attempts import collect_binding_attempts, binding_attempt_summary, published_binding_sources
    binding_attempts=collect_binding_attempts(output,stage)
    binding_summary=binding_attempt_summary(binding_attempts)
    _write(output/"binding-attempts.json",{"schema_version":"binding-attempt-ledger-v1",
        "attempts":binding_attempts,"summary":binding_summary})
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
    fillet=actual_stage.get("fillet_tangent_contract") or {}
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
                  contract and contract.get("satisfied") and
                  validation.get("reconstruction_contract",{}).get("satisfied"))
    strict_relations=validation.get("strict_relation_validation") or {}
    # The public sidecar follows the CURRENT coherent CORE, including rollback,
    # rather than a newer attempted candidate's coverage claim.
    published_coverage=copy.deepcopy(validation.get("reconstruction_contract") or {})
    published_coverage.update(satisfied=bool(integrity and published_coverage.get("satisfied")),
                             publication_integrity_verified=integrity,
                             prediction_dxf_sha256=hashes.get("drawing.dxf"))
    if not validation.get("reconstruction_contract"):
        published_coverage.update(status="not_certified",reasons=["current_export_has_no_reconstruction_contract"])
    _write(output/"reconstruction-contract.json",published_coverage)
    current_parametric_constraints=bool(integrity and kind=="parametric" and
        actual_stage.get("constraint_subset_accepted") and contract and
        contract.get("current_dxf_verified") and fillet.get("solver_accepted") is True and
        strict_relations.get("passed") is True and strict_relations.get("dxf_readback_performed") is True)
    provenance={"schema_version":"cad-reconstruction-provenance-v1",**context,
                "reference_dxf_used_as_prediction_geometry":False,"reference_geometry_sent_to_provider":False,
                "initial_geometry":"local_mask_to_LINE_ARC_fitting",
                "topology_selection_source":stage.get("topology",{}).get("selection_source"),
                "source_arrow_localization":stage.get("radius_target_provider",{}),
                "binding_provider":stage.get("provider",{}),
                "binding_provider_scope":"latest_selected_attempt_receipt; see binding_attempts for all calls",
                "binding_attempts":binding_attempts,"binding_attempt_summary":binding_summary,
                "published_binding_sources":published_binding_sources(actual_stage),
                "solver_requested":"parametric_solve" in stage.get("stages",[]),
                "solver_executed":bool(stage.get("solver")),
                "solver_status":stage.get("solver",{}).get("status"),
                "strict_radius_contract":contract,
                "strict_relation_contract":strict_relations,
                "reconstruction_contract":published_coverage,
                "published_artifact_kind":kind if integrity else "unverified_core",
                "published_source_input_identity":copy.deepcopy(model.get("source_input_identity")),
                "input_identity_scope":"stage_entry_input; published_source_input_identity belongs to current CORE",
                "published_core_sha256":hashes,"published_core_manifest_match":matched,
                "publication_integrity_verified":integrity,"published_metadata_consistent":metadata_consistent,
                "attempted_publication_kind":publication.get("kind"),
                "attempted_publication_status":publication.get("status"),
                "parameterization_accepted":accepted,
                "constraint_subset_accepted":bool(integrity and actual_stage.get("constraint_subset_accepted") and
                                                    contract and contract.get("current_dxf_verified")),
                "source_verified_fillet_tangency_satisfied":bool(current_parametric_constraints and
                    fillet.get("all_source_verified_joints_satisfied")),
                "source_bound_design_fillet_tangency_satisfied":bool(current_parametric_constraints and
                    fillet.get("all_design_constructed_joints_satisfied")),
                "all_fillet_line_joints_certified":bool(current_parametric_constraints and
                    fillet.get("all_line_arc_joints_certified")),
                "fillet_tangency_cohort":fillet.get("cohort", "bound_arrow_verified_radius_arcs"),
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
    from .parametric_solver import (native_measurement_checks, native_source_nominal_angle_checks,
                                    native_axis_relation_checks)
    if solution.get("accepted") is not True:
        raise ValueError("Only an accepted numerical solution may be exported")
    if solution.get("units",baseline["coordinate_system"]["units"])!=baseline["coordinate_system"]["units"]:
        raise ValueError("Solution and source units differ")
    entities=copy.deepcopy(solution["entities"])
    exact=exact_radius_checks(entities,solution.get("constraints",[]))
    if not exact["passed"]:
        raise ValueError("Annotated ARC radius must exactly equal its source value")
    relations=relation_checks(entities,solution.get("constraints",[]))
    if not relations["passed"]:
        raise ValueError("Admitted tangent relations must be strictly satisfied before export")
    measurements=native_measurement_checks(entities,solution.get("constraints",[]))
    if not measurements["passed"]:
        raise ValueError("Supplied linear measurements must be satisfied before export")
    angles=native_source_nominal_angle_checks(entities,solution.get("constraints",[]))
    if not angles["passed"]:
        raise ValueError("Supplied source angles must be strictly satisfied before export")
    axes=native_axis_relation_checks(entities,solution.get("constraints",[]))
    if not axes["passed"]:
        raise ValueError("Admitted horizontal/vertical relations must be exact before export")
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
    native_relations=relation_checks(entities,solution.get("constraints",[]),
                                    dxf_document=ezdxf.readfile(Path(output_dir)/"drawing.dxf"))
    if not native_relations["passed"]:
        raise ValueError("Exported DXF does not satisfy its strict relation obligations")
    model["validation"]["strict_relation_validation"]=native_relations
    native_measurements=native_measurement_checks(entities,solution.get("constraints",[]),
        dxf_document=ezdxf.readfile(Path(output_dir)/"drawing.dxf"))
    if not native_measurements["passed"]:
        raise ValueError("Exported DXF does not satisfy its linear measurement obligations")
    model["validation"]["native_measurement_validation"]=native_measurements
    native_angles=native_source_nominal_angle_checks(entities,solution.get("constraints",[]),
        dxf_document=ezdxf.readfile(Path(output_dir)/"drawing.dxf"))
    if not native_angles["passed"]:
        raise ValueError("Exported DXF does not satisfy its strict source angle obligations")
    model["validation"]["native_source_angle_validation"]=native_angles
    native_axes=native_axis_relation_checks(entities,solution.get("constraints",[]),
        dxf_document=ezdxf.readfile(Path(output_dir)/"drawing.dxf"))
    if not native_axes["passed"]:
        raise ValueError("Exported DXF does not satisfy its exact axis relations")
    model["validation"]["native_axis_relation_validation"]=native_axes
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
    from .reconstruction_priority import mask_is_hard, mask_conflict_receipt
    evidence=graph.get("source_evidence") or {}
    before=evidence.get("baseline_stroke_support") or {}
    after=evidence.get("proposal_stroke_support") or {}
    reasons=[]
    if (graph.get('source_line_preservation') or {}).get('passed') is False:
        reasons.append('source_finite_line_support_lost')
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
    if not mask_validation["passed"] and mask_is_hard(baseline):reasons.append(mask_validation["reason"])
    return {"passed":not reasons,"reasons":list(dict.fromkeys(reasons)),"source_evidence":evidence,"source_sha256":graph.get("source_sha256"),
            "coordinate_mapping":coordinate_check,
            "oracle_mask_validation":mask_validation,
            "priority_resolution":mask_conflict_receipt(baseline,mask_validation),
            "acceptance_basis":"source_hash_stroke_support_closed_simple_geometry_and_DXF_readback",
            "dimensions_verified":False,"reference_verified":False,
            "mask_fidelity_gate_used":mask_validation["applicable"] and mask_is_hard(baseline)}


def _solved_source_validation(image_path, document, baseline, graph, entities):
    """Check source ink again after solving, at the original fixed stroke gate."""
    from .topology import _StrokeEvidence
    from .ocr import canonical_records
    from .reconstruction_priority import mask_is_hard, mask_conflict_receipt
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
    if not mask_validation["passed"] and mask_is_hard(baseline):reasons.append(mask_validation["reason"])
    result={"passed":not reasons,"reasons":reasons,"constructed_radius_preservation":constructed,
            "oracle_mask_validation":mask_validation,
            "priority_resolution":mask_conflict_receipt(baseline,mask_validation),
            "before":before,"after":after,"maximum_support_drop":.035,"maximum_p90_increase_px":grid,
            "scope":("Source-stroke comparison and original oracle raster representation budget after numerical solving; no reference DXF comparison."
                     if mask_validation["applicable"] else "Source-stroke comparison after numerical solving; not GT or a relaxed mask-fidelity gate."),
            "reference_verified":False}
    if any((graph.get("source_evidence") or {}).get(key) for key in
           ("source_line_initial_guess", "source_state_resume")):
        from .source_topology_initial_guess import line_seed_problem_graphs
        domain,parent=line_seed_problem_graphs(graph)
        originals={"initial_topology":_solved_source_validation(image_path,document,baseline,domain,entities),
                   "original_parent":_solved_source_validation(image_path,document,baseline,parent,entities)}
        result["original_line_seed_source_comparators"]=originals
        if (graph.get("source_evidence") or {}).get("source_state_resume"):
            # A resumed numerical solution is only an initialization. Making
            # its improved ink fit a new acceptance baseline ratchets the gate
            # on every continuation and can reject a newly fixed annotation
            # that passes BOTH original immutable comparators. Keep the local
            # comparison visible, without silently changing the source gate.
            result["resumed_initialization_source_comparison"]={
                "passed":passed,"before":before,"after":after,
                "acceptance_gate":False}
            result["before"]=copy.deepcopy(originals["initial_topology"]["before"])
            result["reasons"]=[reason for reason in result["reasons"]
                if reason!="solved_source_stroke_support_degraded"]
            result["passed"]=not result["reasons"]
            result["source_comparison_basis"]="immutable_resume_domain_and_parent"
            result["acceptance_thresholds_changed"]=False
        for check in originals.values():
            if not check["passed"]:
                result["passed"]=False
                result["reasons"]=list(dict.fromkeys([*result["reasons"],*check.get("reasons",[])]))
    return result


def _verified_angle_line_growth(selected, candidate, previous_feedback):
    """Narrow offline-selection exception for a newly solved finite OCR LINE.

    The topology editor and the independent source/solver preflight must have
    accepted the candidate already. This only admits a one-LINE increase where
    a previously unbound angle is now solved; it never bypasses the normal
    branch score, source, regression or independent reference gates.
    """
    base = (selected.get("graph") or {}).get("entities") or []
    graph = candidate.get("graph") or {}
    entities = graph.get("entities") or []
    source = (graph.get("source_evidence") or {}).get("topology_edit") or {}
    after = candidate.get("constraint_feedback") or {}
    record_id = source.get("record_id")
    if (source.get("action") != "restore_annotated_line_support" or
            source.get("resegmentation_applied") is not True or
            source.get("net_entity_reduction") != -1 or
            not isinstance(record_id, str) or
            record_id not in (source.get("angle_support_record_ids") or []) or
            record_id in (previous_feedback.get("bound_record_ids") or []) or
            record_id not in (after.get("satisfied_record_ids") or []) or
            (after.get("source_validation") or {}).get("passed") is not True or
            (after.get("topology_source_validation") or {}).get("passed") is not True or
            (candidate.get("constraint_regression_gate") or {}).get("passed") is not True or
            len(entities) != len(base) + 1 or
            sum(row.get("type") == "LINE" for row in entities) !=
            sum(row.get("type") == "LINE" for row in base) + 1 or
            sum(row.get("type") == "ARC" for row in entities) !=
            sum(row.get("type") == "ARC" for row in base)):
        return False
    grid = float(graph.get("source_grid_pitch_px") or 1.)
    for entity in entities:
        evidence = entity.get("angle_support_evidence") or {}
        if (entity.get("type") != "LINE" or evidence.get("record_id") != record_id or
                evidence.get("requires_angle_binding_and_solve") is not True or
                evidence.get("ground_truth_used") is not False):
            continue
        ends = np.asarray(evidence.get("source_interval_endpoints_px"), float)
        if ends.shape == (2, 2) and np.isfinite(ends).all() and grid > 0 and math.isfinite(grid) and \
                float(np.linalg.norm(ends[1] - ends[0])) >= 4. * grid:
            return True
    return False


def _offline_edit_selection_eligible(selected, candidate, previous_feedback, base_eval, edit_eval):
    if (candidate.get("constraint_regression_gate") or {}).get("passed", True) is not True:
        return False
    score = edit_eval.get("score", 0)
    baseline_score = base_eval.get("score", 1)
    ordinary = (score >= baseline_score and
                (len(candidate["graph"]["entities"]) < len(selected["graph"]["entities"]) or
                 (candidate.get("constraint_feedback") or {}).get("issue_count", 10**9) <
                 previous_feedback.get("issue_count", 0)))
    restored = (score >= baseline_score - .02 and
                _verified_angle_line_growth(selected, candidate, previous_feedback))
    return ordinary or restored


def _source_verified_unmet_angle_line(graph, inventory, feedback, operation):
    """Reserve work only for an unmet, independently witnessed finite line."""
    if operation.get("action") != "restore_annotated_line_support":
        return False
    satisfied = (feedback or {}).get("satisfied_record_ids")
    record_id = operation.get("record_id")
    if not isinstance(satisfied, list) or not isinstance(record_id, str) or record_id in satisfied:
        return False
    from .annotation_line_support import angle_edit_evidence
    try:
        observation = angle_edit_evidence(graph, inventory, operation)
        grid = float(graph.get("source_grid_pitch_px") or 1.)
    except (ValueError, TypeError, KeyError):
        return False
    # A single ARC may swallow a genuine finite straight stroke. Scheduling
    # that proposal does not certify either its new curve tails or its angle.
    # The two-ARC constructor still requires two independent radius witnesses.
    scope_count = len(operation.get("entity_ids") or [])
    if (scope_count not in (1, 2) or
            (scope_count == 2 and observation.get("joint_radius_arrow_verified") is not True) or
            not math.isfinite(grid) or grid <= 0):
        return False
    requested = set(operation.get("entity_ids") or [])
    return any(row.get("entity_id") in requested and
               row.get("whole_line_supported") is not True and
               type(row.get("supported_span_px")) in (int, float) and
               math.isfinite(row["supported_span_px"]) and
               row["supported_span_px"] >= 4. * grid
               for row in observation.get("target_candidates", []))


def _ordered_topology_operations(agent_operations, local_operations, graph, inventory,
                                  feedback, operation_filter=None):
    """Keep provider order while guaranteeing one source-witnessed line trial."""
    reserved = next((row for row in local_operations
                     if _source_verified_unmet_angle_line(graph, inventory, feedback, row)), None)
    agent = list(agent_operations)
    if reserved is not None:
        signature = lambda row: (row.get("action"), tuple(row.get("entity_ids") or []), row.get("record_id"))
        agent = [row for row in agent if signature(row) != signature(reserved)]
        agent.insert(0, reserved)
    operations, seen, skipped_visited = [], set(), []
    for operation in [*agent, *local_operations]:
        ids = tuple(operation.get("entity_ids") or [])
        key = (operation.get("action"), ids, operation.get("record_id"))
        if key in seen:
            continue
        if operation_filter is not None and not operation_filter(operation):
            skipped_visited.append({"operation": operation, "status": "skipped",
                                    "reason": "visited_parent_operation"})
            seen.add(key)
            continue
        operations.append(operation)
        seen.add(key)
        if len(operations) >= 5:
            break
    return operations, skipped_visited


def _pending_source_line_repairs(candidate, inventory, feedback, visited):
    """Schedule finite source-witnessed LINE obligations before optional search.

    This is only a work-order decision. The ordinary edit kernel, source
    checks, binding, solver and regression gates still certify every child.
    Already attempted parent/operation pairs do not suppress the editor on
    later rounds, and newly observed source hints stay off the certified graph.
    """
    from .annotation_line_support import propose_annotation_line_edits
    from .topology_search import operation_fingerprint

    graph = _unmet_edit_observation_parent(candidate,feedback)["graph"]
    parent_hash = geometry_fingerprint(candidate["graph"])
    return _source_line_repair_operations(graph,inventory,feedback,
        lambda operation:operation_fingerprint(parent_hash,operation) not in visited)


def _source_line_repair_operations(graph,inventory,feedback,operation_filter=None):
    """Select from the complete bounded LINE scope list before its top-1 cap."""
    from .annotation_line_support import propose_annotation_line_edits
    # Filtering after top-1 allowed an already solved angle or a visited broad
    # scope to hide the next finite, source-owned LINE repair. Enumerate the
    # existing at-most-three scopes per observed target, then reserve one fresh
    # eligible operation. This changes scheduling, never evidence admission.
    scopes=max(1,3*sum(len(row.get("target_candidates") or [])
                      for row in graph.get("angle_source_observations") or []))
    # The production filter reserves a parent/operation pair. Do not evaluate
    # it on scopes that this top-1 trial will never execute.
    for operation in propose_annotation_line_edits(graph, inventory, limit=scopes):
        if (_source_verified_unmet_angle_line(graph, inventory, feedback, operation)
                and (operation_filter is None or operation_filter(operation))):
            return [operation]
    return []


def _source_binding_locks(graph, bindings, solution, feedback):
    """Remember solved source nominals and their object lineage, never fitted values."""
    from .parametric_solver import _strict_source_nominal

    if (solution.get("accepted") is not True or feedback.get("solver_accepted") is not True
            or feedback.get("post_solve_source_accepted") is not True):
        return []
    entities={row["id"]:row for row in graph.get("entities",[])}
    passed={row.get("id") for row in solution.get("constraints",[]) if row.get("passed") is True}
    verified_radii=set(feedback.get("verified_radius_record_ids") or [])
    locks=[]
    for row in bindings.get("constraints",[]):
        record_id=row.get("record_id")
        if not isinstance(record_id,str) or row.get("id") not in passed:
            continue
        if row.get("kind")=="radius":
            if record_id not in verified_radii:continue
        elif not (_strict_source_nominal(row) or
                  (row.get('source_tolerance') or {}).get('policy')=='source_tolerance_interval'
                  and row.get('required') is True and row.get('nominal_source')=='source_ocr'
                  and row.get('source') in {'ocr_api_binding','ocr_local_binding'}):
            continue
        targets=[entities.get(entity_id,{}) for entity_id in row.get("entities",[])]
        # Production source graphs carry stable IDs. Legacy/adaptor graphs
        # still lock exact semantics, but cannot claim a lineage certificate.
        stable_ids=[target.get("stable_id") for target in targets]
        complete=bool(stable_ids and all(isinstance(sid,str) for sid in stable_ids))
        locks.append({"record_id":record_id,"kind":row.get("kind"),
            "value":row.get("value"),"angle_mode":row.get("angle_mode"),
            "reference_axis":row.get("reference_axis"),
            "strict_source_nominal":_strict_source_nominal(row),
            "target_stable_ids":stable_ids if complete else [],
            "target_lineage_available":complete,
            "fresh_binding_and_solve_verified":True})
        if row.get('source_tolerance') is not None:
            locks[-1]['source_tolerance']=copy.deepcopy(row['source_tolerance'])
    return locks


def _source_locked_constraint_regression(before, after):
    """Do not trade a certified annotation's value/owner for a new obligation.

    Display IDs may change. A freshly rebound descendant can carry the old
    nominal after a split; unrelated objects and same-count value changes
    cannot. The caller's fresh source and joint-solve gates remain mandatory.
    """
    gate=constraint_regression(before,after)
    previous=before.get("source_binding_locks") or []
    current=after.get("source_binding_locks") or []
    ancestry=after.get("entity_ancestry") or {}
    failures=[]
    semantic_keys=("record_id","kind","value","angle_mode","reference_axis","strict_source_nominal","source_tolerance")
    for old in previous:
        semantic=[row for row in current if row.get("fresh_binding_and_solve_verified") is True
                  and all(row.get(key)==old.get(key) for key in semantic_keys)]
        if not semantic:
            failures.append({"record_id":old["record_id"],"reason":"verified_source_nominal_changed_or_unsolved"})
            continue
        required=set(old.get("target_stable_ids") or [])
        if required and not any(required.issubset({ancestor for sid in row.get("target_stable_ids") or []
                for ancestor in [sid,*ancestry.get(sid,[])]}) for row in semantic):
            failures.append({"record_id":old["record_id"],"reason":"verified_source_binding_target_lineage_lost"})
    gate["source_binding_lock_failures"]=failures
    if failures:
        gate["passed"]=False
        gate["reasons"]=list(dict.fromkeys([*gate.get("reasons",[]),
                                            *[row["reason"] for row in failures]]))
    return gate


def _local_edit_branch_gate(selected, candidate, base_eval, edit_eval, *, online_evaluator=False):
    """Apply the same source-only local improvement gate during scheduling and selection."""
    base_metrics=base_eval.get("metrics",{});edit_metrics=edit_eval.get("metrics",{})
    edit_execution=((candidate.get("graph") or {}).get("source_evidence") or {}).get("topology_edit") or {}
    edit_actions=[edit_execution.get("action")]
    edit_actions.extend(row.get("action") for row in edit_execution.get("operations",[]) if isinstance(row,dict))
    validated_partition=bool(edit_execution.get("resegmentation_applied") and
                             edit_execution.get("radius_binding_applied") and
                             len(edit_execution.get("bound_record_ids") or [])>=2)
    annotation_guided=bool(set(edit_actions)&{"refit_chain_as_annotated_arc","insert_annotated_fillet","restore_annotated_line_support"} or
                           validated_partition)
    count_improved=edit_metrics.get("entity_count",10**9) < base_metrics.get("entity_count",0)
    count_preserved_for_annotation=bool(
        annotation_guided and edit_metrics.get("entity_count",10**9) <= base_metrics.get("entity_count",0))
    growth_operations=[row for row in (edit_execution.get("operations") or [edit_execution])
                       if isinstance(row,dict) and row.get("net_entity_reduction",0)<0]
    satisfied_records=set((candidate.get("constraint_feedback") or {}).get("satisfied_record_ids") or [])
    feature_restored=bool(growth_operations) and all(
        (row.get("radius_binding_applied") and (row.get("feature_restoration_validated") or
            (row.get("resegmentation_applied") and len(row.get("bound_record_ids") or [])>=2))) or
        (row.get("action")=="restore_annotated_line_support" and row.get("resegmentation_applied") is True
         and row.get("record_id") in (row.get("angle_support_record_ids") or [])
         and row.get("record_id") in satisfied_records)
        for row in growth_operations)
    type_corrected=bool("refit_entity_as_line" in edit_actions and
                        edit_metrics.get("source_boundary_support",0)>=base_metrics.get("source_boundary_support",0) and
                        edit_metrics.get("entity_count")==base_metrics.get("entity_count"))
    regression=candidate.get("constraint_regression_gate",{"passed":True})
    accepted=bool(
        edit_eval.get("score") is not None and base_eval.get("score") is not None and
        edit_eval["score"] >= base_eval["score"]-.02 and
        edit_metrics.get("source_boundary_support",0.) >= base_metrics.get("source_boundary_support",0.)-.02 and
        edit_metrics.get("unsupported_primitive_count",10**9) <= base_metrics.get("unsupported_primitive_count",0)+1 and
        (count_improved or count_preserved_for_annotation or feature_restored or type_corrected) and
        regression.get("passed") is True)
    return {"accepted":accepted,
            "reason":None if accepted else "edited_candidate_failed_local_improvement_gate",
            "proposed_candidate_id":candidate.get("id"),"base_candidate_id":selected.get("id"),
            "minimum_local_score":round(float(base_eval.get("score",0.))-.02,9),
            "minimum_source_boundary_support":max(0.,float(base_metrics.get("source_boundary_support",0.))-.02),
            "maximum_unsupported_primitive_count":int(base_metrics.get("unsupported_primitive_count",0))+1,
            "must_reduce_entity_count":not (annotation_guided or feature_restored or type_corrected),
            "annotation_guided_primitive_correction":annotation_guided,
            "source_validated_feature_restoration":feature_restored,
            "constraint_regression":regression,
            "selection_source":"online_evaluator" if online_evaluator else "strict_local_improvement"}


def _source_verified_primary_progress(selected, candidate, previous_feedback):
    """Only a fully preflighted, nondegenerate improvement may reserve the next round."""
    from .planning_provider import evaluate_candidates
    from .topology_search import preflight_admissible
    report=candidate.get("constraint_feedback") or {}
    if (not preflight_admissible(report) or report.get("solver_accepted") is not True or
            report.get("post_solve_source_accepted") is not True or
            (report.get("source_validation") or {}).get("passed") is not True or
            (report.get("topology_source_validation") or {}).get("passed") is not True or
            (candidate.get("constraint_regression_gate") or {}).get("passed") is not True or
            geometry_fingerprint(candidate["graph"])==geometry_fingerprint(selected["graph"])):
        return False
    before=set(previous_feedback.get("satisfied_record_ids") or [])
    after=set(report.get("satisfied_record_ids") or [])
    record_gained=bool(after>before)
    base_count=len((selected.get("graph") or {}).get("entities") or [])
    candidate_count=len((candidate.get("graph") or {}).get("entities") or [])
    simpler=bool(candidate_count<base_count and
                 type(previous_feedback.get("issue_count")) is int and
                 type(report.get("issue_count")) is int and
                 report["issue_count"]<previous_feedback["issue_count"])
    if not (record_gained or simpler):
        return False
    local=evaluate_candidates([selected,candidate],max_candidates=2)
    admissible=set(local.get("admissible_candidate_ids") or [])
    if selected["id"] not in admissible or candidate["id"] not in admissible:
        return False
    evaluated={row["candidate_id"]:row for row in local.get("evaluated",[]) if row.get("admissible")}
    return _local_edit_branch_gate(selected,candidate,
                                   evaluated[selected["id"]],evaluated[candidate["id"]])["accepted"]


def _topology_edit_stage(image_path, document, baseline, selected, bundle, output, *,
                          editor_provider=None, evaluator_provider=None, use_api=False,
                          round_index=1, feedback=None, verifier=None, operation_filter=None,
                          candidate_namespace=None, provider_guard=None,
                          defer_after_primary_progress=False, defer_for_source_exploration=False,
                          source_line_preflight_first=False):
    """Propose, execute and independently evaluate bounded local topology edits."""
    from .planning_provider import evaluate_candidates
    from .topology_editing import execute_topology_edits, propose_annotation_arc_edits
    from .topology_search import preflight_admissible, preflight_order

    # SearchBudget.visit_operation is a stateful reservation. Scheduling,
    # priority selection and ordering inspect the same operation several
    # times; reuse its first decision within this stage, rather than rejecting
    # our own reservation as a previously executed edit. A new stage gets a
    # fresh cache, so genuinely visited operations still stay excluded.
    if operation_filter is not None:
        reserve_operation = operation_filter
        reservation_decisions = {}
        def operation_filter(operation):
            key = (operation.get("action"), tuple(operation.get("entity_ids") or []),
                   operation.get("record_id"))
            if key not in reservation_decisions:
                reservation_decisions[key] = bool(reserve_operation(operation))
            return reservation_decisions[key]

    # The current candidate was preflighted without the newly discovered
    # source coordinates. They may suggest edits on this temporary parent,
    # whose children still require their own full source/numeric preflight.
    edit_parent=_unmet_edit_observation_parent(selected,feedback)
    unresolved=((feedback or {}).get("radius_binding_coverage") or {}).get("unresolved",[])
    local_annotation_operations=propose_annotation_arc_edits(
        edit_parent.get("graph") or {},bundle.get("annotation_inventory",[]),limit=4,
        unresolved_radius_record_ids={row["record_id"] for row in unresolved
                                      if isinstance(row,dict) and isinstance(row.get("record_id"),str)})
    # The generic ARC proposer reserves only one LINE scope, possibly an
    # already visited broad scope. Use the scheduler's complete scope list in
    # this priority stage rather than silently reapplying that top-1 cap.
    if source_line_preflight_first:
        reserved=_source_line_repair_operations(edit_parent.get("graph") or {},
            bundle.get("annotation_inventory",[]),feedback,operation_filter)
        if reserved:local_annotation_operations=reserved
    mandatory_line_trial=any(_source_verified_unmet_angle_line(
        edit_parent.get("graph") or {},bundle.get("annotation_inventory",[]),feedback,row)
        for row in local_annotation_operations)
    prioritize_source_line=bool(source_line_preflight_first and mandatory_line_trial and verifier)
    if prioritize_source_line:
        # Even construction and source checks for unrelated local edits can
        # consume the shared clock. Give the already witnessed missing LINE
        # its own first preflight; other hypotheses remain eligible next round.
        local_annotation_operations=[row for row in local_annotation_operations
            if _source_verified_unmet_angle_line(edit_parent.get("graph") or {},
                bundle.get("annotation_inventory",[]),feedback,row)
            and (operation_filter is None or operation_filter(row))][:1]
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
    if prioritize_source_line:
        editor_receipt.update(status="skipped",reason="source_line_numeric_preflight_prioritized")
    elif use_api and editor_provider is not None and overlay.is_file():
        try:
            options={"feedback":feedback} if "feedback" in inspect.signature(editor_provider.propose).parameters else {}
            remaining=provider_guard("editor") if provider_guard else 600.
            if mandatory_line_trial and remaining is not None:
                # An optional editor call cannot consume the final opportunity
                # to bind and solve an already source-witnessed missing LINE.
                # This reserves time inside the existing global ceiling.
                remaining=remaining-120. if remaining>125. else None
            if remaining is not None:
                if "request_timeout_seconds" in inspect.signature(editor_provider.propose).parameters:
                    options["request_timeout_seconds"]=remaining
                editor_receipt=editor_provider.propose(image_path,overlay,edit_parent,bundle.get("annotation_inventory",[]),**options)
            else:
                editor_receipt.update(status="skipped",reason=(
                    "source_repair_preflight_time_reserved" if mandatory_line_trial else
                    "topology_search_provider_budget_exhausted"))
        except InterruptedError:
            raise
        except Exception:
            editor_receipt.update(status="failed",reason="topology_edit_provider_error",
                                  schema_success=False,network_requests=None)
    agent_operations=(editor_receipt.get("operations",[])
                      if editor_receipt.get("schema_success") is True else [])
    operations,skipped_visited=_ordered_topology_operations(
        agent_operations,local_annotation_operations,edit_parent.get("graph") or {},
        bundle.get("annotation_inventory",[]),feedback,operation_filter)
    if operations:
        try:
            edited,execution=execute_topology_edits(image_path,document,baseline,edit_parent,bundle,operations,output)
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
    preflight_queue=[]
    deferred_preflight_candidate_ids=[]
    progress_candidate_id=None
    exploration_candidate_id=None
    completed_source_preflights=0
    if verifier:
        # Source checks are bounded by the edit kernel's five operations plus
        # one combined candidate. Check all before costly binding and solving,
        # so materialization order cannot hide the angular or combined repair.
        source_checks={row["id"]:_topology_source_validation(image_path,baseline,row["graph"])
                       for row in edited}
        queued=preflight_order(edited,source_checks)
        def source_valid_primary(row, action):
            from .source_topology_initial_guess import line_seed_admission
            edit=(((row.get("graph") or {}).get("source_evidence") or {}).get("topology_edit") or {})
            may_solve=(source_checks[row["id"]].get("passed") is True or
                       line_seed_admission(row.get("graph") or {},source_checks[row["id"]])["eligible"])
            return (may_solve and
                    edit.get("action")==action and
                    (action!="restore_annotated_line_support" or
                     (edit.get("resegmentation_applied") is True and edit.get("angle_support_record_ids"))))
        primary_ids=set()
        for action in ("restore_annotated_line_support","apply_nonoverlapping_edits"):
            primary=next((row for row in queued if source_valid_primary(row,action)),None)
            if primary is not None:primary_ids.add(primary["id"])
        completed_primary_ids=set()
        accepts_source_check="source_validation" in inspect.signature(verifier).parameters
        accepts_parent="parent_candidate" in inspect.signature(verifier).parameters
        for row in queued:
            source_check=source_checks[row["id"]]
            certified_deferral=bool(defer_after_primary_progress and primary_ids and
                                    primary_ids<=completed_primary_ids and progress_candidate_id)
            # Numerical infeasibility of one unfinished edit is not proof that
            # its source-valid topology cannot become feasible after another
            # edit. Reserve a later round instead of exhausting every sibling.
            # The intermediate remains diagnostic, with no publishable coverage.
            exploratory_deferral=bool(defer_for_source_exploration and exploration_candidate_id and
                                      completed_source_preflights>=2 and
                                      primary_ids<=completed_primary_ids)
            deferred=bool((certified_deferral or exploratory_deferral) and
                          source_check.get("passed") is True)
            if deferred:
                reason="topology_preflight_deferred_for_next_round"
                report=reconstruction_feedback(row["graph"])
                report.update(solver_status="preflight_not_run",preflight_status="deferred_not_run",
                              preflight_skip_reason=reason,binding_status="not_run",constraint_count=0,
                              solver_accepted=False,post_solve_source_accepted=False,
                              source_validation={"status":"not_run","passed":None,"reasons":[]},
                              topology_source_validation=source_check)
                row["constraint_feedback"]=report
                row["constraint_regression_gate"]={"passed":False,"reasons":[reason],
                                                    "comparison_status":"not_run"}
                deferred_preflight_candidate_ids.append(row["id"])
                queue_status="deferred_not_run"
            else:
                verify_options={}
                if accepts_source_check:verify_options["source_validation"]=source_check
                if accepts_parent:verify_options["parent_candidate"]=selected
                row["constraint_feedback"]=verifier(row,**verify_options)
                report=row["constraint_feedback"]
                # A successful initial-guess repair creates a new graph whose
                # source check was actually recomputed. Preserve the old
                # rejection in its separate receipt rather than relabel it.
                if row.get("source_initial_guess_refinement"):
                    source_check=report.get("topology_source_validation") or source_check
                    source_checks[row["id"]]=source_check
                if report.get("solver_status")=="preflight_not_run":
                    row["constraint_regression_gate"]={"passed":False,
                        "reasons":[report.get("preflight_skip_reason") or "topology_search_preflight_not_run"]}
                    queue_status="not_run_budget"
                else:
                    row["constraint_regression_gate"]=_source_locked_constraint_regression(feedback or {},report)
                    queue_status="passed" if row["constraint_regression_gate"]["passed"] else "rejected"
                    if source_check.get("passed") is True:
                        completed_source_preflights+=1
                        if (not preflight_admissible(report) and report.get("binding_status")!="not_run" and
                                geometry_fingerprint(row["graph"])!=geometry_fingerprint(selected["graph"])):
                            exploration_candidate_id=exploration_candidate_id or row["id"]
                if row["id"] in primary_ids:
                    completed_primary_ids.add(row["id"])
                    if (defer_after_primary_progress and
                            _source_verified_primary_progress(selected,row,feedback or {})):
                        progress_candidate_id=row["id"]
            preflight_queue.append({"candidate_id":row["id"],"source_validation_passed":source_check.get("passed") is True,
                                     "status":queue_status,"reasons":row["constraint_regression_gate"]["reasons"]})
        by_candidate={row["id"]:row for row in edited}
        for operation in execution.get("operations",[]):
            row=by_candidate.get(operation.get("candidate_id"))
            if row is None:continue
            report=row["constraint_feedback"]
            operation["preflight_status"]=("deferred_not_run" if report.get("preflight_status")=="deferred_not_run" else
                "not_run_budget" if report.get("solver_status")=="preflight_not_run" else
                "passed" if row["constraint_regression_gate"]["passed"] else "rejected")
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
                  _offline_edit_selection_eligible(selected,row,feedback or {},base_eval,evaluated[row["id"]])]
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
        branch_gates[candidate_id]=_local_edit_branch_gate(
            selected,by_id[candidate_id],base_eval,edit_eval,
            online_evaluator=bool(evaluator_receipt.get("schema_success")))
    if proposed_id==selected.get("id") and proposed_id in admissible:
        gate.update(reason="evaluator_preserved_base")
    elif proposed_id in branch_gates:
        gate=branch_gates[proposed_id]
        if gate["accepted"]:final=by_id[proposed_id]
    # An online preference cannot erase a newly solved, source-witnessed OCR
    # LINE after the same source, exact-radius and regression gates have passed.
    protected=[row for row in edited if row["id"] in admissible and
               branch_gates.get(row["id"],{}).get("accepted") is True and
               _verified_angle_line_growth(selected,row,feedback or {})]
    missing=[row for row in protected
             if (((row["graph"].get("source_evidence") or {}).get("topology_edit") or {})
                 .get("record_id") not in
                 ((final.get("constraint_feedback") or (feedback if final is selected else {}))
                  .get("satisfied_record_ids") or []))]
    if missing:
        restored=max(missing,key=lambda row:evaluated[row["id"]]["score"])
        provider_selected=proposed_id
        proposed_id=restored["id"]
        final=restored
        gate={**branch_gates[proposed_id],
              "selection_source":"source_verified_required_ocr_topology",
              "provider_selected_candidate_id":provider_selected}
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
             "source_line_preflight_prioritized":prioritize_source_line,
             "preflight_queue":preflight_queue,
              "deferred_preflight_candidate_ids":deferred_preflight_candidate_ids,
              "deferred_for_next_round_after_candidate_id":progress_candidate_id if deferred_preflight_candidate_ids else None,
              "deferred_for_exploration_after_candidate_id":exploration_candidate_id if deferred_preflight_candidate_ids and defer_for_source_exploration else None,
              "completed_source_preflights":completed_source_preflights,
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
        from .reconstruction_priority import mask_is_hard
        if not mask_is_hard(baseline):
            observation["mask_representation_budget_diagnostic"] = observation.pop("boundary_error_budget")
            observation["mask_role"] = "initial_topology_and_soft_localization"
            observation['feasibility_first_initialization']=True
            observation["reference_thresholds_changed"] = False
            observation["optimization_budget"]={"max_wall_seconds":120.,"max_objective_evaluations":30000}
    return observation


def _solve_with_source_observation(solver, graph, constraints, baseline, output, *, budget_seconds=None,
                                   budget_evaluations=None, seed_node_offsets=None, image_stroke_prior=None,
                                   seed_entities=None):
    # Explicit signature compatibility keeps external solver adapters usable;
    # the production solver always declares the observation argument.
    from .source_topology_initial_guess import line_seed_problem_graphs
    domain,_=line_seed_problem_graphs(graph)
    if domain is not graph:
        if seed_entities is None:seed_entities=graph["entities"]
        graph=domain
    options={"output_dir":output}
    if "source_observation" in inspect.signature(solver).parameters:
        options["source_observation"]=_solver_source_observation(baseline,graph)
        observation=options["source_observation"]
        if observation is not None and image_stroke_prior is not None:
            observation["image_stroke_prior"]=image_stroke_prior
        if budget_seconds is not None and observation and "boundary_error_budget" in observation:
            observation["boundary_error_budget"]["max_wall_seconds"]=min(120.,float(budget_seconds))
        if budget_evaluations is not None and observation and "boundary_error_budget" in observation:
            observation["boundary_error_budget"]["max_objective_evaluations"]=min(30000,int(budget_evaluations))
        if observation and 'optimization_budget' in observation:
            if budget_seconds is not None:observation['optimization_budget']['max_wall_seconds']=min(120.,float(budget_seconds))
            if budget_evaluations is not None:observation['optimization_budget']['max_objective_evaluations']=min(30000,int(budget_evaluations))
    if seed_node_offsets is not None:
        if "seed_node_offsets" not in inspect.signature(solver).parameters:
            raise ValueError("Source restart requires a solver with bounded seed support")
        options["seed_node_offsets"]=seed_node_offsets
    if seed_entities is not None and "seed_entities" in inspect.signature(solver).parameters:
        options["seed_entities"]=seed_entities
    return solver(graph,constraints,**options)


def _source_restart_offsets(image_path, document, baseline, graph, entities, sampling_step, *,
                            recover_stroke_support=False):
    """Two deterministic, local starting offsets from the immutable inputs.

    The largest source-mask discrepancy identifies the first joint to probe;
    original-image stroke evidence is the fallback. Neither supplies a new
    dimension or an acceptance verdict. The solver keeps the same graph and
    exact constraints for both starts.
    """
    from .reconstruction_feedback import source_mask_interval_diagnostics
    from .source_support_diagnostics import source_support_diagnostics
    by_id={entity["id"]:entity for entity in entities}
    entity_id=None;quarter=None;direction=None;basis=None
    if recover_stroke_support:
        # Probe the region that LOST ink support, rather than a permanently
        # unsupported mask closure or the largest unrelated mask discrepancy.
        # Both measurements use the same original image and OCR exclusions.
        before=source_support_diagnostics(image_path,document,baseline,graph)
        after=source_support_diagnostics(image_path,document,baseline,{**graph,"entities":entities})
        previous={row["entity_id"]:row for row in before.get("entities",[])}
        losses=[]
        if before.get("status")==after.get("status")=="measured":
            for row in after.get("entities",[]):
                prior=previous.get(row.get("entity_id"))
                if prior is None or row.get("entity_id") not in by_id:continue
                prior_parts={part["quarter"]:part for part in prior.get("quarters",[])}
                for part in row.get("quarters",[]):
                    old=prior_parts.get(part["quarter"])
                    if old is None:continue
                    loss=float(old["stroke_supported_fraction"])-float(part["stroke_supported_fraction"])
                    if loss>0:
                        losses.append((loss*float(row["sample_count"])/4,loss,row["entity_id"],int(part["quarter"])))
        if losses:
            _,_,entity_id,quarter=max(losses)
            basis="largest_original_image_ink_support_loss_from_same_source_graph"
    try:
        local=source_mask_interval_diagnostics(baseline,graph,entities)
        for row in local.get("entities",[]):
            if entity_id is not None:break
            if row.get("entity_id") not in by_id:continue
            source=np.asarray(row.get("worst_source_point_px"),float)
            candidate=np.asarray(row.get("closest_candidate_point_px"),float)
            if source.shape!=(2,) or candidate.shape!=(2,) or not np.isfinite(source).all() or not np.isfinite(candidate).all():
                continue
            from .topology_candidates import _affines
            _,source_to_design,_=_affines(graph,baseline)
            direction=source_to_design(source[None,:])[0]-source_to_design(candidate[None,:])[0]
            if np.linalg.norm(direction)>1e-12:
                entity_id=row["entity_id"];quarter=int(row["worst_source_interval_quarter"])
                basis="largest_immutable_input_mask_interval_deviation"
                break
    except (KeyError,TypeError,ValueError,IndexError,ArithmeticError):
        pass
    if entity_id is None:
        diagnostic=source_support_diagnostics(image_path,document,baseline,{**graph,"entities":entities})
        if diagnostic.get("status")!="measured":return [],{"status":"unavailable"}
        candidates=[]
        for row in diagnostic.get("entities",[]):
            entity=by_id.get(row.get("entity_id"))
            if entity is None:continue
            for part in row.get("quarters",[]):
                candidates.append((float(part["stroke_supported_fraction"]),
                                   -float(part["p90_edge_distance_px"]),entity["id"],int(part["quarter"])))
        if not candidates:return [],{"status":"no_source_quarters"}
        _,_,entity_id,quarter=min(candidates)
        basis="lowest_original_image_stroke_supported_quarter"
    entity=by_id[entity_id]
    if direction is None:
        chord=np.asarray(entity["end"],float)-np.asarray(entity["start"],float)
        direction=np.array([-chord[1],chord[0]])
    length=float(np.linalg.norm(direction))
    if not math.isfinite(length) or length<=1e-12:return [],{"status":"degenerate_weak_primitive"}
    normal=direction/length
    shift=min(2*float(sampling_step),.25 if graph["units"]=="mm" else 2.)
    if not math.isfinite(shift) or shift<=0:return [],{"status":"invalid_source_step"}
    node=entity["start_node"] if quarter<=2 else entity["end_node"]
    offset_nodes=([entity["start_node"],entity["end_node"]] if recover_stroke_support and
                  basis=="largest_original_image_ink_support_loss_from_same_source_graph" else [node])
    offsets=[{nid:(polarity*shift*normal).tolist() for nid in offset_nodes} for polarity in (1.,-1.)]
    return offsets,{"status":"prepared","basis":basis,
                    "entity_id":entity_id,"quarter":quarter,"node_id":node,"node_ids":offset_nodes,"offset_magnitude":shift,
                    "units":graph["units"],"ground_truth_used":False}


def _source_solution_score(source_check):
    """Compare admissible candidates without opening a reference DXF."""
    before=source_check["before"];after=source_check["after"]
    mask=source_check["oracle_mask_validation"]
    return {"mask_max_deviation_px":float(mask["conservative_max_deviation_px"]),
            "stroke_supported_fraction":stroke_support_fraction(after,against=before),
            "p90_edge_distance_px":float(after["p90_edge_distance_px"])}


def _source_score_improves(candidate,incumbent):
    """Demand a source-evidence Pareto improvement; never trade a failed gate."""
    mask=candidate["mask_max_deviation_px"]-incumbent["mask_max_deviation_px"]
    support=candidate["stroke_supported_fraction"]-incumbent["stroke_supported_fraction"]
    p90=candidate["p90_edge_distance_px"]-incumbent["p90_edge_distance_px"]
    return (mask<=1e-6 and support>=-1e-6 and p90<=1e-6 and
            (mask< -1e-4 or support>1e-4 or p90< -1e-4))


def _source_feasibility_restart_admission(graph,constraints,solution):
    """A failed source search may be explored, but never published as a seed.

    Recompute geometry and dimensional evidence before using its weakest
    source interval to choose a bounded starting perturbation. No proposed
    coordinates, branch, radius or constraint replace the original problem.
    """
    from .parametric_solver import (_validate_inputs, _model, _geometry_validation,
        _sample_entity, _constraint_value, _constraint_residual, _strict_measurement_row,
        _strict_source_nominal, _strict_ocr_axis_angle, native_measurement_checks,
        native_source_nominal_angle_checks, MAX_DISPLACEMENT_FRACTION,
        LINEAR_TOLERANCE_MM, ANGLE_TOLERANCE_DEG, STRICT_OCR_ANGLE_CERT_TOLERANCE_DEG,
        STRICT_MEASUREMENT_CERT_TOLERANCE_MM, STRICT_TANGENT_CERT_TOLERANCE_DEG)
    audit={"eligible":False,"reason":"not_a_source_only_feasibility_failure",
           "seed_is_publishable":False,"reference_geometry_used":False}
    validation=solution.get("validation") or {}
    if (solution.get("accepted") is True or solution.get("status") not in
            {"source_budget_search_failed","source_budget_search_exhausted"} or
            validation.get("source_boundary_budget_passed") is not False or
            any(validation.get(key) is not True for key in
                ("geometry_valid","constraint_subset_satisfied","exact_radius_chords_feasible",
                 "source_displacement_passed","winding_preserved"))):
        return audit
    try:
        if not constraints or not _constraint_receipts_complete({"constraints":constraints},solution,graph):
            raise ValueError("constraint_receipts_incomplete")
        candidate=solution.get("candidate_entities");candidate_nodes=solution.get("candidate_nodes")
        nodes,baseline,prepared=_validate_inputs(graph,constraints)
        actual_graph={**graph,"entities":candidate,"nodes":candidate_nodes}
        actual_nodes,entities,_=_validate_inputs(actual_graph,constraints)
        identity=lambda entity:tuple(entity.get(key) for key in
            ("id","type","start_node","end_node","clockwise"))
        if [identity(entity) for entity in entities]!=[identity(entity) for entity in baseline]:
            raise ValueError("candidate_changed_topology_or_branch")
        _,_,_,_,scale,_,_,branches=_model(nodes,baseline)
        before=_geometry_validation(baseline,branches,scale)
        after=_geometry_validation(entities,branches,scale)
        if not after["passed"] or before["signed_area"]*after["signed_area"]<=0:
            raise ValueError("candidate_geometry_or_winding_invalid")
        limit=max(.25 if graph["units"]=="mm" else 2.,scale*MAX_DISPLACEMENT_FRACTION)
        if any(float(np.linalg.norm(_sample_entity(first)-_sample_entity(second),axis=1).max())>limit
               for first,second in zip(baseline,entities)):
            raise ValueError("candidate_source_displacement_invalid")
        indexed={entity["id"]:entity for entity in entities}
        points={node["id"]:np.array([node["x"],node["y"]]) for node in actual_nodes}
        for row in prepared:
            residual=abs(_constraint_residual(row,_constraint_value(row,points,indexed)))
            tolerance=(0. if row["kind"]=="radius" else
                STRICT_OCR_ANGLE_CERT_TOLERANCE_DEG if row["kind"]=="angle" and
                    (_strict_source_nominal(row) or _strict_ocr_axis_angle(row)) else
                STRICT_MEASUREMENT_CERT_TOLERANCE_MM if _strict_measurement_row(row) else
                STRICT_TANGENT_CERT_TOLERANCE_DEG if row["kind"]=="tangent" else
                LINEAR_TOLERANCE_MM if row["kind"].startswith("distance") else ANGLE_TOLERANCE_DEG)
            if not math.isfinite(residual) or residual>tolerance:
                raise ValueError("candidate_actual_constraint_failed")
        audits=(exact_radius_checks(entities,constraints),relation_checks(entities,constraints),
                native_measurement_checks(entities,constraints),native_source_nominal_angle_checks(entities,constraints))
        if not all(row.get("passed") is True for row in audits):
            raise ValueError("candidate_independent_constraint_audit_failed")
    except (ValueError,TypeError,KeyError,IndexError,ArithmeticError) as error:
        audit["reason"]=str(error) if isinstance(error,ValueError) else "candidate_audit_unavailable"
        return audit
    return {**audit,"eligible":True,"reason":"strict_subset_valid_source_budget_unmet",
            "candidate_geometry_independently_checked":True,"candidate_constraints_independently_checked":True}


def _final_source_multistart(image_path,document,baseline,graph,constraints,output,*,solver,
                             budget_seconds=None,initial_seed_entities=None):
    """Bounded final solve: recover source feasibility or refine a valid result.

    At most three solves share the existing 120-second/30000-objective budget.
    The topology beam never calls this helper.  Any restart must independently
    satisfy the unchanged solver, original-image, mask and DXF-readback gates.
    """
    from .source_topology_initial_guess import line_seed_problem_graphs
    source_comparator_graph=graph
    domain,_=line_seed_problem_graphs(graph)
    if domain is not graph:
        if initial_seed_entities is None:initial_seed_entities=graph["entities"]
        graph=domain
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    observation=_solver_source_observation(baseline,graph)
    ceiling=min(120.,float(budget_seconds)) if budget_seconds is not None else 120.
    if not math.isfinite(ceiling) or ceiling<=0:raise ValueError("Final solve needs a positive finite budget")
    deadline=time.monotonic()+ceiling
    ink_observation=None
    if observation is not None:
        from .source_stroke_prior import source_stroke_observation
        try:
            ink_observation=source_stroke_observation(image_path,document,baseline,graph)
        except (OSError,ValueError,TypeError,KeyError,ArithmeticError):
            # Missing ink is unknown. The original source acceptance gate still
            # runs, and no failed image observation is substituted with CAD.
            pass
    remaining_evaluations=30000
    from .reconstruction_priority import mask_is_hard
    source_first=not mask_is_hard(baseline)
    if source_first and ink_observation is not None and initial_seed_entities is not None:
        from .source_stroke_prior import source_stroke_observation
        ink_observation=source_stroke_observation(image_path,document,baseline,graph,
            probe_entities=initial_seed_entities)
    receipt={"schema_version":"final-source-multistart-v2","status":"running",
             "maximum_wall_seconds":ceiling,"maximum_objective_evaluations":30000,
             "maximum_optional_restart_wall_seconds":15.,
             "maximum_optional_restart_objective_evaluations":3000,
             "attempts":[],"selected_seed":"canonical","ground_truth_used":False,
             "canonical_original_ink_prior_enabled":bool(source_first and ink_observation is not None),
             "acceptance_thresholds_changed":False}
    def persist():
        receipt["elapsed_seconds"]=max(0.,ceiling-(deadline-time.monotonic()))
        receipt["objective_evaluations"]=30000-remaining_evaluations
        _write(output/"source-multistart.json",receipt)
    def within_deadline(phase):
        if time.monotonic()<deadline:return True
        receipt["stop_reason"]="shared_wall_deadline_reached"
        receipt["deadline_reached_after_phase"]=phase
        persist();return False
    persist()
    def run(seed,offsets=None,previous_entities=None):
        nonlocal remaining_evaluations
        remaining_time=deadline-time.monotonic()
        evaluation_limit=remaining_evaluations
        if seed=='canonical' and source_first and ink_observation is not None and remaining_time>36.:
            # Reserve source/native audit and image-frame recovery time. A
            # canonical numerical solve consuming the whole ceiling previously
            # made every subsequent recovery branch unreachable.
            remaining_time-=24.
        if seed!="canonical":
            # Optional searches cannot consume the canonical solver's entire
            # remaining allowance.  Keep time for source/DXF audits too.
            recovering_source=(source_first and receipt.get('search_purpose')==
                'recover_original_source_support_without_changing_the_problem')
            remaining_time=(remaining_time-5. if recovering_source else min(15.,remaining_time-5.))
            if not recovering_source:evaluation_limit=min(3000,evaluation_limit)
        if remaining_time<=0 or remaining_evaluations<=0:return None
        directory=output/"source-multistart"/seed;directory.mkdir(parents=True,exist_ok=True)
        started=time.monotonic()
        try:
            current_ink=ink_observation
            if source_first and seed!='canonical' and previous_entities is not None:
                from .source_stroke_prior import source_stroke_observation
                current_ink=source_stroke_observation(image_path,document,baseline,graph,
                    probe_entities=previous_entities)
            result=_solve_with_source_observation(solver,graph,constraints,baseline,directory,
                budget_seconds=remaining_time,budget_evaluations=evaluation_limit,
                seed_node_offsets=offsets,image_stroke_prior=current_ink if source_first or seed!="canonical" else None,
                seed_entities=(initial_seed_entities if seed=="canonical" else previous_entities))
        except (ValueError,TypeError,ArithmeticError,OverflowError) as error:
            # A raised solver has no trustworthy evaluation receipt; spend
            # the remaining allowance instead of launching another restart.
            remaining_evaluations=0
            receipt["attempts"].append({"seed":seed,"elapsed_seconds":time.monotonic()-started,
                "solver_status":"failed","error_type":type(error).__name__,
                "solver_accepted":False,"source_gate_passed":False,"dxf_readback_passed":False})
            persist()
            return None
        consumed=int(((result.get("diagnostics") or {}).get("optimization_budget") or {}).get("objective_evaluations") or
                     ((result.get("diagnostics") or {}).get("source_budget_search") or {}).get("objective_evaluations") or 0)
        remaining_evaluations=max(0,remaining_evaluations-consumed)
        row={"seed":seed,"elapsed_seconds":time.monotonic()-started,"objective_evaluations":consumed,
             "solver_status":result.get("status"),"solver_accepted":result.get("accepted") is True,
             "original_ink_probe_geometry":(current_ink or {}).get('probe_geometry'),
             "source_gate_passed":False,"dxf_readback_passed":None if seed=="canonical" else False}
        receipt["attempts"].append(row)
        _write(directory/"parametric-solution.json",result)
        persist()
        return result
    incumbent=run("canonical")
    if incumbent is None:
        raise ValueError("Final source solve budget was exhausted before the canonical attempt")
    if observation is None or not any(k in observation for k in ('boundary_error_budget','optimization_budget')):
        receipt["status"]="not_applicable";_write(output/"source-multistart.json",receipt)
        return incumbent
    if not within_deadline("canonical_solve"):
        receipt["status"]="canonical_retained_budget_exhausted";persist()
        return incumbent
    continuation_used=False
    checkpoint=(incumbent.get('diagnostics') or {}).get('numerical_continuation_checkpoint') or {}
    if (source_first and incumbent.get('status')=='optimization_budget_exhausted'
            and checkpoint.get('available') is True and checkpoint.get('seed_only') is True
            and checkpoint.get('publication_approved') is False and deadline-time.monotonic()>7.):
        from .parametric_solver import numerical_problem_identity
        same_problem=checkpoint.get('problem_sha256')==numerical_problem_identity(graph,constraints)
        receipt['numerical_continuation_admission']={
            'same_original_problem':same_problem,'seed_only':True,
            'known_equations_may_be_unsatisfied':True,'publication_approved':False}
        if same_problem:
            # The solver independently checks topology, branch, shared nodes
            # and ORIGINAL variable bounds again before accepting this seed.
            continued=run('numerical-continuation',previous_entities=incumbent.get('candidate_entities'))
            continuation_used=True
            if continued is not None and (continued.get('accepted') is not True or
                    _constraint_receipts_complete({'constraints':constraints},continued,graph)):
                _write(output/'unaccepted-canonical-solution.json',incumbent)
                incumbent=continued
                receipt['selected_seed']='numerical-continuation'
            if not within_deadline('numerical_continuation'):
                receipt['status']='continuation_retained_budget_exhausted';persist();return incumbent
    recovery=incumbent.get("accepted") is not True
    incumbent_score=None
    if recovery:
        admission=_source_feasibility_restart_admission(graph,constraints,incumbent)
        receipt["recovery_admission"]=admission
        if not admission["eligible"]:
            receipt["status"]="not_applicable";persist();return incumbent
        seed_entities=incumbent["candidate_entities"]
        receipt["search_purpose"]="recover_source_feasibility_without_changing_the_problem"
    else:
        canonical_check=_solved_source_validation(image_path,document,baseline,source_comparator_graph,incumbent["entities"])
        receipt["attempts"][-1]["source_gate_passed"]=canonical_check["passed"] is True
        if not within_deadline("canonical_source_validation"):
            receipt["status"]="canonical_retained_no_refinement";persist();return incumbent
        if canonical_check.get("passed") is not True:
            # Numerical feasibility does not establish original-image support.
            # Keep the same bounded recovery used for a failed mask budget;
            # every binding and gate is unchanged, only the start may vary.
            recovery=True
            receipt["canonical_source_validation"]=canonical_check
            receipt["search_purpose"]="recover_original_source_support_without_changing_the_problem"
            if source_first:
                receipt['maximum_optional_restart_wall_seconds']=ceiling
                receipt['maximum_optional_restart_objective_evaluations']=30000
                receipt['recovery_budget_policy']='Use remaining shared budget for fixed-nominal ink recovery; no per-attempt 15-second cutoff.'
        elif incumbent.get("underconstrained") is not True:
            receipt["status"]="canonical_retained_no_refinement";persist();return incumbent
        else:
            incumbent_score=_source_solution_score(canonical_check)
            receipt["attempts"][0]["source_score"]=incumbent_score
            receipt["search_purpose"]="source_pareto_refinement"
        seed_entities=incumbent["entities"]
    if (source_first and incumbent.get('accepted') is True and recovery
            and deadline-time.monotonic()>10.):
        # Shape equations do not determine the image-frame translation gauge.
        # Re-localize that gauge from source ink before distorting free contacts
        # or abandoning correct nominal attributes. This is never registration
        # against the reference, and no coordinate-system origin is reset.
        from .source_frame_recovery import source_frame_candidate
        from .parametric_solver import MAX_DISPLACEMENT_FRACTION, _sample_entity
        try:
            shifted,frame=source_frame_candidate(image_path,document,baseline,source_comparator_graph,
                incumbent['entities'],max_seconds=min(8.,deadline-time.monotonic()-5.))
            receipt['source_frame_recovery']=frame
            if shifted is not None:
                extent=np.asarray([e['start'] for e in graph['entities']],float)
                limit=max(.25 if graph['units']=='mm' else 2.,
                    max(float(np.linalg.norm(np.ptp(extent,axis=0))),1e-3)*MAX_DISPLACEMENT_FRACTION)
                displacement=max(float(np.linalg.norm(_sample_entity(new)-_sample_entity(old),axis=1).max())
                    for new,old in zip(shifted,graph['entities']))
                frame.update(original_displacement_limit=limit,original_domain_passed=displacement<=limit)
                if displacement<=limit:
                    candidate=copy.deepcopy(incumbent);candidate['entities']=shifted
                    by_node={e[end+'_node']:e[end] for e in shifted for end in ('start','end')}
                    candidate['nodes']=[{**n,'x':by_node[n['id']][0],'y':by_node[n['id']][1]}
                        for n in candidate.get('nodes',[])]
                    checked=_solved_source_validation(image_path,document,baseline,source_comparator_graph,shifted)
                    frame['source_gate_passed']=checked.get('passed') is True
                    if checked.get('passed') is True:
                        exported=export_parametric(image_path,baseline,{**candidate,'constraints':copy.deepcopy(constraints)},
                            output/'source-frame-recovery')
                        native=all(exported['validation'].get(k,{}).get('passed') is True and
                            exported['validation'][k].get('dxf_readback_performed') is True for k in
                            ('exact_radius_validation','native_source_angle_validation','native_axis_relation_validation',
                             'native_measurement_validation','strict_relation_validation'))
                        frame.update(native_equations_verified=native,accepted=native)
                        if native and within_deadline('source_frame_native_readback'):
                            candidate['diagnostics']['source_frame_recovery']=copy.deepcopy(frame)
                            _write(output/'source-frame-recovery/solved-source-validation.json',checked)
                            receipt.update(status='source_frame_recovered',selected_seed='source-frame-translation')
                            persist();return candidate
        except InterruptedError:raise
        except (OSError,ValueError,TypeError,KeyError,IndexError,ArithmeticError) as error:
            receipt['source_frame_recovery']={'accepted':False,'failure_type':type(error).__name__}
        persist()
    step=(observation.get("boundary_error_budget") or observation["mask_representation_budget_diagnostic"])["sampling_step"]
    if not within_deadline("restart_admission"):
        receipt["status"]="canonical_retained_budget_exhausted";persist();return incumbent
    try:
        seed_options=({"recover_stroke_support":True} if
            receipt.get("search_purpose")=="recover_original_source_support_without_changing_the_problem" and
            "solved_source_stroke_support_degraded" in receipt.get("canonical_source_validation",{}).get("reasons",[]) else {})
        seeds,seed_evidence=_source_restart_offsets(image_path,document,baseline,graph,seed_entities,step,**seed_options)
    except (OSError,ValueError,TypeError,KeyError,IndexError,ArithmeticError):
        seeds,seed_evidence=[],{"status":"unavailable"}
    receipt["seed_evidence"]=seed_evidence
    selected=incumbent
    working_seed=seed_entities
    for index,offsets in enumerate(seeds[:1 if continuation_used else 2],start=1):
        if not within_deadline("source_seed_evidence") or deadline-time.monotonic()<7. or remaining_evaluations<100:
            receipt["stop_reason"]="shared_solve_budget_remaining_too_small";break
        seed=f"source-normal-{index}"
        candidate=run(seed,offsets,working_seed if incumbent.get("accepted") is True else None)
        if candidate is None:break
        row=receipt["attempts"][-1]
        if not within_deadline("optional_solve"):break
        if candidate.get("accepted") is not True:continue
        if (len(candidate.get("constraints") or [])!=len(constraints) or
                not _constraint_receipts_complete({"constraints":constraints},candidate,graph) or
                {check.get("id") for check in candidate.get("constraints") or []}!={check.get("id") for check in constraints} or
                not all(check.get("passed") is True for check in candidate.get("constraints") or [])):
            row["reason"]="constraint_receipts_incomplete";continue
        if not within_deadline("optional_constraint_validation"):break
        try:
            source_check=_solved_source_validation(image_path,document,baseline,source_comparator_graph,candidate["entities"])
        except InterruptedError:
            raise
        except (OSError,ValueError,TypeError,KeyError,IndexError,ArithmeticError):
            row["reason"]="optional_source_check_unavailable";continue
        row["source_gate_passed"]=source_check.get("passed") is True
        if not within_deadline("optional_source_validation"):break
        if not row["source_gate_passed"]:
            if source_first and receipt.get('search_purpose')=='recover_original_source_support_without_changing_the_problem':
                score=_source_solution_score(source_check)
                previous_score=(receipt.get('unpublished_numerical_candidate') or {}).get('source_score') or _source_solution_score(receipt['canonical_source_validation'])
                improves_ink=(score['stroke_supported_fraction']>=previous_score['stroke_supported_fraction']-1e-6 and
                    score['p90_edge_distance_px']<=previous_score['p90_edge_distance_px']+1e-6 and
                    (score['stroke_supported_fraction']>previous_score['stroke_supported_fraction']+1e-4 or
                     score['p90_edge_distance_px']<previous_score['p90_edge_distance_px']-1e-4))
                if improves_ink:
                    # A strictly feasible, still unpublished numerical iterate
                    # can seed the next unknown-attribute solve. The original
                    # graph/domain and the publication gate do not move.
                    try:
                        model=export_parametric(image_path,baseline,{**candidate,'constraints':copy.deepcopy(constraints)},
                            output/'source-multistart'/seed/'unpublished-export')
                        native=all(model['validation'].get(key,{}).get('passed') is True and
                            model['validation'].get(key,{}).get('dxf_readback_performed') is True for key in
                            ('exact_radius_validation','strict_relation_validation','native_measurement_validation','native_source_angle_validation'))
                    except (OSError,ValueError,KeyError,TypeError,ArithmeticError):native=False
                    if native and within_deadline('unpublished_native_readback'):
                        working_seed=candidate['entities']
                        row.update(retained_as_unpublished_next_numerical_seed=True,source_score=score,
                                   dxf_readback_passed=True)
                        _write(output/'source-multistart'/seed/'solved-source-validation.json',source_check)
                        receipt['unpublished_numerical_candidate']={
                            'seed':seed,'solution_path':str(output/'source-multistart'/seed/'parametric-solution.json'),
                            'source_score':score,'source_gate_passed':False,'native_attributes_verified':True,
                            'publication_approved':False,'acceptance_thresholds_changed':False}
            row["reason"]="unchanged_source_gate_failed";continue
        try:
            model=export_parametric(image_path,baseline,
                {**candidate,"constraints":copy.deepcopy(constraints)},
                output/"source-multistart"/seed/"export")
            validation=model["validation"]
            row["dxf_readback_passed"]=all(validation.get(key,{}).get("passed") is True and
                validation.get(key,{}).get("dxf_readback_performed") is True for key in
                ("exact_radius_validation","strict_relation_validation","native_measurement_validation","native_source_angle_validation"))
        except (OSError,ValueError,KeyError,TypeError,ArithmeticError):
            row["reason"]="candidate_export_or_readback_failed";continue
        if not row["dxf_readback_passed"]:
            row["reason"]="candidate_native_constraint_readback_failed";continue
        if not within_deadline("optional_native_readback"):break
        score=_source_solution_score(source_check);row["source_score"]=score
        if incumbent_score is None or _source_score_improves(score,incumbent_score):
            if not within_deadline("optional_promotion"):break
            first_feasible=incumbent_score is None
            incumbent_score=score;selected=candidate;receipt["selected_seed"]=seed
            row["reason"]="source_feasibility_recovered_all_gates_passed" if first_feasible else "source_evidence_pareto_improved"
            if first_feasible and source_first:
                persist()
                break
        else:row["reason"]="source_evidence_not_improved"
        persist()
    receipt["status"]=("selected_source_feasibility_recovery" if recovery else "selected_source_refinement") if selected is not incumbent else "canonical_retained"
    persist()
    selected.setdefault("diagnostics",{})["final_source_multistart"]={
        "status":receipt["status"],"selected_seed":receipt["selected_seed"],
        "receipt_path":str(output/"source-multistart.json"),"ground_truth_used":False}
    return selected


def _retry_source_radius_from_checkpoint(image_path, document, baseline, graph, bindings,
                                         solution, inventory, output, *, provider=None, use_api=False):
    """An optional retry cannot discard the previously certified subset."""
    try:
        return _retry_source_radius_checkpoint_impl(image_path,document,baseline,graph,bindings,
            solution,inventory,output,provider=provider,use_api=use_api)
    except InterruptedError:
        raise
    except Exception as error:
        receipt={'schema_version':'post-solve-radius-retry-v1','accepted':False,
            'reason':'post_solve_radius_retry_failed','failure_type':type(error).__name__,
            'ground_truth_used':False,'publication_approved':False,
            'optimization_domain_recentered':False,'known_subset_preserved':True}
        _write(Path(output)/'radius-checkpoint-refinement.json',receipt)
        return graph,bindings,solution,receipt


def _retry_source_radius_checkpoint_impl(image_path, document, baseline, graph, bindings,
                                         solution, inventory, output, *, provider=None, use_api=False):
    """Retry one source-owned R from a certified initial value, in the same domain.

    An unsuccessful initial search is not an ownership rejection. Preserve its
    receipt, then try once from the jointly satisfied subset. Fresh online
    binding, all old constraints and original source checks still decide use.
    """
    from .source_radius_hypotheses import (radius_hypotheses, refine_source_owned_radius_hypotheses,
                                         certify_refined_radius_seed)
    from .parametric_solver import solve_parametric
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    receipt={'schema_version':'post-solve-radius-retry-v1','accepted':False,
        'maximum_hypothesis_attempts':1,'maximum_hypothesis_wall_seconds':90.,
        'maximum_final_solve_wall_seconds':120.,'optimization_domain_recentered':False,
        'ground_truth_used':False,'publication_approved':False}
    def finish(reason, result=None):
        receipt['reason']=reason
        _write(output/'radius-checkpoint-refinement.json',receipt)
        return (*(result or (graph,bindings,solution)),receipt)
    if (solution.get('accepted') is not True or
            not _constraint_receipts_complete(bindings,solution,graph)):
        return finish('previous_subset_not_jointly_certified')
    if not radius_hypotheses(inventory,graph,bindings.get('constraints') or []):
        return finish('no_source_owned_pending_radius')
    before=_solved_source_validation(image_path,document,baseline,graph,solution['entities'])
    receipt['checkpoint_source_validation']=before
    if before.get('passed') is not True:
        return finish('previous_subset_original_source_check_failed')
    candidate,hypothesis=refine_source_owned_radius_hypotheses(image_path,document,baseline,graph,
        bindings,inventory,output/'source-radius-hypotheses',max_attempts=1,
        seed_entities=solution['entities'])
    receipt['hypothesis']=hypothesis
    if hypothesis.get('status')!='source_verified_seed':
        return finish('checkpoint_radius_hypothesis_not_admitted')
    candidate,new_bindings,certificate=certify_refined_radius_seed(image_path,document,baseline,
        graph,bindings,inventory,candidate,hypothesis,output/'source-radius-certification',
        provider=provider,use_api=use_api)
    receipt['formal_certification']=certificate
    if certificate.get('status')!='accepted_seed':
        return finish('checkpoint_radius_fresh_certification_failed')
    from .reconstruction_priority import mask_is_hard
    ink=None
    if not mask_is_hard(baseline):
        from .source_stroke_prior import source_stroke_observation
        ink=source_stroke_observation(image_path,document,baseline,graph,
            probe_entities=candidate['entities'])
    receipt['final_original_image_observation_supplied_to_solver']=ink is not None
    new_solution=_solve_with_source_observation(solve_parametric,graph,new_bindings['constraints'],
        baseline,output/'final-solve',budget_seconds=120.,seed_entities=candidate['entities'],
        image_stroke_prior=ink)
    receipt['final_solver_status']=new_solution.get('status')
    if (new_solution.get('accepted') is not True or
            not _constraint_receipts_complete(new_bindings,new_solution,graph)):
        return finish('checkpoint_radius_final_joint_solve_failed')
    after=_solved_source_validation(image_path,document,baseline,graph,new_solution['entities'])
    receipt['final_original_source_validation']=after
    if after.get('passed') is not True:
        return finish('checkpoint_radius_final_original_source_check_failed')
    receipt['accepted']=True
    return finish('source_radius_refined_from_certified_checkpoint',(candidate,new_bindings,new_solution))


def _radius_segment_hypotheses(bindings, inherited=()):
    """Carry the latest source-arrow coordinates, never a verdict or target ID."""
    rows = ((bindings.get("radius_binding_coverage") or {}).get("required_mappings") or [])
    result, seen, counts, fresh_records = [], set(), {}, set()

    def add(record_id, evidence):
        if not isinstance(record_id, str) or not isinstance(evidence, dict):
            return False
        try:
            segment = np.asarray(evidence.get("segment_px"), float)
        except (TypeError, ValueError):
            return False
        if segment.shape != (2, 2) or not np.isfinite(segment).all():
            return False
        key = (record_id, tuple(np.round(segment.ravel(), 4)))
        if key in seen:
            return True
        if counts.get(record_id, 0) >= 4:
            return False
        seen.add(key)
        counts[record_id] = counts.get(record_id, 0) + 1
        result.append({"record_id": record_id, "kind": "radius",
                       "source_evidence": {"segment_px": segment.tolist()}})
        return True

    for mapping in rows:
        if not isinstance(mapping, dict):
            continue
        record_id = mapping.get("record_id")
        evidence_rows = mapping.get("source_evidence")
        for evidence in (evidence_rows[:4] if isinstance(evidence_rows, list) else ()):
            if add(record_id, evidence):
                fresh_records.add(record_id)
    # A failed observation on one edited graph does not make a previously
    # observed pixel location untrue. Keep it as a coordinate-only hypothesis
    # for the next graph, where the full source verifier must run again.
    if isinstance(inherited, (list, tuple)):
        for row in inherited[:1000]:
            if not isinstance(row, dict) or row.get("kind") != "radius":
                continue
            record_id = row.get("record_id")
            if record_id not in fresh_records:
                add(record_id, row.get("source_evidence"))
    return result


def _edit_observation_parent(candidate):
    """Offer fresh source coordinates to edits without changing a certified graph."""
    observations = candidate.get("edit_source_observations")
    if not isinstance(observations, dict):
        return candidate
    parent = copy.deepcopy(candidate)
    solved=observations.get("certified_solved_graph")
    if solved is not None:
        identity=lambda e:tuple(e.get(k) for k in ("id","type","start_node","end_node","clockwise"))
        if (observations.get("solver_and_source_certified") is not True or
                observations.get("certified_solved_graph_sha256")!=_canonical_sha256(solved) or
                solved.get("source_sha256")!=parent["graph"].get("source_sha256") or
                [identity(e) for e in solved["entities"]]!=[identity(e) for e in parent["graph"]["entities"]]):
            raise ValueError("edit_parent_certified_state_changed")
        parent["graph"]=copy.deepcopy(solved)
    for field in ("angle_source_observations", "radius_source_segment_hypotheses"):
        if field in observations:
            parent["graph"][field] = copy.deepcopy(observations[field])
    return parent


def _unmet_edit_observation_parent(candidate,feedback):
    """Remove solved angle obligations before the local proposal ranking."""
    parent=_edit_observation_parent(candidate)
    satisfied=set((feedback or {}).get("satisfied_record_ids") or [])
    observations=(parent.get("graph") or {}).get("angle_source_observations") or []
    if not any(row.get("record_id") in satisfied for row in observations):return parent
    parent=copy.deepcopy(parent)
    parent["graph"]["angle_source_observations"]=[copy.deepcopy(row) for row in observations
                                                   if row.get("record_id") not in satisfied]
    return parent


def _source_joint_bootstrap(image_path, document, baseline, selected, bundle, output,
                            feedback, *, remaining_seconds, edit_time_reserve):
    """Build one source-only radius+angle compound, before numerical preflight.

    The intermediate refit is never a beam member. Both operations use the
    ordinary edit kernel and source gate; only a freshly witnessed two-radius
    joint may become a candidate for the shared numerical search budget.
    """
    from .topology_editing import execute_topology_edits
    from .topology_search import (near_nominal_joint_bootstrap_refits,
                                  refresh_source_angle_observations,
                                  unmet_source_angle_joint_operations)

    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    audit={"status":"skipped","reason":None,"parent_candidate_id":selected["id"],
           "operations":[],"ground_truth_used":False}
    def save():
        _write(output/"source-joint-bootstrap-audit.json",audit)
    if remaining_seconds() <= edit_time_reserve+30.:
        audit["reason"]="source_joint_bootstrap_edit_time_reserved";save();return None,audit,None
    edit_parent=_edit_observation_parent(selected)
    refits=near_nominal_joint_bootstrap_refits(
        edit_parent.get("graph") or {},bundle.get("annotation_inventory",[]),feedback,limit=1)
    if not refits:
        audit["reason"]="no_unresolved_arrow_verified_near_nominal_joint_refit";save();return None,audit,None
    first=refits[0];audit["operations"].append(first);save()
    try:
        intermediate_rows,first_execution=execute_topology_edits(
            image_path,document,baseline,edit_parent,bundle,[first],output/"radius-refit")
    except InterruptedError:
        raise
    except (OSError,ValueError,KeyError,TypeError,ArithmeticError):
        audit["reason"]="source_joint_bootstrap_radius_execution_failed";save();return None,audit,None
    audit["radius_execution"]=first_execution
    if not intermediate_rows:
        audit["reason"]="source_joint_bootstrap_radius_candidate_rejected";save();return None,audit,None
    intermediate=intermediate_rows[0]
    intermediate_source=_topology_source_validation(image_path,baseline,intermediate["graph"])
    audit["intermediate"]={"candidate_id":intermediate["id"],
                           "geometry_sha256":geometry_fingerprint(intermediate["graph"]),
                           "source_validation_passed":intermediate_source.get("passed") is True,
                           "source_validation_reasons":intermediate_source.get("reasons",[])}
    _write(output/"intermediate-topology.json",intermediate["graph"])
    _write(output/"intermediate-source-validation.json",intermediate_source)
    save()
    if intermediate_source.get("passed") is not True:
        audit["reason"]="source_joint_bootstrap_intermediate_source_failed";save();return None,audit,None
    if remaining_seconds() <= edit_time_reserve+15.:
        audit["reason"]="source_joint_bootstrap_edit_time_reserved";save();return None,audit,None
    try:
        observations=refresh_source_angle_observations(
            image_path,document,baseline,intermediate["graph"])
    except InterruptedError:
        raise
    except (OSError,ValueError,KeyError,TypeError,ArithmeticError):
        audit["reason"]="source_joint_bootstrap_angle_observation_failed";save();return None,audit,None
    audit["fresh_angle_observation_count"]=len(observations)
    joint_ops=unmet_source_angle_joint_operations(
        intermediate,feedback,bundle.get("annotation_inventory",[]),limit=1)
    if not joint_ops:
        audit["reason"]="source_joint_bootstrap_missing_verified_two_radius_joint";save();return None,audit,None
    joint=joint_ops[0];audit["operations"].append(joint)
    audit["joint_witness"]={"angle_record_id":joint["record_id"],
                            "entity_ids":joint["entity_ids"],"source_verified":True}
    _write(output/"intermediate-topology.json",intermediate["graph"])
    save()
    if remaining_seconds() <= edit_time_reserve+15.:
        audit["reason"]="source_joint_bootstrap_edit_time_reserved";save();return None,audit,None
    try:
        compound_rows,joint_execution=execute_topology_edits(
            image_path,document,baseline,intermediate,bundle,[joint],output/"joint-line")
    except InterruptedError:
        raise
    except (OSError,ValueError,KeyError,TypeError,ArithmeticError):
        audit["reason"]="source_joint_bootstrap_joint_execution_failed";save();return None,audit,None
    audit["joint_execution"]=joint_execution
    if not compound_rows:
        audit["reason"]="source_joint_bootstrap_joint_candidate_rejected";save();return None,audit,None
    compound=compound_rows[0]
    edit=(((compound.get("graph") or {}).get("source_evidence") or {}).get("topology_edit") or {})
    if (edit.get("replacement_types")!=["ARC","LINE","ARC"] or
            len(set(edit.get("bound_record_ids") or []))!=2 or
            edit.get("angle_support_record_ids")!=[joint["record_id"]]):
        audit["reason"]="source_joint_bootstrap_compound_evidence_incomplete";save();return None,audit,None
    compound_source=_topology_source_validation(image_path,baseline,compound["graph"])
    audit["compound"]={"candidate_id":compound["id"],
                       "geometry_sha256":geometry_fingerprint(compound["graph"]),
                       "source_validation_passed":compound_source.get("passed") is True,
                       "source_validation_reasons":compound_source.get("reasons",[]),
                       "bound_radius_record_ids":edit["bound_record_ids"]}
    _write(output/"compound-topology.json",compound["graph"])
    _write(output/"compound-source-validation.json",compound_source)
    save()
    if compound_source.get("passed") is not True:
        audit["reason"]="source_joint_bootstrap_compound_source_failed";save();return None,audit,None
    if remaining_seconds() <= edit_time_reserve:
        audit["reason"]="source_joint_bootstrap_edit_time_reserved";save();return None,audit,None
    compound["id"]=f"seed-joint-{selected['id']}"
    compound["graph"]["candidate_id"]=compound["id"]
    if compound["graph"].get("entity_identity"):
        compound["graph"]["entity_identity"]["display_id_scope"]=compound["id"]
    compound["graph"].setdefault("source_evidence",{})["initial_seed_bootstrap"]={
        "parent_candidate_id":selected["id"],"operations":copy.deepcopy(audit["operations"]),
        "intermediate_geometry_sha256":audit["intermediate"]["geometry_sha256"],
        "ground_truth_used":False}
    audit["compound"]["candidate_id"]=compound["id"]
    _write(output/"compound-topology.json",compound["graph"])
    audit["status"]="source_candidate";save()
    return compound,audit,compound_source


def _pending_source_topology_operations(candidate, feedback, inventory, visited, *, limit=4):
    """Find fresh, source-witnessed annotation edits; proposals are never certificates."""
    from .topology_editing import propose_annotation_arc_edits
    from .topology_search import operation_fingerprint

    graph=_unmet_edit_observation_parent(candidate,feedback)["graph"]
    coverage=feedback.get("radius_binding_coverage") or {}
    radius_records={row.get("record_id") for group in ("required_mappings","unresolved")
                    for row in coverage.get(group) or [] if isinstance(row,dict)
                    and row.get("source_arrow_verified") is True}
    satisfied=set(feedback.get("satisfied_record_ids") or [])
    angle_records={row.get("record_id") for row in graph.get("angle_source_observations") or []
                   if isinstance(row,dict) and row.get("verified") is True
                   and row.get("record_id") not in satisfied}
    unresolved={row.get("record_id") for row in coverage.get("unresolved") or []
                if isinstance(row,dict) and row.get("source_arrow_verified") is True}
    by_id={row.get("id"):row for row in graph.get("entities") or [] if isinstance(row,dict)}
    parent_hash=geometry_fingerprint(candidate["graph"])
    proposed=propose_annotation_arc_edits(graph,inventory,limit=limit,
                                           unresolved_radius_record_ids=unresolved)
    pending=[]
    for operation in proposed:
        if not isinstance(operation,dict):continue
        action=operation.get("action");record_id=operation.get("record_id")
        ids=operation.get("entity_ids") or []
        tags=set(operation.get("evidence_tags") or [])
        if (action not in {"insert_annotated_fillet","refit_chain_as_annotated_arc",
                           "restore_annotated_line_support","split_chain_at_source_features"}
                or not isinstance(ids,list) or not 1<=len(ids)<=8
                or any(entity_id not in by_id for entity_id in ids)
                or not {"annotation_target","source_boundary"}<=tags):
            continue
        if record_id is not None:
            if record_id not in (angle_records if action=="restore_annotated_line_support" else radius_records):
                continue
        elif action=="split_chain_at_source_features":
            witnessed={row.get("record_id") for row in graph.get("annotation_support") or []
                       if isinstance(row,dict) and row.get("kind")=="radius"
                       and row.get("candidate_entity_id") in ids
                       and row.get("arrowhead_verified") is True}
            if len(witnessed)<2:continue
        else:continue
        if operation_fingerprint(parent_hash,operation) in visited:continue
        pending.append({"action":action,"entity_ids":ids,"record_id":record_id,
                        "source_witness":"verified_annotation_target_and_source_boundary"})
    return pending


def _maybe_extend_topology_time_budget(budget, record, previous, selected, inventory,
                                       *, round_number, max_rounds, use_api):
    """Grant one extra bounded clock only for certified progress with pending source work."""
    audit=record["time_budget_extension"]
    if budget.extension_seconds:return False
    audit["evaluated_after_round"]=round_number
    if round_number>=max_rounds:
        audit["reason"]="no_remaining_round";return False
    if budget.preflights>=budget.max_preflights or (use_api and budget.provider_calls>=budget.max_provider_calls):
        audit["reason"]="count_budget_exhausted";return False
    if budget.clock()-budget.started>=budget.max_seconds+min(600.,budget.max_seconds):
        audit["reason"]="maximum_extended_time_already_elapsed";return False
    if record.get("stop_reason") not in (None,"topology_search_time_budget_exhausted"):
        audit["reason"]="non_time_search_stop";return False
    before=previous.get("constraint_feedback") or {}
    after=selected.get("constraint_feedback") or {}
    try:
        progressed=bool(after and _source_verified_primary_progress(previous,selected,before))
    except Exception:
        audit["reason"]="progress_evidence_unavailable";return False
    if not progressed:
        audit["reason"]="no_certified_source_numeric_regression_progress";return False
    try:
        pending=_pending_source_topology_operations(selected,after,inventory,budget.visited_operations)
    except Exception:
        audit["reason"]="pending_source_topology_evidence_unavailable";return False
    if not pending:
        audit["reason"]="no_fresh_source_witnessed_topology_obligation";return False
    if budget.clock()-budget.started>=budget.max_seconds+min(600.,budget.max_seconds):
        audit["reason"]="maximum_extended_time_already_elapsed";return False
    reason="certified_progress_with_fresh_source_witnessed_topology_obligation"
    if not budget.extend_once(reason=reason):return False
    audit.update(status="granted",reason=reason,extension_seconds=budget.extension_seconds,
                 total_max_seconds=budget.max_seconds+budget.extension_seconds,
                 trigger_round=round_number,trigger_candidate_id=selected["id"],
                 trigger_geometry_sha256=geometry_fingerprint(selected["graph"]),
                 gained_satisfied_record_ids=sorted(set(after.get("satisfied_record_ids") or [])-
                                                  set(before.get("satisfied_record_ids") or [])),
                 pending_source_operations=pending,ground_truth_used=False)
    if record.get("stop_reason")=="topology_search_time_budget_exhausted":
        record["stop_reason"]=None
        audit["cleared_prior_time_stop_for_next_round"]=True
    return True


def _topology_edit_loop(image_path, document, baseline, selected, bundle, output, *,
                        editor_provider=None, evaluator_provider=None, use_api=False,
                        progress=None, max_rounds=3, beam_width=3, max_preflights=18,
                        max_provider_calls=12, max_seconds=600., initial_seed_ids=()):
    """Bounded beam of source-certified edits; no GT-driven candidate selection."""
    from .constraint_binding import analyze_constraint_bindings
    from .parametric_solver import solve_parametric
    from .topology_search import (SearchBudget, preflight_admissible, retain_distinct,
                                  source_repair_branch_priority)
    if isinstance(max_rounds,bool) or not 1<=max_rounds<=3:
        raise ValueError("topology_iteration_budget_must_be_1_to_3")
    if type(beam_width) is not int or not 1<=beam_width<=3:
        raise ValueError("topology_beam_width_must_be_1_to_3")
    if not isinstance(initial_seed_ids,(list,tuple)) or len(initial_seed_ids)>5 or any(
            not isinstance(name,str) for name in initial_seed_ids):
        raise ValueError("topology_initial_seeds_must_be_at_most_five_candidate_ids")
    output=Path(output);root=output/"topology-iterations";root.mkdir(parents=True,exist_ok=True)
    emit=progress or (lambda stage,message:None)
    budget=SearchBudget(max_preflights=max_preflights,max_provider_calls=max_provider_calls,max_seconds=max_seconds)
    record={"schema_version":"bounded-topology-iterations-v2","status":"running","max_rounds":max_rounds,
            "beam_width":beam_width,"ranking":"verified_radius_records, satisfied_independent_records, satisfied_structural_constraints, remaining_shape_dof, source_error",
            "rounds":[],"stop_reason":None,"ground_truth_used":False,
            "scope":"Source-only bounded beam; numerical construction is not fresh binding, and model verdicts do not certify reference accuracy."}
    record["time_budget_extension"]={"status":"not_granted","reason":None,
        "original_max_seconds":budget.max_seconds,"extension_seconds":0.,
        "maximum_extension_seconds":min(600.,budget.max_seconds),"total_max_seconds":budget.max_seconds,
        "ground_truth_used":False}
    record["initial_seed_audit"]={"selected_candidate_id":selected["id"],"attempts":[],
                                  "admitted_candidate_ids":[],"ground_truth_used":False}
    cache={};edit_observation_cache={};geometries={};seen_selected={geometry_fingerprint(selected["graph"])}
    def persist():
        record["budget"]=budget.summary()
        record["visited_operations"]=list(budget.visited_operations.values())
        record["visited_geometries"]=list(geometries.values())
        _write(output/"topology-iterations.json",record)
    def verify(candidate, *, source_validation=None, parent_candidate=None):
        graph=candidate["graph"];geometry_key=geometry_fingerprint(graph)
        # Every source hint and construction claim can affect binding or solving.
        # Cache only the exact graph that the preflight actually examined.
        key=_canonical_sha256({"candidate_id":candidate["id"],"graph":graph})
        if key in cache:
            observations=edit_observation_cache.get(key)
            if observations is None:
                candidate.pop("edit_source_observations",None)
            else:
                candidate["edit_source_observations"]=copy.deepcopy(observations)
            result=copy.deepcopy(cache[key]);result["candidate_id"]=candidate["id"]
            return result
        if not budget.reserve_preflight():
            reason=budget.preflight_block_reason() or "topology_search_preflight_budget_exhausted"
            result=reconstruction_feedback(graph)
            result.update(solver_status="preflight_not_run",preflight_status="not_run_budget",
                          preflight_skip_reason=reason,constraint_count=0,binding_status="not_run",
                          source_validation={"status":"not_run","passed":None,"reasons":[]})
            if source_validation is not None:
                result["topology_source_validation"]=source_validation
            return result
        persist()
        directory=root/"constraint-preflight"/key[:16];directory.mkdir(parents=True,exist_ok=True)
        preflight_graph=copy.deepcopy(graph)
        bindings=None;solution=None
        input_graph_sha256=_canonical_sha256(preflight_graph)
        candidate.pop("edit_source_observations",None)
        source_validation=(source_validation if source_validation is not None else
                           _topology_source_validation(image_path,baseline,preflight_graph))
        repair_domain_graph=None;repair_parent_graph=None;repair_parent_constraints=[]
        if parent_candidate is not None:
            from .source_topology_initial_guess import (line_seed_admission,
                repair_source_owned_line_seed)
            from .reconstruction_priority import mask_is_hard
            admission=line_seed_admission(preflight_graph,source_validation)
            if admission["eligible"] or not mask_is_hard(baseline):
                parent_artifact=(parent_candidate.get("constraint_feedback") or {}).get("preflight_artifact")
                try:
                    parent_bindings=json.loads((Path(parent_artifact)/"constraint-bindings.json").read_text(encoding="utf8"))
                    repair_parent_constraints=parent_bindings.get("constraints") or []
                except (TypeError,ValueError,OSError):
                    repair_parent_constraints=[]
                repair_parent_graph=copy.deepcopy(_edit_observation_parent(parent_candidate)["graph"])
            if admission["eligible"]:
                original=copy.deepcopy(preflight_graph)
                repaired,repair_receipt=repair_source_owned_line_seed(
                    image_path,document,baseline,candidate,directory/"initial-guess-repair",
                    parent_graph=repair_parent_graph,parent_constraints=repair_parent_constraints,
                    source_validation=source_validation,budget_seconds=budget.remaining_seconds())
                _write(directory/"initial-source-validation.json",source_validation)
                if repaired is not None:
                    candidate.update(repaired);graph=candidate["graph"]
                    preflight_graph=copy.deepcopy(graph);repair_domain_graph=original
                    geometry_key=geometry_fingerprint(graph)
                    input_graph_sha256=_canonical_sha256(preflight_graph)
                    source_validation=_topology_source_validation(image_path,baseline,preflight_graph)
        _write(directory/"source-validation.json",source_validation)
        if not source_validation["passed"]:
            result=reconstruction_feedback(preflight_graph)
            result.update(solver_status="source_validation_failed",constraint_count=0,
                          binding_status="not_run",source_validation=source_validation)
            try:
                from .source_support_diagnostics import source_support_diagnostics
                diagnostic=source_support_diagnostics(image_path,document,baseline,preflight_graph)
                _write(directory/"source-support-diagnostics.json",diagnostic)
                result["local_source_diagnostics"]=diagnostic
            except (OSError,KeyError,TypeError,ValueError):
                result["local_source_diagnostics"]={"status":"unavailable"}
        else:
            bindings=None
            binding_completed=False
            solver_started=False
            solver_completed=False
            preflight_phase="binding"
            try:
                bindings=analyze_constraint_bindings(image_path,document,baseline,preflight_graph,directory,use_api=False)
                preflight_phase="binding_validation"
                if (not isinstance(bindings,dict) or not isinstance(bindings.get("constraints"),list) or
                        any(not isinstance(row,dict) for row in bindings["constraints"])):
                    raise ValueError("invalid_preflight_binding_receipt")
                binding_completed=True
                if any((preflight_graph.get("source_evidence") or {}).get(key) for key in
                       ("source_line_initial_guess", "source_state_resume")):
                    from .source_topology_initial_guess import retain_line_seed_bindings
                    inventory=json.loads((directory/"binding-candidates.json").read_text(encoding="utf8"))
                    bindings,inventory,_=retain_line_seed_bindings(image_path,document,baseline,
                        preflight_graph,bindings,inventory,directory)
                if repair_domain_graph is not None or (repair_parent_graph is not None and
                        not mask_is_hard(baseline) and repair_parent_constraints):
                    from .source_topology_initial_guess import required_constraints_preserved
                    retained=required_constraints_preserved(repair_parent_graph,preflight_graph,
                        repair_parent_constraints,bindings["constraints"])
                    _write(directory/"fresh-known-source-obligations.json",retained)
                    if not retained["passed"]:
                        raise ValueError("fresh_binding_lost_previous_source_obligation")
                preflight_phase="binding_evidence"
                # These newly observed coordinates are proposals for a child
                # edit, not part of the graph whose binding was just certified.
                candidate["edit_source_observations"]={
                    "schema_version":"edit-source-observations-v1",
                    "scope":"Source-coordinate proposals for independently preflighted child edits only.",
                    "binding_verified":False,"ground_truth_used":False,
                    "angle_source_observations":copy.deepcopy(bindings.get("angle_source_observations",[])),
                    "radius_source_segment_hypotheses":_radius_segment_hypotheses(
                        bindings,graph.get("radius_source_segment_hypotheses") or ())}
                preflight_phase="budget_before_solver"
                if budget.remaining_seconds()<=0:
                    raise TimeoutError("topology_search_time_budget_exhausted")
                preflight_phase="solver"
                solver_started=True
                from .reconstruction_priority import mask_is_hard
                ink=None
                if not mask_is_hard(baseline):
                    from .source_stroke_prior import source_stroke_observation
                    ink=source_stroke_observation(image_path,document,baseline,
                        repair_domain_graph if repair_domain_graph is not None else preflight_graph,
                        probe_entities=preflight_graph['entities'] if repair_domain_graph is not None else None)
                solution=_solve_with_source_observation(solve_parametric,
                    repair_domain_graph if repair_domain_graph is not None else preflight_graph,
                    bindings.get("constraints",[]),baseline,directory,
                    budget_seconds=budget.remaining_seconds(),
                    seed_entities=preflight_graph["entities"] if repair_domain_graph is not None else None,
                    image_stroke_prior=ink)
                solver_completed=True
                preflight_phase="post_solve_validation"
                if _canonical_sha256(preflight_graph)!=input_graph_sha256:
                    raise ValueError("preflight_graph_mutated_during_binding_or_solve")
                result=reconstruction_feedback(preflight_graph,bindings,solution)
                _record_fillet_tangent_contract(directory,preflight_graph,bindings,solution,result)
                diagnostic_entities=solution.get("entities") if solution.get("accepted") else solution.get("candidate_entities")
                if diagnostic_entities:
                    validation=_solved_source_validation(image_path,document,baseline,preflight_graph,diagnostic_entities)
                    if repair_parent_graph is not None:
                        comparator=copy.deepcopy(repair_parent_graph)
                        for entity in comparator["entities"]:entity.pop("radius_binding",None)
                        parent_validation=_solved_source_validation(image_path,document,baseline,comparator,diagnostic_entities)
                        _write(directory/"parent-solved-source-validation.json",parent_validation)
                        validation["original_parent_source_validation"]=parent_validation
                        if not parent_validation["passed"]:
                            validation["passed"]=False
                            validation["reasons"]=list(dict.fromkeys([*validation.get("reasons",[]),
                                *parent_validation.get("reasons",[])]))
                    _write(directory/"solved-source-validation.json",validation)
                    mask_diagnostics=None
                    if not validation["passed"]:
                        try:
                            mask_diagnostics=source_mask_interval_diagnostics(baseline,preflight_graph,diagnostic_entities)
                            mask_diagnostics["geometry_stage"]="solver_candidate" if solution.get("accepted") else "rejected_solver_candidate"
                            _write(directory/"source-mask-interval-diagnostics.json",mask_diagnostics)
                        except (KeyError,TypeError,ValueError,ArithmeticError):
                            mask_diagnostics={"status":"unavailable"}
                        from .source_support_diagnostics import source_support_diagnostics
                        solved_graph={**preflight_graph,"entities":diagnostic_entities}
                        diagnostic=source_support_diagnostics(image_path,document,baseline,solved_graph)
                        diagnostic["geometry_stage"]="solver_candidate" if solution.get("accepted") else "rejected_solver_candidate"
                        _write(directory/"source-support-diagnostics.json",diagnostic)
                        result["local_source_diagnostics"]=diagnostic
                    source_failure_feedback(result,preflight_graph,validation,mask_diagnostics)
            except InterruptedError:
                raise
            except Exception as error:
                # An exhausted search budget is not geometric infeasibility.
                # Keep completed binding evidence even when no solver started,
                # but never turn a partial preflight into a certified result.
                if isinstance(error,TimeoutError):
                    category="budget_exhausted"
                    reason=("topology_search_time_budget_exhausted" if preflight_phase=="budget_before_solver"
                            else "source_preflight_operation_timed_out")
                elif preflight_phase=="binding_validation":
                    category="invalid_binding";reason="source_preflight_invalid_binding"
                elif preflight_phase in {"binding","binding_evidence"}:
                    category="binding_failed";reason="source_constraint_binding_failed"
                elif preflight_phase=="solver":
                    category="solver_failed";reason="source_constraint_solver_failed"
                else:
                    category="post_solve_failed";reason="source_preflight_post_solve_failed"
                # Only built-in type labels are public. Exception messages and
                # arbitrary provider class names can contain local secrets.
                error_type=next((kind.__name__ for kind in (TimeoutError,ValueError,TypeError,KeyError,
                    ArithmeticError,OSError,RuntimeError) if isinstance(error,kind)),"Exception")
                result=reconstruction_feedback(graph,bindings if binding_completed else None)
                result.update(solver_status="preflight_not_run" if category=="budget_exhausted" and not solver_started
                              else "preflight_failed",binding_status="completed" if binding_completed else "failed",
                              solver_accepted=False,post_solve_source_accepted=False,
                              preflight_failure={"category":category,"stage":preflight_phase,"error_type":error_type,
                                  "binding_completed":binding_completed,"solver_started":solver_started,
                                  "solver_completed":solver_completed,"infeasibility_proven":False,
                                  "exception_message_recorded":False},
                              source_validation={"status":"not_run","passed":None,"reasons":[reason]})
                if category=="budget_exhausted" and not solver_started:
                    result.update(preflight_status="not_run_budget",preflight_skip_reason=reason)
                _write(directory/"preflight-failure.json",result["preflight_failure"])
                candidate.pop("edit_source_observations",None)
        # A downstream source check must not silently change the input after
        # binding/solving. Only the untouched candidate graph may receive a
        # trusted checkpoint identity, and a changed working copy fails shut.
        if (_canonical_sha256(preflight_graph)!=input_graph_sha256 or
                _canonical_sha256(graph)!=input_graph_sha256):
            result=reconstruction_feedback(graph)
            result.update(solver_status="preflight_failed",constraint_count=0,binding_status="not_run",
                          source_validation={"passed":False,
                                             "reasons":["preflight_input_mutated_during_preflight"]})
            candidate.pop("edit_source_observations",None)
        result["topology_source_validation"]=source_validation
        result["preflight_artifact"]=str(directory)
        result["source_binding_locks"]=_source_binding_locks(graph,bindings or {},solution or {},result)
        if result.get("solver_accepted") is True and result.get("post_solve_source_accepted") is True:
            _write(directory/"topology.json",graph)
            _write(directory/"preflight-input-identity.json",_preflight_input_identity(image_path,document,baseline,graph))
            if _constraint_receipts_complete(bindings,solution,graph):
                from .source_radius_hypotheses import graph_from_solution
                observed=graph_from_solution(graph,baseline,solution)
                observed.setdefault("source_evidence",{})["proposal_stroke_support"]=copy.deepcopy(validation["after"])
                candidate.setdefault("edit_source_observations",{}).update(
                    certified_solved_graph=observed,certified_solved_graph_sha256=_canonical_sha256(observed),
                    solver_and_source_certified=True,original_optimization_domain_recentered=False)
        _write(directory/"reconstruction-feedback.json",result)
        cache[key]=copy.deepcopy(result)
        if repair_domain_graph is not None:
            updated_key=_canonical_sha256({"candidate_id":candidate["id"],"graph":graph})
            cache[updated_key]=copy.deepcopy(result)
            edit_observation_cache[updated_key]=copy.deepcopy(candidate.get("edit_source_observations"))
        edit_observation_cache[key]=copy.deepcopy(candidate.get("edit_source_observations"))
        geometries.setdefault(geometry_key,{"geometry_sha256":geometry_key,"candidate_ids":[]})["candidate_ids"].append(candidate["id"])
        geometries[geometry_key].update(solver_status=result.get("solver_status"),
            source_passed=(result.get("source_validation") or source_validation).get("passed"),
            failure_reasons=(result.get("source_validation") or {}).get("reasons",[]),
            remaining_shape_dof=result.get("remaining_shape_dof"),issue_count=result.get("issue_count"))
        persist()
        return result
    selected_preflight_started=budget.clock()
    selected["constraint_feedback"]=verify(selected)
    selected_preflight_seconds=max(0.,budget.clock()-selected_preflight_started)
    record["initial_seed_audit"]["selected_preflight"]={
        "solver_status":selected["constraint_feedback"].get("solver_status"),
        "source_validation_passed":(selected["constraint_feedback"].get("topology_source_validation") or {}).get("passed"),
        "preflight_artifact":selected["constraint_feedback"].get("preflight_artifact"),
        "elapsed_seconds":round(selected_preflight_seconds,3)}
    beam=[selected]
    ready_seeds=[];diagnostic_seeds=[];last_seed_preflight_seconds=0.
    edit_time_reserve=min(300.,budget.max_seconds/2.)
    record["initial_seed_audit"]["minimum_edit_reserve_seconds"]=edit_time_reserve
    seed_hashes={geometry_fingerprint(selected["graph"])}
    admitted_seed_ids=set()
    preferred_bootstrap=False
    compound_numeric_rejected=False
    pending_line_repairs=_pending_source_line_repairs(selected,bundle.get("annotation_inventory",[]),
        selected["constraint_feedback"],budget.visited_operations)
    record["initial_seed_audit"]["source_line_priority"]={
        "status":"reserved" if pending_line_repairs else "not_needed",
        "operations":copy.deepcopy(pending_line_repairs),
        "scope":"Reserve a local source/numeric preflight, never automatic candidate admission.",
        "ground_truth_used":False}
    bootstrap_dir=root/"source-joint-bootstrap"
    if pending_line_repairs:
        record["initial_seed_audit"]["source_joint_bootstrap"]={
            "status":"skipped","reason":"source_line_numeric_preflight_prioritized",
            "ground_truth_used":False}
    elif beam_width>1 and budget.preflight_block_reason() is None:
        bootstrap_dir.mkdir(parents=True,exist_ok=True)
        compound,bootstrap_audit,compound_source=_source_joint_bootstrap(
            image_path,document,baseline,selected,bundle,bootstrap_dir,
            selected["constraint_feedback"],remaining_seconds=budget.remaining_seconds,
            edit_time_reserve=edit_time_reserve)
        record["initial_seed_audit"]["source_joint_bootstrap"]=bootstrap_audit
        persist()
        if compound is not None:
            compound_hash=geometry_fingerprint(compound["graph"])
            if compound_hash in seed_hashes:
                bootstrap_audit.update(status="rejected",reason="repeated_geometry")
            else:
                preflight_started=budget.clock()
                compound_feedback=verify(compound,source_validation=compound_source)
                compound["constraint_feedback"]=compound_feedback
                bootstrap_audit["numeric_preflight"]={
                    "solver_status":compound_feedback.get("solver_status"),
                    "preflight_artifact":compound_feedback.get("preflight_artifact"),
                    "elapsed_seconds":round(max(0.,budget.clock()-preflight_started),3),
                    "source_validation_passed":(compound_feedback.get("topology_source_validation") or {}).get("passed")}
                if preflight_admissible(compound_feedback):
                    bootstrap_audit.update(status="admitted",reason=None)
                    ready_seeds.append(compound)
                    beam.append(compound)
                    seed_hashes.add(compound_hash)
                    admitted_seed_ids.add(compound["id"])
                    record["initial_seed_audit"]["admitted_candidate_ids"].append(compound["id"])
                    bundle.setdefault("candidates",[]).append(compound)
                    bundle["candidate_count"]=len(bundle["candidates"])
                    preferred_bootstrap=True
                    parent_hash=geometry_fingerprint(selected["graph"])
                    for operation in bootstrap_audit["operations"]:
                        _,key=budget.visit_operation(parent_hash,operation)
                        budget.visited_operations[key].update(
                            status="source_joint_bootstrap",candidate_id=compound["id"],reasons=[])
                        parent_hash=bootstrap_audit["intermediate"]["geometry_sha256"]
                else:
                    compound_numeric_rejected=True
                    bootstrap_audit.update(status="rejected",reason=(
                        compound_feedback.get("preflight_skip_reason") or
                        "source_joint_bootstrap_numeric_preflight_not_admissible"))
            _write(bootstrap_dir/"source-joint-bootstrap-audit.json",bootstrap_audit)
            persist()
    else:
        record["initial_seed_audit"]["source_joint_bootstrap"]={
            "status":"skipped","reason":"initial_beam_or_preflight_budget_unavailable",
            "ground_truth_used":False}
    # The planner's one selected graph is a preference, not the entire search
    # frontier. Recheck up to two other locally admissible source hypotheses
    # under this same preflight/time budget before any local edit is proposed.
    original_candidates={row["id"]:row for row in bundle.get("candidates",[])
                         if isinstance(row,dict) and isinstance(row.get("id"),str)}
    for candidate_id in initial_seed_ids:
        audit={"candidate_id":candidate_id,"status":"skipped","reason":None}
        record["initial_seed_audit"]["attempts"].append(audit)
        if pending_line_repairs:
            audit["reason"]="source_line_numeric_preflight_prioritized";continue
        if preferred_bootstrap:
            audit["reason"]="source_joint_bootstrap_preferred_for_edit_budget";continue
        if len(beam)>=beam_width:
            audit["reason"]="initial_beam_full";continue
        if candidate_id==selected["id"]:
            audit["reason"]="already_selected";continue
        candidate=original_candidates.get(candidate_id)
        if candidate is None:
            audit["reason"]="candidate_not_in_original_bundle";continue
        fingerprint=geometry_fingerprint(candidate["graph"])
        audit["geometry_sha256"]=fingerprint
        if fingerprint in seed_hashes:
            audit["reason"]="repeated_geometry";continue
        # A second slow seed should not consume the time needed to execute an
        # edit from the first. The first alternate gets one bounded trial; a
        # second is attempted only when its last observed preflight cost fits
        # alongside a half-budget edit reserve.
        if budget.remaining_seconds()<edit_time_reserve+(
                last_seed_preflight_seconds if admitted_seed_ids else 0.):
            audit["reason"]="initial_seed_edit_time_reserved";continue
        if budget.remaining_seconds()<=0:
            audit["reason"]="topology_search_time_budget_exhausted";continue
        source_check=_topology_source_validation(image_path,baseline,candidate["graph"])
        audit["source_validation_passed"]=source_check.get("passed") is True
        audit["source_validation_reasons"]=source_check.get("reasons",[])
        if source_check.get("passed") is not True:
            audit["reason"]="source_validation_failed";continue
        if (compound_numeric_rejected and
                budget.remaining_seconds()<edit_time_reserve+selected_preflight_seconds):
            audit["reason"]="post_compound_failure_edit_time_reserved"
            audit["minimum_remaining_for_alternate_seconds"]=round(
                edit_time_reserve+selected_preflight_seconds,3)
            continue
        preflight_started=budget.clock()
        feedback=verify(candidate,source_validation=source_check)
        last_seed_preflight_seconds=max(0.,budget.clock()-preflight_started)
        candidate["constraint_feedback"]=feedback
        audit["preflight_status"]=feedback.get("solver_status")
        audit["preflight_artifact"]=feedback.get("preflight_artifact")
        audit["preflight_elapsed_seconds"]=round(last_seed_preflight_seconds,3)
        if feedback.get("solver_status")=="preflight_not_run":
            audit["reason"]=feedback.get("preflight_skip_reason") or "preflight_not_run";continue
        if preflight_admissible(feedback):
            audit["status"]="admitted"
            ready_seeds.append(candidate)
        elif feedback.get("binding_status")!="not_run":
            candidate["diagnostic_only"]=True
            candidate["formal_binding_coverage_for_ranking"]=0
            audit["status"]="diagnostic_only"
            diagnostic_seeds.append(candidate)
        else:
            audit["reason"]="preflight_failed";continue
        beam.append(candidate)
        seed_hashes.add(fingerprint)
        admitted_seed_ids.add(candidate_id)
        record["initial_seed_audit"]["admitted_candidate_ids"].append(candidate_id)
        persist()
    # Spend the first edit opportunity on newly opened source hypotheses;
    # the original selected candidate remains the publishable fallback.
    beam=[*ready_seeds,selected,*diagnostic_seeds]
    beam.sort(key=lambda row: source_repair_branch_priority(
        row, annotation_inventory=bundle.get("annotation_inventory", [])))
    record["initial_seed_audit"]["first_round_parent_ids"]=[row["id"] for row in beam]
    persist()
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
            source_line_priority=bool(_pending_source_line_repairs(parent,
                bundle.get("annotation_inventory",[]),feedback,budget.visited_operations))
            proposed,step=_topology_edit_stage(image_path,document,baseline,parent,bundle,directory,
                editor_provider=editor_provider,evaluator_provider=evaluator_provider,use_api=use_api,
                round_index=number,feedback=feedback,verifier=verify,operation_filter=operation_filter,
                candidate_namespace=f"r{number:02d}-b{branch_index:02d}",provider_guard=provider_guard,
                defer_after_primary_progress=number<max_rounds,
                defer_for_source_exploration=number<max_rounds and beam_width>1,
                source_line_preflight_first=source_line_priority)
            step.update(round=number,branch=branch_index,parent_geometry_sha256=parent_hash,feedback=feedback)
            if step["acceptance_gate"].get("accepted") and geometry_fingerprint(proposed["graph"])==parent_hash:
                step["acceptance_gate"].update(accepted=False,reason="repeated_geometry")
                step["final_candidate_id"]=parent["id"]
            for operation in step.get("execution",{}).get("operations",[]):
                value=operation.get("operation")
                if not isinstance(value,dict):continue
                _,key=budget.visit_operation(parent_hash,value)
                operation["parent_operation_sha256"]=key
                if operation.get("preflight_status") in {"deferred_not_run","not_run_budget"}:
                    # The kernel constructed this proposal, but it has no
                    # independent binding/solve receipt. It remains retryable.
                    budget.visited_operations.pop(key,None)
                    continue
                budget.visited_operations[key].update(status=operation.get("preflight_status",operation.get("status")),
                    reasons=operation.get("preflight_reasons") or [operation.get("reason")],
                    candidate_id=operation.get("candidate_id"))
            # The stage filter and execution receipts already mark attempted edits.
            # Unexecuted editor proposals must remain eligible next round.
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
                    incumbent_feedback=selected.get("constraint_feedback") or {}
                    if preflight_admissible(incumbent_feedback) and incumbent_feedback.get("solver_accepted") is True:
                        incumbent_gate=_source_locked_constraint_regression(incumbent_feedback,row["constraint_feedback"])
                        row["incumbent_constraint_regression_gate"]=incumbent_gate
                        if not incumbent_gate["passed"]:
                            continue
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
        credible=retain_distinct(next_pool,width=beam_width,preferred_id=preferred,
            annotation_inventory=bundle.get("annotation_inventory",[]))
        selected=credible[0] if credible else previous_selected
        # One publishable/fallback parent plus at most two kernel-safe diagnostic
        # branches lets a two-region repair span rounds without publishing an
        # intermediate failed numerical candidate or claiming its R coverage.
        exploration=retain_distinct(exploration_pool,width=min(2,beam_width),preferred_id=None,
            annotation_inventory=bundle.get("annotation_inventory",[]))
        beam=[selected];beam_hashes={geometry_fingerprint(selected["graph"])}
        for row in [*exploration,*credible[1:]]:
            key=geometry_fingerprint(row["graph"])
            if key in beam_hashes:continue
            beam.append(row);beam_hashes.add(key)
            if len(beam)>=beam_width:break
        beam=beam[:beam_width]
        # Expand unfinished source-valid repairs before spending another round
        # on the unchanged fallback. Search order does not change `selected` or
        # promote a diagnostic graph to a publishable candidate.
        beam.sort(key=lambda row: (source_repair_branch_priority(
            row, annotation_inventory=bundle.get("annotation_inventory", [])),
            not bool(row.get("diagnostic_only"))))
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
        step["final_candidate_id"]=selected["id"]
        step["selection_origin"]=("initial_seed_promotion" if selected["id"] in admitted_seed_ids else
                                  "initial_selection" if selected["id"]==record["initial_seed_audit"]["selected_candidate_id"] else
                                  "local_edit")
        frontier_changed={geometry_fingerprint(row["graph"]) for row in beam}!=previous_beam
        if not changed and not frontier_changed and not record["stop_reason"]:
            can_revise=use_api and (fresh_operations or retryable_failure) and number<max_rounds
            if retryable_failure and can_revise:
                step["retry_reason"]="provider_failure_within_existing_round_budget"
            if not can_revise:record["stop_reason"]="no_accepted_improvement"
        record["rounds"].append(step)
        record["final_candidate_id"]=selected["id"]
        record["final_feedback"]=verify(selected)
        _maybe_extend_topology_time_budget(budget,record,previous_selected,selected,
            bundle.get("annotation_inventory",[]),round_number=number,
            max_rounds=max_rounds,use_api=use_api)
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
        selected=next((row for row in beam if row["id"] in passing and not row.get("diagnostic_only") and
                       preflight_admissible(row.get("constraint_feedback") or {})),
                      previous_selected if record["rounds"] else selected)
        record["stop_reason"]="final_source_recheck_failed"
    record["final_candidate_id"]=selected["id"]
    record["final_selection_origin"]=("initial_seed_promotion" if selected["id"] in admitted_seed_ids else
                                      "initial_selection" if selected["id"]==record["initial_seed_audit"]["selected_candidate_id"] else
                                      "local_edit")
    record.update(status="completed",stop_reason=record["stop_reason"] or "round_budget_exhausted")
    persist()
    # Preserve old receipt consumers while adding the full per-round audit.
    summary={**(record["rounds"][-1] if record["rounds"] else {}),"rounds":record["rounds"],"max_rounds":max_rounds,
             "stop_reason":record["stop_reason"],"final_candidate_id":selected["id"],
             "beam_width":beam_width,"budget":record["budget"],"visited_operations":record["visited_operations"],
             "time_budget_extension":record["time_budget_extension"],
             "visited_geometries":record["visited_geometries"],"final_source_rechecks":record["final_source_rechecks"],
             "initial_seed_audit":record["initial_seed_audit"],"final_selection_origin":record["final_selection_origin"],
             "accepted_round_count":sum(row["acceptance_gate"].get("accepted") is True for row in record["rounds"]),
             "accepted_local_edit_count":sum(row["acceptance_gate"].get("accepted") is True and
                                              row.get("selection_origin")=="local_edit" for row in record["rounds"]),
             "accepted_seed_promotion_count":sum(row["acceptance_gate"].get("accepted") is True and
                                                  row.get("selection_origin")=="initial_seed_promotion" for row in record["rounds"])}
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
    # The generated base wrapper retains the same source geometry but also
    # carries independently localized annotation support and stable lineage.
    # Try it before the original extraction graph when another candidate
    # fails source validation; otherwise editing the unannotated fallback
    # cannot insert a directed fillet even though its source arrow was found.
    admissible_ids = {row.get("candidate_id") for row in local_evaluation.get("evaluated", [])
                      if row.get("admissible")}
    if (allow_alternatives and base["id"] in admissible_ids
            and selected.get("id") != base["id"] and base["graph"] != base_graph):
        choices.append((base, base["graph"], "generated_base_candidate"))
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
                editor_provider=editor_provider,evaluator_provider=evaluator_provider,use_api=use_api,progress=progress,
                initial_seed_ids=tuple(local.get("ranked_candidate_ids",())) if annotation_evidence else ())
            selected_id=selected["id"]
            selected_graph=selected["graph"]
            if edit_stage.get("final_selection_origin")=="initial_seed_promotion":
                source="bounded_source_seed_promotion"
            elif edit_stage.get("final_selection_origin")=="local_edit":
                source=("multimodal_local_topology_edit" if annotation_evidence else
                        "multimodal_local_topology_edit_without_annotation_leader")
        # The exact preserved graph is not necessarily a generated bundle
        # member. Only newly registered/unchanged generated candidates pass
        # through materialize_selected_candidate's membership/hash guard.
        if preserved_base_before_edit and (edit_stage or {}).get("final_selection_origin","initial_selection")=="initial_selection":
            materialization=preserve_base_materialization()
        elif annotation_evidence or (edit_stage or {}).get("final_selection_origin")=="local_edit":
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
          "radius_source_observations","angle_source_observations","relations","source_arrow_ownership")
    if not isinstance(inventory,dict) or any(key not in inventory for key in ("all_records","all_candidates")):
        return None
    evidence={key:inventory.get(key) for key in keys}
    ownership=evidence.get("source_arrow_ownership")
    if isinstance(ownership,dict) and isinstance(ownership.get("ocr_local_candidate_searches"),list):
        # This one field is measured wall time, not a source observation.
        # Keep budgets/deadlines, completion/truncation status, rejected
        # hypotheses and all geometry. Never drop timing keys recursively.
        evidence["source_arrow_ownership"]={**ownership,"ocr_local_candidate_searches":[
            {key:value for key,value in search.items() if key!="elapsed_seconds"}
            if isinstance(search,dict) else search
            for search in ownership["ocr_local_candidate_searches"]]}
    return _canonical_sha256(evidence)


def _constraint_signature(row):
    identity={key:row.get(key) for key in ("kind","record_id","entities","nodes","value","angle_mode","reference_axis")}
    from .parametric_solver import _strict_source_nominal
    identity["strict_source_nominal"]=_strict_source_nominal(row)
    if "anchors" in row:
        # _validate_inputs normalizes unordered intersection supports, while
        # retaining the ordered endpoints that define a signed measurement.
        identity["anchors"]=row["anchors"]
    if row.get("source_station_relation") is not None:
        identity["source_station_relation"]=row["source_station_relation"]
    if row.get('source_tolerance') is not None:
        from .dimension_policy import _validated_tolerance_metadata
        identity['source_tolerance']=_validated_tolerance_metadata(row['source_tolerance'])
    return _canonical_sha256(identity)


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


def _fillet_tangent_contract(graph, bindings, solution):
    """Audit arrow-verified radius arcs at their actual LINE joints.

    A rounded corner has a geometric tangency obligation, but this diagnostic
    never manufactures a constraint from an attractive fitted join.  A joint
    becomes certified only if the current binding pass verified source ink or
    a source-bound fillet construction and the solver satisfied that relation.
    These are separate evidence classes: construction is not an independent
    measurement of tangency in the source image.
    Missing source evidence stays visibly *unknown*, rather than being counted
    as either a successful fillet or a failed solver constraint.
    """
    bindings, solution = bindings or {}, solution or {}
    bound = bindings.get("constraints") or []
    confirmed = set((bindings.get("radius_binding_coverage") or {}).get("confirmed_arrow_records") or [])
    entities = {row["id"]: row for row in graph.get("entities", [])}
    incident = {}
    for entity in entities.values():
        for node in {entity.get("start_node"), entity.get("end_node")} - {None}:
            incident.setdefault(node, []).append(entity)
    independently_verified = {
        row.get("relation_id") for row in bindings.get("bindings", [])
        if row.get("accepted") is True and row.get("source") == "source_geometry"
        and isinstance(row.get("evidence"), dict)
        and row["evidence"].get("verified") is True
        and row["evidence"].get("evidence_class") != "source_bound_design_construction"
    }
    design_constructed = {
        row.get("relation_id") for row in bindings.get("bindings", [])
        if row.get("accepted") is True and row.get("source") == "source_bound_design_fillet"
        and isinstance(row.get("evidence"), dict)
        and row["evidence"].get("verified") is True
        and row["evidence"].get("evidence_class") == "source_bound_design_construction"
        and row["evidence"].get("independent_source_tangent_measurement") is False
    }
    design_reviewed = {
        row.get("relation_id") for row in bindings.get("bindings", [])
        if row.get("accepted") is True and row.get("source") == "source_design_review"
        and isinstance(row.get("evidence"), dict)
        and row["evidence"].get("verified") is True
        and row["evidence"].get("evidence_class") == "source_design_review"
        and row["evidence"].get("independent_source_tangent_measurement") is False
    }
    solver_accepted = solution.get("accepted") is True
    satisfied = {row.get("id") for row in solution.get("constraints", [])
                 if solver_accepted and row.get("passed") is True}
    failed = {row.get("id") for row in solution.get("constraints", []) if row.get("passed") is False}
    tangent_relations = {}
    for row in bound:
        relation = row.get("id", "")[1:]
        if row.get("kind") != "tangent" or relation not in independently_verified | design_constructed | design_reviewed:
            continue
        design = relation in design_constructed
        if design and (row.get("evidence_class") != "source_bound_design_construction" or
                       row.get("independent_source_tangent_measurement") is not False):
            continue
        if relation in design_reviewed and row.get('evidence_class')!='source_design_review':
            continue
        eids = row.get("entities") or []
        nodes = row.get("nodes") or []
        if len(eids) == 2 and len(set(eids)) == 2 and len(nodes) == 1:
            tangent_relations[(frozenset(eids), nodes[0])] = (row["id"], design, relation in design_reviewed)
    diagnostics = []
    seen = set()
    for radius in bound:
        record_id = radius.get("record_id")
        eids = radius.get("entities") or []
        if radius.get("kind") != "radius" or record_id not in confirmed or len(eids) != 1:
            continue
        key = record_id, eids[0]
        if key in seen:
            continue
        seen.add(key)
        entity = entities.get(eids[0])
        if entity is None or entity.get("type") != "ARC":
            diagnostics.append({"record_id": record_id, "arc_entity_id": eids[0],
                                "status": "radius_target_not_arc", "line_joints": []})
            continue
        joints = []
        for node in (entity.get("start_node"), entity.get("end_node")):
            if node is None:
                continue
            neighbors = [row for row in incident.get(node, []) if row["id"] != entity["id"]]
            if len(neighbors) != 1:
                joints.append({"node_id": node, "status": "ambiguous_incidence",
                               "neighbor_entity_ids": [row["id"] for row in neighbors]})
                continue
            neighbor = neighbors[0]
            if neighbor.get("type") != "LINE":
                continue
            relation_id, design, reviewed = tangent_relations.get(
                (frozenset((entity["id"], neighbor["id"])), node), (None, False, False))
            prefix = "design_reviewed" if reviewed else "design_constructed" if design else "source_verified"
            status = (prefix + "_and_satisfied" if relation_id in satisfied else
                      prefix + "_but_unsatisfied" if relation_id in failed else
                      prefix + "_without_solver_receipt" if relation_id else
                      "source_tangency_unverified")
            joints.append({"node_id": node, "line_entity_id": neighbor["id"],
                           "relation_id": relation_id, "status": status,
                           "evidence_class": "source_design_review" if reviewed else "source_bound_design_construction" if design else
                                             "independent_source_tangent_measurement" if relation_id else None})
        diagnostics.append({"record_id": record_id, "arc_entity_id": entity["id"],
                            "status": "review_line_arc_joints", "line_joints": joints})
    statuses = [joint["status"] for item in diagnostics for joint in item["line_joints"]
                if "line_entity_id" in joint]
    certified = sum(value.startswith("source_verified_") for value in statuses)
    passed = statuses.count("source_verified_and_satisfied")
    constructed = sum(value.startswith("design_constructed_") for value in statuses)
    construction_passed = statuses.count("design_constructed_and_satisfied")
    reviewed = sum(value.startswith("design_reviewed_") for value in statuses)
    review_passed = statuses.count("design_reviewed_and_satisfied")
    ambiguous = sum(joint.get("status") == "ambiguous_incidence"
                    for item in diagnostics for joint in item["line_joints"])
    return {"schema_version": "source-fillet-tangent-contract-v1", "ground_truth_used": False,
            "cohort": "bound_arrow_verified_radius_arcs",
            "solver_accepted": solver_accepted,
            "arrow_verified_radius_arcs": len(diagnostics), "line_arc_joints": len(statuses),
            "source_verified_joints": certified, "satisfied_source_verified_joints": passed,
            "design_constructed_joints": constructed,
            "satisfied_design_constructed_joints": construction_passed,
            "design_reviewed_joints": reviewed, "satisfied_design_reviewed_joints": review_passed,
            "unverified_joints": statuses.count("source_tangency_unverified"),
            "ambiguous_incidence_joints": ambiguous,
            "all_source_verified_joints_satisfied": certified > 0 and certified == passed,
            "all_design_constructed_joints_satisfied": constructed > 0 and constructed == construction_passed,
            "all_line_arc_joints_certified": bool(statuses) and not ambiguous and passed + construction_passed + review_passed == len(statuses),
            "scope": "Diagnostic coverage only; unknown joints are not accepted tangencies or solver failures.",
            "fillets": diagnostics}


def _record_fillet_tangent_contract(output, graph, bindings, solution, feedback=None):
    contract = _fillet_tangent_contract(graph, bindings, solution)
    if feedback is not None:
        feedback["fillet_tangent_contract"] = copy.deepcopy(contract)
    _write(Path(output)/"fillet-tangent-contract.json", contract)
    return contract


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
        if not required<=current and (graph.get('source_evidence') or {}).get('source_state_resume'):
            from .source_topology_initial_guess import retain_line_seed_bindings
            recheck,_,_=retain_line_seed_bindings(image_path,document,baseline,graph,recheck,
                read(recheck_dir/'binding-candidates.json'),recheck_dir)
            current=_constraint_signatures(graph,recheck.get('constraints',[]))
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
        from .parametric_solver import native_axis_relation_checks, solve_parametric
        old_axes=native_axis_relation_checks(solution['entities'],bindings['constraints'])
        if old_axes.get('passed') is not True:
            # Old accepted checkpoints allowed 0.1-degree axis residuals.
            # Rejecting their certificate must not erase the good initial
            # values or unbind independently revalidated source dimensions.
            # Re-solve the COMPLETE unchanged subset in the ORIGINAL domain.
            repair={'reason':'legacy_axis_tolerance_requires_current_exact_equations',
                'previous_native_axis_check':old_axes,'accepted':False,
                'known_equations_changed':False,'original_domain_recentered':False,
                'reference_geometry_used':False,'publication_approved':False}
            _write(output/'legacy-axis-reoptimization.json',repair)
            repaired=_final_source_multistart(image_path,document,baseline,graph,
                bindings['constraints'],output/'legacy-axis-reoptimization',solver=solve_parametric,
                budget_seconds=60.,initial_seed_entities=solution['entities'])
            _write(output/'legacy-axis-reoptimized-solution.json',repaired)
            if not repaired.get('accepted') or not _constraint_receipts_complete(bindings,repaired,graph):
                raise ValueError('checkpoint_current_axis_reoptimization_not_certified')
            repaired_source=_solved_source_validation(image_path,document,baseline,graph,repaired['entities'])
            repair['original_source_validation']=repaired_source
            repair['solver_status']=repaired.get('status')
            repair['accepted']=bool(repaired_source.get('passed') and
                native_axis_relation_checks(repaired['entities'],bindings['constraints']).get('passed'))
            _write(output/'legacy-axis-reoptimization.json',repair)
            if not repair['accepted']:
                raise ValueError('checkpoint_current_axis_reoptimization_source_failed')
            solution,solved=repaired,repaired_source
        export_solution={**solution,"constraints":copy.deepcopy(bindings["constraints"])}
        candidate_dir=output/"export"
        model=export_parametric(image_path,baseline,export_solution,candidate_dir)
        exact=model["validation"]["exact_radius_validation"]
        if not exact.get("passed") or not exact.get("dxf_readback_performed"):
            raise ValueError("checkpoint_exact_radius_readback_failed")
        _record_published_source_validation(model,solved,baseline)
        contract=annotation_radius_contract(bindings,solution)
        contract.update(candidate_satisfied=contract["satisfied"],satisfied=False,
                        all_annotated_radii_verified=False,current_dxf_verified=False,
                        publication_status="candidate_only",exact_radius_validation=copy.deepcopy(exact))
        fresh_feedback=reconstruction_feedback(graph,bindings,solution)
        _record_fillet_tangent_contract(output,graph,bindings,solution,fresh_feedback)
        fresh_feedback["source_validation"]=solved
        fresh_feedback["post_solve_source_accepted"]=True
        receipt={"status":"certified","input_identity":identity,
                 "source_evidence_sha256":_binding_evidence_sha256(inventory),
                 "legacy_receipt_revalidated":not manifest.is_file(),
                 "source_validation_passed":True,
                 "source_and_mask_passed":solved.get("oracle_mask_validation",{}).get("passed") is not False,
                 "priority_resolution":copy.deepcopy(solved.get("priority_resolution")),
                 "constraint_receipts_verified":True,
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
              "checkpoint_exact_radius_readback_failed","checkpoint_current_axis_reoptimization_not_certified",
              "checkpoint_current_axis_reoptimization_source_failed"}
        receipt.update(status="rejected",reason=str(error) if str(error) in safe else "checkpoint_certification_failed",
                       error_type=type(error).__name__)
        _write(output/"certificate.json",receipt)
        return None,receipt


def _final_checkpoint_replacement_gate(trusted, attempt, *, evidence_sha256):
    """Same source evidence plus non-regressing certified coverage, never a merge."""
    same_evidence_bytes=evidence_sha256 is not None and evidence_sha256==trusted["certificate"].get("source_evidence_sha256")
    same=same_evidence_bytes
    transition=attempt.get("source_binding_transition_certificate") or {}
    carried=transition.get("carried_binding_certificate") or {}
    controlled_transition=bool(transition.get("status")=="accepted_seed" and
        transition.get("all_known_constraints_preserved") is True and
        transition.get("fresh_new_radius_binding_complete") is True and
        transition.get("topology_changed") is False and carried.get("passed") is True and
        transition.get("input_identity_verified_against_checkpoint") is True)
    revalidation=attempt.get('source_binding_revalidation_certificate') or {}
    known_carry=revalidation.get('carried_binding_certificate') or {}
    original=trusted['bindings'].get('constraints',[])
    known_revalidated=bool(revalidation.get('status')=='known_source_subset_revalidated' and
        revalidation.get('passed') is True and revalidation.get('ground_truth_used') is False and
        revalidation.get('input_identity_verified_against_checkpoint') is True and
        revalidation.get('previous_constraint_count')==len(original) and bool(original) and
        revalidation.get('previous_typed_constraints_sha256')==_canonical_sha256(
            sorted(_constraint_signatures(trusted['graph'],original))) and
        known_carry.get('passed') is True and len(known_carry.get('checks',[]))==len(original) and
        all(row.get('passed') is True for row in known_carry.get('checks',[])))
    same=same or controlled_transition or known_revalidated
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
            "reasons":list(dict.fromkeys(reasons)),"same_independent_source_evidence":same_evidence_bytes,
            "source_evidence_remeasured_with_certified_known_bindings":controlled_transition,
            "known_source_subset_independently_revalidated":known_revalidated,
            "constraint_regression":regression,"previous_verified_radius_records":sorted(old),
            "attempt_verified_radius_records":sorted(new),"reference_geometry_used":False}


def refine_parametric(image_path, document, baseline, output_dir, *, provider=None, planner_provider=None,
                      editor_provider=None, evaluator_provider=None, use_api=False, progress=None,
                      frozen_topology=None, trusted_preflight_dir=None, source_first=False,
                      allow_frozen_topology_repair=False):
    if source_first:
        from .reconstruction_priority import source_first_model
        baseline=source_first_model(baseline)
    from .topology import build_topology
    from .constraint_binding import analyze_constraint_bindings
    from .parametric_solver import solve_parametric
    output=Path(output_dir)
    current=baseline
    emit=progress or (lambda stage,message:None)
    oracle_mask=baseline.get("oracle_mask_conditioned") is True
    input_ocr_sha256=canonical_document_sha256(document)
    stage={"status":"running","accepted":False,"ground_truth_used":oracle_mask,
           "ground_truth_use":"raster_mask_input_only" if oracle_mask else "none",
           "oracle_mask_conditioned":oracle_mask,"all_dimensions_verified":False,
           "geometry_updated_by_api":False,"dimensions_updated_by_api":False,
           "provider":{"status":"not_invoked","network_requests":0},"stages":[]}
    provenance_context={"input_source_image_sha256":None,
                         "input_ocr_canonical_sha256":input_ocr_sha256,
                         "input_ocr_hash_encoding":"canonical_json_utf8_compact_v1",
                        "mask_input":"gt_raster_development_input" if oracle_mask else "segmentation_or_reviewed_material_mask"}
    def checkpoint(name,message):
        stage["stage"]=name;stage["stages"].append(name)
        _write(output/"parametric-stage.json",stage)
        _write_workflow_provenance(output,stage,provenance_context)
        emit(name,message)
    def radius_seed_attempt(seed_graph,seed_bindings,directory,inventory=None):
        from .source_radius_hypotheses import (refine_source_owned_radius_hypotheses,
                                               certify_refined_radius_seed)
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        path=directory/"binding-candidates.json"
        if inventory is None:
            if not path.is_file():return seed_graph,seed_bindings,None
            inventory=json.loads(path.read_text(encoding="utf8"))
        proposed,hypothesis=refine_source_owned_radius_hypotheses(
            image_path,document,baseline,seed_graph,seed_bindings,inventory,directory/"source-radius-hypotheses")
        stage["source_radius_hypotheses"]=hypothesis
        if hypothesis.get("status")!="source_verified_seed":return seed_graph,seed_bindings,None
        checkpoint("source_radius_rebinding","精确半径候选已通过原始源图检查；重新在线绑定当前有限区间并保留已知约束。")
        proposed,bound,certification=certify_refined_radius_seed(
            image_path,document,baseline,seed_graph,seed_bindings,inventory,proposed,hypothesis,
            directory/"source-radius-certification",provider=provider,use_api=use_api)
        stage["source_radius_seed_certification"]=certification
        if certification.get("status")!="accepted_seed":return seed_graph,seed_bindings,certification
        _write(directory/"initial-radius-binding-candidates.json",inventory)
        _write(directory/"initial-radius-constraint-bindings.json",seed_bindings)
        shutil.copy2(Path(bound["inventory_artifact"]),path)
        _write(directory/"topology.json",proposed)
        _write(directory/"constraint-bindings.json",bound)
        stage["continuous_seed_refinement"]={"topology_changed":False,
            "unknown_coordinates_reoptimized":True,"original_source_comparator_preserved":True}
        return proposed,bound,certification
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
            coverage=reconstruction_contract(updated["entities"],stage,updated["validation"])
            updated["validation"]["reconstruction_contract"]=coverage
            stage["reconstruction_contract"]=copy.deepcopy(coverage)
            _write(candidate_dir/"reconstruction-contract.json",coverage)
            _write(candidate_dir/"validation.json",updated["validation"])
        else:complete_radii=False
        complete=bool(is_parametric and coverage["satisfied"])
        parametric_status=("completed" if complete else "completed_with_unresolved_attributes" if complete_radii
                           else "completed_with_unresolved_radii")
        published_stage={**copy.deepcopy(stage),"status":parametric_status if is_parametric else "source_topology_exported",
                         "accepted":complete,"constraint_subset_accepted":is_parametric,
                         "topology_exported":stage.get("topology_exported",False) or not is_parametric,
                         "geometry_updated_by_api":bool(is_parametric and api_geometry),
                         "dimensions_updated_by_api":bool(is_parametric and api_dimensions),
                         "stage":kind+"_export","stages":[*stage["stages"],kind+"_publish",kind+"_export"]}
        published_stage.pop("publication",None)
        if is_parametric:published_stage["annotation_radius_contract"]=copy.deepcopy(contract)
        updated["parameterization"]=published_stage
        updated["source_input_identity"]=copy.deepcopy(stage["source_input_identity"])
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
            stage.update(status=parametric_status if is_parametric else "running",accepted=complete,
                         constraint_subset_accepted=is_parametric,
                         topology_exported=published_stage["topology_exported"],
                         geometry_updated_by_api=published_stage["geometry_updated_by_api"],
                         dimensions_updated_by_api=published_stage["dimensions_updated_by_api"])
            stage["publication"]["status"]="committed"
            if is_parametric:
                stage["annotation_radius_contract"]=contract
                _write(output/"radius-contract.json",contract)
                _write(output/"reconstruction-contract.json",coverage)
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
    def complete_published_subset(published, original_graph):
        """Continue from a published subset without moving the original domain."""
        nonlocal document
        if not stage.get("constraint_subset_accepted") or stage.get("accepted"):
            return published
        from .ocr import canonical_records
        if not canonical_records(document):
            stage["post_solve_source_completion"]={"accepted":False,"reason":"no_source_annotation_records",
                                                   "ground_truth_used":False}
            return published
        from .source_constraint_completion import complete_source_constraints
        previous_stage=copy.deepcopy(stage)
        metadata_names=("binding-candidates.json","constraint-bindings.json","parametric-solution.json",
            "reconstruction-feedback.json","fillet-tangent-contract.json","radius-contract.json",
            "reconstruction-contract.json","topology.json","topology-plan.json",*CORE)
        metadata_replaced=False
        try:
            bound=json.loads((output/"constraint-bindings.json").read_text(encoding="utf8"))
            sol=json.loads((output/"parametric-solution.json").read_text(encoding="utf8"))
            inventory=json.loads((output/"binding-candidates.json").read_text(encoding="utf8"))
            from .angle_target_provider import unresolved_angle_record_ids,angle_localization_provider
            missing_angles=unresolved_angle_record_ids(document,bound)
            angle_provider=(angle_localization_provider(provider,editor_provider)
                if missing_angles and use_api else None)
            if angle_provider is not None:
                # Frozen topology replay skips initial localization. Even in
                # a fresh run an absent candidate must trigger source search,
                # rather than asking the binder to select a nonexistent ID.
                from .angle_target_provider import locate_angle_targets
                checkpoint('source_angle_relocalization','针对未绑定角度重新查看原图及局部放大图，定位有限斜线与双箭头。')
                document,angle_receipt=locate_angle_targets(image_path,document,
                    output/'source-obligation-angle-retry',angle_provider,
                    record_ids=missing_angles)
                stage['post_solve_angle_target_provider']={k:v for k,v in angle_receipt.items() if k!='proposals'}
            checkpoint("source_constraint_completion","从有效部分解重新读取原图尺寸和接点；保留已知约束，再在原问题中联合求解。")
            bound,sol,inventory,receipt=complete_source_constraints(image_path,document,baseline,
                original_graph,bound,sol,inventory,output/"source-constraint-completion",
                provider=provider,use_api=use_api)
            stage["post_solve_source_completion"]=receipt
            completion_graph=original_graph
            any_completion_accepted=receipt['accepted']
            # A failed derivative observation at a fitted vertex must not
            # starve a source-supported design contact. Review unknown joints
            # separately, then move their contacts in the SAME enlarged solve.
            from .source_joint_completion import complete_source_joints
            from .joint_review_provider import JointReviewProvider
            joint_settings=getattr(provider,'settings',None) if use_api else None
            joint_provider=JointReviewProvider(joint_settings) if joint_settings is not None else None
            checkpoint('source_joint_completion','从原图独立复核圆角与接点关系，固定已知标注并联合求解切点和圆心。')
            bound,sol,inventory,joints=complete_source_joints(image_path,document,baseline,
                original_graph,bound,sol,inventory,output/'source-joint-completion',
                provider=joint_provider,use_api=use_api)
            stage['post_solve_source_joint_completion']=joints
            any_completion_accepted=any_completion_accepted or joints['accepted']
            if frozen_topology is None or allow_frozen_topology_repair:
                from .source_topology_completion import complete_source_topology
                checkpoint('source_topology_completion','依据未满足标注修复对象类型与局部拓扑；保留既有约束并重新联合求解。')
                completion_graph,bound,sol,inventory,structural=complete_source_topology(
                    image_path,document,baseline,original_graph,bound,sol,inventory,
                    output/'source-topology-completion',provider=provider,
                    editor_provider=editor_provider,use_api=use_api)
                stage['post_solve_source_topology_completion']=structural
                any_completion_accepted=any_completion_accepted or structural['accepted']
            if not any_completion_accepted:
                checkpoint("source_constraint_completion_retained","没有新增且联合可行的源约束；保留已有有效DXF和未完成记录。")
                return published
            candidate_dir=output/"source-constraint-completion"/"parametric-export"
            exported=export_parametric(image_path,baseline,
                {**sol,"constraints":copy.deepcopy(bound["constraints"])},candidate_dir)
            source_check=_solved_source_validation(image_path,document,baseline,original_graph,sol["entities"])
            if not source_check.get("passed"):
                receipt.update(accepted=False,reason="completion_export_original_source_check_failed")
                return published
            # Preserve the prior complete metadata set before replacing it.
            for name in metadata_names:
                if (output/name).is_file():
                    shutil.copyfile(output/name,output/"source-constraint-completion"/("previous-"+name))
            metadata_replaced=True
            _write(output/"binding-candidates.json",inventory)
            bound["inventory_artifact"]=str(output/"binding-candidates.json")
            _write(output/"constraint-bindings.json",bound)
            _write(output/"parametric-solution.json",sol)
            _write(output/'topology.json',completion_graph)
            if completion_graph is not original_graph:
                plan=json.loads((output/'topology-plan.json').read_text('utf8'))
                plan['post_solve_structural_completion']={
                    'initial_candidate_id':original_graph.get('candidate_id'),
                    'selected_candidate_id':completion_graph.get('candidate_id'),
                    'receipt_artifact':'source-topology-completion/completion.json',
                    'known_source_obligations_preserved':True,'reference_geometry_used':False}
                _write(output/'topology-plan.json',plan)
                stage['topology'].update(entity_count=len(completion_graph['entities']),
                    node_count=len(completion_graph['nodes']),
                    selected_candidate_id=completion_graph.get('candidate_id'),
                    selection_source='post_solve_source_obligation_completion')
            feedback=reconstruction_feedback(completion_graph,bound,sol)
            stage["fillet_tangent_contract"]=_record_fillet_tangent_contract(output,completion_graph,bound,sol,feedback)
            _write(output/"reconstruction-feedback.json",feedback)
            contract=annotation_radius_contract(bound,sol)
            contract.update(candidate_satisfied=contract["satisfied"],satisfied=False,
                all_annotated_radii_verified=False,current_dxf_verified=False,publication_status="candidate_only",
                exact_radius_validation=copy.deepcopy(exported["validation"]["exact_radius_validation"]))
            stage.update(binding_counts=bound.get("counts",{}),constraints=bound["constraints"],
                radius_binding_coverage=bound.get("radius_binding_coverage"),issues=bound.get("issues",[]),
                solver={k:v for k,v in sol.items() if k not in {"entities","nodes","candidate_entities","candidate_nodes","constraints"}},
                solver_constraint_checks=copy.deepcopy(sol.get("constraints",[])),
                underconstrained=sol.get("underconstrained",True),reconstruction_feedback=feedback,
                annotation_radius_contract=contract,solved_source_validation=source_check,
                selected_binding_provider=bound.get("provider",{}))
            _record_published_source_validation(exported,source_check,baseline)
            exported["validation"].update(solved_source_validation=source_check,
                annotation_radius_contract=copy.deepcopy(contract),
                all_annotated_radii_verified=contract["candidate_satisfied"])
            _write(candidate_dir/"validation.json",exported["validation"])
            selected=bound.get("bindings",[])
            api_dimensions=any(r.get("accepted") and r.get("source")=="ocr_api_binding" for r in selected)
            return publish(candidate_dir,exported,kind="parametric",api_geometry=api_dimensions,
                           api_dimensions=api_dimensions)
        except InterruptedError:
            if metadata_replaced:
                for name in metadata_names:
                    previous=output/"source-constraint-completion"/("previous-"+name)
                    if previous.is_file():shutil.copyfile(previous,output/name)
                stage.clear();stage.update(previous_stage)
            raise
        except Exception as error:
            if metadata_replaced:
                for name in metadata_names:
                    previous=output/"source-constraint-completion"/("previous-"+name)
                    if previous.is_file():shutil.copyfile(previous,output/name)
                stage.clear();stage.update(previous_stage)
            stage["post_solve_source_completion"]={"accepted":False,"reason":"completion_failed_last_valid_preserved",
                "failure_type":type(error).__name__,"ground_truth_used":False}
            checkpoint("source_constraint_completion_retained","源约束补全未完成，保留上一套有效草稿。")
            return published
    try:
        for name in CORE:
            if (output/name).is_file() and not (output/("baseline-"+name)).exists():
                shutil.copyfile(output/name,output/("baseline-"+name))
        stage["source_input_identity"]=source_input_identity(image_path,document)
        provenance_context["input_source_image_sha256"]=stage["source_input_identity"]["source_image_sha256"]
        # Proposals are source-image pixels only. They augment detection and
        # must pass the same local ink/arrow checks before becoming bindings.
        if frozen_topology is None and use_api and editor_provider is not None and hasattr(editor_provider,"settings"):
            checkpoint("radius_target_localization","在线定位半径引线与箭尖；随后由原图像素复核，不读取参考CAD坐标。")
            from .radius_target_provider import RadiusTargetProvider, locate_radius_targets
            document,receipt=locate_radius_targets(image_path,document,baseline,output,
                                                   RadiusTargetProvider(editor_provider.settings),progress=checkpoint)
            stage["radius_target_provider"]={key:value for key,value in receipt.items() if key!="proposals"}
            checkpoint("radius_target_localization_finished","半径箭头提案已保存；未检测或歧义不会被视为没有标注。")
            checkpoint("angle_target_localization","在线结合原图和放大图定位斜线、基准线及角度双箭头；角度值仍由标注给定。")
            from .angle_target_provider import AngleTargetProvider, locate_angle_targets
            document,receipt=locate_angle_targets(image_path,document,output/"multimodal-angle-source",
                AngleTargetProvider(editor_provider.settings))
            stage["angle_target_provider"]={key:value for key,value in receipt.items() if key!="proposals"}
            checkpoint("angle_target_localization_finished","角度源图提案已保存，随后独立核验并参与拓扑修正。")
        checkpoint("topology","从近似边界提取图元连接图，并用原图笔画检查局部误分割。")
        if frozen_topology is None:
            base_graph=build_topology(image_path,document,baseline,output)
            graph=base_graph
        else:
            # Explicit final-stage replay of one frozen source candidate. This
            # does not rerun or claim a new upstream online reconstruction.
            graph=copy.deepcopy(frozen_topology);base_graph=graph
            topology_plan={"selected_candidate_id":graph.get("candidate_id"),
                "selection_source":"frozen_topology_final_stage_replay","local_evaluation":{},
                "selected_preflight_artifact":str(trusted_preflight_dir) if trusted_preflight_dir else None,
                "upstream_reconstruction_rerun":False,"ground_truth_used":False}
            _write(output/"topology.json",graph);_write(output/"topology-plan.json",topology_plan)
        # Use the first graph to identify genuinely missing/ambiguous arrows.
        # Reinspect those BEFORE planning: a new pair of radius arrows on one
        # primitive must be visible to partition/repair search, not arrive only
        # after topology has already been frozen for the numerical solver.
        # An explicitly requested frozen-topology replay still does not edit
        # its graph, and reports that narrower scope separately.
        # This bounded source stage never uses reference geometry and never
        # turns an unknown/out-of-scope arrow into an absent-label exemption.
        if use_api and editor_provider is not None and hasattr(editor_provider,"settings"):
            from .constraint_binding import build_binding_candidates
            from .radius_target_provider import RadiusTargetProvider, locate_radius_targets
            retry_dir=output/"source-obligation-radius-retry"
            initial_inventory=build_binding_candidates(image_path,document,baseline,graph,retry_dir/"before")
            supported={row["record_id"] for row in initial_inventory.get("all_candidates",[])
                if row.get("kind")=="radius" and row.get("local_reliable") is True}
            from .radius_target_provider import radius_retry_selection
            unresolved,retry_selection=radius_retry_selection(document,
                initial_inventory.get('radius_binding_coverage',{}),supported_record_ids=supported,
                prior_attempts=(stage.get('radius_target_provider') or {}).get('attempts',()))
            stage['radius_obligation_retry_selection']=retry_selection
            if unresolved:
                checkpoint("unresolved_radius_source_retry","重读初始对象图中尚未核实且仍有请求预算的半径引线；未知记录继续保留，随后执行拓扑与求解。")
                document,retry_receipt=locate_radius_targets(image_path,document,baseline,retry_dir,
                    RadiusTargetProvider(editor_provider.settings),progress=checkpoint,record_ids=unresolved[:12])
                stage["radius_obligation_retry"]={key:value for key,value in retry_receipt.items() if key!="proposals"}
                _write(output/"effective-radius-source-document.json",document)
                # Fresh proposals are not a contradiction of the known
                # subset. A resumed checkpoint must still pass the full fresh
                # binding/carry certificate below; positive contradictions
                # revoke it there. Ordinary preflight identities remain strict.
                from .source_state import retain_checkpoint_for_fresh_source_recheck
                recheck_saved=retain_checkpoint_for_fresh_source_recheck(
                    baseline,graph,frozen_replay=frozen_topology is not None)
                stage['radius_obligation_retry']['saved_subset_revalidation_required']=recheck_saved
                if not recheck_saved:
                    trusted_preflight_dir=None
                    if frozen_topology is not None:
                        topology_plan["selected_preflight_artifact"]=None
        if frozen_topology is None:
            checkpoint("topology_planning","结合最新引线证据生成 LINE/ARC 拓扑候选，再遍历局部修正与约束求解。")
            graph,topology_plan=_plan_topology(image_path,document,baseline,base_graph,output,
                planner_provider=planner_provider,editor_provider=editor_provider,
                evaluator_provider=evaluator_provider,use_api=use_api,progress=checkpoint)
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
                    solver_constraint_checks=copy.deepcopy(sol.get("constraints",[])),
                    issues=bound.get("issues",[]),radius_binding_coverage=bound.get("radius_binding_coverage"),
                    reconstruction_feedback=prepared["feedback"],solved_source_validation=prepared["source_validation"],
                    fillet_tangent_contract=prepared["feedback"].get("fillet_tangent_contract"),
                    annotation_radius_contract=copy.deepcopy(prepared["contract"]),
                    solver={key:value for key,value in sol.items() if key not in {"entities","nodes","candidate_entities","candidate_nodes","constraints"}},
                    selected_binding_provider=bound.get("provider",{}),binding_selection=selection,
                    underconstrained=sol.get("underconstrained",True))
                for name,value in (("constraint-bindings.json",bound),("parametric-solution.json",sol),
                                   ("reconstruction-feedback.json",prepared["feedback"]),
                                   ("radius-contract.json",stage["annotation_radius_contract"])):
                    _write(output/name,value)
                if stage["fillet_tangent_contract"] is not None:
                    _write(output/"fillet-tangent-contract.json",stage["fillet_tangent_contract"])
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
            attempt_graph,bindings,radius_certificate=radius_seed_attempt(
                graph,copy.deepcopy(trusted["bindings"]),attempt_dir,inventory=trusted["inventory"])
            if not radius_certificate or radius_certificate.get("status")!="accepted_seed":
                bindings=analyze_constraint_bindings(image_path,document,baseline,graph,attempt_dir,provider=provider,use_api=use_api)
                name=("binding-fallback-attempt.json" if (radius_certificate or {}).get("provider") else "binding-initial-attempt.json")
                _write(attempt_dir/name,{"provider":copy.deepcopy(bindings.get("provider",{}))})
            known_revalidation=None
            if not mask_is_hard(baseline) and (not radius_certificate or radius_certificate.get('status')!='accepted_seed'):
                from .source_state import revalidate_checkpoint_binding_subset
                inventory=json.loads((attempt_dir/'binding-candidates.json').read_text(encoding='utf8'))
                bindings,inventory,known_revalidation=revalidate_checkpoint_binding_subset(
                    image_path,document,baseline,trusted,bindings,inventory,attempt_dir)
                known_revalidation['input_identity_verified_against_checkpoint']=(
                    _preflight_input_identity(image_path,document,baseline,graph)==certificate['input_identity'])
                _write(attempt_dir/'checkpoint-known-source-revalidation.json',known_revalidation)
            stage["provider"]=bindings.get("provider",{})
            checkpoint("parametric_solve","独立求解在线重绑定候选，并比较当前证据与已核实约束。")
            seed_options=({"initial_seed_entities":attempt_graph["entities"]} if attempt_graph is not graph else {})
            solution=_final_source_multistart(image_path,document,baseline,graph,
                bindings.get("constraints",[]),attempt_dir,solver=solve_parametric,**seed_options)
            _write(attempt_dir/"parametric-solution.json",solution)
            attempted_entities=solution.get("entities") if solution.get("accepted") else solution.get("candidate_entities")
            source_check=(_solved_source_validation(image_path,document,baseline,graph,attempted_entities)
                          if attempted_entities else {"passed":False,"reasons":["final_numerical_candidate_unavailable"]})
            feedback=reconstruction_feedback(attempt_graph,bindings,solution)
            _record_fillet_tangent_contract(attempt_dir,attempt_graph,bindings,solution,feedback)
            source_failure_feedback(feedback,attempt_graph,source_check)
            _write(attempt_dir/"reconstruction-feedback.json",feedback)
            _write(attempt_dir/"solved-source-validation.json",source_check)
            contract=annotation_radius_contract(bindings,solution)
            contract.update(candidate_satisfied=contract["satisfied"],satisfied=False,
                            all_annotated_radii_verified=False,current_dxf_verified=False,
                            publication_status="candidate_only")
            attempt={"bindings":bindings,"solution":solution,"feedback":feedback,
                     "source_validation":source_check,"contract":contract}
            if known_revalidation is not None:
                attempt['source_binding_revalidation_certificate']=known_revalidation
            if radius_certificate and radius_certificate.get("status")=="accepted_seed":
                # The original graph/source identity is still compared below;
                # only independently remeasured evidence may change here.
                radius_certificate["input_identity_verified_against_checkpoint"]=(
                    _preflight_input_identity(image_path,document,baseline,graph)==certificate["input_identity"])
                attempt["source_binding_transition_certificate"]=radius_certificate
            if solution.get("accepted") is True:
                # Keep a native, fixed-nominal candidate even when unknown
                # geometry still fails the source gate. It is separately
                # inspectable and can seed another solve; it is never promoted
                # by an attribute certificate alone.
                candidate_dir=attempt_dir/"parametric-export"
                updated=export_parametric(image_path,baseline,
                    {**solution,"constraints":copy.deepcopy(bindings.get("constraints",[]))},candidate_dir)
                _record_published_source_validation(updated,source_check,baseline)
                updated['candidate_scope']={'kind':'native_attribute_verified_candidate',
                    'source_gate_passed':source_check.get('passed') is True,
                    'publication_approved':False,'reference_accuracy_verified':False}
                _write(candidate_dir/'model.json',updated)
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
            if attempt.get('model'):
                stage['final_binding_attempt']['native_attribute_candidate']={
                    'directory':str(attempt['export_directory']),
                    'scope':'Bound constraint subset verified by native DXF readback; publication checked separately.',
                    'source_gate_passed':source_check.get('passed') is True,
                    'publication_approved':guard['replace_checkpoint'],
                    'reference_accuracy_verified':False}
            _write(output/"binding-publication-guard.json",stage["final_binding_attempt"])
            if guard["replace_checkpoint"]:
                if attempt_graph is not graph:
                    _write(output/"topology.json",attempt_graph)
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
                stage["fillet_tangent_contract"]=_record_fillet_tangent_contract(
                    output,graph,revoked_bindings,revoked_solution,revoked_feedback)
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
            current=complete_published_subset(current,graph)
            return current,stage
        stage["provider"]={"status":"pending" if use_api else "disabled","network_requests":None if use_api else 0,
                           "http_success":None if use_api else False,"schema_success":False}
        checkpoint("constraint_binding","关联原图标注与候选图元；数值只取自标注，歧义保留为未绑定。")
        bindings=analyze_constraint_bindings(image_path,document,baseline,graph,output,provider=provider,use_api=use_api)
        _write(output/"binding-initial-attempt.json",{"provider":copy.deepcopy(bindings.get("provider",{}))})
        if any((graph.get("source_evidence") or {}).get(key) for key in
               ("source_line_initial_guess", "source_state_resume")):
            from .source_topology_initial_guess import retain_line_seed_bindings
            inventory=json.loads((output/"binding-candidates.json").read_text(encoding="utf8"))
            bindings,inventory,retention=retain_line_seed_bindings(image_path,document,baseline,
                graph,bindings,inventory,output,allow_partial_resume=True)
            stage["initial_line_seed_binding_retention"]=retention
        # A segmentation interval is an initial guess, not an immutable radius
        # attribute. Try a bounded, source-arrow-owned exact-radius solve, then
        # independently rebind the changed finite interval before using it as
        # the final solver seed. The original graph remains the ink comparator.
        source_validation_graph=graph
        graph,bindings,_=radius_seed_attempt(graph,bindings,output)
        stage["provider"]=bindings.get("provider",{})
        stage["binding_counts"]=bindings.get("counts",{})
        stage["constraints"]=bindings.get("constraints",[])
        stage["issues"]=bindings.get("issues",[])
        stage["radius_binding_coverage"]=bindings.get("radius_binding_coverage")
        checkpoint("parametric_solve","联合求解图元尺寸与连接关系，逐项核验已绑定约束。")
        seed_options=({"initial_seed_entities":graph["entities"]} if graph is not source_validation_graph else {})
        solution=_final_source_multistart(image_path,document,baseline,source_validation_graph,
            stage["constraints"],output,solver=solve_parametric,**seed_options)
        # The first R hypothesis started from an approximate fitted graph.
        # Once its known subset has a feasible solution, retry one unresolved
        # source-owned R from that solution without changing the original
        # optimization domain or any acceptance threshold.
        if (solution.get('accepted') is True and
                (stage.get('source_radius_hypotheses') or {}).get('status')=='no_admitted_hypothesis'):
            inventory=json.loads((output/'binding-candidates.json').read_text(encoding='utf8'))
            _write(output/'parametric-solution.json',solution)
            checkpoint('post_solve_radius_retry','已知约束已有有效解；以该解为初值，在原问题中重试一个未绑定半径并重新在线核验。')
            new_graph,new_bindings,new_solution,retry=_retry_source_radius_from_checkpoint(
                image_path,document,baseline,source_validation_graph,bindings,solution,inventory,
                output/'post-solve-radius-retry',provider=provider,use_api=use_api)
            stage['post_solve_radius_retry']=retry
            if retry['accepted']:
                _write(output/'pre-post-solve-constraint-bindings.json',bindings)
                _write(output/'pre-post-solve-parametric-solution.json',solution)
                _write(output/'pre-post-solve-topology.json',graph)
                graph,bindings,solution=new_graph,new_bindings,new_solution
                shutil.copyfile(bindings['inventory_artifact'],output/'binding-candidates.json')
                bindings['inventory_artifact']=str(output/'binding-candidates.json')
                _write(output/'topology.json',graph);_write(output/'constraint-bindings.json',bindings)
                stage.update(provider=bindings.get('provider',{}),binding_counts=bindings.get('counts',{}),
                    constraints=bindings['constraints'],issues=bindings.get('issues',[]),
                    radius_binding_coverage=bindings.get('radius_binding_coverage'))
        _write(output/"parametric-solution.json",solution)
        feedback=reconstruction_feedback(graph,bindings,solution)
        stage["fillet_tangent_contract"]=_record_fillet_tangent_contract(output,graph,bindings,solution,feedback)
        _write(output/"reconstruction-feedback.json",feedback)
        stage["reconstruction_feedback"]=feedback
        stage["solver"]={key:value for key,value in solution.items() if key not in {"entities","nodes","candidate_entities","candidate_nodes","constraints"}}
        stage["solver_constraint_checks"]=copy.deepcopy(solution.get("constraints",[]))
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
        source_check=_solved_source_validation(image_path,document,baseline,source_validation_graph,solution["entities"])
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
        _record_published_source_validation(updated,source_check,baseline)
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
        updated=complete_published_subset(updated,source_validation_graph)
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
