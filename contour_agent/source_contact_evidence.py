"""Fresh source checks for exact ARC-LINE-ARC design-construction relations."""
import hashlib
import math
from pathlib import Path
import numpy as np


def contact_construction_relations(gray,records,image_path,model,graph,transform,relations,candidates):
    from .topology import _StrokeEvidence,_primitive_distance,_tangent
    from .topology_candidates import _base_source_entities,_affines
    from .topology_editing import _source_ring,_ring_path,_sample_entities,_source_polyline_distance
    if graph.get("source_sha256")!=hashlib.sha256(Path(image_path).read_bytes()).hexdigest():return relations
    entities=graph.get("entities") or [];result=list(relations)
    grid=float(graph.get("source_grid_pitch_px") or 0.);tol=float(graph.get("proposal_tolerance_px") or 0.)
    if len(entities)<3 or not 0<grid<=tol or _source_ring(model) is None:return result
    stroke=None;source_entities=None
    for i,line in enumerate(entities):
        proof=line.get("source_contact_construction") or {}
        if (line.get("type")!="LINE" or proof.get("method")!="source_nominal_exact_arc_line_arc_v1"
                or proof.get("ground_truth_used") is not False
                or any(proof.get(k) is not True for k in ("radius_exact","angle_exact","contacts_exact","outer_endpoints_fixed"))):continue
        before,after=entities[(i-1)%len(entities)],entities[(i+1)%len(entities)]
        if (before.get("type")!="ARC" or after.get("type")!="ARC"
                or before.get("end_node")!=line.get("start_node")
                or line.get("end_node")!=after.get("start_node")):continue
        try:
            radius_records=proof["radius_record_ids"];angle_record=proof["angle_record_id"]
            if len(radius_records)!=2 or len(set(radius_records))!=2:continue
            radius_candidates=[]
            for arc,rid in zip((before,after),radius_records):
                rows=[r for r in candidates if r.get("kind")=="radius" and r.get("record_id")==rid
                      and r.get("entities")==[arc["id"]] and r.get("local_reliable") is True
                      and ((r.get("evidence") or {}).get("leader") or {}).get("arrowhead_verified") is True
                      and ((r.get("evidence") or {}).get("whole_primitive_radius") or {}).get("passed") is True]
                if len(rows)!=1 or abs(float(arc["radius"])-float(rows[0]["value"]))>1e-8:break
                radius_candidates.append(rows[0])
            angles=[r for r in candidates if r.get("kind")=="angle" and r.get("record_id")==angle_record
                    and r.get("entities")==[line["id"]] and r.get("local_reliable") is True
                    and r.get("reference_axis") in {"horizontal","vertical"}]
            if len(radius_candidates)!=2 or len(angles)!=1:continue
            vector=np.asarray(line["end"])-line["start"]
            actual=math.degrees(math.atan2(abs(vector[0]),abs(vector[1])) if
                angles[0]["reference_axis"]=="vertical" else math.atan2(abs(vector[1]),abs(vector[0])))
            if abs(actual-float(angles[0]["value"]))>1e-7:continue
            contacts=np.asarray(proof["constructed_contacts_px"],float)
            outer=np.asarray(proof["original_outer_endpoints_px"],float)
            current=transform([line["start"],line["end"]]);far=transform([before["start"],after["end"]])
            if contacts.shape!=(2,2) or outer.shape!=(2,2) or not np.isfinite(contacts).all() or not np.isfinite(outer).all():continue
            if max(np.linalg.norm(current-contacts,axis=1).max(),np.linalg.norm(far-outer,axis=1).max())>1e-4:continue
            dots=[float(_tangent(before,True)@_tangent(line,False)),float(_tangent(line,True)@_tangent(after,False))]
            if min(dots)<1-1e-10:continue
            if stroke is None:stroke=_StrokeEvidence(gray,records,grid)
            ink=stroke.summarize(np.linspace(current[0],current[1],64))
            if ink["stroke_supported_fraction"]<.60 or ink["p90_edge_distance_px"]>max(2*grid,tol+grid):continue
            if source_entities is None:
                _,_,det=_affines(graph,model);source_entities=_base_source_entities(graph,transform,det)
            local=[source_entities[(i-1)%len(entities)],source_entities[i],source_entities[(i+1)%len(entities)]]
            sampled=_sample_entities(local,max_step_px=.5)[0]
            source,_=_ring_path(np.asarray(_source_ring(model),float),far[0],far[1],sampled)
            forward=float(np.min(np.asarray([_primitive_distance(source,e) for e in local]),axis=0).max())
            reverse=float(_source_polyline_distance(sampled,source).max())
            local_mask_passed=max(forward,reverse)<=max(1.,1.5*tol)
            from .reconstruction_priority import mask_is_hard
            if not local_mask_passed and mask_is_hard(model):continue
        except (KeyError,ValueError,TypeError,IndexError,ArithmeticError):continue
        for arc,node,ordered,dot in ((before,line["start_node"],[before["id"],line["id"]],dots[0]),
                                    (after,line["end_node"],[line["id"],after["id"]],dots[1])):
            existing=next((j for j,r in enumerate(result) if r.get("type")=="tangent"
                           and set(r.get("entities") or [])==set(ordered) and r.get("nodes")==[node]),None)
            if existing is not None and result[existing].get("local_reliable") is True:continue
            evidence=dict(method="source_bound_exact_arc_line_arc_tangent_v1",verified=True,
                evidence_class="source_bound_design_construction",independent_source_tangent_measurement=False,
                original_source_image_rechecked=True,ground_truth_used=False,shared_node=node,
                angle_record_id=angle_record,radius_record_ids=radius_records,
                source_line_stroke=ink,local_source_maximum_px=max(forward,reverse),
                local_mask_representation_passed=local_mask_passed,
                local_mask_threshold_px=max(1.,1.5*tol),
                mask_is_publication_requirement=mask_is_hard(model),
                directed_tangent_deviation_degrees=math.degrees(math.acos(float(np.clip(dot,-1.,1.)))))
            row=dict(id=result[existing]["id"] if existing is not None else f"contact-{line['id']}-{node}",
                type="tangent",entities=ordered,nodes=[node],source="source_bound_design_contact",
                local_reliable=True,required=False,evidence=evidence)
            if existing is None:result.append(row)
            else:result[existing]=row
    return result
