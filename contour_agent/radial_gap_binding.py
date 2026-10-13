"""Source-owned oblique radial thickness candidates, never fit-based dimensions.

The fitted graph proposes finite contact locations only. Both contacts need a
fresh original-pixel arrow check, unique finite-object ownership and a common
center proved independently by original radius-leader tails. The source OCR
nominal never selects contacts, owners or a common center.
"""
from __future__ import annotations

from copy import deepcopy
from itertools import product
import math

import numpy as np

from .dimension_policy import source_dimension_interval
from .measurement_anchors import source_curve_center_locations
from .oblique_dimension_evidence import oblique_dimension_witnesses, refine_oblique_arrow_contacts
from .topology import _primitive_distance


_MAX_ENTITIES = 256
_MAX_CONTACT_OPTIONS = 8
_MAX_CONTACT_PAIRS = 32
_RADIAL_ALIGNMENT = math.cos(math.radians(3.))


def _mapping(transform):
    points=np.asarray(transform([[0.,0.],[1.,0.],[0.,1.],[2.,3.]]),float)
    if points.shape!=(4,2) or not np.isfinite(points).all():
        raise ValueError("invalid_source_mapping")
    origin=points[0];matrix=np.column_stack([points[1]-origin,points[2]-origin])
    lengths=np.linalg.norm(matrix,axis=0)
    if (min(lengths)<=1e-12 or abs(float(np.linalg.det(matrix)))<=1e-20 or
            abs(float(matrix[:,0]@matrix[:,1]))>1e-6*lengths.prod() or
            abs(float(lengths[0]-lengths[1]))>1e-6*max(lengths) or
            not np.allclose(points[3],origin+matrix@np.asarray([2.,3.]),rtol=1e-9,atol=1e-6)):
        raise ValueError("source_mapping_is_not_a_rigid_uniform_scale")
    return origin,matrix,np.linalg.inv(matrix),float(lengths.mean())


def _finite_arc_intersections(entity,source_origin,direction,mapping):
    origin,matrix,inverse,scale=mapping
    a=inverse@(source_origin-origin);u=inverse@direction
    center=np.asarray(entity["center"],float);radius=float(entity["radius"])
    start,end=np.asarray(entity["start"],float),np.asarray(entity["end"],float)
    if (center.shape!=(2,) or start.shape!=(2,) or end.shape!=(2,) or
            not np.isfinite(np.r_[center,start,end,radius]).all() or radius<=0):
        return []
    first=math.atan2(*(start-center)[::-1]);last=math.atan2(*(end-center)[::-1])
    sign=-1. if entity.get("clockwise",False) else 1.
    sweep=(sign*(last-first))%(2*math.pi)
    if sweep<=1e-9:
        return []
    relative=a-center;aa=float(u@u);bb=2.*float(relative@u)
    cc=float(relative@relative-radius*radius);discriminant=bb*bb-4.*aa*cc
    if aa<=1e-30 or discriminant<0:
        return []
    result=[]
    for parameter in ((-bb-math.sqrt(discriminant))/(2.*aa),(-bb+math.sqrt(discriminant))/(2.*aa)):
        point=a+parameter*u;angle=math.atan2(*(point-center)[::-1])
        travel=(sign*(angle-first))%(2.*math.pi)
        if travel>sweep+1e-9 and travel<2.*math.pi-1e-9:
            continue
        source=origin+matrix@point
        if not any(np.linalg.norm(source-other)<1e-7 for other in result):
            result.append(source)
    return result


def _common_centers(locations,owners,records):
    result=[]
    for location in locations:
        anchor=location.get("anchor") or {};proof=location.get("evidence") or {}
        ids=[anchor.get("entity_id"),anchor.get("other_entity_id")]
        radius_records=proof.get("source_radius_record_ids") or []
        arrows=proof.get("source_arrow_ids") or []
        tails=proof.get("tail_support") or []
        if (anchor.get("type")!="common_curve_center" or set(ids)!=set(owners) or
                proof.get("verified") is not True or proof.get("center_semantics_verified") is not True or
                proof.get("nominal_used") is not False or proof.get("coordinates_locked") is not False or
                set(proof.get("required_center_equalities") or [])!={"x","y"} or
                set(proof.get("source_supported_entities") or [])!=set(owners) or
                len(radius_records)!=2 or len(set(radius_records))!=2 or
                len(arrows)!=2 or len(set(arrows))!=2 or len(tails)!=2 or
                any(tail.get("verified") is not True for tail in tails)):
            continue
        if any(identifier not in records or (records[identifier].get("parsed") or {}).get("kind")!="radius" or
               (records[identifier].get("source_text_admission") or {}).get("status")=="withheld"
               for identifier in radius_records):
            continue
        center=np.asarray(proof.get("observed_source_center_px"),float)
        if center.shape!=(2,) or not np.isfinite(center).all():
            continue
        if not any(np.linalg.norm(center-old[0])<=1. for old in result):
            result.append((center,proof))
    return result


def radial_gap_candidates(gray,records,graph,transform,band,radius_observations,*,diagnostics=None):
    """Propose source radial gaps only after all independent ownership checks.

    Returns a candidate list. Optional diagnostics is an append-only list of
    source observations/rejection reasons; unknown measurements remain visible.
    This never binds a radius, replaces an annotation by a fitted value or
    certifies publication/reference accuracy.
    """
    log=[] if diagnostics is None else diagnostics
    records=list(records);mapping_records={row.get("id"):row for row in records}
    entities=graph.get("entities") or []
    if (np.asarray(gray).ndim!=2 or np.asarray(gray).dtype!=np.uint8 or
            not math.isfinite(float(band)) or band<=0 or len(entities)>_MAX_ENTITIES or
            graph.get("units")!="mm"):
        log.append({"status":"radial_gap_input_or_units_unresolved","candidate_count":0})
        return []
    try:
        mapping=_mapping(transform)
    except (ValueError,TypeError,ArithmeticError,np.linalg.LinAlgError):
        log.append({"status":"radial_gap_source_mapping_unresolved","candidate_count":0})
        return []
    eligible=[row for row in records if (row.get("parsed") or {}).get("kind")=="length" and
        not (row.get("parsed") or {}).get("reference") and
        (row.get("source_text_admission") or {}).get("status")!="withheld"]
    witnesses=oblique_dimension_witnesses(gray,records)
    witness_records={row.get("record_id") for row in witnesses}
    for row in eligible:
        if row.get("id") not in witness_records:
            log.append({"record_id":row.get("id"),"status":"oblique_source_arrow_pair_unresolved","candidate_count":0})
    if not witnesses:
        return []
    centers=source_curve_center_locations(gray,graph,radius_observations,"x",records=records)
    origin,matrix,inverse,source_scale=mapping
    arcs=[entity for entity in entities if entity.get("type")=="ARC"]
    candidates=[]
    for witness in witnesses:
        record=mapping_records.get(witness.get("record_id"));proof=witness.get("evidence") or {}
        diagnostic={"record_id":witness.get("record_id"),"candidate_count":0,
            "ground_truth_used":False,"nominal_used_for_ownership":False}
        log.append(diagnostic)
        if not record or record not in eligible:
            diagnostic["status"]="source_record_unadmitted";continue
        try:
            nominal,_=source_dimension_interval(record["parsed"])
            tips=np.asarray(witness["endpoints_source_px"],float)
            direction=np.asarray(witness["direction_px"],float)
            arrows=proof["endpoint_arrows"]
            if (tips.shape!=(2,2) or direction.shape!=(2,) or not np.isfinite(np.r_[tips.ravel(),direction]).all() or
                    not .999<=np.linalg.norm(direction)<=1.001 or len(arrows)!=2 or
                    witness.get("arrow_pair_verified") is not True or proof.get("verified") is not True or
                    any(proof.get(key) is not True for key in ("same_source_shaft","arrow_directions_opposed","label_association_unique")) or
                    any(proof.get(key) is not False for key in ("nominal_used_to_rank","nominal_used_to_set_span","ground_truth_used"))):
                raise ValueError("unverified_source_arrow_pair")
            arrow_lengths=[float(arrow["length_px"]) for arrow in arrows]
            if not all(math.isfinite(value) and value>0 for value in arrow_lengths):
                raise ValueError("invalid_arrow_search_window")
            windows=[min(48.,max(7.,2.*value)) for value in arrow_lengths]
        except (ValueError,TypeError,KeyError,ArithmeticError):
            diagnostic["status"]="invalid_source_oblique_measurement";continue
        source_origin=tips.mean(axis=0);options=[[],[]]
        for entity in arcs:
            for contact in _finite_arc_intersections(entity,source_origin,direction,mapping):
                for index in (0,1):
                    if np.linalg.norm(contact-tips[index])<=windows[index]:
                        options[index].append({"entity_id":entity["id"],"contact_source_px":contact})
        diagnostic["finite_contact_proposal_counts"]=[len(rows) for rows in options]
        if not all(options):
            diagnostic["status"]="finite_arc_contact_not_represented";continue
        if max(map(len,options))>_MAX_CONTACT_OPTIONS or len(options[0])*len(options[1])>_MAX_CONTACT_PAIRS:
            diagnostic["status"]="finite_contact_search_budget_exceeded";continue
        verified=[];recheck_count=0;ambiguous_contacts=0
        for first,second in product(*options):
            if first["entity_id"]==second["entity_id"]:
                continue
            contacts=np.asarray([first["contact_source_px"],second["contact_source_px"]])
            recheck_count+=1
            refined=refine_oblique_arrow_contacts(gray,witness,contacts)
            if (refined is None or (refined.get("evidence") or {}).get("contact_pixel_tapers_verified") is not True or
                    (refined.get("evidence") or {}).get("requires_contact_tip_recheck") is not False):
                continue
            contacts=np.asarray(refined["endpoints_source_px"],float)
            if contacts.shape!=(2,2) or not np.isfinite(contacts).all():
                continue
            # A reversed enumeration is the same physical pair, not a second
            # admissible binding. Preserve the source arrow ordering only.
            if np.linalg.norm(contacts-tips[::-1],axis=1).sum()<np.linalg.norm(contacts-tips,axis=1).sum():
                contacts=contacts[::-1]
            actual_owners=[];owner_checks=[]
            for contact in contacts:
                native=inverse@(contact-origin);near=[]
                for entity in entities:
                    if entity.get("type") not in {"LINE","ARC"}:
                        continue
                    distance=float(_primitive_distance([native],entity)[0])*source_scale
                    if distance<=max(2.,float(band)):
                        near.append({"entity_id":entity["id"],"entity_type":entity["type"],"contact_gap_px":distance})
                owner_checks.append(near)
                if len(near)==1 and near[0]["entity_type"]=="ARC":
                    actual_owners.append(near[0]["entity_id"])
            if len(actual_owners)!=2 or len(set(actual_owners))!=2:
                ambiguous_contacts+=1;continue
            if any(np.linalg.norm(contacts-old[0])<=1. and actual_owners==old[1] for old in verified):
                continue
            verified.append((contacts,actual_owners,refined,owner_checks))
        diagnostic.update(contact_pixel_recheck_attempts=recheck_count,
            verified_finite_contact_pairs=len(verified),ambiguous_contact_pairs=ambiguous_contacts)
        if len(verified)!=1:
            diagnostic["status"]=("finite_arc_contact_owner_not_unique" if verified or ambiguous_contacts else
                                  "source_contact_taper_recheck_failed")
            continue
        contacts,owners,refined,owner_checks=verified[0]
        common=_common_centers(centers,owners,mapping_records)
        if len(common)!=1:
            diagnostic["status"]="independent_source_common_center_unresolved";continue
        center,center_proof=common[0]
        vectors=contacts-center;lengths=np.linalg.norm(vectors,axis=1)
        normal=np.asarray([-direction[1],direction[0]])
        center_gap=abs(float((center-source_origin)@normal))
        alignments=np.abs(vectors@direction)/np.maximum(lengths,1e-20)
        same_ray=float(vectors[0]@vectors[1])/max(float(lengths.prod()),1e-20)
        if (min(lengths)<=1. or center_gap>max(3.,float(band)) or
                min(alignments)<_RADIAL_ALIGNMENT or same_ray<_RADIAL_ALIGNMENT):
            diagnostic["status"]="source_dimension_shaft_not_a_common_radial_gap"
            diagnostic.update(center_to_shaft_gap_px=center_gap,radial_alignments=alignments.tolist(),same_ray_alignment=same_ray)
            continue
        evidence={"method":"source_oblique_contacts_and_independent_common_center_v1",
            "verified":True,"arrow_pair_verified":True,"contour_contacts_verified":True,
            "common_center_semantics_verified":True,"source_supported_entities":sorted(owners),
            "source_oblique_dimension":deepcopy(refined),
            "finite_contact_owners":deepcopy(owner_checks),"source_common_center":deepcopy(center_proof),
            "required_center_equalities":["x","y"],"center_to_shaft_gap_px":center_gap,
            "radial_alignments":alignments.tolist(),"same_radial_half_ray":True,
            "nominal_used_for_ownership":False,"fitted_radius_difference_used_as_dimension":False,
            "numeric_constraint_applied":False,"coordinates_locked":False,"ground_truth_used":False}
        candidates.append({"record_id":record["id"],"kind":"radial_gap","value":nominal,
            "entities":sorted(owners),"nodes":[],"measurement_policy":"source_tolerance_interval",
            "source_dimension_semantics":deepcopy(record["parsed"]),"local_reliable":True,
            "_score":0.,"evidence":evidence})
        diagnostic.update(status="source_radial_gap_candidate_available",candidate_count=1,finite_arc_owners=sorted(owners))
    return candidates
