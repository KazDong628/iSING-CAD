"""Exact annotation-driven corner seeds, checked against original image ink.

An oracle-mask corner may be sharp even when the drawing labels a round. This
constructor never fits a radius to that mask. Its result is a topology seed;
ordinary binding, joint solving, source gates and DXF readback remain required.
"""
from __future__ import annotations

import copy
import math

import numpy as np


def two_line_fillet_seed(selected, radius, tolerance, binding, validator):
    middle_arc=len(selected)==3 and [row.get('type') for row in selected]==['LINE','ARC','LINE']
    if ((not middle_arc and (len(selected) != 2 or any(row.get('type') != 'LINE' for row in selected)))
            or not binding or binding.get('arrowhead_verified') is not True
            or validator is None):
        raise ValueError('source_fillet_seed_requires_two_lines_and_verified_arrow')
    if not math.isfinite(radius) or radius <= 0 or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('invalid_source_fillet_seed_radius_or_tolerance')
    a, joint = np.asarray(selected[0]['start'],float), np.asarray(selected[0]['end'],float)
    other, b = np.asarray(selected[-1]['start'],float), np.asarray(selected[-1]['end'],float)
    target = np.asarray(binding.get('verified_arrow_tip_source_px',binding.get('target_source_px')),float)
    if any(p.shape != (2,) or not np.isfinite(p).all() for p in (a,joint,other,b,target)):
        raise ValueError('invalid_source_fillet_seed_points')
    if any(math.dist(first['end'],second['start'])>1e-6 for first,second in zip(selected,selected[1:])):
        raise ValueError('fillet_support_chain_not_connected')
    if middle_arc:
        prior=selected[1].get('radius_binding') or {}
        if prior:
            if (prior.get('record_id')!=binding.get('record_id') or not math.isclose(
                    float(selected[1].get('radius',0.)),radius,rel_tol=1e-8,abs_tol=1e-8)):
                raise ValueError('fillet_reinsertion_requires_same_bound_radius')
        elif binding.get('unique_unbound_arc_source_claim') is not True:
            raise ValueError('fillet_reinsertion_requires_unique_unbound_source_arc')
        u,v=joint-a,b-other
        try:
            intersections=np.linalg.solve(np.column_stack((u,v)),b-a)
        except np.linalg.LinAlgError:
            raise ValueError('source_fillet_seed_parallel_supports') from None
        corner=a+intersections[0]*u
        if (not np.isfinite(corner).all() or min(intersections)<=0):
            raise ValueError('source_fillet_seed_invalid_virtual_corner')
        u,v=corner-a,b-corner
        joint=corner
    else:u,v = joint-a,b-joint
    lengths = np.array([np.linalg.norm(u),np.linalg.norm(v)])
    if min(lengths) <= tolerance:
        raise ValueError('fillet_would_consume_line_support')
    u,v = u/lengths[0],v/lengths[1]
    turn = math.atan2(u[0]*v[1]-u[1]*v[0],float(u@v))
    if not math.radians(5.) <= abs(turn) <= math.radians(160.):
        raise ValueError('source_fillet_seed_invalid_corner_turn')
    distance = radius*math.tan(abs(turn)/2.)
    if min(lengths)-distance <= tolerance:
        raise ValueError('fillet_contact_outside_finite_line_support')
    first,last = joint-distance*u,joint+distance*v
    if middle_arc:
        middle_chord=math.dist(selected[1]['start'],selected[1]['end'])
        # Only redistribute this existing local corner, never extend a
        # support into unrelated source objects. Pixel mismatch is merely an
        # initial-state issue; original ink and fresh ownership still decide.
        if any(np.linalg.norm(contact-outer)>math.dist(old['start'],old['end'])+middle_chord
               for contact,outer,old in ((first,a,selected[0]),(last,b,selected[-1]))):
            raise ValueError('fillet_contact_outside_replaced_radius_arc_domain')
    center = first+np.array([-u[1],u[0]])*math.copysign(radius,turn)
    arc = dict(type='ARC',start=first.tolist(),end=last.tolist(),center=center.tolist(),
               radius=float(radius),clockwise=turn<0)
    from .topology import _primitive_distance
    # Use the existing topology inventory's source-arrow association window,
    # not the mask-fit tolerance. A real arrow can stop short of the ideal
    # nominal-R arc on a scanned drawing. This is no reference-accuracy test.
    arrow_gap=float(_primitive_distance(target[None,:],arc)[0])
    arrow_window=max(12.,tolerance)
    if arrow_gap > arrow_window:
        raise ValueError('source_fillet_seed_arrow_does_not_reach_arc')
    check=validator(arc)
    if not isinstance(check,dict) or check.get('passed') is not True or check.get('ground_truth_used') is not False:
        raise ValueError('source_fillet_seed_original_ink_not_supported')
    arc.update(radius_binding=copy.deepcopy(binding),radius_annotation_evidence=copy.deepcopy(binding),
        radius_binding_status='applied',fillet_construction='original_image_two_line_exact_radius_seed',
        fit_error_px=float(check.get('p90_edge_distance_px',0.)),
        source_refinement=dict(outer_endpoints_fixed=True,support_directions_fixed=True,support_types_fixed=True,
            radius_exact=True,source_path_and_arrow_verified=False,original_image_support_verified=True,
            original_image_check=copy.deepcopy(check),mask_path_used_as_numeric_evidence=False,
            arrow_association_gap_px=arrow_gap,arrow_association_window_px=arrow_window,
            arrow_association_is_reference_accuracy=False,
            original_finite_line_supports_px=[[selected[0]['start'],selected[0]['end']],
                                              [selected[-1]['start'],selected[-1]['end']]],
            replaced_middle_arc=copy.deepcopy(selected[1]) if middle_arc else None,
            constructed_contact_points_px=[first.tolist(),last.tolist()],
            requires_fresh_binding_and_joint_solve=True,ground_truth_used=False))
    lines=[]
    for old,start,end in ((selected[0],a,first),(selected[-1],last,b)):
        row=copy.deepcopy(old);row.update(start=start.tolist(),end=end.tolist())
        lines.append(row)
    return [lines[0],arc,lines[1]]


def original_ink_fillet_feasibility(gray, records, entity, nominal, leader, model, graph, transform, band):
    """Reobserve a source-first constructed fillet; never certify its mask path.

    Only an exact two-LINE construction with a fresh owned arrow is eligible.
    Existing local contact ink thresholds are reused. A successful admission
    still needs the ordinary joint solve, complete source check and native DXF
    checks, and says nothing about independent reference accuracy.
    """
    from .reconstruction_priority import mask_is_hard
    if mask_is_hard(model) or entity.get('fillet_construction')!='original_image_two_line_exact_radius_seed':
        return None
    result=dict(method='original_image_exact_two_line_fillet_admission_v1',entity_id=entity['id'],
        nominal=float(nominal),checked=True,passed=False,status='constructed_fillet_original_image_not_verified',
        ground_truth_used=False,nominal_used_to_rank=False,geometry_modified=False,
        observation='fresh_original_image_ink',mask_path_used_as_numeric_evidence=False,
        reference_accuracy_verified=False,endpoints_may_move_in_joint_solve=True)
    try:
        from .topology import _StrokeEvidence,_tangent
        from .vectorize import _sample_entities
        proof=entity.get('source_refinement') or {};binding=entity.get('radius_binding') or {}
        if (gray is None or records is None or proof.get('ground_truth_used') is not False or
                any(proof.get(k) is not True for k in ('radius_exact','outer_endpoints_fixed',
                    'support_types_fixed','support_directions_fixed','original_image_support_verified')) or
                binding.get('record_id')!=leader.get('record_id') or
                leader.get('arrowhead_verified') is not True or
                (leader.get('shaft_evidence') or {}).get('verified') is not True or
                (leader.get('source_arrow_ownership') or {}).get('status')!='globally_unique_source_claim' or
                not math.isclose(float(entity['radius']),float(nominal),rel_tol=0.,abs_tol=1e-8)):
            return result
        entities=graph['entities'];index=next(i for i,e in enumerate(entities) if e['id']==entity['id'])
        before,after=entities[(index-1)%len(entities)],entities[(index+1)%len(entities)]
        if (before['type']!='LINE' or after['type']!='LINE' or
                before['end_node']!=entity['start_node'] or after['start_node']!=entity['end_node'] or
                min(float(_tangent(before,True)@_tangent(entity,False)),
                    float(_tangent(entity,True)@_tangent(after,False)))<1-1e-10):return result
        center=transform([entity['center']])[0]
        radial=np.asarray(leader['arrowhead']['tip_px'],float)-center
        direction=np.asarray(leader['arrowhead']['direction_px'],float)
        alignment=abs(float(radial@direction/np.linalg.norm(radial)/np.linalg.norm(direction)))
        if alignment<.88:return result
        grid=float(graph['source_grid_pitch_px']);tolerance=float(graph.get('proposal_tolerance_px') or band)
        if grid<=0 or tolerance<=0:return result
        evidence=_StrokeEvidence(gray,records,grid);measurements=[]
        for row in (before,entity,after):
            points=transform(_sample_entities([row],max_step_px=.25)[0])
            check=evidence.summarize(points)
            measurements.append(check)
        passed=all(m['stroke_supported_fraction']>=.60 and
            m['p90_edge_distance_px']<=max(2*grid,tolerance+grid) for m in measurements)
        result.update(passed=passed,status='feasible_pending_joint_solve' if passed else result['status'],
            source_ink_checks=measurements,minimum_stroke_supported_fraction=.60,
            maximum_p90_edge_distance_px=max(2*grid,tolerance+grid),
            nominal_circle_arrow_alignment=alignment,minimum_arrow_alignment=.88,
            scope='Local source-design admission only; full source gate and native solve remain required.')
        return result
    except (KeyError,ValueError,TypeError,IndexError,ArithmeticError):return result
