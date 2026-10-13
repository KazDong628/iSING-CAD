"""Source-defined section stations from an observed reference-offset dimension.

The offset defines a construction station, not an independent hard dimension
on the contour. Source pixels establish ownership and finite attachments only.
"""
from __future__ import annotations
import math
import numpy as np
from .dimension_evidence import _axis_lines
from .execution_control import execution_checkpoint
from .measurement_anchors import derived_anchor_point


def _verified(witness):
    proof=witness.get('binding_evidence') or {}
    return proof.get('verified') is True and not proof.get('ambiguous') and not witness.get('axis_association',{}).get('ambiguous')


def _covers(line,coordinate,tolerance=2.):
    return line['lo']-tolerance<=coordinate<=line['hi']+tolerance


def _pixel_hits(entity, transform, station_axis, coordinate):
    if entity['type']=='LINE':points=np.linspace(entity['start'],entity['end'],2)
    else:
        center=np.asarray(entity['center']);start=np.asarray(entity['start'])-center;end=np.asarray(entity['end'])-center
        a=math.atan2(start[1],start[0]);b=math.atan2(end[1],end[0]);sign=-1 if entity.get('clockwise') else 1
        sweep=(sign*(b-a))%(2*math.pi);angles=a+sign*np.linspace(0,sweep,129)
        points=center+entity['radius']*np.c_[np.cos(angles),np.sin(angles)]
    pixels=transform(points);result=[]
    for a,b in zip(pixels[:-1],pixels[1:]):
        denominator=b[station_axis]-a[station_axis]
        if abs(denominator)<1e-9:continue
        t=(coordinate-a[station_axis])/denominator
        if 0<=t<=1:
            point=a+t*(b-a)
            if not any(np.linalg.norm(point-old)<1. for old in result):result.append(point)
    return result


def source_section_locations(gray, records, graph, transform, band, axis, witnesses, virtual):
    """Couple two finite source extensions, then propose a symbolic section.

    Neither a nearby vertex nor a literal raster station substitutes for the
    section. The provider must still select the dimension/anchor association.
    """
    if axis not in {'x','y'}:return []
    measured=0 if axis=='x' else 1;station_axis=1-measured
    indexed={e['id']:e for e in graph['entities']};record_map={r['id']:r for r in records};result=[]
    station_strokes=_axis_lines(gray,axis)
    measurement_strokes=_axis_lines(gray,'x' if axis=='y' else 'y',minimum_span=max(5,int(3*band)))
    def observed_stroke(strokes,expected,a,b):
        matches=[s for s in strokes if abs(s['cross']-expected['cross'])<=2.
                 and _covers(s,a,2.) and _covers(s,b,2.)]
        return matches[0] if len(matches)==1 else None
    origin,bx,by=transform([[0.,0.],[1.,0.],[0.,1.]])
    basis=np.asarray([bx-origin,by-origin])
    if abs(basis[station_axis,measured])>1e-4*abs(basis[station_axis,station_axis]):return []
    for reference in witnesses:
        execution_checkpoint('section.reference')
        record=record_map.get(reference['record_id'],{});parsed=record.get('parsed') or {}
        if (reference.get('axis')==axis or parsed.get('kind')!='length' or not parsed.get('reference') or
                not _verified(reference)):continue
        ref=reference['line']
        for base_end,section_end in [('lo','hi'),('hi','lo')]:
            base_extension=(ref.get('extension_stations') or {}).get(base_end)
            section_extension=(ref.get('extension_stations') or {}).get(section_end)
            if not base_extension or not section_extension:continue
            origins=[v for v in virtual if v.get('anchor',{}).get('type')=='line_intersection' and
                abs(v['source_px'][station_axis]-ref[base_end])<=2*band and
                _covers(base_extension,v['source_px'][measured],band) and
                _covers(base_extension,ref['cross'],2.)]
            # Independently observed source alternatives remain ambiguous.
            if len(origins)!=1:continue
            virtual_origin=origins[0]
            actual_base=observed_stroke(station_strokes,base_extension,
                                       virtual_origin['source_px'][measured],ref['cross'])
            if actual_base is None:continue
            for witness in witnesses:
                if witness.get('axis')!=axis or not _verified(witness):continue
                if record_map.get(witness['record_id'],{}).get('parsed',{}).get('reference'):continue
                line=witness['line']
                for end,extension in (line.get('extension_stations') or {}).items():
                    if end not in {'lo','hi'}:continue
                    # A label-local ROI can clip an otherwise continuous
                    # extension. Remeasure the entire original finite stroke;
                    # never extrapolate the clipped line through a blank gap.
                    actual_section=observed_stroke(station_strokes,section_extension,line[end],ref['cross'])
                    if actual_section is None:continue
                    actual_measurement=observed_stroke(measurement_strokes,extension,
                                                       ref[section_end],line['cross'])
                    if actual_measurement is None:continue
                    extension=actual_measurement
                    if (abs(actual_section['cross']-ref[section_end])>2. or
                            not _covers(actual_section,line[end],2.) or
                            not _covers(actual_section,ref['cross'],2.) or
                            not _covers(extension,ref[section_end],2.) or
                            not _covers(extension,line['cross'],2.)):continue
                    # A continuing dimension extension may cross a station
                    # incidentally (e.g. an extremum's level). Only a finite
                    # termination at this station establishes section contact.
                    termination_gap=min(abs(extension['lo']-ref[section_end]),
                                        abs(extension['hi']-ref[section_end]))
                    if termination_gap>max(2*band,2*extension.get('thickness',1.)):continue
                    contact=np.zeros(2);contact[station_axis]=ref[section_end];contact[measured]=line[end]
                    candidates=[]
                    for entity in graph['entities']:
                        for point in _pixel_hits(entity,transform,station_axis,contact[station_axis]):
                            if abs(point[measured]-contact[measured])<=2*band:
                                candidates.append((entity,point))
                    if len(candidates)!=1:continue
                    entity,pixel=candidates[0]
                    direction=1 if (ref[section_end]-ref[base_end])*basis[station_axis,station_axis]>0 else -1
                    station=dict(type='source_offset_from_line_intersection',entities=virtual_origin['anchor']['entities'],
                        axis='x' if station_axis==0 else 'y',direction=direction,
                        source_record_id=reference['record_id'],source_offset=parsed['nominal'],reference=True)
                    # Pixel geometry chooses the finite root before any nominal
                    # station offset participates in the solver expression.
                    if entity['type']=='ARC':
                        center=transform([entity['center']])[0]
                        branch='positive' if (pixel[measured]-center[measured])*basis[measured,measured]>=0 else 'negative'
                    else:branch='linear'
                    anchor=dict(type='section_curve_intersection',entity_id=entity['id'],axis=axis,station=station,branch=branch)
                    try:point=derived_anchor_point(anchor,indexed,allow_infeasible=True)
                    except (ValueError,ArithmeticError,KeyError):continue
                    evidence=dict(method='source_reference_offset_finite_section_v1',verified=True,
                        source_section_semantics_verified=True,reference_record_id=reference['record_id'],
                        dimension_record_id=witness['record_id'],reference_dimension_line=ref,
                        section_extension=actual_section,measurement_extension=extension,
                        section_termination_gap_px=termination_gap,
                        source_attachment_px=contact.tolist(),nominal_used_for_station_ownership=False,
                        coordinates_locked=False,reference_offset_is_station_definition_not_independent_equation=True)
                    result.append(dict(id='section_'+entity['id']+'_'+reference['record_id']+'_'+witness['record_id']+'_'+end,
                        x=float(point[0]),y=float(point[1]),source_px=contact.tolist(),anchor=anchor,evidence=evidence))
    return result
