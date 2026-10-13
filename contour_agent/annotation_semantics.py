"""Source-text obligation inventory; never estimates or applies CAD attributes.

Ratios retain their literal numerator/denominator until a finite source LINE
and its reference axis have been verified. Hole and process notes remain
visible as separate feature families, not exterior-profile distances.
"""
from __future__ import annotations
import math
import re
import unicodedata

from .ocr import canonical_records, parse_dimension


def inclination_ratio(text):
    normalized=unicodedata.normalize('NFKC',str(text)).upper().replace(' ','')
    match=re.fullmatch(r'(?:(?:INCLINE|INCLINATION|SLOPE|TAPER))?'
        r'(\d+(?:\.\d+)?)IN(\d+(?:\.\d+)?)',normalized)
    if not match:return None
    numerator,denominator=map(float,match.groups())
    if not all(math.isfinite(v) and 0<v<=1e6 for v in (numerator,denominator)):
        return None
    return {'kind':'inclination_ratio','numerator':numerator,'denominator':denominator,
        'reference_axis':None,'direction':None,'numeric_constraint_applied':False,
        'reference_axis_and_finite_line_required':True}


def annotation_feature(row):
    text=unicodedata.normalize('NFKC',str(row.get('text','')))
    compact=re.sub(r'\s+','',text).upper()
    parsed=row.get('parsed') or parse_dimension(text)
    ratio=inclination_ratio(text)
    if ratio:return {**ratio,'family':'inclination','requires_profile_resolution':True}
    if re.search(r'(?:ROUGH.?TURNED|WHENROUGH|MHENROUGH)',compact):
        return {'family':'manufacturing_stage','requires_profile_resolution':False,
                'note':'Retained process-state dimension; never silently substituted for final geometry.'}
    if re.fullmatch(r'\d+[-X×][⌀ØΦ∅]\d+(?:\.\d+)?HOLES?',compact):
        return {'family':'hole_feature','requires_profile_resolution':False,
                'note':'Hole count/diameter requires a separate feature contract, not an exterior distance.'}
    if re.search(r'(?:INCLINE|INCLINATION|SLOPE|TAPER|\dIN\d)',compact):
        return {'family':'unresolved_inclination','requires_profile_resolution':True}
    if parsed.get('kind')=='angle' and re.match(r'^R',compact):
        return {'family':'conflicting_angle_prefix','requires_profile_resolution':True}
    if re.search(r'RA.*R\d',compact):
        return {'family':'mixed_radius_and_surface_glyphs','requires_profile_resolution':True}
    if parsed.get('kind')=='unknown' and re.search(r'R\d',compact):
        return {'family':'unparsed_radius','requires_profile_resolution':True}
    if parsed.get('kind')=='unknown' and re.search(r'[⌀ØΦ∅]\d',compact):
        return {'family':'compound_diameter','requires_profile_resolution':True}
    return None


def source_annotation_inventory(document):
    rows=[]
    for row in canonical_records(document):
        feature=annotation_feature(row)
        if feature:rows.append({'record_id':row['id'],'source_text':row['text'],**feature})
    return {'schema_version':'source-annotation-families-v1','records':rows,
        'unresolved_profile_record_ids':[r['record_id'] for r in rows if r['requires_profile_resolution']],
        'non_exterior_feature_record_ids':[r['record_id'] for r in rows if not r['requires_profile_resolution']],
        'source_coverage_complete':False,'gt_geometry_used':False,
        'scope':'Text-family audit only. No inferred axis, binding, scope waiver or numeric CAD equation.'}


def ocr_review_priority(row):
    feature=annotation_feature(row)
    if feature:
        return 0 if feature['requires_profile_resolution'] else None
    kind=(row.get('parsed') or parse_dimension(row.get('text',''))).get('kind')
    if kind=='length':return 1
    if kind=='unknown' and re.search(r'\d',row.get('text','')):return 2
    return None
