"""Source-only OCR/primitive binding candidates and independently checked choices.

This module never reads reference CAD. Candidate generation is deliberately wider
than the final pixel-fit gate: dimensions may correct a coarse geometry proposal.
Unbound and rejected source dimensions remain in the inventory/denominator.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
from itertools import product
import json
import math
import unicodedata
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .binding_provider import validate_selection
from .dimension_evidence import _axis_lines, _linear_witnesses
from .ocr import canonical_records, parse_dimension
from .structural_evidence import structural_evidence


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _box(row):
    try:
        points = np.asarray(row.get("box"), float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
            return None
        return points
    except (TypeError, ValueError):
        return None


def _source_text_evidence(gray, row):
    """Flag a narrow, source-observed Delta/leading-4 ambiguity, not new OCR.

    A hollow glyph that widens to a full bottom bar and has no descending
    stem cannot safely provide a leading numeric 4. Other glyphs are untested,
    not certified. This guard never changes the OCR value to a guessed symbol.
    """
    text=unicodedata.normalize("NFKC",str(row.get("text",""))).replace(" ","").lstrip("(")
    result={"method":"leading_four_closed_triangle_no_descender_v1","checked":False,
            "symbol_confusion":False,"text_confirmed":False}
    if not text.startswith("4") or row.get("parsed",{}).get("kind")!="length":return result
    box=_box(row)
    if box is None:return result
    lo=np.floor(box.min(axis=0)).astype(int);hi=np.ceil(box.max(axis=0)).astype(int)+1
    lo=np.maximum(lo,[0,0]);hi=np.minimum(hi,[gray.shape[1],gray.shape[0]])
    if np.any(hi-lo<8):return result
    crop=gray[lo[1]:hi[1],lo[0]:hi[0]]
    _,ink=cv2.threshold(crop,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    count,labels,stats,_=cv2.connectedComponentsWithStats(ink,8)
    major=[]
    for index,(x,y,width,height,area) in enumerate(stats[1:],1):
        if height<max(8,crop.shape[0]*.30) or width<3 or area<max(12,crop.size*.012):continue
        if width>crop.shape[1]*.9 and height<crop.shape[0]*.5:continue
        major.append((int(x),int(y),int(width),int(height),index))
    if not major:return result
    x,y,width,height,index=min(major,key=lambda item:item[0])
    mask=(labels[y:y+height,x:x+width]==index).astype(np.uint8)*255
    contours,hierarchy=cv2.findContours(mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    holes=[c for i,c in enumerate(contours) if hierarchy[0][i][3]>=0 and cv2.contourArea(c)>=width*height*.075]
    def row_width(fraction):
        row=mask[min(height-1,int(height*fraction))]
        occupied=np.flatnonzero(row)
        return float(occupied[-1]-occupied[0]+1) if len(occupied) else 0.
    upper=float(np.median([row_width(f) for f in (.15,.20,.25)]))
    lower=float(np.median([row_width(f) for f in (.80,.85,.90)]))
    bottom=float(np.median([row_width(f) for f in (.90,.93,.96)]))
    hole_vertices=0
    if len(holes)==1:
        hull=cv2.convexHull(holes[0]);hole_vertices=len(cv2.approxPolyDP(hull,.065*cv2.arcLength(hull,True),True))
    suspect=bool(len(holes)==1 and hole_vertices in (3,4) and lower>=.80*width and
                 bottom>=.72*width and upper<=.70*lower)
    result.update(checked=True,symbol_confusion=suspect,
                  reason="closed_triangle_without_digit_four_descender" if suspect else None,
                  first_glyph_box_px=[int(lo[0]+x),int(lo[1]+y),int(lo[0]+x+width),int(lo[1]+y+height)],
                  large_holes=len(holes),hole_hull_vertices=hole_vertices,
                  width_px=width,height_px=height,upper_width_px=upper,lower_width_px=lower,bottom_width_px=bottom,
                  requires_source_text_confirmation=suspect)
    return result


def _source_transform(model, graph):
    nodes = graph.get("nodes", [])
    if len(nodes) >= 3:
        design = np.asarray([[n["x"], n["y"], 1.] for n in nodes], float)
        target = np.asarray([n["source_px"] for n in nodes], float)
        if np.linalg.matrix_rank(design) == 3:
            transform = np.linalg.lstsq(design, target, rcond=None)[0]
            return lambda xy: np.c_[np.asarray(xy, float), np.ones(len(xy))] @ transform
    system = graph.get("coordinate_system") or model.get("coordinate_system") or {}
    origin = system.get("origin_source_px", [0., 0.])
    scale = model.get("scale", {}).get("pixels_per_mm") if graph.get("units") == "mm" else 1.
    scale = float(scale or 1.)
    return lambda xy: np.asarray(xy, float) * [scale, -scale] + origin


def _samples(entity, transform):
    start, end = np.asarray(entity["start"], float), np.asarray(entity["end"], float)
    if entity["type"] == "LINE":
        return transform(np.linspace(start, end, 16))
    center = np.asarray(entity["center"], float)
    a, b = [math.atan2(*(point-center)[::-1]) for point in (start, end)]
    sweep = -((a-b) % (2*math.pi)) if entity.get("clockwise") else ((b-a) % (2*math.pi))
    theta = np.linspace(a, a+sweep, 64)
    return transform(center + float(entity["radius"]) * np.c_[np.cos(theta), np.sin(theta)])


def _leaders(gray, records):
    """Detected source strokes, not model-claimed arrows. OCR boxes are removed."""
    ratio = min(1., 1600/max(gray.shape))
    small = cv2.resize(gray, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
    for row in records:
        box = _box(row)
        if box is not None:
            cv2.fillPoly(small, [np.round(box*ratio).astype(np.int32)], 255)
    edge = cv2.Canny(small, 70, 180)
    raw = cv2.HoughLinesP(edge, 1, np.pi/720, threshold=22, minLineLength=22, maxLineGap=6)
    if raw is None:
        return []
    lines = [np.asarray(item[0], float).reshape(2,2)/ratio for item in raw]
    return sorted(lines, key=lambda p: -float(np.linalg.norm(p[1]-p[0])))[:700]


def _label_ray_entry(point, away_from_target, low, high, maximum_gap):
    """The shaft must lead into this label, not merely pass beside its box."""
    enter,leave=0.,float(maximum_gap)
    for axis in range(2):
        velocity=float(away_from_target[axis])
        if abs(velocity)<1e-10:
            if point[axis]<low[axis] or point[axis]>high[axis]:return None
            continue
        a=(low[axis]-point[axis])/velocity;b=(high[axis]-point[axis])/velocity
        enter=max(enter,min(a,b));leave=min(leave,max(a,b))
        if leave<enter:return None
    return enter


def _arrowhead_evidence(gray, endpoint, direction, label_size, band, target_points=None):
    """Verify a directed filled-arrow taper from the original source pixels.

    A line or a single crossing is insufficient: the tip region must be
    narrower than two successive body sections, followed by a thinner shaft.
    No OCR nominal, predicted radius or reference geometry is used.
    """
    if gray is None:return None
    normal=np.array([-direction[1],direction[0]])
    best=None
    def width_at(point,extent):
        offsets=np.arange(-extent,extent+.5,1.)
        points=point+offsets[:,None]*normal
        x=np.rint(points[:,0]).astype(int);y=np.rint(points[:,1]).astype(int)
        valid=(x>=0)&(x<gray.shape[1])&(y>=0)&(y<gray.shape[0])
        ink=np.zeros(len(points),bool);ink[valid]=gray[y[valid],x[valid]]<170
        middle=len(ink)//2
        # Hough may describe one edge of the shaft instead of its centre.
        central=[i for i in range(max(0,middle-2),min(len(ink),middle+3)) if ink[i]]
        if not central:return 0.
        index=min(central,key=lambda i:abs(i-middle));left=right=index
        while left>0 and ink[left-1]:left-=1
        while right+1<len(ink) and ink[right+1]:right+=1
        if left==0 or right==len(ink)-1:return float(2*extent+1)
        return float(right-left+1)
    for length in sorted(set(max(7.,min(90.,label_size*f)) for f in (.10,.16,.24,.36,.52))):
        for shift in np.linspace(-.35*length,.65*length,7):
            tip=endpoint+direction*shift
            # Filter before ranking: a stronger taper elsewhere on a crossing
            # stroke must not hide a valid arrow at this contour target. This
            # uses the same tip-to-arc band checked by the caller afterwards.
            if target_points is not None and float(np.min(np.linalg.norm(target_points-tip,axis=1)))>max(10.,band*1.7):
                continue
            widths=[width_at(tip-direction*length*f,max(5.,.6*length)) for f in (.12,.42,.72,1.20,1.55)]
            narrow,middle,body,shaft,tail=widths
            shaft=max(shaft,tail,1.)
            if not (narrow>=1 and middle>=2 and body>=max(3.,1.65*shaft) and
                    middle>=.45*body and narrow<=.78*body and body<=.9*length):continue
            # A valid taper gets stronger towards its base; a terminal T or X
            # intersection usually widens at the tip and fails this condition.
            if middle>1.35*body:continue
            ahead=tip+np.asarray([.35,.60,.85])[:,None]*length*direction
            ahead_ink=[]
            for point in ahead:
                pixels=point+np.asarray([-1.,0.,1.])[:,None]*normal
                x=np.rint(pixels[:,0]).astype(int);y=np.rint(pixels[:,1]).astype(int)
                valid=(x>=0)&(x<gray.shape[1])&(y>=0)&(y<gray.shape[0])
                ahead_ink.append(bool(np.any(gray[y[valid],x[valid]]<170)))
            forward_ink=float(np.mean(ahead_ink))
            if forward_ink>1/3:continue
            score=body/max(shaft,1.) + body/max(narrow,1.)-abs(shift)/max(length,1.)
            item={"method":"directed_source_ink_taper_and_shaft","tip_px":tip.tolist(),
                  "direction_px":direction.tolist(),"length_px":length,"cross_section_widths_px":widths,
                  "forward_ink_fraction":forward_ink,"score":score,"verified":True}
            if best is None or score>best["score"]:best=item
    return best


def _leader_contour_visibility(label_point, target_point, contours, band):
    """Check the directed label-to-tip path against the current source contour.

    A hatch stroke can pass the local arrow taper test at the opposite material
    boundary. Such a stroke must not bind an arc beyond the first boundary it
    crosses. The existing coarse target band allows adjacent primitives at the
    same junction; it does not allow a separate intervening material boundary.
    """
    start, end = np.asarray(label_point, float), np.asarray(target_point, float)
    ray = end-start
    length = float(np.linalg.norm(ray))
    target_band = max(10., band*1.7)
    first = None
    if length > 1e-8:
        for contour in contours:
            points = np.asarray(contour, float)
            for a, b in zip(points[:-1], points[1:]):
                edge = b-a
                relative = a-start
                denominator = ray[0]*edge[1]-ray[1]*edge[0]
                if abs(denominator) > 1e-9:
                    t = (relative[0]*edge[1]-relative[1]*edge[0])/denominator
                    u = (relative[0]*ray[1]-relative[1]*ray[0])/denominator
                    if not (-1e-9 <= t <= 1.+1e-9 and -1e-9 <= u <= 1.+1e-9):
                        continue
                elif abs(relative[0]*ray[1]-relative[1]*ray[0]) <= 1e-7:
                    # A shaft following a boundary is also not a free path to a
                    # distant radius. Include its earliest overlapping point.
                    ta, tb = (float(np.dot(point-start, ray)/(length*length)) for point in (a,b))
                    if max(ta,tb) < 0. or min(ta,tb) > 1.:
                        continue
                    t = max(0., min(ta,tb))
                else:
                    continue
                t = float(np.clip(t,0.,1.))
                if first is None or t < first:
                    first = t
    remaining = None if first is None else (1.-first)*length
    return {"method":"first_source_contour_intersection",
            "label_exit_px":start.tolist(), "target_tip_px":end.tolist(),
            "first_intersection_px":None if first is None else (start+first*ray).tolist(),
            "first_intersection_to_target_px":remaining,
            "target_support_band_px":target_band,
            "verified":remaining is None or remaining <= target_band}


def _leader_evidence(box, arc, center, lines, band, gray=None, contours=None, rejections=None):
    lo, hi = box.min(axis=0), box.max(axis=0)
    size = max(12., float(np.linalg.norm(hi-lo)))
    best = None
    for index, segment in enumerate(lines):
        for label_end, target_end in (segment, segment[::-1]):
            label_gap = float(np.linalg.norm(np.maximum(lo-label_end, 0)+np.minimum(hi-label_end, 0)))
            if label_gap > max(18., size*.75):
                continue
            direction=target_end-label_end;length=float(np.linalg.norm(direction))
            if length<1e-8:continue
            unit=direction/length
            ray_gap=_label_ray_entry(label_end,-unit,lo-2.,hi+2.,max(18.,size*.75))
            if ray_gap is None:continue
            distances = np.linalg.norm(arc-target_end, axis=1)
            target_index = int(np.argmin(distances))
            tip_gap = float(distances[target_index])
            if tip_gap > max(10., band*1.7):
                continue
            radial = arc[target_index]-center
            denominator = float(np.linalg.norm(direction)*np.linalg.norm(radial))
            radial_alignment = abs(float(np.dot(direction,radial)/denominator)) if denominator else 0.
            if radial_alignment < .88:
                continue
            arrow=_arrowhead_evidence(gray,target_end,unit,size,band,arc)
            if arrow is None:continue
            arrow_gap=float(np.min(np.linalg.norm(arc-np.asarray(arrow["tip_px"]),axis=1)))
            if arrow_gap>max(10.,band*1.7):continue
            visibility = _leader_contour_visibility(label_end-unit*ray_gap,arrow["tip_px"],
                                                     contours if contours is not None else [arc],band)
            if not visibility["verified"]:
                if rejections is not None and len(rejections)<3:
                    rejections.append({"leader_id":f"line{index:03d}",
                                       "reason":"earlier_source_contour_intersection",**visibility})
                continue
            score = tip_gap + label_gap*.2 + (1-radial_alignment)*band*3
            item = {"method": "detected_source_leader_to_arc", "leader_id": f"line{index:03d}",
                    "segment_px": segment.tolist(), "label_gap_px": label_gap,
                    "arc_endpoint_gap_px": tip_gap, "radial_alignment": radial_alignment, "score": score,
                    "label_ray_intersection_gap_px":ray_gap,"arrowhead_verified":True,
                    "arrow_tip_to_arc_gap_px":arrow_gap,"arrowhead":arrow,
                    "contour_visibility":visibility}
            if best is None or score < best["score"]:
                best = item
    return best


def _annotation_leader_segments(graph, record_id):
    """Reuse upstream source observations, never its candidate ID or verdict.

    Candidate edits rotate/renumber entities. Re-test each observed segment
    against the current graph and original pixels before it can bind a radius.
    """
    result = []
    for row in graph.get("annotation_support", []):
        if row.get("record_id") != record_id:
            continue
        segment = (row.get("source_evidence") or {}).get("segment_px")
        try:
            points = np.asarray(segment, float)
            if points.shape == (2, 2) and np.isfinite(points).all():
                result.append(points)
        except (TypeError, ValueError):
            continue
    return result


def _annotation_issues(graph, records, candidates):
    """Describe unresolved semantics separately from a successful graph edit."""
    by_record = defaultdict(list)
    for candidate in candidates:
        by_record[candidate["record_id"]].append(candidate)
    entities = {e["id"]: e for e in graph.get("entities", [])}
    issues = []
    for record in records:
        if record.get("parsed", {}).get("kind") != "radius":
            continue
        record_id, nominal = record["id"], record["parsed"]["nominal"]
        proposals = by_record[record_id]
        reliable = [p for p in proposals if p.get("local_reliable")]
        supports = [s for s in graph.get("annotation_support", []) if s.get("record_id") == record_id]
        targets = []
        for support in supports:
            entity = entities.get(support.get("candidate_entity_id"))
            if entity is None:
                continue
            target = {"entity_id": entity["id"], "entity_type": entity["type"],
                      "association_verified": any(p["entities"] == [entity["id"]] for p in reliable)}
            if entity["type"] == "ARC":
                target.update(fitted_radius=entity["radius"],
                              nominal_difference=float(entity["radius"]) - float(nominal),
                              fixed_endpoints_radius_feasible=math.dist(entity["start"], entity["end"]) <= 2 * float(nominal) + 1e-8)
            targets.append(target)
        issues.append({"record_id": record_id, "kind": "radius", "nominal": nominal,
                       "status": "binding_candidate_requires_solve" if reliable else "unresolved_source_association",
                       "candidate_count": len(proposals), "upstream_target_hypotheses": targets,
                       "geometry_edit_is_not_numeric_binding": True,
                       "reason": ("radius_target_is_line_requires_topology_edit" if any(t["entity_type"] == "LINE" for t in targets)
                                  else "radius_requires_joint_or_topology_edit" if any(t.get("fixed_endpoints_radius_feasible") is False for t in targets)
                                  else None),
                       "dimensions_verified": False})
    return issues


def _constructed_radius_priors(graph, records):
    """Track exact local construction without accepting its binding assertions.

    A geometry executor can construct the requested nominal before the source
    arrow-to-entity association has passed this module's independent checks.
    Keep that history visible, but never promote metadata to a solver equation.
    """
    by_id = {row["id"]: row for row in records}
    result = []
    def finite_number(value):
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None
    for entity in graph.get("entities", []):
        construction = entity.get("radius_binding")
        if not isinstance(construction, dict):
            continue
        nominal = finite_number(construction.get("nominal"))
        radius = finite_number(entity.get("radius"))
        record_id = construction.get("record_id")
        record = by_id.get(record_id, {}) if isinstance(record_id, str) else {}
        parsed = record.get("parsed", {})
        matches_ocr = bool(parsed.get("kind") == "radius" and nominal is not None and nominal > 0
                           and parsed.get("nominal") == nominal)
        exact = bool(entity.get("type") == "ARC" and nominal is not None and radius is not None
                     and nominal > 0 and abs(radius - nominal) <= max(1e-7, nominal * 1e-9))
        result.append({"entity_id": entity["id"], "stable_id": entity.get("stable_id"),
                       "record_id": record_id if isinstance(record_id,str) else None,
                       "nominal": nominal, "current_radius": radius,
                       "construction_matches_source_ocr": matches_ocr,
                       "exact_constructed_radius_preserved": exact,
                       "status": "constructed_geometry_prior_unverified" if matches_ocr and exact
                                 else "construction_metadata_requires_review",
                       "source_binding_verified": False, "radius_constraint_pending_solve": False,
                       "dimensions_verified": False,
                       "upstream_dimension_bound_claim_ignored": entity.get("dimension_bound") is True,
                       "scope": "Construction history only. Fresh source evidence and an accepted numerical radius constraint are required."})
    return result


def _stroke_interval(line):
    # Connected-component thickness describes occupied pixel centres; the
    # extra half pixel represents their area, not an extra geometric tolerance.
    half=max(1.,float(line.get("thickness",1)))/2
    return float(line["cross"])-half,float(line["cross"])+half


def _interval_gap(value, interval):
    return max(interval[0]-float(value),float(value)-interval[1],0.)


def _extension_supports(lines, station, cross, node_cross, band):
    options = [line for line in lines if _interval_gap(station,_stroke_interval(line)) <= band
               and line["lo"]-band <= min(cross,node_cross) and line["hi"]+band >= max(cross,node_cross)]
    return [{**line,"ink_station_interval_px":list(_stroke_interval(line)),
             "intersection_gap_px":_interval_gap(station,_stroke_interval(line))}
            for line in sorted(options,key=lambda line:(_interval_gap(station,_stroke_interval(line)),abs(line["cross"]-station)))]


def _extension_support(lines, station, cross, node_cross, band):
    options=_extension_supports(lines,station,cross,node_cross,band)
    return options[0] if options else None


def _observed_station_groups(options, axis, band, graph):
    """Group by one measured, connected source extension, not nearby node IDs.

    Nodes outside that extension's covered span never enter a supported group.
    Prefer nodes actually inside its ink strip over coarse corner neighbours;
    if none coincide, retain the original coarse proposal band explicitly.
    Different detected source strokes remain separate alternatives.
    """
    grouped=defaultdict(list)
    unsupported=[]
    for option in options:
        support=option[3]
        if support is None:
            unsupported.append(option);continue
        interval=_stroke_interval(support)
        if _interval_gap(option[2]["source_px"][axis],interval)>band:
            unsupported.append(option);continue
        key=tuple(float(support[k]) for k in ("cross","lo","hi","thickness"))
        grouped[key].append(option)
    groups=[]
    for values in grouped.values():
        support=values[0][3];interval=_stroke_interval(support)
        direct=[value for value in values if _interval_gap(value[2]["source_px"][axis],interval)==0]
        members=direct or values
        by_id={v[2]["id"]:v for v in members}
        supporting=[];supported_ids=set()
        transform=_source_transform({},graph)
        for entity in graph.get("entities",[]):
            ends=[entity.get("start_node"),entity.get("end_node")]
            if not all(n in by_id for n in ends):continue
            sampled=_samples(entity,transform)
            if np.ptp(sampled[:,1-axis])<band*4:continue
            if max(_interval_gap(p[axis],interval) for p in sampled)>band:continue
            if min(sampled[:,1-axis])<support["lo"]-band or max(sampled[:,1-axis])>support["hi"]+band:continue
            supporting.append(entity["id"]);supported_ids.update(ends)
        if supported_ids:
            members=[v for v in members if v[2]["id"] in supported_ids]
        elif not direct:
            # A lone nearby curve intersection is not an observed station.
            unsupported.extend(members);continue
        # The closest projected source station is the reproducible anchor;
        # the full equivalence class remains evidence rather than hidden choice.
        representative=min(members,key=lambda v:(_interval_gap(v[2]["source_px"][axis],interval),
                                                 abs(v[2]["source_px"][axis]-support["cross"]),v[1],v[2]["id"]))
        evidence={"observed_station_px":support["cross"],"ink_station_interval_px":list(interval),
                  "member_nodes":[v[2]["id"] for v in members],
                  "member_source_px":[v[2]["source_px"] for v in members],
                  "excluded_coarse_neighbours":[v[2]["id"] for v in values if v not in members],
                  "representative_node":representative[2]["id"],
                  "representative_rule":"closest_projected_observed_stroke_then_nearest_dimension_line",
                  "support_kind":"source_ink_station" if direct else "coarse_projection_onto_observed_station",
                  "extension":support,"source_supported_entities":supporting,
                  "coordinate_equality_enforced":False}
        groups.append((representative,evidence))
    if groups:
        return groups
    # Keep unverified candidates reviewable, without letting them compete with
    # independently observed source stations or become hard dimensions.
    return [(value,{"member_nodes":[value[2]["id"]],"support_kind":"unsupported_projection"})
            for value in sorted(unsupported,key=lambda v:(v[0],v[1]))[:5]]


def _length_candidates(gray, records, graph, band, pixels_per_mm=None):
    result = []
    nodes = graph.get("nodes", [])
    perpendicular = {"x": _axis_lines(gray,"y"), "y": _axis_lines(gray,"x")}
    for witness in _linear_witnesses(gray, records, split_chains=True):
        axis = 0 if witness["axis"] == "x" else 1
        line = witness["line"]
        expected_span = witness["nominal"]*pixels_per_mm if pixels_per_mm else None
        span_tolerance = max(band*2,expected_span*.035) if expected_span else None
        compatible_span = bool(expected_span and abs((line["hi"]-line["lo"])-expected_span)<=span_tolerance)
        choices = []
        for end in ("lo", "hi"):
            options = []
            for node in nodes:
                point = node["source_px"]
                gap = abs(float(point[axis])-line[end])
                if gap > band*2:
                    continue
                supports = _extension_supports(perpendicular[witness["axis"]],line[end],line["cross"],point[1-axis],band)
                for support in supports or [None]:
                    options.append((gap, abs(point[1-axis]-line["cross"]), node, support))
            choices.append(_observed_station_groups(options,axis,band,graph))
        alternatives = []
        for (first,first_group), (second,second_group) in product(*choices):
            if first[2]["id"] == second[2]["id"]:
                continue
            ordered = sorted([first[2],second[2]],key=lambda n: float(n["x" if axis == 0 else "y"]))
            span = float(ordered[1]["x" if axis == 0 else "y"])-float(ordered[0]["x" if axis == 0 else "y"])
            if span <= 0:
                continue
            supported=first_group["support_kind"]!="unsupported_projection" and second_group["support_kind"]!="unsupported_projection"
            evidence = {"method": "source_dimension_line_and_observed_station_groups", "axis": witness["axis"],
                        "dimension_line": line, "line_endpoints_px": [[line["lo"],line["cross"]],[line["hi"],line["cross"]]] if axis==0 else [[line["cross"],line["lo"]],[line["cross"],line["hi"]]],
                        "endpoint_gaps_px": [first[0],second[0]], "extension_lines": [first[3],second[3]],
                        "observed_station_groups":[first_group,second_group],
                        "coarse_endpoint_band_px": band*2, "fitted_value": span,
                        "expected_dimension_span_px":expected_span,"source_span_tolerance_px":span_tolerance,
                        "source_span_scale_compatible":compatible_span,
                        "arrowhead_verified": False}
            alternatives.append({"record_id":witness["record_id"],"kind":"distance_x" if axis==0 else "distance_y",
                                 "entities":[],"nodes":[n["id"] for n in ordered],"value":witness["nominal"],
                                 "evidence":evidence,"local_reliable":bool(supported and compatible_span),
                                 "_score": first[0]+second[0]+.005*(first[1]+second[1])})
        alternatives.sort(key=lambda row: row["_score"])
        for candidate in alternatives[:3]:
            candidate["local_reliable"] = candidate["local_reliable"] and len(alternatives)==1
            candidate["evidence"]["alternative_station_pairs"] = len(alternatives)
            result.append(candidate)
    return result


def _draw_topology(image_path, graph, model, path):
    with Image.open(image_path) as loaded:
        source = loaded.convert("RGB")
    factor = min(1., 1600/max(source.size))
    source.thumbnail((1600,1600), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(source)
    font = ImageFont.load_default(size=13)
    transform = _source_transform(model, graph)
    for entity in graph.get("entities", []):
        points = _samples(entity,transform)*factor
        draw.line([tuple(p) for p in points], fill=(205,35,42), width=2)
        position = tuple(points[len(points)//2])
        text = entity["id"]
        bounds = draw.textbbox(position,text,font=font)
        draw.rectangle(bounds,fill="white")
        draw.text(position,text,fill=(180,0,10),font=font)
    for node in graph.get("nodes", []):
        x,y = np.asarray(node["source_px"])*factor
        draw.ellipse((x-3,y-3,x+3,y+3),fill=(0,74,180))
        draw.text((x+3,y+3),node["id"],fill=(0,50,155),font=font,stroke_width=1,stroke_fill="white")
    source.save(path)


def build_binding_candidates(image_path, document, model, graph, output_dir):
    image_path, out = Path(image_path), Path(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    gray = cv2.imdecode(np.fromfile(str(image_path),np.uint8),cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError("source image unreadable")
    all_records = canonical_records(document)
    records = [{"id":r["id"],"text":str(r.get("text","")),"parsed":r["parsed"],"box":r.get("box")} for r in all_records]
    text_evidence={row["id"]:_source_text_evidence(gray,row) for row in records}
    for row in records:row["source_text_evidence"]=text_evidence[row["id"]]
    eligible = [r for r in records if r["parsed"].get("kind") in {"radius","length","diameter","angle"}
                and r["parsed"].get("nominal") is not None and r["parsed"]["nominal"] > 0 and _box(r) is not None]
    band = max(4.,float(graph.get("proposal_tolerance_px",max(gray.shape)/500)))
    transform = _source_transform(model,graph)
    pixels_per_mm=(model.get("scale") or {}).get("pixels_per_mm")
    candidates = _length_candidates(gray,eligible,graph,band,pixels_per_mm)
    arcs = [(entity,_samples(entity,transform),transform([entity["center"]])[0])
            for entity in graph.get("entities",[]) if entity.get("type")=="ARC"]
    contours = [_samples(entity,transform) for entity in graph.get("entities",[])]
    leaders = _leaders(gray,eligible) if arcs else []
    for row in eligible:
        if row["parsed"]["kind"] != "radius":
            continue
        box = _box(row)
        options = []
        for entity,points,center in arcs:
            distance = float(np.min(np.linalg.norm(points-box.mean(axis=0),axis=1)))
            if distance > max(gray.shape)*.3:
                continue
            inherited = _annotation_leader_segments(graph, row["id"])
            occluded = []
            leader = _leader_evidence(box,points,center,leaders + inherited,band,gray,contours,occluded)
            evidence = {"method":"source_label_proximity", "label_to_arc_px":distance,
                        "coarse_proposal_band_px":band, "fitted_radius":entity["radius"],
                        "nominal_difference":float(entity["radius"])-row["parsed"]["nominal"],
                        "leader":leader,"nominal_used_to_rank":False,
                        "occluded_leader_hypotheses":occluded,
                        "upstream_leader_hypotheses_rechecked":len(inherited),
                        "evidence_chain":["source_ocr_box","observed_leader","verified_directed_arrow","current_primitive"],
                        "numeric_constraint_applied":False}
            score = leader["score"] if leader else 100000+distance
            options.append({"record_id":row["id"],"kind":"radius","entities":[entity["id"]],"nodes":[],
                            "value":row["parsed"]["nominal"],"evidence":evidence,"local_reliable":False,"_score":score})
        options.sort(key=lambda candidate:candidate["_score"])
        if options:
            # Keep rejected paths visible even when their arc falls outside the
            # three returned proximity candidates. A crossing can be a real but
            # ambiguous engineering leader; it requires review, not automatic
            # numeric admission.
            review_paths=[{"entity_id":option["entities"][0],**path}
                          for option in options for path in option["evidence"]["occluded_leader_hypotheses"]][:6]
            best = options[0]
            independent = best["evidence"]["leader"] is not None and (
                len(options)==1 or options[1]["_score"]-best["_score"] > max(5.,band*.5))
            best["local_reliable"] = independent
            for candidate in options[:3]:
                candidate["evidence"]["uniquely_supported_leader"] = bool(independent and candidate is best)
                candidate["evidence"]["alternative_arcs"] = len(options)
                candidate["evidence"]["ambiguous_crossing_leaders_requiring_review"] = review_paths
                candidates.append(candidate)
    # Stable ID order is independent of provider output. Keep the full inventory;
    # only the provider packet is capped. Never mistake truncation for uniqueness.
    candidates.sort(key=lambda row:(row["record_id"],row["_score"],row["kind"],row["nodes"],row["entities"]))
    for index,candidate in enumerate(candidates):
        candidate["id"] = f"c{index:03d}"
        candidate.pop("_score",None); candidate.pop("_stations",None)
        candidate["source"] = "source_image_geometry_candidate"
    for candidate in candidates:
        candidate["evidence"]["source_text"]=text_evidence[candidate["record_id"]]
        if text_evidence[candidate["record_id"]]["symbol_confusion"]:
            candidate["local_reliable"]=False
    reliable_counts=Counter(c["record_id"] for c in candidates if c["local_reliable"])
    for candidate in candidates:
        if reliable_counts[candidate["record_id"]]>1:
            candidate["local_reliable"]=False
            candidate["evidence"]["record_has_multiple_supported_bindings"]=True
    relations=structural_evidence(gray,records,graph,transform,_samples,band)
    # Prioritize supported records, then keep alternatives together in the packet.
    by_record=defaultdict(list)
    for candidate in candidates: by_record[candidate["record_id"]].append(candidate)
    selected_records=[]; selected_candidates=[]
    for row in sorted(eligible,key=lambda r:(not any(c["local_reliable"] for c in by_record[r["id"]]),not bool(by_record[r["id"]]),r["id"])):
        group=by_record[row["id"]]
        if len(selected_records)>=24 or len(selected_candidates)+len(group)>48:
            continue
        selected_records.append(row);selected_candidates.extend(group)
    topology=out/"binding-topology.png"
    _draw_topology(image_path,graph,model,topology)
    inventory={"version":"source-constraint-binding-v2","units":graph.get("units") or (graph.get("coordinate_system") or {}).get("units"),
               "records":selected_records,"candidates":selected_candidates,"relations":relations,
               "all_records":records,"all_candidates":candidates,"source_image_sha256":hashlib.sha256(image_path.read_bytes()).hexdigest(),
               "artifacts":{"topology":str(topology),"inventory":str(out/"binding-candidates.json")},
               "counts":{"ocr_records":len(records),"recognized_dimensions":len(eligible),"all_candidates":len(candidates),
                         "input_records":len(selected_records),"input_candidates":len(selected_candidates),
                         "structural_candidates":len(relations),
                         "source_verified_relations":sum(r["local_reliable"] for r in relations)},
               "annotation_diagnostics":_annotation_issues(graph,eligible,candidates),
               "constructed_radius_priors":_constructed_radius_priors(graph,records),
               "ground_truth_used":False,"issues":[],"proposal_tolerance_px":band}
    _write(out/"binding-candidates.json",inventory)
    return inventory


def _disabled(status="disabled",reason=None):
    return {"status":status,"reason":reason,"network_requests":0,"http_success":False,"schema_success":False,
            "image_sent":False,"ground_truth_sent":False,"bindings":[],"relations":[]}


def _constraint(candidate, source):
    return {"id":"k"+candidate["id"],"kind":candidate["kind"],"record_id":candidate["record_id"],
            "entities":candidate["entities"],"nodes":candidate["nodes"],"value":candidate["value"],"source":source,
            "candidate_id":candidate["id"]}


def analyze_constraint_bindings(image_path, document, model, graph, output_dir, *, provider=None, use_api=False):
    inventory=build_binding_candidates(image_path,document,model,graph,output_dir)
    receipt=_disabled()
    if use_api and (inventory["candidates"] or inventory["relations"]):
        if provider is None:
            receipt=_disabled("not_configured","binding_provider_missing")
        else:
            try:
                receipt=provider.select(image_path,inventory["artifacts"]["topology"],inventory)
                if receipt.get("schema_success"):
                    validate_selection(json.dumps({"bindings":receipt.get("bindings"),"relations":receipt.get("relations")},allow_nan=False))
            except InterruptedError:
                raise
            except Exception:
                # Unknown exceptions are deliberately not serialized: they may
                # include headers. Request count remains unknown, never fake zero.
                receipt={**_disabled("failed"),"error_code":"binding_provider_error","network_requests":None,"http_success":None}
    elif use_api:
        receipt=_disabled("skipped","no_source_binding_candidates")
    # Providers can send a smaller packet than the generic inventory cap (for
    # example a Responses vision budget). Only the recorded transmitted IDs
    # establish model abstention or authorize accepting a returned selection.
    # Legacy/custom providers without this receipt retain local source fallback
    # but cannot claim an API-confirmed binding or a model abstention.
    sent_sets=[]
    sent_inventory_verified=True
    for key, rows in (("input_record_ids",inventory["records"]),
                      ("input_candidate_ids",inventory["candidates"]),
                      ("input_relation_ids",inventory["relations"])):
        value=receipt.get(key)
        valid=(isinstance(value,list) and all(isinstance(item,str) for item in value)
               and len(value)==len(set(value)) and set(value)<={row["id"] for row in rows})
        sent_inventory_verified=sent_inventory_verified and valid
        sent_sets.append(set(value) if valid else set())
    sent_records,allowed,allowed_relations=sent_sets if sent_inventory_verified else (set(),set(),set())
    receipt={**receipt,"input_inventory_verified":sent_inventory_verified}
    candidates={c["id"]:c for c in inventory["all_candidates"]}
    records={r["id"]:r for r in inventory["all_records"]}
    graph_entities={e["id"]:e for e in graph.get("entities",[])}
    graph_nodes={n["id"]:n for n in graph.get("nodes",[])}
    selections=receipt.get("bindings",[]) if receipt.get("schema_success") else []
    repeats=Counter(row["record_id"] for row in selections)
    decisions=[];proposals=[]

    def check(candidate, record_id):
        if candidate is None:return "unknown_candidate_id"
        row=records.get(record_id)
        if row is None:return "unknown_record_id"
        if candidate["record_id"]!=record_id:return "record_candidate_mismatch"
        expected={"radius":"radius","distance_x":"length","distance_y":"length","distance":"length","angle":"angle"}.get(candidate["kind"])
        if expected!=row["parsed"]["kind"]:return "record_type_mismatch"
        if candidate["value"]!=row["parsed"].get("nominal"):return "source_nominal_mismatch"
        if inventory["units"]!="mm":return "units_unresolved"
        if any(e not in graph_entities for e in candidate["entities"]) or any(n not in graph_nodes for n in candidate["nodes"]):return "unknown_graph_id"
        if candidate["kind"]=="radius" and (len(candidate["entities"])!=1 or graph_entities[candidate["entities"][0]]["type"]!="ARC"):return "entity_type_mismatch"
        if candidate.get("evidence",{}).get("source_text",{}).get("symbol_confusion"):return "source_symbol_confusion_requires_confirmation"
        if not candidate["local_reliable"]:return "ambiguous_or_insufficient_independent_source_evidence"
        if candidate["kind"]=="radius" and not (candidate.get("evidence",{}).get("leader") or {}).get("arrowhead_verified"):
            return "source_arrowhead_not_verified"
        return None

    for selected in selections:
        candidate=candidates.get(selected["candidate_id"])
        error=check(candidate,selected["record_id"])
        observed=selected.get("observed_text")
        if error is None:
            if not isinstance(observed,str) or not observed.strip():error="source_observed_text_missing"
            else:
                parsed=parse_dimension(observed);original=records[selected["record_id"]]["parsed"]
                if parsed.get("kind")!=original.get("kind") or parsed.get("nominal")!=original.get("nominal"):
                    error="source_observed_text_mismatch"
        error=error or ("candidate_not_sent" if selected["candidate_id"] in candidates and selected["candidate_id"] not in allowed else None)
        error=error or ("record_not_sent" if selected["record_id"] not in sent_records else None)
        error=error or ("duplicate_record_selection" if repeats[selected["record_id"]]>1 else None)
        decision={**selected,"source":"ocr_api_binding","accepted":error is None,"reason":error}
        decisions.append(decision)
        if error is None:proposals.append((_constraint(candidate,"ocr_api_binding"),decision))
    selected_records={row["record_id"] for row in selections if row["record_id"] in sent_records}
    provider_abstained=sent_records-selected_records if use_api and receipt.get("schema_success") else set()
    for candidate in inventory["all_candidates"]:
        if candidate["record_id"] in selected_records:continue
        error=check(candidate,candidate["record_id"])
        if error is None and candidate["record_id"] in provider_abstained:error="provider_abstained_from_sent_record"
        decision={"record_id":candidate["record_id"],"candidate_id":candidate["id"],"source":"ocr_local_binding","accepted":error is None,"reason":error}
        decisions.append(decision)
        if error is None:proposals.append((_constraint(candidate,"ocr_local_binding"),decision))
    # Reject EVERY member of a contradictory group, not whichever comes later.
    groups=defaultdict(list)
    for constraint,decision in proposals:
        groups[(constraint["kind"],tuple(constraint["entities"]),tuple(constraint["nodes"]))].append((constraint,decision))
    constraints=[]
    for group in groups.values():
        values={float(constraint["value"]) for constraint,_ in group}
        if len(values)>1:
            for _,decision in group:decision.update(accepted=False,reason="conflicting_constraints")
        else:
            constraints.append(group[0][0])
            for _,decision in group[1:]:decision.update(accepted=False,reason="duplicate_equivalent_constraint")
    relations={r["id"]:r for r in inventory["relations"]}
    relation_selections=receipt.get("relations",[]) if receipt.get("schema_success") else []
    relation_counts=Counter(r["relation_id"] for r in relation_selections)
    selected_relation_ids=set(relation_counts)&allowed_relations
    relation_rows=[(selected,"api_and_source") for selected in relation_selections]
    for relation in inventory["relations"]:
        if relation["id"] not in selected_relation_ids:
            relation_rows.append(({"relation_id":relation["id"]},"local_source_fallback"))
    orientation=defaultdict(set)
    for selected,method in relation_rows:
        relation=relations.get(selected["relation_id"])
        if method=="api_and_source" and selected["relation_id"] not in allowed_relations:
            continue
        if relation and relation["type"] in {"horizontal","vertical"} and (method=="api_and_source" or relation.get("local_reliable")):
            for entity in relation["entities"]:orientation[entity].add(relation["type"])
    relation_proposals=[]
    for selected,method in relation_rows:
        relation=relations.get(selected["relation_id"])
        reason="unknown_relation_id" if relation is None else "duplicate_relation_selection" if relation_counts[selected["relation_id"]]>1 else None
        if reason is None:
            entity_ids=relation["entities"]
            if relation["type"] in {"horizontal","vertical"}:
                if len(entity_ids)!=1 or graph_entities.get(entity_ids[0],{}).get("type")!="LINE":
                    reason="relation_entity_type_mismatch"
                elif len(orientation[entity_ids[0]])>1:
                    reason="conflicting_relations"
            elif relation["type"]=="tangent" and (len(entity_ids)!=2 or len(set(entity_ids))!=2 or any(e not in graph_entities for e in entity_ids)):
                reason="relation_entity_type_mismatch"
            elif relation["type"]=="tangent":
                ends=[{graph_entities[e].get("start_node"),graph_entities[e].get("end_node")} for e in entity_ids]
                shared=(ends[0]&ends[1])-{None}
                if len(shared)!=1 or (relation["nodes"] and set(relation["nodes"])!=shared):
                    reason="tangent_requires_unique_shared_node"
        if reason is None and method=="api_and_source" and selected["relation_id"] not in allowed_relations:
            reason="relation_not_sent"
        if reason is None and not (relation.get("local_reliable") and (relation.get("evidence") or {}).get("verified")):
            reason="insufficient_independent_source_relation_evidence"
        if (reason is None and method=="local_source_fallback" and use_api and receipt.get("schema_success")
                and selected["relation_id"] in allowed_relations):
            reason="provider_abstained_from_sent_relation"
        decision={**selected,"source":"source_geometry","admission_method":method,
                  "accepted":reason is None,"reason":reason,"evidence":relation.get("evidence") if relation else None}
        decisions.append(decision)
        if reason is None:
            relation_proposals.append(({"id":"k"+relation["id"],"kind":relation["type"],"record_id":None,"entities":relation["entities"],
                                        "nodes":relation["nodes"],"value":None,"source":"source_geometry","required":False,
                                        "admission_method":method},decision))
    seen_relations=set()
    for constraint,decision in relation_proposals:
        signature=(constraint["kind"],tuple(sorted(constraint["entities"])),tuple(sorted(constraint["nodes"])))
        if signature in seen_relations:
            decision.update(accepted=False,reason="duplicate_equivalent_relation")
            continue
        seen_relations.add(signature);constraints.append(constraint)
    bound={c["record_id"] for c in constraints if c["record_id"] is not None}
    constructed_priors=inventory.get("constructed_radius_priors",[])
    for prior in constructed_priors:
        matched=next((c for c in constraints if c["kind"]=="radius" and
                      c["entities"]==[prior["entity_id"]] and c["record_id"]==prior["record_id"]
                      and c["value"]==prior["nominal"]),None)
        if matched is not None:
            prior.update(status="source_verified_constraint_pending_solve",source_binding_verified=True,
                         radius_constraint_pending_solve=True,constraint_id=matched["id"])
    rejected=[row for row in decisions if not row["accepted"]]
    result={"status":"completed","constraints":constraints,"bindings":decisions,"provider":receipt,
            "counts":{**inventory["counts"],"api_selected":len(selections),
                      "api_accepted":sum(row["accepted"] and row["source"]=="ocr_api_binding" for row in decisions),
                      "local_accepted":sum(row["accepted"] and row["source"]=="ocr_local_binding" for row in decisions),
                      "structural_accepted":sum(row["accepted"] and row["source"]=="source_geometry" for row in decisions),
                      "structural_local_accepted":sum(row["accepted"] and row.get("admission_method")=="local_source_fallback" for row in decisions),
                      "structural_api_accepted":sum(row["accepted"] and row.get("admission_method")=="api_and_source" for row in decisions),
                      "rejected":len(rejected),"bound_source_records":len(bound),
                      "unbound_dimensions":inventory["counts"]["recognized_dimensions"]-len(bound),"constraints":len(constraints)},
            "units":inventory["units"],"inventory_artifact":inventory["artifacts"]["inventory"],"topology_artifact":inventory["artifacts"]["topology"],
            "issues":sorted({row["reason"] for row in rejected}),"ground_truth_used":False,"dimensions_verified":False,
            "annotation_diagnostics":inventory.get("annotation_diagnostics",[]),
            "constructed_radius_priors":constructed_priors,
            "scope":"Only independently supported source bindings; unbound dimensions and global constraint completeness remain unverified."}
    _write(Path(output_dir)/"constraint-bindings.json",result)
    return result
