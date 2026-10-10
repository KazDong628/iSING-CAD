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
from .structural_evidence import structural_evidence, source_ink, _ink_trace


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


def _raw_boundary_tangent_relations(gray, records, model, graph, transform, band, relations):
    """Inspect source boundary joints without using fitted curve tangency as proof.

    The immutable extraction boundary locates two exterior paths. Its quadratic
    derivatives must agree at three scales before original ink is measured on
    those paths. A mask alone, a fitted arc or a model verdict cannot admit a
    tangent constraint. Existing source-verified relations stay unchanged.
    """
    raw = (model.get("extraction") or {}).get("raw_polyline_px")
    if raw is None:
        return relations
    try:
        ring = np.asarray(raw, float)
        if ring.ndim != 2 or ring.shape[1] != 2 or len(ring) < 12 or not np.isfinite(ring).all():
            return relations
        if np.linalg.norm(ring[-1] - ring[0]) >= 1e-7:
            return relations
        ring = ring[:-1]
        ring = ring[np.r_[True, np.linalg.norm(np.diff(ring, axis=0), axis=1) > 1e-7]]
        delta = np.roll(ring, -1, axis=0) - ring
        length = np.linalg.norm(delta, axis=1)
        if len(ring) < 12 or np.any(length <= 1e-7):
            return relations
        cumulative = np.r_[0., np.cumsum(length)]
        perimeter = float(cumulative[-1])
        grid = float(graph.get("source_grid_pitch_px", 1.))
        if not math.isfinite(grid) or grid <= 0:
            return relations
    except (ValueError, TypeError, IndexError):
        return relations
    closed = np.vstack([ring, ring[0]])
    incident = defaultdict(list)
    for entity in graph.get("entities", []):
        for node in {entity.get("start_node"), entity.get("end_node")} - {None}:
            incident[node].append(entity)
    result = list(relations)
    by_joint = {(frozenset(row["entities"]), tuple(row.get("nodes", []))): index
                for index, row in enumerate(result) if row["type"] == "tangent"}
    used_ids = {row["id"] for row in result}
    ink = None
    for node, adjacent in incident.items():
        if len(adjacent) != 2 or len({entity["id"] for entity in adjacent}) != 2:
            continue
        if not any(entity.get("type") == "ARC" for entity in adjacent):
            continue
        shared = ({adjacent[0].get("start_node"), adjacent[0].get("end_node")} &
                  {adjacent[1].get("start_node"), adjacent[1].get("end_node")}) - {None}
        if shared != {node}:
            continue
        key = frozenset(entity["id"] for entity in adjacent), (node,)
        index = by_joint.get(key)
        if index is not None and result[index].get("local_reliable"):
            continue
        samples = [_samples(entity, transform) for entity in adjacent]
        spans = [float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) for points in samples]
        span = min(96., min(spans) * .45, perimeter * .05)
        if span < max(24., 4 * band, 8 * grid):
            continue
        entity = adjacent[0]
        endpoint = entity["end"] if entity.get("end_node") == node else entity["start"]
        point = transform([endpoint])[0]
        fraction = np.clip(np.sum((point - ring) * delta, axis=1) / length ** 2, 0., 1.)
        projections = ring + fraction[:, None] * delta
        gaps = np.linalg.norm(projections - point, axis=1)
        segment = int(np.argmin(gaps))
        if float(gaps[segment]) > band:
            continue
        station = float(cumulative[segment] + fraction[segment] * length[segment])
        def path(direction, distances):
            positions = (station + direction * distances) % perimeter
            return np.c_[np.interp(positions, cumulative, closed[:, 0]),
                         np.interp(positions, cumulative, closed[:, 1])]
        directions, mask_sides = [], []
        for side in (-1., 1.):
            fits = []
            for ratio in (.5, .75, 1.):
                distances = np.linspace(0., span * ratio, 49)
                points = path(side, distances)
                parameter = distances / (span * ratio)
                coefficients = np.polynomial.polynomial.polyfit(parameter, points, 2)
                predicted = np.polynomial.polynomial.polyval(parameter, coefficients).T
                vector = coefficients[1]
                norm = float(np.linalg.norm(vector))
                if norm < 1e-8:
                    break
                fits.append((vector / norm, float(np.percentile(np.linalg.norm(points - predicted, axis=1), 90))))
            if len(fits) != 3:
                break
            disagreement = max(math.degrees(math.acos(float(np.clip(a[0] @ b[0], -1., 1.))))
                               for a, b in product(fits, fits))
            residual = max(item[1] for item in fits)
            mask_sides.append({"scale_disagreement_degrees": disagreement,
                               "fit_residual_p90_px": residual, "span_px": span})
            if disagreement > 2. or residual > max(1.25, grid * .75):
                break
            directions.append(fits[-1][0])
        if len(directions) != 2:
            continue
        mask_angle = math.degrees(math.acos(float(np.clip(-directions[0] @ directions[1], -1., 1.))))
        if mask_angle > 3.:
            continue
        if ink is None:
            _, ink = source_ink(gray, records)
        traces = [_ink_trace(ink, path(side, np.linspace(0., span / .45, 129)), band)
                  for side in (-1., 1.)]
        ink_angle = None
        verified = False
        if all(trace.get("verified") for trace in traces):
            ink_angle = math.degrees(math.acos(float(np.clip(-np.dot(
                traces[0]["tangent_direction_px"], traces[1]["tangent_direction_px"]), -1., 1.))))
            verified = ink_angle <= 3.
        evidence = {"method": "raw_mask_multiscale_and_two_sided_source_ink_tangent_v1",
                    "verified": verified, "shared_node": node, "sides": traces,
                    "mask_sides": mask_sides, "mask_observed_deviation_degrees": mask_angle,
                    "observed_deviation_degrees": ink_angle, "tolerance_degrees": 3.,
                    "boundary_observation": "immutable_extraction_raw_polyline_px",
                    "ground_truth_geometry_used": False,
                    "reason": None if verified else "source_tangent_evidence_insufficient"}
        if index is not None:
            item = dict(result[index])
            evidence["previous_geometry_seeded_evidence"] = item.get("evidence")
        else:
            number = len(result)
            while f"rel{number:03d}" in used_ids:
                number += 1
            item = {"id": f"rel{number:03d}", "type": "tangent",
                    "entities": [entity["id"] for entity in adjacent], "nodes": [node],
                    "source": "source_boundary_hypothesis", "required": False}
            used_ids.add(item["id"])
        item.update(local_reliable=verified, evidence=evidence)
        if index is None:
            result.append(item)
        else:
            result[index] = item
    return result


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
    return float(enter)


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
            score=float(body/max(shaft,1.) + body/max(narrow,1.)-abs(shift)/max(length,1.))
            item={"method":"directed_source_ink_taper_and_shaft","tip_px":tip.tolist(),
                  "direction_px":direction.tolist(),"length_px":length,"cross_section_widths_px":widths,
                  "forward_ink_fraction":forward_ink,"score":score,"verified":True}
            if best is None or score>best["score"]:best=item
    return best


def _leader_contour_visibility(label_point, target_point, contours, band):
    """Check the directed label-to-tip path against the current source contour.

    A hatch stroke can pass the local arrow taper test at the opposite material
    boundary. The first crossing remains an independent observation; callers
    may admit a farther arrow only with complete source shaft and label proof.
    The target band itself never hides a separate intervening boundary.
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
            from .source_arrow_localization import source_label_shaft_ownership
            ownership = source_label_shaft_ownership(gray, {"parsed": {"kind": "radius"}, "box": box.tolist()},
                                                     arrow["tip_px"], unit)
            if ownership["repetitive_label_crossing"]:
                if rejections is not None and len(rejections) < 3:
                    rejections.append({"leader_id": f"line{index:03d}", "reason": ownership["reason"],
                                       "source_label_association": ownership})
                continue
            visibility = _leader_contour_visibility(label_end-unit*ray_gap,arrow["tip_px"],
                                                     contours if contours is not None else [arc],band)
            full = None
            if not visibility["verified"]:
                from .source_arrow_localization import verify_source_hough_leader
                full = verify_source_hough_leader(gray,
                    {"id": "source-hough", "parsed": {"kind": "radius"}, "box": box.tolist()},
                    [label_end, target_end], arc, band, contours if contours is not None else [arc])
                if full is None:
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
                    "contour_visibility":visibility,"source_label_association":ownership}
            if full is not None:
                item.update({key: full[key] for key in ("method", "shaft_evidence", "crossing_source_contour",
                             "crossing_admission", "proposal_origin", "model_proposal_used")})
            if best is None or score < best["score"]:
                best = item
    return best


def _annotation_leader_segments(graph, record_id):
    """Reuse upstream source coordinates, never its candidate ID or verdict.

    Candidate edits rotate/renumber entities. Re-test each observed segment
    against the current graph and original pixels before it can bind a radius.
    """
    result = []
    seen = set()
    for row in [*(graph.get("annotation_support") or []),
                *(graph.get("radius_source_segment_hypotheses") or [])]:
        if row.get("record_id") != record_id:
            continue
        segment = (row.get("source_evidence") or {}).get("segment_px")
        try:
            points = np.asarray(segment, float)
            if points.shape == (2, 2) and np.isfinite(points).all():
                key = tuple(np.round(points.ravel(), 4))
                if key in seen:
                    continue
                seen.add(key)
                result.append(points)
        except (TypeError, ValueError):
            continue
    return result


def verify_source_arrow_proposal(gray, record, proposal, boundary_points, band, contours=None):
    """Locally verify a provider's source-pixel hypothesis, never its verdict.

    Radius leaders may cross material before reaching their real arrowhead.
    Crossing admission requires the entire shaft, label ray and a filled
    source arrow. Source Hough callers may use this identical proof; the
    proposal origin does not substitute for or weaken any pixel check.
    """
    if record.get("parsed", {}).get("kind") != "radius" or not isinstance(proposal, dict):
        return None
    if proposal.get("record_id", record.get("id")) != record.get("id"):
        return None
    box = _box(record)
    try:
        shaft = np.asarray(proposal.get("shaft_px"), float)
        tip = np.asarray(proposal.get("tip_px"), float)
        boundary = np.asarray(boundary_points, float)
    except (TypeError, ValueError, AttributeError):
        return None
    if (box is None or shaft.shape != (2,) or tip.shape != (2,) or
            not np.isfinite([shaft, tip]).all() or boundary.ndim != 2 or
            boundary.shape[1] != 2 or not len(boundary) or not np.isfinite(boundary).all()):
        return None
    if any(not (0 <= point[0] < gray.shape[1] and 0 <= point[1] < gray.shape[0]) for point in (shaft, tip)):
        return None
    lo, hi = box.min(axis=0), box.max(axis=0)
    size = max(12., float(np.linalg.norm(hi-lo)))
    delta = tip-shaft
    length = float(np.linalg.norm(delta))
    if length < 6.:
        return None
    direction = delta/length
    ray_gap = _label_ray_entry(shaft, -direction, lo-2., hi+2., max(18., size*.75))
    if ray_gap is None:
        return None
    arrow = _arrowhead_evidence(gray, tip, direction, size, band, boundary)
    if arrow is None:
        return None
    actual_tip = np.asarray(arrow["tip_px"], float)
    from .source_arrow_localization import source_label_shaft_ownership
    ownership = source_label_shaft_ownership(gray, record, actual_tip, direction)
    if ownership["repetitive_label_crossing"]:
        return None
    # A provider cannot point at one object and gain a stronger arrow far away.
    if math.dist(actual_tip, tip) > max(10., .55*float(arrow["length_px"])):
        return None
    target_gap = float(np.min(np.linalg.norm(boundary-actual_tip, axis=1)))
    if target_gap > max(10., band*1.7):
        return None
    label_exit = shaft-direction*ray_gap
    path_length = float(np.linalg.norm(actual_tip-label_exit))
    if path_length < 6.:
        return None
    points = np.linspace(label_exit, actual_tip, max(8, int(math.ceil(path_length))+1))
    normal = np.array([-direction[1], direction[0]])
    pixels = points[:, None, :]+np.arange(-2., 3.)[None, :, None]*normal
    x = np.rint(pixels[:, :, 0]).astype(int); y = np.rint(pixels[:, :, 1]).astype(int)
    valid = (x >= 0) & (x < gray.shape[1]) & (y >= 0) & (y < gray.shape[0])
    ink = np.zeros(x.shape, bool); ink[valid] = gray[y[valid], x[valid]] < 170
    support = np.any(ink, axis=1)
    longest = running = 0
    for present in support:
        running = 0 if present else running+1
        longest = max(longest, running)
    maximum_gap = max(5., min(12., size*.10))
    shaft_evidence = {"method": "full_source_shaft_pixel_support", "sample_count": len(points),
                      "supported_fraction": float(np.mean(support)), "maximum_unobserved_run_px": longest,
                      "maximum_allowed_gap_px": maximum_gap,
                      "verified": bool(np.mean(support) >= .88 and longest <= maximum_gap)}
    if not shaft_evidence["verified"]:
        return None
    visibility = _leader_contour_visibility(label_exit, actual_tip,
                                           contours if contours is not None else [boundary], band)
    return {"method": "agent_proposed_source_arrow_locally_verified",
            "segment_px": [shaft.tolist(), tip.tolist()], "label_gap_px": ray_gap,
            "label_ray_intersection_gap_px": ray_gap, "arrowhead_verified": True,
            "arrowhead": arrow, "target_source_px": actual_tip.tolist(),
            "arrow_tip_to_arc_gap_px": target_gap, "arc_endpoint_gap_px": target_gap,
            "contour_visibility": visibility, "shaft_evidence": shaft_evidence,
            "source_label_association": ownership,
            "crossing_source_contour": not visibility["verified"],
            "crossing_admission": "explicit_directed_proposal_with_full_source_shaft_and_arrow",
            "score": target_gap+.2*ray_gap, "nominal_used_to_rank": False}


def _radius_source_observations(gray, records, graph, transform, leaders, band, rejections=None, ownership_audit=None):
    """Observe directed arrows before deciding whether their target is an ARC.

    A LINE at a radius arrow is a topology defect, not an absent dimension.
    Neither upstream entity IDs nor upstream arrow verdicts are accepted here;
    only their source segment coordinates are reused and checked in the image.
    """
    paths = []
    for entity in graph.get("entities", []):
        coarse = _samples(entity, transform)
        # Include interiors of long LINEs when checking an arrow tip. Existing
        # arc samples are already in source coordinates; this is only sampling,
        # not a geometry repair or a nominal-radius fit.
        dense = [np.linspace(a, b, max(2, int(math.ceil(np.linalg.norm(b-a)/2.))+1))
                 for a, b in zip(coarse[:-1], coarse[1:])]
        if dense:
            paths.append((entity, np.vstack(dense)))
    if not paths:
        return []
    boundary = np.vstack([points for _, points in paths])
    contours = [points for _, points in paths]
    target_band = max(10., band*1.7)
    observations = []
    for row in records:
        if row.get("parsed", {}).get("kind") != "radius":
            continue
        box = _box(row)
        if box is None:
            continue
        lo, hi = box.min(axis=0), box.max(axis=0)
        size = max(12., float(np.linalg.norm(hi-lo)))
        # Hough segments often end on the arrow body, not at its tapered tip.
        # This is only a conservative search window: _arrowhead_evidence can
        # advance by at most .65 of its largest inspected arrow length. The
        # actual tip must still pass the original-pixel verifier and the
        # current-contour target_band below.
        prelocalization_band = target_band + .65*max(7., min(90., size*.52))
        found, seen = [], set()
        from .source_arrow_localization import native_radius_leader_segments, verify_source_hough_leader
        global_segments = leaders + _annotation_leader_segments(graph, row["id"])
        native_segments = native_radius_leader_segments(gray, row, records)
        for segment_index, segment in enumerate(global_segments + native_segments):
            native_segment = segment_index >= len(global_segments)
            key = tuple(np.round(np.asarray(segment).ravel(), 4))
            if key in seen:
                continue
            seen.add(key)
            for label_end, target_end in (segment, segment[::-1]):
                gap = float(np.linalg.norm(np.maximum(lo-label_end, 0)+np.minimum(hi-label_end, 0)))
                if gap > max(18., size*.75):
                    continue
                delta = target_end-label_end
                length = float(np.linalg.norm(delta))
                if length < 1e-8:
                    continue
                direction = delta/length
                ray_gap = _label_ray_entry(label_end, -direction, lo-2., hi+2., max(18., size*.75))
                if ray_gap is None:
                    continue
                distances = np.linalg.norm(boundary-target_end, axis=1)
                if float(distances.min()) > prelocalization_band:
                    continue
                local_boundary = boundary[distances <= max(120., size)]
                arrow = _arrowhead_evidence(gray, target_end, direction, size, band, local_boundary)
                if arrow is None:
                    continue
                tip = np.asarray(arrow["tip_px"], float)
                from .source_arrow_localization import source_label_shaft_ownership
                ownership = source_label_shaft_ownership(gray, row, tip, direction)
                if ownership["repetitive_label_crossing"]:
                    if rejections is not None and len(rejections) < 128:
                        rejections.append({"record_id": row["id"], "reason": ownership["reason"],
                                           "tip_px": tip.tolist(), "segment_px": np.asarray(segment).tolist(),
                                           "source_label_association": ownership})
                    continue
                visibility = _leader_contour_visibility(label_end-direction*ray_gap, tip, contours, band)
                # The first contour intersection and a local taper do not
                # establish that the complete shaft belongs to this text.
                # Every detection channel uses the same original-pixel proof.
                full = verify_source_hough_leader(gray, row, [label_end, target_end], local_boundary,
                                                 band, contours, verifier=verify_source_arrow_proposal)
                if full is None:
                    continue
                arrow = full["arrowhead"]
                tip = np.asarray(arrow["tip_px"], float)
                visibility = full["contour_visibility"]
                targets = [{"entity_id": entity["id"], "entity_type": entity["type"],
                            "tip_gap_px": float(np.min(np.linalg.norm(points-tip, axis=1)))}
                           for entity, points in paths]
                targets = sorted((target for target in targets if target["tip_gap_px"] <= target_band),
                                 key=lambda target: (target["tip_gap_px"], target["entity_id"]))
                if not targets:
                    continue
                item = {"record_id": row["id"], "nominal": row["parsed"].get("nominal"),
                        "method": "source_arrow_rechecked_independent_of_primitive_type",
                        "segment_px": np.asarray(segment).tolist(), "arrowhead": arrow,
                        "arrowhead_verified": True, "contour_visibility": visibility,
                        "target_candidates": targets, "nominal_used_to_rank": False,
                        "source_label_association": ownership}
                if full is not None:
                    item.update({key: full[key] for key in ("method", "label_gap_px", "label_ray_intersection_gap_px",
                                 "shaft_evidence", "crossing_source_contour",
                                 "crossing_admission", "proposal_origin", "model_proposal_used")})
                    item["detection_resolution"] = "native_local" if native_segment else "global_scaled"
                # Physical-arrow deduplication is global and checks direction
                # and shaft alignment as well as tip proximity. A local tip-
                # only merge can erase neighboring genuine parallel arrows
                # before the ownership audit sees the competing evidence.
                found.append(item)
        for proposal in (row.get("source_arrow_proposals") or [])[:2]:
            from .source_arrow_localization import localize_source_arrow_proposal
            verified = localize_source_arrow_proposal(gray, row, proposal, boundary, band, contours,
                                                      verifier=verify_source_arrow_proposal)
            if verified is None:
                continue
            tip = np.asarray(verified["arrowhead"]["tip_px"], float)
            targets = [{"entity_id": entity["id"], "entity_type": entity["type"],
                        "tip_gap_px": float(np.min(np.linalg.norm(points-tip, axis=1)))}
                       for entity, points in paths]
            targets = sorted((target for target in targets if target["tip_gap_px"] <= target_band),
                             key=lambda target: (target["tip_gap_px"], target["entity_id"]))
            if not targets:
                continue
            item = {**verified, "record_id": row["id"], "nominal": row["parsed"].get("nominal"),
                    "target_candidates": targets}
            # A provider proposal is another independently verified source
            # observation, never a reason to overwrite a nearby detected tip.
            found.append(item)
        if found:
            from .source_arrow_localization import (recover_source_attached_radius_arrow,
                                                     same_original_ink_arrow_shaft,
                                                     source_arrow_label_attachment)
            attachments = [(item, source_arrow_label_attachment(gray, row, item)) for item in found]
            def close_source_intersection(item):
                gap = (item.get("contour_visibility") or {}).get("first_intersection_to_target_px")
                return gap is not None and gap <= 1.
            # A complete but wrong neighboring extension can obscure the
            # OCR-attached arrow. Exhaust a bounded original-ink search only
            # when no existing observation has both glyph attachment and an
            # actual first contour intersection near its tip.
            if not any(link.get("strong_text_adjacency") and close_source_intersection(item)
                       for item, link in attachments):
                seed, _ = min(attachments, key=lambda pair: (
                    pair[1].get("normalized_text_to_shaft", math.inf),
                    (pair[0].get("arrowhead") or {}).get("score", math.inf)))
                recovered = recover_source_attached_radius_arrow(
                    gray, row, seed, boundary, band, contours, verifier=verify_source_arrow_proposal)
                if recovered is not None:
                    tip = np.asarray(recovered["arrowhead"]["tip_px"], float)
                    targets = [{"entity_id": entity["id"], "entity_type": entity["type"],
                                "tip_gap_px": float(np.min(np.linalg.norm(points-tip, axis=1)))}
                               for entity, points in paths]
                    targets = sorted((target for target in targets if target["tip_gap_px"] <= target_band),
                                     key=lambda target: (target["tip_gap_px"], target["entity_id"]))
                    if targets:
                        recovered = {**recovered, "record_id": row["id"],
                                     "nominal": row["parsed"].get("nominal"),
                                     "target_candidates": targets}
                        chosen_targets = {target["entity_id"] for target in targets}
                        retained = []
                        for item, link in attachments:
                            old_tip = np.asarray(item["arrowhead"]["tip_px"], float)
                            old_targets = {target["entity_id"] for target in item.get("target_candidates") or []}
                            old_first = (item.get("contour_visibility") or {}).get("first_intersection_to_target_px")
                            same_ink_shaft = (link.get("strong_text_adjacency") and
                                              same_original_ink_arrow_shaft(gray, item, recovered))
                            dominated = (math.dist(old_tip, tip) <= 8. and bool(old_targets & chosen_targets)
                                         and (not link.get("strong_text_adjacency") or same_ink_shaft)
                                         and (old_first is None or old_first > 2.))
                            if dominated:
                                if rejections is not None and len(rejections) < 128:
                                    rejections.append({"record_id": row["id"],
                                        "reason": "inferior_near_tip_source_arrow_hypothesis",
                                        "tip_px": old_tip.tolist(),
                                        "source_text_shaft_attachment": link,
                                        "first_intersection_to_target_px": old_first,
                                        "same_original_ink_shaft_verified": bool(same_ink_shaft),
                                        "replacement_has_original_ink_and_text_support": True})
                            else:
                                retained.append(item)
                        found = retained+[recovered]
        observations.extend(found)
    from .source_arrow_localization import resolve_source_arrow_ownership
    observations, ownership_rejections, audit = resolve_source_arrow_ownership(
        gray, records, observations, band=band)
    if rejections is not None:
        rejections.extend(ownership_rejections)
    if ownership_audit is not None:
        ownership_audit.update(audit)
    return observations


def _radius_text_audit(record):
    """Keep plausible source R labels even when numeric parsing is unresolved.

    This detects a radius marker, not a guessed number or a verified arrow.
    Surface finish Ra and ordinary alphabetic notes are separate text kinds.
    """
    parsed = record.get("parsed") or {}
    text = "".join(unicodedata.normalize("NFKC", str(record.get("text", ""))).split())
    marker_text = text.lstrip("(").rstrip(")")
    suffix = marker_text[1:] if marker_text[:1].lower() == "r" else None
    marker = bool(suffix is not None and not marker_text.lower().startswith("ra") and
                  (not suffix or not suffix[0].isalpha() or
                   suffix.lower() in {"o", "i", "l", "nan", "inf", "infinity"}))
    if not marker and parsed.get("kind") != "radius":
        return None
    source = parse_dimension(record.get("text", ""))
    def positive(value):
        return (isinstance(value, (int, float)) and not isinstance(value, bool) and
                math.isfinite(value) and value > 0)
    if source.get("kind") != "radius" or not positive(source.get("nominal")):
        issue = "source_radius_text_invalid_or_nonpositive"
    elif (parsed.get("kind") != "radius" or not positive(parsed.get("nominal")) or
          parsed.get("nominal") != source["nominal"]):
        issue = "source_radius_parsed_metadata_mismatch"
    else:
        issue = None
    return {"valid": issue is None, "issue": issue,
            "nominal": source["nominal"] if issue is None else None}


def radius_binding_coverage(inventory, graph, constraints=(), decisions=()):
    """Audit the complete radius denominator; missing detection is not absence.

    This certifies association coverage only. Exact numeric satisfaction must
    still be checked after solving and again after DXF export/readback.
    """
    text_audits = {row["id"]: _radius_text_audit(row) for row in inventory.get("all_records", [])}
    records = [row for row in inventory.get("all_records", []) if text_audits[row["id"]] is not None]
    entities = {row["id"]: row for row in graph.get("entities", [])}
    candidates = defaultdict(list)
    observations = defaultdict(list)
    ownership_rejections = defaultdict(list)
    for row in inventory.get("all_candidates", []):
        if row.get("kind") == "radius":
            candidates[row["record_id"]].append(row)
    for row in inventory.get("radius_source_observations", []):
        if row.get("arrowhead_verified"):
            observations[row["record_id"]].append(row)
    for row in inventory.get("radius_source_rejections", []):
        ownership_rejections[row.get("record_id")].append(row)
    required, bound, unresolved, ambiguous, unknown, confirmed = [], [], [], [], [], []
    unresolved_text = []
    for record in records:
        record_id = record["id"]
        text_audit = text_audits[record_id]
        if not text_audit["valid"]:
            unresolved_text.append(record_id)
            unresolved.append({"record_id": record_id, "text": record.get("text"), "nominal": None,
                               "reason": "unresolved_radius_text", "text_parse_issue": text_audit["issue"],
                               "candidate_entity_ids": []})
            continue
        nominal = text_audit["nominal"]
        directed = [row for row in candidates[record_id]
                    if (row.get("evidence", {}).get("leader") or {}).get("arrowhead_verified")]
        supported = [row for row in directed if row.get("local_reliable")]
        repartition = [row for row in directed if row.get("evidence", {}).get("whole_primitive_radius", {}).get("status") == "requires_topology_repartition"]
        observed = observations[record_id]
        targets = sorted({entity_id for row in directed for entity_id in row.get("entities", [])} |
                         {target["entity_id"] for row in observed for target in row.get("target_candidates", [])})
        base = {"record_id": record_id, "text": record.get("text"), "nominal": nominal}
        if not directed and not observed:
            unknown.append(record_id)
            rejected_claims = ownership_rejections[record_id]
            conflict = next((row for row in rejected_claims if row.get("reason") in {
                "shared_source_arrow_ownership_ambiguous", "source_arrow_claim_owned_by_other_label"}), None)
            unresolved.append({**base, "reason": conflict["reason"] if conflict else "source_arrow_not_verified",
                               "candidate_entity_ids": [],
                               "source_arrow_ownership_rejection_reasons": sorted({
                                   row["reason"] for row in rejected_claims if row.get("reason")})})
            continue
        confirmed.append(record_id)
        chosen_ids = supported[0].get("entities", []) if len(supported) == 1 else []
        chosen = chosen_ids[0] if len(chosen_ids) == 1 else None
        mapping = {**base, "entity_id": chosen,
                   "entity_type": entities.get(chosen, {}).get("type"),
                   "candidate_entity_ids": targets, "required": True,
                   "enforcement": "exact", "source_arrow_verified": True,
                   "source_evidence": [{"method": row.get("method"), "segment_px": row.get("segment_px"),
                                         "arrowhead": row.get("arrowhead")} for row in observed] or
                                      [row["evidence"]["leader"] for row in directed]}
        required.append(mapping)
        accepted = [row for row in constraints if row.get("kind") == "radius"
                    and row.get("record_id") == record_id and row.get("value") == nominal
                    and row.get("entities") == [chosen] and entities.get(chosen, {}).get("type") == "ARC"]
        # Equivalent OCR records may describe the very same equation. Preserve
        # their coverage without adding duplicate equations to the solver.
        if not accepted and chosen is not None:
            equivalent = any(row.get("record_id") == record_id and
                             row.get("reason") == "duplicate_equivalent_constraint" for row in decisions)
            if equivalent:
                accepted = [row for row in constraints if row.get("kind") == "radius"
                            and row.get("value") == nominal and row.get("entities") == [chosen]]
        if accepted:
            bound.append({**mapping, "constraint_id": accepted[0]["id"], "numeric_satisfaction_verified": False})
            continue
        if repartition:
            reason = "requires_topology_repartition"
            mapping["status"] = reason
            mapping["whole_primitive_radius"] = [row["evidence"]["whole_primitive_radius"] for row in repartition]
        elif (len(observed) > 1 and len({target["entity_id"] for row in observed
                                       for target in row.get("target_candidates", [])}) > 1) or (len(directed) > 1 and not supported):
            reason = "ambiguous_source_arrow_targets"
            ambiguous.append({**base, "candidate_entity_ids": targets, "reason": reason})
        elif targets and all(entities.get(target, {}).get("type") == "LINE" for target in targets):
            reason = "radius_target_is_line_requires_topology_edit"
        elif chosen is None:
            reason = "source_arrow_target_mapping_unresolved"
        else:
            reasons = sorted({row.get("reason") for row in decisions if row.get("record_id") == record_id and row.get("reason")})
            reason = reasons[0] if reasons else "radius_constraint_not_admitted"
        unresolved.append({**base, "reason": reason, "candidate_entity_ids": targets,
                           "source_arrow_verified": True})
    return {"schema_version": "source-radius-binding-coverage-v1",
            "recognized_radius_records": [row["id"] for row in records],
            "confirmed_arrow_records": confirmed, "required_mappings": required,
            "bound_mappings": bound, "unresolved": unresolved, "ambiguous": ambiguous,
            "unknown_arrow_records": unknown, "verified_absent_arrow_records": [],
            "unresolved_radius_text_records": unresolved_text, "unresolved_radius_text_count": len(unresolved_text),
            "radius_text_scope": "All supplied OCR records with parsed or plausible raw radius markers; full-image OCR recall is not certified.",
            "recognized_count": len(records), "required_count": len(required), "bound_count": len(bound),
            "unresolved_count": len(unresolved), "ambiguous_count": len(ambiguous),
            "all_confirmed_arrows_bound": len(required) == len(bound),
            "all_radius_records_resolved": not unresolved,
            "numeric_satisfaction_verified": False, "ground_truth_used": False,
            "scope": "Source association only. Undetected arrows remain unresolved, never a no-arrow exemption; solver and DXF readback must separately prove exact radii."}


def _radius_primitive_feasibility(entity, nominal, leader, model, graph, transform, band):
    """Check the whole observed source interval after arrow-only target choice.

    A point on a coarse ARC does not prove that its full interval is one circle.
    Hold the OCR radius fixed and fit only its centre to the source segmentation
    interval, without modifying geometry or ranking alternative entities. The
    existing source deviation budget is retained. This bounded feasibility
    search is an admission screen, not a proof of global solve feasibility.
    """
    from scipy.optimize import least_squares, minimize
    from scipy.spatial import cKDTree
    from .topology import _sample_path

    result = {"method": "fixed_nominal_radius_source_interval_v1", "entity_id": entity["id"],
              "nominal": float(nominal), "checked": False, "passed": False,
              "status": "source_boundary_observation_unavailable", "ground_truth_used": False,
              "nominal_used_to_rank": False, "geometry_modified": False,
              "scope": "Admission feasibility only; joint solve and source/DXF validation remain required."}
    raw = (model.get("extraction") or {}).get("raw_polyline_px")
    try:
        ring = np.asarray(raw, float)
        if ring.ndim != 2 or ring.shape[1] != 2 or len(ring) < 3 or not np.isfinite(ring).all():
            return result
        if np.linalg.norm(ring[-1]-ring[0]) > 1e-8:
            ring = np.vstack([ring, ring[0]])
        ring = _sample_path(ring, .5)[:-1]
        if len(ring) < 3:
            return result
        radius_px = float(nominal)*float(np.linalg.norm(transform([[1., 0.], [0., 0.]])[0]-transform([[1., 0.], [0., 0.]])[1]))
        budget = (model.get("curve_fit") or {}).get("total_deviation_budget_px", band)
        if isinstance(budget, bool) or not isinstance(budget, (float, int)) or not math.isfinite(budget) or budget <= 0 or radius_px <= 0:
            return result
        primitive = _samples(entity, transform)
        indices = cKDTree(ring).query(primitive)[1]
        unwrapped = np.unwrap(indices*2*np.pi/len(ring))*len(ring)/(2*np.pi)
        start, end = int(round(unwrapped[0])), int(round(unwrapped[-1]))
        step = 1 if end >= start else -1
        interval = np.arange(start, end+step, step)
        if len(interval) < 3 or len(interval) >= len(ring):
            result["status"] = "source_boundary_interval_unresolved"
            return result
        observed = ring[interval % len(ring)]
        samples = observed[np.unique(np.rint(np.linspace(0, len(observed)-1, min(256, len(observed)))).astype(int))]
        center = transform([entity["center"]])[0]
        starts = [center, 2*samples.mean(axis=0)-center]
        fits = []
        for initial in starts:
            fit = least_squares(lambda xy: np.linalg.norm(samples-xy, axis=1)-radius_px,
                                initial, max_nfev=150)
            residual = np.abs(np.linalg.norm(samples-fit.x, axis=1)-radius_px)
            maximum = minimize(lambda x: x[2], np.r_[fit.x, residual.max()], method="SLSQP",
                               constraints=[{"type": "ineq", "fun": lambda x: x[2]-np.abs(np.linalg.norm(samples-x[:2], axis=1)-radius_px)}],
                               options={"maxiter": 150, "ftol": 1e-8})
            for fitted_center in (fit.x, maximum.x[:2]):
                full_residual = np.abs(np.linalg.norm(observed-fitted_center, axis=1)-radius_px)
                if np.isfinite(full_residual).all():
                    fits.append((float(full_residual.max()), fitted_center, full_residual))
        if not fits:
            result["status"] = "fixed_radius_feasibility_search_failed"
            return result
        residual, center, residuals = min(fits, key=lambda row: row[0])
        sampling_bound = float(np.max(np.linalg.norm(np.diff(observed, axis=0), axis=1))/2)
        arrow = leader.get("arrowhead") or {}
        radial = np.asarray(arrow.get("tip_px"), float)-center
        direction = np.asarray(arrow.get("direction_px"), float)
        alignment = abs(float(np.dot(radial, direction)/np.linalg.norm(radial)))
        passed = bool(residual+sampling_bound <= budget and alignment >= .88)
        a, b = transform([entity["start"], entity["end"]])
        chord = float(np.linalg.norm(b-a))
        fixed_endpoint_residual = None
        if 0 < chord <= 2*radius_px:
            normal = np.array([-(b-a)[1], (b-a)[0]])/chord
            offset = math.sqrt(max(0., radius_px**2-(chord/2)**2))
            fixed_endpoint_residual = min(float(np.max(np.abs(np.linalg.norm(observed-((a+b)/2+sign*offset*normal), axis=1)-radius_px)))
                                          for sign in (-1., 1.))
        result.update(checked=True, passed=passed,
                      status="feasible_pending_joint_solve" if passed else "requires_topology_repartition",
                      observation="initial_extraction_raw_polyline_px", sample_count=len(observed),
                      source_interval_px=[observed[0].tolist(), observed[-1].tolist()],
                      radius_source_px=radius_px, center_source_px=center.tolist(),
                      max_radial_residual_px=residual, p90_radial_residual_px=float(np.quantile(residuals, .9)),
                      sampling_bound_px=sampling_bound, conservative_max_residual_px=residual+sampling_bound,
                      source_deviation_budget_px=float(budget),
                      budget_source="initial_curve_fit_total_deviation_budget_px" if "total_deviation_budget_px" in (model.get("curve_fit") or {}) else "proposal_tolerance_px",
                      nominal_circle_arrow_alignment=alignment, minimum_arrow_alignment=.88,
                      fixed_endpoints_chord_feasible=chord <= 2*radius_px,
                      fixed_endpoints_max_radial_residual_px=fixed_endpoint_residual,
                      endpoints_may_move_in_joint_solve=True)
        return result
    except (KeyError, TypeError, ValueError, IndexError, ArithmeticError):
        result["status"] = "fixed_radius_feasibility_search_failed"
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
        repartition = [p for p in proposals if p.get("evidence", {}).get("whole_primitive_radius", {}).get("status") == "requires_topology_repartition"]
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
                       "status": "requires_topology_repartition" if repartition else "binding_candidate_requires_solve" if reliable else "unresolved_source_association",
                       "candidate_count": len(proposals), "upstream_target_hypotheses": targets,
                       "geometry_edit_is_not_numeric_binding": True,
                       "reason": ("requires_topology_repartition" if repartition else "radius_target_is_line_requires_topology_edit" if any(t["entity_type"] == "LINE" for t in targets)
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


def _angle_source_observations(gray, records, graph, transform, band):
    """Associate angular labels with independently observed straight supports.

    Angular arrows can meet an extension of the contour LINE, not the material
    boundary itself. Both opposing arrowheads, the axial reference stroke and
    a collinear source stroke must therefore be observed. The nominal angle is
    deliberately unavailable to target ranking and straight-stroke detection.
    ARC targets are retained as topology-repair evidence, never angle bindings.
    """
    angles = [r for r in records if r.get("parsed", {}).get("kind") == "angle"
              and isinstance(r["parsed"].get("nominal"), (int, float))
              and 0 < r["parsed"]["nominal"] < 90 and _box(r) is not None]
    if not angles:
        return []
    origin, px, py = transform([[0., 0.], [1., 0.], [0., 1.]])
    aligned = (abs((px-origin)[1]) <= 1e-6 * max(np.linalg.norm(px-origin), 1e-9)
               and abs((py-origin)[0]) <= 1e-6 * max(np.linalg.norm(py-origin), 1e-9))
    if not aligned:
        return []
    leaders = _leaders(gray, records)
    boundary = [(e, _samples(e, transform)) for e in graph.get("entities", [])]
    observations = []
    for axis in ("vertical", "horizontal"):
        swap = axis == "horizontal"
        oriented = gray.T if swap else gray
        orient = lambda p: np.asarray(p, float)[..., ::-1] if swap else np.asarray(p, float)
        reference_lines = _axis_lines(oriented, "y")
        for row in angles:
            box = orient(_box(row)); low, high = box.min(axis=0), box.max(axis=0)
            size = max(12., float(max(high-low)))
            references = [line for line in reference_lines
                          if line["lo"] <= high[1]+size*.15 and line["hi"] >= high[1]-size*.35
                          and low[0]-size*.6 <= line["cross"] <= high[0]+size*.6
                          and line["span"] >= size*1.5]
            # Merge the two detected edges and collinear pieces of a measured
            # stroke. This does not extend its finite observed support domain.
            groups = []
            for raw in leaders:
                line = orient(raw); delta = line[1]-line[0]
                if abs(delta[1]) < max(24., size*.4):
                    continue
                slope = float(delta[0]/delta[1])
                if not .035 < abs(slope) < .8:
                    continue
                intercept = float(line[0, 0]-slope*line[0, 1])
                x = slope*high[1]+intercept
                if abs(x-box.mean(axis=0)[0]) > size*1.5:
                    continue
                if line[:, 1].max() < low[1]-size*2 or line[:, 1].min() > high[1]+size*2:
                    continue
                group = next((g for g in groups if abs(g["slope"]-slope) < .025
                              and abs(g["slope"]*high[1]+g["intercept"]-x) <= max(5., band*1.5)), None)
                if group is None:
                    groups.append({"slope": slope, "intercept": intercept, "pieces": [line]})
                else:
                    group["pieces"].append(line)
                    points = np.concatenate(group["pieces"])
                    group["slope"], group["intercept"] = np.polyfit(points[:, 1], points[:, 0], 1)
            for group in groups:
                points = np.concatenate(group["pieces"]); slope, intercept = group["slope"], group["intercept"]
                observed_angle = math.degrees(math.atan(abs(slope)))
                # This loose branch check excludes a crossing hatch stroke; it
                # never replaces the annotation by a measured/fitted angle.
                if abs(observed_angle-row["parsed"]["nominal"]) > 8.:
                    continue
                ymin, ymax = float(points[:, 1].min()), float(points[:, 1].max())
                if ymax-ymin < max(40., size*.7):
                    continue
                source_line = np.asarray([[slope*ymin+intercept, ymin], [slope*ymax+intercept, ymax]])
                targets = []
                for entity, samples in boundary:
                    samples = orient(samples)
                    distances = np.abs(samples[:, 0]-slope*samples[:, 1]-intercept)/math.sqrt(1+slope*slope)
                    keep = ((distances <= max(5., band*1.7)) &
                            (samples[:, 1] >= ymin-band) & (samples[:, 1] <= ymax+band))
                    ids = np.flatnonzero(keep)
                    if len(ids) < 3:
                        continue
                    # Disconnected intersections of a curved primitive are not
                    # a straight support interval.
                    runs = np.split(ids, np.flatnonzero(np.diff(ids) != 1)+1)
                    ids = max(runs, key=len)
                    span = float(np.ptp(samples[ids, 1]))
                    restoration = entity.get("angle_support_evidence") or {}
                    restored_segment = restoration.get("source_line") or {}
                    restored_points = np.asarray([restored_segment.get("start_px"),
                                                  restored_segment.get("end_px")], dtype=float)
                    short_restored_line = (
                        entity["type"] == "LINE" and restoration.get("record_id") == row.get("id")
                        and restoration.get("method") == "verified_two_radius_joint_line_topology_restore"
                        and restoration.get("ground_truth_used") is False
                        and restoration.get("requires_angle_binding_and_solve") is True
                        and restored_points.shape == (2, 2) and np.isfinite(restored_points).all()
                        and np.max(np.linalg.norm(restored_points - orient(source_line), axis=1)) <= 1.)
                    # Short finite straight supports restored between two
                    # source-arrow-verified radii need a smaller observation
                    # window. The independently detected source stroke and
                    # opposing angle arrows are still checked below; numeric
                    # angle tolerances and global acceptance do not change.
                    minimum_span = max(12., 4.*band) if short_restored_line else max(32., size*.6)
                    if span < minimum_span:
                        continue
                    targets.append({"entity_id": entity["id"], "entity_type": entity["type"],
                                    "source_interval": [float(ids[0]/(len(samples)-1)), float(ids[-1]/(len(samples)-1))],
                                    "supported_span_px": span, "maximum_stroke_gap_px": float(max(distances[ids])),
                                    "whole_line_supported": bool(entity["type"] == "LINE" and
                                        float(max(distances)) <= max(5., band*1.7) and
                                        span >= .78*float(np.ptp(samples[:, 1])))})
                if not targets:
                    continue
                for reference in references:
                    xref = reference["cross"]
                    if abs(slope*high[1]+intercept-xref) < size*.2:
                        continue
                    # A scanned angular dimension arc need not be horizontal;
                    # independently locate the two opposing tapered arrows.
                    arrow_options = {"reference": [], "target": []}
                    for y in np.linspace(high[1]-size*.38, high[1]+size*.24, 23):
                        for role, x in (("reference", xref), ("target", slope*y+intercept)):
                            for sign in (-1., 1.):
                                arrow = _arrowhead_evidence(oriented, np.asarray([x, y]), np.asarray([sign, 0.]), size, band)
                                if arrow is None:
                                    continue
                                tip = np.asarray(arrow["tip_px"])
                                gap = abs(tip[0] - (xref if role == "reference" else slope*tip[1]+intercept))
                                if gap <= max(10., band*1.7):
                                    arrow_options[role].append(arrow)
                    pairs = [(a, b) for a in arrow_options["reference"] for b in arrow_options["target"]
                             if a["direction_px"][0]*b["direction_px"][0] < 0
                             and abs(a["tip_px"][1]-b["tip_px"][1]) <= size*.4
                             and abs(a["tip_px"][0]-b["tip_px"][0]) >= size*.2]
                    if not pairs:
                        continue
                    a, b = max(pairs, key=lambda pair: pair[0]["score"]+pair[1]["score"])
                    def original_arrow(arrow):
                        return {**arrow, "tip_px": orient(arrow["tip_px"]).tolist(),
                                "direction_px": orient(arrow["direction_px"]).tolist()}
                    observations.append({"record_id": row["id"], "nominal": row["parsed"]["nominal"],
                                         "reference_axis": axis, "verified": True,
                                         "reference_stroke": dict(reference),
                                         "source_line": {"start_px": orient(source_line[0]).tolist(),
                                                         "end_px": orient(source_line[1]).tolist(),
                                                         "observed_direction_deg_from_axis": observed_angle,
                                                         "detected_piece_count": len(group["pieces"])},
                                         "target_candidates": targets,
                                         "evidence": {"method": "source_angular_arrows_axis_and_straight_support_v1",
                                                      "verified": True, "nominal_used_to_rank": False,
                                                      "reference_arrow": original_arrow(a), "target_arrow": original_arrow(b),
                                                      "coarse_direction_tolerance_deg": 8.,
                                                      "source_coordinate_axes_aligned": True}})
    return observations


def _angle_candidates(observations, graph):
    result = []
    entity_map = {e["id"]: e for e in graph.get("entities", [])}
    by_record = defaultdict(list)
    for observation in observations:
        if observation.get("verified"):
            by_record[observation["record_id"]].append(observation)
    for record_id, rows in by_record.items():
        signatures = {(row["reference_axis"], target["entity_id"]) for row in rows
                      for target in row["target_candidates"] if target.get("whole_line_supported")}
        # A curved or differently targeted observation still makes an ambiguous
        # label unresolved; choosing only LINE alternatives would hide a conflict.
        all_targets = {(row["reference_axis"], target["entity_id"]) for row in rows for target in row["target_candidates"]}
        for axis, entity_id in sorted(signatures):
            observation = next(row for row in rows if row["reference_axis"] == axis and
                               any(t["entity_id"] == entity_id and t.get("whole_line_supported") for t in row["target_candidates"]))
            # A large adjacent ARC can lie within the straight support band
            # close to the tangent join. It must not disqualify an independently
            # observed whole LINE, nor be included in the angle constraint.
            target_line = entity_map.get(entity_id, {})
            line_nodes = {target_line.get("start_node"), target_line.get("end_node")}-{None}
            line_span = max(t["supported_span_px"] for row in rows for t in row["target_candidates"]
                            if t["entity_id"] == entity_id and t.get("whole_line_supported"))
            def local_neighbor(other_axis, other_id):
                other = entity_map.get(other_id, {})
                shared = line_nodes & {other.get("start_node"), other.get("end_node")}
                targets = [t for row in rows for t in row["target_candidates"] if t["entity_id"] == other_id]
                if not (other_axis == axis and other.get("type") == "ARC" and shared and targets):
                    return False
                if max(t["supported_span_px"] for t in targets) < line_span*.6:
                    return True
                # A gentle radius can look straight within the image band near
                # its tangent join, even over a span as long as the short LINE.
                # Treat only a bounded interval touching that shared endpoint
                # as the radius's local tangent, not as a second angle object.
                for target in targets:
                    interval = target.get("source_interval")
                    if (not isinstance(interval, (list, tuple)) or len(interval) != 2
                            or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in interval)):
                        continue
                    lo, hi = sorted(interval)
                    if (other.get("start_node") in shared and lo <= .12 and hi <= .4) or (
                            other.get("end_node") in shared and lo >= .6 and hi >= .88):
                        return True
                return False
            unique = len(signatures) == 1 and all((a, e) == (axis, entity_id) or local_neighbor(a, e) for a, e in all_targets)
            result.append({"record_id": record_id, "kind": "angle", "reference_axis": axis,
                           "angle_mode": "unsigned", "entities": [entity_id], "nodes": [],
                           "value": observation["nominal"], "local_reliable": unique,
                           "evidence": {**observation["evidence"], "angle_observation": observation,
                                        "numeric_constraint_applied": False}, "_score": 0.})
    return result


def _constructed_fillet_tangent_relations(gray, records, image_path, graph, transform,
                                          relations, candidates):
    """Recheck exact LINE/ARC/LINE construction against current source pixels.

    This is a design tangency of an admitted local fillet, not an additional OCR
    dimension or a verdict carried from a prior graph.  An ordinary ARC refit
    has no construction witness and cannot enter this path.
    """
    if graph.get("source_sha256") != hashlib.sha256(Path(image_path).read_bytes()).hexdigest():
        return relations
    from .topology import _StrokeEvidence, _primitive_distance, _tangent
    entities = graph.get("entities") or []
    if len(entities) < 3:
        return relations
    grid = float(graph.get("source_grid_pitch_px") or 0.)
    tolerance = float(graph.get("proposal_tolerance_px") or 0.)
    if not all(math.isfinite(value) and value > 0 for value in (grid, tolerance)):
        return relations
    accepted_radius = {(row["record_id"], row["entities"][0]): row for row in candidates
        if row.get("kind") == "radius" and row.get("local_reliable") is True
        and len(row.get("entities") or []) == 1
        and (row.get("evidence") or {}).get("whole_primitive_radius", {}).get("passed") is True
        and ((row.get("evidence") or {}).get("leader") or {}).get("arrowhead_verified") is True}
    result = list(relations)
    by_joint = {(frozenset(row.get("entities") or []), tuple(row.get("nodes") or [])): index
                for index, row in enumerate(result) if row.get("type") == "tangent"}
    used_ids = {row.get("id") for row in result}
    stroke = None
    stroke_cache = {}
    for index, arc in enumerate(entities):
        if (arc.get("type") != "ARC" or arc.get("fillet_construction") not in {
                "existing_finite_line_supports", "existing_lines_short_connector_redistribution",
                "existing_line_arc_line_tangent_reinsertion"}):
            continue
        refinement = arc.get("source_refinement") or {}
        binding = arc.get("radius_binding") or {}
        record_id = binding.get("record_id")
        radius = accepted_radius.get((record_id, arc.get("id")))
        try:
            radius_matches = math.isclose(float(arc.get("radius", 0.)), float(radius["value"]),
                                          rel_tol=1e-8, abs_tol=1e-8) if radius is not None else False
        except (TypeError, ValueError, KeyError):
            radius_matches = False
        if (radius is None or any(refinement.get(key) is not True for key in (
                "outer_endpoints_fixed", "support_directions_fixed", "support_types_fixed",
                "radius_exact", "source_path_and_arrow_verified"))
                or refinement.get("ground_truth_used") is not False
                or not radius_matches):
            continue
        before, after = entities[(index-1) % len(entities)], entities[(index+1) % len(entities)]
        if (before.get("type") != "LINE" or after.get("type") != "LINE"
                or before.get("end_node") != arc.get("start_node")
                or arc.get("end_node") != after.get("start_node")
                or arc.get("start_node") == arc.get("end_node")):
            continue
        try:
            original = np.asarray(refinement["original_finite_line_supports_px"], float)
            contacts = np.asarray(refinement["constructed_contact_points_px"], float)
            current = [transform(np.asarray([row["start"], row["end"]], float))
                       for row in (before, after)]
            middle = refinement.get("original_middle_source_geometry")
            if (original.shape != (2, 2, 2) or contacts.shape != (2, 2)
                    or not np.isfinite(original).all() or not np.isfinite(contacts).all()
                    or any(not np.isfinite(segment).all() for segment in current)
                    or np.linalg.norm(current[0][0]-original[0][0]) > 1e-4
                    or np.linalg.norm(current[1][1]-original[1][1]) > 1e-4
                    or np.linalg.norm(current[0][1]-contacts[0]) > 1e-4
                    or np.linalg.norm(current[1][0]-contacts[1]) > 1e-4):
                continue
            old_lengths = np.linalg.norm(np.diff(original, axis=1)[:, 0], axis=1)
            new_lengths = np.asarray([np.linalg.norm(np.diff(segment, axis=0)[0]) for segment in current])
            if np.any(old_lengths < max(24., 4*grid)) or np.any(new_lengths <= tolerance):
                continue
            old_directions = (original[:, 1]-original[:, 0])/old_lengths[:, None]
            new_directions = np.asarray([(segment[1]-segment[0])/length
                                         for segment, length in zip(current, new_lengths)])
            if np.any(np.sum(old_directions*new_directions, axis=1) < 1-1e-8):
                continue
            extensions = new_lengths-old_lengths
            for side, extension in zip(("before", "after"), extensions):
                if extension <= 1e-4:
                    continue
                name = ("replaced_arc_domain_redistribution" if middle and middle.get("type") == "ARC"
                        else "connector_domain_redistribution")
                receipt = next((row for row in refinement.get(name) or []
                                if row.get("side") == side and
                                abs(float(row.get("extension_px", -1))-extension) <= 1e-4), None)
                if (receipt is None or not isinstance(middle, dict)
                        or extension > math.dist(middle["start"], middle["end"])
                        or float(_primitive_distance(contacts[0 if side == "before" else 1][None, :], middle)[0]) >
                           (max(2*tolerance, 2.) if middle["type"] == "ARC" else tolerance)):
                    raise ValueError("fillet_extension_not_in_original_middle_domain")
            dots = (float(_tangent(before, True) @ _tangent(arc, False)),
                    float(_tangent(arc, True) @ _tangent(after, False)))
            if min(dots) < 1-1e-10:
                continue
            if stroke is None:
                stroke = _StrokeEvidence(gray, records, grid)
            support = []
            for segment in original:
                key = tuple(np.round(segment.ravel(), 4))
                if key not in stroke_cache:
                    stroke_cache[key] = stroke.summarize(np.linspace(segment[0], segment[1], 64))
                measurement = stroke_cache[key]
                if (measurement["stroke_supported_fraction"] < .60 or
                        measurement["p90_edge_distance_px"] > max(2*grid, tolerance+grid)):
                    break
                support.append({"stroke_supported_fraction": measurement["stroke_supported_fraction"],
                                "p90_edge_distance_px": measurement["p90_edge_distance_px"]})
            if len(support) != 2:
                continue
        except (ValueError, TypeError, KeyError, IndexError, ZeroDivisionError):
            continue
        for neighbor, node, dot, ordered_entities in (
                (before, arc["start_node"], dots[0], [before["id"], arc["id"]]),
                (after, arc["end_node"], dots[1], [arc["id"], after["id"]])):
            key = frozenset((arc["id"], neighbor["id"])), (node,)
            old_index = by_joint.get(key)
            if old_index is not None and result[old_index].get("local_reliable") is True:
                continue
            evidence = {"method": "source_bound_exact_fillet_construction_tangent_v1",
                        "verified": True, "evidence_class": "source_bound_design_construction",
                        "independent_source_tangent_measurement": False,
                        "record_id": record_id, "shared_node": node,
                        "construction": arc["fillet_construction"],
                        "original_finite_line_strokes": support,
                        "directed_tangent_deviation_degrees": math.degrees(math.acos(min(1., max(-1., dot)))),
                        "radius_candidate_id": radius.get("id"),
                        "original_source_image_rechecked": True,
                        "ocr_angle_claimed": False, "ground_truth_used": False}
            if old_index is None:
                number = len(result)
                while f"rel{number:03d}" in used_ids:
                    number += 1
                item = {"id": f"rel{number:03d}", "type": "tangent",
                        "entities": ordered_entities, "nodes": [node],
                        "required": False}
                used_ids.add(item["id"])
                by_joint[key] = len(result)
                result.append(item)
                old_index = len(result)-1
            result[old_index].update(source="source_bound_design_fillet", entities=ordered_entities,
                                     local_reliable=True, nodes=[node], evidence=evidence,
                                     construction_arc_entity_id=arc["id"],
                                     construction_record_id=record_id)
    return result


def _source_owned_angle_conflict(group, candidates, graph, model):
    """Resolve only a clearly nearer arrow on the same finite source LINE.

    Angular arrows can land on an extension of a material line. If two
    different labels extrapolate to the same line, their independently
    observed arrowheads must localize the finite target. A close/ambiguous
    pair remains unresolved; OCR values never rank the ownership.
    """
    if len(group) != 2 or any(row.get("kind") != "angle" for row, _ in group):
        return None
    entity_ids = group[0][0].get("entities")
    if (not isinstance(entity_ids, list) or len(entity_ids) != 1 or
            any(row.get("entities") != entity_ids for row, _ in group)):
        return None
    entity = next((row for row in graph.get("entities", []) if row.get("id") == entity_ids[0]), None)
    if not entity or entity.get("type") != "LINE":
        return None
    try:
        start, end = _source_transform(model, graph)([entity["start"], entity["end"]])
        direction = end - start
        length = float(np.linalg.norm(direction))
        if not math.isfinite(length) or length <= 0:
            return None
        measured = []
        for constraint, decision in group:
            candidate = candidates.get(decision.get("candidate_id"), {})
            observation = (candidate.get("evidence") or {}).get("angle_observation") or {}
            evidence = observation.get("evidence") or {}
            arrow = evidence.get("target_arrow") or {}
            reference_arrow = evidence.get("reference_arrow") or {}
            target = next((row for row in observation.get("target_candidates", [])
                           if row.get("entity_id") == entity_ids[0] and row.get("whole_line_supported") is True), None)
            tip = np.asarray(arrow.get("tip_px"), float)
            if (observation.get("verified") is not True or arrow.get("verified") is not True
                    or reference_arrow.get("verified") is not True
                    or target is None or tip.shape != (2,) or not np.isfinite(tip).all()):
                return None
            projection = float(np.clip(np.dot(tip - start, direction) / (length * length), 0., 1.))
            distance = float(np.linalg.norm(tip - (start + projection * direction)))
            measured.append((distance, constraint, decision, tip))
    except (KeyError, TypeError, ValueError, ArithmeticError):
        return None
    measured.sort(key=lambda row: row[0])
    near, far = measured
    grid = float(graph.get("source_grid_pitch_px") or 1.)
    if (not math.isfinite(grid) or grid <= 0 or near[0] > 1.25 * length + 3. * grid
            or far[0] - near[0] < max(3. * grid, .5 * length)
            or far[0] < 1.5 * near[0]):
        return None
    # The farther label must have a different, finite, similarly directed
    # source-topology LINE nearer to its arrow. This witness prevents two
    # labels on the same extension from being silently forced onto one side.
    transform = _source_transform(model, graph)
    other_line_witness = False
    for other in graph.get("entities", []):
        if other.get("type") != "LINE" or other.get("id") == entity_ids[0]:
            continue
        try:
            other_start, other_end = transform([other["start"], other["end"]])
            other_direction = other_end - other_start
            other_length = float(np.linalg.norm(other_direction))
            # Unsigned angular dimensions may lie on opposite sides of the
            # reference axis; compare the source slopes' magnitudes.
            source_angle = math.degrees(math.atan2(abs(direction[0]), abs(direction[1])))
            other_angle = math.degrees(math.atan2(abs(other_direction[0]), abs(other_direction[1])))
            if (not math.isfinite(other_length) or other_length <= 0 or
                    abs(source_angle - other_angle) > 8.):
                continue
            def gap(tip):
                fraction = float(np.clip(np.dot(tip - other_start, other_direction) /
                                         (other_length * other_length), 0., 1.))
                return float(np.linalg.norm(tip - (other_start + fraction * other_direction)))
            if (near[0] + .5 * length < gap(near[3]) and
                    gap(far[3]) + .5 * length < far[0]):
                other_line_witness = True
                break
        except (KeyError, TypeError, ValueError, ArithmeticError):
            continue
    if not other_line_witness:
        return None
    return near[2]


def _radius_joint_conflict_preflight(candidate, observations, candidates):
    """Find a source-proven target before checking its fixed-radius interval.

    Multiple full-shaft observations of one label can localize the same arrow
    on opposite sides of a thick/crossing source stroke. A unique observation
    must identify the proposed ARC, every other observation must still include
    it, and every extra ARC at the joint must have a *different*, independently
    verified radius label. This preflight intentionally does not require the
    proposed radius's feasibility result: the ambiguous-tip score margin keeps
    that check from running during ordinary single-label admission.
    """
    if candidate.get("kind") != "radius" or len(candidate.get("entities", [])) != 1:
        return None
    evidence = candidate.get("evidence") or {}
    leader = evidence.get("leader") or {}
    if (not leader.get("arrowhead_verified") or
            (leader.get("radial_alignment") or 0.) < .88 or
            (evidence.get("source_text") or {}).get("symbol_confusion")):
        return None
    record_id, chosen = candidate["record_id"], candidate["entities"][0]
    observed = [row for row in observations if row.get("record_id") == record_id
                and row.get("arrowhead_verified")]
    if len(observed) < 2:
        return None
    if any(not (row.get("shaft_evidence") or {}).get("verified") or
           not (row.get("source_text_shaft_attachment") or {}).get("strong_text_adjacency") or
           (row.get("source_arrow_ownership") or {}).get("status") != "globally_unique_source_claim"
           for row in observed):
        return None
    # Do not silently discard a competing LINE at the same tip: that can be
    # an unresolved fillet topology, not evidence that the ARC wins.
    if any(target.get("entity_type") != "ARC" for row in observed
           for target in row.get("target_candidates") or []):
        return None
    target_sets = [{target.get("entity_id") for target in row.get("target_candidates") or []
                    if target.get("entity_id")} for row in observed]
    if (not all(chosen in targets for targets in target_sets) or
            not any(targets == {chosen} for targets in target_sets)):
        return None
    unique_leader = next(row for row, targets in zip(observed, target_sets) if targets == {chosen})
    alternatives = sorted(set().union(*target_sets) - {chosen})
    if not alternatives:
        return None
    # A second uniquely directed tip to another ARC means two genuine arrow
    # targets may exist. Do not erase that evidence by resolving the joint.
    if any(targets == {other} for other in alternatives for targets in target_sets):
        return None
    conflicting = []
    for other in alternatives:
        claims = [row for row in candidates
                  if row is not candidate and row.get("kind") == "radius" and
                  row.get("record_id") != record_id and row.get("entities") == [other] and
                  row.get("value") != candidate.get("value") and row.get("local_reliable") and
                  (row.get("evidence") or {}).get("uniquely_supported_leader") and
                  not ((row.get("evidence") or {}).get("source_text") or {}).get("symbol_confusion") and
                  ((row.get("evidence") or {}).get("leader") or {}).get("arrowhead_verified") and
                  ((row.get("evidence") or {}).get("whole_primitive_radius") or {}).get("passed")]
        if not claims:
            return None
        if any(sum(bool(row.get("local_reliable")) for row in candidates
                   if row.get("record_id") == claim["record_id"]) != 1 for claim in claims):
            return None
        conflicting.append({"entity_id": other,
                            "record_ids": sorted({claim["record_id"] for claim in claims})})
    audit = {"method": "source_arrow_common_target_and_incompatible_radius_claim_v1",
            "entity_id": chosen, "source_observation_count": len(observed),
            "independent_incompatible_claims": conflicting,
            "nominal_used_to_resolve_constraint_conflict": True,
            "nominal_used_to_rank_arrow_geometry": False,
            "ground_truth_used": False}
    return audit, unique_leader


def _radius_joint_conflict_resolution(candidate, observations, candidates):
    """Admit the common target only after its original source interval passes."""
    preflight = _radius_joint_conflict_preflight(candidate, observations, candidates)
    if preflight is None or not ((candidate.get("evidence") or {}).get("whole_primitive_radius") or {}).get("passed"):
        return None
    audit, _ = preflight
    return {**audit, "fixed_nominal_source_interval_feasible": True}


def _radius_line_joint_preflight(candidate, observations, graph):
    """Exclude an adjacent LINE beside an independently observed radius ARC.

    A radius arrow at a tangent join can be within the target band of both
    primitives. A second, independently verified localization must identify
    the ARC alone; the LINE must share its endpoint. Coarse fitted tangency
    cannot be an admission requirement because the radius/fillet solver must
    correct that property. The OCR radius is still checked against the
    original source interval before this binding can be used.
    """
    if candidate.get("kind") != "radius" or len(candidate.get("entities", [])) != 1:
        return None
    evidence = candidate.get("evidence") or {}
    leader = evidence.get("leader") or {}
    if (not leader.get("arrowhead_verified") or
            (leader.get("radial_alignment") or 0.) < .88 or
            (evidence.get("source_text") or {}).get("symbol_confusion")):
        return None
    chosen = candidate["entities"][0]
    entities = {row["id"]: row for row in graph.get("entities", [])}
    arc = entities.get(chosen)
    if arc is None or arc.get("type") != "ARC":
        return None
    observed = [row for row in observations if row.get("record_id") == candidate["record_id"]]
    if len(observed) < 2 or any(
            not row.get("arrowhead_verified") or
            not (row.get("shaft_evidence") or {}).get("verified") or
            not (row.get("source_text_shaft_attachment") or {}).get("strong_text_adjacency") or
            (row.get("source_arrow_ownership") or {}).get("status") != "globally_unique_source_claim"
            for row in observed):
        return None
    target_sets = [{target.get("entity_id") for target in row.get("target_candidates") or []
                    if target.get("entity_id")} for row in observed]
    if not all(chosen in targets for targets in target_sets):
        return None
    unique = [row for row, targets in zip(observed, target_sets) if targets == {chosen}]
    extra = set().union(*target_sets) - {chosen}
    if not unique or len(extra) != 1:
        return None
    line = entities.get(next(iter(extra)))
    if line is None or line.get("type") != "LINE":
        return None
    common = ({arc.get("start_node"), arc.get("end_node")}-{None}) & (
        {line.get("start_node"), line.get("end_node")}-{None})
    if len(common) != 1:
        return None
    node = next(iter(common))
    audit = {"method": "source_radius_arrow_at_verified_line_arc_joint_v2",
             "entity_id": chosen, "neighboring_line_id": line["id"],
             "shared_node_id": node,
             "source_observation_count": len(observed),
             "arc_only_observation_count": len(unique),
             "fitted_tangency_used_to_admit": False,
             "nominal_used_to_rank_arrow_geometry": False, "ground_truth_used": False}
    return audit, unique[0]


def _radius_one_arrow_incompatible_claim_preflight(candidate, observations, candidates, graph):
    """At one shared ARC joint, an independently bound incompatible R owns its ARC.

    This is only a source-constraint exclusion: one full verified shaft must
    target exactly two adjacent ARCs and the other ARC needs its own unique,
    feasible and source-bound radius label. It never chooses by fitted radius.
    """
    if candidate.get("kind") != "radius" or len(candidate.get("entities", [])) != 1:
        return None
    evidence = candidate.get("evidence") or {}
    leader = evidence.get("leader") or {}
    if (not leader.get("arrowhead_verified") or
            (leader.get("radial_alignment") or 0.) < .88 or
            (evidence.get("source_text") or {}).get("symbol_confusion")):
        return None
    chosen = candidate["entities"][0]
    observed = [row for row in observations if row.get("record_id") == candidate["record_id"]]
    if len(observed) != 1:
        return None
    row = observed[0]
    if (not row.get("arrowhead_verified") or
            not (row.get("shaft_evidence") or {}).get("verified") or
            not (row.get("source_text_shaft_attachment") or {}).get("strong_text_adjacency") or
            (row.get("source_arrow_ownership") or {}).get("status") != "globally_unique_source_claim"):
        return None
    targets = row.get("target_candidates") or []
    if (len(targets) != 2 or any(target.get("entity_type") != "ARC" for target in targets) or
            len({target.get("entity_id") for target in targets}) != 2 or
            chosen not in {target.get("entity_id") for target in targets}):
        return None
    other = next(target["entity_id"] for target in targets if target["entity_id"] != chosen)
    entities = {entity["id"]: entity for entity in graph.get("entities", [])}
    first, second = entities.get(chosen), entities.get(other)
    if (first is None or second is None or first.get("type") != "ARC" or
            second.get("type") != "ARC" or not (
                ({first.get("start_node"), first.get("end_node")}-{None}) &
                ({second.get("start_node"), second.get("end_node")}-{None}))):
        return None
    claims = [item for item in candidates if item is not candidate and
              item.get("kind") == "radius" and item.get("record_id") != candidate["record_id"] and
              item.get("entities") == [other] and item.get("value") != candidate.get("value") and
              item.get("local_reliable") and
              (item.get("evidence") or {}).get("uniquely_supported_leader") and
              ((item.get("evidence") or {}).get("leader") or {}).get("arrowhead_verified") and
              not (((item.get("evidence") or {}).get("source_text") or {}).get("symbol_confusion")) and
              ((item.get("evidence") or {}).get("whole_primitive_radius") or {}).get("passed")]
    if len(claims) != 1 or sum(bool(item.get("local_reliable")) for item in candidates
                               if item.get("record_id") == claims[0]["record_id"]) != 1:
        return None
    audit = {"method": "one_source_arrow_and_independent_incompatible_radius_claim_v1",
             "entity_id": chosen, "source_observation_count": 1,
             "independent_incompatible_claims": [{"entity_id": other,
                                                  "record_ids": [claims[0]["record_id"]]}],
             "nominal_used_to_resolve_constraint_conflict": True,
             "nominal_used_to_rank_arrow_geometry": False, "ground_truth_used": False}
    return audit, row


def _radius_ocr_owned_target_preflight(candidate, observations, candidates):
    """Resolve separate arrow hypotheses only when source text owns one ARC.

    A verified Hough stroke can cross the OCR box without belonging to its
    glyphs. One fully verified, strongly glyph-attached arrow must uniquely
    target the proposed ARC; every weaker arrow must target a different ARC
    already uniquely owned by another incompatible, source-feasible label.
    This never uses constructed CAD radii or reference geometry.
    """
    if candidate.get("kind") != "radius" or len(candidate.get("entities", [])) != 1:
        return None
    evidence = candidate.get("evidence") or {}
    leader = evidence.get("leader") or {}
    if (not leader.get("arrowhead_verified") or
            (leader.get("radial_alignment") or 0.) < .88 or
            (evidence.get("source_text") or {}).get("symbol_confusion")):
        return None
    chosen = candidate["entities"][0]
    observed = [row for row in observations if row.get("record_id") == candidate["record_id"]]
    if len(observed) < 2 or any(
            not row.get("arrowhead_verified") or
            not (row.get("shaft_evidence") or {}).get("verified") or
            (row.get("source_arrow_ownership") or {}).get("status") != "globally_unique_source_claim" or
            not (row.get("source_text_shaft_attachment") or {}).get("checked")
            for row in observed):
        return None
    strong = [row for row in observed if
              (row.get("source_text_shaft_attachment") or {}).get("strong_text_adjacency")]
    if len(strong) != 1:
        return None
    attached = strong[0]
    targets = attached.get("target_candidates") or []
    if (len(targets) != 1 or targets[0].get("entity_id") != chosen or
            targets[0].get("entity_type") != "ARC"):
        return None
    competitors = {}
    for row in observed:
        if row is attached:
            continue
        for target in row.get("target_candidates") or []:
            other = target.get("entity_id")
            if not other or other == chosen or target.get("entity_type") != "ARC":
                return None
            competitors[other] = None
    if not competitors:
        return None
    for other in competitors:
        claims = [item for item in candidates if item is not candidate and
                  item.get("kind") == "radius" and item.get("record_id") != candidate["record_id"] and
                  item.get("entities") == [other] and item.get("value") != candidate.get("value") and
                  item.get("local_reliable") and
                  (item.get("evidence") or {}).get("uniquely_supported_leader") and
                  ((item.get("evidence") or {}).get("leader") or {}).get("arrowhead_verified") and
                  not (((item.get("evidence") or {}).get("source_text") or {}).get("symbol_confusion")) and
                  ((item.get("evidence") or {}).get("whole_primitive_radius") or {}).get("passed")]
        if len(claims) != 1 or sum(bool(item.get("local_reliable")) for item in candidates
                                   if item.get("record_id") == claims[0]["record_id"]) != 1:
            return None
        competitors[other] = claims[0]["record_id"]
    audit = {"method": "single_strong_source_text_arrow_and_incompatible_occupied_targets_v1",
             "entity_id": chosen, "source_observation_count": len(observed),
             "strong_text_attached_arrow_count": 1,
             "independent_incompatible_claims": [
                 {"entity_id": other, "record_ids": [rid]} for other, rid in sorted(competitors.items())],
             "nominal_used_to_resolve_constraint_conflict": True,
             "nominal_used_to_rank_arrow_geometry": False, "ground_truth_used": False}
    return audit, attached


def _admit_joint_radius_conflicts(candidates, observations, model, graph, transform, band):
    """Run the original fixed-radius source gate after a source-only joint proof."""
    radius_entities = {entity["id"]: entity for entity in graph.get("entities", [])
                       if entity.get("type") == "ARC"}
    for candidate in candidates:
        if candidate.get("kind") != "radius" or candidate.get("local_reliable"):
            continue
        preflight = (_radius_joint_conflict_preflight(candidate, observations, candidates) or
                     _radius_line_joint_preflight(candidate, observations, graph) or
                     _radius_one_arrow_incompatible_claim_preflight(candidate, observations, candidates, graph) or
                     _radius_ocr_owned_target_preflight(candidate, observations, candidates))
        if preflight is None:
            continue
        entity = radius_entities.get(candidate["entities"][0])
        if entity is None:
            continue
        evidence = candidate["evidence"]
        if "whole_primitive_radius" not in evidence:
            _, unique_leader = preflight
            evidence["whole_primitive_radius"] = _radius_primitive_feasibility(
                entity, candidate["value"], unique_leader, model, graph, transform, band)
        if (evidence.get("whole_primitive_radius") or {}).get("passed"):
            candidate["local_reliable"] = True
            evidence["source_joint_radius_conflict_resolution"] = {
                **preflight[0], "fixed_nominal_source_interval_feasible": True}
            evidence["multiple_directed_source_targets_require_review"] = False


def build_binding_candidates(image_path, document, model, graph, output_dir):
    image_path, out = Path(image_path), Path(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    gray = cv2.imdecode(np.fromfile(str(image_path),np.uint8),cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError("source image unreadable")
    all_records = canonical_records(document)
    records = [{"id":r["id"],"text":str(r.get("text","")),"parsed":r["parsed"],"box":r.get("box"),
                "source_arrow_proposals":r.get("source_arrow_proposals", [])} for r in all_records]
    text_evidence={row["id"]:_source_text_evidence(gray,row) for row in records}
    for row in records:row["source_text_evidence"]=text_evidence[row["id"]]
    eligible = [r for r in records if r["parsed"].get("kind") in {"radius","length","diameter","angle"}
                and r["parsed"].get("nominal") is not None and r["parsed"]["nominal"] > 0 and _box(r) is not None]
    band = max(4.,float(graph.get("proposal_tolerance_px",max(gray.shape)/500)))
    transform = _source_transform(model,graph)
    pixels_per_mm=(model.get("scale") or {}).get("pixels_per_mm")
    candidates = _length_candidates(gray,eligible,graph,band,pixels_per_mm)
    angle_observations = _angle_source_observations(gray, records, graph, transform, band)
    candidates.extend(_angle_candidates(angle_observations, graph))
    arcs = [(entity,_samples(entity,transform),transform([entity["center"]])[0])
            for entity in graph.get("entities",[]) if entity.get("type")=="ARC"]
    contours = [_samples(entity,transform) for entity in graph.get("entities",[])]
    leaders = _leaders(gray,eligible)
    radius_rejections = []
    arrow_ownership = {}
    radius_observations = _radius_source_observations(gray, eligible, graph, transform, leaders, band,
                                                      radius_rejections, arrow_ownership)
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
            # Keep rejected legacy paths for review, but only observations
            # admitted by the complete global source-arrow ownership audit
            # may establish a fresh binding. Per-arc detection cannot bypass
            # a different label's claim to the same physical arrow.
            legacy_leader = _leader_evidence(box,points,center,leaders + inherited,band,gray,contours,occluded)
            leader = None
            for observed in radius_observations:
                if observed["record_id"] != row["id"]:
                    continue
                target = next((target for target in observed["target_candidates"]
                               if target["entity_id"] == entity["id"]), None)
                if target is None:
                    continue
                direction = np.asarray(observed["arrowhead"]["direction_px"], float)
                radial = np.asarray(observed["arrowhead"]["tip_px"], float)-center
                radial_length = float(np.linalg.norm(radial))
                alignment = abs(float(np.dot(direction, radial)/radial_length)) if radial_length else 0.
                if alignment < .88:
                    continue
                verified = {**observed, "radial_alignment": alignment,
                            "score": target["tip_gap_px"]+.2*observed["label_gap_px"]+(1-alignment)*band*3}
                if leader is None or verified["score"] < leader["score"]:
                    leader = verified
            evidence = {"method":"source_label_proximity", "label_to_arc_px":distance,
                        "coarse_proposal_band_px":band, "fitted_radius":entity["radius"],
                        "nominal_difference":float(entity["radius"])-row["parsed"]["nominal"],
                        "leader":leader,"nominal_used_to_rank":False,
                        "occluded_leader_hypotheses":occluded,
                        "upstream_leader_hypotheses_rechecked":len(inherited),
                        "legacy_leader_not_an_ownership_certificate":legacy_leader is not None and leader is None,
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
            independent = bool(best["evidence"]["leader"] is not None and (
                len(options)==1 or options[1]["_score"]-best["_score"] > max(5.,band*.5)))
            best["local_reliable"] = independent
            if independent:
                entity = next(entity for entity, _, _ in arcs if entity["id"] == best["entities"][0])
                feasibility = _radius_primitive_feasibility(entity, row["parsed"]["nominal"], best["evidence"]["leader"],
                                                             model, graph, transform, band)
                best["evidence"]["whole_primitive_radius"] = feasibility
                best["local_reliable"] = bool(feasibility["passed"])
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
        if candidate["kind"] == "radius":
            observed = [row for row in radius_observations if row["record_id"] == candidate["record_id"]]
            candidate["evidence"]["directed_source_target_count"] = len(observed)
            distinct_targets = {target["entity_id"] for row in observed for target in row.get("target_candidates", [])}
            candidate["evidence"]["source_arrow_ownership_rejection_reasons"] = sorted({
                row["reason"] for row in radius_rejections if row.get("record_id") == candidate["record_id"] and row.get("reason")})
            if len(observed) > 1 and len(distinct_targets) > 1:
                candidate["local_reliable"] = False
                candidate["evidence"]["multiple_directed_source_targets_require_review"] = True
    # At a thick shared joint the best/runner-up tip scores can be closer than
    # the ordinary independent-leader margin. First establish the target from
    # full source arrows and a separately feasible incompatible label; only
    # then test this candidate's original raw-mask interval at its OCR radius.
    _admit_joint_radius_conflicts(candidates, radius_observations, model, graph, transform, band)
    reliable_counts=Counter(c["record_id"] for c in candidates if c["local_reliable"])
    for candidate in candidates:
        if reliable_counts[candidate["record_id"]]>1:
            candidate["local_reliable"]=False
            candidate["evidence"]["record_has_multiple_supported_bindings"]=True
    relations=structural_evidence(gray,records,graph,transform,_samples,band)
    relations=_raw_boundary_tangent_relations(gray,records,model,graph,transform,band,relations)
    relations=_constructed_fillet_tangent_relations(
        gray,records,image_path,graph,transform,relations,candidates)
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
                         "recognized_angles":sum(r["parsed"]["kind"] == "angle" for r in eligible),
                         "source_verified_angle_records":len({r["record_id"] for r in angle_observations if r.get("verified")}),
                         "angle_candidates":sum(c["kind"] == "angle" for c in candidates),
                         "source_verified_relations":sum(r["local_reliable"] for r in relations)},
               "annotation_diagnostics":_annotation_issues(graph,eligible,candidates),
               "radius_source_observations":radius_observations,
               "angle_source_observations":angle_observations,
               "radius_source_rejections":radius_rejections,
               "source_arrow_ownership":arrow_ownership,
               "constructed_radius_priors":_constructed_radius_priors(graph,records),
               "ground_truth_used":False,"issues":[],"proposal_tolerance_px":band}
    inventory["radius_binding_coverage"] = radius_binding_coverage(inventory, graph)
    _write(out/"binding-candidates.json",inventory)
    return inventory


def _disabled(status="disabled",reason=None):
    return {"status":status,"reason":reason,"network_requests":0,"http_success":False,"schema_success":False,
            "image_sent":False,"ground_truth_sent":False,"bindings":[],"relations":[]}


def _constraint(candidate, source):
    result = {"id":"k"+candidate["id"],"kind":candidate["kind"],"record_id":candidate["record_id"],
            "entities":candidate["entities"],"nodes":candidate["nodes"],"value":candidate["value"],"source":source,
            "candidate_id":candidate["id"]}
    if candidate["kind"] == "radius":
        result.update(required=True, enforcement="exact", nominal_source="source_ocr",
                      source_arrow_verified=True)
    if candidate["kind"] == "angle" and candidate.get("reference_axis"):
        result.update(reference_axis=candidate["reference_axis"], angle_mode="unsigned",
                      required=True, nominal_source="source_ocr", source_arrow_verified=True)
    return result


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
        if candidate["kind"] == "angle" and candidate.get("reference_axis"):
            if (candidate["reference_axis"] not in {"horizontal", "vertical"} or len(candidate["entities"]) != 1
                    or graph_entities[candidate["entities"][0]]["type"] != "LINE" or candidate["nodes"]):
                return "axis_angle_requires_one_line"
            observed = candidate.get("evidence", {}).get("angle_observation", {})
            if not (observed.get("verified") and observed.get("reference_axis") == candidate["reference_axis"]
                    and observed.get("record_id") == candidate["record_id"] and observed.get("nominal") == candidate["value"]
                    and observed.get("evidence", {}).get("reference_arrow", {}).get("verified")
                    and observed.get("evidence", {}).get("target_arrow", {}).get("verified")
                    and any(t.get("entity_id") == candidate["entities"][0] and t.get("whole_line_supported")
                            for t in observed.get("target_candidates", []))):
                return "source_angular_arrows_not_verified"
        if candidate.get("evidence",{}).get("source_text",{}).get("symbol_confusion"):return "source_symbol_confusion_requires_confirmation"
        feasibility = candidate.get("evidence", {}).get("whole_primitive_radius")
        if feasibility is not None and feasibility.get("passed") is not True:
            return feasibility.get("status", "whole_primitive_radius_unverified")
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
        groups[(constraint["kind"],tuple(constraint["entities"]),tuple(constraint["nodes"]),constraint.get("reference_axis"))].append((constraint,decision))
    constraints=[]
    for group in groups.values():
        values={float(constraint["value"]) for constraint,_ in group}
        if len(values)>1:
            owner=_source_owned_angle_conflict(group,candidates,graph,model)
            if owner is None:
                for _,decision in group:decision.update(accepted=False,reason="conflicting_constraints")
            else:
                for constraint,decision in group:
                    if decision is owner:
                        constraints.append(constraint)
                    else:
                        decision.update(accepted=False,reason="ambiguous_angle_arrow_targets_another_finite_line")
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
            method=("source_constructed_fillet" if relation.get("source") == "source_bound_design_fillet"
                    else "local_source_fallback")
            relation_rows.append(({"relation_id":relation["id"]},method))
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
        if reason is None and relation.get("source") == "source_bound_design_fillet":
            if not any(row.get("kind") == "radius" and
                       row.get("record_id") == relation.get("construction_record_id") and
                       row.get("entities") == [relation.get("construction_arc_entity_id")]
                       for row in constraints):
                reason="constructed_fillet_radius_not_independently_bound"
        if (reason is None and method=="local_source_fallback" and use_api and receipt.get("schema_success")
                and selected["relation_id"] in allowed_relations):
            reason="provider_abstained_from_sent_relation"
        source=("source_bound_design_fillet" if relation and relation.get("source") == "source_bound_design_fillet"
                else "source_geometry")
        decision={**selected,"source":source,"admission_method":method,
                  "accepted":reason is None,"reason":reason,"evidence":relation.get("evidence") if relation else None}
        decisions.append(decision)
        if reason is None:
            relation_proposals.append(({"id":"k"+relation["id"],"kind":relation["type"],"record_id":None,"entities":relation["entities"],
                                        "nodes":relation["nodes"],"value":None,"source":"source_geometry","required":False,
                                        "evidence_class":(relation.get("evidence") or {}).get("evidence_class"),
                                        "independent_source_tangent_measurement":(relation.get("evidence") or {}).get("independent_source_tangent_measurement"),
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
                      "constructed_fillet_tangent_accepted":sum(row["accepted"] and row["source"]=="source_bound_design_fillet" for row in decisions),
                      "structural_local_accepted":sum(row["accepted"] and row.get("admission_method")=="local_source_fallback" for row in decisions),
                      "structural_api_accepted":sum(row["accepted"] and row.get("admission_method")=="api_and_source" for row in decisions),
                      "rejected":len(rejected),"bound_source_records":len(bound),
                      "bound_angle_records":sum(c["kind"] == "angle" for c in constraints),
                      "unbound_dimensions":inventory["counts"]["recognized_dimensions"]-len(bound),"constraints":len(constraints)},
            "units":inventory["units"],"inventory_artifact":inventory["artifacts"]["inventory"],"topology_artifact":inventory["artifacts"]["topology"],
            "issues":sorted({row["reason"] for row in rejected}),"ground_truth_used":False,"dimensions_verified":False,
            "annotation_diagnostics":inventory.get("annotation_diagnostics",[]),
            "angle_source_observations":inventory.get("angle_source_observations",[]),
            "radius_binding_coverage":radius_binding_coverage(inventory, graph, constraints, decisions),
            "constructed_radius_priors":constructed_priors,
            "scope":"Only independently supported source bindings; unbound dimensions and global constraint completeness remain unverified."}
    _write(Path(output_dir)/"constraint-bindings.json",result)
    return result
