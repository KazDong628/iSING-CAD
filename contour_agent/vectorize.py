"""Fit source-derived polylines with connected native CAD lines and arcs.

Endpoints remain shared. Radius labels may snap an already matching local arc;
the source-record binding and pixel residual are retained, never hidden.
"""
from __future__ import annotations
import math
from bisect import bisect_left, bisect_right
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Polygon


def _line(points, tolerance):
    a, b = points[0], points[-1]
    delta = b - a
    length = np.linalg.norm(delta)
    if length < 1e-8:
        return None
    u = delta / length
    along = (points - a) @ u
    # A fit is a finite CAD segment, not its infinitely extended supporting
    # line. Collinear backtracking past an endpoint must retain its true error.
    nearest = a+np.clip(along, 0., length)[:, None]*u
    residual = np.linalg.norm(points-nearest, axis=1)
    if residual.max() > tolerance or np.any(np.diff(along) < -tolerance):
        return None
    return {"type": "LINE", "start": a.tolist(), "end": b.tolist(), "fit_error_px": float(residual.max())}


def _arc(points, tolerance):
    if len(points) < 4:
        return None
    a, b = points[0], points[-1]
    chord = b - a
    length = np.linalg.norm(chord)
    if length < tolerance * 2:
        return None
    middle = (a + b) / 2
    normal = np.array([-chord[1], chord[0]]) / length
    delta = points - middle
    projection = delta @ normal
    denom = 2 * float(projection @ projection)
    if denom < 1e-10:
        return None
    offset = float(projection @ (np.sum(delta * delta, axis=1) - length ** 2 / 4)) / denom
    center = middle + offset * normal
    radius = float(np.linalg.norm(center - a))
    # Co-circular polygon corners alone do not establish a source circular arc.
    interior = np.concatenate([points[:-1]*(1-f)+points[1:]*f for f in (.25,.5,.75)])
    support = np.vstack([points,interior])
    errors = np.abs(np.linalg.norm(support - center, axis=1) - radius)
    theta = np.unwrap(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]))
    sweep = float(theta[-1] - theta[0])
    if errors.max() > tolerance or not .035 < abs(sweep) < math.pi * 1.8:
        return None
    if np.any(np.diff(theta) * np.sign(sweep) < -.01):
        return None
    return {"type": "ARC", "start": a.tolist(), "end": b.tolist(), "center": center.tolist(),
            "radius": radius, "clockwise": sweep < 0, "fit_error_px": float(errors.max())}


def _bind_radius(choice, points, contour_points, radius_records, pixels_per_mm, tolerance_px):
    """Bind only the final selected local arc, using the same source tolerance."""
    pts = contour_points
    if choice["type"] == "ARC" and pixels_per_mm:
        fitted_mm = choice["radius"] / pixels_per_mm
        proposals = []
        for row in radius_records:
            value = row.get("parsed", {}).get("nominal")
            box = row.get("box")
            if row.get("parsed", {}).get("kind") != "radius" or not value or value <= 0 or not box:
                continue
            if abs(fitted_mm - value) > max(.5, .025 * value):
                continue
            label_center = np.mean(np.asarray(box, float), axis=0)
            distance = float(np.min(np.linalg.norm(points - label_center, axis=1)))
            if distance > max(80, np.ptp(pts, axis=0).max() * .18):
                continue
            a, b = points[0], points[-1]
            chord = b - a; chord_length = float(np.linalg.norm(chord))
            radius = value * pixels_per_mm
            if chord_length >= radius * 2:
                continue
            midpoint = (a+b)/2
            normal = np.array([-chord[1], chord[0]]) / chord_length
            offset = math.sqrt(radius**2 - (chord_length/2)**2)
            centers = [midpoint + offset*normal, midpoint - offset*normal]
            center = min(centers, key=lambda c: np.linalg.norm(c - choice["center"]))
            interior=np.concatenate([points[:-1]*(1-f)+points[1:]*f for f in (.25,.5,.75)])
            error = float(np.max(np.abs(np.linalg.norm(np.vstack([points,interior])-center, axis=1)-radius)))
            if error <= tolerance_px:
                proposals.append((error, distance, row, center, radius))
        if proposals:
            error, distance, row, center, radius = min(proposals, key=lambda p:(p[0],p[1]))
            choice.update(center=center.tolist(), radius=float(radius), fit_error_px=error,
                          radius_binding={"record_id": row["id"], "text": row["text"], "nominal": row["parsed"]["nominal"],
                                          "method": "local pixel arc fit plus compatible OCR radius", "label_distance_px": distance})
    return choice


def _merge_primitive_runs(pts, runs, tolerance_px, *, candidate_budget=20000):
    """Shortest path over source landmarks, refitting complete source support.

    The seed's 100-vertex search limit does not limit a merge: a candidate can
    span any number of source vertices or adjacent seed primitives. No source
    vertex is moved or discarded. A finite candidate budget bounds CPU use;
    every seed edge remains available even if the optional search is exhausted.
    """
    count = len(runs)
    boundaries = [runs[0][0]] + [run[1] for run in runs]
    arclength = np.r_[0., np.cumsum(np.linalg.norm(np.diff(pts,axis=0),axis=1))]
    landmarks = set(boundaries)
    # Permit old greedy endpoints to move to nearby ORIGINAL source vertices.
    # Quartiles use arclength, not vertex count, so dense straight runs do not
    # receive disproportionate influence. No smoothing/resampling is applied.
    for first,last,_ in runs:
        for fraction in (.25,.5,.75):
            target = arclength[first]+fraction*(arclength[last]-arclength[first])
            index = int(np.searchsorted(arclength,target))
            nearest = min((max(first,index-1),min(last,index)),key=lambda p:abs(arclength[p]-target))
            landmarks.add(nearest)
    landmarks = sorted(landmarks)
    candidates = {(first,last):entity for first,last,entity in runs}
    pairs=set();budget_exhausted=False
    def add_pair(first,last):
        pair=(first,last)
        if pair in candidates or pair in pairs:
            return True
        if len(pairs)>=candidate_budget:
            return False
        pairs.add(pair)
        return True
    # Local endpoint relocation over up to three adjacent seed spans. Longer
    # merges remain available between ANY seed endpoints, without a point cap.
    for i,first in enumerate(boundaries[:-1]):
        last = boundaries[min(i+3,count)]
        local = landmarks[bisect_left(landmarks,first):bisect_right(landmarks,last)]
        for k,a in enumerate(local):
            for b in local[k+1:]:
                if not add_pair(a,b):
                    budget_exhausted=True
                    break
            if budget_exhausted:break
        if budget_exhausted:break
    # Generate longer seed merges lazily: the budget bounds pair allocation as
    # well as fitting work, including pathologically fragmented input masks.
    if not budget_exhausted:
        for width in range(2,count+1):
            for first in range(count-width+1):
                if not add_pair(boundaries[first],boundaries[first+width]):
                    budget_exhausted=True
                    break
            if budget_exhausted:break
    pairs = sorted(pairs,key=lambda pair:(pair[1]-pair[0],pair[0]))
    tested = 0
    for first,last in pairs[:candidate_budget]:
        support = pts[first:last+1]
        candidate = _line(support, tolerance_px) or _arc(support, tolerance_px)
        tested += 1
        if candidate:
            candidates[first,last] = candidate
    # Minimize object count, then integrated squared source residual, then arc
    # count. The second objective avoids an arbitrary longest-prefix choice
    # among equally compact valid representations.
    costs = {point:(math.inf, math.inf, math.inf) for point in landmarks}
    costs[0] = (0, 0., 0)
    previous = {}
    incoming = {}
    for (first,last),entity in candidates.items():
        incoming.setdefault(last,[]).append((first,entity))
    for last in landmarks[1:]:
        for first,entity in incoming.get(last,[]):
            length = float(arclength[last]-arclength[first])
            cost = (costs[first][0]+1,
                    costs[first][1]+entity["fit_error_px"]**2*length,
                    costs[first][2]+(entity["type"]=="ARC"))
            if cost < costs[last]:
                costs[last] = cost
                previous[last] = (first, entity)
    merged, last = [], boundaries[-1]
    while last:
        first, entity = previous[last]
        merged.append((first,last,dict(entity)))
        last = first
    merged.reverse()
    evidence = {"method":"source-supported-primitive-merge-dp-v2",
                "seed_entity_count":count,"merged_entity_count":len(merged),
                "candidate_evaluations":tested,"candidate_budget":candidate_budget,
                "candidate_budget_exhausted":budget_exhausted,
                "landmark_count":len(landmarks),
                "optimization_scope":"shortest path over seed endpoints and nearby source arclength quartile vertices; no claim of global optimum over all source points",
                "source_points_moved":False}
    return merged,evidence


def fit_polyline(polyline_px, *, tolerance_px=2.0, radius_records=(), pixels_per_mm=None, optimize=True):
    pts = _closed_ring(polyline_px)[:-1]
    # Start at the most pronounced vertex so a smooth arc is not split by the seam.
    before, after = pts - np.roll(pts, 1, axis=0), np.roll(pts, -1, axis=0) - pts
    lengths = np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1)
    turns = np.arccos(np.clip(np.sum(before * after, axis=1) / np.maximum(lengths, 1e-12), -1, 1))
    start = int(np.argmax(turns))
    pts = np.roll(pts, -start, axis=0)
    pts = np.vstack([pts, pts[0]])
    runs, i = [], 0
    while i < len(pts) - 1:
        choice, end_index = _line(pts[i:i+2], tolerance_px), i + 1
        for j in range(i + 2, min(len(pts), i + 100)):
            current = _line(pts[i:j+1], tolerance_px) or _arc(pts[i:j+1], tolerance_px)
            if current:
                choice, end_index = current, j
            elif j > end_index + 7:
                break
        if choice is None:
            raise ValueError("Source segment cannot form a finite primitive")
        runs.append((i,end_index,choice))
        i = end_index
    if optimize:
        runs,evidence = _merge_primitive_runs(pts,runs,tolerance_px)
    else:
        evidence={"method":"unmerged_source_seed","seed_entity_count":len(runs),
                  "merged_entity_count":len(runs),"source_points_moved":False}
    entities=[];unbound=[]
    for first,last,choice in runs:
        unbound.append(dict(choice))
        choice = _bind_radius(choice,pts[first:last+1],pts,radius_records,pixels_per_mm,tolerance_px)
        choice["id"] = f"auto_{len(entities):03d}"
        choice["source_support_vertex_count"] = last-first+1
        entities.append(choice)
    bindings={}
    for index,entity in enumerate(entities):
        if entity.get("radius_binding"):
            bindings.setdefault(entity["radius_binding"]["record_id"],[]).append(index)
    ambiguous=[]
    for record_id,indices in bindings.items():
        if len(indices)<2:
            continue
        ambiguous.append({"record_id":record_id,"entity_ids":[entities[i]["id"] for i in indices],
                          "reason":"one_radius_annotation_is_compatible_with_multiple_source_arcs",
                          "action":"no_radius_snap_applied_to_conflicting_arcs"})
        for index in indices:
            entities[index].update(unbound[index])
            entities[index].pop("radius_binding",None)
    evidence["ambiguous_radius_bindings"]=ambiguous
    entities[0]["fitting_optimization"] = evidence
    return entities


def _closed_ring(polyline):
    points = np.asarray(polyline, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError("Source contour must contain finite Nx2 coordinates")
    # Repeated vertices do not define extra geometry and would make zero lines.
    points = points[np.r_[True, np.linalg.norm(np.diff(points,axis=0),axis=1)>1e-10]]
    if len(points) and np.linalg.norm(points[0]-points[-1]) <= 1e-10:
        points = points[:-1]
    if len(points) < 3:
        raise ValueError("Source contour has fewer than three distinct vertices")
    return np.vstack([points,points[0]])


def _line_entities(ring):
    return [{"id":f"auto_{i:03d}","type":"LINE","start":a.tolist(),"end":b.tolist(),
             "fit_error_px":0.,"source":"raw_contour_polyline_fallback"}
            for i,(a,b) in enumerate(zip(ring[:-1],ring[1:]))]


def _sample_entities(entities, max_step_px=.5, max_samples=250_000):
    """Sample ON every primitive with bounded arclength between neighbors."""
    samples, count, max_spacing, max_sagitta = [], 0, 0., 0.
    for entity in entities:
        a,b = np.asarray(entity["start"],float),np.asarray(entity["end"],float)
        if entity["type"] == "LINE":
            length=float(np.linalg.norm(b-a))
            n=max(1,int(math.ceil(length/max_step_px)))
            if count+n > max_samples:
                raise ValueError("Source fidelity audit exceeds the 250000 sample limit")
            sample=a+(b-a)*np.linspace(0,1,n,endpoint=False)[:,None]
            spacing=length/n
        elif entity["type"] == "ARC":
            center=np.asarray(entity["center"],float);radius=float(entity["radius"])
            if not np.isfinite(center).all() or not math.isfinite(radius) or radius <= 0:
                raise ValueError("Arc has invalid center or radius")
            if max(abs(np.linalg.norm(a-center)-radius),abs(np.linalg.norm(b-center)-radius)) > max(1e-7,radius*1e-10):
                raise ValueError("Arc endpoint does not lie on its declared circle")
            first=math.atan2(a[1]-center[1],a[0]-center[0]);last=math.atan2(b[1]-center[1],b[0]-center[0])
            sweep=-((first-last)%(2*math.pi)) if entity["clockwise"] else ((last-first)%(2*math.pi))
            length=abs(sweep)*radius;n=max(1,int(math.ceil(length/max_step_px)))
            if count+n > max_samples:
                raise ValueError("Source fidelity audit exceeds the 250000 sample limit")
            angles=first+sweep*np.linspace(0,1,n,endpoint=False)
            sample=center+radius*np.column_stack([np.cos(angles),np.sin(angles)])
            spacing=length/n
            max_sagitta=max(max_sagitta,radius*(1-math.cos(abs(sweep)/(2*n))))
        else:
            raise ValueError("Only finite LINE and ARC primitives can be audited")
        if not np.isfinite(sample).all():
            raise ValueError("Nonfinite fitted primitive coordinates")
        samples.append(sample);count+=n;max_spacing=max(max_spacing,spacing)
    if not samples:
        raise ValueError("No fitted primitives to audit")
    # Include last endpoint: covers open chains diagnostically without inventing
    # a connector. Chain closure is checked separately before acceptance.
    return np.vstack([*samples,np.asarray(entities[-1]["end"],float)]),max_spacing,max_sagitta


def _distance_evidence(first_samples, second_samples, first_spacing, second_spacing):
    first_to_second=cKDTree(second_samples).query(first_samples,workers=1)[0]
    second_to_first=cKDTree(first_samples).query(second_samples,workers=1)[0]
    sampled=max(float(first_to_second.max()),float(second_to_first.max()))
    # Every curve point is within half the maximum arclength sampling step of
    # a sampled point. The distance-to-a-set function is 1-Lipschitz. Samples
    # lie on the primitives, so these bounds also cover curved arcs exactly.
    uncertainty=max(first_spacing,second_spacing)/2
    return {"sampled_symmetric_hausdorff_px":sampled,
            "conservative_lower_bound_px":max(0.,sampled-uncertainty),
            "conservative_upper_bound_px":sampled+uncertainty,
            "sampling_uncertainty_bound_px":uncertainty,
            "source_to_fit_sample_max_px":float(first_to_second.max()),
            "fit_to_source_sample_max_px":float(second_to_first.max()),
            "source_to_fit_sample_p95_px":float(np.percentile(first_to_second,95)),
            "fit_to_source_sample_p95_px":float(np.percentile(second_to_first,95))}


def assess_fit_quality(source_polyline_px, entities, *, max_step_px=.5):
    """Measure vectorization fidelity to source pixels, never to a reference.

    Boundary distance is bidirectional with a conservative sampling upper
    bound, in original image pixels. Area overlap/topology use sampled arc
    chords and are diagnostic, not exact analytic curve-intersection proofs.
    """
    if not math.isfinite(max_step_px) or max_step_px <= 0:
        raise ValueError("Sampling step must be finite and positive")
    ring=_closed_ring(source_polyline_px)
    source_polygon=Polygon(ring)
    source,source_spacing,_=_sample_entities(_line_entities(ring),max_step_px)
    fitted,fit_spacing,sagitta=_sample_entities(entities,max_step_px)
    distances=_distance_evidence(source,fitted,source_spacing,fit_spacing)
    polygon=Polygon(fitted)
    gaps=[math.dist(e["end"],entities[(i+1)%len(entities)]["start"]) for i,e in enumerate(entities)]
    valid=bool(source_polygon.is_valid and source_polygon.area>0 and polygon.is_valid and polygon.area>0 and max(gaps)<=1e-7)
    area_iou=source_polygon.intersection(polygon).area/source_polygon.union(polygon).area if valid else None
    return {"source_boundary_deviation_px":distances,"area_iou":area_iou,
            "source_area_px2":float(source_polygon.area),"fitted_area_px2":float(polygon.area),
            "sampled_topology_valid":valid,"max_endpoint_gap_px":max(gaps),
            "arc_chord_max_sagitta_px":sagitta,"source_sample_count":len(source),"fitted_sample_count":len(fitted),
            "distance_method":"bidirectional_arclength_sampled_Hausdorff_with_Lipschitz_bound",
            "topology_method":"sampled_arc_chord_polygon_and_exact_chain_endpoint_gap"}


def fit_polyline_with_diagnostics(polyline_px, *, source_polyline_px=None,
                                  tolerance_px=2.0, radius_records=(), pixels_per_mm=None):
    """Fit native curves, enforcing a source-pixel fidelity budget.

    The budget is the existing fitting tolerance plus a measured conservative
    bound for any earlier contour simplification. A violating or invalid fit
    falls back to the original closed contour's straight segments. This changes
    representation only; it does not improve segmentation or certify dimensions.
    """
    if isinstance(tolerance_px,bool) or not math.isfinite(tolerance_px) or tolerance_px <= 0:
        raise ValueError("Fitting tolerance must be finite and positive")
    fit_input=_closed_ring(polyline_px)
    source=_closed_ring(source_polyline_px if source_polyline_px is not None else polyline_px)
    source_polygon=Polygon(source)
    if not source_polygon.is_valid or source_polygon.area<=0:
        raise ValueError("Raw source contour must be a valid nondegenerate closed polygon")
    if np.array_equal(fit_input,source):
        simplification_bound=0.
    else:
        simplification=assess_fit_quality(source,_line_entities(fit_input))
        simplification_bound=simplification["source_boundary_deviation_px"]["conservative_upper_bound_px"]
    budget=float(tolerance_px+simplification_bound)
    entities=fit_polyline(fit_input,tolerance_px=tolerance_px,radius_records=radius_records,pixels_per_mm=pixels_per_mm)
    optimization=entities[0].get("fitting_optimization") if entities else None
    attempted=None;fallback_reason=None
    try:
        attempted=assess_fit_quality(source,entities)
        if not attempted["sampled_topology_valid"]:
            fallback_reason="fitted_chain_or_sampled_topology_invalid"
        elif attempted["source_boundary_deviation_px"]["conservative_upper_bound_px"] > budget:
            fallback_reason="source_boundary_deviation_exceeds_pixel_budget"
    except (ValueError,ArithmeticError) as error:
        fallback_reason="fit_audit_failed"
        attempted={"audit_error":str(error)}
    rejected_optimization=None
    # A rejected compact representation must not displace a valid existing
    # seed fit. Audit that seed with EXACTLY the same budget and topology gate;
    # this is a representation rollback, never a relaxed acceptance threshold.
    if fallback_reason and optimization and optimization.get("method")=="source-supported-primitive-merge-dp-v2":
        rejected_optimization={"reason":fallback_reason,"entity_count":len(entities),"quality":attempted}
        seed=fit_polyline(fit_input,tolerance_px=tolerance_px,radius_records=radius_records,
                          pixels_per_mm=pixels_per_mm,optimize=False)
        try:
            seed_quality=assess_fit_quality(source,seed)
            seed_valid=(seed_quality["sampled_topology_valid"] and
                        seed_quality["source_boundary_deviation_px"]["conservative_upper_bound_px"]<=budget)
            rejected_optimization["seed_quality"]=seed_quality
        except (ValueError,ArithmeticError) as error:
            seed_valid=False
            rejected_optimization["seed_quality"]={"audit_error":str(error)}
        if seed_valid:
            entities=seed;attempted=seed_quality;fallback_reason=None
            rejected_optimization["action"]="retained_revalidated_unmerged_fit"
        else:
            rejected_optimization["action"]="raw_polyline_fallback_required"
    if fallback_reason:
        entities=_line_entities(source)
        # Exact same straight segments as the raw input: no sampled estimate is
        # needed to establish zero representation error for this fallback.
        final={"source_boundary_deviation_px":{"sampled_symmetric_hausdorff_px":0.,"conservative_lower_bound_px":0.,
               "conservative_upper_bound_px":0.,"sampling_uncertainty_bound_px":0.},
               "area_iou":1.,"source_area_px2":float(source_polygon.area),"fitted_area_px2":float(source_polygon.area),
               "sampled_topology_valid":True,"max_endpoint_gap_px":0.,"arc_chord_max_sagitta_px":0.,
               "distance_method":"identical_raw_polyline_segments","topology_method":"valid_original_polygon"}
    else:
        final=attempted
    quality={**final,"passed":True,"method":"source-pixel-fidelity-gate-v1","fallback_used":bool(fallback_reason),
             "fallback_reason":fallback_reason,"representation":"raw_polyline_lines" if fallback_reason else "fitted_lines_and_arcs",
             "tolerance_px":float(tolerance_px),"pre_fit_simplification_upper_bound_px":simplification_bound,
             "total_deviation_budget_px":budget,"raw_source_vertex_count":len(source)-1,
             "fit_input_vertex_count":len(fit_input)-1,"entity_count":len(entities),
             "line_count":sum(e["type"]=="LINE" for e in entities),"arc_count":sum(e["type"]=="ARC" for e in entities),
             "fitting_optimization":optimization,"rejected_optimization":rejected_optimization,
             "optimization_summary":{"candidate_accepted":bool(optimization and not rejected_optimization and not fallback_reason),
                 "seed_entity_count":optimization.get("seed_entity_count") if optimization else None,
                 "candidate_entity_count":optimization.get("merged_entity_count") if optimization else None,
                 "published_entity_count":len(entities),
                 "rejection_reason":rejected_optimization["reason"] if rejected_optimization else fallback_reason},
             "optimization_rollback_used":bool(rejected_optimization and not fallback_reason),
             "ambiguous_radius_bindings":entities[0].get("fitting_optimization",{}).get("ambiguous_radius_bindings",[]),
             "attempted_fit":attempted,"dimensions_verified":False,"reference_verified":False,"engineering_certified":False,
             "scope":"Representation fidelity to the selected source segmentation contour only; not segmentation correctness, dimension solving or reference accuracy."}
    return {"entities":entities,"quality":quality}
