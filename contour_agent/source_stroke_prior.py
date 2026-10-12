"""Soft, source-only ink observations for the constrained numerical solve.

The drawing supplies observations, not CAD dimensions. These samples never
replace annotation equalities, the immutable mask budget or the final ink gate.
"""
from __future__ import annotations

import math
import cv2
import numpy as np

from .ocr import canonical_records
from .topology import _StrokeEvidence
from .parametric_solver import _sample_entity, _sweep


def _unique_ink_centres(evidence, pixels, normals, offsets, centre_budget):
    """Observe complete, unique dark strips, without choosing a CAD-nearest run."""
    points=pixels[:,None,:]+offsets[None,:,None]*normals[:,None,:]
    ij=np.rint(points*evidence.scale).astype(int)
    h,w=evidence.source_gray.shape
    inside=(ij[...,0]>=0)&(ij[...,0]<w)&(ij[...,1]>=0)&(ij[...,1]<h)
    x=np.clip(ij[...,0],0,w-1);y=np.clip(ij[...,1],0,h-1)
    dark=inside&(evidence.source_gray[y,x]<170)
    targets=pixels.copy();valid=np.zeros(len(pixels),bool)
    for i,occupied in enumerate(dark):
        starts=np.flatnonzero(occupied&~np.r_[False,occupied[:-1]])
        ends=np.flatnonzero(occupied&~np.r_[occupied[1:],False])
        options=[]
        for lo,hi in zip(starts,ends):
            if lo==0 or hi==len(offsets)-1 or offsets[hi]-offsets[lo]>max(7.,evidence.grid):continue
            gradients=[]
            for boundary,side in ((lo,-1),(hi,1)):
                indices=(np.arange(max(0,boundary-2),boundary+1) if side<0 else
                         np.arange(boundary,min(len(offsets),boundary+3)))
                gradient=np.c_[evidence.edge_gx[y[i,indices],x[i,indices]],
                               evidence.edge_gy[y[i,indices],x[i,indices]]]
                magnitude=np.linalg.norm(gradient,axis=1)
                unit=gradient[np.argmax(magnitude)]/max(float(magnitude.max()),1e-12)
                gradients.append(unit)
            # Discard crossing hatch directions before checking uniqueness.
            # Two parallel compatible runs remain ambiguous; proximity never
            # decides which track is the material boundary.
            if (all(abs(float(v@normals[i]))>=math.cos(math.radians(45.)) for v in gradients) and
                    float(gradients[0]@gradients[1])<=-.8):options.append((lo,hi))
        if len(options)!=1:continue
        lo,hi=options[0]
        if abs(float((offsets[lo]+offsets[hi])/2))>centre_budget:continue
        centre=pixels[i]+float((offsets[lo]+offsets[hi])/2)*normals[i]
        cx,cy=np.rint(centre*evidence.scale).astype(int)
        if not (0<=cx<w and 0<=cy<h) or evidence.interior_text_mask[cy,cx]:continue
        targets[i]=centre;valid[i]=True
    # The same original edge/dark-interior orientation checks remain required.
    _,supported=evidence.query(targets)
    return targets,valid&supported


def source_stroke_observation(image_path, document, baseline, graph, *, probe_entities=None):
    image = cv2.imdecode(np.fromfile(str(image_path), np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("source_image_unavailable")
    grid = float(graph.get("source_grid_pitch_px") or 1.)
    evidence = _StrokeEvidence(image, canonical_records(document), grid)
    ppm = float(baseline["scale"]["pixels_per_mm"]) if graph["units"] == "mm" else 1.
    origin = np.asarray(baseline["coordinate_system"]["origin_source_px"], float)
    budget = float((baseline.get("curve_fit") or {}).get("total_deviation_budget_px") or grid)
    # Search only the original input uncertainty tube, never the whole image.
    radius = min(8., max(grid, budget))
    offsets = np.linspace(-radius, radius, 2*int(math.ceil(radius*2))+1)
    from .reconstruction_priority import mask_is_hard
    source_first=not mask_is_hard(baseline)
    rows = []
    probes=graph['entities'] if probe_entities is None else probe_entities
    if (not isinstance(probes,list) or len(probes)!=len(graph['entities']) or
            [(e.get('id'),e.get('type')) for e in probes]!=
            [(e.get('id'),e.get('type')) for e in graph['entities']]):
        raise ValueError('stroke_probe_must_preserve_source_topology')
    for entity in probes:
        length=(math.dist(entity['start'],entity['end']) if entity['type']=='LINE'
                else entity['radius']*_sweep(entity))
        # Twenty-five samples on every primitive undersample long arcs while
        # oversampling tiny fillets. Cover the source path at a bounded spatial
        # density; keep the existing 65-sample schema/CPU ceiling. The weights
        # below still normalize by sample count and primitive arclength.
        count=int(np.clip(math.ceil(length*ppm/max(2.,2*grid)),25,65))
        points = _sample_entity(entity, count, midpoints=True)
        if not np.isfinite(points).all():raise ValueError('nonfinite_stroke_probe')
        pixels = points*np.array([ppm, -ppm])+origin
        tangent = np.gradient(pixels, axis=0)
        normal = np.c_[-tangent[:, 1], tangent[:, 0]]
        normal /= np.maximum(np.linalg.norm(normal, axis=1)[:, None], 1e-12)
        if source_first:
            # Observe both edges of a thick stroke even when a mask follows
            # one edge. A wider observation aperture does not authorize a
            # larger centre displacement or change any solve/publication gate.
            aperture=radius+max(3.5,grid/2)
            centre_offsets=np.linspace(-aperture,aperture,2*int(math.ceil(aperture*2))+1)
            targets,valid=_unique_ink_centres(evidence,pixels,normal,centre_offsets,radius)
        else:
            choices = []
            for shift in offsets:
                trial = pixels+shift*normal
                distances, supported = evidence.query(trial)
                cost = np.where(supported, abs(shift)+.02*distances, np.inf)
                choices.append((trial, cost))
            costs = np.asarray([value[1] for value in choices])
            best = np.argmin(costs, axis=0)
            valid = np.isfinite(costs[best, np.arange(len(points))])
            targets = np.asarray([choices[int(best[i])][0][i] for i in range(len(points))])
        # Unsupported samples stay absent. A crop closure without image ink is
        # not turned into invented source evidence.
        rows.append({"entity_id": entity["id"], "sample_count": len(points),
                     "indices": np.flatnonzero(valid).tolist(),
                     "target_points": ((targets[valid]-origin)/[ppm, -ppm]).tolist(),
                     "normals": (normal[valid]*[1., -1.]).tolist()})
    return {"schema_version": "original-image-stroke-observation-v1", "units": graph["units"],
            "source": "original_image_direction_compatible_ink", "reference_dxf_read": False,
            "hard_constraint": False, "search_radius_px": radius, "entities": rows,
            "probe_geometry":'current_numerical_candidate' if probe_entities is not None else 'original_problem_graph',
            "matching":'source_strip_centre_to_native_primitive' if source_first else 'sample_normal_displacement',
            "measurement_aperture_radius_px":radius+max(3.5,grid/2) if source_first else radius,
            "optimization_domain_recentered":False}


def stroke_prior(observation, graph, entities, scale):
    if observation is None:
        return None, {"enabled": False}
    if (not isinstance(observation, dict) or observation.get("units") != graph["units"] or
            observation.get("source") != "original_image_direction_compatible_ink" or
            observation.get("reference_dxf_read") is not False or observation.get("hard_constraint") is not False):
        raise ValueError("invalid_original_image_stroke_observation")
    lengths = np.array([math.dist(e["start"], e["end"]) if e["type"] == "LINE" else
                        e["radius"]*_sweep(e) for e in entities])
    shares = lengths/max(float(lengths.sum()), 1e-12)
    indices = {entity["id"]: i for i, entity in enumerate(entities)}
    groups = []
    seen = set()
    for row in observation.get("entities", []):
        eid, count = row.get("entity_id"), row.get("sample_count")
        if eid not in indices or eid in seen or type(count) is not int or not 3 <= count <= 65:
            raise ValueError("invalid_stroke_observation_entity")
        seen.add(eid)
        selected = np.asarray(row.get("indices"), int)
        targets, normals = np.asarray(row.get("target_points"), float), np.asarray(row.get("normals"), float)
        if not len(selected):
            continue
        if (selected.ndim != 1 or len(set(selected.tolist())) != len(selected) or
                np.any(selected < 0) or np.any(selected >= count) or
                targets.shape != (len(selected), 2) or normals.shape != targets.shape or
                not np.isfinite(targets).all() or not np.isfinite(normals).all() or
                not np.allclose(np.linalg.norm(normals, axis=1), 1., atol=1e-8)):
            raise ValueError("invalid_stroke_observation_samples")
        index = indices[eid]
        weight = math.sqrt(len(graph["nodes"])*shares[index]/count)/scale
        groups.append((index, count, selected, targets, normals, weight))
    primitive_matching=observation.get('matching')=='source_strip_centre_to_native_primitive'
    # Finite branches matter: an observed point elsewhere on the same circle
    # is not evidence for this ARC. Permit contact migration only across the
    # same shared topological node, never onto an unrelated contour interval.
    from .topology import _primitive_distance
    connected=[]
    for entity in entities:
        node_ids={entity.get('start_node'),entity.get('end_node')}-{None}
        connected.append([j for j,other in enumerate(entities) if other['id']==entity['id'] or
            node_ids.intersection({other.get('start_node'),other.get('end_node')}-{None})])
    def residuals(decoded):
        values=[]
        for index,count,selected,targets,normals,weight in groups:
            entity=decoded[index]
            if primitive_matching:
                value=np.minimum.reduce([_primitive_distance(targets,decoded[j]) for j in connected[index]])
            else:
                value=np.sum((_sample_entity(entity,count,midpoints=True)[selected]-targets)*normals,axis=1)
            values.append(value*weight)
        return np.concatenate(values) if values else np.zeros(0)
    return residuals, {"enabled": bool(groups), "source": observation["source"],
                       "sample_count": sum(len(row[2]) for row in groups),
                       "hard_constraint": False, "reference_dxf_read": False,
                       "method": ("arclength_weighted_source_strip_centre_to_native_primitive" if primitive_matching else
                                  "arclength_weighted_normal_ink_residual; no_tangential_sample_lock"),
                       "finite_primitive_matching": primitive_matching,
                       "contact_migration_scope": "own_and_shared_node_neighbors_only" if primitive_matching else None,
                       "entity_observed_fractions":{entities[index]['id']:len(selected)/count
                           for index,count,selected,targets,normals,weight in groups},
                       "annotation_equalities_and_final_gates_unchanged": True}
