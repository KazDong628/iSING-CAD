"""Bounded exact ARC-LINE-ARC construction hypotheses from source nominals."""
from __future__ import annotations

import itertools
import math
import numpy as np


def tangent_arc_line_arc(points, observation, radii, clockwise, tolerance, *, source_line_validator=None):
    """Keep outer endpoints; solve contacts analytically for a trial line offset.

    Only the line offset is searched. Radii and angle are source nominals,
    and both contact incidences and tangencies hold by construction. This
    returns a proposal, never a binding or permission to publish it.
    """
    from scipy.optimize import minimize_scalar
    from .annotation_line_support import _source_nominal_line, _dense_path
    from .topology_editing import _sample_entities, _source_polyline_distance
    from .topology import _primitive_distance
    points=np.asarray(points,float);r=np.asarray(radii,float)
    if (points.ndim!=2 or points.shape[1]!=2 or len(points)<7 or not np.isfinite(points).all()
            or r.shape!=(2,) or not np.isfinite(r).all() or np.any(r<=0)
            or len(clockwise)!=2 or any(type(v) is not bool for v in clockwise)
            or not math.isfinite(tolerance) or tolerance<=0):return None
    witness=np.asarray([observation["source_line"]["start_px"],observation["source_line"]["end_px"]],float)
    line=_source_nominal_line(witness,observation,3*tolerance)
    if line is None:return None
    d=np.asarray(line["end"])-line["start"];d/=np.linalg.norm(d)
    if np.dot(points[-1]-points[0],d)<0:d=-d
    n=np.array([-d[1],d[0]]);anchors=points[[0,-1]]
    u=anchors@d;v=anchors@n;c0=float(np.mean(witness@n))
    dense=_dense_path(points,tolerance);best=None
    def materialize(c,signs,roots):
        height=c+np.asarray(signs)*r-v;radicand=r*r-height*height
        if np.min(radicand)<=0:return None
        stations=u+np.asarray(roots)*np.sqrt(radicand)
        contacts=stations[:,None]*d+c*n
        if stations[1]-stations[0]<max(12.,4*tolerance):return None
        if max(np.linalg.norm(contacts-witness.mean(axis=0),axis=1))>np.linalg.norm(witness[1]-witness[0])+3*tolerance:return None
        arcs=[]
        for index,(a,b) in enumerate([(anchors[0],contacts[0]),(contacts[1],anchors[1])]):
            center=contacts[index]+signs[index]*r[index]*n
            angles=np.arctan2(np.array([a,b])[:,1]-center[1],np.array([a,b])[:,0]-center[0])
            sweep=float((angles[0]-angles[1])%(2*np.pi) if clockwise[index] else (angles[1]-angles[0])%(2*np.pi))
            if not .02<sweep<math.pi*1.8:return None
            arcs.append(dict(type="ARC",start=a.tolist(),end=b.tolist(),center=center.tolist(),
                             radius=float(r[index]),clockwise=clockwise[index],fit_error_px=0.))
        chain=[arcs[0],dict(type="LINE",start=contacts[0].tolist(),end=contacts[1].tolist(),
                           fit_error_px=0.,source_nominal_angle_constructed=True),arcs[1]]
        forward=np.min(np.array([_primitive_distance(dense,e) for e in chain]),axis=0)
        sampled=_sample_entities(chain,max_step_px=1.)[0]
        reverse=_source_polyline_distance(sampled,points)
        error=max(float(forward.max()),float(reverse.max()))
        if source_line_validator is not None and not source_line_validator(chain[1]):
            return None
        return error,chain
    # At most 16 analytic branches, 33 starting offsets and 48 scalar steps.
    for signs in itertools.product((-1.,1.),repeat=2):
        for roots in itertools.product((-1.,1.),repeat=2):
            low=max(c0-3*tolerance,*[v[i]-signs[i]*r[i]-r[i]+1e-6 for i in (0,1)])
            high=min(c0+3*tolerance,*[v[i]-signs[i]*r[i]+r[i]-1e-6 for i in (0,1)])
            if high<=low:continue
            valid=[]
            for c in np.linspace(low,high,33):
                out=materialize(c,signs,roots)
                if out is not None:valid.append((out[0],c,out[1]))
            if not valid:continue
            initial=min(valid,key=lambda row:row[0]);step=(high-low)/32
            # A raster support gate is discontinuous. Preserve the feasible
            # seed when scalar refinement steps outside the supported stroke.
            if source_line_validator is not None and (best is None or initial[0]<best[0]):
                best=(initial[0],initial[2])
            optimum=minimize_scalar(lambda c:(materialize(c,signs,roots) or (1e6,None))[0],
                bounds=(max(low,initial[1]-step),min(high,initial[1]+step)),method="bounded",
                options={"maxiter":48,"xatol":1e-8})
            out=materialize(optimum.x,signs,roots)
            if out and (best is None or out[0]<best[0]):best=out
    if best is None or best[0]>max(1.,1.5*tolerance):return None
    error,chain=best
    for entity in chain:entity["fit_error_px"]=error
    chain[1]["source_contact_construction"]={
        "method":"source_nominal_exact_arc_line_arc_v1","publication_approved":False,
        "radius_exact":True,"angle_exact":True,"contacts_exact":True,
        "outer_endpoints_fixed":True,"ground_truth_used":False,
        "original_outer_endpoints_px":anchors.tolist(),
        "constructed_contacts_px":[chain[1]["start"],chain[1]["end"]],
        "maximum_bidirectional_source_error_px":error,
        "source_line_support_checked_during_search":source_line_validator is not None,
        "maximum_analytic_branches":16,"maximum_initial_offsets_per_branch":33,
        "maximum_scalar_iterations_per_branch":48}
    return chain
