"""Read declared DXF package raster registrations for label preparation only.

Expressions are parsed as affine arithmetic, never evaluated as Python code.
These proposals are checked against the source by gt_registration, not accepted
as accurate merely because a package contains them. Runtime prediction does not
import this module.
"""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image


def affine_expression(expression, aliases):
    if not isinstance(expression, str) or len(expression) > 240:
        raise ValueError("Expected a short affine expression")
    tree = ast.parse(expression.replace("×", "*"), mode="eval")
    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return np.array([0., 0., float(node.value)])
        if isinstance(node, ast.Name) and node.id.lower() in aliases:
            return np.asarray(aliases[node.id.lower()], dtype=float)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        if isinstance(node, ast.BinOp):
            a, b = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add): return a+b
            if isinstance(node.op, ast.Sub): return a-b
            if isinstance(node.op, ast.Mult):
                if np.all(a[:2] == 0): return a[2]*b
                if np.all(b[:2] == 0): return b[2]*a
            if isinstance(node.op, ast.Div) and np.all(b[:2] == 0) and b[2] != 0:
                return a/b[2]
        raise ValueError("Registration expression must contain only affine arithmetic")
    result=visit(tree.body)
    if not np.isfinite(result).all(): raise ValueError("Nonfinite registration")
    return result.tolist()


def _walk(value, prefix="", depth=0):
    if not isinstance(value,dict) or depth>12:return
    yield prefix,value
    for k,v in value.items():
        if isinstance(v,dict):yield from _walk(v,prefix+"/"+str(k),depth+1)


def package_transforms(reference_source, source_image, *, dataset_root):
    """Return bounded declared affine proposals with sidecar hashes.

    reference_source is a relative DXF path or {'path': ..., 'member': ...}.
    ZIP-only packages simply use image registration without sidecar proposals.
    """
    if isinstance(reference_source,dict):reference_source=reference_source.get("path", "")
    root=Path(dataset_root).resolve();path=(root/str(reference_source)).resolve()
    if not path.is_relative_to(root) or path.suffix.lower()!=".dxf":return []
    with Image.open(source_image) as image:width,height=image.size
    factors={1.}
    # Package overlays sometimes use a resized copy of the same source sheet.
    # Their resolution only proposes a scale; source-image evidence decides.
    for overlay in path.parent.glob("*.png"):
        if not overlay.resolve().is_relative_to(root):continue
        if "overlay" not in overlay.name.lower():continue
        try:
            with Image.open(overlay) as im:
                if abs((im.width/im.height)/(width/height)-1)<.04:
                    factors.add(round(width/im.width,8))
        except (OSError,ValueError):pass
    proposals=[];seen=set()
    for sidecar in sorted(path.parent.glob("*.json")):
        if not sidecar.resolve().is_relative_to(root):continue
        if sidecar.stat().st_size>5_000_000 or "ocr" in sidecar.name.lower():continue
        try:raw=sidecar.read_bytes();doc=json.loads(raw.decode("utf-8-sig"))
        except (ValueError,OSError):continue
        if not isinstance(doc,dict):continue
        coordinates=str(doc.get("coordinate_system",{})).lower()
        radial_y=("axial" in str(doc.get("coordinate_system",{}).get("x","")).lower()
                  if isinstance(doc.get("coordinate_system"),dict) else False)
        dxf_minus_r="-r" in coordinates
        for location,record in _walk(doc):
            if not any(word in location.lower() for word in ("registration","transform","calibration","pixel_from","pixel_mapping","raster")):continue
            normalized={str(k).lower():v for k,v in record.items()}
            ex=next((normalized[k] for k in ("x_px","px_x","u_px","pixel_x","x") if isinstance(normalized.get(k),str)),None)
            ey=next((normalized[k] for k in ("y_px","px_y","v_px","pixel_y","y") if isinstance(normalized.get(k),str)),None)
            matrices=[]
            if ex and ey:
                x=np.array([1.,0.,0.]);y=np.array([0.,1.,0.])
                # r/z follow declared engineering axes; expression placement is
                # an additional proposal when old packages omit the axis map.
                aliases={"x":x,"x_mm":x,"y":y,"y_mm":y,"r":y if radial_y else x,"z":x if radial_y else y}
                if dxf_minus_r:aliases.update(r=-y,z=x)
                aliases.update(r_mm=aliases["r"],z_mm=aliases["z"])
                try:matrices.append([affine_expression(ex,aliases),affine_expression(ey,aliases)])
                except (ValueError,SyntaxError,TypeError):pass
                if not dxf_minus_r:
                    aliases.update(r=y,z=x,r_mm=y,z_mm=x)
                    try:matrices.append([affine_expression(ex,aliases),affine_expression(ey,aliases)])
                    except (ValueError,SyntaxError,TypeError):pass
            def number(keys,default=None):
                return next((float(normalized[k]) for k in keys if type(normalized.get(k)) in (int,float)),default)
            scale=number(("scale_px_per_mm","px_per_mm","k_px_per_mm","px_scale","scale"))
            sx=number(("sx_px_per_mm","scale_x_px_per_mm","sx"),scale)
            sy=number(("sy_px_per_mm","scale_y_px_per_mm","sy"),scale)
            x0=number(("x0_px","u0_px","x0","px_x0","x0_px_at_r0","x_axis_px"))
            y0=number(("y0_px","v0_px","y0","px_y0","y_bottom_rim_px","top_rim_y0_px","y_bottom_px","bottom_px"))
            if all(v is not None for v in (sx,sy,x0,y0)):matrices.append([[sx,0,x0],[0,-sy,y0]])
            for matrix in matrices:
                matrix=np.asarray(matrix,dtype=float)
                if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix[:,:2]))<1e-12:continue
                for factor in sorted(factors):
                    adjusted=matrix*factor;key=tuple(np.round(adjusted,6).flatten())
                    if key in seen:continue
                    seen.add(key)
                    proposals.append({"matrix":adjusted.tolist(),"source":sidecar.relative_to(root).as_posix()+"#"+location,
                                      "source_sha256":hashlib.sha256(raw).hexdigest(),"resolution_factor":factor})
                    if len(proposals)>=32:return proposals
    return proposals
