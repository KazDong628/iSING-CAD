"""Image -> measured scale -> connected CAD geometry, without a case template.

Artifact creation, visual agreement, physical scale and reference accuracy are
separate outcomes. No reference files or per-case numeric defaults are consumed.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import cv2
import ezdxf
import numpy as np
from PIL import Image
from shapely.geometry import Polygon
from .dimension_evidence import estimate_scale
from .ocr import canonical_records
from .raster import extract_main_profile
from .outline_fallback import extract_unhatched
from .vectorize import fit_polyline_with_diagnostics


def _write_json(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf8")


def build_automatic(image_path, document, output_dir, *, progress=None, segmentation_checkpoint=None,
                    segmentation_mask=None, segmentation_review=None):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    emit = progress or (lambda stage,message: None)
    review = segmentation_review or {}
    oracle_mask = bool(segmentation_mask and
                       (review.get("status") == "oracle_mask" or review.get("oracle_mask_conditioned") is True))
    gt_assisted = bool(segmentation_mask and review.get("ground_truth_used") is True)
    if segmentation_mask:
        emit("segment", "使用GT派生的材料掩膜开展条件性重建实验；不把它计为模型分割或人工复核。" if oracle_mask
             else "使用人工确认的材料掩膜继续构建；保留模型原始分割用于审计。")
        from .mask_geometry import extract_mask_profile
        with Image.open(image_path) as source_image:
            original_size = source_image.size
        with Image.open(segmentation_mask) as reviewed_image:
            reviewed = np.asarray(reviewed_image.convert("L")) >= 128
        extraction = extract_mask_profile(reviewed.astype(np.uint8), original_size, out / "reviewed-evidence")
        model = {}
        try:
            previous = json.loads((out / "learned-evidence/segmentation.json").read_text(encoding="utf8"))
            model = previous.get("model") if isinstance(previous, dict) else {}
        except (OSError, ValueError, TypeError):
            model = {}
        extraction.update(model={"label_status": "registered_dxf_gt_oracle_input"} if oracle_mask else
                          {"label_status": "registered_dxf_gt_edited_input"} if gt_assisted else model or {"label_status": "unknown"},
                          learned_segmentation=not oracle_mask, reviewed_segmentation=not oracle_mask,
                          oracle_mask_conditioned=oracle_mask, review=segmentation_review or {})
        extraction["issues"].append(
            "输入为GT派生的二值掩膜；本实验只考察掩膜条件下的CAD重建，不能作为盲测、模型分割精度或独立参考精度证据。"
            if oracle_mask else "GT派生掩膜已人工涂改；仍属于使用GT的开发实验，不能作为盲测或模型分割精度证据。"
            if gt_assisted else "材料掩膜已经人工检查；该确认仅针对像素分割，不证明尺寸或参考几何精度。")
        _write_json(out / "reviewed-evidence/segmentation.json", extraction)
    elif segmentation_checkpoint:
        emit("segment", "使用微调U-Net从原图预测零件材料区域，记录当前模型的训练标签来源。")
        from .segmentation import cached_extract
        extraction = cached_extract(segmentation_checkpoint, image_path, out / "learned-evidence")
    else:
        emit("segment", "从原图的剖面线与闭合区域自动提取材料外边界。")
        extraction = extract_main_profile(image_path, out / "evidence")
    if not extraction.get("polyline_px") and not segmentation_checkpoint and not segmentation_mask:
        emit("segment", "周期性剖面线不足，自动尝试非周期剖面区域提取。")
        primary_diagnostic = {"status": extraction["status"], "issues": extraction.get("issues", []),
                              "evidence": extraction.get("evidence", {})}
        extraction = extract_unhatched(image_path, document, out / "fallback-evidence")
        extraction["fallback_used"] = True
        extraction["primary_diagnostic"] = primary_diagnostic
    if not extraction.get("polyline_px"):
        raise ValueError("所选轮廓提取器未找到有效主轮廓，请检查当前模型的分割结果。")
    # Preserve the pre-fit source contour before measurement/vectorization can
    # fail. This evidence is separate from the red fitted-CAD overlay below.
    source=cv2.imdecode(np.fromfile(str(image_path),np.uint8),cv2.IMREAD_COLOR)
    if source is None:
        raise ValueError("无法读取主轮廓叠加图的原图。")
    display_scale=min(1.,2200/max(source.shape[:2]))
    source=cv2.resize(source,None,fx=display_scale,fy=display_scale)
    contour_overlay=source.copy()
    contour_points=np.rint(np.asarray(extraction["polyline_px"],float)*display_scale).astype(np.int32)
    cv2.polylines(contour_overlay,[contour_points],True,(210,90,0),max(2,round(max(source.shape[:2])/800)))
    cv2.imencode('.png',contour_overlay)[1].tofile(str(out/'contour-overlay.png'))
    emit("measure", "从标注文字和尺寸线端点自动估计比例，检查多个尺寸是否一致。")
    scale = estimate_scale(image_path, document)
    scaled = scale["status"] == "resolved"
    pixel_scale = float(scale["pixels_per_mm"]) if scaled else 1.0
    raw = np.asarray(extraction["polyline_px"], float)
    base_x, base_y = float(raw[:,0].min()), float(raw[:,1].max())
    if scaled and scale["axis"] == "x" and scale.get("diameter_semantics") == "half_section_radius_station":
        base_x = scale["axis_origin_px"]
    emit("vectorize", "沿源轮廓拟合直线与圆弧，优化连接点并合并冗余片段；使用原像素保真门检查结果。")
    fitting_tolerance = max(2., max(extraction["image_size"].values()) / 2200 * 1.7)
    try:
        fitting = fit_polyline_with_diagnostics(raw,source_polyline_px=extraction.get("raw_polyline_px") or raw,
                                                tolerance_px=fitting_tolerance,radius_records=canonical_records(document),
                                                pixels_per_mm=pixel_scale if scaled else None)
    except (ValueError,ArithmeticError) as error:
        _write_json(out/"curve-fit.json",{"passed":False,"method":"source-pixel-fidelity-gate-v1",
                    "error":str(error),"dimensions_verified":False,"reference_verified":False,
                    "engineering_certified":False,"source_contour_preserved":"contour-overlay.png"})
        raise
    fitted,curve_fit=fitting["entities"],fitting["quality"]
    _write_json(out/"curve-fit.json",curve_fit)
    def xy(p): return [(float(p[0])-base_x)/pixel_scale, (base_y-float(p[1]))/pixel_scale]
    entities=[]
    for entity in fitted:
        e={**entity,"start":xy(entity["start"]),"end":xy(entity["end"])}
        if entity["type"] == "ARC":
            e.update(center=xy(entity["center"]),radius=entity["radius"]/pixel_scale,clockwise=not entity["clockwise"])
        entities.append(e)
    if not entities:
        raise ValueError("提取边界未能形成有效的CAD实体。")
    doc=ezdxf.new("R2010"); doc.units=4 if scaled else 0
    doc.layers.new("AUTO_MAIN_PROFILE", dxfattribs={"color":3})
    modelspace=doc.modelspace()
    path_commands=[]
    sample_points=[]
    for index,e in enumerate(entities):
        if index==0:path_commands.append(f"M {e['start'][0]} {-e['start'][1]}")
        if e["type"]=="LINE":
            modelspace.add_line(e["start"],e["end"],dxfattribs={"layer":"AUTO_MAIN_PROFILE"})
            path_commands.append(f"L {e['end'][0]} {-e['end'][1]}")
            sample_points.append(e["start"])
        else:
            center=np.asarray(e["center"])
            a=math.atan2(e["start"][1]-center[1],e["start"][0]-center[0]);b=math.atan2(e["end"][1]-center[1],e["end"][0]-center[0])
            sweep=-((a-b)%(2*math.pi)) if e["clockwise"] else ((b-a)%(2*math.pi))
            modelspace.add_arc(e["center"],e["radius"],math.degrees(b if e["clockwise"] else a)%360,math.degrees(a if e["clockwise"] else b)%360,dxfattribs={"layer":"AUTO_MAIN_PROFILE"})
            path_commands.append(f"A {e['radius']} {e['radius']} 0 {int(abs(sweep)>math.pi)} {int(e['clockwise'])} {e['end'][0]} {-e['end'][1]}")
            theta=np.linspace(a,a+sweep,max(4,int(abs(sweep)*e["radius"]*pixel_scale/3)),endpoint=False)
            sample_points.extend((center+e["radius"]*np.column_stack([np.cos(theta),np.sin(theta)])).tolist())
    path_commands.append("Z")
    polygon=Polygon(sample_points)
    gaps=[math.dist(e["end"],entities[(i+1)%len(entities)]["start"]) for i,e in enumerate(entities)]
    issues=list(extraction.get("issues",[]))+list(scale.get("issues",[]))
    if curve_fit["fallback_used"]:
        issues.append("曲线拟合未满足源像素轮廓误差或拓扑要求，已保留原始闭合折线；这不修正分割错误，也不证明尺寸精度。")
    if curve_fit.get("optimization_rollback_used"):
        issues.append("更紧凑的拟合未通过原像素保真门，已保留重新验证通过的原分段；未放宽误差容限。")
    if curve_fit.get("ambiguous_radius_bindings"):
        issues.append("同一半径标注与多个源圆弧兼容，无法唯一绑定；冲突圆弧已保留像素拟合参数，未施加该标注。")
    geometry_valid=bool(polygon.is_valid and polygon.area>0 and max(gaps)<1e-7)
    material_connectivity=extraction.get("evidence",{}).get("connectivity")
    complete_material=material_connectivity is None or material_connectivity.get("complete_exterior_candidate") is True
    if not geometry_valid:issues.append("拟合后的轮廓存在自交、退化或未闭合连接。")
    doc.saveas(out/"drawing.dxf")
    readback=ezdxf.readfile(out/"drawing.dxf")
    readback_validation=_verify_dxf_readback(readback,entities,expected_units=4 if scaled else 0)
    readback_ok=readback_validation["passed"]
    if not readback_ok:issues.append("DXF回读实体的类型、坐标或圆弧参数与导出模型不一致。")
    bounds=polygon.bounds
    x0,y0,x1,y1=bounds; pad=max(x1-x0,y1-y0)*.05
    svg=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0-pad} {-y1-pad} {x1-x0+2*pad} {y1-y0+2*pad}"><title>自动图像主轮廓：原图证据拟合，非模板默认值</title><path d="{" ".join(path_commands)}" fill="#dbe9e2" stroke="#087f72" stroke-width="1.6" vector-effect="non-scaling-stroke"/></svg>'
    (out/"preview.svg").write_text(svg,encoding="utf8")
    # Fitted geometry overlay uses source image coordinates for direct inspection.
    mapped=np.array([[(p[0]*pixel_scale+base_x)*display_scale,(base_y-p[1]*pixel_scale)*display_scale] for p in sample_points],dtype=np.int32)
    cv2.polylines(source,[mapped],True,(0,0,220),max(2,round(max(source.shape[:2])/800)))
    cv2.imencode('.png',source)[1].tofile(str(out/'overlay.png'))
    validation={"passed":geometry_valid and readback_ok and curve_fit["passed"],"entity_count":len(entities),"max_gap_mm":max(gaps) if scaled else None,
                "material_connectivity":material_connectivity,"complete_material_exterior":complete_material,
                "self_intersection":not polygon.is_valid,"dxf_readback":readback_validation,"scaled_mm":scaled,
                "dimensions_verified":False,"reference_verified":False,"engineering_certified":False,
                "radius_bindings":[e["radius_binding"] for e in entities if e.get("radius_binding")],"curve_fit":curve_fit,"issues":issues,
                "meaning":"Connected CAD artifact validity only; dimensional constraints and reference accuracy are separate."}
    learned_algorithm = ("oracle-gt-mask-conditioned-v1" if oracle_mask else "gt-assisted-human-reviewed-mask-v1" if gt_assisted else
                         "human-reviewed-unet-mask-v1" if segmentation_mask else
                         ("unet-resnet18-dxf-supervised" if extraction.get("model",{}).get("label_status")=="registered_dxf_gt_experimental" else "unet-resnet18-weak-v1"))
    result={"mode":"autonomous_image","algorithm_version":learned_algorithm if (segmentation_checkpoint or segmentation_mask) else "hatch-dimension-vector-v3","automatic_completion":geometry_valid and readback_ok and complete_material,
            "complete_material_exterior":complete_material,"completion_class":"automatic_source_draft" if complete_material else "incomplete_material_exterior_draft",
            "manual_intervention":bool(segmentation_review) and not oracle_mask,"template_used":False,
            "ground_truth_used":gt_assisted or oracle_mask,"oracle_mask_conditioned":oracle_mask,
            "ground_truth_use":"raster_mask_input_only" if oracle_mask else "human_edited_gt_raster_mask" if gt_assisted else "none",
            "extraction":extraction,"scale":scale,
            "vectorization_algorithm":"source-supported-primitive-merge-dp-v2",
            "entities":entities,"validation":validation,"curve_fit":curve_fit,"bounds":{"min_x":x0,"min_y":y0,"max_x":x1,"max_y":y1},
            "coordinate_system":{"units":"mm" if scaled else "pixel","x":"image right","y":"image up",
                                 "origin_source_px":[base_x,base_y],"unit_assumption":"millimetres for unitless source dimensions" if scaled else None},
            "polyline_px":extraction["polyline_px"],"issues":issues,
            "fitted_polyline_px":[[p[0]*pixel_scale+base_x,base_y-p[1]*pixel_scale] for p in sample_points],
            "scope":"Automatically traced main material boundary with measured global scale and local radius bindings; not full dimension-constraint solving."}
    _write_json(out/"model.json",result);_write_json(out/"validation.json",validation);_write_json(out/"dimension-evidence.json",scale)
    emit("export", f"自动导出{len(entities)}个CAD实体；"+("毫米比例已从尺寸线求得。" if scaled else "比例尚未唯一确定，产物明确采用像素单位。"))
    return result


def _verify_dxf_readback(document, expected, *, expected_units, tolerance=1e-7):
    """Compare every saved primitive, including the CCW storage of CAD arcs."""
    actual = list(document.modelspace())
    checks = []
    for index, (saved, source) in enumerate(zip(actual, expected)):
        check = {"index": index, "id": source.get("id"), "type": source.get("type"), "passed": False}
        if saved.dxftype() != source.get("type") or source.get("type") not in {"LINE", "ARC"}:
            check["reason"] = "entity_type_mismatch"
            checks.append(check)
            continue
        try:
            def point_error(actual_point, expected_point):
                a = np.asarray(tuple(actual_point), dtype=float)
                b = np.asarray([*expected_point[:2], 0.], dtype=float)
                if a.shape != (3,) or not np.isfinite(a).all() or not np.isfinite(b).all():
                    raise ValueError("nonfinite coordinate")
                return float(np.linalg.norm(a-b))

            if source["type"] == "LINE":
                errors = [point_error(saved.dxf.start, source["start"]), point_error(saved.dxf.end, source["end"])]
            else:
                clockwise = bool(source["clockwise"])
                first, last = (source["end"], source["start"]) if clockwise else (source["start"], source["end"])
                radius = float(saved.dxf.radius)
                if not math.isfinite(radius) or radius <= 0:
                    raise ValueError("invalid radius")
                errors = [point_error(saved.dxf.center, source["center"]), abs(radius-float(source["radius"])),
                          point_error(saved.start_point, first), point_error(saved.end_point, last)]
                extrusion = np.asarray(tuple(saved.dxf.extrusion), dtype=float)
                if not np.isfinite(extrusion).all() or np.linalg.norm(extrusion-[0.,0.,1.]) > tolerance:
                    raise ValueError("nonplanar arc")
            maximum = max(errors)
            if not math.isfinite(maximum):
                raise ValueError("nonfinite geometry error")
            check.update(passed=maximum <= tolerance, max_geometry_error=maximum)
            if not check["passed"]:
                check["reason"] = "geometry_mismatch"
        except (KeyError, TypeError, ValueError, AttributeError, ArithmeticError):
            check["reason"] = "invalid_geometry"
        checks.append(check)
    count_matches = len(actual) == len(expected)
    units_match = document.units == expected_units
    return {"passed": bool(count_matches and units_match and len(checks) == len(expected) and all(c["passed"] for c in checks)),
            "entity_count_matches": count_matches, "units_match": units_match, "checked_entities": len(checks),
            "geometry_tolerance": tolerance, "geometry_tolerance_units": "mm" if expected_units == 4 else "pixel",
            "entities": checks}
