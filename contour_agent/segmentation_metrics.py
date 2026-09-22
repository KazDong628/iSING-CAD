"""NumPy/OpenCV mask agreement metrics and reproducible annotation noise.

Metrics measure agreement with the supplied binary target. A weak or pseudo
target does not turn these metrics into independent truth or CAD accuracy.
Noise augmentation darkens a copy of the source image and never changes a mask.
"""
from __future__ import annotations

import math

import cv2
import numpy as np


def _binary(mask, name):
    array = np.asarray(mask)
    if array.ndim != 2 or not array.size:
        raise ValueError(f"{name} must be a nonempty two-dimensional binary mask")
    if not (np.issubdtype(array.dtype, np.number) or array.dtype == np.bool_) or np.iscomplexobj(array):
        raise ValueError(f"{name} must contain real binary values")
    if not np.isfinite(array).all() or not np.all((array == 0) | (array == 1) | (array == 255)):
        raise ValueError(f"{name} must contain binary 0/1 or 0/255 values, not probabilities")
    return array != 0


def _boundary(mask):
    # Image-border foreground has a boundary against the outside background.
    eroded = cv2.erode(mask.astype(np.uint8), np.ones((3,3), np.uint8),
                       borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return mask & (eroded == 0)


def segmentation_metrics(pred_binary, target_binary, tolerance_px=3):
    """Return region and symmetric boundary agreement without importing torch.

    Both masks empty: IoU, Dice, precision/recall and boundary F1 are 1, boundary
    distances are 0. Exactly one empty: all overlap/boundary scores are 0 and
    boundary distances are None (undefined). ``empty_case`` exposes this rule
    so an evaluator can report empty cases separately instead of inflating a
    foreground result. Nonempty boundaries use Euclidean pixel-centre distance.
    """
    prediction, target = _binary(pred_binary, "pred_binary"), _binary(target_binary, "target_binary")
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target must have identical shapes")
    if isinstance(tolerance_px, (bool, np.bool_)) or not math.isfinite(tolerance_px) or tolerance_px < 0:
        raise ValueError("tolerance_px must be finite and nonnegative")
    predicted_count, target_count = int(prediction.sum()), int(target.sum())
    intersection = int(np.count_nonzero(prediction & target))
    union = predicted_count+target_count-intersection
    pred_boundary, target_boundary = _boundary(prediction), _boundary(target)
    pred_boundary_count, target_boundary_count = int(pred_boundary.sum()), int(target_boundary.sum())
    empty_case = "both_empty" if not union else "empty_prediction" if not predicted_count else "empty_target" if not target_count else "none"
    if empty_case == "both_empty":
        boundary_precision = boundary_recall = boundary_f1 = 1.
        assd = hd95 = hausdorff = 0.
    elif empty_case != "none":
        boundary_precision = boundary_recall = boundary_f1 = 0.
        assd = hd95 = hausdorff = None
    else:
        target_distance = cv2.distanceTransform((~target_boundary).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        prediction_distance = cv2.distanceTransform((~pred_boundary).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        forward = target_distance[pred_boundary]
        backward = prediction_distance[target_boundary]
        boundary_precision = float(np.mean(forward <= tolerance_px))
        boundary_recall = float(np.mean(backward <= tolerance_px))
        denominator = boundary_precision+boundary_recall
        boundary_f1 = 2*boundary_precision*boundary_recall/denominator if denominator else 0.
        assd = float((forward.mean()+backward.mean())/2)
        hd95 = float(max(np.percentile(forward,95), np.percentile(backward,95)))
        hausdorff = float(max(forward.max(), backward.max()))
    return {"iou": intersection/union if union else 1.,
            "dice": 2*intersection/(predicted_count+target_count) if union else 1.,
            "pixel_precision": intersection/predicted_count if predicted_count else float(not target_count),
            "pixel_recall": intersection/target_count if target_count else float(not predicted_count),
            "boundary_precision": boundary_precision, "boundary_recall": boundary_recall,
            "boundary_f1": boundary_f1, "boundary_tolerance_px": float(tolerance_px),
            "average_symmetric_boundary_distance_px": assd,
            "hausdorff95_px": hd95, "hausdorff_distance_px": hausdorff,
            "predicted_foreground_pixels": predicted_count, "target_foreground_pixels": target_count,
            "intersection_pixels": intersection, "union_pixels": union,
            "predicted_boundary_pixels": pred_boundary_count, "target_boundary_pixels": target_boundary_count,
            "empty_case": empty_case,
            "boundary_definition": "foreground minus 3x3 erosion, outside image treated as background",
            "distance_definition": "Euclidean pixel centres; ASSD averages directed means; HD95 is the maximum directed 95th percentile",
            "scope": "Agreement with the supplied target only. Weak/pseudo target agreement is not independent truth, dimensional accuracy or manufacturing certification."}


def add_annotation_noise(image_rgb, seed, mask=None, strength=1.0):
    """Return a reproducible RGB copy with thin annotation-like interference.

    Input is uint8 HWC with three RGB channels. ``strength`` is in [0,2]. Up to
    min(12%, 6%*strength) of image pixels may be darkened; complete strokes are
    skipped when the budget is exhausted. Noise never lightens existing ink.
    An optional binary mask restricts additional parallel hatch strokes only;
    dimension lines, arrowheads and short ASCII labels may cross the whole
    image. The target mask itself is never altered or returned as a new target.
    """
    image = np.asarray(image_rgb)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or not image.shape[0] or not image.shape[1]:
        raise ValueError("image_rgb must be a nonempty uint8 HWC RGB image")
    if isinstance(strength, (bool, np.bool_)) or not math.isfinite(strength) or not 0 <= strength <= 2:
        raise ValueError("strength must be finite and in [0,2]")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    target = _binary(mask, "mask") if mask is not None else None
    height, width = image.shape[:2]
    if target is not None and target.shape != (height,width):
        raise ValueError("Optional mask must have the same spatial shape as image_rgb")
    result = image.copy()
    if strength == 0:
        return result
    rng = np.random.default_rng(int(seed))
    ink = np.zeros((height,width), np.uint8)
    budget = int(height*width*min(.12,.06*strength))
    short = min(height,width)
    thickness = max(1,min(2,round(short/900)))

    def merge(layer):
        nonlocal ink
        candidate = np.maximum(ink,layer)
        changed = np.any(image > (255-candidate)[...,None], axis=2)
        if int(changed.sum()) > budget:
            return False
        ink = candidate
        return True

    dimension_count = max(1,min(12,round((2+short/230)*strength)))
    for _ in range(dimension_count):
        horizontal = bool(rng.integers(0,2))
        extent, cross_extent = (width,height) if horizontal else (height,width)
        lo = int(rng.uniform(.04,.48)*extent)
        hi = min(extent-1,lo+max(5,int(rng.uniform(.17,.46)*extent)))
        cross = min(cross_extent-1,int(rng.uniform(.08,.92)*cross_extent))
        extension = max(2,min(18,round(short*.025)))
        arrow = max(2,min(12,round(short*.014)))
        shade = int(rng.integers(160,236))
        layer = np.zeros_like(ink)
        def point(along, across):
            return (int(along),int(across)) if horizontal else (int(across),int(along))
        cv2.line(layer,point(lo,cross),point(hi,cross),shade,thickness,cv2.LINE_AA)
        for endpoint, direction in ((lo,1),(hi,-1)):
            cv2.line(layer,point(endpoint,cross-extension),point(endpoint,cross+extension),shade,thickness,cv2.LINE_AA)
            tip = point(endpoint,cross)
            for side in (-1,1):
                cv2.line(layer,tip,point(endpoint+direction*arrow,cross+side*max(1,arrow//3)),shade,thickness,cv2.LINE_AA)
        merge(layer)

    text_count = max(1,min(12,round((2+short/300)*strength)))
    font_scale = float(np.clip(short/900,.28,.9))
    for _ in range(text_count):
        nominal = int(rng.integers(5,401))
        text = str(rng.choice([str(nominal),f"R{nominal}",f"{nominal}+/-1","A","B","Ra3.2","1:2"]))
        text_width, text_height = cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,font_scale,thickness)[0]
        x = int(rng.integers(0,max(1,width-text_width)))
        y = int(rng.integers(min(height-1,text_height+1),max(min(height-1,text_height+1)+1,height)))
        layer = np.zeros_like(ink)
        cv2.putText(layer,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,font_scale,int(rng.integers(165,226)),thickness,cv2.LINE_AA)
        merge(layer)

    if target is not None and target.any():
        angle = math.radians(float(rng.choice([30,45,60,120,135,150])))
        ys, xs = np.indices((height,width), dtype=np.float32)
        offset = -math.sin(angle)*xs+math.cos(angle)*ys
        pitch = max(8.,short*.045/max(.5,strength))
        phase = float(rng.uniform(0,pitch))
        shade = int(rng.integers(110,176))
        # Increase pitch if the complete hatch pattern would exceed the budget.
        # This keeps parallel strokes intact instead of deleting random pixels.
        for _ in range(6):
            layer = ((np.mod(offset-phase,pitch) < thickness) & target).astype(np.uint8)*shade
            if merge(layer):
                break
            pitch *= 1.7
    np.minimum(image,(255-ink)[...,None],out=result)
    return result
