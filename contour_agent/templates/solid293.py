"""Dimension-driven 293 template, authored from the declared calibration case.

Only topology and seven *tangent directions* are calibrated shape priors.
No endpoint or centre coordinate table is stored.  Every endpoint, centre and
straight length is recomputed analytically from the supplied dimensions.

The upper/lower chains have respectively 3/4 unmeasured degrees of freedom.
Freezing the seven join directions makes the construction deterministic; it
does not turn these priors into measured drawing dimensions.  The unlabeled
transition radius and the simplified LM tread also require explicit review.
"""

from copy import deepcopy

TEMPLATE_ID = "solid293-v1"
CALIBRATION_CASE = "solid-arrow-ping__img_000293"

# Authored once from the declared calibration profile's tangent orientations.
# These are shape assumptions, NOT held-out measurements or runtime GT access.
UPPER_JOIN_DIRECTIONS_DEG = (-10.8066747746, -43.2336291888, -27.0006655187)
LOWER_JOIN_DIRECTIONS_DEG = (164.6762898521, 144.7070586158, 136.6774922831, 158.5081103677)


def _parameter(pid, label, unit, value, minimum, maximum, hint, source="dimension"):
    return {"id": pid, "label": label, "unit": unit, "default": value,
            "min": minimum, "max": maximum, "source_kind": source,
            "required": True, "ocr_hint": hint}


PARAMETERS = [
    _parameter("d_left", "左侧实体面直径", "mm", 188, 10, 2000, "Ø188；排除Ø184粗加工线"),
    _parameter("d_shoulder", "左侧理论尖角直径", "mm", 274, 10, 2000, "Ø274"),
    _parameter("d_right", "右侧斜线端点直径", "mm", 710, 10, 3000, "Ø710"),
    _parameter("d_outer", "简化踏面外径", "mm", 840, 10, 4000, "Ø840"),
    _parameter("left_base", "左侧底面距基准高度", "mm", 25, 0, 500, "左下25；不是S1/S2厚度"),
    _parameter("left_height", "左侧实体面轴向高度", "mm", 178, 1, 1000, "178±1；原OCR可能错误"),
    _parameter("right_height", "右侧轴向高度", "mm", 135, 1, 1000, "135 +3/0"),
    _parameter("angle_left", "左侧斜线与竖直夹角", "deg", 12, 1, 35, "两处12°，需确认同值"),
    _parameter("angle_right", "右侧斜线与竖直夹角", "deg", 15, 1, 35, "两处15°，需确认同值"),
    _parameter("r_fillet", "左侧上下圆角半径", "mm", 5, 0.1, 100, "两处R5，需确认同值"),
    _parameter("r_upper_left", "上侧左过渡半径", "mm", 62, 0.1, 1000, "R62"),
    _parameter("r_upper_bend", "上侧中央弯曲半径", "mm", 180, 0.1, 2000, "R180"),
    _parameter("r_upper_right", "上侧右圆弧半径", "mm", 40, 0.1, 1000, "右上R40，独立绑定"),
    _parameter("r_lower_right", "下侧右圆弧半径", "mm", 40, 0.1, 1000, "右下R40，独立绑定"),
    _parameter("r_lower_bend", "下侧中央弯曲半径", "mm", 199, 0.1, 2000, "R199"),
    _parameter("r_lower_mid", "下侧中间过渡半径", "mm", 110, 0.1, 2000, "R110"),
    _parameter("r_lower_left_bend", "下侧左弯曲半径", "mm", 130, 0.1, 2000, "R130"),
    _parameter("r_lower_left", "下侧左圆弧半径", "mm", 40, 0.1, 1000, "左下R40，独立绑定"),
    _parameter("r_transition", "未标注上侧过渡圆弧估计半径", "mm", 124.354072397,
               0.1, 2000, "未标注；来自校准形状先验，不能当作OCR尺寸", "estimated"),
]

ASSUMPTIONS = [
    {"id": "template_topology", "required": True,
     "label": "确认本图适用293校准模板的21段拓扑；忽略局部孔/详图、剖面线、粗加工线及R3微小圆角"},
    {"id": "calibration_shape_priors", "required": True,
     "label": "确认7个未由尺寸确定的切线方向采用293校准形状先验，并核准未标注过渡半径；这不是唯一尺寸解"},
    {"id": "simplified_tread", "required": True,
     "label": "确认LM踏面缺少完整定义，本次以外径处直线简化，不能作为完整踏面制造轮廓"},
    {"id": "nominal_dimensions", "required": True,
     "label": "确认使用名义尺寸；公差、粗加工尺寸、测量点及δ1/δ2不作为本模板驱动约束"},
]


def template_schema():
    return deepcopy({
        "id": TEMPLATE_ID, "name": "293 回转剖面 · 尺寸驱动校准模板",
        "calibration_case": CALIBRATION_CASE,
        "description": "21实体解析切线链；参数重建全部圆心/切点。仅293为声明校准；新拓扑需另建模板。",
        "version": "1.0.0", "parameters": PARAMETERS, "assumptions": ASSUMPTIONS,
        "shape_priors": {
            "source_kind": "calibration",
            "source": "293 mainprofile-GT-v4 declared calibration, tangent-direction authorship only",
            "upper_join_directions_deg": list(UPPER_JOIN_DIRECTIONS_DEG),
            "lower_join_directions_deg": list(LOWER_JOIN_DIRECTIONS_DEG),
            "unmeasured_degrees_of_freedom": {"upper": 3, "lower": 4},
            "engineering_acceptance": False,
        },
    })
