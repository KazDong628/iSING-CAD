"""Conservative dimension parser and registered-layout bindings.

Numeric spelling corrections are deliberately not inferred from expected CAD values.
The 293 binding table is calibration metadata, not a cross-drawing recognizer.
"""
from __future__ import annotations
import math
import re
import unicodedata
from typing import Any

NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"

def parse_dimension(raw: str) -> dict:
    text = unicodedata.normalize("NFKC", str(raw)).replace(" ", "").replace("−", "-")
    out: dict[str, Any] = {"raw_text": str(raw), "kind": "unknown", "nominal": None,
                           "upper_deviation": None, "lower_deviation": None, "reference": False}
    if re.match(r"(?i)^ra", text):
        out["kind"] = "surface_finish"
        return out
    if re.fullmatch(r"[δΔ][₁₂12]?", text):
        out["kind"] = "symbol"
        return out
    if text.startswith("(") and text.endswith(")"):
        out["reference"] = True
        text = text[1:-1]
    prefix, unit = "", "mm"
    if text.startswith(("⌀", "Ø", "ø", "Φ", "φ", "∅")):
        prefix, kind, text = text[0], "diameter", text[1:]
    elif text.startswith(("R", "r")):
        prefix, kind, text = text[0], "radius", text[1:]
    else:
        kind = "length"
    if text.endswith(("°", "º")):
        kind, unit, text = "angle", "deg", text[:-1]
    match = re.fullmatch(rf"({NUMBER})(.*)", text)
    if not match:
        return out
    nominal = float(match.group(1))
    suffix = match.group(2)
    upper = lower = None
    if suffix:
        symmetric = re.fullmatch(rf"±({NUMBER})", suffix)
        asymmetric = re.fullmatch(rf"\^\{{?({NUMBER})\}}?_\{{?({NUMBER})\}}?", suffix)
        # An explicit diameter/radius marker disambiguates a compact unilateral
        # tolerance from an ordinary numeric range such as "20-25". Do not
        # recover partial strings, merged datum labels, or missing digits.
        unilateral = re.fullmatch(r"([+-])((?:\d+(?:\.\d*)?|\.\d+))", suffix) if prefix and kind in {"diameter", "radius"} else None
        if symmetric:
            magnitude = float(symmetric.group(1))
            if magnitude < 0:
                return out
            upper, lower = magnitude, -magnitude
        elif asymmetric:
            upper, lower = float(asymmetric.group(1)), float(asymmetric.group(2))
            if upper < lower:
                return out
        elif unilateral:
            deviation = float(unilateral.group(2))
            upper, lower = (deviation, 0.) if unilateral.group(1) == "+" else (0., -deviation)
        else:
            return out
    if not math.isfinite(nominal) or abs(nominal) > 1e6:
        return out
    if any(value is not None and (not math.isfinite(value) or abs(value) > 1e6) for value in (upper, lower)):
        return out
    out.update(kind=kind, nominal=nominal, unit=unit, upper_deviation=upper, lower_deviation=lower)
    return out

# Original 4170 x 2551 OCR centers. The association is deliberately authored once.
# Roles are not selected by matching the expected nominal value.
BINDINGS_293 = {
    "d_left": (384, 1670, "diameter"), "d_shoulder": (573, 228, "diameter"),
    "d_right": (1676, 2113, "diameter"), "d_outer": (1544, 2388, "diameter"),
    "left_base": (1028, 1907, "length"), "left_height": (82, 1220, "length"),
    "right_height": (3770, 1446, "length"), "angle_left": (1408, 1453, "angle"),
    "angle_right": (2956, 1318, "angle"), "r_fillet": (1286, 425, "radius"),
    "r_upper_left": (1558, 666, "radius"), "r_upper_bend": (2488, 875, "radius"),
    "r_upper_right": (2608, 1289, "radius"), "r_lower_right": (2526, 1700, "radius"),
    "r_lower_bend": None, "r_lower_mid": (1766, 1360, "radius"),
    "r_lower_left_bend": (1705, 1262, "radius"), "r_lower_left": (1544, 1248, "radius"),
    "r_transition": None,
}

def canonical_records(document: dict) -> list[dict]:
    records = []
    for index, item in enumerate(document.get("records", [])):
        row = dict(item)
        row["id"] = f"r{index:03d}"
        row["parsed"] = parse_dimension(row.get("text", ""))
        records.append(row)
    return records

def bind_parameters(document: dict, schema: dict, *, calibrated_layout: bool) -> tuple[list[dict], list[dict]]:
    records = canonical_records(document)
    size = document.get("meta", {}).get("original_size", {})
    width, height = float(size.get("width", 0)), float(size.get("height", 0))
    layout_valid = width > 0 and height > 0 and abs(width / height - 4170 / 2551) < 0.025
    rows = []
    for parameter in schema["parameters"]:
        pid = parameter["id"]
        row = dict(parameter)
        row.update(value=None, suggested_value=parameter.get("default"), source_record_id=None,
                   source_kind="unresolved", needs_review=True, note="需要明确的尺寸来源或人工输入。")
        binding = BINDINGS_293.get(pid)
        if binding is None:
            row["note"] = "原始 OCR 没有可唯一绑定的尺寸；建议值来自已声明的模板校准。"
            if parameter.get("source_kind") == "estimated":
                row["note"] = "未标注过渡形状参数；建议值为校准拟合，须明确确认。"
        elif layout_valid:
            x, y, expected_kind = binding
            candidates = []
            for rec in records:
                box = rec.get("box", [])
                if not box:
                    continue
                cx = sum(float(p[0]) for p in box) / len(box) / width
                cy = sum(float(p[1]) for p in box) / len(box) / height
                dist = math.hypot(cx - x / 4170, cy - y / 2551)
                if dist < .022:
                    candidates.append((dist, rec))
            candidates.sort(key=lambda item: item[0])
            if candidates:
                _, rec = candidates[0]
                row.update(source_record_id=rec["id"], raw_text=rec.get("text"), source_box=rec.get("box"))
                parsed = rec["parsed"]
                nominal = parsed.get("nominal")
                ambiguous = len(candidates) > 1 and candidates[1][0] - candidates[0][0] < .004
                lo, hi = parameter.get("min", -1e6), parameter.get("max", 1e6)
                if parsed["kind"] == expected_kind and nominal is not None and lo <= nominal <= hi and not ambiguous:
                    row.update(value=nominal, source_kind="ocr_registered_layout", needs_review=not calibrated_layout,
                               tolerance={"upper": parsed.get("upper_deviation"), "lower": parsed.get("lower_deviation")},
                               note="已注册版式的 OCR 尺寸绑定。" if calibrated_layout else "模板版式候选绑定，需核对原图位置。")
                else:
                    row["note"] = "OCR 类型、取值或位置存在歧义；不会用模板数字自动覆盖。"
        rows.append(row)
    return rows, records
