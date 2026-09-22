"""Audit existing pilot DXFs against references; never generates CAD or calls APIs."""
from __future__ import annotations

import argparse
import base64
import csv
from datetime import datetime, timezone
import hashlib
import html
import io
import json
from pathlib import Path
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from contour_agent.config import Settings
from contour_agent.dataset import build_catalog
from contour_agent.autonomous_qualification import _reference_for_scoring
from contour_agent.dxf_comparison import compare_dxf_entities
from tools.run_segmentation_pilot import PILOT_CASE_IDS, RUN_PATTERN


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
    temporary.replace(path)


def _inside(root, relative):
    root = Path(root).resolve(); expected = root / relative
    resolved = expected.resolve(strict=True)
    if resolved != expected or not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("Comparison input must be a real file inside its declared directory")
    return resolved


def _load_summary(settings, path):
    runtime = settings.runtime_root.resolve()
    if path is None:
        pilot_root = runtime / "segmentation" / "pilot"
        pointer = json.loads(_inside(pilot_root, "latest.json").read_text(encoding="utf8"))
        path = _inside(pilot_root, pointer["report"])
        if pointer.get("report_sha256") != _sha(path): raise ValueError("Pilot latest report hash mismatch")
    else:
        requested = Path(path).absolute()
        path = requested.resolve(strict=True)
        if requested != path or not path.is_relative_to(runtime): raise ValueError("Pilot summaries must be inside runtime")
    report = json.loads(path.read_text(encoding="utf8"))
    if (report.get("status") != "completed" or not re.fullmatch(RUN_PATTERN, str(report.get("run_id", "")))
        or len(report.get("cases", [])) != 4 or {row.get("case_id") for row in report["cases"]} != set(PILOT_CASE_IDS)):
        raise ValueError("Expected a completed four-case pilot summary")
    return report, path


def _plot(result, destination, title):
    physical = result["alignment"]["coordinate_unit"] == "mm"
    figure, axis = plt.subplots(figsize=(11, 5.5))
    for side, color, zorder in (("reference", "#c44861", 1), ("prediction", "#087b8b", 2)):
        curves = result["visualization"][side]
        for index, curve in enumerate(curves):
            points = curve["points"]
            axis.plot([p[0] for p in points], [p[1] for p in points], color=color,
                      linewidth=1.4, alpha=.88, linestyle="--" if curve["assumption_layer"] else "-",
                      label=side if index == 0 else None, zorder=zorder)
            if side == "reference":
                point = points[len(points)//2]
                axis.annotate(f"r{index:04d}", point, color=color, fontsize=6, xytext=(3, 3), textcoords="offset points")
    if physical:
        for side in ("prediction", "reference"):
            worst = result["error_localization"].get(f"worst_{side}_curves", [])[:1]
            for row in worst:
                point = row["worst_point_mm"]
                axis.scatter(*point, marker="x", color="#292b28", s=45, zorder=4)
                axis.annotate(f'{row["entity_id"]}: {row["max_error_mm"]:.2f} mm', point, fontsize=8,
                              xytext=(7, -13), textcoords="offset points", color="#292b28")
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_title(title + ("\nD4 + translation; no scale fit; shape diagnostic" if physical else "\nINDEPENDENT BBOX NORMALIZATION - NO PHYSICAL SCALE ACCURACY"), fontsize=11)
    unit = "mm" if physical else "dimensionless (each file scaled separately)"
    axis.set_xlabel(unit); axis.set_ylabel(unit)
    axis.grid(alpha=.15); axis.legend(loc="best")
    figure.tight_layout(); figure.savefig(destination, dpi=150); plt.close(figure)


PARAMETER_FIELDS = ("side", "scope", "id", "type", "layer", "coordinate_unit", "source_unit_code", "length", "radius", "center_x", "center_y",
                    "start_x", "start_y", "end_x", "end_y", "line_angle_deg", "start_angle_deg", "end_angle_deg", "sweep_deg",
                    "start_node", "end_node", "assumption_layer", "vertices")


def _parameter_rows(result):
    for side in ("prediction", "reference"):
        for scope, entities in (("raw_modelspace", result[side]["raw_modelspace"]["entities"]),
                                ("filtered_profile", result[side]["filtered_profile"]["entities"])):
            for entity in entities:
                row = {key: entity.get(key) for key in PARAMETER_FIELDS}
                row.update(side=side, scope=scope, source_unit_code=result[side]["source_units"])
                for name in ("center", "start", "end"):
                    values = entity.get(name) or [None, None]
                    row[f"{name}_x"], row[f"{name}_y"] = values
                vertices = entity.get("vertices_xy_bulge", entity.get("vertices_xy"))
                row["vertices"] = json.dumps(vertices) if vertices is not None else None
                yield row


def _csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def _joint_rows(result):
    for side in ("prediction", "reference"):
        for node in result[side]["connections"]["nodes"]:
            yield {"side": side, "node_id": node["id"], "degree": node["degree"], "joint_type": node["joint_type"],
                   "coordinate_unit": result[side]["connections"]["coordinate_unit"], "maximum_snap_distance": node["maximum_snap_distance"],
                   "nearest_other_endpoint_distance": node["nearest_other_endpoint_distance"],
                   "interior_tangent_angle_deg": node["interior_tangent_angle_deg"],
                   "deviation_from_tangent_continuity_deg": node["deviation_from_tangent_continuity_deg"],
                   "intended_tangency_known": False, "entity_ids": "|".join(node["entity_ids"])}


def _summary_row(role, row):
    p, r = row["comparison"]["prediction"], row["comparison"]["reference"]
    score = row["comparison"]["physical_score"]; metrics = score.get("registered_metrics") or {}
    return {"role": role, "run_id": row["run_id"], "case_id": row["case_id"], "job_id": row.get("job_id"),
            "prediction_raw": p["raw_modelspace"]["count"], "reference_raw": r["raw_modelspace"]["count"],
            "prediction_filtered": p["filtered_profile"]["count"], "reference_filtered": r["filtered_profile"]["count"],
            "prediction_types": json.dumps(p["filtered_profile"]["types"], sort_keys=True),
            "reference_types": json.dumps(r["filtered_profile"]["types"], sort_keys=True),
            "prediction_units": p["source_units"], "reference_units": r["source_units"],
            "physical_comparison_status": score["status"], "max_error_mm": metrics.get("max_error_mm"),
            "p95_error_mm": metrics.get("p95_error_mm"), "rms_error_mm": metrics.get("rms_error_mm"),
            "length_error_mm": metrics.get("length_error_mm"), "reference_within_0_1mm": score["reference_within_0_1mm"],
            "unique_parameter_agreements": row["comparison"]["entity_correspondence"]["counts"].get("unique_full_parameter_agreement", 0)
            if row["comparison"]["entity_correspondence"]["status"] == "physical_diagnostic" else None}


def _changes(runs):
    if len(runs) < 2: return []
    before = {row["case_id"]: row for row in runs[0]["cases"]}; changes = []
    for after in runs[1]["cases"]:
        old = before[after["case_id"]]; p = old["comparison"]; q = after["comparison"]
        same_reference = p["reference"]["sha256"] == q["reference"]["sha256"]
        pm = p["physical_score"].get("registered_metrics") or {}; qm = q["physical_score"].get("registered_metrics") or {}
        baseline_count = p["prediction"]["filtered_profile"]["count"]
        candidate_count = q["prediction"]["filtered_profile"]["count"]
        comparable = same_reference and "max_error_mm" in pm and "max_error_mm" in qm
        changes.append({"case_id": after["case_id"], "same_reference_sha256": same_reference,
                        "prediction_file_changed": p["prediction"]["sha256"] != q["prediction"]["sha256"],
                        "baseline_filtered_entities": baseline_count, "candidate_filtered_entities": candidate_count,
                        "reference_filtered_entities": q["reference"]["filtered_profile"]["count"],
                        "baseline_raw_objects": p["prediction"]["raw_modelspace"]["count"],
                        "candidate_raw_objects": q["prediction"]["raw_modelspace"]["count"],
                        "reference_raw_objects": q["reference"]["raw_modelspace"]["count"],
                        "filtered_entity_count_change": candidate_count - baseline_count,
                        "filtered_entity_reduction_percent": 100 * (baseline_count - candidate_count) / baseline_count if baseline_count else None,
                        "max_error_change_mm": qm["max_error_mm"] - pm["max_error_mm"] if comparable else None,
                        "p95_error_change_mm": qm["p95_error_mm"] - pm["p95_error_mm"] if comparable else None,
                        "rms_error_change_mm": qm["rms_error_mm"] - pm["rms_error_mm"] if comparable else None,
                        "max_error_improved": qm["max_error_mm"] < pm["max_error_mm"] if comparable else None,
                        "meaning": "Parameter/entity count reduction is not a precision improvement. Error delta is reported only when both runs have physical metrics against identical GT bytes."})
    return changes


def _table(rows, keys):
    escape = lambda value: html.escape(str(value), quote=True)
    labels = {"case_id":"图纸", "baseline_raw_objects":"旧原始对象", "candidate_raw_objects":"新原始对象",
              "reference_raw_objects":"GT 原始对象", "baseline_filtered_entities":"旧主轮廓对象",
              "candidate_filtered_entities":"新主轮廓对象", "reference_filtered_entities":"GT 主轮廓对象",
              "filtered_entity_reduction_percent":"对象减少 (%)", "max_error_change_mm":"最大偏差变化 (mm)",
              "p95_error_change_mm":"P95 变化 (mm)", "rms_error_change_mm":"RMS 变化 (mm)",
              "max_error_improved":"最大偏差下降", "prediction_types":"预测对象类型", "reference_types":"GT 对象类型",
              "max_error_mm":"最大偏差 (mm)", "p95_error_mm":"P95 (mm)", "rms_error_mm":"RMS (mm)",
              "length_error_mm":"总长度差 (mm)", "reference_within_0_1mm":"完整参考 0.1 mm 通过",
              "unique_parameter_agreements":"完整参数唯一匹配数", "id":"对象编号", "type":"类型", "layer":"图层",
              "coordinate_unit":"坐标单位", "length":"长度", "radius":"半径", "center":"圆心",
              "start":"起点", "end":"终点", "start_angle_deg":"起始角 (°)", "end_angle_deg":"终止角 (°)",
              "sweep_deg":"扫角 (°)", "assumption_layer":"假设闭合层", "side":"对象来源",
              "node_id":"接头编号", "joint_type":"连接类型", "degree":"连接度",
              "maximum_snap_distance":"最大接头间隙", "deviation_from_tangent_continuity_deg":"切向差 (°)",
              "entity_ids":"相连对象", "prediction_id":"预测编号", "prediction_type":"预测类型",
              "status":"对应状态", "reference_id":"GT 编号", "fragment_candidate_ids":"可能覆盖的 GT 片段",
              "parameter_errors":"已匹配参数误差", "scope":"参考范围", "sample_count":"采样数"}
    values = {"prediction":"预测", "reference":"GT", "drawing_units_unspecified":"原绘图单位（未声明）",
              "original_drawing_units":"原绘图单位", "unmatched_geometry":"未匹配",
              "possible_fragment_coverage":"可能的局部覆盖", "unique_full_parameter_agreement":"完整参数唯一匹配",
              "ambiguous_multiple_full_candidates":"多个候选，存在歧义"}
    def value(value):
        if value is None: return "—"
        if isinstance(value, bool): return "是" if value else "否"
        if isinstance(value, float): return f"{value:.6g}"
        if isinstance(value, (list, dict)): return json.dumps(value, ensure_ascii=False)
        return values.get(str(value),str(value))
    return '<div class="table-wrap"><table><thead><tr>' + ''.join(f'<th>{escape(labels.get(key,key))}</th>' for key in keys) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{escape(value(row.get(key)))}</td>' for key in keys) + '</tr>' for row in rows) + '</tbody></table></div>'


def _html(report, output):
    esc = lambda value: html.escape(str(value), quote=True)
    cards = []
    for run in report["runs"]:
        for row in run["cases"]:
            result = row["comparison"]; p = result["prediction"]; r = result["reference"]
            data = base64.b64encode((output / row["files"]["overlay"]).read_bytes()).decode("ascii")
            unit_note = "毫米物理评分；对齐仅供形状诊断，不改动DXF" if result["alignment"]["coordinate_unit"] == "mm" else "未执行毫米参数对比：物理尺度未确定或配准不可用。图中两轮廓分别按包围盒归一化，不能用于毫米参数精度。"
            if not r["source_units"]:
                unit_note += " 参考 DXF 的 INSUNITS=0，原始长度/半径单位未知，不能标成毫米。"
            if not r["connections"]["closed"]:
                endpoints = [node["nearest_other_endpoint_distance"] for node in r["connections"]["nodes"] if node["nearest_other_endpoint_distance"] is not None]
                coordinate_unit = "mm" if r["connections"]["coordinate_unit"] == "mm" else "原绘图单位（非毫米）"
                distance = f"；未连接端点的最近其他端点距离最大为 {max(endpoints):.6g} {coordinate_unit}" if endpoints else ""
                unit_note += f' 参考主轮廓端点审计未闭合，有 {len(r["connections"]["components"])} 个连通分量{distance}。本工具保留取整缝隙和缺口，不补连参考线。'
            details = []
            for side, label in (("prediction", "预测"), ("reference", "参考")):
                for scope, scope_label in (("raw_modelspace", "原始对象参数"), ("filtered_profile", "过滤后主轮廓参数")):
                    keys = ("id", "type", "layer", "coordinate_unit", "length", "radius", "center", "start", "end", "start_angle_deg", "end_angle_deg", "sweep_deg", "assumption_layer")
                    details.append(f'<details><summary>{label} · {scope_label}（{result[side][scope]["count"]}）</summary>{_table(result[side][scope]["entities"], keys)}</details>')
            summary = _summary_row(run["role"], row)
            localization = result["error_localization"]
            if "reference_directed_scope" in localization:
                scopes = [{"scope": name, **values} for name, values in localization["reference_directed_scope"].items() if values]
                details.insert(0, '<details open><summary>最大误差定位与 core / closure 分开诊断</summary><p>以下 core / closure 是参考到预测的单向覆盖误差，不能替代上方完整双向评分，也不能单凭这张表断定是分割、定标还是参考简化导致。</p>' + _table(scopes, ("scope","max_error_mm","p95_error_mm","rms_error_mm")) + _table(localization["worst_reference_curves"], ("entity_id","type","layer","max_error_mm","worst_point_mm","parameters")) + _table(localization["worst_prediction_curves"], ("entity_id","type","max_error_mm","worst_point_mm","nearest_target_entity_ids_at_worst_point")) + '</details>')
            match_rows = [{**item, "parameter_errors": item["parameter_errors"]} for item in result["entity_correspondence"]["rows"]]
            cards.append(f'<article><h2>{esc(run["role"])} · {esc(row["case_id"])}</h2><p>Job {esc(row.get("job_id"))} · 预测原始/主轮廓 {p["raw_modelspace"]["count"]}/{p["filtered_profile"]["count"]}；参考 {r["raw_modelspace"]["count"]}/{r["filtered_profile"]["count"]}</p><p class="warning">{esc(unit_note)}</p><img src="data:image/png;base64,{data}" alt="Prediction and reference diagnostic overlay"><p>青色为预测，红色为参考；虚线为命名假设/闭合层。参考标签 r0000 等只用于查表，不表示与预测同序号对应。</p>{_table([summary], ("prediction_types","reference_types","max_error_mm","p95_error_mm","rms_error_mm","length_error_mm","reference_within_0_1mm","unique_parameter_agreements"))}{"".join(details)}<details><summary>连接与切向夹角（角点不自动认定为错误）</summary>{_table(list(_joint_rows(result)), ("side","node_id","joint_type","degree","maximum_snap_distance","deviation_from_tangent_continuity_deg","entity_ids"))}</details><details><summary>碎片、候选对应与未匹配实体</summary><p>只有双向唯一且完整参数支持的实体才给出参数误差；不按索引强制配对。</p>{_table(match_rows, ("prediction_id","prediction_type","status","reference_id","fragment_candidate_ids","parameter_errors"))}<pre>{esc(json.dumps({"unmatched_reference_ids":result["entity_correspondence"].get("unmatched_reference_ids"),"fragment_groups":result["entity_correspondence"].get("one_to_many_fragment_candidates")},ensure_ascii=False,indent=2))}</pre></details><details><summary>弧长、半径、弧角分布与参考范围</summary><pre>{esc(json.dumps({side:{"arc_diagnostics":result[side]["filtered_profile"]["arc_diagnostics"],"assumed_layers":result[side]["filtered_profile"]["assumed_layers"]} for side in ("prediction","reference")},ensure_ascii=False,indent=2))}</pre></details></article>')
    url = f'/api/dxf-comparison/{report["run_id"]}'
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DXF 实体与参数独立审计</title><style>body{{margin:0;background:#f4f1e9;color:#1c3430;font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif}}header{{padding:30px;background:#1c3430;color:white}}main{{max-width:1360px;padding:24px;margin:auto}}article{{background:white;border:1px solid #d6ded7;padding:22px;margin:22px 0}}img{{width:100%;max-height:620px;object-fit:contain}}h1{{margin:0}}h2{{overflow-wrap:anywhere}}.warning{{padding:12px;background:#fff4da;border-left:4px solid #ce9238}}.table-wrap{{overflow:auto;max-height:560px;margin:12px 0}}table{{border-collapse:collapse;font-size:12px;width:100%}}th,td{{border:1px solid #d5dfd7;padding:6px;white-space:nowrap}}th{{position:sticky;top:0;background:#eef4ee}}details{{margin:12px 0;border-top:1px solid #d5dfd7;padding-top:10px}}summary{{cursor:pointer;font-weight:600}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}}a{{display:inline-block;margin:6px 12px 6px 0;color:#08776a}}@media(max-width:650px){{main{{padding:8px}}article{{padding:12px}}}}</style></head><body><header><h1>DXF 实体与参数独立审计</h1><p>保留原始对象、过滤后主轮廓、连接关系与独立物理评分；不以实体数量判精度。</p></header><main><p class="warning">四图按已知效果选择，属于已暴露开发子集。全数据集分母 50，本工具审计 4 张，不能外推其余 46 张。固定 0.1 mm 阈值保持不变。归一化图不证明物理定标；参考 closure / tread 简化层仍包含在原评分中。</p><nav><a href="{url}/comparison-artifacts.zip">下载完整审计 ZIP</a><a href="{url}/comparison.json">JSON 全部参数</a><a href="{url}/comparison.csv">CSV 汇总</a></nav>{_table(report["changes"], ("case_id","baseline_raw_objects","candidate_raw_objects","reference_raw_objects","baseline_filtered_entities","candidate_filtered_entities","reference_filtered_entities","filtered_entity_reduction_percent","max_error_change_mm","p95_error_change_mm","rms_error_change_mm","max_error_improved")) if report["changes"] else ""}{''.join(cards)}</main></body></html>'''


def compare_pilot_runs(settings, *, baseline_summary=None, candidate_summary=None, output_root=None):
    """Compare existing predictions only; return ``(report, comparison_path)``."""
    baseline, baseline_path = _load_summary(settings, baseline_summary)
    selected = [("baseline", baseline, baseline_path)]
    if candidate_summary is not None:
        candidate, candidate_path = _load_summary(settings, candidate_summary)
        if candidate["run_id"] == baseline["run_id"]: raise ValueError("Before/after runs must have different identifiers")
        selected.append(("candidate", candidate, candidate_path))
    run_id = selected[-1][1]["run_id"]
    runtime = settings.runtime_root.resolve()
    destination = Path(output_root).resolve() if output_root else runtime / "dxf-comparison"
    if not destination.is_relative_to(runtime): raise ValueError("Comparison output must stay within runtime")
    output = destination / run_id
    output.mkdir(parents=True, exist_ok=True)
    if output.resolve() != output: raise ValueError("Comparison output must not be redirected")
    catalog = build_catalog(settings.dataset_root); cases = {case["id"]: case for case in catalog["cases"]}
    if len(cases) != 50: raise ValueError("Frozen 50-case catalog required")
    runs = []; packaged = []; summaries = []
    for role, summary, path in selected:
        rows = []
        for case in summary["cases"]:
            case_id = case["case_id"]
            prediction = _inside(path.parent, f"cases/{case_id}/drawing.dxf")
            expected_hash = case.get("artifacts", {}).get("drawing.dxf", {}).get("packaged_sha256")
            if not expected_hash or _sha(prediction) != expected_hash: raise ValueError("Pilot prediction artifact hash mismatch")
            reference, receipt = _reference_for_scoring(settings, cases[case_id], output / "reference-only" / role)
            if reference is None: raise ValueError(f"Reference is unavailable for declared pilot case {case_id}")
            result = compare_dxf_entities(prediction, reference)
            relative = Path("cases") / role / case_id; folder = output / relative
            folder.mkdir(parents=True, exist_ok=True)
            if folder.resolve() != folder: raise ValueError("Comparison artifact directory must not be redirected")
            _plot(result, folder / "aligned-diagnostic.png", f"{role}: {case_id}")
            _csv(folder / "parameters.csv", list(_parameter_rows(result)), PARAMETER_FIELDS)
            joint_rows = list(_joint_rows(result))
            _csv(folder / "connections.csv", joint_rows, tuple(joint_rows[0]) if joint_rows else ("side", "node_id"))
            match_rows = [{**item, "parameter_errors": json.dumps(item["parameter_errors"]),
                           "nearest_candidates": json.dumps(item["nearest_candidates"])} for item in result["entity_correspondence"]["rows"]]
            _csv(folder / "correspondence.csv", match_rows, ("prediction_id", "prediction_type", "status", "reference_id", "parameter_errors", "nearest_candidates"))
            row = {"case_id": case_id, "run_id": summary["run_id"], "job_id": case.get("job_id"),
                   "model_sha256": summary.get("checkpoint_sha256"), "source": case.get("source"),
                   "model_exposure": case.get("model_exposure"), "reference_receipt": receipt,
                   "reference_hash_matches_pilot": receipt.get("reference_sha256") == (case.get("reference") or {}).get("reference_sha256"),
                   "dimension_binding_evidence": case.get("dimensions"), "comparison": result,
                   "files": {"overlay": (relative / "aligned-diagnostic.png").as_posix(),
                             "parameters": (relative / "parameters.csv").as_posix(),
                             "connections": (relative / "connections.csv").as_posix(),
                             "correspondence": (relative / "correspondence.csv").as_posix()}}
            rows.append(row); summaries.append(_summary_row(role, row))
            packaged.extend(folder / name for name in ("aligned-diagnostic.png", "parameters.csv", "connections.csv", "correspondence.csv"))
        runs.append({"role": role, "run_id": summary["run_id"], "pilot_summary_sha256": _sha(path), "cases": rows})
    report = {"schema_version": "pilot-dxf-comparison-v1", "run_id": run_id, "status": "completed",
              "created_at": datetime.now(timezone.utc).isoformat(), "dataset_total": 50, "selected_count": 4, "not_selected": 46,
              "development_only": True, "blind_test": False, "engineering_verified": False,
              "strict_tolerance_mm": .1, "runs": runs, "changes": _changes(runs),
              "policy": "Read-only comparison of existing predictions. GT is used only in this scoring/reporting module; no model, scale solver, vectorizer or provider is invoked.",
              "files": {"html": "index.html", "json": "comparison.json", "csv": "comparison.csv", "zip": "comparison-artifacts.zip"}}
    report_path = output / "comparison.json"
    _write(report_path, report)
    _csv(output / "comparison.csv", summaries, tuple(summaries[0]))
    (output / "index.html").write_text(_html(report, output), encoding="utf8")
    packaged.extend([report_path, output / "comparison.csv", output / "index.html"])
    archive = output / "comparison-artifacts.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as package:
        for file in packaged: package.write(file, file.relative_to(output).as_posix())
    with zipfile.ZipFile(archive) as package:
        if package.testzip() is not None: raise ValueError("Comparison ZIP integrity check failed")
    _write(destination / "latest.json", {"run_id": run_id, "status": "completed", "report": f"{run_id}/comparison.json",
                                         "html": f"{run_id}/index.html", "archive": f"{run_id}/comparison-artifacts.zip",
                                         "report_sha256": _sha(report_path), "archive_sha256": _sha(archive)})
    return report, report_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, help="Completed pilot_summary.json; default is latest pilot")
    parser.add_argument("--candidate", type=Path, help="Optional later pilot_summary.json for before/after comparison")
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--output-root", type=Path)
    arguments = parser.parse_args(); settings = Settings()
    if arguments.runtime: settings.runtime_root = arguments.runtime.resolve()
    report, path = compare_pilot_runs(settings, baseline_summary=arguments.baseline,
                                    candidate_summary=arguments.candidate, output_root=arguments.output_root)
    print(json.dumps({"report": str(path), "run_id": report["run_id"], "runs": [{"role": run["role"], "cases": len(run["cases"])} for run in report["runs"]]}, ensure_ascii=False))


if __name__ == "__main__": main()
