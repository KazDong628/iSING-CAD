"""Read frozen controlled-run receipts; never generate CAD, open GT, or call APIs."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
from urllib.parse import quote


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def numeric(value):
    return type(value) in (int, float) and math.isfinite(value)


def geometry_signature(entities):
    fields = ("id", "type", "start", "end", "center", "radius", "clockwise")
    return [{key: row.get(key) for key in fields} for row in entities]


def endpoint_connectivity(exported, tolerance=1e-5):
    """Native endpoint closure, kept separate from self-intersection/tangency."""
    points, unsupported = [], []
    for primitive in exported:
        if primitive.dxftype() == "LINE":
            ends = (primitive.dxf.start, primitive.dxf.end)
        elif primitive.dxftype() == "ARC":
            ends = (primitive.start_point, primitive.end_point)
        else:
            unsupported.append(primitive.dxftype())
            continue
        points.extend((float(p[0]), float(p[1])) for p in ends)
    parents = list(range(len(points)))
    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index
    def union(a, b):
        parents[find(a)] = find(b)
    for i, first in enumerate(points):
        for j in range(i):
            if math.hypot(first[0] - points[j][0], first[1] - points[j][1]) <= tolerance:
                union(i, j)
    degree = Counter(find(i) for i in range(len(points)))
    non_degree_two = sum(n != 2 for n in degree.values())
    closed = bool(points and not unsupported and not non_degree_two)
    for i in range(0, len(points), 2):
        union(i, i + 1)
    return {"closed_endpoint_degree": closed, "component_count": len({find(i) for i in range(len(points))}),
            "non_degree_two_node_count": non_degree_two, "endpoint_tolerance_drawing_units": tolerance,
            "unsupported_native_types": sorted(set(unsupported)),
            "self_intersections_or_design_tangency_verified": False}


def native_export_check(path, model):
    """Re-read current DXF and compare source-bound native R, without candidate flags."""
    import ezdxf
    if not path.is_file():
        return {"status": "missing_prediction", "exact_radius_count": 0, "exact_radius_checks": []}
    try:
        doc = ezdxf.readfile(path)
        exported = list(doc.modelspace())
        entities = model.get("entities", [])
        by_id = {row.get("id"): index for index, row in enumerate(entities)}
        published = model.get("parameterization") or {}
        constraints = published.get("constraints", [])
        checks = []
        for constraint in constraints:
            if constraint.get("kind") != "radius":
                continue
            ids = constraint.get("entities") or []
            entity_id = ids[0] if len(ids) == 1 else None
            index = by_id.get(entity_id)
            row = entities[index] if index is not None else {}
            primitive = exported[index] if index is not None and index < len(exported) else None
            actual = float(primitive.dxf.radius) if primitive is not None and primitive.dxftype() == "ARC" else None
            nominal = constraint.get("value")
            passed = bool(len(entities) == len(exported) and row.get("type") == "ARC" and
                          numeric(nominal) and nominal > 0 and actual == nominal and row.get("radius") == nominal)
            checks.append({"record_id": constraint.get("record_id"), "entity_id": entity_id,
                           "nominal": nominal, "current_dxf_radius": actual, "passed": passed,
                           "absolute_residual_mm": abs(actual - nominal) if numeric(actual) and numeric(nominal) else None,
                           "radius_tolerance_mm": 0.})
        coverage = ((model.get("validation") or {}).get("annotation_radius_contract") or {}).get("coverage") or published.get("radius_binding_coverage") or {}
        recognized = coverage.get("recognized_count")
        exemptions = len(coverage.get("verified_absent_arrow_records", []))
        exact_count = len({(row["record_id"], row["entity_id"]) for row in checks if row["passed"]})
        ambiguous_ledger = (len({row["record_id"] for row in checks}) != len(checks) or
                            len({row["entity_id"] for row in checks}) != len(checks) or
                            any(not isinstance(row["record_id"], str) or not row["record_id"] for row in checks))
        return {"status": "read_back", "sha256": digest(path), "units": int(doc.units),
                "native_object_count": len(exported), "native_types": dict(Counter(e.dxftype() for e in exported)),
                "connectivity": endpoint_connectivity(exported),
                "exact_radius_checks": checks, "exact_radius_count": exact_count,
                "ambiguous_or_duplicate_radius_ledger": ambiguous_ledger,
                "recognized_radius_count": recognized, "required_radius_count": coverage.get("required_count"),
                "verified_absent_arrow_count": exemptions,
                "unknown_arrow_count": len(coverage.get("unknown_arrow_records", [])),
                "unresolved_radius_count": coverage.get("unresolved_count"),
                "all_recognized_radii_exact": bool(checks and not ambiguous_ledger and exact_count == len(checks) and
                                                    isinstance(recognized, int) and recognized - exemptions == exact_count and
                                                    coverage.get("all_radius_records_resolved") is True),
                "source_association_audited": False,
                "scope": "Independent native ARC value readback of published bindings; no proof of arrow association or OCR completeness."}
    except Exception as error:
        return {"status": "readback_failed", "failure_type": type(error).__name__, "exact_radius_count": 0, "exact_radius_checks": []}


def transport_receipts(after):
    """Stage-level receipts; do not double-count duplicated publication snapshots."""
    stage = read(after / "parametric-stage.json")
    values = [("dimension_analysis", (read(after / "dimension-analysis.json").get("provider") or {})),
              ("radius_target_localization", stage.get("radius_target_provider") or {}),
              ("final_binding", stage.get("provider") or {})]
    plan = read(after / "topology-plan.json")
    iterations = read(after / "topology-iterations.json") or plan.get("topology_editing") or {}
    rounds = iterations.get("rounds", []) if isinstance(iterations, dict) else []
    for key in (("provider", "planning_provider") if rounds else
                ("provider", "planning_provider", "editor_provider", "evaluator_provider")):
        if isinstance(plan.get(key), dict):
            values.append(("topology_" + key, plan[key]))
    # Each round mirrors its selected branch at the top level. Read the branch
    # ledger once so failed calls survive reporting without double-counting the
    # mirrored winner. Identical receipts in different branches are real calls.
    seen = set()
    for round_index, round_row in enumerate(rounds, 1):
        if not isinstance(round_row, dict):
            continue
        branches = round_row.get("branches") or [round_row]
        for branch_index, branch in enumerate(branches, 1):
            if not isinstance(branch, dict):
                continue
            identity = (round_row.get("round", round_index), branch.get("branch", branch_index))
            for role in ("editor", "evaluator"):
                receipt = branch.get(role)
                key = (*identity, role)
                if isinstance(receipt, dict) and key not in seen:
                    seen.add(key)
                    values.append((f"topology_round_{identity[0]}_branch_{identity[1]}_{role}", receipt))
    keys = ("status", "http_status", "http_success", "schema_success", "any_http_success", "any_schema_success",
            "partial_schema_success", "any_partial_schema_success", "response_structure_valid",
            "schema_rejection_scope", "rejected_record_ids", "unresolved_rejected_record_ids",
            "network_requests", "known_network_requests", "elapsed_seconds", "model", "protocol", "error_code")
    return [{"stage": name, **{key: value.get(key) for key in keys if key in value}} for name, value in values if value]


def collect_attempt(root, run):
    manifest = read(run / "run-manifest.json")
    after = run / "after"
    model = read(after / "model.json")
    published = model.get("parameterization") or {}
    stage = read(after / "parametric-stage.json")
    native = native_export_check(after / "drawing.dxf", model)
    frozen = bool(manifest.get("prediction_frozen") is not False and
                  manifest.get("prediction_frozen_at") and manifest.get("prediction_sha256"))
    artifact_current = bool(frozen and native.get("sha256") == manifest.get("prediction_sha256"))
    code_current = manifest.get("code", {}).get("unchanged") is True
    evaluation = read(run / "evaluation" / "summary.json")
    eval_sha = evaluation.get("provenance", {}).get("prediction_sha256")
    evaluation_current = bool(artifact_current and eval_sha == native.get("sha256"))
    geometry = evaluation.get("geometry") or {}
    registered = geometry.get("registered_shape_diagnostic") or {}
    direct = geometry.get("native_coordinate_frame") or {}
    target = evaluation.get("target_acceptance") or {}
    feedback = published.get("reconstruction_feedback") or {}
    solver = published.get("solver") or {}
    diagnostics = solver.get("diagnostics") or {}
    solution = read(after / "parametric-solution.json")
    matching_solution = bool(solution.get("entities") and geometry_signature(solution["entities"]) == geometry_signature(model.get("entities", [])))
    checks = solution.get("constraints", []) if matching_solution else []
    residuals = {unit: max((row["absolute_residual"] for row in checks
                           if row.get("residual_unit") == unit and numeric(row.get("absolute_residual"))), default=None)
                 for unit in ("mm", "degree")}
    shape_dof = diagnostics.get("remaining_shape_dof", feedback.get("remaining_shape_dof"))
    full = bool(manifest.get("status") == "completed" and artifact_current and code_current and evaluation_current and
                target.get("all_requested_checks_passed") is True and
                native.get("all_recognized_radii_exact") is True and shape_dof == 0 and
                published.get("all_dimensions_verified") is True)
    reference = {"status": evaluation.get("status"), "receipt_matches_current_dxf": evaluation_current,
                 "reference_sha256": evaluation.get("provenance", {}).get("reference_sha256"),
                 "objects": evaluation.get("objects"), "checks_0_1mm": evaluation.get("checks") if evaluation_current else None,
                 "checks_1mm": target.get("checks") if evaluation_current else None,
                 "target_all_checks_passed": target.get("all_requested_checks_passed") if evaluation_current else None,
                 "registered_max_mm": registered.get("conservative_max_error_mm") if evaluation_current else None,
                 "registered_rms_mm": registered.get("rms_error_mm") if evaluation_current else None,
                 "native_max_mm": direct.get("conservative_max_error_mm") if evaluation_current else None,
                 "matched_primitive_count_0_1mm": evaluation.get("primitive_matching", {}).get("matched_count"),
                 "matched_primitive_count_1mm": target.get("primitive_matching", {}).get("matched_count")}
    links = {}
    for name, relative in (("dxf", "after/drawing.dxf"), ("overlay", "evaluation/overlay.svg"),
                           ("manifest", "run-manifest.json"), ("evaluation", "evaluation/summary.json")):
        if (run / relative).is_file():
            links[name] = (run / relative).relative_to(root).as_posix()
    return {"attempt": run.relative_to(root).as_posix(), "label": manifest.get("label", "unknown"),
            "case_id": manifest.get("case_id", "unknown"), "status": manifest.get("status", "unreadable_manifest"),
            "created_at": manifest.get("created_at"), "last_stage": manifest.get("last_stage"),
            "failure_type": manifest.get("failure_type"),
            "interruption_reason": manifest.get("interruption_reason", manifest.get("reason")),
            "artifact_frozen": frozen,
            "artifact_state": ("frozen_current" if artifact_current else "frozen_artifact_mismatch" if frozen else
                               "intermediate_only" if native.get("status") == "read_back" else "no_readable_prediction"),
            "frozen_prediction_still_current": artifact_current, "code_unchanged_during_prediction": code_current,
            "code_root": manifest.get("code", {}).get("root"),
            "code_sha256": manifest.get("code", {}).get("sha256_before", {}),
            "input_sha256": {key: manifest.get(key) for key in ("source_image_sha256", "source_ocr_sha256", "oracle_mask_sha256")},
            "transport": transport_receipts(after), "native": native, "reference": reference,
            "published_parameterization": {"status": published.get("status"), "latest_attempt_status": stage.get("status"),
                "solver_status": solver.get("status"), "constraint_subset_accepted": published.get("constraint_subset_accepted"),
                "bound_source_records": published.get("binding_counts", {}).get("bound_source_records"),
                "recognized_dimensions": published.get("binding_counts", {}).get("recognized_dimensions"),
                "remaining_shape_dof": shape_dof, "constraint_rank": diagnostics.get("constraint_rank"),
                "closed": native.get("connectivity", {}).get("closed_endpoint_degree"),
                "geometry_valid": solver.get("validation", {}).get("geometry_valid", (model.get("validation") or {}).get("passed")),
                "current_solution_receipt_matches_model": matching_solution,
                "maximum_recorded_constraint_residual_by_unit": residuals,
                "all_dimensions_verified": published.get("all_dimensions_verified")},
            "full_user_target_passed": full, "files": links}


def assemble(root, cases=("HDSA-65-main", "044-main"), labels=("baseline", "one-fix")):
    root = Path(root).resolve(strict=True)
    attempts = [collect_attempt(root, path.parent) for path in sorted(root.glob("*/run-manifest.json"))]
    # Latest is chronological, never minimum GT error. Every failed run remains.
    latest = []
    for label in labels:
        for case in cases:
            rows = [row for row in attempts if row["label"] == label and row["case_id"] == case]
            if rows:
                latest.append(max(rows, key=lambda row: (row.get("created_at") or "", row["attempt"])))
            else:
                latest.append({"attempt": None, "label": label, "case_id": case, "status": "not_run",
                               "full_user_target_passed": False, "files": {}})
    pairs = []
    if labels:
        for case in cases:
            base = next((r for r in latest if r["label"] == labels[0] and r["case_id"] == case), {})
            for label in labels[1:]:
                candidate = next((r for r in latest if r["label"] == label and r["case_id"] == case), {})
                completed_states = ("completed", "completed_with_unresolved_attributes")
                comparable = bool(base.get("status") in completed_states and candidate.get("status") in completed_states and
                                  base.get("frozen_prediction_still_current") and candidate.get("frozen_prediction_still_current") and
                                  base.get("code_unchanged_during_prediction") and candidate.get("code_unchanged_during_prediction") and
                                  base.get("reference", {}).get("receipt_matches_current_dxf") and
                                  candidate.get("reference", {}).get("receipt_matches_current_dxf") and
                                  base.get("input_sha256") == candidate.get("input_sha256") and
                                  base.get("reference", {}).get("reference_sha256") and
                                  base.get("reference", {}).get("reference_sha256") == candidate.get("reference", {}).get("reference_sha256"))
                changes = [key for key in sorted(set(base.get("code_sha256", {})) | set(candidate.get("code_sha256", {})))
                           if base.get("code_sha256", {}).get(key) != candidate.get("code_sha256", {}).get(key)]
                before = base.get("reference", {}).get("registered_max_mm")
                after = candidate.get("reference", {}).get("registered_max_mm")
                pairs.append({"case_id": case, "baseline_attempt": base.get("attempt"), "candidate_attempt": candidate.get("attempt"),
                              "candidate_label": label, "same_frozen_inputs_and_reference": comparable,
                              "comparison_scope": "completed_current_frozen_runs_only", "changed_code_files": changes,
                              "registered_max_delta_mm": after - before if comparable and numeric(before) and numeric(after) else None,
                              "exact_radius_count_delta": candidate.get("native", {}).get("exact_radius_count", 0) - base.get("native", {}).get("exact_radius_count", 0) if comparable else None,
                              "publication_decision": "No automatic promotion; source checks and all-dimensional coverage are separate from GT diagnostics."})
    return {"schema_version": "controlled-cad-report-v1", "created_at": datetime.now(timezone.utc).isoformat(),
            "condition": "frozen_GT_derived_raster_mask_plus_original_image_and_OCR", "held_out": False,
            "calibration_or_development": True, "reference_coordinates_used_for_prediction": False,
            "all_attempts": attempts, "latest_chronological": latest, "comparisons": pairs,
            "coverage": {"actual_attempts": len(attempts), "actual_attempts_full_pass": sum(r["full_user_target_passed"] for r in attempts),
                         "expected_label_case_slots": len(cases) * len(labels),
                         "latest_slots_full_pass": sum(r["full_user_target_passed"] for r in latest),
                         "not_run_slots": sum(r["status"] == "not_run" for r in latest)},
            "limits": ["All failed/interrupted attempts and not-run slots remain visible; no best-GT selection.",
                       "Online transport receipts are stage snapshots, not parameter accuracy or a deduplicated total request count.",
                       "Partial schema success preserves only individually validated records; it never counts as full schema success or verified geometry.",
                       "Intermediate artifacts from interrupted runs are shown as drafts and never enter completed-run comparisons.",
                       "0.1 mm strict evaluation stays unchanged; 1 mm is supplemental and annotated R remains exactly equal.",
                       "Oracle-mask development success cannot establish segmentation accuracy or held-out generalization.",
                       "This reporter reads existing evaluation receipts only. It never opens a GT DXF or invokes a provider.",
                       "Zero shape DOF alone does not prove correct binding, unique global geometry, or independent reference accuracy."]}


def render(report):
    def esc(value):
        return html.escape(str(value), quote=True)
    def value(item):
        if item is None:
            return "—"
        if isinstance(item, bool):
            return "通过" if item else "未通过"
        if isinstance(item, float):
            return f"{item:.6f}"
        if isinstance(item, dict):
            return esc(json.dumps(item, ensure_ascii=False))
        return esc(item)
    def rows(items):
        result = []
        for row in items:
            native, reference, param = row.get("native", {}), row.get("reference", {}), row.get("published_parameterization", {})
            links = " · ".join(f'<a href="{quote(path, safe="/")}">{esc(name)}</a>' for name, path in row.get("files", {}).items())
            cells = [row["label"], row["case_id"], row["status"], row.get("artifact_state"), native.get("native_types"),
                     f'{native.get("exact_radius_count", 0)} / {native.get("recognized_radius_count", "?")}',
                     param.get("remaining_shape_dof"), reference.get("registered_max_mm"), reference.get("native_max_mm"),
                     (reference.get("checks_0_1mm") or {}).get("registered_shape_within_0_1mm"),
                     (reference.get("checks_1mm") or {}).get("registered_shape_within_target"), row["full_user_target_passed"]]
            result.append("<tr>" + "".join(f"<td>{value(cell)}</td>" for cell in cells) + f"<td>{links}</td></tr>")
        return "".join(result)
    headers = ("版本", "图纸", "运行状态", "产物状态", "DXF 原生对象", "严格 R / 已识别 R", "剩余形状自由度", "配准最大误差 mm",
               "原坐标最大误差 mm", "0.1 mm 形状", "1 mm 形状", "完整目标", "证据")
    table_head = "<table><thead><tr>" + "".join(f"<th>{esc(v)}</th>" for v in headers) + "</tr></thead><tbody>"
    details = []
    for row in report["all_attempts"]:
        transport_rows = []
        for receipt in row.get("transport", []):
            cells = (receipt.get("stage"), receipt.get("status"), receipt.get("http_success"),
                     receipt.get("schema_success"), receipt.get("partial_schema_success", receipt.get("any_partial_schema_success")),
                     receipt.get("rejected_record_ids"), receipt.get("unresolved_rejected_record_ids"))
            transport_rows.append("<tr>" + "".join(f"<td>{value(cell)}</td>" for cell in cells) + "</tr>")
        transport_head = ("阶段", "阶段状态", "HTTP 成功", "完整结构成功", "部分结构成功", "曾拒绝的记录", "仍未解决的拒绝记录")
        transport_table = ("<table><thead><tr>" + "".join(f"<th>{esc(cell)}</th>" for cell in transport_head) +
                           "</tr></thead><tbody>" + "".join(transport_rows) + "</tbody></table>")
        body = {key: row.get(key) for key in ("artifact_state", "interruption_reason", "transport", "native", "published_parameterization", "reference")}
        details.append(f'<details><summary>{esc(row["attempt"])}：传输、当前 DXF 半径、求解和参考检查</summary>{transport_table}<pre>{esc(json.dumps(body, ensure_ascii=False, indent=2))}</pre></details>')
    return ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>受控 CAD 迭代测试</title>'
            '<style>body{font:16px/1.55 system-ui,sans-serif;margin:32px;color:#243240;background:#f7f9fb}h1{font-size:26px}table{border-collapse:collapse;width:100%;background:white;font-size:13px}th,td{padding:10px;border:1px solid #dce2e8;text-align:left}th{background:#eaf0f5}section{overflow:auto;margin:24px 0}pre{white-space:pre-wrap;font:12px/1.5 monospace;background:#fff;padding:16px}details{margin:16px 0}a{color:#006e79}p{max-width:1050px}</style>'
            '<h1>回退基线与单项改进：受控测试</h1><p>输入为固定原图、OCR 与 GT 派生栅格掩膜，属于开发实验。预测冻结后才执行 GT DXF 评估。1 mm 是附加检查，原 0.1 mm 和标注 R 严格相等要求保留。</p>'
            f'<p>全部实际尝试：{report["coverage"]["actual_attempts_full_pass"]} / {report["coverage"]["actual_attempts"]} 完整通过。预定版本 × 图纸：{report["coverage"]["latest_slots_full_pass"]} / {report["coverage"]["expected_label_case_slots"]}；尚未运行 {report["coverage"]["not_run_slots"]} 项。</p>'
            '<p>表中采用每个版本、每幅图纸最新一次运行，未按 GT 误差选择最好结果。形状通过不能代替对象、约束、精确半径或全部属性通过。</p>'
            '<section><h2>最新结果</h2>' + table_head + rows(report["latest_chronological"]) + '</tbody></table></section>'
            '<section><h2>全部尝试（包含失败和中断）</h2>' + table_head + rows(report["all_attempts"]) + '</tbody></table></section>'
            '<h2>同输入比较</h2><pre>' + esc(json.dumps(report["comparisons"], ensure_ascii=False, indent=2)) + '</pre>'
            + "".join(details) + '<h2>解释边界</h2><ul>' + "".join(f"<li>{esc(v)}</li>" for v in report["limits"]) + '</ul></html>')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", default=["baseline", "one-fix"])
    parser.add_argument("--cases", nargs="+", default=["HDSA-65-main", "044-main"])
    args = parser.parse_args(argv)
    report = assemble(args.root, tuple(args.cases), tuple(args.labels))
    (args.root / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
    (args.root / "report.html").write_text(render(report), encoding="utf8")
    print(json.dumps(report["coverage"], ensure_ascii=False))


if __name__ == "__main__":
    main()
