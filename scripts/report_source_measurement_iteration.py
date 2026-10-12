"""Compare frozen source-measurement iterations; never read GT CAD directly."""
from __future__ import annotations

import html
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import report_source_priority_iteration as base
from scripts.report_source_initialization_iteration import pipeline_gaps


def read(path):
    return json.loads(path.read_text("utf8")) if path.is_file() else {}


def measurement_graph(directory):
    """Symbolic incidence and source obligations, not pixel coordinate locks.

    Missing candidates, selected equations and native satisfaction are kept
    separate. A common-center measurement contributes its two equalities only
    after the corresponding measurement has actually entered the solver.
    """
    inventory = read(directory/"binding-candidates.json")
    bindings = read(directory/"constraint-bindings.json")
    solution = read(directory/"parametric-solution.json")
    validation = read(directory/"validation.json")
    constraints = bindings.get("constraints", [])
    def identity(row):
        return (row.get("record_id"), row.get("kind"), tuple(sorted(row.get("entities", []))),
            tuple(row.get("nodes", [])), json.dumps(row.get("anchors"), sort_keys=True),
            json.dumps(row.get("source_station_relation"), sort_keys=True))
    known = {identity(row) for row in constraints}
    locations, edges = {}, []
    for candidate in inventory.get("all_candidates", []):
        if candidate.get("kind") not in {"distance", "distance_x", "distance_y"}:
            continue
        anchors = candidate.get("anchors") or [{"type": "node", "node_id": n} for n in candidate.get("nodes", [])]
        keys = []
        for anchor in anchors:
            key = json.dumps(anchor, sort_keys=True)
            locations[key] = anchor
            keys.append(key)
        edges.append({"candidate_id": candidate["id"], "record_id": candidate["record_id"],
            "kind": candidate["kind"], "locations": keys,
            "source_admitted": candidate.get("local_reliable") is True,
            "selected_into_current_constraints": identity(candidate) in known,
            "source_station_relation": candidate.get("source_station_relation"),
            "observed_interval_method": (candidate.get("evidence", {}).get("dimension_line") or {}).get("segmentation_method"),
            "source_center_proofs": [g.get("construction") for g in
                candidate.get("evidence", {}).get("observed_station_groups", [])
                if (g.get("anchor") or {}).get("type") in {"curve_center", "common_curve_center"}]})
    represented = {edge["record_id"] for edge in edges}
    missing = [{"record_id": row["id"], "text": row.get("text"), "kind": row["parsed"]["kind"]}
        for row in inventory.get("all_records", []) if row.get("parsed", {}).get("kind") in {"length", "diameter"}
        and row["id"] not in represented]
    center_equations = []
    seen = set()
    for row in constraints:
        for anchor in row.get("anchors", []):
            if anchor.get("type") != "common_curve_center":
                continue
            pair = tuple(sorted((anchor["entity_id"], anchor["other_entity_id"])))
            if pair not in seen:
                center_equations.append({"entities": pair, "equal_coordinates": ["x", "y"],
                    "source_measurement_record": row.get("record_id"), "literal_coordinate_locked": False})
                seen.add(pair)
    return {"schema_version": "symbolic-source-measurement-incidence-v1",
        "locations": list(locations.values()), "candidate_measurements": edges,
        "unrepresented_measurement_obligations": missing,
        "admitted_common_center_equations": center_equations,
        "solver_measurement_certificate": solution.get("strict_measurement_contract"),
        "native_measurement_readback_certificate": validation.get("native_measurement_validation"),
        "constraint_rank": solution.get("diagnostics", {}).get("constraint_rank"),
        "remaining_shape_dof": solution.get("diagnostics", {}).get("remaining_shape_dof"),
        "scope": "Candidates are not equations or a success certificate. Soft pixel losses are excluded from constraint rank."}


def main():
    root = Path(__file__).resolve().parents[1]/"runtime/cad-controlled-20261008"
    base.SLOTS = [("v75", "source-initialization-v75-online", "此前冻结基线"),
                  ("v77", "source-measurement-v77-online", "长引线、共同圆心及远端尺寸箭头／在线最终阶段重放")]
    report = base.assemble(root)
    report.update(schema_version="source-measurement-iteration-review-v1", new_attempt_slots=2,
        fresh_full_end_to_end_runs=0, report_reads_gt_dxf=False, report_calls_provider=False,
        reference_thresholds_changed=False, cloud_gpt_equivalence_established=False,
        regression_logs=["pytest-measurement-v77-full.txt", "pytest-measurement-v77-new.txt"],
        historical_report="source-initialization-v75-review.html",
        changes=["Source-only finite long radial shaft remeasurement with explicit owned OCR occlusions",
            "Common-center symbolic anchors retain both ARC references and exact center x/y equalities",
            "Native measurement certification and constraint rank include common-center equalities",
            "Remote short dimension extensions recovered by two observed opposing arrows on a full label-owned shaft",
            "No annotation nominal, CAD fit or GT coordinate ranks a source arrow or center location"])
    extra = "<p>两张已曝光开发图；本轮两次真实在线最终阶段重放，复用冻结 OCR 和上游拓扑，使用 GT 派生栅格掩膜。不是全新网页端到端测试。预测先冻结，GT 仅由独立评估进程读取。所有未通过项保留。</p>"
    extra += "<p>共同圆心以当前两条 ARC 的未知圆心和精确相等方程表示；原图测得的圆心只关联尺寸，不写成 CAD 坐标目标。105、351 能生成候选不等于已经在线绑定、求解和发布。</p>"
    for row in report["rows"]:
        directory = root/row["run"]
        manifest = read(directory/"run-manifest.json")
        row["pipeline_gaps"] = pipeline_gaps(directory)
        row["snapshot_integrity"] = {key: manifest.get(key) for key in
            ("upstream_receipts_unchanged", "prediction_unchanged_after_evaluation")}
        row["snapshot_integrity"]["code_unchanged"] = manifest.get("code", {}).get("unchanged")
        row["measurement_stages"] = {"published": measurement_graph(directory/"after")}
        attempt = directory/"after/final-binding-attempt"
        if attempt.is_dir():
            row["measurement_stages"]["unpublished_last_attempt"] = measurement_graph(attempt)
        for path in sorted((directory/"after/source-constraint-completion").glob("round-*")):
            row["measurement_stages"][str(path.relative_to(directory/"after"))] = measurement_graph(path)
        (directory/"symbolic-measurement-graph.json").write_text(json.dumps(row["measurement_stages"],
            ensure_ascii=False, indent=2), "utf8")
        extra += "<details><summary>"+html.escape(row["run"])+"：测量图、缺失方程和发布状态</summary><pre>"+html.escape(
            json.dumps({"integrity": row["snapshot_integrity"], "measurement_stages": row["measurement_stages"],
                        "pipeline_gaps": row["pipeline_gaps"]}, ensure_ascii=False, indent=2))+"</pre></details>"
    page = base.render(report).replace("标注优先：冲突修复与独立评估", "V77：原图长引线、符号圆心与尺寸方程")
    page = page.replace("<main>", extra+"<main>", 1)
    (root/"source-measurement-v77-review.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf8")
    (root/"source-measurement-v77-review.html").write_text(page, "utf8")
    print(json.dumps({"rows": len(report["rows"]), "new_online_attempts": 2,
        "complete": sum(row["source_goal_complete"] for row in report["rows"])}))


if __name__ == "__main__":
    main()
