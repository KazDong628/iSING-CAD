"""Compare named frozen source-obligation runs; never open GT or invoke an API."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import re
import sys
from urllib.parse import quote

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from contour_agent.parametric_solver import STRICT_OCR_ANGLE_CERT_TOLERANCE_DEG
from contour_agent.relation_contract import relation_checks, _native_geometry, _geometry, _joint_check
from scripts.report_controlled_iteration import collect_attempt, digest, numeric, read


CASES = (("hdsa", "HDSA-65-main"), ("044", "044-main"))
VERSIONS = ("v11", "v12")
REQUIRED_FILES = ("drawing.dxf", "model.json", "parametric-solution.json", "constraint-bindings.json",
                  "parametric-stage.json", "binding-candidates.json")


def _elapsed(manifest):
    def seconds(end):
        try:
            return max(0., (datetime.fromisoformat(end.replace("Z", "+00:00")) -
                           datetime.fromisoformat(manifest["created_at"].replace("Z", "+00:00"))).total_seconds())
        except (AttributeError, KeyError, TypeError, ValueError):
            return None
    history = manifest.get("stage_history") or []
    finished = [row.get("at") for row in history if isinstance(row, dict) and row.get("stage") == "finished"]
    return {"prediction_seconds": seconds(manifest.get("prediction_frozen_at")),
            "total_run_seconds": seconds(finished[-1]) if finished else None,
            "scope": "Wall time from run creation; total includes independent evaluation when finished."}


def _frozen_receipts(run, manifest):
    frozen = manifest.get("frozen_artifacts") or {}
    rows = []
    for name in REQUIRED_FILES:
        path = run / "after" / name
        actual = digest(path) if path.is_file() else None
        rows.append({"file": name, "expected_sha256": frozen.get(name), "actual_sha256": actual,
                     "passed": bool(actual and frozen.get(name) == actual)})
    return {"passed": all(row["passed"] for row in rows), "checks": rows}


def _native_angle_checks(entities, constraints, inventory, document):
    exported = list(document.modelspace())
    positions = {row.get("id"): index for index, row in enumerate(entities)}
    records = [row for row in inventory.get("all_records", inventory.get("records", []))
               if (row.get("parsed") or {}).get("kind") == "angle"]
    rows = []
    for record in records:
        nominal = (record.get("parsed") or {}).get("nominal")
        matching = [row for row in constraints if row.get("kind") == "angle" and row.get("record_id") == record.get("id")]
        result = {"record_id": record.get("id"), "nominal_degrees": nominal,
                  "passed": False, "bound": bool(matching), "native_degrees": None,
                  "tolerance_degrees": STRICT_OCR_ANGLE_CERT_TOLERANCE_DEG}
        try:
            if len(matching) != 1:
                raise ValueError("unbound_or_ambiguous_angle")
            constraint = matching[0]
            ids = constraint.get("entities") or []
            axis = constraint.get("reference_axis")
            if len(ids) != 1 or axis not in {"horizontal", "vertical"}:
                raise ValueError("not_a_supported_single_line_axis_angle")
            index = positions[ids[0]]
            if len(exported) != len(entities):
                raise ValueError("native_count_mismatch")
            native = _native_geometry(exported[index], _geometry(entities[index]))
            if native["type"] != "LINE" or not numeric(nominal) or constraint.get("value") != nominal:
                raise ValueError("annotation_or_line_type_mismatch")
            dx, dy = (abs(native["end"][i] - native["start"][i]) for i in (0, 1))
            value = math.degrees(math.atan2(dx, dy) if axis == "vertical" else math.atan2(dy, dx))
            residual = abs(value - nominal)
            result.update(entity_id=ids[0], reference_axis=axis, native_degrees=value,
                          absolute_residual_degrees=residual,
                          passed=residual <= STRICT_OCR_ANGLE_CERT_TOLERANCE_DEG)
        except (KeyError, ValueError, TypeError, IndexError) as error:
            result["reason"] = str(error)
        rows.append(result)
    return {"recognized_count": len(rows), "bound_count": sum(row["bound"] for row in rows),
            "strict_native_satisfied_count": sum(row["passed"] for row in rows), "checks": rows,
            "scope": "Recognized single-line axis angles; source association and complete OCR detection are not certified here."}


def _curved_join_audit(entities, constraints, document, strict):
    native, mapping_issues = {}, []
    exported = list(document.modelspace())
    try:
        if len(exported) != len(entities):
            raise ValueError("native_count_mismatch")
        for entity, primitive in zip(entities, exported):
            source = _geometry(entity)
            if source["id"] in native:
                raise ValueError("duplicate_entity_id")
            native[source["id"]] = _native_geometry(primitive, source)
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        mapping_issues.append(str(error))
    by_relation = {}
    for row in strict.get("checks", []):
        ids, node = row.get("entity_ids"), row.get("node_id")
        if isinstance(ids, list) and len(ids) == 2 and node:
            by_relation[(frozenset(ids), node)] = row
    # Presence of a declared relation must survive a failed certificate, which
    # may lack normalized IDs. Such joins are failed, never recast as unknown.
    declared = {}
    for row in constraints:
        ids, nodes = row.get("entities") or [], row.get("nodes") or []
        if row.get("kind") == "tangent" and len(ids) == 2 and len(nodes) == 1:
            declared[(frozenset(ids), nodes[0])] = row.get("id")
    checks = []
    for index, left in enumerate(entities):
        right = entities[(index + 1) % len(entities)]
        if "ARC" not in (left.get("type"), right.get("type")):
            continue
        node = left.get("end_node")
        key = (frozenset((left.get("id"), right.get("id"))), node)
        certificate = by_relation.get(key)
        admitted = key in declared or certificate is not None
        result = {"entities": [left.get("id"), right.get("id")], "node_id": node,
                  "types": [left.get("type"), right.get("type")],
                  "relationship_status": "admitted_passed" if certificate and certificate.get("passed") else
                                         "admitted_failed" if admitted else "unknown",
                  "constraint_id": declared.get(key), "native_geometry": None,
                  "tangency_inferred_from_appearance": False}
        if not mapping_issues and node == right.get("start_node"):
            try:
                result["native_geometry"] = _joint_check(native[left["id"]], native[right["id"]], node)
            except (KeyError, TypeError, ValueError) as error:
                result["native_diagnostic_error"] = str(error)
        checks.append(result)
    return {"curved_join_count": len(checks),
            "unknown_count": sum(row["relationship_status"] == "unknown" for row in checks),
            "unknown_arc_arc_count": sum(row["relationship_status"] == "unknown" and row["types"] == ["ARC", "ARC"] for row in checks),
            "admitted_passed_count": sum(row["relationship_status"] == "admitted_passed" for row in checks),
            "admitted_failed_count": sum(row["relationship_status"] == "admitted_failed" for row in checks),
            "mapping_issues": mapping_issues, "checks": checks,
            "scope": "Unknown means no admitted relationship, not proof a join should be tangent. All ARC-ARC joins remain in the denominator."}


def _binding_receipt(stage):
    provider = stage.get("provider") or {}
    coverage = provider.get("inventory_coverage") or {}
    pages = provider.get("pages")
    actual_pages = pages if isinstance(pages, list) else ([provider] if provider.get("network_requests", 0) else [])
    fields = ("page_index", "status", "http_success", "schema_success", "selection_payload_verified", "http_status",
              "network_requests", "elapsed_seconds", "error_code", "input_record_ids", "input_candidate_ids", "input_relation_ids")
    sent_records = provider.get("sent_record_ids", provider.get("input_record_ids", []))
    sent_relations = provider.get("sent_relation_ids", provider.get("input_relation_ids", []))
    valid_records = provider.get("input_record_ids", []) if provider.get("schema_success") else []
    return {"status": provider.get("status"), "protocol": provider.get("protocol"),
            "network_requests": provider.get("network_requests"), "http_success": provider.get("http_success"),
            "schema_success": provider.get("schema_success"),
            "schema_success_scope": "valid retained page subset" if isinstance(pages, list) else "single bounded response",
            "all_pages_succeeded": provider.get("all_pages_succeeded", provider.get("schema_success")),
            "coverage_complete": provider.get("coverage_complete"), "actual_page_count": len(actual_pages),
            "all_record_count": coverage.get("all_record_count"),
            "source_eligible_record_count": coverage.get("source_eligible_record_count"),
            "actual_sent_record_ids": sent_records, "actual_sent_record_count": len(sent_records),
            "valid_response_record_ids": valid_records, "valid_response_record_count": len(valid_records),
            "actual_sent_relation_ids": sent_relations, "actual_sent_relation_count": len(sent_relations),
            "all_relation_count": coverage.get("all_relation_count"),
            "record_ids_not_sent": coverage.get("record_ids_not_sent"),
            "relation_ids_not_sent": coverage.get("relation_ids_not_sent"),
            "elapsed_seconds": provider.get("elapsed_seconds"), "total_timeout_seconds": provider.get("total_timeout_seconds"),
            "pages": [{key: row.get(key) for key in fields if key in row} for row in actual_pages],
            "scope": "Latest online binding attempt; selected publication may retain an earlier verified numerical result."}


def _solution_publication_link(model, solution, manifest, frozen_receipts):
    """Geometry equality alone also matches a rejected solver's baseline."""
    published = model.get("parameterization") or {}
    checks = {"solution_accepted": solution.get("accepted") is True,
              "published_solver_accepted": (published.get("solver") or {}).get("accepted") is True,
              "published_subset_accepted": published.get("constraint_subset_accepted") is True or published.get("accepted") is True,
              "frozen_export_and_receipts_verified": frozen_receipts.get("passed") is True,
              "solution_geometry_matches": False, "candidate_geometry_matches": False,
              "source_obligations_match": False,
              "solver_diagnostics_agree": all((solution.get("diagnostics") or {}).get(key) ==
                  ((published.get("solver") or {}).get("diagnostics") or {}).get(key)
                  for key in ("remaining_shape_dof", "remaining_dof", "constraint_rank"))}
    def geometry(values):
        if not isinstance(values, list) or not values: raise ValueError("missing_geometry")
        normalized = [_geometry(entity) for entity in values]
        if len({entity["id"] for entity in normalized}) != len(normalized): raise ValueError("duplicate_entity_id")
        return normalized
    def obligations(values, entities):
        if not isinstance(values, list) or not values: raise ValueError("no_formal_constraints")
        indexed = {entity["id"]: entity for entity in entities}
        result = {}
        fields = ("kind", "entities", "nodes", "value", "record_id", "source", "reference_axis",
                  "required", "nominal_source", "source_arrow_verified")
        for row in values:
            cid = row.get("id")
            if not isinstance(cid, str) or not cid or cid in result: raise ValueError("invalid_constraint_identity")
            item = {key: row.get(key) for key in fields}
            item["entities"], item["nodes"] = row.get("entities", []), row.get("nodes", [])
            if row.get("kind") in {"horizontal", "vertical"} and not item["nodes"] and len(item["entities"]) == 1:
                entity = indexed[item["entities"][0]]
                item["nodes"] = [entity["start_node"], entity["end_node"]]
            if row.get("kind") == "angle": item["angle_mode"] = row.get("angle_mode", "unsigned")
            result[cid] = item
        return result
    try:
        current = geometry(model.get("entities"))
        checks["solution_geometry_matches"] = geometry(solution.get("entities")) == current
        checks["candidate_geometry_matches"] = geometry(solution.get("candidate_entities")) == current
        checks["source_obligations_match"] = obligations(published.get("constraints"), current) == obligations(solution.get("constraints"), current)
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        issue = str(error)
    else:
        issue = None
    return {"verified": all(checks.values()), "checks": checks, "issue": issue,
            "prediction_sha256": manifest.get("prediction_sha256"),
            "solution_receipt_sha256": (manifest.get("frozen_artifacts") or {}).get("parametric-solution.json"),
            "scope": "Accepted solution and candidate geometry, source obligations, and immutable export receipts must agree; a rejected baseline match is insufficient."}


def _attribute_scopes(published, solution, bindings, inventory, link):
    constraints = published.get("constraints") or []
    counts = published.get("binding_counts") or {}
    records = inventory.get("all_records", inventory.get("records", []))
    supported = {"radius", "diameter", "length", "angle"}
    recognized_ids = {r.get("id") for r in records if isinstance(r, dict) and isinstance(r.get("id"), str)
                      and (r.get("parsed") or {}).get("kind") in supported}
    def count(value):
        return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None
    recognized = count(counts.get("recognized_dimensions"))
    if recognized is None:
        recognized = count((inventory.get("counts") or {}).get("recognized_dimensions"))
        if recognized is None and records: recognized = len(recognized_ids)
    attribute_kinds = {"radius", "distance", "distance_x", "distance_y", "angle"}
    bound_ids = {c.get("record_id") for c in constraints if isinstance(c, dict) and
                 c.get("kind") in attribute_kinds and isinstance(c.get("record_id"), str) and c["record_id"]}
    diagnostics = (published.get("solver") or {}).get("diagnostics") or {}
    solver_applies = bool(constraints and (published.get("solver") or {}).get("accepted") is True and
                          (published.get("constraint_subset_accepted") is True or published.get("accepted") is True))
    current = {"recognized_dimensions": recognized, "bound_source_records": len(bound_ids),
               "unbound_dimensions": max(0, recognized - len(bound_ids)) if recognized is not None else None,
               "formal_constraint_count": len(constraints), "bound_record_ids": sorted(bound_ids),
               "remaining_shape_dof": diagnostics.get("remaining_shape_dof") if solver_applies else None,
               "dof_source": "published_model_solver" if solver_applies else "not_certified_for_current_export",
               "recognized_source": "published_model" if count(counts.get("recognized_dimensions")) is not None else "source_ocr_inventory",
               "scope": "Current published model obligations only; no binding or DOF fallback to a later candidate."}
    attempt_counts = bindings.get("counts") or {}
    attempted = {key: attempt_counts.get(key) for key in ("recognized_dimensions", "bound_source_records", "unbound_dimensions")}
    attempted.update(formal_constraint_count=len(bindings.get("constraints") or []),
                     remaining_shape_dof=(solution.get("diagnostics") or {}).get("remaining_shape_dof"),
                     solver_status=solution.get("status"), solver_accepted=solution.get("accepted") is True,
                     applies_to_current_core=link["verified"],
                     diagnostics_geometry="accepted_solution" if solution.get("accepted") is True else "rejected_candidate",
                     scope="Latest binding and optimization attempt; rejected candidate diagnostics do not describe the retained DXF.")
    return current, attempted


def collect_slot(root, version, suffix, case_id):
    run = root / f"angle-semantics-{version}-{suffix}"
    manifest = read(run / "run-manifest.json")
    row = {"version": version, "case_id": case_id, "run": run.name, "status": "not_run",
           "full_user_target_passed": False, "files": {}, "ground_truth_opened": False, "online_api_called": False}
    if not manifest:
        row["status"] = "missing_manifest" if run.exists() else "not_run"
        return row
    row.update(status=manifest.get("status", "unknown"), last_stage=manifest.get("last_stage"),
               failure_type=manifest.get("failure_type"), elapsed=_elapsed(manifest))
    row["files"]["manifest"] = f"{run.name}/run-manifest.json"
    frozen = bool(manifest.get("prediction_frozen") is not False and manifest.get("prediction_frozen_at") and manifest.get("prediction_sha256"))
    if not frozen:
        row["artifact_state"] = "not_frozen_no_native_audit"
        return row
    base = collect_attempt(root, run)
    row.update(controlled_attempt=base, files=base.get("files", {}), artifact_state=base["artifact_state"],
               reference=base.get("reference", {}), native=base.get("native", {}),
               transport=base.get("transport", []), published=base.get("published_parameterization", {}))
    before = _frozen_receipts(run, manifest)
    row["frozen_receipt_integrity"] = before
    if not before["passed"] or not base.get("frozen_prediction_still_current"):
        row["strict_audit_status"] = "frozen_receipt_mismatch"
        return row
    after = run / "after"
    model, stage = read(after / "model.json"), read(after / "parametric-stage.json")
    solution, bindings = read(after / "parametric-solution.json"), read(after / "constraint-bindings.json")
    published = model.get("parameterization") or {}
    inventory = read(after / "binding-candidates.json")
    entities, constraints = model.get("entities", []), published.get("constraints", [])
    try:
        import ezdxf
        document = ezdxf.readfile(after / "drawing.dxf")
        strict = relation_checks(entities, constraints, dxf_document=document)
        angles = _native_angle_checks(entities, constraints, inventory, document)
        curved = _curved_join_audit(entities, constraints, document, strict)
        row.update(strict_audit_status="read_back", strict_relations=strict, angles=angles, curved_joins=curved)
    except (OSError, ValueError, TypeError, KeyError, IndexError) as error:
        row.update(strict_audit_status="readback_failed", strict_audit_failure_type=type(error).__name__)
        return row
    link = _solution_publication_link(model, solution, manifest, before)
    row["solution_publication_link"] = link
    row["attributes"], row["attempt_attributes"] = _attribute_scopes(published, solution, bindings, inventory, link)
    # collect_attempt is a legacy collector: its geometry-only equality can
    # match rejected solver baseline_entities. Correct the report's copy, not
    # the collector or any historical/evaluation artifact.
    current_receipt = base["published_parameterization"]
    current_receipt["geometry_only_solution_match"] = current_receipt.get("current_solution_receipt_matches_model")
    current_receipt["current_solution_receipt_matches_model"] = link["verified"]
    current_receipt.update({key: row["attributes"][key] for key in ("recognized_dimensions", "bound_source_records", "remaining_shape_dof")})
    if not link["verified"]:
        current_receipt["maximum_recorded_constraint_residual_by_unit"] = {"mm": None, "degree": None}
    row["online_binding"] = _binding_receipt(stage)
    row["source_reconstruction_contract"] = (model.get("validation") or {}).get("reconstruction_contract")
    row["files"].update({"binding_receipt": f"{run.name}/after/parametric-stage.json",
                          "bindings": f"{run.name}/after/constraint-bindings.json", "solver": f"{run.name}/after/parametric-solution.json"})
    final_integrity = _frozen_receipts(run, manifest)
    row["audit_inputs_unchanged"] = final_integrity == before
    row["full_user_target_passed"] = bool(base.get("full_user_target_passed") and
        row["attributes"]["formal_constraint_count"] > 0 and row["attributes"]["remaining_shape_dof"] == 0 and
        strict.get("passed") and strict.get("dxf_readback_performed") and not curved["mapping_issues"] and
        curved["unknown_count"] == 0 and angles["strict_native_satisfied_count"] == angles["recognized_count"] and
        row["audit_inputs_unchanged"])
    return row


def assemble(root, versions=VERSIONS):
    root = Path(root).resolve(strict=True)
    if isinstance(versions, str):
        raise ValueError("versions must be a nonempty unique sequence")
    versions = tuple(versions)
    if (not versions or any(not isinstance(value, str) or not re.fullmatch(r"v[0-9]+", value) for value in versions)
            or len(versions) != len(set(versions))):
        raise ValueError("versions must be a nonempty unique sequence of v-number names")
    rows = [collect_slot(root, version, suffix, case) for suffix, case in CASES for version in versions]
    comparisons = []
    for _, case in CASES:
        case_rows = [row for row in rows if row["case_id"] == case]
        for before, after in zip(case_rows, case_rows[1:]):
            old, new = before.get("controlled_attempt", {}), after.get("controlled_attempt", {})
            comparable = bool(old.get("input_sha256") and old.get("input_sha256") == new.get("input_sha256") and
                              old.get("reference", {}).get("reference_sha256") and
                              old.get("reference", {}).get("reference_sha256") == new.get("reference", {}).get("reference_sha256") and
                              old.get("code_unchanged_during_prediction") and new.get("code_unchanged_during_prediction") and
                              all(row.get("frozen_receipt_integrity", {}).get("passed") and row.get("audit_inputs_unchanged") and
                                  row.get("reference", {}).get("receipt_matches_current_dxf") and
                                  row.get("status") in {"completed", "completed_with_unresolved_attributes"}
                                  for row in (before, after)))
            comparison = {"case_id": case, "from_version": before["version"], "to_version": after["version"],
                          "same_frozen_inputs_and_reference": comparable,
                          "selection": "Adjacent versions in the requested order; no best-GT run selection",
                          "delta_direction": "to_version minus from_version",
                          "delta_scope": "Completed, unchanged frozen predictions with current independent evaluation receipts only",
                          "automatic_promotion": False}
            metrics = {"registered_max_delta_mm": ("reference", "registered_max_mm"),
                       "registered_rms_delta_mm": ("reference", "registered_rms_mm"),
                       "exact_bound_radius_delta": ("native", "exact_radius_count"),
                       "strict_angle_satisfied_delta": ("angles", "strict_native_satisfied_count"),
                       "strict_tangent_satisfied_delta": ("strict_relations", "satisfied_count"),
                       "unknown_curved_joins_delta": ("curved_joins", "unknown_count"),
                       "matched_primitive_count_1mm_delta": ("reference", "matched_primitive_count_1mm"),
                       "remaining_shape_dof_delta": ("attributes", "remaining_shape_dof")}
            for name, (section, key) in metrics.items():
                first, last = before.get(section, {}).get(key), after.get(section, {}).get(key)
                comparison[name] = last - first if comparable and numeric(first) and numeric(last) else None
            comparisons.append(comparison)
    return {"schema_version": "strict-joint-iteration-review-v1", "created_at": datetime.now(timezone.utc).isoformat(),
            "ground_truth_opened": False, "online_api_called": False, "condition": "GT-derived raster mask development experiment",
            "held_out": False, "versions": list(versions), "rows": rows, "comparisons": comparisons,
            "coverage": {"expected_slots": len(versions) * len(CASES), "fully_passed_slots": sum(row["full_user_target_passed"] for row in rows),
                         "not_run_slots": sum(row["status"] in {"not_run", "missing_manifest"} for row in rows),
                         "frozen_audited_slots": sum(row.get("strict_audit_status") == "read_back" for row in rows)},
            "limits": ["1 mm 仅为补充形状及图元检查；原有 0.1 mm 检查和标注半径严格相等要求不变。",
                       "本报告把已准入相切约束按 1e-7 度重新读回，不将旧版本 0.1 度通过收据视为严格相切。",
                       "未知接点保留在分母中；没有证据时，不因为看起来平滑就断言必须相切。",
                       f"失败、未运行及未冻结结果保留在 {len(versions) * len(CASES)} 个预定槽位；未冻结产物不做原生精度审计。",
                       "差值只比较指定列表中的相邻版本，方向为后版本减前版本；未运行、失败、未冻结或收据过期时不计算差值。",
                       "GT 仅由此前独立评估器读取；此脚本只读现有评估收据，不读取 GT DXF，也不调用 API。",
                       "每页 API 成功、数值约束通过、全标注覆盖和 GT 精度是不同判据；分页成本与旧单页调用不同。"]}


def render(report):
    esc = lambda value: html.escape(str(value), quote=True)
    def val(value):
        if value is None: return "—"
        if isinstance(value, bool): return "通过" if value else "未通过"
        return f"{value:.6g}" if isinstance(value, float) else esc(value)
    def ratio(first, second):
        return f"{val(first)} / {val(second)}"
    def links(row):
        titles = {"dxf": "DXF", "overlay": "GT 叠图", "manifest": "运行收据", "evaluation": "独立评估",
                  "binding_receipt": "API 收据", "bindings": "绑定", "solver": "求解", "audit": "本次原生审计"}
        return " · ".join(f'<a href="{esc(quote(path, safe="/"))}">{esc(titles.get(key, key))}</a>' for key, path in row.get("files", {}).items())
    table_rows, details = [], []
    for row in report["rows"]:
        native, angles, strict = [row.get(key, {}) for key in ("native", "angles", "strict_relations")]
        reference, attributes, curved = [row.get(key, {}) for key in ("reference", "attributes", "curved_joins")]
        attempt = row.get("attempt_attributes", {})
        types = native.get("native_types", {})
        object_count = native.get("native_object_count")
        cells = [row["case_id"], row["version"], row["status"],
                 " + ".join(f"{count} {kind}" for kind, count in sorted(types.items())) or "—",
                 ratio(native.get("exact_radius_count"), native.get("recognized_radius_count")),
                 ratio(angles.get("strict_native_satisfied_count"), angles.get("recognized_count")),
                 ratio(strict.get("satisfied_count"), strict.get("required_count")),
                 ratio(curved.get("unknown_count"), curved.get("curved_join_count")),
                 val(reference.get("registered_max_mm")), val(reference.get("registered_rms_mm")),
                 ratio(reference.get("matched_primitive_count_1mm"), object_count),
                 ratio(attributes.get("bound_source_records"), attributes.get("recognized_dimensions")),
                 val(attributes.get("remaining_shape_dof")),
                 ratio(attempt.get("bound_source_records"), attempt.get("recognized_dimensions")),
                 val(attempt.get("remaining_shape_dof")), val(attempt.get("solver_status")),
                 val(row.get("elapsed", {}).get("prediction_seconds")),
                 "完整通过" if row["full_user_target_passed"] else "未完整达标"]
        table_rows.append("<tr>" + "".join(f"<td>{esc(value)}</td>" for value in cells) + "</tr>")
        online = row.get("online_binding", {})
        online_text = (f"绑定实际发送 {val(online.get('actual_sent_record_count'))}/{val(online.get('all_record_count'))} 条 OCR；"
                       f"关系 {val(online.get('actual_sent_relation_count'))}/{val(online.get('all_relation_count'))}；"
                       f"实际 {val(online.get('actual_page_count'))} 页，HTTP {val(online.get('http_success'))}，"
                       f"全部页结构 {val(online.get('all_pages_succeeded'))}。有效页子集 {val(online.get('valid_response_record_count'))} 条，"
                       f"绑定耗时 {val(online.get('elapsed_seconds'))} 秒；运行总耗时 {val(row.get('elapsed', {}).get('total_run_seconds'))} 秒。")
        small_rows = []
        for joint in curved.get("checks", []):
            measure = joint.get("native_geometry") or {}
            small_rows.append("<tr>" + "".join(f"<td>{esc(value)}</td>" for value in (
                " / ".join(joint["entities"]), " / ".join(joint["types"]), joint["relationship_status"],
                val(measure.get("angle_residual_deg")), val(measure.get("endpoint_gap")))) + "</tr>")
        tables = '<table><thead><tr><th>连接对象</th><th>类型</th><th>关系证据</th><th>切向偏折 °</th><th>端点间距</th></tr></thead><tbody>' + "".join(small_rows) + '</tbody></table>' if small_rows else '<p>暂无可审计的冻结产物。</p>'
        overlay = row.get("files", {}).get("overlay")
        image = f'<a href="{esc(quote(overlay, safe="/"))}"><img loading="lazy" src="{esc(quote(overlay, safe="/"))}" alt="{esc(row["case_id"])} {esc(row["version"])} 独立比较叠图"></a>' if overlay else ""
        details.append(f'<section><h2>{esc(row["case_id"])} · {esc(row["version"])}</h2><p>{links(row)}</p><p>{online_text}</p>{image}'
                       '<details><summary>当前发布义务与最新候选诊断（分别记录）</summary><pre>' +
                       esc(json.dumps({"current_export_attributes": attributes, "latest_attempt_attributes": attempt,
                                       "solution_publication_link": row.get("solution_publication_link")}, ensure_ascii=False, indent=2)) + '</pre></details>'
                       f'<details><summary>所有含圆弧的接点（包括 ARC–ARC）</summary>{tables}</details>'
                       '<details><summary>API 分页与阶段收据</summary><pre>' + esc(json.dumps({"binding": online, "stages": row.get("transport", [])}, ensure_ascii=False, indent=2)) + '</pre></details></section>')
    coverage = report["coverage"]
    version_title = " / ".join(version.upper() for version in report.get("versions", VERSIONS))
    comparison_rows = []
    for pair in report["comparisons"]:
        cells = [pair["case_id"], f'{pair["from_version"]} → {pair["to_version"]}',
                 "可比" if pair["same_frozen_inputs_and_reference"] else "暂无有效比较",
                 val(pair.get("registered_max_delta_mm")), val(pair.get("registered_rms_delta_mm")),
                 val(pair.get("matched_primitive_count_1mm_delta")), val(pair.get("strict_tangent_satisfied_delta")),
                 val(pair.get("unknown_curved_joins_delta")), val(pair.get("remaining_shape_dof_delta"))]
        comparison_rows.append("<tr>" + "".join(f"<td>{esc(value)}</td>" for value in cells) + "</tr>")
    comparison_table = ('<section><h2>相邻版本变化</h2><p>以下差值均为后版本减前版本。缺少有效的同输入冻结收据时显示“—”，不把未运行当作零。</p>'
                        '<div class="scroll"><table><thead><tr><th>图纸</th><th>版本</th><th>可比性</th><th>最大误差变化 mm</th><th>RMS 变化 mm</th>'
                        '<th>完整图元变化</th><th>严格 G1 通过数变化</th><th>未知接点变化</th><th>形状自由度变化</th></tr></thead><tbody>'
                        + "".join(comparison_rows) + '</tbody></table></div></section>')
    headings = ["图纸", "版本", "运行状态", "原生对象", "精确 R / 已识别", "精确角 / 已识别", "严格 G1 / 已准入", "未知接点 / 含弧接点",
                "最大误差 mm", "RMS mm", "完整图元 1 mm", "当前绑定 / 已识别", "当前形状自由度",
                "最新候选绑定 / 已识别", "最新候选自由度", "最新求解状态", "预测秒数", "完整目标"]
    return ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>标注与严格接点约束：{esc(version_title)} 在线测试</title><style>body{{font:15px/1.6 system-ui,sans-serif;margin:28px;color:#243540;background:#f6f8fa}}h1{{font-size:25px}}h2{{font-size:20px}}section{{margin:28px 0;padding:18px;background:white;border:1px solid #dfe5ea}}table{{border-collapse:collapse;width:100%;font-size:12px;background:white}}th,td{{border:1px solid #dfe5ea;padding:8px;text-align:left;white-space:nowrap}}th{{background:#eaf1f4}}.scroll{{overflow:auto}}a{{color:#086b7a}}img{{display:block;max-width:100%;max-height:600px;margin:14px auto}}pre{{white-space:pre-wrap;font-size:12px}}details{{margin-top:16px}}p{{max-width:1200px}}</style>'
            f'<h1>标注与严格接点约束：{esc(version_title)} 在线测试</h1><p>固定原图、OCR 和 GT 派生栅格掩膜的开发实验。轮廓接近不等于对象属性、连接关系或标注覆盖完整。该报告只重新审计已冻结的产物。</p>'
            '<p>当前绑定与自由度仅描述已发布 DXF。最新候选即使已绑定标注，求解失败后也不能把其绑定数或自由度计入保留的轮廓；当前未认证自由度显示“—”。</p>'
            f'<p><strong>完整目标通过 {coverage["fully_passed_slots"]} / {coverage["expected_slots"]}</strong>；冻结原生审计 {coverage["frozen_audited_slots"]} 项，未运行 {coverage["not_run_slots"]} 项。失败与缺失结果没有从分母删除。</p>'
            '<div class="scroll"><table><thead><tr>' + "".join(f"<th>{esc(value)}</th>" for value in headings) + '</tr></thead><tbody>' + "".join(table_rows) + '</tbody></table></div>'
            + comparison_table + "".join(details) + '<section><h2>判读边界</h2><ul>' + "".join(f"<li>{esc(value)}</li>" for value in report["limits"]) + '</ul></section></html>')


def write_report(root, output_prefix="strict-joint-review-20261010", versions=VERSIONS):
    root = Path(root).resolve(strict=True)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", output_prefix):
        raise ValueError("output prefix must be a local filename stem")
    report = assemble(root, versions=versions)
    audit_dir = root / (output_prefix + "-audits")
    audit_dir.mkdir(exist_ok=True)
    for row in report["rows"]:
        if row.get("strict_audit_status") != "read_back":
            continue
        filename = row["run"] + "-" + row["native"]["sha256"][:12] + ".json"
        audit = {key: row.get(key) for key in ("run", "version", "case_id", "strict_relations", "angles", "curved_joins",
                                             "attributes", "attempt_attributes", "solution_publication_link",
                                             "frozen_receipt_integrity", "audit_inputs_unchanged")}
        audit.update(prediction_sha256=row["native"]["sha256"], created_at=report["created_at"],
                     ground_truth_opened=False, online_api_called=False,
                     auditor_sha256=digest(Path(__file__)), relation_checker_sha256=digest(PROJECT_ROOT / "contour_agent/relation_contract.py"))
        (audit_dir / filename).write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
        row["files"]["audit"] = (audit_dir / filename).relative_to(root).as_posix()
    json_path, html_path = root / (output_prefix + ".json"), root / (output_prefix + ".html")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
    html_path.write_text(render(report), encoding="utf8")
    return report, json_path, html_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-prefix", default="strict-joint-review-20261010")
    parser.add_argument("--versions", nargs="+", default=list(VERSIONS), help="Ordered version names, for example v11 v12 v13")
    args = parser.parse_args(argv)
    report, json_path, html_path = write_report(args.root, args.output_prefix, versions=args.versions)
    print(json.dumps({"coverage": report["coverage"], "json": str(json_path), "html": str(html_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
