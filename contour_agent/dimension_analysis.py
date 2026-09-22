"""Source-only dimension inventory and optional bounded LLM interpretation.

Parsing agreement is separate from geometric constraint satisfaction. The
provider sees only OCR text/IDs; it never supplies CAD points or reference data.
"""
from __future__ import annotations

from .ocr import canonical_records
from .provider import ProviderError

FIELDS = ("kind", "nominal", "upper_deviation", "lower_deviation")


def analyze_dimensions(document, model, *, provider=None, use_api=False):
    records = canonical_records(document)
    scale = model.get("scale") or {}
    bindings = {}
    for item in scale.get("bindings", []):
        if item.get("record_id"):
            bindings.setdefault(item["record_id"], []).append({"role": "scale_evidence", "axis": scale.get("axis")})
    for entity in model.get("entities", []):
        binding = entity.get("radius_binding") or {}
        if binding.get("record_id"):
            bindings.setdefault(binding["record_id"], []).append({
                "role": "local_radius_fit", "entity_id": entity.get("id"),
                "nominal": binding.get("nominal"), "fitted_radius": entity.get("radius"),
                "units": (model.get("coordinate_system") or {}).get("units"),
            })
    decisions = [{"id": row["id"], "text": str(row.get("text", ""))[:500],
                  "box": row.get("box"), "local": {key: row["parsed"].get(key) for key in FIELDS},
                  "bindings": bindings.get(row["id"], []), "provider_agrees": None,
                  "geometric_constraint_verified": False}
                 for row in records if row["parsed"]["kind"] != "unknown"]
    selected = sorted(decisions, key=lambda row: (not bool(row["bindings"]),
                      {"radius": 0, "diameter": 1, "length": 2, "angle": 3}.get(row["local"]["kind"], 4), row["id"]))[:16]
    receipt = {"status": "disabled", "network_requests": 0, "http_success": False,
               "schema_success": False, "ground_truth_sent": False, "image_sent": False}
    if use_api and selected:
        try:
            if provider is None:
                raise ProviderError("not_configured", "尺寸解析服务尚未配置。")
            receipt = provider.normalize([{"id": row["id"], "text": row["text"]} for row in selected])
            # Exact source agreement only; a model proposal cannot override an
            # OCR value or manufacture an unobserved dimensional constraint.
            parsed = {row["id"]: row for row in receipt["dimensions"]}
            for row in selected:
                proposal = parsed.get(row["id"])
                if proposal:
                    row["provider_proposal"] = {key: proposal.get(key) for key in FIELDS}
                    row["provider_agrees"] = all(proposal.get(key) == row["local"].get(key) for key in FIELDS)
        except ProviderError as error:
            receipt = {"status": "failed", "error_code": error.code, "http_status": error.status_code,
                       "network_requests": error.network_requests, "http_success": error.http_success,
                       "schema_success": False, "ground_truth_sent": False, "image_sent": False,
                       "elapsed_seconds": error.elapsed_seconds, "model": error.model,
                       "protocol": error.protocol}
        except Exception:
            # A failed auxiliary parser must not discard an already valid CAD.
            # Never publish arbitrary provider exception strings or raw bodies.
            receipt = {"status": "failed", "error_code": "dimension_analysis_error",
                       "network_requests": None, "http_success": None, "schema_success": False,
                       "ground_truth_sent": False, "image_sent": False}
    elif use_api:
        receipt["status"] = "skipped"
        receipt["reason"] = "no_parseable_source_dimensions"
    agreed = sum(row["provider_agrees"] is True for row in decisions)
    conflicts = sum(row["provider_agrees"] is False for row in decisions)
    return {"status": "completed", "provider": receipt, "decisions": decisions,
            "counts": {"ocr_records": len(records), "recognized_dimensions": len(decisions),
                       "bound_source_records": sum(bool(row["bindings"]) for row in decisions),
                       "api_selected": len(selected) if use_api else 0, "api_agreed": agreed, "api_conflicts": conflicts,
                       "unbound_dimensions": sum(not row["bindings"] for row in decisions)},
            "scale_status": scale.get("status"), "units": (model.get("coordinate_system") or {}).get("units"),
            "geometry_updated_by_api": False, "dimensions_verified": False, "ground_truth_used": False,
            "scope": "OCR interpretation and existing local scale/radius bindings; not a full dimensional constraint solver."}
