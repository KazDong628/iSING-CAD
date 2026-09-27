"""Bounded screenshot-guided selection among existing source-only topology candidates."""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from pathlib import Path

import httpx

from .api_wire import (extract_text, prepare_request, request_headers, endpoint_allowed,
                       anthropic_thinking_mode_requested)
from .planning_provider import _image_payload, evaluate_candidates


PROMPT = """You are reviewing a 2D mechanical main-contour reconstruction after a user marked or cropped a bad region.
Choose only one candidate from the supplied admissible candidate IDs, or abstain. The images are ordered as stated in the JSON packet: user feedback screenshot, current exported overlay, then candidate overlays. Ground truth is unavailable. Never invent coordinates or dimensions. If and only if the user's instruction explicitly says a visible boundary side must be one straight segment, add a bounded high-level operation; the local geometry engine will locate endpoints from existing source-only topology and may reject it. Radius-labelled entities listed in protected_radius_entities must remain arcs; never flatten them merely because the raster edge is rough. Internal hatching is never a main-contour primitive.
Return exactly one JSON object with keys candidate_id, observation, proposed_action, evidence_tags, rationale_code, confidence, operations. candidate_id is a listed ID or null. observation and proposed_action are concise Chinese strings of at most 160 characters describing visible evidence and the auditable action, not hidden reasoning. evidence_tags is a list containing only user_feedback, source_boundary, annotation_leader, continuity, primitive_count, hatching_excluded, user_geometry_instruction. operations is a list of at most four objects with exactly action, side, basis. action is replace_boundary_chain_with_line or exclude_hatching_from_boundary; side is top, right, bottom, left, or crop; basis is a concise Chinese quote-like paraphrase of the user's explicit instruction, at most 120 characters. rationale_code is one of feedback_region_improved, restore_visible_detail, simplify_unsupported_primitives, preserve_current, ambiguous_feedback. confidence is low, medium, high, or abstain. No Markdown or extra keys."""


class FeedbackProvider:
    def __init__(self, settings):
        self.settings = settings

    def select(self, screenshot_path, current_overlay, candidates, instruction, current_candidate_id=None):
        return asyncio.run(self._select(screenshot_path, current_overlay, candidates, instruction, current_candidate_id))

    async def _select(self, screenshot_path, current_overlay, candidates, instruction, current_candidate_id):
        started, settings = time.monotonic(), self.settings
        local = evaluate_candidates(candidates, max_candidates=5)
        allowed = set(local["bounded_candidate_ids"])
        receipt = {
            "status": "failed", "protocol": settings.wire_api + "-screenshot-topology-revision-v1",
            "model": settings.model, "network_requests": 0, "http_success": False, "schema_success": False,
            "image_sent": False, "ground_truth_sent": False, "coordinates_sent": False,
            "selected_candidate_id": None, "observation": "", "proposed_action": "",
            "evidence_tags": [], "operations": [], "rationale_code": "ambiguous_feedback", "confidence": "abstain",
            "current_candidate_id": current_candidate_id, "input_candidate_ids": local["bounded_candidate_ids"],
            "total_timeout_seconds": min(600.0, max(0.001, float(settings.api_timeout))),
            "anthropic_thinking_mode_requested": anthropic_thinking_mode_requested(settings),
        }

        def finish():
            receipt["elapsed_seconds"] = round(time.monotonic() - started, 3)
            return receipt

        if not settings.api_key:
            receipt.update(status="not_configured", error_code="not_configured")
            return finish()
        if not endpoint_allowed(settings):
            receipt["error_code"] = "invalid_endpoint"
            return finish()
        if not allowed:
            receipt.update(status="skipped", error_code="no_admissible_candidates")
            return finish()
        by_id = {row.get("id"): row for row in candidates if isinstance(row, dict)}
        try:
            images = []
            metadata = []
            for label, path in [("feedback", screenshot_path), ("current", current_overlay)]:
                payload, meta = _image_payload(Path(path))
                images.append({"type": "image_url", "image_url": {"url": payload}})
                metadata.append({"kind": label, **meta})
            for candidate_id in local["bounded_candidate_ids"]:
                payload, meta = _image_payload(Path(by_id[candidate_id]["overlay_path"]))
                images.append({"type": "image_url", "image_url": {"url": payload}})
                metadata.append({"kind": "candidate", "candidate_id": candidate_id, **meta})
        except (OSError, ValueError, TypeError, KeyError):
            receipt["error_code"] = "invalid_revision_images"
            return finish()
        packet = {
            "instruction": str(instruction).strip()[:1200],
            "current_candidate_id": current_candidate_id,
            "candidate_order": local["bounded_candidate_ids"],
            "candidates": local["bounded_candidates"],
            "protected_radius_entities": {
                candidate_id: sorted({row.get("candidate_entity_id")
                                      for row in (by_id[candidate_id].get("graph") or {}).get(
                                          "annotation_support", [])
                                      if isinstance(row, dict) and row.get("kind") == "radius"
                                      and row.get("status") == "candidate_supported"
                                      and next((entity.get("type") for entity in
                                                (by_id[candidate_id].get("graph") or {}).get("entities", [])
                                                if entity.get("id") == row.get("candidate_entity_id")), None) == "ARC"
                                      and row.get("candidate_entity_id")})
                for candidate_id in local["bounded_candidate_ids"]
            },
        }
        packet_text = json.dumps(packet, ensure_ascii=False, separators=(",", ":"))
        receipt.update(image_sent=True, input_images=metadata,
                       packet_sha256=hashlib.sha256(packet_text.encode()).hexdigest())
        payload = {
            "model": settings.model, "temperature": 0, "max_tokens": 700,
            "messages": [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": [{"type": "text", "text": packet_text}, *images]},
            ],
        }
        endpoint, wire_payload = prepare_request(settings, payload)
        budget = receipt["total_timeout_seconds"]
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(budget, connect=min(10.0, budget)),
                                         trust_env=settings.trust_env, verify=True,
                                         follow_redirects=False) as client:
                receipt["network_requests"] = 1
                response = await client.post(endpoint, headers=request_headers(settings), json=wire_payload)
                receipt.update(http_status=response.status_code, http_success=response.status_code == 200)
                if response.status_code != 200:
                    receipt["error_code"] = "http_error"
                    return finish()
                text, source, reason, _ = extract_text(settings, response)
                receipt["response_text_source"] = source
                receipt["finish_reason"] = reason
                value = json.loads(text)
                required = {"candidate_id", "observation", "proposed_action", "evidence_tags", "rationale_code", "confidence", "operations"}
                if not isinstance(value, dict) or set(value) != required:
                    raise ValueError("schema_mismatch")
                candidate_id = value["candidate_id"]
                if candidate_id is not None and candidate_id not in allowed:
                    raise ValueError("unknown_candidate")
                if not all(isinstance(value[name], str) and len(value[name]) <= 160 for name in ("observation", "proposed_action")):
                    raise ValueError("invalid_summary")
                tags = value["evidence_tags"]
                allowed_tags = {"user_feedback", "source_boundary", "annotation_leader", "continuity", "primitive_count", "hatching_excluded", "user_geometry_instruction"}
                if not isinstance(tags, list) or len(tags) > 8 or not all(isinstance(tag, str) for tag in tags):
                    raise ValueError("invalid_tags")
                # Evidence tags are explanatory metadata. Keep the provider compatible
                # when it adds a harmless synonym, while geometry operations and the
                # selected candidate remain strictly allow-listed below.
                discarded_tag_count = sum(tag not in allowed_tags for tag in tags)
                tags = list(dict.fromkeys(tag for tag in tags if tag in allowed_tags))[:5]
                operations = value["operations"]
                allowed_actions = {"replace_boundary_chain_with_line", "exclude_hatching_from_boundary"}
                allowed_sides = {"top", "right", "bottom", "left", "crop"}
                if not isinstance(operations, list) or len(operations) > 4:
                    raise ValueError("invalid_operations")
                for operation in operations:
                    if (not isinstance(operation, dict) or set(operation) != {"action", "side", "basis"}
                        or operation.get("action") not in allowed_actions or operation.get("side") not in allowed_sides
                        or not isinstance(operation.get("basis"), str) or len(operation["basis"]) > 120):
                        raise ValueError("invalid_operations")
                if value["rationale_code"] not in {"feedback_region_improved", "restore_visible_detail", "simplify_unsupported_primitives", "preserve_current", "ambiguous_feedback"}:
                    raise ValueError("invalid_rationale")
                if value["confidence"] not in {"low", "medium", "high", "abstain"}:
                    raise ValueError("invalid_confidence")
                receipt.update(status="succeeded", schema_success=True, selected_candidate_id=candidate_id,
                               observation=value["observation"], proposed_action=value["proposed_action"],
                               evidence_tags=tags, operations=operations,
                               discarded_evidence_tag_count=discarded_tag_count,
                               rationale_code=value["rationale_code"], confidence=value["confidence"],
                               response_text_sha256=hashlib.sha256(text.encode()).hexdigest(), response_text_chars=len(text))
        except (asyncio.TimeoutError, httpx.TimeoutException):
            receipt["error_code"] = "timeout"
        except httpx.TransportError:
            receipt["error_code"] = "transport_error"
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            receipt["error_code"] = str(error) if str(error) in {
                "schema_mismatch", "unknown_candidate", "invalid_summary", "invalid_tags", "invalid_operations", "invalid_rationale", "invalid_confidence"
            } else "invalid_response"
        return finish()
