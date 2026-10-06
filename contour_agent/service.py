"""Small agent state machine: extract -> review -> solve -> independently validate."""
from __future__ import annotations
import hashlib
import io
import json
import math
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image
from .config import Settings, provider_registry
from .dataset import build_catalog, read_ocr, resolve_inside
from .geometry import solve_profile, template_schema
from .ocr import bind_parameters
from .provider import DimensionProvider, ProviderError
from .store import JobStore
from .automatic import build_automatic
from .vision_provider import VisionProvider
from .dimension_analysis import analyze_dimensions
from .parametric_pipeline import refine_parametric
from .feedback_provider import FeedbackProvider
from .feedback_geometry import create_feedback_candidate

def now():
    return datetime.now(timezone.utc).isoformat()

def digest(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _parameterization_needs_review(job):
    stage=job.get("parameterization") or {}
    if not stage:return False
    contract=stage.get("annotation_radius_contract")
    return bool(stage.get("accepted") is not True or
                (contract is not None and (not contract.get("satisfied") or
                                          contract.get("publication_status")=="candidate_only")))

class AgentService:
    def __init__(self, settings: Settings, *, recover_running=False):
        self.settings = settings
        self.provider_profiles, self.default_provider_id = provider_registry(settings)
        self.catalog = build_catalog(settings.dataset_root)
        self.cases = {row["id"]: row for row in self.catalog["cases"]}
        self.store = JobStore(settings.runtime_root)
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="contour")
        self.lock = threading.RLock()
        self._futures = {}
        self._deleted_job_ids = set()
        active_settings = self.provider_profiles[self.default_provider_id].settings(settings)
        self.provider = DimensionProvider(active_settings)
        self.automatic_dimension_provider = DimensionProvider(active_settings, max_attempts=1)
        self.vision_provider = VisionProvider(active_settings)
        from .binding_provider import BindingProvider
        from .planning_provider import PlanningProvider
        from .topology_edit_provider import TopologyEditProvider, TopologyEvaluationProvider
        self.binding_provider = BindingProvider(active_settings)
        self.planning_provider = PlanningProvider(active_settings)
        self.topology_edit_provider = TopologyEditProvider(active_settings)
        self.topology_evaluation_provider = TopologyEvaluationProvider(active_settings)
        self.feedback_provider = FeedbackProvider(active_settings)
        self._provider_clients = {self.default_provider_id: {
            "dimension": self.provider, "automatic_dimension": self.automatic_dimension_provider,
            "vision": self.vision_provider, "binding": self.binding_provider,
            "planning": self.planning_provider, "topology_edit": self.topology_edit_provider,
            "topology_evaluate": self.topology_evaluation_provider, "feedback": self.feedback_provider,
        }}
        for job in self.store.list(1000) if recover_running else []:
            if job["status"] in {"queued", "extracting", "solving", "auditing"}:
                if job.get("mode") == "autonomous_image":
                    previous_status = job["status"]
                    directory = Path(job["artifact_directory"]) if job.get("artifact_directory") else None
                    def saved_object(name):
                        if not directory:return {}
                        try:
                            value=json.loads((directory/name).read_text(encoding="utf8"))
                            return value if isinstance(value,dict) else {}
                        except (OSError,ValueError):return {}
                    terminal_receipts={"succeeded","failed","not_configured","skipped"}
                    analysis=job.get("dimension_analysis") or {}
                    persisted_analysis=saved_object("dimension-analysis.json")
                    if (persisted_analysis.get("provider") or {}).get("status") in terminal_receipts:
                        analysis=persisted_analysis
                        job["dimension_analysis"]=analysis
                    dimension_receipt=analysis.get("provider") or {}
                    if dimension_receipt.get("status")=="pending":
                        dimension_receipt.update(status="interrupted",error_code="service_restart",
                                                 network_requests=None,http_success=None,schema_success=False)
                        analysis["provider"]=dimension_receipt
                        job["dimension_analysis"]=analysis
                    parametric=saved_object("parametric-stage.json") or job.get("parameterization") or {}
                    persisted_bindings=saved_object("constraint-bindings.json")
                    persisted_receipt=persisted_bindings.get("provider") or {}
                    if persisted_receipt.get("status") in terminal_receipts:
                        parametric.update(provider=persisted_receipt,
                                          binding_counts=persisted_bindings.get("counts",{}),
                                          constraints=persisted_bindings.get("constraints",[]))
                    binding_receipt=parametric.get("provider") or {}
                    if binding_receipt.get("status")=="pending":
                        binding_receipt.update(status="interrupted",network_requests=None,http_success=None,
                                               schema_success=False,error_code="service_restart")
                        parametric.update(provider=binding_receipt)
                    if parametric:
                        if parametric.get("status") in {"running","pending"}:
                            parametric.update(status="interrupted",accepted=False,
                                              geometry_updated_by_api=False,dimensions_updated_by_api=False)
                        job["parameterization"]=parametric
                    published=False
                    if directory and directory.is_dir():
                        published=self._recover_automatic_geometry(job,directory)
                        self._publish_automatic_artifacts(job,directory)
                        if job.get("parameterization"):
                            from .parametric_pipeline import _write
                            _write(directory/"parametric-stage.json",job["parameterization"])
                    job["status"] = ("completed" if job.get("automatic_completion") else "needs_review") if published else "failed"
                    if published and _parameterization_needs_review(job):job["status"]="needs_review"
                    job["automatic_completion"] = bool(published and job.get("automatic_completion"))
                    if not published:
                        job["validation"]={**(job.get("validation") or {}),"passed":False,"recovery_integrity_verified":False}
                        job.setdefault("issues",[]).append("重启后未能验证一致的DXF、模型和验证文件，产物保留为未通过检查。")
                    if previous_status == "auditing" and (job.get("provider") or {}).get("status") != "succeeded":
                        receipt = dict(job.get("provider") or {})
                        receipt.update(status="interrupted", error_code="service_restart", schema_success=False,
                                       verdict="uncertain", dimension_certified=False, network_request_state="unknown",
                                       network_requests=None,http_success=None)
                        job["provider"] = receipt
                        job.setdefault("issues", []).append("服务重启使在线视觉复核中断，不能计为在线核验通过。")
                    if directory and directory.is_dir() and job.get("dimension_analysis"):
                        self._save_dimension_analysis(job,directory)
                    self._event(job, "recovered", "服务重启；已完成的自动产物已保留，未重新调用在线接口。" if published
                                else "服务重启；自动识别尚未完成，已保留阶段记录，可重新运行。")
                else:
                    job["status"] = "needs_review" if job.get("parameters") else "failed"
                    self._event(job, "recovered", "服务重启；已完成阶段保留，请检查参数后重新确认。")
                self.store.save(job)

    def _event(self, job, stage, message):
        job["updated_at"] = now()
        job.setdefault("events", []).append({"stage": stage, "message": message, "time": job["updated_at"]})

    def _persist_if_active(self, job):
        with self.lock:
            try:
                current = self.store.get(job["id"])
            except KeyError:
                return False
            if current["status"] == "cancelled" or current.get("solve_revision", 0) != job.get("solve_revision", 0):
                return False
            self.store.save(job)
            return True

    def _job_directory(self, job_id):
        jobs_root = (self.settings.runtime_root / "jobs").resolve()
        target = (jobs_root / job_id).resolve()
        if target.parent != jobs_root:
            raise ValueError("任务产物目录无效。")
        return target

    def _submit(self, job_id, function, *args):
        """Track workers so deleting a cancelled task can clean late writes."""
        future = self.executor.submit(function, *args)
        if not hasattr(future, "add_done_callback"):
            return future
        with self.lock:
            self._futures[job_id] = future

        def finished(done):
            with self.lock:
                if self._futures.get(job_id) is done:
                    self._futures.pop(job_id, None)
                cleanup = job_id in self._deleted_job_ids
                self._deleted_job_ids.discard(job_id)
            if cleanup:
                target = self._job_directory(job_id)
                if target.exists():
                    shutil.rmtree(target)

        future.add_done_callback(finished)
        return future

    def _select_provider(self, provider_id, use_api):
        selected = str(provider_id or self.default_provider_id)
        if selected not in self.provider_profiles:
            raise ValueError("未知在线模型配置。")
        if provider_id and use_api and not self.provider_profiles[selected].api_key:
            raise ValueError("所选在线模型尚未在服务端配置密钥。")
        return selected

    def _provider_bundle(self, job):
        provider_id = job.get("provider_id") or self.default_provider_id
        if provider_id in self._provider_clients:
            return self._provider_clients[provider_id]
        profile = self.provider_profiles.get(provider_id)
        if profile is None:
            raise ValueError("任务引用的在线模型配置不存在。")
        from .binding_provider import BindingProvider
        from .planning_provider import PlanningProvider
        from .topology_edit_provider import TopologyEditProvider, TopologyEvaluationProvider
        configured = profile.settings(self.settings)
        bundle = {"dimension": DimensionProvider(configured),
                  "automatic_dimension": DimensionProvider(configured, max_attempts=1),
                  "vision": VisionProvider(configured), "binding": BindingProvider(configured),
                  "planning": PlanningProvider(configured),
                  "topology_edit": TopologyEditProvider(configured),
                  "topology_evaluate": TopologyEvaluationProvider(configured),
                  "feedback": FeedbackProvider(configured)}
        self._provider_clients[provider_id] = bundle
        return bundle

    def _provider_metadata(self, provider_id):
        return self.provider_profiles[provider_id].public(default=provider_id == self.default_provider_id)

    def _new(self, case_id, template_id, image_path, ocr_path, use_api, uploaded=False, provider_id=None):
        schema = template_schema()
        supported = template_id == schema["id"]
        provider_id = self._select_provider(provider_id, use_api)
        job_id = uuid.uuid4().hex
        job = {"id": job_id, "status": "queued" if supported else "unsupported", "case_id": case_id,
               "template_id": template_id, "created_at": now(), "updated_at": now(), "use_api": bool(use_api),
               "uploaded": uploaded, "source_image": str(image_path), "source_ocr": str(ocr_path),
               "source_image_url": f"/api/jobs/{job_id}/image", "parameters": [], "assumptions": [],
               "events": [], "issues": [], "artifacts": {}, "validation": None,
               "provider": {"status": "pending" if use_api else "disabled", "network_requests": 0},
               "geometry": None, "engineering_accepted": False, "automatic_completion": False,
               "manual_confirmation": None, "split": "calibration" if case_id == schema["calibration_case"] else "unqualified",
               "provider_id": provider_id, "provider_profile": self._provider_metadata(provider_id),
               "scope": "template_assisted_main_profile_with_simplified_tread"}
        if not supported:
            job["issues"] = ["当前只有293已注册模板；该图尚不支持自动重建，不会按文件相似度套用。"]
            self._event(job, "unsupported", job["issues"][0])
        else:
            self._event(job, "queued", "已创建任务：只使用原图、OCR和声明模板；不读取参考DXF。")
        self.store.save(job)
        return job

    def create_case(self, case_id: str, use_api=True, template_id: str | None = None, *, asynchronous=True,
                    provider_id=None):
        case = self.cases.get(case_id)
        if not case:
            raise ValueError("未知数据集样本。")
        if not case.get("image") or not case.get("ocr"):
            raise ValueError("样本缺少图像或 OCR。")
        # A catalog sample cannot be silently reclassified into the calibration template.
        supported = case.get("supported_template")
        if template_id and template_id != supported:
            raise ValueError("该数据集样本未注册此模板；请通过上传模式明确审查新图纸。")
        job = self._new(case_id, supported, resolve_inside(self.settings.dataset_root, case["image"]),
                        resolve_inside(self.settings.dataset_root, case["ocr"]), use_api, provider_id=provider_id)
        if job["status"] == "queued":
            if asynchronous:
                self._submit(job["id"], self._extract, job["id"])
            else:
                self._extract(job["id"])
        return self.store.get(job["id"])

    def create_auto_case(self, case_id: str, use_api=True, *, asynchronous=True, use_segmentation=False,
                         require_segmentation_review=False, conversation_id=None, provider_id=None):
        case = self.cases.get(case_id)
        if not case or not case.get("image") or not case.get("ocr"):
            raise ValueError("样本不存在或缺少图像及OCR文件。")
        return self.create_auto_source(resolve_inside(self.settings.dataset_root,case["image"]),
                                       resolve_inside(self.settings.dataset_root,case["ocr"]),use_api,
                                       case_id=case_id,asynchronous=asynchronous,use_segmentation=use_segmentation,
                                       require_segmentation_review=require_segmentation_review,
                                       conversation_id=conversation_id, provider_id=provider_id)

    def create_auto_source(self, image: Path, ocr: Path, use_api=True, *, case_id="upload", asynchronous=True,
                           use_segmentation=False, require_segmentation_review=False, conversation_id=None,
                           provider_id=None):
        if use_segmentation and not (self.settings.segmentation_checkpoint and Path(self.settings.segmentation_checkpoint).is_file()):
            raise ValueError("尚未配置可用的分割模型checkpoint，请先完成训练。")
        provider_id=self._select_provider(provider_id,use_api)
        job_id=uuid.uuid4().hex
        job={"id":job_id,"mode":"autonomous_image","status":"queued","case_id":case_id,"template_id":None,
             "created_at":now(),"updated_at":now(),"use_api":bool(use_api),"uploaded":case_id=="upload",
             "source_image":str(image),"source_ocr":str(ocr),"source_image_url":f"/api/jobs/{job_id}/image",
             "parameters":[],"assumptions":[],"events":[],"issues":[],"artifacts":{},"validation":None,
             "provider":{"status":"pending" if use_api else "disabled","network_requests":0},
             "geometry":None,"engineering_accepted":False,"automatic_completion":False,"manual_confirmation":None,
             "manual_intervention":False,"split":"calibration" if case_id==template_schema()["calibration_case"] else "unqualified",
             "scope":"autonomous_source_image_main_contour","solve_revision":1,"use_segmentation":bool(use_segmentation),
             "require_segmentation_review":bool(require_segmentation_review and use_segmentation),
             "segmentation_review":{"status":"not_required" if not (require_segmentation_review and use_segmentation) else "pending"},
             "conversation_id":conversation_id}
        job.update(provider_id=provider_id, provider_profile=self._provider_metadata(provider_id))
        self._event(job,"queued","启动自动绘制：从原图识别主轮廓，不调用样本专用模板，不读取参考DXF。")
        self.store.save(job)
        if asynchronous:self._submit(job_id,self._automatic,job_id)
        else:self._automatic(job_id)
        return self.store.get(job_id)

    def _resolve_feedback_evidence(self, parent):
        """Find the nearest revision ancestor with a complete candidate bundle."""
        root = self.settings.runtime_root.resolve()
        evidence_parent, resolved, selection_file, ancestor_depth = parent, None, None, 0
        visited = set()
        while evidence_parent and evidence_parent.get("id") not in visited and ancestor_depth < 16:
            visited.add(evidence_parent.get("id"))
            try:
                candidate = Path(evidence_parent.get("artifact_directory", "")).resolve(strict=True)
            except OSError:
                candidate = None
            if candidate is not None and candidate.is_relative_to(root):
                selection = next((name for name in ("topology-plan.json", "feedback-plan.json")
                                  if (candidate / name).is_file()), None)
                required = ("topology-candidates.json", "baseline-model.json", "overlay.png")
                if selection and all((candidate / name).is_file() for name in required):
                    resolved, selection_file = candidate, selection
                    break
            ancestor_id = evidence_parent.get("parent_job_id")
            if not ancestor_id:
                evidence_parent = None
                break
            try:
                evidence_parent = self.store.get(ancestor_id)
            except KeyError:
                evidence_parent = None
                break
            ancestor_depth += 1
        if resolved is None or evidence_parent is None:
            raise ValueError("当前任务缺少多候选拓扑证据，不能执行截图修正。")
        return evidence_parent, resolved, selection_file, ancestor_depth

    def create_feedback_revision(self, parent_job_id: str, screenshot: Path, instruction: str,
                                 *, conversation_id=None, asynchronous=True, provider_id=None):
        """Create a source-bound topology revision from a screenshot conversation turn."""
        parent = self.store.get(parent_job_id)
        if parent.get("mode") not in {"autonomous_image", "autonomous_revision"}:
            raise ValueError("只有自动主轮廓任务可以进行截图修正。")
        if parent.get("status") not in {"completed", "needs_review", "failed"}:
            raise ValueError("请等待当前主轮廓任务结束后再提交截图修正。")
        evidence_parent, resolved, selection_file, ancestor_depth = self._resolve_feedback_evidence(parent)
        instruction = str(instruction).strip()
        if not instruction or len(instruction) > 2000:
            raise ValueError("请用不超过2000字说明需要修改的区域。")
        provider_id = self._select_provider(provider_id or parent.get("provider_id"), True)
        job_id = uuid.uuid4().hex
        job = {
            "id": job_id, "mode": "autonomous_revision", "status": "queued",
            "case_id": parent.get("case_id", "feedback-revision"), "template_id": None,
            "created_at": now(), "updated_at": now(), "use_api": True,
            "uploaded": parent.get("uploaded", False), "source_image": parent["source_image"],
            "source_ocr": parent["source_ocr"], "source_image_url": f"/api/jobs/{job_id}/image",
            "parameters": [], "assumptions": [], "events": [], "issues": [], "artifacts": {},
            "validation": None, "provider": {"status": "pending", "network_requests": 0},
            "geometry": None, "engineering_accepted": False, "automatic_completion": False,
            "manual_confirmation": None, "manual_intervention": True, "split": parent.get("split", "unqualified"),
            "scope": "screenshot_guided_source_topology_revision", "solve_revision": 1,
            "use_segmentation": parent.get("use_segmentation", False), "parent_job_id": parent_job_id,
            "evidence_parent_job_id": evidence_parent.get("id"),
            "evidence_ancestor_depth": ancestor_depth, "parent_selection_file": selection_file,
            "conversation_id": conversation_id, "feedback_instruction": instruction,
            "feedback_screenshot": str(screenshot), "parent_artifact_directory": str(resolved),
            "provider_id": provider_id, "provider_profile": self._provider_metadata(provider_id),
        }
        self._event(job, "feedback_queued", "已读取当前任务上下文；准备比较已保存的源图拓扑候选。")
        self.store.save(job)
        if asynchronous:
            self._submit(job_id, self._feedback_revision, job_id)
        else:
            self._feedback_revision(job_id)
        return self.store.get(job_id)

    def _feedback_revision(self, job_id):
        job = self.store.get(job_id)
        if job.get("status") == "cancelled":
            return
        providers = self._provider_bundle(job)
        output = self.settings.runtime_root / "jobs" / job_id / "feedback-001"
        output.mkdir(parents=True, exist_ok=True)
        parent_output = Path(job["parent_artifact_directory"])
        try:
            from .planning_provider import evaluate_candidates
            from .topology_candidates import materialize_selected_candidate
            from .parametric_pipeline import _export_chain, _topology_source_validation
            bundle = json.loads((parent_output / "topology-candidates.json").read_text(encoding="utf-8"))
            selection_file = job.get("parent_selection_file", "topology-plan.json")
            if selection_file not in {"topology-plan.json", "feedback-plan.json"}:
                raise ValueError("截图修正的候选选择证据文件无效。")
            parent_plan = json.loads((parent_output / selection_file).read_text(encoding="utf-8"))
            baseline = json.loads((parent_output / "baseline-model.json").read_text(encoding="utf-8"))
            candidates = bundle.get("candidates") or []
            local = evaluate_candidates(candidates, max_candidates=5)
            job["status"] = "auditing"
            self._event(job, "feedback_observe", "截图、当前叠加图与候选轮廓已组成有界视觉比较包。")
            if not self._persist_if_active(job):
                return
            receipt = providers["feedback"].select(
                Path(job["feedback_screenshot"]), parent_output / "overlay.png", candidates,
                job["feedback_instruction"], parent_plan.get("selected_candidate_id"),
            )
            job["provider"] = receipt
            plan = {
                "schema_version": "screenshot-topology-revision-v1", "status": receipt.get("status"),
                "parent_job_id": job["parent_job_id"], "instruction": job["feedback_instruction"],
                "evidence_parent_job_id": job.get("evidence_parent_job_id", job["parent_job_id"]),
                "evidence_ancestor_depth": job.get("evidence_ancestor_depth", 0),
                "current_candidate_id": parent_plan.get("selected_candidate_id"),
                "selected_candidate_id": receipt.get("selected_candidate_id"),
                "observation": receipt.get("observation", ""),
                "proposed_action": receipt.get("proposed_action", ""),
                "evidence_tags": receipt.get("evidence_tags", []),
                "operations": receipt.get("operations", []),
                "rationale_code": receipt.get("rationale_code"), "confidence": receipt.get("confidence"),
                "provider": receipt, "local_evaluation": local, "ground_truth_used": False,
                "private_chain_of_thought_exposed": False,
                "trace_scope": "Auditable observations, evidence IDs, decisions and checks; no hidden chain-of-thought.",
            }
            shutil.copyfile(job["feedback_screenshot"], output / "feedback-screenshot.png")
            (output / "feedback-plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
            selected_id = receipt.get("selected_candidate_id") if receipt.get("schema_success") is True else None
            admissible = set(local.get("admissible_candidate_ids") or [])
            if selected_id not in admissible:
                job.update(status="needs_review", automatic_completion=False, validation=None,
                           issues=["在线修正没有选择通过本地来源与拓扑门禁的候选；当前版本保持不变。"])
                job["artifact_directory"] = str(output)
                self._publish_automatic_artifacts(job, output)
                self._event(job, "feedback_abstained", job["issues"][0])
                self._persist_if_active(job)
                return
            selected = next(row for row in candidates if row.get("id") == selected_id)
            geometry_operations = [row for row in receipt.get("operations", [])
                                   if row.get("action") == "replace_boundary_chain_with_line"]
            if geometry_operations:
                try:
                    selected, geometry_audit = create_feedback_candidate(
                        Path(job["source_image"]), read_ocr(Path(job["source_ocr"])), baseline, selected, bundle,
                        Path(job["feedback_screenshot"]), parent_output / "overlay.png", receipt["operations"], output,
                    )
                    revised_local = evaluate_candidates([selected], max_candidates=1)
                    if selected["id"] not in revised_local.get("admissible_candidate_ids", []):
                        raise ValueError("截图直线修正没有通过来源、闭合或图元有效性门禁。")
                    bundle["candidates"].append(selected)
                    plan.update(provider_selected_candidate_id=selected_id,
                                selected_candidate_id=selected["id"], geometry_revision=geometry_audit,
                                operation_status="applied_by_local_geometry_kernel")
                    selected_id = selected["id"]
                    if geometry_audit.get("geometry_changed"):
                        self._event(job, "feedback_geometry", "已将截图定位到原图；按用户指令把目标边界链替换为直线，剖面线不参与轮廓建模。")
                    else:
                        self._event(job, "feedback_geometry", "截图目标在所选候选中已经是一条直线；已按满足指令处理并继续验证。")
                except ValueError as error:
                    plan.update(operation_status="rejected_by_local_geometry_gate", operation_error=str(error)[:240])
                    (output / "feedback-plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
                    job.update(status="needs_review", automatic_completion=False, validation=None,
                               issues=[str(error)[:240]], artifact_directory=str(output))
                    self._publish_automatic_artifacts(job, output)
                    self._event(job, "feedback_geometry_rejected", job["issues"][0])
                    self._persist_if_active(job)
                    return
            (output / "feedback-plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
            (output / "topology-candidates.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
            (output / "topology-plan.json").write_text(json.dumps({
                "schema_version": "feedback-selected-topology-v1", "status": "selected",
                "selected_candidate_id": selected_id, "parent_job_id": job["parent_job_id"],
                "evidence_parent_job_id": job.get("evidence_parent_job_id", job["parent_job_id"]),
                "selection_source": "screenshot_feedback_agent_and_local_geometry_gate",
                "ground_truth_used": False,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            materialize_selected_candidate(selected, bundle, output)
            graph = selected["graph"]
            source_validation = _topology_source_validation(Path(job["source_image"]), baseline, graph)
            if not source_validation.get("passed"):
                raise ValueError("截图修正候选未通过源笔画、闭合和坐标一致性门禁。")
            self._event(job, "feedback_validate", "修正候选已通过源图哈希、笔画支持、闭合与坐标一致性检查。")
            result = _export_chain(Path(job["source_image"]), baseline, graph["entities"], output,
                                   topology=True, source_validation=source_validation)
            parameterization = {
                "status": "feedback_topology_exported", "accepted": False, "topology_exported": True,
                "ground_truth_used": False, "geometry_updated_by_api": False,
                "topology_selected_by_api": True,
                "geometry_updated_by_user_instruction": bool(
                    geometry_operations and geometry_audit.get("geometry_changed")),
                "local_geometry_operations_applied": sum(
                    row.get("status") == "applied" for row in geometry_audit.get("operations", []))
                    if geometry_operations else 0,
                "dimensions_updated_by_api": False, "all_dimensions_verified": False,
                "reason": ("screenshot_guided_line_revision" if geometry_operations and geometry_audit.get("geometry_changed")
                           else "screenshot_instruction_already_satisfied" if geometry_operations
                           else "screenshot_guided_candidate_selection"),
                "selected_candidate_id": selected_id, "parent_job_id": job["parent_job_id"],
                "provider": receipt, "source_validation": source_validation,
            }
            result["parameterization"] = parameterization
            result["revision"] = {"parent_job_id": job["parent_job_id"],
                                  "evidence_parent_job_id": job.get("evidence_parent_job_id", job["parent_job_id"]),
                                  "instruction": job["feedback_instruction"],
                                  "feedback_plan": "feedback-plan.json"}
            (output / "model.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            (output / "parametric-stage.json").write_text(json.dumps(parameterization, ensure_ascii=False, indent=2), encoding="utf-8")
            for name in ("segmentation-mask.png", "segmentation-overlay.png", "raw-segmentation-mask.png",
                         "raw-segmentation-overlay.png", "segmentation-refinement.json", "segmentation.json",
                         "dimension-evidence.json", "dimension-analysis.json", "baseline-model.json",
                         "baseline-drawing.dxf", "baseline-preview.svg", "baseline-overlay.png"):
                source = parent_output / name
                if source.is_file() and not (output / name).exists():
                    shutil.copyfile(source, output / name)
            job.update(status="completed", validation=result["validation"], automatic_completion=True,
                       artifact_directory=str(output), parameterization=parameterization,
                       geometry={"entities": result["entities"], "bounds": result["bounds"],
                                 "coordinate_system": result["coordinate_system"]},
                       completion_class="screenshot_guided_source_topology_revision",
                       issues=[] if selected_id != parent_plan.get("selected_candidate_id") else
                              ["在线检查保留了当前候选；已重新导出并验证，但轮廓对象没有变化。"])
            self._publish_automatic_artifacts(job, output)
            self._event(job, "feedback_export", f"截图修正已导出 {len(result['entities'])} 个 LINE/ARC 图元并完成 DXF 回读。")
            self._persist_if_active(job)
        except InterruptedError:
            return
        except Exception as error:
            job["status"] = "failed"
            job["automatic_completion"] = False
            job["issues"] = [str(error)[:300] if isinstance(error, ValueError) else "截图修正未完成；原任务产物保持不变。"]
            job["artifact_directory"] = str(output)
            self._publish_automatic_artifacts(job, output)
            self._event(job, "feedback_failed", job["issues"][0])
            self._persist_if_active(job)

    def _stage_segmentation_review(self, job, image_path, output):
        from .segmentation import cached_extract
        evidence_dir = output / "learned-evidence"
        cached_extract(self.settings.segmentation_checkpoint, image_path, evidence_dir)
        mask_path = evidence_dir / "prediction-mask.png"
        overlay_path = evidence_dir / "prediction-overlay.png"
        if not mask_path.is_file() or not overlay_path.is_file():
            raise ValueError("分割模型没有生成可供人工检查的掩膜与叠加图。")
        with Image.open(mask_path) as mask:
            size = [mask.width, mask.height]
        job["artifact_directory"] = str(output)
        job["segmentation_review"] = {
            "status": "pending", "mask_width": size[0], "mask_height": size[1],
            "model_mask_sha256": digest(mask_path), "reviewed": False,
            "instructions": "绿色为材料区域；添加补全缺失区域，擦除标注线、剖面线和多余区域。",
        }
        self._publish_automatic_artifacts(job, output)
        job["status"] = "awaiting_segmentation_review"
        self._event(job, "segmentation_review", "分割图已生成；流水线暂停，等待人工添加、擦除并确认材料区域。")
        self._persist_if_active(job)

    def import_oracle_dxf(self, job_id, content: bytes):
        """Derive a reviewable GT mask without retaining or publishing DXF geometry."""
        from .oracle_mask_import import generate_oracle_mask

        with self.lock:
            job = self.store.get(job_id)
            if job.get("mode") != "autonomous_image" or job.get("status") != "awaiting_segmentation_review":
                raise ValueError("该任务当前不在分割检查阶段。")
            root = self.settings.runtime_root.resolve()
            output = Path(job.get("artifact_directory", "")).resolve(strict=True)
            predicted_path = output / "segmentation-mask.png"
            if not output.is_relative_to(root) or not predicted_path.is_file():
                raise ValueError("任务缺少原始模型分割掩膜。")
            with Image.open(predicted_path) as image:
                mask_size = image.size
            image_path, ocr_path = Path(job["source_image"]), Path(job["source_ocr"])
        pixels, receipt = generate_oracle_mask(content, image_path, ocr_path, predicted_path,
                                               mask_size, temporary_root=root)
        with self.lock:
            job = self.store.get(job_id)
            if job.get("status") != "awaiting_segmentation_review" or Path(job.get("artifact_directory", "")).resolve() != output:
                raise ValueError("任务状态已变化，GT掩膜未导入。")
            if digest(image_path) != receipt["source_image_sha256"] or digest(ocr_path) != receipt["source_ocr_sha256"]:
                raise ValueError("导入期间原图或OCR发生变化，GT掩膜未导入。")
            mask_path = output / "oracle-generated-mask.png"
            temporary = output / "oracle-generated-mask.png.tmp"
            temporary.write_bytes(pixels)
            temporary.replace(mask_path)
            receipt_path = output / "oracle-mask-import.json"
            temporary_receipt = output / "oracle-mask-import.json.tmp"
            temporary_receipt.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf8")
            temporary_receipt.replace(receipt_path)
            job["oracle_mask_import"] = receipt
            self._publish_automatic_artifacts(job, output)
            self._event(job, "oracle_mask_generated", "已从上传的GT DXF生成同尺寸掩膜；请检查叠加位置后确认。此项仅用于开发实验。")
            self.store.save(job)
            return self.public_job(job)

    def submit_segmentation_review(self, job_id, content: bytes, *, source="human_review", asynchronous=True):
        if source not in {"human_review", "gt_oracle", "gt_oracle_edited"}:
            raise ValueError("未知掩膜来源；仅支持人工审核或声明的GT实验掩膜。")
        if not content or len(content) > 12_000_000:
            raise ValueError("审核掩膜必须是小于12MB的PNG图像。")
        with self.lock:
            job = self.store.get(job_id)
            if job.get("mode") != "autonomous_image" or job.get("status") != "awaiting_segmentation_review":
                raise ValueError("该任务当前不在分割人工检查阶段。")
            output = Path(job.get("artifact_directory", ""))
            root = self.settings.runtime_root.resolve()
            try:
                resolved = output.resolve(strict=True)
            except OSError:
                raise ValueError("分割审核产物目录不存在。") from None
            predicted_path = resolved / "segmentation-mask.png"
            if not resolved.is_relative_to(root) or not predicted_path.is_file():
                raise ValueError("任务缺少原始模型分割掩膜。")
            try:
                with Image.open(io.BytesIO(content)) as submitted:
                    submitted.load()
                    if submitted.format != "PNG":
                        raise ValueError("审核掩膜必须使用PNG格式。")
                    reviewed = np.asarray(submitted.convert("L")) >= 128
                with Image.open(predicted_path) as predicted_image:
                    predicted = np.asarray(predicted_image.convert("L")) >= 128
            except (OSError, ValueError, Image.DecompressionBombError):
                raise ValueError("无法读取审核掩膜PNG。") from None
            if reviewed.shape != predicted.shape:
                raise ValueError("审核掩膜尺寸必须与模型分割图完全一致。")
            foreground = int(reviewed.sum())
            if foreground < 3 or foreground >= reviewed.size:
                raise ValueError("审核掩膜必须保留有效且非满幅的材料区域。")
            gt_derived = source in {"gt_oracle", "gt_oracle_edited"}
            oracle_mask = source == "gt_oracle"
            import_receipt = job.get("oracle_mask_import") or {}
            if gt_derived:
                generated_path = resolved / "oracle-generated-mask.png"
                if (not generated_path.is_file() or import_receipt.get("mask_sha256") != digest(generated_path)
                        or import_receipt.get("source_image_sha256") != digest(Path(job["source_image"]))
                        or import_receipt.get("source_ocr_sha256") != digest(Path(job["source_ocr"]))):
                    raise ValueError("必须先上传当前图纸对应的GT DXF并成功生成掩膜。")
                with Image.open(generated_path) as generated_image:
                    generated = np.asarray(generated_image.convert("L")) >= 128
                if reviewed.shape != generated.shape or (oracle_mask and not np.array_equal(reviewed, generated)):
                    raise ValueError("声明为原始GT的掩膜必须与服务端DXF生成的掩膜完全一致；人工涂改请标记为GT辅助修订。")
            model_mask = resolved / "model-segmentation-mask.png"
            model_overlay = resolved / "model-segmentation-overlay.png"
            if not model_mask.exists():
                shutil.copyfile(predicted_path, model_mask)
            current_overlay = resolved / "segmentation-overlay.png"
            if current_overlay.is_file() and not model_overlay.exists():
                shutil.copyfile(current_overlay, model_overlay)
            reviewed_mask = resolved / "reviewed-segmentation-mask.png"
            temporary = reviewed_mask.with_suffix(".png.tmp")
            Image.fromarray(reviewed.astype(np.uint8) * 255).save(temporary, format="PNG")
            temporary.replace(reviewed_mask)
            with Image.open(job["source_image"]) as source_image:
                source_image = source_image.convert("RGB").resize((reviewed.shape[1], reviewed.shape[0]), Image.Resampling.LANCZOS)
                overlay = np.asarray(source_image).copy()
            overlay[reviewed] = (overlay[reviewed] * .58 + np.asarray([45, 205, 159]) * .42).astype(np.uint8)
            reviewed_overlay = resolved / "reviewed-segmentation-overlay.png"
            Image.fromarray(overlay).save(reviewed_overlay, format="PNG")
            shutil.copyfile(reviewed_mask, resolved / "segmentation-mask.png")
            shutil.copyfile(reviewed_overlay, resolved / "segmentation-overlay.png")
            added = int((reviewed & ~predicted).sum())
            removed = int((predicted & ~reviewed).sum())
            record = {
                "schema_version": "job-segmentation-review-v1", "status": "approved",
                "reviewed_at": now(), "reviewed": not oracle_mask, "ground_truth_used": gt_derived,
                "oracle_mask_conditioned": oracle_mask,
                "source": "registered_dxf_gt" if oracle_mask else "registered_dxf_gt_edited" if gt_derived else "human_review",
                "calibration_or_development": gt_derived, "held_out": False,
                "ground_truth_dxf_coordinates_used_for_mask_creation": gt_derived,
                "ground_truth_dxf_coordinates_used_for_prediction": False,
                "ground_truth_dxf_coordinates_sent_to_provider": False,
                "source_gt_sha256": import_receipt.get("source_gt_sha256") if gt_derived else None,
                "model_mask_sha256": digest(model_mask), "reviewed_mask_sha256": digest(reviewed_mask),
                "mask_size": {"width": reviewed.shape[1], "height": reviewed.shape[0]},
                "model_foreground_pixels": int(predicted.sum()), "reviewed_foreground_pixels": foreground,
                "added_pixels": added, "removed_pixels": removed,
                "scope": ("GT-derived raster mask is a declared development input; it does not certify automatic segmentation, dimensions or independent reference accuracy."
                          if gt_derived else "Human pixel-mask review only; does not certify dimensions or reference accuracy."),
            }
            (resolved / "segmentation-review.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf8")
            job["segmentation_review"] = {**record, "mask_path": str(reviewed_mask)}
            job["manual_intervention"] = not oracle_mask
            job["ground_truth_used"] = gt_derived
            job["oracle_mask_conditioned"] = oracle_mask
            job["status"] = "queued"
            job["solve_revision"] = int(job.get("solve_revision", 1)) + 1
            self._publish_automatic_artifacts(job, resolved)
            self._event(job, "segmentation_approved", ("已导入GT派生掩膜作为开发测试输入；" if oracle_mask else
                        "GT派生掩膜经人工修订；" if gt_derived else "人工分割已确认：")
                        + f"补全 {added} 像素，擦除 {removed} 像素；继续拓扑与参数化构建。")
            self.store.save(job)
        if asynchronous:
            self._submit(job_id, self._automatic, job_id)
        else:
            self._automatic(job_id)
        return self.store.get(job_id)

    def _automatic(self, job_id):
        job=self.store.get(job_id)
        if job.get("status") == "cancelled":
            return
        providers=self._provider_bundle(job)
        image_path,ocr_path=Path(job["source_image"]),Path(job["source_ocr"])
        output=self.settings.runtime_root/"jobs"/job_id/"automatic-001"
        def progress(stage,message):
            job["status"]="extracting" if stage in {"segment","measure"} else "solving"
            if (output/"parametric-stage.json").is_file():
                job["parameterization"]=json.loads((output/"parametric-stage.json").read_text(encoding="utf8"))
            self._publish_automatic_artifacts(job, output)
            self._event(job,stage,message)
            if not self._persist_if_active(job):raise InterruptedError("任务已取消")
        try:
            document=read_ocr(ocr_path)
            with Image.open(image_path) as source:
                width,height=source.size
                if width*height>80_000_000:raise ValueError("图像像素超过80百万限制。")
                source.verify()
            original=document["meta"].get("original_size",{})
            if original and (original.get("width"),original.get("height"))!=(width,height):
                raise ValueError("OCR坐标尺寸与原图不一致。")
            job["source"]={"image_sha256":digest(image_path),"ocr_sha256":digest(ocr_path),"width":width,"height":height,"ocr_records":len(document["records"])}
            review = job.get("segmentation_review") or {}
            if job.get("use_segmentation") and job.get("require_segmentation_review") and review.get("status") != "approved":
                self._stage_segmentation_review(job, image_path, output)
                return
            if job.get("use_segmentation") and review.get("status") == "approved":
                kwargs = {"segmentation_mask": review["mask_path"],
                          "segmentation_review": {key: value for key, value in review.items() if key != "mask_path"}}
            else:
                kwargs = {"segmentation_checkpoint": self.settings.segmentation_checkpoint} if job.get("use_segmentation") else {}
            result=build_automatic(image_path,document,output,progress=progress,**kwargs)
            if review.get("status") == "approved" and (output / "reviewed-evidence/segmentation.json").is_file():
                shutil.copyfile(output / "reviewed-evidence/segmentation.json", output / "segmentation.json")
            job["scale"]=result["scale"]
            job["extraction"]={k:v for k,v in result["extraction"].items() if k not in {"polyline_px","raw_polyline_px","candidates"}}
            job["validation"]=result["validation"]
            job["geometry"]={"entities":result["entities"],"bounds":result["bounds"],"coordinate_system":result["coordinate_system"]}
            job["artifact_directory"]=str(output)
            names={"dxf":"drawing.dxf","svg":"preview.svg","model":"model.json","validation":"validation.json",
                   "overlay":"overlay.png","dimensions":"dimension-evidence.json"}
            job["artifacts"]={k:f"/api/jobs/{job_id}/artifacts/{v}" for k,v in names.items() if (output/v).is_file()}
            self._publish_automatic_artifacts(job, output)
            job["curve_fit"]=result.get("curve_fit")
            job["automatic_completion"]=result["automatic_completion"]
            job["issues"]=result["issues"]
            job["completion_class"]="automatic_scaled_draft" if result["validation"]["scaled_mm"] else "automatic_pixel_draft"
            # Publish the local draft before any bounded remote stage starts.
            job["dimension_analysis"]=analyze_dimensions(document,result)
            self._save_dimension_analysis(job,output)
            self._publish_automatic_artifacts(job, output)
            if not self._persist_if_active(job):return
            if job["use_api"]:
                job["status"]="auditing"
                job["dimension_analysis"]["provider"]["status"]="pending"
                self._save_dimension_analysis(job,output)
                self._event(job,"dimension_api","本地轮廓已可下载；在线解析最多16条原图尺寸，记录与本地证据的一致性。")
                if not self._persist_if_active(job):return
                job["dimension_analysis"]=analyze_dimensions(document,result,provider=providers["automatic_dimension"],use_api=True)
                self._save_dimension_analysis(job,output)
                dimension_receipt=job["dimension_analysis"]["provider"]
                self._event(job,"dimension_result","在线尺寸解析已完成。" if dimension_receipt.get("status")=="succeeded" else "在线尺寸解析不可用；本地尺寸证据与CAD产物保留。")
                if not self._persist_if_active(job):return
            if job.get("use_segmentation"):
                result,parameterization=refine_parametric(image_path,document,result,output,
                    provider=providers["binding"],planner_provider=providers["planning"],
                    editor_provider=providers["topology_edit"],evaluator_provider=providers["topology_evaluate"],
                    use_api=job["use_api"],progress=progress)
                job["parameterization"]=parameterization
                job["validation"]=result["validation"]
                job["geometry"]={"entities":result["entities"],"bounds":result["bounds"],"coordinate_system":result["coordinate_system"]}
                job["curve_fit"]=result.get("curve_fit")
                job["automatic_completion"]=result["automatic_completion"]
                if parameterization.get("accepted"):
                    job["completion_class"]="partial_parametric_draft"
                elif parameterization.get("constraint_subset_accepted"):
                    job["completion_class"]="partial_parametric_draft_unresolved_radii"
                elif parameterization.get("topology_exported"):
                    job["completion_class"]="source_topology_draft"
                self._publish_automatic_artifacts(job,output)
                if not self._persist_if_active(job):return
            if job["use_api"]:
                job["status"]="auditing"
                self._event(job,"vision","本地自动轮廓已可下载；在线模型正在检查候选边界与原图是否一致。")
                if not self._persist_if_active(job):return
                receipt=providers["vision"].inspect(image_path,contour_px=result.get("fitted_polyline_px",result["polyline_px"]))
                job["provider"]=receipt
                if receipt.get("status")=="succeeded":
                    verdict=receipt.get("verdict","uncertain")
                    self._event(job,"vision_result",f"在线视觉复核结果：{verdict}。该结果不代表尺寸精度验证。")
                    if verdict!="match":job["issues"].extend(receipt.get("issues",[]) or ["在线模型未确认候选轮廓与原图一致。"])
                else:
                    self._event(job,"vision_unavailable","在线视觉复核未完成；自动生成的轮廓和原图叠加已保留，不转为人工补参数流程。")
                    job["issues"].append("在线视觉复核未完成，不能计为在线核验通过。")
            job["status"]="completed" if result["validation"]["passed"] else "failed"
            if result["validation"]["passed"] and _parameterization_needs_review(job):
                job["status"]="needs_review"
            if result.get("complete_material_exterior") is False:
                job["completion_class"]="incomplete_material_exterior_draft"
                job["automatic_completion"]=False
                job["status"]="needs_review" if result["validation"]["passed"] else "failed"
                self._event(job,"material_connectivity","材料区域仍存在断裂或仅角点连接；已保留主区域草稿，不能计为完整主轮廓自动重建。")
            self._event(job,job["status"],"自动绘制结束：请区分闭合产物、尺寸比例、视觉复核与独立参考误差。")
            self._persist_if_active(job)
        except InterruptedError:
            return
        except Exception as error:
            self._publish_automatic_artifacts(job, output)
            job["status"]="failed"
            if (job.get("provider") or {}).get("status") == "pending":
                job["provider"] = {"status":"skipped", "network_requests":0, "reason":"local_generation_failed"}
            message=str(error)[:300] if isinstance(error,(ValueError,ArithmeticError)) else "自动识别失败，未生成通过检查的轮廓。"
            job["issues"]=[message]
            self._event(job,"failed",message)
            self._persist_if_active(job)

    def _publish_automatic_artifacts(self, job, output):
        """Publish actual intermediate files, including when a later stage fails."""
        aliases={"segmentation-mask.png":"learned-evidence/prediction-mask.png",
                 "segmentation-overlay.png":"learned-evidence/prediction-overlay.png",
                 "raw-segmentation-mask.png":"learned-evidence/raw-prediction-mask.png",
                 "raw-segmentation-overlay.png":"learned-evidence/raw-prediction-overlay.png",
                 "segmentation-refinement.json":"learned-evidence/refinement.json",
                 "segmentation.json":"learned-evidence/segmentation.json"}
        for name,relative in aliases.items():
            source=output/relative
            target=output/name
            if source.is_file() and not target.exists():
                shutil.copyfile(source,target)
        names={"dxf":"drawing.dxf","svg":"preview.svg","model":"model.json","validation":"validation.json",
               "overlay":"overlay.png","dimensions":"dimension-evidence.json",
               "segmentation_mask":"segmentation-mask.png","segmentation_overlay":"segmentation-overlay.png",
               "raw_segmentation_mask":"raw-segmentation-mask.png","raw_segmentation_overlay":"raw-segmentation-overlay.png",
               "segmentation_refinement":"segmentation-refinement.json",
               "segmentation_evidence":"segmentation.json","contour_overlay":"contour-overlay.png",
               "model_segmentation_mask":"model-segmentation-mask.png",
               "model_segmentation_overlay":"model-segmentation-overlay.png",
               "oracle_generated_mask":"oracle-generated-mask.png",
               "oracle_mask_import":"oracle-mask-import.json",
               "reviewed_segmentation_mask":"reviewed-segmentation-mask.png",
               "reviewed_segmentation_overlay":"reviewed-segmentation-overlay.png",
               "segmentation_review":"segmentation-review.json",
               "cad_overlay":"overlay.png","curve_fit":"curve-fit.json","dimension_analysis":"dimension-analysis.json"}
        names.update(topology="topology.json",topology_overlay="topology-overlay.png",
                     topology_candidates="topology-candidates.json",topology_plan="topology-plan.json",
                     topology_edit_proposals="topology-edit-proposals.json",
                     topology_iterations="topology-iterations.json",
                     reconstruction_feedback="reconstruction-feedback.json",
                     corrections="correction-evidence.json",constraint_bindings="constraint-bindings.json",
                     binding_candidates="binding-candidates.json",binding_overlay="binding-topology.png",
                     parametric_solution="parametric-solution.json",parameterization="parametric-stage.json",
                     radius_targets="radius-targets.json",radius_contract="radius-contract.json",
                     workflow_provenance="workflow-provenance.json",
                     baseline_dxf="baseline-drawing.dxf",baseline_svg="baseline-preview.svg",
                     baseline_overlay="baseline-overlay.png",baseline_model="baseline-model.json")
        names.update(feedback_plan="feedback-plan.json", feedback_screenshot="feedback-screenshot.png")
        for key,name in names.items():
            if (output/name).is_file():
                job["artifact_directory"]=str(output)
                job.setdefault("artifacts",{})[key]=f"/api/jobs/{job['id']}/artifacts/{name}"

    @staticmethod
    def _save_dimension_analysis(job,output):
        path=output/"dimension-analysis.json"
        temporary=path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(job["dimension_analysis"],ensure_ascii=False,indent=2),encoding="utf8")
        temporary.replace(path)

    @staticmethod
    def _recover_automatic_geometry(job,directory):
        """Verify a coherent export, or restore a separately verified baseline.

        A saved job's old passed flag is never evidence that publication finished.
        Hash manifests also detect mixed previews/diagnostics after a hard stop.
        """
        import ezdxf
        from .automatic import _verify_dxf_readback
        from .parametric_pipeline import CORE
        publication=(job.get("parameterization") or {}).get("publication") or {}
        def verified(prefix="", expected_hashes=None):
            try:
                if not all((directory/(prefix+name)).is_file() for name in CORE):return None
                if expected_hashes and (set(expected_hashes)!=set(CORE) or any(
                    digest(directory/(prefix+name))!=expected_hashes[name] for name in CORE)):
                    return None
                model=json.loads((directory/(prefix+"model.json")).read_text(encoding="utf8"))
                validation=json.loads((directory/(prefix+"validation.json")).read_text(encoding="utf8"))
                curve_fit=json.loads((directory/(prefix+"curve-fit.json")).read_text(encoding="utf8"))
                if not isinstance(model,dict) or not isinstance(validation,dict):return None
                incomplete_material = model.get("complete_material_exterior") is False and validation.get("complete_material_exterior") is False
                if model.get("validation")!=validation or not validation.get("passed") or (not model.get("automatic_completion") and not incomplete_material):
                    return None
                if model.get("curve_fit")!=curve_fit:return None
                units=(model.get("coordinate_system") or {}).get("units")
                if units not in {"mm","pixel"} or (units=="mm")!=bool(validation.get("scaled_mm")):return None
                readback=_verify_dxf_readback(ezdxf.readfile(directory/(prefix+"drawing.dxf")),model["entities"],
                                             expected_units=4 if units=="mm" else 0)
                if not readback["passed"]:return None
                return model,validation
            except (OSError,ValueError,KeyError,TypeError,ezdxf.DXFError):return None
        try:
            # A finished rollback deliberately contains baseline, not candidate,
            # hashes. Otherwise an incomplete publication must match one full set.
            candidate_hashes=publication.get("candidate_sha256") if publication.get("status") in {"pending","committed"} else None
            saved=verified(expected_hashes=candidate_hashes)
            if saved is None:
                rollback_prefix=publication.get("rollback_prefix","baseline-")
                if rollback_prefix not in {"last-valid-","baseline-"}:return False
                rollback_hashes=publication.get("rollback_sha256") if rollback_prefix=="last-valid-" else publication.get("baseline_sha256")
                saved=verified(rollback_prefix,rollback_hashes)
                if saved is None and rollback_prefix!="baseline-":
                    rollback_prefix="baseline-"
                    saved=verified(rollback_prefix,publication.get("baseline_sha256"))
                if saved is None:return False
                for name in CORE:
                    temporary=directory/(name+".recover")
                    shutil.copyfile(directory/(rollback_prefix+name),temporary)
                    temporary.replace(directory/name)
                if verified() is None:return False
                parametric=job.setdefault("parameterization",{})
                parametric.update(status="interrupted",accepted=False,reason="incomplete_publication_last_valid_restored",
                                  geometry_updated_by_api=False,dimensions_updated_by_api=False,
                                  topology_exported=bool(saved[0].get("parameterization",{}).get("topology_exported")))
                parametric.setdefault("publication",{}).update(status="rolled_back",recovered=True)
                job.setdefault("issues",[]).append("新草稿发布未完整结束，已回读核验并恢复上一套有效轮廓。")
                job["completion_class"]="automatic_scaled_draft" if saved[1].get("scaled_mm") else "automatic_pixel_draft"
            model,validation=saved
            job.update(validation=validation,automatic_completion=bool(model.get("automatic_completion")),scale=model["scale"],curve_fit=model.get("curve_fit"),
                       geometry={"entities":model["entities"],"bounds":model["bounds"],"coordinate_system":model["coordinate_system"]})
            if model.get("parameterization",{}).get("accepted"):
                # model.json snapshots the export. A later durable stage or
                # binding receipt must not be replaced by older model metadata.
                verified_stage={**model["parameterization"],**(job.get("parameterization") or {}),
                                "accepted":True,"status":"completed",
                                "geometry_updated_by_api":model["parameterization"].get("geometry_updated_by_api",False),
                                "dimensions_updated_by_api":model["parameterization"].get("dimensions_updated_by_api",False)}
                if publication:
                    verified_stage["publication"]={**publication,"status":"committed","recovered":True}
                job["parameterization"]=verified_stage
                job["completion_class"]="partial_parametric_draft"
            elif model.get("parameterization",{}).get("constraint_subset_accepted"):
                job["parameterization"]={**model["parameterization"],**(job.get("parameterization") or {}),
                                         "accepted":False,"constraint_subset_accepted":True,
                                         "status":"completed_with_unresolved_radii"}
                job["completion_class"]="partial_parametric_draft_unresolved_radii"
                job["status"]="needs_review"
            elif model.get("parameterization",{}).get("topology_exported"):
                recovered_stage=job.get("parameterization") or model["parameterization"]
                recovered_stage.update(accepted=False,topology_exported=True,
                                       geometry_updated_by_api=False,dimensions_updated_by_api=False)
                job["parameterization"]=recovered_stage
                job["completion_class"]="source_topology_draft"
            job["extraction"]={k:v for k,v in model.get("extraction",{}).items() if k not in {"polyline_px","raw_polyline_px","candidates"}}
            if model.get("complete_material_exterior") is False:
                job["completion_class"]="incomplete_material_exterior_draft"
            return True
        except (OSError,ValueError,KeyError,TypeError,ezdxf.DXFError):
            return False

    def create_upload(self, image: Path, ocr: Path, template_id: str, use_api=True, provider_id=None):
        if template_id != template_schema()["id"]:
            raise ValueError("未知模板。")
        job = self._new("upload", template_id, image, ocr, use_api, uploaded=True, provider_id=provider_id)
        self._submit(job["id"], self._extract, job["id"])
        return job

    def _extract(self, job_id):
        job = self.store.get(job_id)
        providers = self._provider_bundle(job)
        job["status"] = "extracting"
        self._event(job, "source", "检查图片与 OCR 坐标尺寸，建立尺寸来源记录。")
        if not self._persist_if_active(job):
            return
        try:
            image_path, ocr_path = Path(job["source_image"]), Path(job["source_ocr"])
            document = read_ocr(ocr_path)
            with Image.open(image_path) as image:
                width, height = image.size
                if width * height > 60_000_000:
                    raise ValueError("图像像素超过60百万限制。")
                image.verify()
            original = document["meta"].get("original_size", {})
            if original and (original.get("width"), original.get("height")) != (width, height):
                raise ValueError("OCR坐标尺寸与上传原图不一致；请使用未经缩放的配对原图。")
            document["meta"]["original_size"] = {"width": width, "height": height}
            job["source"] = {"image_sha256": digest(image_path), "ocr_sha256": digest(ocr_path),
                             "width": width, "height": height, "ocr_records": len(document["records"])}
            schema = template_schema()
            rows, records = bind_parameters(document, schema, calibrated_layout=not job["uploaded"])
            job["parameters"] = rows
            job["assumptions"] = [{**a, "confirmed": False} for a in schema["assumptions"]]
            job["shape_priors"] = schema.get("shape_priors")
            self._event(job, "local_parse", f"本地解析完成：{sum(r['value'] is not None for r in rows)}/{len(rows)}个驱动参数有明确OCR候选。")
            if not self._persist_if_active(job):
                return
            if job["use_api"]:
                source_ids = list(dict.fromkeys(r["source_record_id"] for r in rows if r.get("source_record_id")))[:16]
                by_id = {r["id"]: r for r in records}
                selected = [by_id[source_id] for source_id in source_ids]
                try:
                    report = providers["dimension"].normalize(selected)
                    job["provider"] = report
                    decisions = {r["id"]: r for r in report["dimensions"]}
                    for row in rows:
                        decision = decisions.get(row.get("source_record_id"))
                        if decision:
                            row["provider_proposal"] = decision
                            local = by_id[row["source_record_id"]]["parsed"]
                            keys = ("kind", "nominal", "upper_deviation", "lower_deviation")
                            agrees = all(decision.get(k) == local.get(k) for k in keys)
                            row["provider_agrees_with_local"] = agrees
                            if not agrees:
                                row["needs_review"] = True
                                row["note"] += " API解释与本地规则不同；候选不自动采用。"
                    self._event(job, "api", f"真实API已返回{len(report['dimensions'])}条短文本解析；数值仍以原图证据和人工核对为准。")
                except ProviderError as error:
                    job["provider"] = {"status": "failed", "code": error.code, "message": error.message,
                                       "status_code": error.status_code, "network_requests": error.network_requests,
                                       "http_success": error.http_success, "schema_success": False}
                    self._event(job, "api_fallback", error.message)
            job["status"] = "needs_review"
            job["issues"] = [f"{r['label']}：{r['note']}" for r in rows if r["needs_review"]]
            self._event(job, "review", "请补齐缺失参数并逐项确认模板假设；API失败不会清除已识别尺寸。")
            self._persist_if_active(job)
        except Exception as error:
            job["status"] = "failed"
            message = str(error)[:240] if isinstance(error, ValueError) else "输入处理失败，请检查图片及OCR文件。"
            job["issues"] = [message]
            self._event(job, "failed", message)
            self._persist_if_active(job)

    def confirm(self, job_id, values, confirmed_assumptions, confirm_bindings, *, asynchronous=True, actor="user"):
        with self.lock:
            job = self.store.get(job_id)
            if job["status"] not in {"needs_review", "completed", "failed"} or not job.get("parameters"):
                raise ValueError("任务当前不能确认；请等待尺寸提取完成。")
            if not confirm_bindings:
                raise ValueError("需要明确确认尺寸与图中位置的对应关系。")
            schema = template_schema()
            expected = {row["id"] for row in schema["parameters"]}
            if set(values) != expected:
                raise ValueError("必须提供全部19个参数，不能默默使用缺失字段的默认值。")
            normalized = {}
            for field in schema["parameters"]:
                value = values[field["id"]]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not field["min"] <= value <= field["max"]:
                    raise ValueError(f"{field['label']}必须在{field['min']}至{field['max']}{field['unit']}之间。")
                normalized[field["id"]] = float(value)
            expected_assumptions = {a["id"] for a in schema["assumptions"] if a.get("required")}
            if not expected_assumptions.issubset(set(confirmed_assumptions)):
                raise ValueError("请确认全部模板假设，特别是未标注形状先验和简化踏面。")
            if set(confirmed_assumptions) - {a["id"] for a in schema["assumptions"]}:
                raise ValueError("包含未知假设。")
            previous = {r["id"]: r.get("value") for r in job["parameters"]}
            for row in job["parameters"]:
                row["extracted_value"] = row.get("value")
                row["value"] = normalized[row["id"]]
                row["confirmed_by"] = actor
                row["needs_review"] = False
                row["confirmation_source"] = "manual_input" if previous[row["id"]] != row["value"] else "confirmed_ocr"
            for assumption in job["assumptions"]:
                assumption["confirmed"] = assumption["id"] in confirmed_assumptions
            job["manual_confirmation"] = {"actor": actor, "time": now(), "parameter_count": len(normalized),
                                          "changed_or_supplied": [pid for pid in normalized if previous[pid] != normalized[pid]],
                                          "assumptions": list(confirmed_assumptions)}
            job["status"] = "solving"
            job["artifacts"] = {}
            job.pop("artifact_directory", None)
            job["validation"] = None
            job["geometry"] = None
            job["issues"] = []
            job["engineering_accepted"] = False
            job["solve_revision"] = int(job.get("solve_revision", 0)) + 1
            self._event(job, "solve", "参数与适用范围已确认，按公式重新计算圆心、切点和完整轮廓。")
            self.store.save(job)
        if asynchronous:
            self._submit(job_id, self._solve, job_id, normalized)
        else:
            self._solve(job_id, normalized)
        return self.store.get(job_id)

    def _solve(self, job_id, parameters):
        job = self.store.get(job_id)
        if job.get("status") == "cancelled":
            return
        output = self.settings.runtime_root / "jobs" / job_id / f"revision-{job['solve_revision']:03d}"
        try:
            result = solve_profile(parameters, output)
            job["validation"] = result["validation"]
            job["geometry"] = {k: result.get(k) for k in ("entities", "bounds", "assumptions")}
            job["artifact_directory"] = str(output)
            known = {"dxf": "drawing.dxf", "svg": "preview.svg", "model": "model.json", "validation": "validation.json"}
            job["artifacts"] = {k: f"/api/jobs/{job_id}/artifacts/{name}" for k, name in known.items() if (output / name).is_file()}
            passed = bool(result["validation"].get("passed"))
            job["status"] = "completed" if passed else "needs_review"
            job["issues"] = result["validation"].get("issues", [])
            # A confirmed simplified, prior-conditioned template is never certified manufacturing geometry.
            job["engineering_accepted"] = False
            job["automatic_completion"] = False
            job["completion_class"] = "confirmed_template_assisted" if passed else "geometry_failed"
            self._event(job, "completed" if passed else "validation", "DXF已导出并回读验证；结果属于已确认模板范围，包含声明的形状先验和简化踏面。" if passed else "几何检查未通过，请修改相关尺寸。")
            self._persist_if_active(job)
        except Exception as error:
            job["status"] = "needs_review"
            message = str(error)[:350] if isinstance(error, (ValueError, ArithmeticError)) else "几何重建失败；参数已保留。"
            job["issues"] = [message]
            self._event(job, "geometry_failed", message)
            self._persist_if_active(job)

    def cancel(self, job_id):
        with self.lock:
            job = self.store.get(job_id)
            if job["status"] not in {"queued", "extracting", "solving", "auditing"}:
                raise ValueError("仅运行中的任务可以取消。")
            job["status"] = "cancelled"
            self._event(job, "cancelled", "已取消后续处理；正在进行的网络请求会在规定期限内结束，结果不会发布。")
            self.store.save(job)
            return job

    def delete_job(self, job_id):
        """Delete a terminal job and only its runtime-owned artifact directory."""
        with self.lock:
            job = self.store.get(job_id)
            if job.get("status") in {"queued", "extracting", "solving", "auditing"}:
                raise ValueError("任务仍在运行，请先停止任务再删除历史记录。")
            if len(job_id) != 32 or any(ch not in "0123456789abcdef" for ch in job_id):
                raise ValueError("任务标识无效。")
            target = self._job_directory(job_id)
            future = self._futures.get(job_id)
            if future is not None and not future.done():
                self._deleted_job_ids.add(job_id)
                future.cancel()
            if target.exists():
                shutil.rmtree(target)
            if not self.store.delete(job_id):
                raise KeyError(job_id)
            return job

    def public_job(self, job):
        result = {key: value for key, value in job.items() if key not in {"source_image", "source_ocr", "artifact_directory"}}
        if isinstance(result.get("segmentation_review"), dict):
            result["segmentation_review"] = {key: value for key, value in result["segmentation_review"].items()
                                             if key != "mask_path"}
        return result

    def close(self):
        self.executor.shutdown(wait=False, cancel_futures=True)
