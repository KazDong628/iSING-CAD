"""Local web workbench and persisted asynchronous job API."""
from __future__ import annotations
import json
import re
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from .config import ROOT, Settings, load_local_env
from .dataset import resolve_inside
from .geometry import template_schema
from .service import AgentService
from .store import RuntimeLease
from .conversation_store import ConversationStore
from .model_transcript import build_model_transcript
from .segmentation_review import create_segmentation_review_router

class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str = Field(max_length=200)
    template_id: str | None = None
    use_api: bool = True
    use_segmentation: bool = False
    require_segmentation_review: bool = False
    provider_id: str | None = Field(default=None, max_length=80)
    mode: Literal["autonomous_image", "template_assisted"] = "autonomous_image"

class Confirmation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    parameters: dict
    confirmed_assumptions: list[str]
    confirm_bindings: bool

class ConversationCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(default="新建主轮廓任务", max_length=80)

def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is None:
        load_local_env()
        settings = Settings()
    lease = RuntimeLease(settings.runtime_root)
    try:
        service = AgentService(settings, recover_running=True)
    except Exception:
        lease.close()
        raise

    @asynccontextmanager
    async def lifespan(app):
        yield
        service.close()
        lease.close()

    app = FastAPI(title="Contour Agent / 主轮廓工作台", version="0.1.0", lifespan=lifespan)
    app.state.service = service
    conversations = ConversationStore(settings.runtime_root / "conversations")
    app.state.conversations = conversations
    app.include_router(create_segmentation_review_router(Path(settings.segmentation_manifest) if settings.segmentation_manifest else settings.runtime_root / "segmentation/data/manifest.json"))

    @app.middleware("http")
    async def browser_boundary(request: Request, call_next):
        origin = request.headers.get("origin")
        if request.method in {"POST", "PUT", "DELETE", "PATCH"} and origin:
            parsed = urlparse(origin)
            if parsed.netloc != request.url.netloc:
                return JSONResponse({"detail": "拒绝跨来源修改请求。"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.exception_handler(ValueError)
    async def invalid(request, error):
        return JSONResponse({"detail": str(error)[:400]}, status_code=400)

    @app.exception_handler(KeyError)
    async def missing(request, error):
        return JSONResponse({"detail": "任务或资源不存在。"}, status_code=404)

    @app.get("/api/health")
    def health():
        return {"status": "ok", "version": "0.1.0"}

    @app.get("/api/config")
    def config():
        return settings.public()

    @app.get("/agent", include_in_schema=False)
    def agent_workbench():
        return FileResponse(ROOT / "web" / "agent.html", media_type="text/html", headers={"Cache-Control": "no-store"})

    @app.get("/segmentation-results", include_in_schema=False)
    def segmentation_results():
        report = settings.runtime_root / "segmentation/gt-comparison/index.html"
        if not report.is_file() or not report.resolve().is_relative_to(settings.runtime_root.resolve()):
            raise HTTPException(404, "本轮 GT 分割评估尚未生成。")
        return FileResponse(report, media_type="text/html", headers={"Cache-Control": "no-store"})

    @app.get("/connectivity-results", include_in_schema=False)
    def connectivity_results():
        root = settings.runtime_root.resolve()
        report = root / "segmentation/connectivity-results/index.html"
        if not report.is_file() or report.resolve() != report or not report.resolve().is_relative_to(root):
            raise HTTPException(404, "连通性与参数化重建对比尚未生成。")
        return FileResponse(report, media_type="text/html", headers={"Cache-Control": "no-store"})

    @app.get("/connectivity-results/summary.json", include_in_schema=False)
    def connectivity_summary():
        root = settings.runtime_root.resolve()
        report = root / "segmentation/connectivity-results/summary.json"
        if not report.is_file() or report.resolve() != report or not report.resolve().is_relative_to(root):
            raise HTTPException(404, "连通性重建摘要尚未生成。")
        return FileResponse(report, media_type="application/json", headers={"Cache-Control": "no-store"})

    def pilot_file(run_id, name):
        if not re.fullmatch(r"\d{8}T\d{12}Z(?:-[0-9a-f]{8})?", run_id) or name not in {
                "index.html", "pilot_summary.json", "pilot_summary.csv", "pilot-artifacts.zip"}:
            raise HTTPException(404, "试验产物不存在。")
        root=settings.runtime_root.resolve()
        path=root/"segmentation"/"pilot"/run_id/name
        if not path.is_file() or path.resolve()!=path or not path.resolve().is_relative_to(root):
            raise HTTPException(404, "试验产物不存在。")
        media={"index.html":"text/html","pilot_summary.json":"application/json",
               "pilot_summary.csv":"text/csv","pilot-artifacts.zip":"application/zip"}
        return FileResponse(path,media_type=media[name],filename=None if name=="index.html" else name,
                            headers={"Cache-Control":"no-store"})

    @app.get("/segmentation-pilot", include_in_schema=False)
    def segmentation_pilot():
        pointer=settings.runtime_root/"segmentation/pilot/latest.json"
        try:
            if pointer.resolve()!=pointer.absolute():
                raise ValueError("Invalid report pointer")
            latest=json.loads(pointer.read_text(encoding="utf8"))
            run_id=latest.get("run_id","")
            if not isinstance(run_id,str):raise ValueError("Invalid run identity")
        except (OSError,ValueError,TypeError,AttributeError):
            raise HTTPException(404,"四图自动构建试验报告尚未生成。") from None
        return pilot_file(run_id,"index.html")

    @app.get("/api/segmentation/pilot/{run_id}/{artifact_name}")
    def segmentation_pilot_artifact(run_id: str, artifact_name: str):
        return pilot_file(run_id,artifact_name)

    def comparison_file(run_id, name):
        media={"index.html":"text/html", "comparison.json":"application/json",
               "comparison.csv":"text/csv", "comparison-artifacts.zip":"application/zip"}
        if not re.fullmatch(r"\d{8}T\d{12}Z(?:-[0-9a-f]{8})?", run_id) or name not in media:
            raise HTTPException(404,"DXF 对比产物不存在。")
        root=settings.runtime_root.resolve()
        path=root/"dxf-comparison"/run_id/name
        if not path.is_file() or path.resolve()!=path or not path.resolve().is_relative_to(root):
            raise HTTPException(404,"DXF 对比产物不存在。")
        return FileResponse(path,media_type=media[name],filename=None if name=="index.html" else name,
                            headers={"Cache-Control":"no-store"})

    @app.get("/dxf-comparison", include_in_schema=False)
    def dxf_comparison():
        pointer=settings.runtime_root/"dxf-comparison/latest.json"
        try:
            if pointer.resolve()!=pointer.absolute():
                raise ValueError("Invalid comparison pointer")
            latest=json.loads(pointer.read_text(encoding="utf8"))
            run_id=latest.get("run_id","")
            if not isinstance(run_id,str):raise ValueError("Invalid run identity")
        except (OSError,ValueError,TypeError,AttributeError):
            raise HTTPException(404,"DXF 与 GT 对比报告尚未生成。") from None
        return comparison_file(run_id,"index.html")

    @app.get("/api/dxf-comparison/{run_id}/{artifact_name}")
    def dxf_comparison_artifact(run_id: str, artifact_name: str):
        return comparison_file(run_id,artifact_name)

    @app.get("/api/catalog")
    def catalog():
        return service.catalog

    @app.get("/api/templates")
    def templates():
        return {"templates": [template_schema()]}

    @app.get("/api/jobs")
    def jobs():
        return {"jobs": [service.public_job(j) for j in service.store.list()]}

    @app.post("/api/jobs", status_code=202)
    def create_job(payload: JobRequest):
        if payload.mode == "autonomous_image":
            options = {"use_segmentation": True, "require_segmentation_review": payload.require_segmentation_review} if payload.use_segmentation else {}
            return service.public_job(service.create_auto_case(payload.case_id,payload.use_api,
                                                               provider_id=payload.provider_id,**options))
        return service.public_job(service.create_case(payload.case_id, payload.use_api, payload.template_id,
                                                      provider_id=payload.provider_id))

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        return service.public_job(service.store.get(job_id))

    @app.post("/api/jobs/{job_id}/confirm", status_code=202)
    def confirm_job(job_id: str, payload: Confirmation):
        return service.public_job(service.confirm(job_id, payload.parameters, payload.confirmed_assumptions, payload.confirm_bindings))

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        return service.public_job(service.cancel(job_id))

    @app.get("/api/cases/{case_id}/image")
    def case_image(case_id: str):
        case = service.cases.get(case_id)
        if not case or not case.get("image"):
            raise HTTPException(404, "图片不存在。")
        return FileResponse(resolve_inside(settings.dataset_root, case["image"]))

    @app.get("/api/jobs/{job_id}/image")
    def job_image(job_id: str):
        job = service.store.get(job_id)
        return FileResponse(job["source_image"])

    @app.put("/api/jobs/{job_id}/segmentation-review", status_code=202)
    async def approve_job_segmentation(job_id: str, mask: UploadFile = File(...)):
        if (mask.content_type or "").lower() != "image/png":
            raise ValueError("审核掩膜必须使用PNG格式。")
        content = await mask.read(12_000_001)
        return service.public_job(service.submit_segmentation_review(job_id, content))

    @app.get("/api/jobs/{job_id}/artifacts/{name}")
    def artifact(job_id: str, name: str):
        if name not in {"drawing.dxf", "preview.svg", "validation.json", "model.json", "overlay.png", "dimension-evidence.json",
                        "segmentation-mask.png","segmentation-overlay.png","segmentation.json","contour-overlay.png",
                        "raw-segmentation-mask.png","raw-segmentation-overlay.png","segmentation-refinement.json",
                        "curve-fit.json","dimension-analysis.json","topology.json","topology-overlay.png",
                        "topology-candidates.json","topology-plan.json",
                        "correction-evidence.json","constraint-bindings.json","binding-candidates.json","binding-topology.png",
                        "parametric-stage.json","parametric-solution.json","baseline-drawing.dxf","baseline-preview.svg",
                        "baseline-overlay.png","baseline-model.json","feedback-plan.json","feedback-screenshot.png",
                        "model-segmentation-mask.png","model-segmentation-overlay.png",
                        "reviewed-segmentation-mask.png","reviewed-segmentation-overlay.png","segmentation-review.json"}:
            raise HTTPException(404, "产物不存在。")
        job = service.store.get(job_id)
        directory = job.get("artifact_directory")
        if not directory:
            raise HTTPException(404, "任务尚未生成产物。")
        path = Path(directory) / name
        if not path.is_file():
            raise HTTPException(404, "产物不存在。")
        if name.endswith(".svg"):
            return FileResponse(path, media_type="image/svg+xml")
        if name.endswith(".png"):
            return FileResponse(path, media_type="image/png")
        return FileResponse(path, filename=name)

    def public_conversation(value):
        result = json.loads(json.dumps(value, ensure_ascii=False))
        jobs = {}
        for job_id in result.get("memory", {}).get("job_ids", [])[-20:]:
            try:
                jobs[job_id] = service.public_job(service.store.get(job_id))
            except KeyError:
                continue
        result["jobs"] = jobs
        return result

    @app.get("/api/conversations")
    def list_conversations():
        return {"conversations": [{key: row.get(key) for key in ("id", "title", "created_at", "updated_at", "memory")}
                                  for row in conversations.list()]}

    @app.post("/api/conversations", status_code=201)
    def create_conversation(payload: ConversationCreate):
        return public_conversation(conversations.create(payload.title))

    @app.get("/api/conversations/{conversation_id}")
    def get_conversation(conversation_id: str):
        return public_conversation(conversations.get(conversation_id))

    @app.delete("/api/conversations/{conversation_id}")
    def delete_conversation(conversation_id: str):
        conversation = conversations.get(conversation_id)
        remembered = list(dict.fromkeys(conversation.get("memory", {}).get("job_ids", [])))
        owned = [job for job in service.store.list(10_000)
                 if job.get("conversation_id") == conversation_id and job.get("id") not in remembered]
        jobs = []
        for job_id in remembered:
            try:
                job = service.store.get(job_id)
                if job.get("conversation_id") == conversation_id:
                    jobs.append(job)
            except KeyError:
                continue
        jobs.extend(owned)
        if any(job.get("status") in {"queued", "extracting", "solving", "auditing"} for job in jobs):
            raise HTTPException(409, "该任务仍在运行，请先点击停止，待状态变为已停止后再删除。")
        for job in jobs:
            try:
                service.delete_job(job["id"])
            except KeyError:
                continue
        upload_root = (settings.runtime_root / "conversation-uploads").resolve()
        upload_directory = (upload_root / conversation_id).resolve()
        if upload_directory.parent != upload_root:
            raise ValueError("对话上传目录无效。")
        if upload_directory.exists():
            shutil.rmtree(upload_directory)
        conversations.delete(conversation_id)
        return {"deleted": True, "conversation_id": conversation_id,
                "deleted_job_count": len(jobs)}

    @app.get("/api/jobs/{job_id}/model-transcript")
    def get_model_transcript(job_id: str):
        return build_model_transcript(service.store.get(job_id), settings.runtime_root)

    @app.post("/api/conversations/{conversation_id}/turns", status_code=202)
    async def conversation_turn(
        conversation_id: str,
        prompt: str = Form(...),
        case_id: str = Form(""),
        job_id: str = Form(""),
        use_api: bool = Form(True),
        use_segmentation: bool = Form(True),
        provider_id: str = Form(""),
        image: UploadFile | None = File(None),
        ocr: UploadFile | None = File(None),
        screenshot: UploadFile | None = File(None),
    ):
        conversation = conversations.get(conversation_id)
        prompt = prompt.strip()
        if not prompt or len(prompt) > 2000:
            raise ValueError("请输入1至2000字的任务或修改说明。")
        attachments = []
        upload_root = settings.runtime_root / "conversation-uploads" / conversation_id / uuid.uuid4().hex
        source_job = None
        if screenshot is not None:
            suffix = Path(screenshot.filename or "").suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
                raise ValueError("反馈截图仅支持 JPG、PNG、WebP。")
            content = await screenshot.read(12_000_001)
            if len(content) > 12_000_000:
                raise ValueError("反馈截图最大12MB。")
            upload_root.mkdir(parents=True, exist_ok=True)
            screenshot_path = upload_root / ("feedback" + suffix)
            screenshot_path.write_bytes(content)
            attachments.append({"kind": "feedback_screenshot", "name": Path(screenshot.filename or "截图").name[:120]})
            parent_job_id = job_id.strip() or conversation.get("memory", {}).get("last_job_id")
            if not parent_job_id:
                raise ValueError("当前对话还没有可修改的主轮廓任务。")
            source_job = service.create_feedback_revision(parent_job_id, screenshot_path, prompt,
                                                          conversation_id=conversation_id,
                                                          provider_id=provider_id.strip() or None)
            assistant_text = "已结合当前任务上下文读取截图。正在比较保存的拓扑候选、执行来源门禁并重新导出 DXF。"
        elif image is not None or ocr is not None:
            if image is None or ocr is None:
                raise ValueError("首次上传需要同时提供工程图原图和配对 OCR JSON。")
            suffix = Path(image.filename or "").suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
                raise ValueError("工程图仅支持 JPG、PNG、WebP。")
            image_bytes = await image.read(20_000_001)
            ocr_bytes = await ocr.read(5_000_001)
            if len(image_bytes) > 20_000_000 or len(ocr_bytes) > 5_000_000:
                raise ValueError("工程图最大20MB，OCR JSON最大5MB。")
            upload_root.mkdir(parents=True, exist_ok=True)
            image_path, ocr_path = upload_root / ("source" + suffix), upload_root / "ocr.json"
            image_path.write_bytes(image_bytes)
            ocr_path.write_bytes(ocr_bytes)
            attachments.extend([{"kind": "source_image", "name": Path(image.filename or "原图").name[:120]},
                                {"kind": "ocr", "name": Path(ocr.filename or "OCR.json").name[:120]}])
            source_job = service.create_auto_source(image_path, ocr_path, use_api,
                                                    case_id="conversation-upload", use_segmentation=use_segmentation,
                                                    require_segmentation_review=use_segmentation,
                                                    conversation_id=conversation_id,
                                                    provider_id=provider_id.strip() or None)
            assistant_text = ("文件已进入分割阶段；生成材料掩膜后会暂停，待你添加、擦除并确认后再继续拓扑、约束与 DXF。"
                              if use_segmentation else "文件已进入自动轮廓流水线。")
        elif case_id.strip():
            source_job = service.create_auto_case(case_id.strip(), use_api=use_api, use_segmentation=use_segmentation,
                                                  require_segmentation_review=use_segmentation,
                                                  conversation_id=conversation_id,
                                                  provider_id=provider_id.strip() or None)
            assistant_text = ("已启动真实分割任务；分割图生成后会暂停，等待你人工检查并确认。"
                              if use_segmentation else "已使用数据集原图和 OCR 创建真实重建任务；运行轨迹会持续写入当前对话。")
        else:
            parent_job_id = job_id.strip() or conversation.get("memory", {}).get("last_job_id")
            if parent_job_id:
                current = service.public_job(service.store.get(parent_job_id))
                assistant_text = f"我记得当前任务 {parent_job_id[:8]}，状态为 {current.get('status')}。如需修改，请附上局部截图并描述问题。"
            else:
                assistant_text = "当前对话还没有图纸上下文。请拖入工程图和 OCR JSON，或从数据集选择一张图纸。"
        conversations.add_message(conversation_id, "user", prompt, attachments=attachments,
                                  parent_job_id=job_id.strip() or None)
        if source_job is not None:
            conversations.remember_job(conversation_id, source_job["id"])
            conversations.add_message(conversation_id, "assistant", assistant_text, job_id=source_job["id"],
                                      trace_kind="revision" if source_job.get("mode") == "autonomous_revision" else "reconstruction")
        else:
            conversations.add_message(conversation_id, "assistant", assistant_text)
        updated_conversation = conversations.get(conversation_id)
        if updated_conversation.get("title") == "新建主轮廓任务":
            if case_id.strip():
                updated_conversation["title"] = case_id.strip()[:80]
            elif screenshot is not None:
                updated_conversation["title"] = f"截图修订 · {(job_id.strip() or '当前任务')[:8]}"
            elif image is not None:
                updated_conversation["title"] = Path(image.filename or "上传图纸").stem[:80]
            else:
                updated_conversation["title"] = prompt[:36]
            conversations.save(updated_conversation)
        return public_conversation(updated_conversation)

    @app.post("/api/uploads", status_code=202)
    async def upload(image: UploadFile = File(...), ocr: UploadFile = File(...),
                     template_id: str = Form(""), use_api: bool = Form(True), mode: str = Form("autonomous_image"),
                     use_segmentation: bool = Form(False), require_segmentation_review: bool = Form(False),
                     provider_id: str = Form("")):
        if mode not in {"autonomous_image", "template_assisted"}:
            raise ValueError("未知绘制模式。")
        suffix = Path(image.filename or "").suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            raise ValueError("图像仅支持 JPG、PNG、WebP。")
        image_bytes = await image.read(20_000_001)
        ocr_bytes = await ocr.read(5_000_001)
        if len(image_bytes) > 20_000_000 or len(ocr_bytes) > 5_000_000:
            raise ValueError("图像最大20MB，OCR JSON最大5MB。")
        directory = settings.runtime_root / "uploads" / uuid.uuid4().hex
        directory.mkdir(parents=True)
        image_path, ocr_path = directory / ("source" + suffix), directory / "ocr.json"
        image_path.write_bytes(image_bytes)
        ocr_path.write_bytes(ocr_bytes)
        if mode=="autonomous_image":
            options = {"use_segmentation": True, "require_segmentation_review": require_segmentation_review} if use_segmentation else {}
            return service.public_job(service.create_auto_source(image_path,ocr_path,use_api,
                                                                 provider_id=provider_id.strip() or None,**options))
        return service.public_job(service.create_upload(image_path, ocr_path, template_id, use_api,
                                                        provider_id=provider_id.strip() or None))

    @app.get("/api/evaluations")
    def evaluations():
        directory = settings.runtime_root / "evaluations"
        reports = []
        if directory.exists():
            for path in sorted(directory.glob("*.json"), reverse=True)[:20]:
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    reports.append({"name": path.name, **data})
                except (ValueError, OSError):
                    continue
        return {"reports": reports}

    @app.get("/api/evaluations/{report_name}/cases/{case_id}/artifacts/{artifact_name}")
    def evaluation_artifact(report_name: str, case_id: str, artifact_name: str):
        media_types = {"drawing.dxf": "application/dxf", "preview.svg": "image/svg+xml",
                       "overlay.png": "image/png", "model.json": "application/json",
                       "validation.json": "application/json", "dimension-evidence.json": "application/json"}
        media_types.update({name:"application/json" for name in ("topology.json", "topology-candidates.json", "topology-plan.json", "correction-evidence.json", "constraint-bindings.json", "binding-candidates.json", "parametric-stage.json", "parametric-solution.json", "baseline-model.json")})
        media_types.update({name:"image/png" for name in ("topology-overlay.png", "binding-topology.png", "baseline-overlay.png")})
        media_types.update({"baseline-drawing.dxf":"application/dxf", "baseline-preview.svg":"image/svg+xml"})
        media_types.update({name:"image/png" for name in ("segmentation-mask.png", "segmentation-overlay.png", "raw-segmentation-mask.png", "raw-segmentation-overlay.png")})
        media_types.update({name:"application/json" for name in ("segmentation.json", "segmentation-refinement.json")})
        missing_artifact = HTTPException(404, "评估产物不存在。")
        if artifact_name not in media_types or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,180}\.json", report_name):
            raise missing_artifact
        try:
            root = Path(settings.runtime_root).resolve(strict=True)
            report_path = root / "evaluations" / report_name
            # Reject symlinks/junctions even when they point to another file in
            # runtime. A report cannot turn this endpoint into a file browser.
            if report_path.resolve(strict=True) != report_path or not report_path.is_file() or report_path.stat().st_size > 50_000_000:
                raise missing_artifact
            data = json.loads(report_path.read_text(encoding="utf8"))
            if not isinstance(data, dict) or data.get("mode") != "autonomous_image":
                raise missing_artifact
            run_id = data.get("run_id")
            if not isinstance(run_id, str) or not re.fullmatch(r"\d{8}T\d{12}Z(?:-[0-9a-f]{8})?", run_id):
                raise missing_artifact
            trials = data.get("trials")
            if not isinstance(trials, list):
                raise missing_artifact
            trial = next((row for row in reversed(trials) if isinstance(row, dict) and row.get("case_id") == case_id), None)
            if not trial or not isinstance(trial.get("job_id"), str) or not re.fullmatch(r"[0-9a-f]{32}", trial["job_id"]):
                raise missing_artifact
            artifacts = trial.get("artifacts")
            reported_path = artifacts.get(artifact_name) if isinstance(artifacts, dict) else None
            if not isinstance(reported_path, str):
                raise missing_artifact
            # Reconstruct the actual generator output location; untrusted
            # report paths are only checked for equality and never served.
            expected = root / "autonomous-qualification" / run_id / "agent-runtime" / "jobs" / trial["job_id"] / "automatic-001" / artifact_name
            actual = expected.resolve(strict=True)
            if actual != expected or not actual.is_relative_to(root) or not actual.is_file():
                raise missing_artifact
            if Path(reported_path).resolve(strict=True) != actual:
                raise missing_artifact
        except (OSError, RuntimeError, ValueError, TypeError):
            raise missing_artifact from None
        inline = artifact_name in {"overlay.png", "preview.svg"}
        response = FileResponse(actual, media_type=media_types[artifact_name], filename=None if inline else artifact_name)
        if artifact_name == "preview.svg":
            response.headers["Content-Security-Policy"] = "sandbox; default-src 'none'; style-src 'unsafe-inline'"
        return response

    web = ROOT / "web"
    web.mkdir(exist_ok=True)

    # Windows registry MIME associations may classify .js as text/plain.
    # The browser correctly blocks that response under nosniff, so do not rely
    # on host MIME guesses for the workbench's executable and stylesheet assets.
    @app.get("/static/app.js", include_in_schema=False)
    def javascript():
        return FileResponse(web / "app.js", media_type="text/javascript")

    @app.get("/static/workflow.js", include_in_schema=False)
    def workflow_javascript():
        return FileResponse(web / "workflow.js", media_type="text/javascript")

    @app.get("/static/styles.css", include_in_schema=False)
    def stylesheet():
        return FileResponse(web / "styles.css", media_type="text/css")

    @app.get("/static/agent.js", include_in_schema=False)
    def agent_javascript():
        return FileResponse(web / "agent.js", media_type="text/javascript")

    @app.get("/static/agent.css", include_in_schema=False)
    def agent_stylesheet():
        return FileResponse(web / "agent.css", media_type="text/css")

    app.mount("/static", StaticFiles(directory=web), name="static")

    @app.get("/")
    def index():
        return FileResponse(web / "index.html", media_type="text/html")

    return app
