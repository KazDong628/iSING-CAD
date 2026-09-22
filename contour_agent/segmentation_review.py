"""Local, explicit user correction of registered source-derived masks.

The frozen manifest and dataset are never modified. Loading a mask is not a
review decision; only an explicit PNG save creates a reviewed record.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from .config import ROOT

MAX_EDGE = 1536
MAX_BODY = 12_000_000
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,179}")


def create_segmentation_review_router(manifest_path: Path | None = None) -> APIRouter:
    """Mount with ``app.include_router(create_segmentation_review_router(...))``.

    Manifest rows may use image/mask paths relative to the manifest directory,
    or absolute paths within that same directory. It can be a row list or an
    object containing rows, cases, or samples. No arbitrary filesystem route is
    exposed. Source hashes are inherited, never recalculated from resized data.
    """
    manifest_path = Path(manifest_path or ROOT / "runtime/segmentation/data/manifest.json").absolute()
    data_root = manifest_path.parent.resolve()
    router = APIRouter()
    lock = threading.RLock()

    def safe_path(value, *, exists=True):
        if not isinstance(value, (str, Path)) or not str(value) or "\x00" in str(value):
            raise HTTPException(404, "注册资源不存在。")
        raw = Path(value)
        candidate = raw if raw.is_absolute() else data_root / raw
        try:
            actual = candidate.resolve(strict=exists)
            if not actual.is_relative_to(data_root) or actual != candidate.absolute():
                raise HTTPException(404, "注册资源不存在。")
            if exists and not actual.is_file():
                raise HTTPException(404, "注册资源不存在。")
            return actual
        except (OSError, RuntimeError, ValueError):
            raise HTTPException(404, "注册资源不存在。") from None

    def read_json(path, *, missing=None):
        if not path.exists() and missing is not None:
            return missing
        path = safe_path(path)
        try:
            if path.stat().st_size > 5_000_000:
                raise ValueError("oversize")
            return json.loads(path.read_text(encoding="utf8"))
        except (ValueError, OSError):
            raise HTTPException(503, "标签清单或审核记录无法读取，请检查本地数据。") from None

    def manifest():
        data = read_json(manifest_path)
        rows = data if isinstance(data, list) else next((data[k] for k in ("rows", "cases", "samples") if isinstance(data, dict) and isinstance(data.get(k), list)), None)
        if not isinstance(rows, list):
            raise HTTPException(503, "标签清单格式无效。")
        result = {}
        for row in rows:
            if not isinstance(row, dict):
                raise HTTPException(503, "标签清单格式无效。")
            case_id = row.get("id", row.get("case_id"))
            source_hash = row.get("source_image_sha256", row.get("sourcehash", row.get("source_sha256")))
            if not isinstance(case_id, str) or not SAFE_ID.fullmatch(case_id) or case_id in result:
                raise HTTPException(503, "标签清单包含无效或重复样本ID。")
            if not isinstance(source_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", source_hash):
                raise HTTPException(503, "标签清单缺少原图SHA256。")
            result[case_id] = {**row, "id": case_id, "source_image_sha256": source_hash}
        return result

    def get_row(case_id):
        if not SAFE_ID.fullmatch(case_id):
            raise HTTPException(404, "样本不存在。")
        row = manifest().get(case_id)
        if row is None:
            raise HTTPException(404, "样本不存在。")
        return row

    def dimensions(path):
        try:
            with Image.open(path) as image:
                width, height = image.size
                if image.format not in {"PNG", "JPEG", "WEBP"} or min(width, height) < 1 or max(width, height) > MAX_EDGE:
                    raise HTTPException(422, "校正图像最大边须不超过1536像素，请使用准备后的数据。")
                return (width, height), image.format
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            raise HTTPException(422, "注册图像无法读取。") from None

    def reviews():
        data = read_json(data_root / "reviews.json", missing={})
        if not isinstance(data, dict):
            raise HTTPException(503, "审核记录格式无效，未覆盖已有记录。")
        return data

    def reviewed_record(case_id, row, records):
        record = records.get(case_id)
        if not isinstance(record, dict) or record.get("reviewed") is not True or record.get("label_source") != "local_user_review" or record.get("source_image_sha256") != row["source_image_sha256"]:
            return None
        expected = f"reviewed_masks/{case_id}.png"
        if record.get("mask") != expected:
            raise HTTPException(404, "审核掩膜路径无效。")
        path = safe_path(expected)
        if record.get("mask_sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
            raise HTTPException(409, "审核掩膜与记录不一致，请检查本地文件。")
        return record

    def file_response(path, media_type):
        return FileResponse(path, media_type=media_type, headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})

    @router.get("/segmentation-review", include_in_schema=False)
    def page():
        return file_response(ROOT / "web/segmentation-review.html", "text/html")

    @router.get("/api/segmentation-review/cases")
    def cases():
        with lock:
            records = reviews()
            result = []
            for case_id, row in manifest().items():
                item = {"id": case_id, "split": row.get("split"), "source_image_sha256": row["source_image_sha256"], "available": False, "reviewed": False,
                        "trainable": row.get("trainable", True), "registration_status": (row.get("registration") or {}).get("status"),
                        "label_issues": row.get("issues", [])}
                try:
                    size, _ = dimensions(safe_path(row.get("image")))
                    record = reviewed_record(case_id, row, records)
                    mask_size, mask_kind = dimensions(safe_path(record["mask"] if record else row.get("mask")))
                    if mask_size != size or mask_kind != "PNG":
                        raise HTTPException(422, "掩膜必须为与注册图像同尺寸的PNG。")
                    item.update(available=True, width=size[0], height=size[1], reviewed=bool(record),
                                label_source="local_user_review" if record else row.get("label_source", "source_heuristic"),
                                reviewed_at=record.get("timestamp") if record else None)
                except HTTPException as error:
                    item["issue"] = error.detail
                result.append(item)
            return {"cases": result, "max_edge": MAX_EDGE, "review_policy": "Only explicit user save marks a mask reviewed; review does not certify CAD dimensions or change split."}

    @router.get("/api/segmentation-review/cases/{case_id}/image")
    def image(case_id: str):
        path = safe_path(get_row(case_id).get("image"))
        _, kind = dimensions(path)
        return file_response(path, {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}[kind])

    @router.get("/api/segmentation-review/cases/{case_id}/mask")
    def mask(case_id: str):
        with lock:
            row = get_row(case_id)
            record = reviewed_record(case_id, row, reviews())
            path = safe_path(record["mask"] if record else row.get("mask"))
            size, kind = dimensions(path)
            expected, _ = dimensions(safe_path(row.get("image")))
            if size != expected or kind != "PNG":
                raise HTTPException(422, "掩膜必须为与注册图像同尺寸的PNG。")
            return file_response(path, "image/png")

    @router.put("/api/segmentation-review/cases/{case_id}/review")
    async def save_review(case_id: str, request: Request):
        origin = request.headers.get("origin")
        if origin and urlparse(origin).netloc != request.url.netloc:
            raise HTTPException(403, "拒绝跨来源保存。")
        if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "image/png":
            raise HTTPException(415, "请提交PNG掩膜。")
        row = get_row(case_id)
        expected, _ = dimensions(safe_path(row.get("image")))
        chunks, total = [], 0
        async for chunk in request.stream():
            total += len(chunk)
            if total > MAX_BODY:
                raise HTTPException(413, "PNG文件超过12MB限制。")
            chunks.append(chunk)
        try:
            with Image.open(io.BytesIO(b"".join(chunks))) as uploaded:
                if uploaded.format != "PNG" or uploaded.size != expected or max(uploaded.size) > MAX_EDGE or uploaded.mode not in {"1", "L", "RGB", "RGBA"}:
                    raise HTTPException(422, "PNG必须与注册图像同尺寸，且最大边不超过1536像素。")
                if uploaded.mode == "RGBA" and uploaded.getchannel("A").getextrema() != (255, 255):
                    raise HTTPException(422, "保存掩膜必须是不透明的二值PNG。")
                binary = uploaded.convert("L")
                if any(count for index, count in enumerate(binary.histogram()) if index not in (0, 255)):
                    raise HTTPException(422, "掩膜只能包含0和255两个类别值。")
                buffer = io.BytesIO()
                binary.save(buffer, format="PNG")
                payload = buffer.getvalue()
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            raise HTTPException(422, "PNG掩膜无法读取。") from None
        with lock:
            # Re-read under the lock, preserving reviews of other cases.
            records = reviews()
            output_dir = safe_path("reviewed_masks", exists=False)
            output_dir.mkdir(exist_ok=True)
            output = safe_path(f"reviewed_masks/{case_id}.png", exists=False)
            record = {"mask": output.relative_to(data_root).as_posix(), "reviewed": True,
                      "label_source": "local_user_review", "source_image_sha256": row["source_image_sha256"],
                      "timestamp": datetime.now(timezone.utc).isoformat(), "mask_sha256": hashlib.sha256(payload).hexdigest(),
                      "width": expected[0], "height": expected[1]}
            records[case_id] = record
            review_path = safe_path("reviews.json", exists=False)
            token = uuid.uuid4().hex
            temporary_mask = output_dir / f".{token}.png.tmp"
            temporary_json = data_root / f".{token}.reviews.tmp"
            try:
                with temporary_mask.open("xb") as stream:
                    stream.write(payload)
                with temporary_json.open("x", encoding="utf8") as stream:
                    json.dump(records, stream, ensure_ascii=False, indent=2, allow_nan=False)
                temporary_mask.replace(output)
                temporary_json.replace(review_path)
            finally:
                temporary_mask.unlink(missing_ok=True)
                temporary_json.unlink(missing_ok=True)
            return {"case_id": case_id, "split": row.get("split"), "review": record}

    return router
