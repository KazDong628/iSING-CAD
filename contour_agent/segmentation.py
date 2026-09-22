"""Small GPU segmentation experiment with declared label provenance and splits.

The learned model never reads masks, OCR, references or templates at inference.
ImageNet encoder initialization is real; weak-label scores are NOT gold accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
import threading
from collections import Counter
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MODEL_SPEC = {"architecture": "Unet", "encoder_name": "resnet18", "decoder_channels": [128, 64, 32, 16, 8], "classes": 1}
UPSTREAM = {"repository": "https://github.com/qubvel-org/segmentation_models.pytorch", "tag": "v0.5.0", "commit": "420ce84b0c2df0286fa9bb2bd1499eea625c9b33", "license": "MIT"}


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def letterbox(image, size=512, mask=None):
    """Preserve aspect ratio; return exact offsets for reversing inference."""
    h, w = image.shape[:2]
    ratio = min(size / w, size / h)
    nw, nh = max(1, round(w * ratio)), max(1, round(h * ratio))
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    canvas[top:top+nh, left:left+nw] = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR)
    target = None
    if mask is not None:
        target = np.zeros((size, size), dtype=np.uint8)
        target[top:top+nh, left:left+nw] = cv2.resize(mask.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST)
    return canvas, target, {"left": left, "top": top, "width": nw, "height": nh, "original_width": w, "original_height": h}


def unletterbox(probability, transform):
    x, y, w, h = [transform[k] for k in ("left", "top", "width", "height")]
    return cv2.resize(probability[y:y+h, x:x+w].astype(np.float32), (transform["original_width"], transform["original_height"]), interpolation=cv2.INTER_LINEAR)


def tensor_image(image):
    import torch
    normalized = image.astype(np.float32) / 255.
    normalized = (normalized - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
    return torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1)))


def make_model():
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=1, decoder_channels=tuple(MODEL_SPEC["decoder_channels"]))


def pretrained_encoder(model, cache):
    """Download official torchvision weights, validate their published hash prefix."""
    import httpx
    import torch
    path = Path(cache) / "resnet18-f37072fd.pth"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temp = path.with_suffix(".download")
        with httpx.stream("GET", "https://download.pytorch.org/models/resnet18-f37072fd.pth", trust_env=False, timeout=60, follow_redirects=True) as response:
            response.raise_for_status()
            with temp.open("wb") as stream:
                for part in response.iter_bytes(): stream.write(part)
        if not sha256(temp).startswith("f37072fd"):
            raise ValueError("Official encoder weight hash does not match")
        temp.replace(path)
    digest = sha256(path)
    if not digest.startswith("f37072fd"):
        raise ValueError("Encoder weight cache failed hash validation")
    model.encoder.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return {"source": "torchvision_resnet18_imagenet1k_v1", "url": "https://download.pytorch.org/models/resnet18-f37072fd.pth", "sha256": digest}


def _manifest_digest(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise ValueError(f"Manifest requires a valid {name} SHA256")
    return value.lower()


def _manifest_file(root, value, expected_hash, name):
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"Manifest {name} path is invalid")
    file = Path(value)
    file = (file if file.is_absolute() else root/file).resolve()
    if not file.is_relative_to(root) or not file.is_file():
        raise ValueError(f"Manifest {name} escapes prepared data or is missing")
    digest = _manifest_digest(expected_hash, name)
    if sha256(file) != digest:
        raise ValueError(f"Manifest {name} content hash changed; restore the frozen artifact or prepare a new data directory")
    return file, digest


def _registration_provenance(item):
    registration = item.get("registration")
    if not isinstance(registration, dict) or registration.get("status") != "accepted":
        raise ValueError(f"Registered DXF sample {item['id']} requires accepted registration; exclude failed candidates from training")
    digest = _manifest_digest(registration.get("source_gt_sha256"), "source_gt_sha256")
    try:
        matrix = np.asarray(registration.get("transform"), dtype=float)
    except (TypeError, ValueError):
        raise ValueError("Registration transform must be a finite affine 2x3 or 3x3 matrix") from None
    if matrix.shape not in {(2,3),(3,3)} or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix[:2,:2])) < 1e-12:
        raise ValueError("Registration transform must be a finite nonsingular affine 2x3 or 3x3 matrix")
    if matrix.shape == (3,3) and not np.allclose(matrix[2], [0,0,1], atol=1e-12, rtol=0):
        raise ValueError("Registration transform must be affine, not a projective warp")
    if not isinstance(registration.get("quality"), dict) or not registration["quality"]:
        raise ValueError("Registered DXF labels require recorded registration quality")
    try:
        json.dumps(registration, allow_nan=False)
    except (TypeError, ValueError):
        raise ValueError("Registration provenance must contain finite JSON values") from None
    return {**registration, "source_gt_sha256": digest, "transform": matrix.tolist(),
            "acceptance_meaning": "Quantitative automatic registration acceptance; not human pixel review or engineering certification"}


def _ignore_array(row, size=None):
    if not row.get("ignore_mask"):
        return None
    with Image.open(row["ignore_mask"]) as image:
        if image.format != "PNG" or image.mode not in {"1", "L"}:
            raise ValueError("Ignore mask must be a binary grayscale PNG (255 means ignored)")
        mask = np.asarray(image.convert("L"))
    if size is not None and mask.shape != (size[1],size[0]):
        raise ValueError("Ignore mask and prepared image sizes differ")
    if not np.all((mask == 0) | (mask == 255)) or np.all(mask == 255):
        raise ValueError("Ignore mask must be binary and leave at least one supervised pixel")
    return mask != 0


def _label_status(rows):
    sources = {row["label_source"] for row in rows}
    if sources == {"registered_dxf_gt"}: return "registered_dxf_gt_experimental"
    if sources == {"source_heuristic"}: return "weak_supervision_experimental"
    if sources == {"local_user_review"}: return "local_user_review_experimental"
    return "mixed_supervision_experimental" if sources else "no_usable_supervision"


def read_manifest(path):
    """Validate consumed artifacts and explicit reviews without reading sources.

    Source-image hashes are provenance claims from the frozen preparation step;
    prepared image/weak-mask/review-mask hashes are checked against actual files.
    Manual review overrides retain the frozen weak-label identity alongside the
    effective mask identity. A stale or changed review is never silently used.
    """
    path = Path(path).resolve()
    manifest_bytes = path.read_bytes()
    document = json.loads(manifest_bytes.decode("utf8"))
    rows = document if isinstance(document, list) else next((document[k] for k in ("cases","rows","samples") if isinstance(document,dict) and isinstance(document.get(k),list)), None)
    if not isinstance(rows,list) or not rows:
        raise ValueError("Manifest contains no samples")
    reviews_path = path.parent/"reviews.json"
    reviews_bytes = None
    if reviews_path.exists():
        if not reviews_path.resolve().is_relative_to(path.parent) or not reviews_path.is_file():
            raise ValueError("Review registry escapes prepared data")
        reviews_bytes = reviews_path.read_bytes()
    reviews = json.loads(reviews_bytes.decode("utf8")) if reviews_bytes is not None else {}
    if not isinstance(reviews,dict):
        raise ValueError("Review registry must be a mapping of sample IDs to review records")
    prepared, validated, seen = [], [], set()
    for row in rows:
        if not isinstance(row,dict):
            raise ValueError("Manifest samples must be objects")
        item = dict(row); item["id"] = item.get("id",item.get("case_id"))
        case_id = item["id"]
        if not isinstance(case_id,str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,179}",case_id):
            raise ValueError("Manifest sample ID is unsafe or missing")
        if case_id.casefold() in seen:
            raise ValueError(f"Manifest contains duplicate sample ID: {case_id}")
        seen.add(case_id.casefold())
        if item.get("split") not in {"train","val","test"}:
            raise ValueError(f"Manifest sample {case_id} has an invalid split")
        if not isinstance(item.get("group"),str) or not item["group"].strip():
            raise ValueError(f"Manifest sample {case_id} is missing its leakage-control group")
        if "trainable" in item and not isinstance(item["trainable"],bool):
            raise ValueError(f"Manifest sample {case_id} has a nonboolean trainable flag")
        source_hash = _manifest_digest(item.get("source_image_sha256",item.get("image_sha256")),"source_image_sha256")
        if item.get("image_sha256") is not None and _manifest_digest(item["image_sha256"],"image_sha256") != source_hash:
            raise ValueError(f"Manifest sample {case_id} has conflicting source-image hashes")
        item.update(source_image_sha256=source_hash, image_sha256=source_hash)
        for hash_field in ("prepared_image_sha256","mask_sha256"):
            if item.get(hash_field) is not None:
                item[hash_field] = _manifest_digest(item[hash_field],hash_field)
        if item.get("registration") is not None and not isinstance(item["registration"],dict):
            raise ValueError("Registration provenance must be a mapping")
        validated.append(item)
        if item.get("trainable") is False or not item.get("image") or not item.get("mask"):
            continue
        image, image_hash = _manifest_file(path.parent,item["image"],item.get("prepared_image_sha256"),f"{case_id} prepared image")
        mask, mask_hash = _manifest_file(path.parent,item["mask"],item.get("mask_sha256"),f"{case_id} frozen mask")
        if item.get("label_source") not in {"source_heuristic", "registered_dxf_gt"}:
            raise ValueError(f"Manifest sample {case_id} has an unsupported frozen label source; explicit user reviews belong in reviews.json")
        if item["label_source"] == "registered_dxf_gt":
            item["registration"] = _registration_provenance(item)
        if item.get("ignore_mask"):
            ignore_path, ignore_hash = _manifest_file(path.parent,item["ignore_mask"],item.get("ignore_mask_sha256"),f"{case_id} ignore mask")
            item.update(ignore_mask=str(ignore_path), ignore_mask_sha256=ignore_hash)
            with Image.open(image) as source:
                ignored = _ignore_array(item, source.size)
            item["ignore_pixels"] = int(ignored.sum())
            item["ignore_fraction"] = float(ignored.mean())
        elif item.get("ignore_mask_sha256") is not None:
            raise ValueError("Ignore mask hash requires its registered mask path")
        item.update(image=str(image), mask=str(mask), prepared_image_sha256=image_hash, mask_sha256=mask_hash,
                    frozen_mask=str(mask), frozen_mask_sha256=mask_hash,
                    frozen_label_source=item["label_source"], reviewed=False)
        review = reviews.get(case_id,{})
        if not isinstance(review,dict) or ("reviewed" in review and not isinstance(review["reviewed"],bool)):
            raise ValueError(f"Review record for {case_id} is malformed")
        if review.get("reviewed") is True:
            if review.get("label_source") != "local_user_review":
                raise ValueError(f"Review for {case_id} lacks explicit local_user_review provenance")
            if _manifest_digest(review.get("source_image_sha256"),"review source_image_sha256") != source_hash:
                raise ValueError(f"Review for {case_id} belongs to a different source image; re-review the current source")
            expected = f"reviewed_masks/{case_id}.png"
            if review.get("mask") != expected:
                raise ValueError(f"Review for {case_id} must use its registered reviewed_masks file")
            reviewed_mask, reviewed_hash = _manifest_file(path.parent,expected,review.get("mask_sha256"),f"{case_id} reviewed mask")
            if not reviewed_mask.is_relative_to(path.parent/"reviewed_masks"):
                raise ValueError(f"Review for {case_id} escapes the reviewed mask directory")
            item.update(mask=str(reviewed_mask),mask_sha256=reviewed_hash,reviewed=True,label_source="local_user_review")
        prepared.append(item)
    # Check all declared rows, including failed/untrainable rows, so later review
    # cannot silently introduce a source/group already allocated to another split.
    for key in ("group","source_image_sha256","prepared_image_sha256"):
        mapping = {}
        for row in validated:
            value = row.get(key)
            if value is not None:
                mapping.setdefault(value,set()).add(row["split"])
        if any(len(splits) != 1 for splits in mapping.values()):
            raise ValueError(f"Cross-split leakage in {key}")
    artifacts = [{key:row[key] for key in ("id","split","group","source_image_sha256","prepared_image_sha256","frozen_mask_sha256","mask_sha256","label_source","reviewed","registration","ignore_mask_sha256","ignore_pixels","ignore_fraction") if key in row} for row in prepared]
    identities = [{key:row[key] for key in ("id","split","group","source_image_sha256")} for row in validated]
    artifact_bytes = json.dumps(artifacts,sort_keys=True,separators=(",",":")).encode("utf8")
    return prepared, {"manifest_sha256":hashlib.sha256(manifest_bytes).hexdigest(),
                      "reviews_sha256":hashlib.sha256(reviews_bytes).hexdigest() if reviews_bytes is not None else None,
                      "consumed_artifacts_sha256":hashlib.sha256(artifact_bytes).hexdigest(),"consumed_artifacts":artifacts,
                      "case_identities":identities,
                      "total_source_cases":len(rows),"usable_cases":len(prepared),
                      "label_status":_label_status(prepared),
                      "label_source_counts":dict(Counter(row["label_source"] for row in prepared)),
                      "frozen_label_source_counts":dict(Counter(row["frozen_label_source"] for row in prepared)),
                      "registration_status_counts":dict(Counter((row.get("registration") or {}).get("status", "not_applicable") for row in validated)),
                      "ignore_mask_cases":sum(bool(row.get("ignore_mask")) for row in prepared),
                      "human_reviewed_cases":sum(row["reviewed"] is True for row in prepared),
                      "split_claim":"development_split_not_blind_test"}


class DrawingDataset:
    def __init__(self, rows, size, augment=False, seed=42):
        self.rows, self.size, self.augment = rows, size, augment
        self.generator = random.Random(seed)
        # Prepared <=1536 images are cached once to avoid repeated decoding.
        self.samples = [(np.asarray(Image.open(row["image"]).convert("RGB")), np.asarray(Image.open(row["mask"]).convert("L")) > 127) for row in rows]
        if any(image.shape[:2] != mask.shape for image,mask in self.samples):
            raise ValueError("Prepared image and segmentation mask sizes differ")
        self.ignore_masks = [_ignore_array(row, (image.shape[1], image.shape[0])) for row,(image,_) in zip(rows,self.samples)]

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        import torch
        image, mask = self.samples[index]
        ignored = self.ignore_masks[index]
        image, mask, transform = letterbox(image, self.size, mask)
        ignore = np.zeros((self.size,self.size),np.uint8)
        if ignored is not None:
            x,y,w,h = [transform[k] for k in ("left","top","width","height")]
            ignore[y:y+h,x:x+w] = cv2.resize(ignored.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)
        if self.augment:
            if self.generator.random() < .5: image, mask, ignore = image[:, ::-1].copy(), mask[:, ::-1].copy(), ignore[:, ::-1].copy()
            # Small affine changes affect the image and target identically.
            matrix = cv2.getRotationMatrix2D((self.size/2, self.size/2), self.generator.uniform(-8, 8), self.generator.uniform(.9, 1.05))
            image = cv2.warpAffine(image, matrix, (self.size, self.size), borderValue=(255,255,255))
            mask = cv2.warpAffine(mask, matrix, (self.size, self.size), flags=cv2.INTER_NEAREST, borderValue=0)
            ignore = cv2.warpAffine(ignore, matrix, (self.size, self.size), flags=cv2.INTER_NEAREST, borderValue=0)
            if self.generator.random() < .75:
                from .segmentation_metrics import add_annotation_noise
                image = add_annotation_noise(image, self.generator.randrange(2**31), mask=mask, strength=1.0)
                if isinstance(image, tuple): image = image[0]
        target = mask.astype(np.float32)
        target[ignore != 0] = -1.
        return tensor_image(image), torch.from_numpy(np.ascontiguousarray(target[None]))


def binary_loss(logits, target):
    import torch
    valid = target >= 0
    target = target.clamp(0,1)
    losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (losses*valid).sum()/valid.sum().clamp_min(1)
    probability = logits.sigmoid()*valid
    target = target*valid
    dice = (2*(probability*target).sum((1,2,3)) + 1) / (probability.sum((1,2,3))+target.sum((1,2,3))+1)
    return bce + 1 - dice.mean()


def evaluate_loader(model, loader, device):
    import torch
    model.eval(); values=[]
    with torch.inference_mode():
        for images, masks in loader:
            predictions = model(images.to(device)).sigmoid() > .5
            targets, valid = masks.to(device) > .5, masks.to(device) >= 0
            intersection = (predictions & targets & valid).sum((1,2,3)).cpu().numpy()
            union = ((predictions | targets) & valid).sum((1,2,3)).cpu().numpy()
            observed = valid.sum((1,2,3)).cpu().numpy() > 0
            values.extend(((intersection[observed] + 1e-6)/(union[observed]+1e-6)).tolist())
    return float(np.mean(values)) if values else None


def train(manifest, output, *, epochs=20, size=512, batch_size=2, seed=42, device=None):
    import torch
    from torch.utils.data import DataLoader
    torch.set_num_threads(4); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if size < 128 or size % 32 or size > 1024: raise ValueError("size must be a multiple of32 between128 and1024")
    if not 1 <= epochs <= 500 or not 1 <= batch_size <= 16: raise ValueError("Invalid training budget")
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if (output / "best.pt").exists(): raise ValueError("Use a new run directory; checkpoints are never overwritten across runs")
    rows, provenance = read_manifest(manifest)
    training = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "val"]
    if not training or not validation: raise ValueError("Train and validation groups must both be nonempty")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(); initialization = pretrained_encoder(model, ROOT / "runtime/segmentation/weights")
    model.to(device)
    train_loader = DataLoader(DrawingDataset(training, size, True, seed), batch_size=batch_size, shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(DrawingDataset(validation, size), batch_size=1, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": 1e-4},
                                  {"params": list(model.decoder.parameters())+list(model.segmentation_head.parameters()), "lr": 1e-3}], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=device.startswith("cuda"))
    report = {"status":"running", "model":MODEL_SPEC, "upstream":UPSTREAM, "initialization":initialization,
              "seed":seed,"size":size,"batch_size":batch_size,"epochs_planned":epochs,"device":device,
              "gpu":torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
              "split_counts":{split:sum(r["split"]==split for r in rows) for split in ("train","val","test")},
              "train_ids":[r["id"] for r in training], "val_ids":[r["id"] for r in validation],
              "reviewed_counts":{split:sum(r["split"]==split and r.get("reviewed") is True for r in rows) for split in ("train","val","test")},
              "provenance":provenance,"label_status":provenance["label_status"],
              "validation_meaning":"agreement with declared development labels; registered DXF masks are derived supervision, not human pixel review",
              "checkpoint_selection_metric":"mean validation region IoU on nonignored pixels; full boundary metrics are reported separately at evaluation",
              "test_used_for_training_or_checkpoint_selection":False,"automatic_promotion":False,"history":[]}
    started=time.monotonic(); best=-1.
    legacy_weak_names = provenance["label_status"] == "weak_supervision_experimental"
    report["initial_val_label_iou"] = evaluate_loader(model, val_loader, device)
    if report["initial_val_label_iou"] is None: raise ValueError("Validation has no supervised pixels")
    if legacy_weak_names: report["initial_val_weak_iou"] = report["initial_val_label_iou"]
    write_json(output/"training.json",report)
    for epoch in range(epochs):
        model.train()
        frozen = epoch < min(2, max(0, epochs-1))
        for parameter in model.encoder.parameters(): parameter.requires_grad_(not frozen)
        if frozen: model.encoder.eval()
        losses=[]
        for images,masks in train_loader:
            images,masks=images.to(device),masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", enabled=device.startswith("cuda")):
                loss=binary_loss(model(images),masks)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        iou=evaluate_loader(model,val_loader,device)
        item={"epoch":epoch+1,"train_loss":float(np.mean(losses)),"val_label_iou":iou,"encoder_frozen":frozen,
              "elapsed_seconds":round(time.monotonic()-started,2)}
        if legacy_weak_names: item["val_weak_iou"] = iou
        report["history"].append(item)
        if iou > best:
            best=iou
            checkpoint={"state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()}, "model":MODEL_SPEC,
                        "size":size,"epoch":epoch+1,"val_label_iou":iou,"provenance":provenance,
                        "label_status":provenance["label_status"], "inference_inputs":"image_only"}
            if legacy_weak_names: checkpoint["val_weak_iou"] = iou
            temporary=output/"best.tmp"; torch.save(checkpoint,temporary);temporary.replace(output/"best.pt")
            report["best_epoch"]=epoch+1;report["best_val_label_iou"]=best
            if legacy_weak_names: report["best_val_weak_iou"] = best
        write_json(output/"training.json",report);print(json.dumps(item),flush=True)
    report.update(status="completed",elapsed_seconds=round(time.monotonic()-started,2),checkpoint_sha256=sha256(output/"best.pt"),
                  peak_gpu_memory_mb=round(torch.cuda.max_memory_allocated()/1024**2,1) if device.startswith("cuda") else None)
    write_json(output/"training.json",report)
    return report


class Segmenter:
    """Image-only inference; no access to training labels or drawing identifiers."""
    def __init__(self, checkpoint, device=None, *, expected_manifest_sha256=None):
        import torch
        torch.set_num_threads(4)
        # Only locally produced state dictionaries; no Python object deserialization.
        state=torch.load(checkpoint,map_location="cpu",weights_only=True)
        if state.get("model") != MODEL_SPEC: raise ValueError("Unsupported segmentation checkpoint specification")
        size=state.get("size")
        if isinstance(size,bool) or not isinstance(size,int) or not 128 <= size <= 1024 or size % 32:
            raise ValueError("Checkpoint size must be a nonboolean integer multiple of32 between128 and1024")
        training_provenance=state.get("provenance")
        if training_provenance is not None and not isinstance(training_provenance,dict):
            raise ValueError("Checkpoint training provenance must be a mapping when present")
        if expected_manifest_sha256 is not None:
            training_manifest=(training_provenance or {}).get("manifest_sha256")
            if not isinstance(training_manifest,str) or not re.fullmatch(r"[0-9a-fA-F]{64}",training_manifest):
                raise ValueError("Checkpoint lacks valid training manifest provenance; prediction is supported, but development evaluation requires the frozen training manifest")
            if training_manifest.lower() != _manifest_digest(expected_manifest_sha256,"evaluation manifest"):
                raise ValueError("Checkpoint training manifest does not match the evaluation manifest; use the same frozen data split")
        self.device=device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model=make_model();self.model.load_state_dict(state["state_dict"],strict=True)
        self.model.to(self.device).eval();self.size=size
        self.metadata={"checkpoint_sha256":sha256(checkpoint),"label_status":state.get("label_status"),
                       "size":self.size,"device":self.device,"training_provenance":training_provenance}

    def predict_array(self, image_rgb, *, inference_size=None):
        import torch
        size = self.size if inference_size is None else inference_size
        if isinstance(size, bool) or not isinstance(size, int) or not 128 <= size <= 1024 or size % 32:
            raise ValueError("Inference size must be a multiple of 32 between 128 and 1024")
        boxed,_,transform=letterbox(image_rgb,size)
        with torch.inference_mode():
            probability=self.model(tensor_image(boxed)[None].to(self.device)).sigmoid()[0,0].cpu().numpy()
        return unletterbox(probability,transform)

    def extract(self,image_path,output_dir=None):
        from .mask_geometry import extract_mask_profile
        with Image.open(image_path) as source:
            if source.width*source.height > 80_000_000: raise ValueError("Image exceeds processing limit")
            original_size=source.size
            # Bound mask size, retain the exact mapping to original source pixels.
            source=source.convert("RGB");source.thumbnail((1536,1536),Image.Resampling.LANCZOS)
            rgb=np.asarray(source)
        probability=self.predict_array(rgb)
        raw_probability=probability.copy()
        from .segmentation_refinement import refine_probabilities, connectivity, detail_inference_size
        detail_size=detail_inference_size(self.size)
        refinement={"algorithm":"guarded-boundary-multiresolution-v1","status":"skipped",
                    "reason":"No higher bounded inference resolution available.","ground_truth_used":False,
                    "before":connectivity(probability>=.5),"after":connectivity(probability>=.5)}
        if detail_size > self.size:
            try:
                detail=self.predict_array(rgb,inference_size=detail_size)
                probability,refinement=refine_probabilities(rgb,probability,detail,model_size=self.size)
                refinement["detail_inference_size"]=detail_size
            except (RuntimeError, MemoryError) as error:
                # Preserve global output if the optional detail pass exhausts resources.
                refinement.update(status="failed_preserved_coarse",reason=type(error).__name__)
        result=extract_mask_profile(probability,original_size,output_dir)
        result["evidence"]["refinement"]=refinement
        result["model"]=self.metadata;result["learned_segmentation"]=True
        result["issues"].append(f"Experimental model label status: {self.metadata.get('label_status') or 'unknown'}. Segmentation confidence does not establish dimensional accuracy.")
        if output_dir:
            output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
            Image.fromarray(((probability>=.5)*255).astype(np.uint8)).save(output_dir/"prediction-mask.png")
            Image.fromarray(((raw_probability>=.5)*255).astype(np.uint8)).save(output_dir/"raw-prediction-mask.png")
            np.savez_compressed(output_dir/"prediction-probabilities.npz",raw=raw_probability,refined=probability)
            raw_overlay=rgb.copy();raw_mask=raw_probability>=.5
            raw_overlay[raw_mask]=(raw_overlay[raw_mask]*.6+np.array([0,170,120])*.4).astype(np.uint8)
            Image.fromarray(raw_overlay).save(output_dir/"raw-prediction-overlay.png")
            write_json(output_dir/"refinement.json",refinement)
            overlay=rgb.copy();mask=probability>=.5
            overlay[mask]=(overlay[mask]*.6+np.array([0,170,120])*.4).astype(np.uint8)
            Image.fromarray(overlay).save(output_dir/"prediction-overlay.png")
            write_json(output_dir/"segmentation.json",result)
        return result


def _verify_same_source_split(proof, current):
    previous = proof.get("case_identities") or proof.get("consumed_artifacts")
    actual = current.get("case_identities")
    if not isinstance(previous,list) or not previous or not isinstance(actual,list):
        raise ValueError("Label-change comparison requires source IDs/hashes/splits from training or an explicit comparison manifest")
    indexed = {row.get("id"):row for row in previous if isinstance(row,dict)}
    if len(indexed) != len(previous) or set(indexed) != {row["id"] for row in actual}:
        raise ValueError("Comparison manifests must retain exactly the same source case IDs, including unavailable-label rows")
    for row in actual:
        old = indexed[row["id"]]
        if old.get("split") != row["split"] or _manifest_digest(old.get("source_image_sha256"),"comparison source") != row["source_image_sha256"]:
            raise ValueError(f"Comparison source hash or split changed for {row['id']}")
        if old.get("group") is not None and old["group"] != row["group"]:
            raise ValueError(f"Comparison source group changed for {row['id']}")
    return {"verified":True,"case_count":len(actual),"rule":"Exact case IDs, original-image SHA256 and split; recorded groups must also match. Label changes are explicit."}


def _valid_region_metrics(prediction, target, valid):
    count = int(valid.sum())
    intersection = int((prediction & target & valid).sum())
    union = int(((prediction | target) & valid).sum())
    denominator = int(((prediction & valid).sum()+(target & valid).sum()))
    return {"iou":(intersection/union if union else 1.) if count else None,
            "dice":(2*intersection/denominator if denominator else 1.) if count else None,
            "supervised_pixels":count,"ignored_pixels":int(valid.size-count),
            "boundary_metrics":"Not computed on masked fragments; complete-boundary metrics remain in clean/noise."}


def _excluded_split_cases(manifest, provenance, split, scored_ids):
    raw = Path(manifest).read_bytes()
    if hashlib.sha256(raw).hexdigest() != provenance["manifest_sha256"]:
        raise ValueError("Manifest changed during evaluation")
    document = json.loads(raw)
    rows = document if isinstance(document,list) else next(document[k] for k in ("cases","rows","samples") if isinstance(document.get(k),list))
    excluded = []
    for row in rows:
        case_id = row.get("id",row.get("case_id"))
        if row["split"] != split or case_id in scored_ids:
            continue
        reasons = []
        if row.get("trainable") is False: reasons.append("declared_untrainable")
        if not row.get("image"): reasons.append("missing_prepared_image")
        if not row.get("mask"): reasons.append("missing_label_mask")
        registration = row.get("registration") or {}
        excluded.append({"id":case_id,"split":split,"reasons":reasons or ["unavailable_supervision"],
                         "registration_status":registration.get("status"),
                         "failed_quality_checks":registration.get("failed_quality_checks",[]),
                         "issues":[str(issue)[:500] for issue in row.get("issues",[]) if isinstance(issue,str)]})
    return excluded


def evaluate(manifest, checkpoint, output, *, split="test", allow_label_change=False, comparison_manifest=None):
    from .segmentation_metrics import segmentation_metrics,add_annotation_noise
    rows, provenance=read_manifest(manifest)
    selected=[r for r in rows if r["split"]==split]
    declared_ids = [row["id"] for row in provenance["case_identities"] if row["split"] == split]
    scored_ids = [row["id"] for row in selected]
    if not declared_ids: raise ValueError("Requested development split is empty")
    excluded = _excluded_split_cases(manifest,provenance,split,set(scored_ids))
    comparison_provenance = None
    if comparison_manifest is not None:
        _,comparison_provenance = read_manifest(comparison_manifest)
        allow_label_change = True  # Supplying this path explicitly requests comparison.
        _verify_same_source_split(comparison_provenance,provenance)
    expected_manifest = comparison_provenance["manifest_sha256"] if comparison_provenance else None if allow_label_change else provenance["manifest_sha256"]
    segmenter=Segmenter(checkpoint,expected_manifest_sha256=expected_manifest)
    training_provenance=segmenter.metadata["training_provenance"]
    if not isinstance(training_provenance,dict):
        raise ValueError("Evaluation requires checkpoint training provenance")
    training_manifest = _manifest_digest(training_provenance.get("manifest_sha256"),"checkpoint training manifest")
    comparison_identity = _verify_same_source_split(comparison_provenance or training_provenance,provenance) if allow_label_change else None
    reviews_match=(training_provenance["reviews_sha256"] == provenance["reviews_sha256"]) if "reviews_sha256" in training_provenance else None
    training_artifacts=training_provenance.get("consumed_artifacts_sha256")
    current_artifacts=provenance.get("consumed_artifacts_sha256")
    labels_match=(training_artifacts == current_artifacts) if training_artifacts is not None and current_artifacts is not None else None
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    records=[]
    for index,row in enumerate(selected):
        source_image=np.asarray(Image.open(row["image"]).convert("RGB"));mask=np.asarray(Image.open(row["mask"]).convert("L"))>127
        ignored = _ignore_array(row,(source_image.shape[1],source_image.shape[0]))
        if ignored is None:
            ignore_target = np.zeros((512,512),np.uint8)
        else:
            _,ignore_target,_ = letterbox(source_image,512,ignored)
        # Preserve native prepared source details for models trained above 512.
        # Only binary predictions/targets share the fixed scoring grid.
        image,target,transform=letterbox(source_image,512,mask)
        x,y,w,h=[transform[k] for k in ("left","top","width","height")]
        noise_layer=add_annotation_noise(np.full_like(image,255),20260920+index,mask=target,strength=1.)
        if isinstance(noise_layer,tuple):noise_layer=noise_layer[0]
        native_layer=cv2.resize(noise_layer[y:y+h,x:x+w],(source_image.shape[1],source_image.shape[0]),interpolation=cv2.INTER_NEAREST)
        noisy_source=np.minimum(source_image,native_layer)
        clean_native=segmenter.predict_array(source_image)>=.5
        noisy_native=segmenter.predict_array(noisy_source)>=.5
        if clean_native.shape != mask.shape or noisy_native.shape != mask.shape:
            raise ValueError("Segmenter predictions must retain prepared source image dimensions")
        _,clean,_=letterbox(source_image,512,clean_native)
        _,disturbed,_=letterbox(source_image,512,noisy_native)
        clean=clean.astype(bool);disturbed=disturbed.astype(bool)
        noise,_,_=letterbox(noisy_source,512)
        clean_score=segmentation_metrics(clean[y:y+h,x:x+w],target[y:y+h,x:x+w])
        noisy_score=segmentation_metrics(disturbed[y:y+h,x:x+w],target[y:y+h,x:x+w])
        valid = ignore_target[y:y+h,x:x+w] == 0
        clean_valid = _valid_region_metrics(clean[y:y+h,x:x+w],target[y:y+h,x:x+w].astype(bool),valid)
        noisy_valid = _valid_region_metrics(disturbed[y:y+h,x:x+w],target[y:y+h,x:x+w].astype(bool),valid)
        case_dir=out/row["id"];case_dir.mkdir(exist_ok=True)
        overlay=noise.copy();overlay[disturbed]=(overlay[disturbed]*.5+np.array([0,170,120])*.5).astype(np.uint8)
        Image.fromarray(overlay).save(case_dir/"noise-prediction.png")
        Image.fromarray((clean*255).astype(np.uint8)).save(case_dir/"prediction.png")
        records.append({"id":row["id"],"reviewed":row.get("reviewed",False),"label_source":row.get("label_source"),
                        "inference_input_size":{"width":source_image.shape[1],"height":source_image.shape[0]},
                        "registration":row.get("registration"),"ignore_mask_sha256":row.get("ignore_mask_sha256"),
                        "clean":clean_score,"noise":noisy_score,"clean_valid_region":clean_valid,"noise_valid_region":noisy_valid})
    valid_scores = [row["clean_valid_region"]["iou"] for row in records if row["clean_valid_region"]["iou"] is not None]
    report={"split":split,"count":len(records),"provenance":provenance,"model":segmenter.metadata,"cases":records,
            "status":"completed" if records else "no_usable_labels", "total":len(declared_ids),
            "excluded":len(excluded),"excluded_case_ids":[row["id"] for row in excluded],"excluded_cases":excluded,
            "declared_split_count":len(declared_ids),"scored_case_ids":scored_ids,
            "unavailable_label_case_ids":[case_id for case_id in declared_ids if case_id not in scored_ids],
            "label_status":provenance["label_status"],
            "checkpoint_manifest_match":training_manifest == provenance["manifest_sha256"],"reviews_match":reviews_match,
            "allow_label_change":bool(allow_label_change),"comparison_identity":comparison_identity,
            "comparison_manifest_sha256":comparison_provenance["manifest_sha256"] if comparison_provenance else None,
            "label_provenance":{"training_reviews_sha256":training_provenance.get("reviews_sha256"),
                                "evaluation_reviews_sha256":provenance["reviews_sha256"],
                                "training_consumed_artifacts_sha256":training_artifacts,
                                "evaluation_consumed_artifacts_sha256":current_artifacts,
                                "consumed_artifacts_match":labels_match,
                                "meaning":"Source/split identity follows the strict manifest gate or an explicitly verified comparison manifest. Review and effective-label changes are separately reported; missing historical hashes mean unknown."},
            "true_accuracy_verified":False,"meaning":"Agreement with declared development targets, including registered DXF-derived masks when specified; not independently human-verified pixel or CAD accuracy.",
            "mean_clean_iou":float(np.mean([r["clean"]["iou"] for r in records])) if records else None,"mean_noisy_iou":float(np.mean([r["noise"]["iou"] for r in records])) if records else None,
            "mean_clean_valid_region_iou":float(np.mean(valid_scores)) if valid_scores else None,
            "mean_clean_boundary_f1":float(np.mean([r["clean"]["boundary_f1"] for r in records])) if records else None,
            "mean_noisy_boundary_f1":float(np.mean([r["noise"]["boundary_f1"] for r in records])) if records else None,
            "inference_input":"native_prepared_image; model performs its own configured input resize",
            "scoring_grid":{"size":512,"prediction":"threshold native probability at 0.5, then nearest-neighbor binary resize","padding_scored":False},
            "noise_policy":"Fixed seeded annotation layer on white 512 scoring canvas, cropped and nearest-resized to native prepared source before darkening. Same layer for every checkpoint; source image itself is never downsampled before inference.",
            "development_only":True,"blind_test":False,
            "ignore_policy":"Only loss/checkpoint selection and valid_region IoU exclude declared ignore pixels. clean/noise and boundary metrics always use complete masks."}
    write_json(out/"evaluation.json",report);return report


_model_lock = threading.RLock()


@lru_cache(maxsize=1)
def _cached_model(checkpoint, modified_ns):
    return Segmenter(checkpoint)


def cached_extract(checkpoint, image_path, output_dir):
    # Serialize GPU use across the workbench's two worker threads.
    checkpoint=Path(checkpoint).resolve()
    with _model_lock:
        return _cached_model(str(checkpoint),checkpoint.stat().st_mtime_ns).extract(image_path,output_dir)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="command",required=True)
    prepare=sub.add_parser("prepare");prepare.add_argument("--dataset",type=Path,default=ROOT/"__dataset");prepare.add_argument("--output",type=Path,default=ROOT/"runtime/segmentation/data")
    training=sub.add_parser("train");training.add_argument("--manifest",type=Path,default=ROOT/"runtime/segmentation/data/manifest.json");training.add_argument("--output",type=Path,required=True);training.add_argument("--epochs",type=int,default=20);training.add_argument("--size",type=int,default=512);training.add_argument("--batch-size",type=int,default=2);training.add_argument("--device")
    prediction=sub.add_parser("predict");prediction.add_argument("--checkpoint",type=Path,required=True);prediction.add_argument("--image",type=Path,required=True);prediction.add_argument("--output",type=Path,required=True)
    evaluation=sub.add_parser("evaluate");evaluation.add_argument("--manifest",type=Path,default=ROOT/"runtime/segmentation/data/manifest.json");evaluation.add_argument("--checkpoint",type=Path,required=True);evaluation.add_argument("--output",type=Path,required=True);evaluation.add_argument("--split",choices=("val","test"),default="test")
    evaluation.add_argument("--allow-label-change",action="store_true");evaluation.add_argument("--comparison-manifest",type=Path)
    args=parser.parse_args()
    if args.command=="prepare":
        from .segmentation_data import prepare_dataset
        result=prepare_dataset(args.dataset,args.output)
    elif args.command=="train": result=train(args.manifest,args.output,epochs=args.epochs,size=args.size,batch_size=args.batch_size,device=args.device)
    elif args.command=="predict":result=Segmenter(args.checkpoint).extract(args.image,args.output)
    else:result=evaluate(args.manifest,args.checkpoint,args.output,split=args.split,allow_label_change=args.allow_label_change,comparison_manifest=args.comparison_manifest)
    if args.command in {"train","evaluate"}:
        fields=("status","split","count","total","excluded","label_status","best_epoch","best_val_label_iou","elapsed_seconds","checkpoint_sha256","split_counts","mean_clean_iou","mean_noisy_iou","mean_clean_boundary_f1","mean_noisy_boundary_f1")
        summary={key:result[key] for key in fields if key in result}
        summary["label_source_counts"]=result.get("provenance",{}).get("label_source_counts")
        summary["report_path"]=str(Path(args.output)/("training.json" if args.command == "train" else "evaluation.json"))
    else:
        summary={k:v for k,v in result.items() if k not in {"cases","rows","history","polyline_px","raw_polyline_px","candidates","train_ids","val_ids"}}
    print(json.dumps(summary,ensure_ascii=False,default=str),flush=True)


if __name__=="__main__":main()
