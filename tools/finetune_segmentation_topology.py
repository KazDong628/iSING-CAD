"""Bounded warm-start experiment; development validation selects, never test."""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from contour_agent.segmentation import (DrawingDataset, MODEL_SPEC, _verify_same_source_split,
                                       binary_loss, make_model, read_manifest, sha256, write_json)
from contour_agent.segmentation_metrics import segmentation_metrics
from contour_agent.segmentation_topology_training import connectivity_metrics, material_regularization, audit_manifest


def evaluate(model, loader, device, ids):
    model.eval()
    rows = []
    with torch.inference_mode():
        for index, (images, targets) in enumerate(loader):
            probability = model(images.to(device)).sigmoid()[0, 0].cpu().numpy()
            target = targets.numpy()[0, 0]
            valid, truth, prediction = target >= 0, target > .5, probability >= .5
            union = int(((prediction | truth) & valid).sum())
            iou = int((prediction & truth & valid).sum()) / union if union else 1.
            geometry = segmentation_metrics(prediction, truth)
            pred_top = connectivity_metrics(prediction, valid=valid)
            gt_top = connectivity_metrics(truth, valid=valid)
            topology_error = sum(abs(pred_top[key]-gt_top[key]) for key in ("components_4", "components_8", "holes_4", "holes_8"))
            rows.append({"id": ids[index], "valid_region_iou": iou,
                         "complete_supervision": bool(valid.all()), "full_mask": geometry,
                         "prediction_topology": pred_top, "target_topology": gt_top,
                         "topology_count_error": topology_error if valid.all() else None})
    complete = [r for r in rows if r["complete_supervision"]]
    distances = [r["full_mask"]["average_symmetric_boundary_distance_px"] for r in complete]
    return {"mean_iou": float(np.mean([r["valid_region_iou"] for r in rows])),
            "mean_boundary_distance_px": float(np.mean(distances)) if distances and all(v is not None for v in distances) else None,
            "mean_topology_count_error": float(np.mean([r["topology_count_error"] for r in complete])) if complete else None,
            "complete_topology_cases": len(complete), "cases": rows}


def improves_without_regression(candidate, incumbent):
    """Frozen development criterion; no threshold relaxation or test tuning."""
    fields = ("mean_boundary_distance_px", "mean_topology_count_error")
    if any(candidate[k] is None or incumbent[k] is None for k in fields): return False
    no_regression = candidate["mean_iou"] >= incumbent["mean_iou"] and all(candidate[k] <= incumbent[k] for k in fields)
    strict_gain = candidate["mean_iou"] > incumbent["mean_iou"] or any(candidate[k] < incumbent[k] for k in fields)
    return no_regression and strict_gain


def finetune(manifest, checkpoint, output, *, epochs=12, batch_size=2, seed=142,
             boundary_weight=.2, topology_weight=.03, device=None, allow_label_change=False):
    if not 1 <= epochs <= 12 or not 1 <= batch_size <= 4:
        raise ValueError("This experiment is bounded to 1-12 epochs and batch size 1-4")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()): raise ValueError("Use a new empty experiment directory")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    rows, provenance = read_manifest(manifest)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state.get("model") != MODEL_SPEC: raise ValueError("Unsupported initial model")
    initial_proof = state.get("provenance") or {}
    _verify_same_source_split(initial_proof, provenance)
    changed_labels = initial_proof.get("consumed_artifacts_sha256") != provenance["consumed_artifacts_sha256"]
    if changed_labels and not allow_label_change: raise ValueError("Different derived labels require explicit --allow-label-change")
    training, validation = [r for r in rows if r["split"] == "train"], [r for r in rows if r["split"] == "val"]
    if not training or not validation: raise ValueError("Nonempty train and validation partitions required")
    size = state["size"]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(); model.load_state_dict(state["state_dict"], strict=True); model.to(device)
    train_loader = DataLoader(DrawingDataset(training, size, True, seed), batch_size=batch_size, shuffle=True,
                              num_workers=0, generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(DrawingDataset(validation, size), batch_size=1, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": 1e-5},
                                 {"params": list(model.decoder.parameters())+list(model.segmentation_head.parameters()), "lr": 8e-5}], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=device.startswith("cuda"))
    started = time.monotonic()
    initial = evaluate(model, val_loader, device, [r["id"] for r in validation])
    shutil.copy2(checkpoint, output/"best.pt")
    incumbent = initial
    report = {"status": "running", "format": "material-topology-finetune-v1", "model": MODEL_SPEC,
              "initial_checkpoint": str(Path(checkpoint).resolve()), "initial_checkpoint_sha256": sha256(checkpoint),
              "initial_training_manifest_sha256": initial_proof.get("manifest_sha256"), "provenance": provenance,
              "label_change_explicitly_allowed": bool(allow_label_change), "labels_changed": changed_labels,
              "epochs_planned": epochs, "seed": seed, "size": size, "batch_size": batch_size, "device": device,
              "loss": {"base": "BCE+Dice", "boundary_weight": boundary_weight, "topology_weight": topology_weight,
                       "topology": "soft 2x2 Euler local/global comparison in both 4/8 adjacencies; not a topology guarantee"},
              "train_ids": [r["id"] for r in training], "val_ids": [r["id"] for r in validation],
              "test_used_for_training_or_checkpoint_selection": False, "automatic_promotion": False,
              "selection_rule": "Pareto non-regression against incumbent: valid IoU not lower, full-supervision boundary distance and component/hole count error not higher; at least one strict improvement. Initial checkpoint participates.",
              "augmentation": "Existing paired whole-image affine/flip and annotation-noise augmentation; no partial crops",
              "validation_meaning": "Development label agreement, not blind accuracy or manufacturing verification",
              "initial_validation": initial, "best_epoch": 0, "best_validation": initial, "history": []}
    write_json(output/"training.json", report)
    for epoch in range(epochs):
        model.train()
        losses, terms = [], []
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", enabled=device.startswith("cuda")):
                logits = model(images)
                regularization, diagnostics = material_regularization(logits, targets, boundary_weight=boundary_weight, topology_weight=topology_weight)
                loss = binary_loss(logits, targets) + regularization
            if not torch.isfinite(loss): raise ValueError("Nonfinite loss; candidate checkpoint not replaced")
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            scaler.step(optimizer); scaler.update()
            losses.append(float(loss.detach().cpu()))
            terms.append({key: float(value.detach().cpu()) for key, value in diagnostics.items()})
        scheduler.step()
        metrics = evaluate(model, val_loader, device, [r["id"] for r in validation])
        selected = improves_without_regression(metrics, incumbent)
        if selected:
            incumbent = metrics
            trained = {"state_dict": {k:v.detach().cpu() for k,v in model.state_dict().items()}, "model": MODEL_SPEC,
                       "size": size, "epoch": epoch+1, "val_label_iou": metrics["mean_iou"], "provenance": provenance,
                       "label_status": provenance["label_status"], "inference_inputs": "image_only",
                       "warm_start_checkpoint_sha256": report["initial_checkpoint_sha256"],
                       "topology_training": report["loss"]}
            temporary = output/"best.tmp"; torch.save(trained, temporary); temporary.replace(output/"best.pt")
            report.update(best_epoch=epoch+1, best_validation=metrics)
        item = {"epoch": epoch+1, "train_loss": float(np.mean(losses)), "loss_terms": {k: float(np.mean([r[k] for r in terms])) for k in terms[0]},
                "validation": metrics, "selected": selected, "elapsed_seconds": round(time.monotonic()-started, 2)}
        report["history"].append(item)
        write_json(output/"training.json", report)
        print(json.dumps({k:v for k,v in item.items() if k != "validation"} | {"val_iou": metrics["mean_iou"],
                         "val_boundary_px": metrics["mean_boundary_distance_px"], "val_topology_error": metrics["mean_topology_count_error"]}), flush=True)
    report.update(status="completed", elapsed_seconds=round(time.monotonic()-started, 2),
                  checkpoint_sha256=sha256(output/"best.pt"), trained_checkpoint_selected=report["best_epoch"] > 0,
                  peak_gpu_memory_mb=round(torch.cuda.max_memory_allocated()/1024**2, 1) if device.startswith("cuda") else None)
    write_json(output/"training.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True); parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2); parser.add_argument("--seed", type=int, default=142)
    parser.add_argument("--boundary-weight", type=float, default=.2); parser.add_argument("--topology-weight", type=float, default=.03)
    parser.add_argument("--allow-label-change", action="store_true"); parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        report = audit_manifest(args.manifest, Path(args.output)/"gt-connectivity-audit.json")
        print(json.dumps({k:report[k] for k in ("total_cases", "audited_cases", "review_flagged_cases")}))
    else:
        report = finetune(args.manifest, args.checkpoint, args.output, epochs=args.epochs, batch_size=args.batch_size,
                          seed=args.seed, boundary_weight=args.boundary_weight, topology_weight=args.topology_weight,
                          allow_label_change=args.allow_label_change)
        print(json.dumps({"status": report["status"], "best_epoch": report["best_epoch"], "checkpoint_sha256": report["checkpoint_sha256"]}))
