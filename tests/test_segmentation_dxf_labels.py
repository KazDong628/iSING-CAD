"""Derived DXF supervision provenance and ignore/comparison checks; CPU only."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from contour_agent import segmentation


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row(root, case_id="a", split="train", *, gt=True):
    root.mkdir(parents=True, exist_ok=True)
    image = np.full((40,80,3),255,np.uint8)
    image[10:30,20:60] = 0
    image[0,0] = sum(case_id.encode()) % 255
    mask = np.zeros((40,80),np.uint8); mask[10:30,20:60] = 255
    image_path, mask_path = root/f"{case_id}-image.png", root/f"{case_id}-mask.png"
    Image.fromarray(image).save(image_path); Image.fromarray(mask).save(mask_path)
    result = {"id":case_id,"split":split,"group":"group-"+case_id,
              "image":image_path.name,"mask":mask_path.name,
              "source_image_sha256":hashlib.sha256(("original-"+case_id).encode()).hexdigest(),
              "prepared_image_sha256":digest(image_path),"mask_sha256":digest(mask_path),
              "label_source":"registered_dxf_gt" if gt else "source_heuristic","trainable":True}
    if gt:
        result["registration"] = {"status":"accepted","source_gt_sha256":"d"*64,
                                  "transform":[[2,0,10],[0,-2,30],[0,0,1]],
                                  "quality":{"region_overlap":.85,"boundary_distance_px":2.1}}
    return result


def manifest(root, rows):
    path = root/"manifest.json"
    path.write_text(json.dumps({"cases":rows}),encoding="utf8")
    return path


def add_ignore(root, item, value=None):
    ignored = np.zeros((40,80),np.uint8)
    ignored[:, :10] = 255
    if value is not None: ignored[:] = value
    path = root/f"{item['id']}-ignore.png"
    Image.fromarray(ignored).save(path)
    item.update(ignore_mask=path.name, ignore_mask_sha256=digest(path))
    return path


def test_registered_dxf_provenance_is_derived_supervision_not_human_review(tmp_path):
    item = row(tmp_path); item["reviewed"] = True
    # This provenance-only reference is deliberately unreadable: the loader
    # consumes prepared masks, never source DXF coordinates at train/inference.
    item["registration"]["source_gt_path"] = "missing-reference-not-opened.dxf"
    prepared, provenance = segmentation.read_manifest(manifest(tmp_path,[item]))
    assert prepared[0]["label_source"] == "registered_dxf_gt"
    assert prepared[0]["reviewed"] is False
    assert provenance["label_status"] == "registered_dxf_gt_experimental"
    assert provenance["human_reviewed_cases"] == 0
    recorded = provenance["consumed_artifacts"][0]["registration"]
    assert recorded["source_gt_sha256"] == "d"*64
    assert recorded["transform"] == item["registration"]["transform"]
    assert recorded["quality"] == item["registration"]["quality"]
    assert "not human" in recorded["acceptance_meaning"]


@pytest.mark.parametrize("fault", ["missing", "unaccepted", "bad_hash", "missing_transform", "nan", "singular", "projective", "missing_quality"])
def test_trainable_dxf_requires_complete_finite_registration_provenance(tmp_path,fault):
    item = row(tmp_path)
    registration = item["registration"]
    if fault == "missing": item.pop("registration")
    elif fault == "unaccepted": registration["status"] = "needs_review"
    elif fault == "bad_hash": registration["source_gt_sha256"] = "not-a-hash"
    elif fault == "missing_transform": registration.pop("transform")
    elif fault == "nan": registration["transform"][0][0] = float("nan")
    elif fault == "singular": registration["transform"][0][0] = 0
    elif fault == "projective": registration["transform"][2] = [1,0,1]
    else: registration["quality"] = {}
    with pytest.raises(ValueError):
        segmentation.read_manifest(manifest(tmp_path,[item]))


def test_missing_registration_masks_remain_in_identity_and_coverage_denominators(tmp_path):
    first, second = row(tmp_path,"a"), row(tmp_path,"b",split="val")
    second.update(mask=None, mask_sha256=None, trainable=False, registration={"status":"failed"})
    prepared, provenance = segmentation.read_manifest(manifest(tmp_path,[first,second]))
    assert len(prepared) == 1
    assert provenance["total_source_cases"] == 2 and provenance["usable_cases"] == 1
    assert len(provenance["case_identities"]) == 2
    assert provenance["registration_status_counts"] == {"accepted":1,"failed":1}


@pytest.mark.parametrize("field", ["source_gt_sha256", "quality"])
def test_gt_source_and_registration_quality_affect_consumed_provenance_hash(tmp_path,field):
    item = row(tmp_path)
    _, before = segmentation.read_manifest(manifest(tmp_path,[item]))
    item["registration"][field] = "e"*64 if field == "source_gt_sha256" else {"region_overlap":.8}
    _, after = segmentation.read_manifest(manifest(tmp_path,[item]))
    assert before["consumed_artifacts_sha256"] != after["consumed_artifacts_sha256"]


def test_mixed_labels_are_reported_without_relabeling_heuristic_as_gt(tmp_path):
    first, second = row(tmp_path,"a"), row(tmp_path,"b",gt=False)
    _, provenance = segmentation.read_manifest(manifest(tmp_path,[first,second]))
    assert provenance["label_status"] == "mixed_supervision_experimental"
    assert provenance["label_source_counts"] == {"registered_dxf_gt":1,"source_heuristic":1}


@pytest.mark.parametrize("fault", ["changed", "gray", "all_ignored", "wrong_size"])
def test_ignore_mask_hash_binary_values_and_dimensions_are_validated(tmp_path,fault):
    item = row(tmp_path)
    path = add_ignore(tmp_path,item,127 if fault == "gray" else 255 if fault == "all_ignored" else None)
    if fault == "wrong_size":
        Image.new("L",(41,80),0).save(path); item["ignore_mask_sha256"] = digest(path)
    if fault == "changed": path.write_bytes(b"changed ignore mask")
    with pytest.raises(ValueError):
        segmentation.read_manifest(manifest(tmp_path,[item]))


def test_ignore_mask_preprocessing_preserves_pixel_alignment_and_metadata(tmp_path):
    pytest.importorskip("torch")
    item = row(tmp_path); add_ignore(tmp_path,item)
    prepared, provenance = segmentation.read_manifest(manifest(tmp_path,[item]))
    assert provenance["ignore_mask_cases"] == 1
    assert prepared[0]["ignore_pixels"] == 400
    dataset = segmentation.DrawingDataset(prepared,128)
    image, target = dataset[0]
    assert image.device.type == "cpu" and target.device.type == "cpu"
    ignored = target.numpy()[0] < 0
    assert ignored.sum() == 64*16
    assert ignored[32:96,:16].all()
    assert not ignored[:32].any()


def test_ignored_pixels_have_zero_loss_gradient_and_do_not_lower_region_iou():
    torch = pytest.importorskip("torch")
    target = torch.tensor([[[[1.,-1.],[0.,-1.]]]])
    logits = torch.tensor([[[[10.,-3.],[-10.,3.]]]],requires_grad=True)
    loss = segmentation.binary_loss(logits,target)
    changed = logits.detach().clone(); changed[0,0,0,1]=200; changed[0,0,1,1]=-200
    assert segmentation.binary_loss(changed,target).item() == pytest.approx(loss.item())
    loss.backward()
    assert torch.equal(logits.grad[target<0],torch.zeros(2))

    class Prediction(torch.nn.Module):
        def forward(self,images): return logits.detach()

    score = segmentation.evaluate_loader(Prediction(),[(torch.zeros(1,3,2,2),target)],"cpu")
    assert score == pytest.approx(1.)


def test_no_ignore_binary_loss_matches_existing_bce_plus_dice():
    torch = pytest.importorskip("torch")
    logits=torch.tensor([[[[.2,-.4],[.7,-.1]]]])
    target=torch.tensor([[[[1.,0.],[1.,0.]]]])
    probability=logits.sigmoid()
    expected=torch.nn.functional.binary_cross_entropy_with_logits(logits,target)+1-(2*(probability*target).sum()+1)/(probability.sum()+target.sum()+1)
    assert segmentation.binary_loss(logits,target).item() == pytest.approx(expected.item())


@pytest.fixture
def label_comparison(tmp_path,monkeypatch):
    old, new = tmp_path/"old",tmp_path/"new"
    old_rows=[row(old,"a",split="test",gt=False),row(old,"b",split="test",gt=False)]
    new_rows=[row(new,"a",split="test"),row(new,"b",split="test")]
    new_rows[1].update(mask=None,mask_sha256=None,trainable=False,registration={"status":"failed"})
    old_manifest=manifest(old,old_rows); new_manifest=manifest(new,new_rows)
    # Mirrors the older v1 checkpoint: manifest digest but no per-case records.
    provenance={"manifest_sha256":digest(old_manifest)}
    allocations=[]

    class FakeSegmenter:
        def __init__(self,checkpoint,**kwargs):
            expected=kwargs.get("expected_manifest_sha256")
            if expected is not None and expected != provenance["manifest_sha256"]:
                raise ValueError("Checkpoint training manifest does not match evaluation manifest")
            allocations.append(expected)
            self.metadata={"training_provenance":provenance,"label_status":"weak_supervision_experimental"}
        def predict_array(self,image): return np.zeros(image.shape[:2],np.float32)

    monkeypatch.setattr(segmentation,"Segmenter",FakeSegmenter)
    return old_manifest,new_manifest,new_rows,provenance,allocations


def test_old_checkpoint_compares_against_new_gt_only_with_verified_original_manifest(tmp_path,label_comparison):
    old,new,_,provenance,_=label_comparison
    with pytest.raises(ValueError,match="manifest"):
        segmentation.evaluate(new,"stub.pt",tmp_path/"not-permitted")
    report=segmentation.evaluate(new,"stub.pt",tmp_path/"comparison",comparison_manifest=old)
    assert report["checkpoint_manifest_match"] is False
    assert report["comparison_manifest_sha256"] == provenance["manifest_sha256"]
    assert report["comparison_identity"]["verified"]
    assert report["comparison_identity"]["case_count"] == 2
    assert report["count"] == 1 and report["declared_split_count"] == 2
    assert report["total"] == 2 and report["excluded"] == 1
    assert report["excluded_case_ids"] == ["b"]
    assert report["excluded_cases"][0]["reasons"] == ["declared_untrainable","missing_label_mask"]
    assert report["excluded_cases"][0]["registration_status"] == "failed"
    assert report["unavailable_label_case_ids"] == ["b"]
    assert report["cases"][0]["label_source"] == "registered_dxf_gt"
    assert report["cases"][0]["reviewed"] is False
    assert report["true_accuracy_verified"] is False
    assert report["label_provenance"]["consumed_artifacts_match"] is None


def test_unknown_old_case_provenance_cannot_be_silently_allowed(tmp_path,label_comparison):
    _,new,_,_,_=label_comparison
    with pytest.raises(ValueError,match="source IDs/hashes/splits"):
        segmentation.evaluate(new,"stub.pt",tmp_path/"output",allow_label_change=True)
    assert not (tmp_path/"output").exists()


@pytest.mark.parametrize("fault", ["source_hash", "split", "group", "id", "original_manifest"])
def test_label_comparison_rejects_changed_source_identity_and_wrong_baseline(tmp_path,label_comparison,fault):
    old,new,rows,_,allocations=label_comparison
    if fault == "source_hash": rows[0]["source_image_sha256"]="1"*64
    elif fault == "split": rows[1]["split"]="val"
    elif fault == "group": rows[0]["group"]="changed-family"
    elif fault == "id": rows[0]["id"]="another-drawing"
    else:
        data=json.loads(old.read_text());data["changed"]=True;old.write_text(json.dumps(data))
    manifest(new.parent,rows)
    with pytest.raises(ValueError):
        segmentation.evaluate(new,"stub.pt",tmp_path/"output",comparison_manifest=old)
    assert not (tmp_path/"output").exists()
    assert not allocations


def test_ignore_region_is_not_used_to_hide_full_boundary_error(tmp_path,label_comparison):
    old,new,rows,_,_=label_comparison
    ignored=np.zeros((40,80),np.uint8);ignored[10:30,20:60]=255
    path=new.parent/"ignored-material.png";Image.fromarray(ignored).save(path)
    rows[0].update(ignore_mask=path.name,ignore_mask_sha256=digest(path))
    manifest(new.parent,rows)
    report=segmentation.evaluate(new,"stub.pt",tmp_path/"evaluation",comparison_manifest=old)
    result=report["cases"][0]
    assert result["clean"]["iou"] == 0
    assert result["clean"]["boundary_f1"] == 0
    assert result["clean_valid_region"]["iou"] == 1
    assert result["clean_valid_region"]["ignored_pixels"] > 0
    assert report["mean_clean_iou"] == 0 and report["mean_clean_boundary_f1"] == 0


def test_all_failed_labels_are_reported_without_successful_empty_average(tmp_path,label_comparison):
    old,new,rows,_,_=label_comparison
    rows[0].update(trainable=False,mask=None,mask_sha256=None,
        registration={"status":"rejected","failed_quality_checks":["ambiguity_margin"]},
        issues=["Registration has competing placements"])
    manifest(new.parent,rows)
    report=segmentation.evaluate(new,"stub.pt",tmp_path/"empty-evaluation",comparison_manifest=old)
    assert report["count"] == 0 and report["total"] == report["excluded"] == 2
    assert report["status"] == "no_usable_labels"
    assert report["mean_clean_iou"] is None and report["mean_clean_boundary_f1"] is None
    assert report["excluded_cases"][0]["failed_quality_checks"] == ["ambiguity_margin"]
    assert report["excluded_cases"][0]["issues"] == ["Registration has competing placements"]
    assert report["cases"] == [] and report["blind_test"] is False
    json.dumps(report,allow_nan=False)


def test_native_inputs_preserve_detail_with_fixed_binary_scoring_grid(tmp_path,label_comparison,monkeypatch):
    import cv2
    old,new,rows,provenance,_=label_comparison
    height,width=640,1024
    source=np.repeat(((np.indices((height,width)).sum(axis=0)%2)*255).astype(np.uint8)[...,None],3,axis=2)
    target=np.zeros((height,width),np.uint8);target[111:527,223:823]=255
    image_path=new.parent/"native.png";mask_path=new.parent/"native-mask.png"
    Image.fromarray(source).save(image_path);Image.fromarray(target).save(mask_path)
    rows[0].update(image=image_path.name,prepared_image_sha256=digest(image_path),mask=mask_path.name,mask_sha256=digest(mask_path))
    manifest(new.parent,rows)
    calls=[];model_size=[512]
    class NativeSegmenter:
        def __init__(self,*args,**kwargs):
            self.metadata={"training_provenance":provenance,"size":model_size[0]}
        def predict_array(self,image):
            calls.append(image.copy())
            assert image.shape == source.shape
            return target.astype(np.float32)/255
    monkeypatch.setattr(segmentation,"Segmenter",NativeSegmenter)
    first=segmentation.evaluate(new,"stub.pt",tmp_path/"model512",comparison_manifest=old)
    model_size[0]=768
    second=segmentation.evaluate(new,"stub.pt",tmp_path/"model768",comparison_manifest=old)
    assert len(calls) == 4
    assert np.array_equal(calls[0],source) and np.array_equal(calls[2],source)
    assert np.array_equal(calls[1],calls[3]) and np.any(calls[1] != source)
    assert np.all(calls[1] <= source)
    # A 512 pre-downsample/upscale would destroy this one-pixel checkerboard.
    degraded=cv2.resize(cv2.resize(source,(512,320),interpolation=cv2.INTER_AREA),(width,height))
    assert not np.array_equal(calls[0],degraded)
    assert first["mean_clean_iou"] == second["mean_clean_iou"] == 1
    assert first["cases"][0]["inference_input_size"] == {"width":1024,"height":640}
    assert first["scoring_grid"]["size"] == second["scoring_grid"]["size"] == 512
    assert "nearest-neighbor" in second["scoring_grid"]["prediction"]
