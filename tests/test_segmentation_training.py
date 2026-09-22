"""CPU-only preprocessing, manifest provenance and checkpoint input checks."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

import contour_agent.segmentation as segmentation


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(root, case_id="example", split="train", marker=1):
    (root/"images").mkdir(parents=True,exist_ok=True)
    (root/"masks").mkdir(parents=True,exist_ok=True)
    image = np.full((40,80,3),255,np.uint8)
    image[10:30,20:60] = 0
    image[0,0] = marker
    mask = np.zeros((40,80),np.uint8)
    mask[10:30,20:60] = 255
    image_path, mask_path = root/"images"/(case_id+".png"),root/"masks"/(case_id+".png")
    Image.fromarray(image).save(image_path)
    Image.fromarray(mask).save(mask_path)
    source_hash = hashlib.sha256(("original-"+case_id).encode()).hexdigest()
    return {"id":case_id,"split":split,"group":"group-"+case_id,
            "image":str(image_path.relative_to(root)),"mask":str(mask_path.relative_to(root)),
            "source_image_sha256":source_hash,"image_sha256":source_hash,
            "prepared_image_sha256":_digest(image_path),"mask_sha256":_digest(mask_path),
            "label_source":"source_heuristic","reviewed":False,"trainable":True}


def _manifest(root, rows, wrapper="cases"):
    path = root/"manifest.json"
    value = rows if wrapper is None else {wrapper:rows}
    path.write_text(json.dumps(value),encoding="utf-8")
    return path


def _review(root, row):
    directory = root/"reviewed_masks"
    directory.mkdir(exist_ok=True)
    mask = np.zeros((40,80),np.uint8)
    mask[9:31,18:62] = 255
    path = directory/(row["id"]+".png")
    Image.fromarray(mask).save(path)
    record = {"reviewed":True,"label_source":"local_user_review",
              "source_image_sha256":row["source_image_sha256"],
              "mask":f"reviewed_masks/{row['id']}.png","mask_sha256":_digest(path)}
    (root/"reviews.json").write_text(json.dumps({row["id"]:record}),encoding="utf-8")
    return record


@pytest.mark.parametrize("wrapper", [None,"cases","rows","samples"])
def test_manifest_reads_supported_wrappers_and_hashes_consumed_artifacts(tmp_path,wrapper):
    row = _row(tmp_path)
    path = _manifest(tmp_path,[row],wrapper)
    prepared,provenance = segmentation.read_manifest(path)
    assert len(prepared) == 1
    assert Path(prepared[0]["image"]).is_absolute()
    assert prepared[0]["frozen_mask_sha256"] == prepared[0]["mask_sha256"] == row["mask_sha256"]
    assert prepared[0]["reviewed"] is False
    assert provenance["manifest_sha256"] == _digest(path)
    assert provenance["total_source_cases"] == provenance["usable_cases"] == 1
    assert provenance["reviews_sha256"] is None
    assert len(provenance["consumed_artifacts_sha256"]) == 64
    assert provenance["consumed_artifacts"][0]["label_source"] == "source_heuristic"


@pytest.mark.parametrize("field", ["image","mask"])
def test_changed_prepared_artifacts_are_rejected(tmp_path,field):
    row = _row(tmp_path)
    path = _manifest(tmp_path,[row])
    (tmp_path/row[field]).write_bytes(b"changed frozen artifact")
    with pytest.raises(ValueError,match="content hash changed"):
        segmentation.read_manifest(path)


@pytest.mark.parametrize("field", ["prepared_image_sha256","mask_sha256","group","split"])
def test_missing_required_provenance_fields_are_rejected(tmp_path,field):
    row = _row(tmp_path)
    row.pop(field)
    with pytest.raises(ValueError):
        segmentation.read_manifest(_manifest(tmp_path,[row]))


def test_missing_source_hash_is_rejected(tmp_path):
    row = _row(tmp_path)
    row.pop("source_image_sha256");row.pop("image_sha256")
    with pytest.raises(ValueError,match="source_image_sha256"):
        segmentation.read_manifest(_manifest(tmp_path,[row]))


@pytest.mark.parametrize("case_id", ["../outside","a/b","a\\b","/absolute",""])
def test_unsafe_ids_cannot_be_used_as_evaluation_output_paths(tmp_path,case_id):
    row = _row(tmp_path)
    row["id"] = case_id
    with pytest.raises(ValueError,match="unsafe"):
        segmentation.read_manifest(_manifest(tmp_path,[row]))


def test_duplicate_ids_are_case_insensitively_rejected(tmp_path):
    first,second = _row(tmp_path,"first"),_row(tmp_path,"second",marker=2)
    second["id"] = "FIRST"
    with pytest.raises(ValueError,match="duplicate"):
        segmentation.read_manifest(_manifest(tmp_path,[first,second]))


def test_prepared_path_cannot_escape_data_root(tmp_path):
    root = tmp_path/"data";root.mkdir()
    row = _row(root)
    outside = tmp_path/"outside.png";outside.write_bytes(b"outside")
    row.update(image="../outside.png",prepared_image_sha256=_digest(outside))
    with pytest.raises(ValueError,match="escapes"):
        segmentation.read_manifest(_manifest(root,[row]))


@pytest.mark.parametrize("identity", ["group","source_image_sha256","prepared_image_sha256"])
def test_cross_split_identity_leakage_is_rejected(tmp_path,identity):
    first,second = _row(tmp_path,"first","train"),_row(tmp_path,"second","val",marker=2)
    second[identity] = first[identity]
    if identity == "source_image_sha256":
        second["image_sha256"] = first["image_sha256"]
    if identity == "prepared_image_sha256":
        (tmp_path/second["image"]).write_bytes((tmp_path/first["image"]).read_bytes())
    with pytest.raises(ValueError,match="Cross-split leakage"):
        segmentation.read_manifest(_manifest(tmp_path,[first,second]))


def test_untrainable_case_stays_in_denominator_but_is_not_consumed(tmp_path):
    first,second = _row(tmp_path,"first"),_row(tmp_path,"second",marker=2)
    second["trainable"] = False
    prepared,provenance = segmentation.read_manifest(_manifest(tmp_path,[first,second]))
    assert [row["id"] for row in prepared] == ["first"]
    assert provenance["total_source_cases"] == 2
    assert provenance["usable_cases"] == 1


def test_string_false_cannot_silently_enable_training(tmp_path):
    row = _row(tmp_path);row["trainable"] = "false"
    with pytest.raises(ValueError,match="nonboolean trainable"):
        segmentation.read_manifest(_manifest(tmp_path,[row]))


def test_review_override_preserves_frozen_and_effective_mask_provenance(tmp_path):
    row = _row(tmp_path)
    record = _review(tmp_path,row)
    prepared,provenance = segmentation.read_manifest(_manifest(tmp_path,[row]))
    actual = prepared[0]
    assert actual["reviewed"] is True and actual["label_source"] == "local_user_review"
    assert actual["mask_sha256"] == record["mask_sha256"]
    assert actual["frozen_mask_sha256"] == row["mask_sha256"]
    assert actual["frozen_label_source"] == "source_heuristic"
    assert actual["mask"] != actual["frozen_mask"]
    assert provenance["reviews_sha256"] == _digest(tmp_path/"reviews.json")
    assert provenance["consumed_artifacts"][0]["mask_sha256"] == record["mask_sha256"]


@pytest.mark.parametrize("fault", ["stale_source","changed_mask","wrong_provenance","other_case_mask"])
def test_invalid_review_never_silently_becomes_reviewed_or_weak(tmp_path,fault):
    row = _row(tmp_path)
    record = _review(tmp_path,row)
    if fault == "stale_source": record["source_image_sha256"] = "0"*64
    elif fault == "changed_mask": (tmp_path/record["mask"]).write_bytes(b"modified review")
    elif fault == "wrong_provenance": record["label_source"] = "automatic_prediction"
    else: record["mask"] = "reviewed_masks/another.png"
    (tmp_path/"reviews.json").write_text(json.dumps({row["id"]:record}),encoding="utf-8")
    with pytest.raises(ValueError):
        segmentation.read_manifest(_manifest(tmp_path,[row]))


def test_manifest_reviewed_flag_alone_does_not_claim_human_review(tmp_path):
    row = _row(tmp_path);row["reviewed"] = True
    prepared,_ = segmentation.read_manifest(_manifest(tmp_path,[row]))
    assert prepared[0]["reviewed"] is False


def test_letterbox_mask_alignment_padding_and_inverse_crop():
    image = np.zeros((7,13,3),np.uint8)
    mask = np.ones((7,13),bool)
    boxed,target,transform = segmentation.letterbox(image,32,mask)
    assert boxed.shape == (32,32,3) and target.shape == (32,32)
    assert (transform["width"],transform["height"]) == (32,17)
    assert set(np.unique(target)) == {0,1}
    assert np.array_equal(boxed[:,:,0] == 0,target == 1)
    probability = np.ones((32,32),np.float32)
    probability[transform["top"]:transform["top"]+transform["height"],:] = .3
    restored = segmentation.unletterbox(probability,transform)
    assert restored.shape == (7,13)
    assert np.allclose(restored,.3)


def test_tensor_image_uses_rgb_imagenet_normalization_on_cpu():
    torch = pytest.importorskip("torch")
    image = np.array([[[255,0,128]]],np.uint8)
    tensor = segmentation.tensor_image(image)
    expected = (image.astype(np.float32)/255-np.array([.485,.456,.406]))/np.array([.229,.224,.225])
    assert tensor.device.type == "cpu" and tensor.dtype == torch.float32
    assert tensor.shape == (3,1,1) and tensor.is_contiguous()
    assert np.allclose(tensor.numpy()[:,0,0],expected[0,0],atol=1e-6)


def test_dataset_transforms_apply_identically_without_changing_cached_pair(tmp_path,monkeypatch):
    pytest.importorskip("torch")
    import contour_agent.segmentation_metrics as metrics
    row = _row(tmp_path)
    prepared,_ = segmentation.read_manifest(_manifest(tmp_path,[row]))
    monkeypatch.setattr(metrics,"add_annotation_noise",lambda image,*args,**kwargs:image)
    monkeypatch.setattr(segmentation,"tensor_image",lambda image:image)
    dataset = segmentation.DrawingDataset(prepared,128,augment=True,seed=7)
    original_image,original_mask = (array.copy() for array in dataset.samples[0])
    image,target = dataset[0]
    binary = target.numpy()[0] > .5
    dark = image[:,:,0] < 127
    assert np.count_nonzero(binary & dark)/np.count_nonzero(binary | dark) > .96
    assert np.array_equal(dataset.samples[0][0],original_image)
    assert np.array_equal(dataset.samples[0][1],original_mask)


def test_segmenter_checkpoint_is_strict_and_predict_array_stays_image_only_cpu(tmp_path,monkeypatch):
    torch = pytest.importorskip("torch")
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.bias = torch.nn.Parameter(torch.tensor(0.))
        def forward(self,value):
            return torch.zeros_like(value[:,:1])+self.bias
    checkpoint = tmp_path/"tiny.pt"
    torch.save({"model":segmentation.MODEL_SPEC,"state_dict":Tiny().state_dict(),"size":128,
                "label_status":"weak_supervision_experimental"},checkpoint)
    monkeypatch.setattr(segmentation,"make_model",Tiny)
    monkeypatch.setattr(torch.cuda,"is_available",lambda:pytest.fail("CPU test queried CUDA"))
    segmenter = segmentation.Segmenter(checkpoint,device="cpu")
    prediction = segmenter.predict_array(np.full((31,73,3),255,np.uint8))
    assert prediction.shape == (31,73)
    assert np.allclose(prediction,.5)
    assert segmenter.metadata["checkpoint_sha256"] == _digest(checkpoint)
    assert segmenter.metadata["device"] == "cpu"
    assert segmenter.metadata["training_provenance"] is None
    bad = {"model":{**segmentation.MODEL_SPEC,"classes":2},"state_dict":{},"size":128}
    torch.save(bad,checkpoint)
    with pytest.raises(ValueError,match="Unsupported"):
        segmentation.Segmenter(checkpoint,device="cpu")


def _tiny_checkpoint(tmp_path,monkeypatch,*,size=128,provenance=None):
    torch = pytest.importorskip("torch")
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.bias = torch.nn.Parameter(torch.tensor(-1.))
        def forward(self,value):
            return torch.zeros_like(value[:,:1])+self.bias
    checkpoint = tmp_path/"test-checkpoint.pt"
    state = {"model":segmentation.MODEL_SPEC,"size":size,"state_dict":Tiny().state_dict(),
             "label_status":"weak_supervision_experimental"}
    if provenance is not None: state["provenance"] = provenance
    torch.save(state,checkpoint)
    monkeypatch.setattr(segmentation,"make_model",Tiny)
    # Default evaluation inference uses CPU without querying the real GPU.
    monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
    return checkpoint


@pytest.mark.parametrize("size", [True,False,None,0,-128,127,129,1056,128.0,128.5,"128"])
def test_checkpoint_rejects_invalid_size_before_model_allocation(tmp_path,monkeypatch,size):
    checkpoint = _tiny_checkpoint(tmp_path,monkeypatch,size=size)
    monkeypatch.setattr(segmentation,"make_model",lambda:pytest.fail("Invalid size allocated a model"))
    with pytest.raises(ValueError,match="Checkpoint size"):
        segmentation.Segmenter(checkpoint,device="cpu")


@pytest.mark.parametrize("size", [128,512,1024])
def test_checkpoint_valid_sizes_and_training_provenance_are_preserved(tmp_path,monkeypatch,size):
    provenance = {"manifest_sha256":"1"*64,"reviews_sha256":None,"split_claim":"development_split_not_blind_test"}
    checkpoint = _tiny_checkpoint(tmp_path,monkeypatch,size=size,provenance=provenance)
    segmenter = segmentation.Segmenter(checkpoint,device="cpu")
    assert segmenter.size == size
    assert segmenter.metadata["training_provenance"] == provenance


@pytest.mark.parametrize("provenance", [None,{}, {"manifest_sha256":"invalid"},{"manifest_sha256":"0"*64}])
def test_evaluation_rejects_unknown_or_different_training_manifest(tmp_path,monkeypatch,provenance):
    row = _row(tmp_path,split="test")
    manifest = _manifest(tmp_path,[row])
    checkpoint = _tiny_checkpoint(tmp_path,monkeypatch,provenance=provenance)
    monkeypatch.setattr(segmentation,"make_model",lambda:pytest.fail("Invalid provenance allocated a model"))
    output = tmp_path/"evaluation"
    with pytest.raises(ValueError,match="manifest"):
        segmentation.evaluate(manifest,checkpoint,output)
    assert not output.exists()


def test_evaluation_matches_current_frozen_manifest_and_effective_labels(tmp_path,monkeypatch):
    row = _row(tmp_path,split="test")
    manifest = _manifest(tmp_path,[row])
    _,provenance = segmentation.read_manifest(manifest)
    checkpoint = _tiny_checkpoint(tmp_path,monkeypatch,provenance=provenance)
    report = segmentation.evaluate(manifest,checkpoint,tmp_path/"evaluation")
    assert report["checkpoint_manifest_match"] is True
    assert report["reviews_match"] is True
    assert report["label_provenance"]["consumed_artifacts_match"] is True
    assert report["model"]["training_provenance"] == provenance
    assert report["model"]["device"] == "cpu"
    assert report["true_accuracy_verified"] is False


def test_new_reviews_are_allowed_and_train_evaluation_label_difference_is_recorded(tmp_path,monkeypatch):
    row = _row(tmp_path,split="test")
    manifest = _manifest(tmp_path,[row])
    _,training_provenance = segmentation.read_manifest(manifest)
    checkpoint = _tiny_checkpoint(tmp_path,monkeypatch,provenance=training_provenance)
    _review(tmp_path,row)
    report = segmentation.evaluate(manifest,checkpoint,tmp_path/"evaluation")
    assert report["checkpoint_manifest_match"] is True
    assert report["reviews_match"] is False
    assert report["label_provenance"]["training_reviews_sha256"] is None
    assert report["label_provenance"]["evaluation_reviews_sha256"] == _digest(tmp_path/"reviews.json")
    assert report["label_provenance"]["consumed_artifacts_match"] is False
    assert report["cases"][0]["reviewed"] is True
    assert report["model"]["training_provenance"]["reviews_sha256"] is None


def test_missing_historical_review_and_artifact_hashes_are_unknown_not_equal(tmp_path,monkeypatch):
    row = _row(tmp_path,split="test")
    manifest = _manifest(tmp_path,[row])
    provenance = {"manifest_sha256":_digest(manifest)}
    checkpoint = _tiny_checkpoint(tmp_path,monkeypatch,provenance=provenance)
    report = segmentation.evaluate(manifest,checkpoint,tmp_path/"evaluation")
    assert report["checkpoint_manifest_match"] is True
    assert report["reviews_match"] is None
    assert report["label_provenance"]["consumed_artifacts_match"] is None
