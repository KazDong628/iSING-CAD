import hashlib
import json
import numpy as np
from PIL import Image
from contour_agent import gt_dataset


def test_gt_labels_preserve_split_and_keep_missing_denominator(tmp_path,monkeypatch):
    from contour_agent import gt_registration
    source=tmp_path/'dataset';source.mkdir()
    base=tmp_path/'base';base.mkdir()
    rows=[]
    def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    for i,split in enumerate(('train','val','test')):
        image=source/f'case{i}.png';Image.new('RGB',(32,24),(200+i,210,220)).save(image)
        prepared=base/f'image{i}.png';prepared.write_bytes(image.read_bytes())
        mask=base/f'mask{i}.png';Image.fromarray(np.pad(np.ones((5,5),np.uint8)*255,((0,19),(0,27)))).save(mask)
        rows.append(dict(id=f'case{i}',image=str(prepared),mask=str(mask),source_image=str(image),
            source_image_sha256=digest(image),image_sha256=digest(image),prepared_image_sha256=digest(prepared),
            mask_sha256=digest(mask),group=f'g{i}',split=split,label_source='source_heuristic',reviewed=False,
            prepared_size={'width':32,'height':24},original_size={'width':32,'height':24},trainable=True))
    manifest=base/'manifest.json';manifest.write_text(json.dumps({'cases':rows}))
    def reference(root,cid):
        if cid=='case2':return dict(status='missing',issues=['missing'])
        return dict(status='ready',source={'path':str(source/'source.dxf')},source_sha256='a'*64,
                    polygon_xy=[[2,2],[20,2],[20,20],[2,20],[2,2]],units='mm',issues=[])
    monkeypatch.setattr(gt_dataset,'load_case_reference',reference)
    monkeypatch.setattr(gt_registration,'register_training_polygon',lambda *a,**k:dict(
        registered_polyline_px=reference(None,'case0')['polygon_xy'],transform_2x3=[[1,0,0],[0,1,0]],
        quality={k:1. for k in ('edge_support','frame_inside_ratio','hint_precision','hint_coverage','ambiguity_margin')},issues=[]))
    report=gt_dataset.prepare_gt_dataset(source,manifest,tmp_path/'out',workers=1)
    assert report['summary']['total']==3 and report['summary']['ready']==2
    assert [r['split'] for r in report['cases']]==['train','val','test']
    assert report['cases'][2]['mask'] is None and not report['cases'][2]['trainable']
    assert report['cases'][0]['label_source']=='registered_dxf_gt'
    assert report['cases'][0]['registration']['source_gt_sha256']=='a'*64
    assert np.asarray(Image.open(report['cases'][0]['mask']))[15,15]==255
    assert np.asarray(Image.open(rows[0]['mask']))[15,15]==0
    assert digest(source/'case0.png')==rows[0]['source_image_sha256']
