import copy
import json

import pytest
from PIL import Image

from scripts.run_source_obligation_replay import restore_frozen_radius_proposals


def case(tmp_path):
    image=tmp_path/'source.png';Image.new('RGB',(40,30),'white').save(image)
    source=tmp_path/'upstream';(source/'after').mkdir(parents=True)
    document={'meta':{},'records':[{'text':'R15','bbox':[2,2,9,9]}]}
    final=copy.deepcopy(document)
    final['records'][0]['source_arrow_proposals']=[{'record_id':'r000','tip_px':[10.,15.],'shaft_px':[8.,9.]}]
    path=source/'after/effective-radius-source-document.json'
    path.write_text(json.dumps(final),encoding='utf8')
    return image,source,document,final,path


def test_replay_restores_late_arrow_evidence_without_mutating_reviewed_ocr(tmp_path):
    image,source,document,final,path=case(tmp_path)
    document['records'][0]['source_arrow_proposals']=[{'record_id':'r000','tip_px':[5.,5.],'shaft_px':[3.,3.]}]
    before=copy.deepcopy(document);raw=path.read_bytes()
    restored,receipt=restore_frozen_radius_proposals(image,document,source)
    assert restored == final and document == before and path.read_bytes() == raw
    assert receipt['new_network_requests'] == 0
    assert receipt['fresh_source_pixel_verification_required'] is True


@pytest.mark.parametrize('change',['nominal','identity','design_coordinate','nan','count'])
def test_replay_rejects_semantic_change_or_non_pixel_hint(tmp_path,change):
    image,source,document,final,path=case(tmp_path)
    row=final['records'][0];proposal=row['source_arrow_proposals'][0]
    if change=='nominal':row['text']='R114'
    if change=='identity':proposal['record_id']='r001'
    if change=='design_coordinate':proposal['tip_px']=[-12.,15.]
    if change=='nan':proposal['tip_px'][0]=float('nan')
    if change=='count':row['source_arrow_proposals']*=3
    path.write_text(json.dumps(final),encoding='utf8')
    with pytest.raises(ValueError,match='frozen_radius'):
        restore_frozen_radius_proposals(image,document,source)
