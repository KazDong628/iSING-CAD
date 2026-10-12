import json
from pathlib import Path

import pytest

from contour_agent.parametric_pipeline import _write


def test_stage_atomic_temporary_does_not_extend_long_artifact_name(tmp_path,monkeypatch):
    destination=tmp_path/'parametric-solution.json'
    original=Path.write_text
    written=[]
    def bounded_write(path,*args,**kwargs):
        # Reproduce the Windows failure where the final path fits but appending
        # .tmp overflows it. The sibling temporary must be strictly shorter.
        assert path.parent==destination.parent
        if len(path.name)>=len(destination.name):
            raise FileNotFoundError('temporary_path_exceeds_final_path_budget')
        written.append(path)
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'write_text',bounded_write)
    _write(destination,{'status':'accepted','value':12.})
    assert json.loads(destination.read_text('utf8'))['value']==12.
    assert len(written)==1 and not written[0].exists()


def test_failed_serialization_preserves_previous_stage_receipt(tmp_path):
    destination=tmp_path/'stage.json'
    _write(destination,{'status':'previous_valid_subset'})
    before=destination.read_bytes()
    with pytest.raises(ValueError):_write(destination,{'radius':float('nan')})
    assert destination.read_bytes()==before
