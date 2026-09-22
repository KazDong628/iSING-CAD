import copy
import hashlib
import json
from pathlib import Path
import zipfile

import ezdxf
import pytest

from contour_agent.config import Settings
from tools import compare_pilot_dxf as tool
from tools.run_segmentation_pilot import PILOT_CASE_IDS


def _rectangle(path, unit=4):
    path.parent.mkdir(parents=True, exist_ok=True)
    document = ezdxf.new(); document.units = unit
    points = [(0,0),(2,0),(2,1),(0,1)]
    for start, end in zip(points, points[1:]+points[:1]): document.modelspace().add_line(start,end)
    document.saveas(path)


@pytest.fixture
def pilot_inputs(tmp_path, monkeypatch):
    settings = Settings(dataset_root=tmp_path/"dataset", runtime_root=tmp_path/"runtime", api_key="")
    settings.dataset_root.mkdir()
    ids = [*PILOT_CASE_IDS,*[f"other-{index}" for index in range(46)]]
    catalog = {"cases":[{"id":case_id,"gt_dxf":f"{case_id}.dxf"} for case_id in ids]}
    for index,case_id in enumerate(PILOT_CASE_IDS):
        reference=settings.dataset_root/f"{case_id}.dxf"
        _rectangle(reference,unit=0 if index==1 else 4)
        if index==1:
            document=ezdxf.readfile(reference)
            document.modelspace().delete_entity(list(document.modelspace())[-1])
            document.saveas(reference)
    monkeypatch.setattr(tool,"build_catalog",lambda root:catalog)
    paths = []
    for number in (1,2):
        run_id = f"20260920T13000000000{number}Z-12345678"
        folder = settings.runtime_root/"segmentation"/"pilot"/run_id
        rows = []
        for index,case_id in enumerate(PILOT_CASE_IDS):
            prediction = folder/"cases"/case_id/"drawing.dxf"
            _rectangle(prediction,unit=0 if index in (1,3) else 4)
            rows.append({"case_id":case_id,"job_id":f"{number*10+index:032x}",
                         "artifacts":{"drawing.dxf":{"packaged_sha256":tool._sha(prediction)}},
                         "reference":{"reference_sha256":tool._sha(settings.dataset_root/f"{case_id}.dxf")}})
        path=folder/"pilot_summary.json"
        path.write_text(json.dumps({"run_id":run_id,"status":"completed","cases":rows}),encoding="utf8")
        paths.append(path)
    pointer=settings.runtime_root/"segmentation"/"pilot"/"latest.json"
    pointer.write_text(json.dumps({"report":f"{paths[0].parent.name}/pilot_summary.json","report_sha256":tool._sha(paths[0])}))
    return settings,paths


def test_compare_existing_runs_packages_parameters_and_preserves_missing_units(pilot_inputs):
    settings, paths=pilot_inputs
    report,path=tool.compare_pilot_runs(settings,candidate_summary=paths[1])
    assert report["dataset_total"]==50 and report["selected_count"]==4 and report["not_selected"]==46
    assert len(report["runs"])==2 and len(report["changes"])==4
    assert not report["blind_test"] and not report["engineering_verified"]
    rows=report["runs"][0]["cases"]
    assert rows[0]["reference_hash_matches_pilot"]
    assert tool._summary_row("baseline",rows[1])["unique_parameter_agreements"] is None
    assert report["changes"][1]["max_error_change_mm"] is None
    assert report["changes"][1]["max_error_improved"] is None
    assert report["changes"][0]["filtered_entity_reduction_percent"]==0
    page=(path.parent/"index.html").read_text(encoding="utf8")
    assert "data:image/png;base64," in page
    assert "最大误差定位与 core / closure" in page
    assert "不能替代" in page
    assert "reference_accuracy_improved" not in page
    assert "参考 DXF 的 INSUNITS=0" in page
    assert "参考主轮廓端点审计未闭合" in page and "原绘图单位（非毫米）" in page
    with zipfile.ZipFile(path.parent/"comparison-artifacts.zip") as package:
        assert package.testzip() is None
        assert not any(name.endswith(".dxf") or "reference-only" in name for name in package.namelist())
        assert sum(name.endswith("parameters.csv") for name in package.namelist())==8
        assert sum(name.endswith("connections.csv") for name in package.namelist())==8
        parameters=package.read("cases/baseline/CL60-main/parameters.csv").decode("utf-8-sig")
        assert "raw_modelspace" in parameters and "filtered_profile" in parameters
    latest=json.loads((path.parent.parent/"latest.json").read_text())
    assert latest["report_sha256"]==tool._sha(path)


def test_tampered_prediction_cannot_be_compared(pilot_inputs):
    settings,paths=pilot_inputs
    prediction=paths[0].parent/"cases"/PILOT_CASE_IDS[0]/"drawing.dxf"
    prediction.write_bytes(prediction.read_bytes()+b"tampered")
    with pytest.raises(ValueError,match="artifact hash"):
        tool.compare_pilot_runs(settings)
    assert not (settings.runtime_root/"dxf-comparison"/"latest.json").exists()


def test_latest_summary_path_cannot_escape_pilot_directory(pilot_inputs,tmp_path):
    settings,paths=pilot_inputs
    outside=tmp_path/"unrelated.json";outside.write_text("{}")
    pointer=settings.runtime_root/"segmentation"/"pilot"/"latest.json"
    pointer.write_text(json.dumps({"report":str(outside),"report_sha256":tool._sha(outside)}))
    with pytest.raises(ValueError,match="inside"):
        tool.compare_pilot_runs(settings)


def test_error_deltas_do_not_conflate_max_with_overall_improvement():
    def row(maximum,p95,rms,count):
        return {"case_id":"x","comparison":{
            "prediction":{"sha256":str(count),"raw_modelspace":{"count":count},"filtered_profile":{"count":count}},
            "reference":{"sha256":"same","raw_modelspace":{"count":4},"filtered_profile":{"count":4}},
            "physical_score":{"registered_metrics":{"max_error_mm":maximum,"p95_error_mm":p95,"rms_error_mm":rms}}}}
    change=tool._changes([{"cases":[row(10,2,1,10)]},{"cases":[row(9,3,2,5)]}])[0]
    assert change["max_error_improved"]
    assert change["p95_error_change_mm"]==change["rms_error_change_mm"]==1
    assert change["filtered_entity_reduction_percent"]==50
    assert "reference_accuracy_improved" not in change


def test_zip_failure_keeps_previous_pointer(pilot_inputs,monkeypatch):
    settings,paths=pilot_inputs
    root=settings.runtime_root/"dxf-comparison";root.mkdir()
    pointer=root/"latest.json";pointer.write_text('{"old":true}')
    monkeypatch.setattr(tool.zipfile.ZipFile,"testzip",lambda self:"broken")
    with pytest.raises(ValueError,match="integrity"):
        tool.compare_pilot_runs(settings)
    assert json.loads(pointer.read_text())=={"old":True}
