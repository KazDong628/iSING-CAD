"""Synthetic fresh-job bundles only: no model inference or provider requests."""
import hashlib
import json
from pathlib import Path
import sqlite3
import zipfile

import ezdxf
from PIL import Image
import pytest

from contour_agent.config import Settings
from tools import run_segmentation_pilot as pilot


RUN_ID = "20260920T130000000000Z-a1b2c3d4"


def _dxf(path, units=4):
    document = ezdxf.new()
    document.units = units
    vertices = [(0, 0), (10, 0), (10, 5), (0, 5)]
    for start, end in zip(vertices, vertices[1:] + vertices[:1]):
        document.modelspace().add_line(start, end)
    document.saveas(path)


@pytest.fixture
def fresh_run(monkeypatch, tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    checkpoint = tmp_path / "synthetic-checkpoint.pt"
    checkpoint.write_bytes(b"synthetic weights; never loaded")
    settings = Settings(dataset_root=dataset, runtime_root=tmp_path / "runtime",
                        segmentation_checkpoint=str(checkpoint), api_key="sk-synthetic-private-key")
    case_ids = [*pilot.PILOT_CASE_IDS, *[f"unattempted-{i}" for i in range(46)]]
    catalog = {"cases": [{"id": case_id, "image": f"{case_id}.png", "gt_dxf": None} for case_id in case_ids]}
    for index, case_id in enumerate(pilot.PILOT_CASE_IDS):
        Image.new("RGB", (20, 10), color=(index * 20, 180, 200)).save(dataset / f"{case_id}.png")
    monkeypatch.setattr(pilot, "build_catalog", lambda root: catalog)
    control = {"catalog": catalog, "calls": [], "before_persist": None, "after_persist": None, "jobs": []}

    def qualify(config, **options):
        control["calls"].append((config, options))
        directory = config.runtime_root / "autonomous-qualification" / RUN_ID
        agent_runtime = directory / "agent-runtime"
        agent_runtime.mkdir(parents=True)
        connection = sqlite3.connect(agent_runtime / "jobs.sqlite3")
        connection.execute("CREATE TABLE jobs(id TEXT PRIMARY KEY, updated TEXT NOT NULL, document TEXT NOT NULL)")
        trials = []
        for index, case_id in enumerate(pilot.PILOT_CASE_IDS):
            job_id = f"{index + 1:032x}"
            generated = agent_runtime / "jobs" / job_id / "automatic-001"
            generated.mkdir(parents=True)
            units = 0 if index == 3 else 4
            _dxf(generated / "drawing.dxf", units)
            (generated / "preview.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"></svg>')
            for name in ("overlay.png", "segmentation-mask.png", "segmentation-overlay.png", "contour-overlay.png"):
                Image.new("L", (20, 10), color=255).save(generated / name)
            validation = {"passed": True, "scaled_mm": units == 4, "dimensions_verified": False, "radius_bindings": []}
            for name in ("model.json", "dimension-evidence.json", "curve-fit.json", "segmentation.json"):
                (generated / name).write_text(json.dumps({"generated": True, "provider_error": settings.api_key}))
            (generated / "validation.json").write_text(json.dumps(validation))
            provider = {"status": "succeeded", "http_success": True, "schema_success": True, "network_requests": 1} if options["online"] else {"status": "disabled", "network_requests": 0}
            analysis = {"status": "completed", "provider": dict(provider), "counts": {"recognized_dimensions": 8}, "dimensions_verified": False, "geometry_updated_by_api": False}
            (generated / "dimension-analysis.json").write_text(json.dumps(analysis))
            source = {"image_sha256": pilot._hash(dataset / f"{case_id}.png"), "width": 20, "height": 10}
            job = {"id": job_id, "case_id": case_id, "use_segmentation": True, "status": "completed", "source": source,
                   "validation": validation, "scale": {"status": "resolved" if units == 4 else "unresolved"},
                   "provider": provider, "dimension_analysis": analysis, "curve_fit": {"status": "fitted"},
                   "extraction": {"model": {"checkpoint_sha256": pilot._hash(checkpoint)}}}
            control["jobs"].append(job)
            comparison = {"status": "compared" if index in (0, 2) else "unscaled_prediction", "reference_compared": index in (0, 2),
                          "reference_within_0_1mm": False, "engineering_verified": False, "tolerance_mm": .1,
                          "registered_metrics": {"max_error_mm": 13.0, "p95_error_mm": 8.0, "rms_error_mm": 4.0} if index in (0, 2) else None}
            trials.append({"case_id": case_id, "job_id": job_id, "status": "completed", "automatic_completion": True,
                           "artifacts": {name: str((generated / name).resolve()) for name in pilot.CORE_ARTIFACTS},
                           "comparison": comparison, "validation": validation, "provider": provider, "source": source,
                           "model_exposure": {"split": "test", "role": "development_test", "blind_test": False}})
        report = {"run_id": RUN_ID, "status": "completed", "catalog_count": 50, "trials": trials,
                  "online_requested": options["online"], "segmentation_requested": True,
                  "summary": {"total": 50, "attempted": 4, "not_attempted": 46, "reference_within_0_1mm": 0},
                  "qualification": {"strict_requested_scope_passed": False, "strict_whole_dataset_passed": False},
                  "online_request_summary": {"logical_calls": {"total": 4 if options["online"] else 0, "http_successful": 4 if options["online"] else 0}, "network_attempts": {"total": 4 if options["online"] else 0}}}
        if control["before_persist"]:
            control["before_persist"](report, control["jobs"], directory)
        for job in control["jobs"]:
            connection.execute("INSERT INTO jobs VALUES(?,?,?)", (job["id"], "now", json.dumps(job)))
        connection.commit()
        connection.close()
        path = config.runtime_root / "evaluations" / f"{RUN_ID}_autonomous.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report), encoding="utf8")
        if control["after_persist"]:
            control["after_persist"](report, path)
        return report, path

    monkeypatch.setattr(pilot, "qualify_autonomous", qualify)
    return settings, control


def test_fresh_bundle_binds_sources_artifacts_and_fifty_case_denominator(fresh_run):
    settings, control = fresh_run
    report, path = pilot.run_segmentation_pilot(settings)
    assert control["calls"] == [(settings, {"online": False, "repeats": 1, "cases": list(pilot.PILOT_CASE_IDS), "use_segmentation": True})]
    assert report["counts"] == {"dataset_total": 50, "selected": 4, "not_selected": 46, "attempted": 4,
                                "dxf_exists": 4, "valid_dxf": 4, "scaled_mm": 3, "reference_compared": 2,
                                "reference_within_0_1mm": 0, "all_dimensions_verified": 0}
    assert report["development_only"] and not report["blind_test"] and not report["engineering_verified"]
    assert len({row["job_id"] for row in report["cases"]}) == 4
    assert all(row["source"]["image_hash_verified"] and row["segmentation"]["checkpoint_hash_verified"] for row in report["cases"])
    assert all(row["segmentation"]["mask_present"] and row["fresh_job_journal_found"] for row in report["cases"])
    assert report["cases"][3]["dxf"]["units"] == "pixel_or_unspecified"
    page = (path.parent / "index.html").read_text(encoding="utf8")
    assert "data:image/png;base64," in page and "file://" not in page and "href=\"E:" not in page
    assert f"/api/evaluations/{RUN_ID}_autonomous.json/cases/CL60-main/artifacts/drawing.dxf" in page
    assert f"/api/segmentation/pilot/{RUN_ID}/pilot-artifacts.zip" in page
    with zipfile.ZipFile(path.parent / "pilot-artifacts.zip") as archive:
        assert archive.testzip() is None
        assert all(not name.startswith("/") and ".." not in Path(name).parts for name in archive.namelist())
        for row in report["cases"]:
            receipt = row["artifacts"]["drawing.dxf"]
            assert hashlib.sha256(archive.read(receipt["package_path"])).hexdigest() == row["dxf"]["sha256"]
        assert not any("reference-only" in name or ".sqlite3" in name or name.endswith(".pt") for name in archive.namelist())
        assert all(settings.api_key.encode() not in archive.read(name) for name in archive.namelist())
    pointer = json.loads((path.parent.parent / "latest.json").read_text())
    assert pointer["run_id"] == RUN_ID and pointer["status"] == "completed"
    assert pointer["report_sha256"] == pilot._hash(path)
    assert pointer["archive_sha256"] == pilot._hash(path.parent / "pilot-artifacts.zip")


def test_dimensions_and_vision_are_distinct_calls_with_unknowns_preserved(fresh_run):
    settings, control = fresh_run

    def change_receipts(report, jobs, directory):
        jobs[1]["dimension_analysis"]["provider"] = {"status": "failed", "network_requests": 2, "http_success": False, "schema_success": False}
        jobs[2].pop("dimension_analysis")
        jobs[3]["dimension_analysis"]["provider"] = {"status": "failed"}

    control["before_persist"] = change_receipts
    report, _ = pilot.run_segmentation_pilot(settings, online=True)
    assert report["vision_provider_statistics"]["logical_calls"]["total"] == 4
    assert report["vision_provider_statistics"]["network_attempts"]["total"] == 4
    dims = report["dimension_provider_statistics"]
    assert (dims["requested_cases"], dims["logical_calls"], dims["http_successful"], dims["missing_receipts"]) == (4, 3, 1, 1)
    assert dims["network_attempts"] is None and dims["known_network_attempts"] == 3
    assert dims["http_unknown"] == dims["schema_unknown"] == 1
    assert report["counts"]["reference_within_0_1mm"] == 0


def test_binding_calls_and_partial_solve_are_not_dimension_accuracy(fresh_run):
    settings,control=fresh_run
    def add_bindings(report,jobs,directory):
        jobs[0]["parameterization"]={"accepted":True,"all_dimensions_verified":False,
            "provider":{"status":"succeeded","network_requests":1,"http_success":True,"schema_success":True},
            "constraints":[{"kind":"radius","value":3}],"solver":{"status":"accepted","underconstrained":True}}
        jobs[1]["parameterization"]={"accepted":False,"provider":{"status":"interrupted","network_requests":None,"http_success":None,"schema_success":False}}
    control["before_persist"]=add_bindings
    report,_=pilot.run_segmentation_pilot(settings,online=True)
    statistics=report["binding_provider_statistics"]
    assert statistics["logical_calls"]==2 and statistics["http_successful"]==1
    assert statistics["network_attempts"] is None and statistics["known_network_attempts"]==1
    assert statistics["http_unknown"]==1 and statistics["missing_receipts"]==2
    assert report["cases"][0]["parameterization"]["accepted"]
    assert report["counts"]["all_dimensions_verified"]==0
    assert report["vision_provider_statistics"]["logical_calls"]["total"]==4
    assert report["dimension_provider_statistics"]["logical_calls"]==4


def test_failed_case_and_runtime_invalidity_remain_in_denominator(fresh_run):
    settings, control = fresh_run

    def fail(report, jobs, directory):
        trial = report["trials"][0]
        Path(trial["artifacts"]["drawing.dxf"]).unlink()
        trial.update(status="failed", automatic_completion=False, comparison={})
        jobs[0]["validation"]["passed"] = False
        jobs[1]["validation"]["passed"] = False  # Closed endpoint audit alone cannot override runtime rejection.

    control["before_persist"] = fail
    report, _ = pilot.run_segmentation_pilot(settings)
    assert report["counts"]["dataset_total"] == 50 and report["counts"]["attempted"] == 4
    assert report["counts"]["dxf_exists"] == 3 and report["counts"]["valid_dxf"] == 2
    assert report["cases"][0]["status"] == "failed"
    assert report["cases"][1]["dxf"]["candidate_validation"]["passed"]
    assert not report["cases"][1]["dxf"]["geometry_valid"]


def test_unitless_reference_stays_unscorable_and_is_not_packaged(fresh_run):
    settings, control = fresh_run
    reference = settings.dataset_root / "unitless-reference.dxf"
    _dxf(reference, units=0)
    control["catalog"]["cases"][1]["gt_dxf"] = reference.name
    report, path = pilot.run_segmentation_pilot(settings)
    row = report["cases"][1]
    assert row["reference"]["units"] == "unspecified"
    assert row["reference"]["reference_sha256"] == pilot._hash(reference)
    assert not row["reference"]["comparison"]["reference_within_0_1mm"]
    with zipfile.ZipFile(path.parent / "pilot-artifacts.zip") as package:
        assert reference.name not in package.namelist()


def test_unbound_artifact_path_is_omitted_instead_of_followed(fresh_run, tmp_path):
    settings, control = fresh_run
    outside = tmp_path / "private.json"
    outside.write_text('{"private":"never package arbitrary report paths"}')

    def redirect(report, jobs, directory):
        report["trials"][0]["artifacts"]["model.json"] = str(outside)

    control["before_persist"] = redirect
    report, path = pilot.run_segmentation_pilot(settings)
    assert "model.json" not in report["cases"][0]["artifacts"]
    assert any("not bound" in issue for issue in report["cases"][0]["issues"])
    with zipfile.ZipFile(path.parent / "pilot-artifacts.zip") as package:
        assert all(b"never package arbitrary" not in package.read(name) for name in package.namelist())


@pytest.mark.parametrize("mutation,message", [
    (lambda report: report.update(status="running"), "finished qualification"),
    (lambda report: report["trials"][0].update(job_id="../../outside"), "safe identifiers"),
    (lambda report: report.update(online_requested=True), "online modes"),
    (lambda report: report.update(run_id="../../outside"), "run identifier"),
])
def test_invalid_report_does_not_update_latest(fresh_run, mutation, message):
    settings, control = fresh_run
    control["before_persist"] = lambda report, jobs, directory: mutation(report)
    destination = settings.runtime_root / "segmentation" / "pilot"
    destination.mkdir(parents=True)
    pointer = destination / "latest.json"
    pointer.write_text('{"previous_completed_run":true}')
    with pytest.raises(ValueError, match=message):
        pilot.run_segmentation_pilot(settings)
    assert json.loads(pointer.read_text()) == {"previous_completed_run": True}


def test_pointer_updates_only_after_zip_finishes(fresh_run, monkeypatch):
    settings, _ = fresh_run
    destination = settings.runtime_root / "segmentation" / "pilot"
    destination.mkdir(parents=True)
    pointer = destination / "latest.json"
    pointer.write_text('{"previous_completed_run":true}')
    monkeypatch.setattr(pilot.zipfile.ZipFile, "testzip", lambda self: "corrupt-entry")
    with pytest.raises(ValueError, match="integrity"):
        pilot.run_segmentation_pilot(settings)
    assert json.loads(pointer.read_text()) == {"previous_completed_run": True}


def test_output_cannot_escape_runtime_before_any_generation(fresh_run, tmp_path):
    settings, control = fresh_run
    with pytest.raises(ValueError, match="within"):
        pilot.run_segmentation_pilot(settings, output_root=tmp_path / "outside")
    assert control["calls"] == []


def test_reused_report_is_rejected(fresh_run):
    settings, control = fresh_run
    directory = settings.runtime_root / "evaluations"
    directory.mkdir(parents=True)
    (directory / f"{RUN_ID}_autonomous.json").write_text("{}")
    with pytest.raises(ValueError, match="newly generated"):
        pilot.run_segmentation_pilot(settings)


def test_identity_mismatches_are_disclosed_not_invented(fresh_run):
    settings, control = fresh_run

    def mismatch(report, jobs, directory):
        jobs[0]["source"]["image_sha256"] = "0" * 64
        jobs[0]["extraction"]["model"]["checkpoint_sha256"] = "0" * 64
        Path(settings.segmentation_checkpoint).write_bytes(b"changed weights after fresh run")

    control["before_persist"] = mismatch
    report, _ = pilot.run_segmentation_pilot(settings)
    assert not report["checkpoint_unchanged_during_run"]
    assert not report["cases"][0]["source"]["image_hash_verified"]
    assert not report["cases"][0]["segmentation"]["checkpoint_hash_verified"]
    assert len(report["cases"][0]["issues"]) >= 2
