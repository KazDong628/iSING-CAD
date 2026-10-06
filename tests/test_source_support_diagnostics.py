import cv2
import numpy as np

from contour_agent.source_support_diagnostics import source_support_diagnostics


def test_local_diagnostics_locate_bad_start_without_approving_geometry(tmp_path):
    image = np.full((120, 200), 255, np.uint8)
    cv2.line(image, (20, 72), (55, 72), 0, 2)
    cv2.line(image, (56, 60), (160, 60), 0, 2)
    path = tmp_path / "source.png"
    cv2.imencode(".png", image)[1].tofile(path)
    graph = {"units": "pixel", "source_grid_pitch_px": 1.,
             "coordinate_system": {"origin_source_px": [0., 0.]},
             "entities": [{"id": "g000", "type": "LINE", "start": [20., -60.], "end": [160., -60.]}]}
    report = source_support_diagnostics(path, {}, {}, graph)
    quarters = report["entities"][0]["quarters"]
    assert report["ground_truth_used"] is False
    assert report["acceptance_thresholds_changed"] is False
    assert quarters[0]["stroke_supported_fraction"] < .2
    assert quarters[-1]["stroke_supported_fraction"] > .9
    assert "passed" not in report
