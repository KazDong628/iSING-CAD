import hashlib
import math
from pathlib import Path
import zipfile

import ezdxf
import numpy as np
from PIL import Image
import pytest
from shapely.geometry import Polygon

from contour_agent.dxf_supervision import audit_references, load_case_reference, load_reference_polygon


def save(document, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    document.saveas(path)
    return path


def square(path, *, units=4, gap=0):
    document = ezdxf.new("R2010"); document.units = units
    space = document.modelspace()
    # Deliberately unordered and reversed, as real CAD storage is not a path.
    space.add_line((10, 10), (10, 0))
    space.add_line((0, 0), (10, 0))
    space.add_line((0, 10), (0, gap))
    space.add_line((0, 10), (10, 10))
    return save(document, path)


def test_orders_reversed_entities_and_ignores_construction(tmp_path):
    path = square(tmp_path / "shape.dxf")
    document = ezdxf.readfile(path)
    document.modelspace().add_circle((5000, 5000), 500, dxfattribs={"layer": "CONSTRUCTION_DEBUG"})
    document.modelspace().add_text("Reference annotations are not instructions")
    document.saveas(path)
    before = path.read_bytes()
    result = load_reference_polygon(path)
    assert result["status"] == "ready"
    assert result["polygon_xy"][0] == result["polygon_xy"][-1]
    assert Polygon(result["polygon_xy"]).exterior.is_ccw
    assert result["main_area"] == pytest.approx(100)
    assert result["bounds"] == [0, 0, 10, 10]
    assert result["geometry_info"]["excluded"]["layer:CONSTRUCTION_DEBUG"] == 1
    assert result["source_sha256"] == hashlib.sha256(before).hexdigest()
    assert path.read_bytes() == before
    assert result["registration_verified"] is False and result["engineering_certified"] is False


def test_arc_flattening_and_inch_conversion(tmp_path):
    document = ezdxf.new(); document.units = ezdxf.units.IN
    space = document.modelspace()
    space.add_arc((0, 0), 1, 0, 180)
    space.add_line((-1, 0), (1, 0))
    result = load_reference_polygon(save(document, tmp_path / "inch.dxf"), chord_tolerance_mm=.005)
    assert result["status"] == "ready"
    assert result["units"]["millimetre_conversion_factor"] == pytest.approx(25.4)
    assert result["main_area"] == pytest.approx(math.pi / 2 * 25.4 ** 2, rel=.001)
    points = np.asarray(result["polygon_xy"])
    assert points[:, 0].min() == pytest.approx(-25.4) and points[:, 1].max() == pytest.approx(25.4)


@pytest.mark.parametrize("legacy", [False, True])
def test_closed_polylines_include_bulge_arcs(tmp_path, legacy):
    document = ezdxf.new(); document.units = 4
    space = document.modelspace()
    if legacy:
        entity = space.add_polyline2d([(0, 0), (2, 0)], close=True)
        entity.vertices[0].dxf.bulge = 1
    else:
        space.add_lwpolyline([(0, 0, 1), (2, 0, 0)], format="xyb", close=True)
    result = load_reference_polygon(save(document, tmp_path / "bulge.dxf"), chord_tolerance_mm=.0005)
    assert result["status"] == "ready"
    assert result["main_area"] == pytest.approx(math.pi / 2, rel=.002)
    assert len(result["polygon_xy"]) > 20


def test_multiple_components_and_nested_holes_are_retained(tmp_path):
    document = ezdxf.new(); document.units = 4
    space = document.modelspace()
    space.add_circle((0, 0), 10)
    space.add_circle((0, 0), 3)
    space.add_lwpolyline([(30, 0), (40, 0), (40, 5), (30, 5)], close=True)
    result = load_reference_polygon(save(document, tmp_path / "holes.dxf"), chord_tolerance_mm=.005)
    assert result["status"] == "ready" and result["component_count"] == 2
    assert len(result["polygons"][0]["holes"]) == 1
    assert result["main_area"] == pytest.approx(math.pi * 91, rel=.002)
    assert result["polygons"][1]["area"] == pytest.approx(50)
    assert result["selection_ambiguous"] is False


def test_transformed_insert_and_ocs_coordinates(tmp_path):
    document = ezdxf.new(); document.units = 4
    block = document.blocks.new("SHAPE")
    block.add_lwpolyline([(0, 0), (2, 0), (2, 1), (0, 1)], close=True)
    document.modelspace().add_blockref("SHAPE", (10, 20), dxfattribs={"rotation": 90, "xscale": 2, "yscale": 2, "layer": "MAIN"})
    result = load_reference_polygon(save(document, tmp_path / "block.dxf"))
    assert result["status"] == "ready" and result["main_area"] == pytest.approx(8)
    assert result["bounds"] == pytest.approx([8, 20, 10, 24])
    assert result["geometry_info"]["layers"] == ["MAIN"]
    other = ezdxf.new(); other.units = 4
    other.modelspace().add_circle((2, 3), 1, dxfattribs={"extrusion": (0, 0, -1)})
    ocs = load_reference_polygon(save(other, tmp_path / "ocs.dxf"), chord_tolerance_mm=.001)
    assert ocs["status"] == "ready"
    assert ocs["bounds"] == pytest.approx([-3, 2, -1, 4])


def test_unitless_geometry_not_asserted_to_be_millimetres(tmp_path):
    result = load_reference_polygon(square(tmp_path / "unitless.dxf", units=0))
    assert result["status"] == "ready" and result["units"]["coordinate_units"] == "dxf_unit"
    assert result["units"]["millimetre_conversion_factor"] is None
    assert result["units"]["physical_units_declared"] is False


def test_tiny_endpoint_mismatch_is_disclosed_but_large_gap_is_not_closed(tmp_path):
    tiny = load_reference_polygon(square(tmp_path / "tiny.dxf", gap=.0008))
    assert tiny["status"] == "ready"
    assert tiny["join_info"]["max_endpoint_snap"] == pytest.approx(.0004)
    large = load_reference_polygon(square(tmp_path / "open.dxf", gap=1))
    assert large["status"] == "invalid" and large["polygon_xy"] == []
    assert large["join_info"]["open_or_branched_components"] == 1
    assert large["join_info"]["node_degree_counts"]["1"] == 2
    assert large["join_info"]["max_nearest_endpoint_distance"] == pytest.approx(1)


def test_self_intersection_is_not_repaired_with_buffer_or_hull(tmp_path):
    document = ezdxf.new(); document.units = 4
    document.modelspace().add_lwpolyline([(0, 0), (10, 10), (0, 10), (10, 0)], close=True)
    path = save(document, tmp_path / "bowtie.dxf")
    original = path.read_bytes()
    result = load_reference_polygon(path, repair_topology=False)
    assert result["status"] == "invalid" and result["polygon_xy"] == []
    assert any("Self-intersection" in issue for issue in result["issues"])
    recovered = load_reference_polygon(path)
    assert recovered["status"] == "ready" and recovered["topology_repaired"] is True
    assert recovered["topology_repair"]["face_count"] == 2
    assert recovered["topology_repair"]["closure_edges_added"] == 0
    assert recovered["main_area"] == pytest.approx(25)
    assert recovered["main_selection"] == "largest_bounded_planar_face"
    assert recovered["selection_ambiguous"] is True
    assert path.read_bytes() == original


def test_planar_repair_retains_main_face_and_discloses_dangling_reference_line(tmp_path):
    path = square(tmp_path / "branched.dxf")
    document = ezdxf.readfile(path)
    document.modelspace().add_line((0, 0), (-.2, 0))
    document.saveas(path)
    result = load_reference_polygon(path)
    assert result["status"] == "ready" and result["topology_repaired"]
    assert result["main_area"] == pytest.approx(100)
    assert result["topology_repair"]["dangle_length"] == pytest.approx(.2)
    assert result["topology_repair"]["native_reference_modified"] is False


def test_zip_read_does_not_extract_and_matches_direct_geometry(tmp_path):
    path = square(tmp_path / "reference.dxf")
    archive = tmp_path / "reference.zip"
    with zipfile.ZipFile(archive, "w") as package:
        package.write(path, "nested/reference.dxf")
    original_files = set(tmp_path.rglob("*"))
    direct = load_reference_polygon(path)
    zipped = load_reference_polygon(archive, archive_member="nested/reference.dxf")
    assert zipped["status"] == "ready"
    assert direct["polygon_xy"] == zipped["polygon_xy"]
    assert direct["source_sha256"] == zipped["source_sha256"]
    assert set(tmp_path.rglob("*")) == original_files
    assert load_reference_polygon(archive, archive_member="../reference.dxf")["status"] == "invalid"


def test_invalid_preferred_reference_can_use_a_valid_alternate(tmp_path):
    root = tmp_path / "dataset"
    bad = square(root / "GT/044_main_profile_GT_v2.dxf", gap=2)
    good = square(root / "GT/044_main_profile_GT_v1.dxf")
    result = load_case_reference(root, "044-main")
    assert result["status"] == "ready" and Path(result["source"]["path"]) == good
    assert result["source_attempts"][0]["path"] == bad.relative_to(root).as_posix()
    assert result["source_attempts"][0]["status"] == "invalid"


def test_partial_reference_not_promoted_and_missing_cases_stay_in_inventory(tmp_path):
    root = tmp_path / "dataset"
    (root / "origin").mkdir(parents=True)
    for name in ("044-main", "245-main", "188-main"):
        Image.new("RGB", (8, 8), "white").save(root / "origin" / (name + ".png"))
    square(root / "GT/044_main_profile_GT_v1.dxf")
    square(root / "GT/188_main_profile_scored_only_v2.dxf")
    result = audit_references(root)
    assert result["summary"] == {"total": 3, "ready": 1, "invalid": 1, "missing": 1}
    assert {row["case_id"]: row["status"] for row in result["cases"]} == {"044-main": "ready", "188-main": "invalid", "245-main": "missing"}


def test_corrupt_dxf_has_explicit_failure(tmp_path):
    path = tmp_path / "corrupt.dxf"; path.write_bytes(b"this is not DXF")
    result = load_reference_polygon(path)
    assert result["status"] == "invalid" and result["polygon_xy"] == []
    assert any("Reference geometry unavailable" in issue for issue in result["issues"])
