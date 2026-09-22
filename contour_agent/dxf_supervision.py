"""Read-only DXF geometry for explicitly authorized TRAINING supervision.

This module is not imported by runtime prediction. It orders existing boundary
entities, converts declared units, and samples curves; it does not register a
reference to an image, repair a large gap, or certify the reference geometry.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import math
from pathlib import Path, PurePosixPath
import zipfile

import ezdxf
from ezdxf.path import make_path
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, LineString
from shapely.geometry.polygon import orient
from shapely.ops import unary_union, polygonize_full
from shapely.validation import explain_validity

from .dataset import IMAGE_SUFFIXES, _matches_case, _profile_rank, resolve_inside
from .evaluation import _NON_PROFILE, _ASSUMED

MAX_DXF_BYTES = 50_000_000
MAX_ENTITIES = 20_000
MAX_POINTS = 300_000


def _safe_member(name):
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or ":" in name or "\x00" in name:
        raise ValueError("Unsafe archive member")
    return name


def _geometry_filename(name):
    name = str(name).lower()
    return name.endswith(".dxf") and not any(word in name for word in ("construction", "debug", "measurement_audit"))


def _reference_inventory(dataset_root):
    root = Path(dataset_root).resolve()
    gt = resolve_inside(root, "GT")
    result = []
    if not gt.is_dir():
        return result
    for candidate in sorted(gt.rglob("*")):
        if not candidate.is_file():
            continue
        path = resolve_inside(root, candidate.relative_to(root))
        relative = path.relative_to(root).as_posix()
        if _geometry_filename(path.name):
            result.append({"path": relative, "member": None})
        elif path.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(path) as archive:
                    for entry in archive.infolist():
                        if not entry.is_dir() and _geometry_filename(entry.filename):
                            try:
                                _safe_member(entry.filename)
                            except ValueError:
                                continue
                            result.append({"path": relative, "member": entry.filename})
            except (OSError, zipfile.BadZipFile):
                # A matching invalid container is retained as a failed source.
                result.append({"path": relative, "member": None, "inventory_error": "invalid_archive"})
    return result


def _case_candidates(inventory, case_id):
    candidates = [dict(row) for row in inventory if _matches_case(Path(row["path"]), case_id)
                  or row.get("member") and _matches_case(Path(row["member"]), case_id)]
    def rank(row):
        name = row.get("member") or row["path"]
        partial = any(word in name.lower() for word in ("scored_only", "strict_body"))
        profile = _profile_rank(Path(name))
        # A complete file takes precedence over an explicitly partial profile.
        return (partial, *profile[:-1], bool(row.get("member")), profile[-1])
    return sorted(candidates, key=rank)


def reference_candidates(dataset_root, case_id):
    """Return ranked direct/ZIP references without extracting or mutating them."""
    return _case_candidates(_reference_inventory(dataset_root), case_id)


def _read_document(path, archive_member):
    path = Path(path)
    if archive_member is not None:
        _safe_member(archive_member)
        with zipfile.ZipFile(path) as archive:
            info = archive.getinfo(archive_member)
            if info.file_size > MAX_DXF_BYTES:
                raise ValueError("Archived DXF exceeds size limit")
            raw = archive.read(info)
        document = ezdxf.readzip(path, archive_member)
    else:
        if path.stat().st_size > MAX_DXF_BYTES:
            raise ValueError("DXF exceeds size limit")
        raw = path.read_bytes()
        document = ezdxf.readfile(path)
    return document, hashlib.sha256(raw).hexdigest()


def _flatten(document, chord_tolerance_mm):
    units = int(document.units)
    factor = float(ezdxf.units.conversion_factor(units, ezdxf.units.MM)) if units else 1.
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("Invalid unit conversion")
    paths, excluded, types, layers, issues = [], Counter(), Counter(), set(), []
    point_count = 0

    def visit(entity, inherited_layer=None, depth=0):
        nonlocal point_count
        layer = str(entity.dxf.layer)
        if layer == "0" and inherited_layer:
            layer = inherited_layer
        if _NON_PROFILE.search(layer):
            excluded["layer:" + layer] += 1
            return
        kind = entity.dxftype()
        if kind == "INSERT":
            if depth >= 5:
                raise ValueError("Nested block depth exceeds limit")
            for child in entity.virtual_entities():
                visit(child, layer, depth + 1)
            return
        if kind not in {"LINE", "ARC", "CIRCLE", "LWPOLYLINE", "POLYLINE", "SPLINE", "ELLIPSE"}:
            excluded["type:" + kind] += 1
            return
        if sum(types.values()) >= MAX_ENTITIES:
            raise ValueError("DXF entity limit exceeded")
        types[kind] += 1
        path = make_path(entity)
        points = np.asarray([tuple(point) for point in path.flattening(chord_tolerance_mm / factor, segments=4)], dtype=float) * factor
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.isfinite(points).all():
            raise ValueError("Non-finite or empty reference curve")
        if np.ptp(points[:, 2]) > 1e-7:
            raise ValueError("Nonplanar reference curve")
        # Translation of a 2D drawing along Z is harmless; varying Z is not.
        xy = points[:, :2]
        keep = np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-12]
        xy = xy[keep]
        if len(xy) < 2:
            excluded["degenerate:" + kind] += 1
            return
        point_count += len(xy)
        if point_count > MAX_POINTS:
            raise ValueError("DXF sampling point limit exceeded")
        paths.append(xy)
        layers.add(layer)

    for original in document.modelspace():
        visit(original)
    return paths, {"source_units": units, "coordinate_units": "mm" if units else "dxf_unit",
                   "millimetre_conversion_factor": factor if units else None,
                   "physical_units_declared": bool(units), "entity_types": dict(types),
                   "layers": sorted(layers), "assumed_layers": sorted(layer for layer in layers if _ASSUMED.search(layer)),
                   "excluded": dict(excluded), "issues": issues}


def _closed_rings(paths, join_tolerance):
    """Join only existing endpoints; no convex hull or inferred closing edge."""
    if not paths:
        return [], {"open_or_branched_components": 0, "max_endpoint_snap": 0., "issues": ["No profile geometry"]}
    endpoints = np.asarray([point for path in paths for point in (path[0], path[-1])])
    parents = list(range(len(endpoints)))
    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index
    for a, b in cKDTree(endpoints).query_pairs(join_tolerance):
        parents[find(a)] = find(b)
    clusters = {}
    for index in range(len(endpoints)):
        clusters.setdefault(find(index), []).append(index)
    nodes, point_node, maximum = [], {}, 0.
    for cluster in clusters.values():
        node = endpoints[cluster].mean(axis=0)
        error = float(np.linalg.norm(endpoints[cluster] - node, axis=1).max())
        if error > join_tolerance:
            raise ValueError("Transitive endpoint cluster exceeds join tolerance")
        maximum = max(maximum, error)
        for index in cluster:
            point_node[index] = len(nodes)
        nodes.append(node)
    edges, adjacency = [], {}
    for index, path in enumerate(paths):
        a, b = point_node[index * 2], point_node[index * 2 + 1]
        points = path.copy(); points[0] = nodes[a]; points[-1] = nodes[b]
        edges.append((a, b, points))
        adjacency.setdefault(a, []).append(index)
        adjacency.setdefault(b, []).append(index)
    unseen, rings, bad_components, issues = set(range(len(edges))), [], 0, []
    while unseen:
        seed = min(unseen)
        component, pending, component_nodes = set(), [seed], set()
        while pending:
            index = pending.pop()
            if index in component:
                continue
            component.add(index)
            for node in edges[index][:2]:
                component_nodes.add(node)
                pending.extend(edge for edge in adjacency[node] if edge not in component)
        unseen -= component
        if any(len(adjacency[node]) != 2 for node in component_nodes):
            bad_components += 1
            continue
        current = start = edges[seed][0]
        remaining = set(component)
        coordinates = []
        while remaining:
            index = min(edge for edge in adjacency[current] if edge in remaining)
            a, b, points = edges[index]
            forward = current == a
            points = points if forward else points[::-1]
            coordinates.extend(points.tolist() if not coordinates else points[1:].tolist())
            current = b if forward else a
            remaining.remove(index)
        if current != start:
            bad_components += 1
            continue
        polygon = Polygon(coordinates)
        if not polygon.is_valid or polygon.area <= 1e-10:
            issues.append("Closed component excluded: " + explain_validity(polygon))
            continue
        rings.append(polygon)
    if bad_components:
        issues.append(f"{bad_components} open/branched component(s) excluded; no missing closure was invented.")
    nearest = cKDTree(endpoints).query(endpoints, k=2)[0][:, 1]
    exceptional = [{"xy": nodes[node].tolist(), "degree": len(edges_at_node)}
                   for node, edges_at_node in adjacency.items() if len(edges_at_node) != 2]
    return rings, {"open_or_branched_components": bad_components, "max_endpoint_snap": maximum,
                   "max_nearest_endpoint_distance": float(nearest.max()),
                   "node_degree_counts": dict(Counter(str(len(edges_at_node)) for edges_at_node in adjacency.values())),
                   "exceptional_nodes": exceptional[:40], "issues": issues}


def _material_polygons(rings):
    rings = sorted(rings, key=lambda polygon: (-polygon.area, tuple(polygon.bounds)))
    parent = [None] * len(rings)
    for index, ring in enumerate(rings):
        containers = []
        for other in range(index):
            if rings[other].contains(ring):
                containers.append(other)
            elif rings[other].intersects(ring):
                raise ValueError("Reference rings overlap or touch ambiguously")
        if containers:
            parent[index] = min(containers, key=lambda other: rings[other].area)
    depths = []
    for index in range(len(rings)):
        depths.append(0 if parent[index] is None else depths[parent[index]] + 1)
    result = []
    for index, ring in enumerate(rings):
        if depths[index] % 2:
            continue
        holes = [list(child.exterior.coords) for k, child in enumerate(rings) if parent[k] == index and depths[k] % 2]
        polygon = orient(Polygon(ring.exterior.coords, holes), sign=1.)
        if not polygon.is_valid:
            raise ValueError("Invalid material polygon with holes")
        result.append({"exterior": [list(point) for point in polygon.exterior.coords],
                       "holes": [[list(point) for point in interior.coords] for interior in polygon.interiors],
                       "area": float(polygon.area), "bounds": list(polygon.bounds)})
    return sorted(result, key=lambda item: (-item["area"], item["bounds"]))


def _planar_face_repair(paths, join_tolerance):
    """Derive bounded faces without adding a segment across an open gap.

    This is training-label derivation only. Existing intersections are noded and
    overlapping segments unified; endpoint movement is separately bounded.
    Loops, dangling pieces and alternate bounded regions stay disclosed.
    """
    if not paths:
        return [], {"method": "bounded_planar_faces", "face_count": 0, "closure_edges_added": 0}
    ends = np.asarray([point for path in paths for point in (path[0], path[-1])])
    parents = list(range(len(ends)))
    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index
    for left, right in cKDTree(ends).query_pairs(join_tolerance):
        parents[find(left)] = find(right)
    clusters = {}
    for index in range(len(ends)):
        clusters.setdefault(find(index), []).append(index)
    maximum = 0.
    for indices in clusters.values():
        midpoint = ends[indices].mean(axis=0)
        error = float(np.linalg.norm(ends[indices] - midpoint, axis=1).max())
        if error > join_tolerance:
            raise ValueError("Planar repair endpoint movement exceeds tolerance")
        maximum = max(maximum, error)
        ends[indices] = midpoint
    lines = []
    for index, original in enumerate(paths):
        points = original.copy(); points[0] = ends[index * 2]; points[-1] = ends[index * 2 + 1]
        lines.append(LineString(points))
    network = unary_union(lines)
    faces, cuts, dangles, invalid = polygonize_full(network)
    valid = sorted((polygon for polygon in faces.geoms if polygon.is_valid and polygon.area > 1e-10),
                   key=lambda polygon: (-polygon.area, tuple(polygon.bounds)))
    polygons = []
    for polygon in valid:
        polygon = orient(polygon, sign=1.)
        polygons.append({"exterior": [list(point) for point in polygon.exterior.coords],
                         "holes": [[list(point) for point in interior.coords] for interior in polygon.interiors],
                         "area": float(polygon.area), "bounds": list(polygon.bounds)})
    return polygons, {"method": "endpoint_snap_then_unary_union_polygonize_full",
                      "selection": "largest_bounded_planar_face", "face_count": len(polygons),
                      "face_areas": [polygon["area"] for polygon in polygons],
                      "secondary_face_area": sum(polygon["area"] for polygon in polygons[1:]),
                      "cut_length": float(cuts.length), "dangle_length": float(dangles.length),
                      "invalid_ring_length": float(invalid.length), "max_endpoint_snap": maximum,
                      "closure_edges_added": 0, "native_reference_modified": False,
                      "scope": "Derived training region; original CAD topology and dimensional certification are not repaired."}


def load_reference_polygon(path, *, archive_member=None, chord_tolerance_mm=.05, join_tolerance_mm=.001, repair_topology=True):
    """Return a JSON-safe geometry receipt from a direct DXF or ZIP member.

    polygon_xy is the largest material component's CCW exterior, explicitly
    closed and sampled in CAD XY. Additional components and holes are retained
    in polygons. Declared DXF units convert to mm; unitless inputs remain marked
    dxf_unit. When strict rings fail, optional planar-face derivation is explicitly
    marked topology_repaired. Registration quality remains the caller's task.
    """
    result = {"status": "invalid", "polygon_xy": [], "polygons": [],
              "source": {"path": str(Path(path).resolve()), "member": archive_member},
              "label_source": "gt_dxf_training_supervision", "registration_verified": False,
              "engineering_certified": False, "issues": [], "topology_repaired": False,
              "chord_tolerance_mm": chord_tolerance_mm, "join_tolerance_mm": join_tolerance_mm}
    try:
        if not all(math.isfinite(value) and value > 0 for value in (chord_tolerance_mm, join_tolerance_mm)):
            raise ValueError("Positive finite geometry tolerances required")
        document, digest = _read_document(path, archive_member)
        result["source_sha256"] = digest
        paths, info = _flatten(document, chord_tolerance_mm)
        result["units"] = {key: info[key] for key in ("source_units", "coordinate_units", "millimetre_conversion_factor", "physical_units_declared")}
        result["geometry_info"] = info
        rings, joins = _closed_rings(paths, join_tolerance_mm)
        result["join_info"] = joins
        result["issues"].extend(joins["issues"])
        polygons = _material_polygons(rings)
        if not polygons and repair_topology:
            polygons, repair = _planar_face_repair(paths, join_tolerance_mm)
            result["topology_repair"] = repair
            if polygons:
                result["topology_repaired"] = True
                result["issues"].append("Training region recovered from existing planar faces; native reference topology was not modified.")
        if not polygons:
            result["issues"].append("No valid closed material polygon available.")
            return result
        result.update(status="ready", polygon_xy=polygons[0]["exterior"], polygons=polygons,
                      component_count=len(polygons), main_area=polygons[0]["area"], bounds=polygons[0]["bounds"],
                      main_selection="largest_bounded_planar_face" if result["topology_repaired"] else "largest_closed_material_component",
                      selection_ambiguous=len(polygons) > 1 and polygons[1]["area"] >= .9 * polygons[0]["area"])
        if len(polygons) > 1:
            result["issues"].append("Multiple closed components retained; polygon_xy explicitly selects the largest one.")
        if info["assumed_layers"]:
            result["issues"].append("Reference includes declared closure/simplified/soft geometry layers.")
        if not info["physical_units_declared"]:
            result["issues"].append("DXF units are unspecified; coordinates were not asserted to be millimetres.")
        return result
    except (OSError, ValueError, TypeError, KeyError, ezdxf.DXFError, zipfile.BadZipFile, OverflowError) as error:
        result["issues"].append(f"Reference geometry unavailable ({type(error).__name__}): {error}")
        return result


def _load_case(root, case_id, candidates, chord_tolerance_mm):
    attempts, seen = [], set()
    for candidate in candidates:
        name = candidate.get("member") or candidate["path"]
        if any(word in name.lower() for word in ("scored_only", "strict_body")):
            attempts.append({**candidate, "status": "partial_reference_excluded"})
            continue
        result = load_reference_polygon(resolve_inside(root, candidate["path"]), archive_member=candidate.get("member"), chord_tolerance_mm=chord_tolerance_mm)
        digest = result.get("source_sha256")
        attempts.append({**candidate, "status": result["status"], "source_sha256": digest,
                         "duplicate_source_content": digest in seen if digest else False, "issues": result["issues"]})
        if digest:
            seen.add(digest)
        if result["status"] == "ready":
            result.update(case_id=case_id, alternatives=candidates, source_attempts=attempts)
            return result
    return {"case_id": case_id, "status": "missing" if not candidates else "invalid", "polygon_xy": [], "polygons": [],
            "alternatives": candidates, "source_attempts": attempts, "registration_verified": False,
            "issues": ["No matching GT DXF file or ZIP member found." if not candidates else "No eligible closed main polygon among the matching reference sources."]}


def load_case_reference(dataset_root, case_id, *, chord_tolerance_mm=.05):
    """Choose an eligible closed main profile, retaining failed alternatives."""
    root = Path(dataset_root).resolve()
    return _load_case(root, case_id, reference_candidates(root, case_id), chord_tolerance_mm)


def audit_references(dataset_root):
    """Inventory every source image, including missing or invalid supervision."""
    root = Path(dataset_root).resolve()
    origin = resolve_inside(root, "origin")
    inventory = _reference_inventory(root)
    cases = []
    for image in sorted(origin.iterdir()):
        if image.suffix.lower() not in IMAGE_SUFFIXES or not image.is_file():
            continue
        result = _load_case(root, image.stem, _case_candidates(inventory, image.stem), .05)
        result["source_image"] = image.relative_to(root).as_posix()
        result["polygon_point_count"] = len(result.pop("polygon_xy"))
        result["component_areas"] = [polygon["area"] for polygon in result.pop("polygons")]
        cases.append(result)
    return {"label_source": "gt_dxf_training_supervision", "registration_verified": False,
            "summary": {"total": len(cases), **dict(Counter(case["status"] for case in cases))}, "cases": cases}
