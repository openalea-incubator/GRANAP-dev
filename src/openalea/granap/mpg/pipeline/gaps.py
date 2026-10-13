"""
The gap fusion, converted (devplan_metafspm.md, §34, B2d): GRANAP's Organ.fuse_gaps and find_gaps
(organ_class.py:403, 459), the union of the cells and their filled mass computed once instead of up to three times
each (find_gaps, _filled_cell_mass called by both, fuse_gaps), and the cells' properties recalculated in bulk.

Exact: the unions are of the same polygons in the same order, so their results are the same; the properties use the
same GEOS centroid, distance and area on each polygon (the angle stays np.arctan2 cell by cell). Every other line is
GRANAP's.
"""
from collections import defaultdict
from typing import Any, Dict

import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree


def _filled_mass(covered):
    """Organ._filled_cell_mass from the cells' union: that union with its internal pockets filled in."""
    parts = covered.geoms if hasattr(covered, "geoms") else [covered]
    return unary_union([Polygon(p.exterior) for p in parts if p.geom_type == "Polygon"])


def find_gaps(organ, sliver_width=None, min_area=None, exclude_voids=True, enclosed_only=False, unions=None):
    """GRANAP's find_gaps; *unions* ({"covered", "mass"}) keeps the unions computed for the caller."""
    polys = [c.polygon for c in organ.all_cells.get_cells()
             if c.polygon is not None and not c.polygon.is_empty]
    if not polys:
        return []
    unions = {} if unions is None else unions
    outline = organ.generate_base_shape()
    if "covered" not in unions:
        unions["covered"] = unary_union(polys)
    covered = unions["covered"]
    gaps = outline.difference(covered)
    if exclude_voids:
        for v in organ._empty_void_polygons():
            gaps = gaps.difference(v)
    if gaps.is_empty:
        return []

    if sliver_width is None:
        med_area = float(np.median([p.area for p in polys]))
        med_diam = 2.0 * np.sqrt(med_area / np.pi) if med_area > 0 else 0.0
        sliver_width = 0.15 * med_diam
    if sliver_width > 0:
        gaps = gaps.buffer(-0.5 * sliver_width).buffer(0.5 * sliver_width)

    pieces = [g for g in (gaps.geoms if hasattr(gaps, "geoms") else [gaps])
              if g.geom_type == "Polygon" and not g.is_empty]
    if enclosed_only:
        # representative_point() is guaranteed inside the piece, unlike a centroid
        # on a crescent, so a surface gap curving around a tip is not mistaken for
        # an enclosed one.
        if "mass" not in unions:
            unions["mass"] = _filled_mass(covered)
        mass = unions["mass"]
        pieces = [g for g in pieces if g.representative_point().within(mass)]
    if min_area is not None:
        pieces = [g for g in pieces if g.area >= min_area]
    pieces.sort(key=lambda g: g.area, reverse=True)
    return pieces


def recalculate_cell_properties(cells) -> None:
    """CellManager.recalculate_cell_properties: angle, radius, area and id_cell of every cell, the geometry in bulk."""
    with_polygon = [i for i, cell in enumerate(cells) if cell.polygon is not None]
    polygons = np.array([cells[i].polygon for i in with_polygon], dtype=object)
    radii = shapely.distance(shapely.centroid(polygons), Point(0, 0)).tolist() if with_polygon else []
    areas = shapely.area(polygons).tolist() if with_polygon else []
    geometry = dict(zip(with_polygon, zip(radii, areas)))
    for i, cell in enumerate(cells):
        cell.angle = np.arctan2(cell.y, cell.x)
        if i in geometry:
            cell.radius, cell.area = geometry[i]
        else:
            cell.radius = np.sqrt(cell.x**2 + cell.y**2)
            cell.area = cell.diameter**2 * np.pi / 4
        cell.id_cell = i


def fuse_gaps(organ, sliver_width: float = None, min_area: float = None) -> int:
    """Absorb each detected gap into the nearest cell of its own tissue zone (GRANAP's, the unions shared)."""
    cells = [c for c in organ.all_cells.get_cells()
             if c.polygon is not None and not c.polygon.is_empty]
    if not cells:
        return 0
    enclosed_only = organ.FUSE_ENCLOSED_GAPS_ONLY
    unions: Dict[str, Any] = {}
    gaps = find_gaps(organ, sliver_width=sliver_width, min_area=min_area, enclosed_only=enclosed_only,
                     unions=unions)
    if not gaps:
        return 0

    med_area = float(np.median([c.polygon.area for c in cells]))
    med_d = 2.0 * np.sqrt(med_area / np.pi) if med_area > 0 else 0.0
    eps = max(med_d * 1e-3, 1e-9)
    # How far to dilate each reported gap to recover the true (un-opened) hole.
    recover = 1.5 * sliver_width if sliver_width else 0.25 * med_d

    covered = unions["covered"]                     # unary_union of the same cells' polygons, in the same order
    raw = organ.generate_base_shape().difference(covered)
    for v in organ._empty_void_polygons():
        raw = raw.difference(v)
    if enclosed_only:
        # Clip the recoverable area too. Without this the ``recover`` dilation
        # below reconnects an enclosed hole near the surface to the ribbon, and
        # the cell fused with it smears along the ribbon anyway — defeating the
        # filter find_gaps just applied.
        if "mass" not in unions:
            unions["mass"] = _filled_mass(covered)
        raw = raw.intersection(unions["mass"])

    tree = STRtree([c.polygon for c in cells])
    fused = 0
    for g in gaps:
        hole = raw.intersection(g.buffer(recover)) if recover > 0 else g
        if hole.is_empty:
            hole = g
        probe = hole.buffer(eps)
        cand = [cells[i] for i in tree.query(probe) if cells[i].polygon.intersects(probe)]
        if not cand:
            continue

        # Dominant bordering tissue = the zone the gap belongs to; then the cell
        # of that zone with the longest shared border (its nearest cell).
        by_type: Dict[str, float] = defaultdict(float)
        best: Dict[str, Any] = {}
        for c in cand:
            # .boundary (not .exterior) so a MultiPolygon cell doesn't crash.
            shared = c.polygon.boundary.intersection(probe).length
            by_type[c.type] += shared
            if c.type not in best or shared > best[c.type][0]:
                best[c.type] = (shared, c)
        if by_type and max(by_type.values()) > 0:
            zone = max(by_type, key=by_type.get)
            target = best[zone][1]
        else:                                   # no shared border -> nearest overall
            target = min(cand, key=lambda c: c.polygon.distance(hole))

        merged = unary_union([target.polygon, hole])
        if merged.geom_type != "Polygon":
            parts = [p for p in merged.geoms if p.geom_type == "Polygon"]
            merged = max(parts, key=lambda p: p.area) if parts else target.polygon
        target.polygon = merged
        fused += 1

    recalculate_cell_properties(organ.all_cells.cells)
    if organ._cells_gdf is not None:
        cell_dicts = [c.cell_to_dict() for c in organ.all_cells.cells]
        for i, c in enumerate(organ.all_cells.cells):
            cell_dicts[i]["geometry"] = c.polygon
        organ._cells_gdf = gpd.GeoDataFrame(cell_dicts)
    return fused
