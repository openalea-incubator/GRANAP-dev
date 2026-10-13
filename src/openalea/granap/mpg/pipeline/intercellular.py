"""
The intercellular spaces, converted (devplan_metafspm.md, §34, B2c).

    apply_intercellular(organ, ics)         GRANAP's Organ._apply_intercellular (organ_class.py:784), the cells'
                                            smoothing in bulk
    seat_air_spaces(cells, hosts, ...)      GRANAP's special_tissues.seat_air_spaces, the hosts carved in bulk and the
                                            cells simplified by the converted simplification (merge.py)

Exact:
  * the smoothing (GeometryProcessor.buffer_polygon with a smooth factor) buffers every cell at once (the same GEOS
    buffer), resamples each ring with GRANAP's own resample_coords, then runs the 10 Laplacian passes on all the
    resampled rings together: the same arithmetic, element by element. A cell whose buffer is not one Polygon goes
    through GRANAP's function;
  * the carving runs the same GEOS difference on each host, in bulk.

Every other line is GRANAP's.
"""
from typing import List

import numpy as np
import shapely
from shapely.geometry import MultiPolygon
from shapely.strtree import STRtree

from openalea.granap.cell_class import Cell
from openalea.granap.geometry_collection import GeometryProcessor
from openalea.granap.mpg.pipeline.merge import _set_polygons, simplify_cells

_ITERATIONS = 10                 # GeometryProcessor.smoothing_polygon's default
_POINTS = 200                    # its resampling


def smoothed_polygons(polygons, smooth_factors) -> list:
    """[GeometryProcessor.buffer_polygon(p, 0, smooth_factor=s)] for each polygon and factor (all > 0), in bulk."""
    buffered = shapely.buffer(np.asarray(polygons, dtype=object), 0., quad_segs=16)
    out = [None] * len(polygons)
    batch, rows, factors = [], [], []
    for i, (polygon, b, s) in enumerate(zip(polygons, buffered, smooth_factors)):
        if b.geom_type != "Polygon" or b.is_empty:
            out[i] = GeometryProcessor.buffer_polygon(polygon, 0, smooth_factor=s)   # GRANAP's own path
            continue
        coords = shapely.get_coordinates(b.exterior)
        resampled = GeometryProcessor.resample_coords(coords, target_n_points=_POINTS)
        if not np.allclose(resampled[0], resampled[-1]):
            out[i] = GeometryProcessor.buffer_polygon(polygon, 0, smooth_factor=s)   # an open ring: GRANAP's
            continue
        batch.append(resampled)
        rows.append(i)
        factors.append(s)
    if batch:
        coords = np.stack(batch).astype(float)                   # (cells, points, 2), closed rings
        s = np.asarray(factors, dtype=float)[:, None, None]
        for _ in range(_ITERATIONS):
            pts = coords[:, :-1]
            prev_pts = np.roll(pts, 1, axis=1)
            next_pts = np.roll(pts, -1, axis=1)
            smoothed_pts = (1 - s) * pts + s * (prev_pts + next_pts) / 2.0
            coords = np.concatenate([smoothed_pts, smoothed_pts[:, :1]], axis=1)
        polygons_out = shapely.polygons(coords)
        for i, p in zip(rows, polygons_out):
            out[i] = p
    return out


def seat_air_spaces(cell_manager, host_cells: List[Cell], air_union, air_faces, *, protect_topology: bool = False,
                    min_area: float = 1e-6) -> None:
    """Carve intercellular lacunae out of their host cells and insert them (GRANAP's, the carving in bulk)."""
    present = [cell for cell in host_cells if cell.polygon is not None]
    if present:
        carved = shapely.difference(np.array([cell.polygon for cell in present], dtype=object), air_union)
        kept = ~shapely.is_empty(carved) & (shapely.area(carved) > min_area)
        _set_polygons(present, [c if k else None for c, k in zip(carved, kept)])

    id_cell = len(cell_manager.cells)
    for face in air_faces:
        id_cell += 1
        lacuna = Cell(
            x=face.centroid.x,
            y=face.centroid.y,
            diameter=np.sqrt(face.area / np.pi) * 2,
            id_cell=id_cell,
            id_layer=0,
            id_group=id_cell,
            type="air space",
            polygon=face,
        )
        if protect_topology:
            lacuna.protect_topology = True
            lacuna.protect_shape = True
        cell_manager.cells.append(lacuna)

    cell_manager.cells = simplify_cells(cell_manager.cells)


def apply_intercellular(organ, ics: dict) -> None:
    """Apply one inter_cellular_spaces entry to the relevant tissue cells."""
    tissues = ics.get("tissue", [])
    if isinstance(tissues, str):
        tissues = [tissues]
    if not tissues:
        return

    smoothness = ics.get("smoothness", 0)
    if isinstance(smoothness, (int, float)):
        smoothness_per_tissue = [float(smoothness)] * len(tissues)
    else:
        smoothness_per_tissue = [float(s) for s in smoothness]

    if not any(smoothness_per_tissue):
        return

    all_tissue_cells = []
    cell_smoothness: dict = {}
    for tissue_name, s in zip(tissues, smoothness_per_tissue):
        cells = organ.all_cells.get_cells_by_type(tissue_name)
        for c in cells:
            cell_smoothness[id(c)] = s
        all_tissue_cells.extend(cells)

    tissue_polys = [c.polygon for c in all_tissue_cells if c.polygon is not None]
    if len(tissue_polys) < 2:
        return

    full_union = GeometryProcessor.union_polygons(tissue_polys)
    min_diameter = min(c.diameter for c in all_tissue_cells)
    full_union_buffed = full_union.buffer(-min_diameter * 0.5)

    # The cells' smoothing in bulk: buffer_polygon(polygon, 0, smooth_factor=s) for each (s > 0), the buffer alone
    # for s == 0, as buffer_polygon returns it
    with_polygon = [cell for cell in all_tissue_cells if cell.polygon is not None]
    factors = [cell_smoothness[id(cell)] for cell in with_polygon]
    smooth = [i for i, s in enumerate(factors) if s > 0]
    shrunk_all = [None] * len(with_polygon)
    for i, p in zip(smooth, smoothed_polygons([with_polygon[i].polygon for i in smooth], [factors[i] for i in smooth])):
        shrunk_all[i] = p
    for i, s in enumerate(factors):
        if not s > 0:
            shrunk_all[i] = with_polygon[i].polygon.buffer(0, resolution=16)
    smoothed = [shrunk for shrunk in shrunk_all if not shrunk.is_empty]

    if not smoothed:
        return

    smoothed_union = GeometryProcessor.union_polygons(smoothed)
    air_region = full_union.difference(smoothed_union)

    if isinstance(air_region, MultiPolygon):
        raw_air_polys = list(air_region.geoms)
    elif air_region.is_empty:
        return
    else:
        raw_air_polys = [air_region]

    r_values = [np.sqrt(p.area / np.pi) for p in tissue_polys]
    tol = float(np.median(r_values)) * 0.05

    air_space_polys = []
    for poly in raw_air_polys:
        if poly.intersects(full_union_buffed):
            simplified = poly.simplify(tol, preserve_topology=True)
            if not simplified.is_empty and simplified.area > 1E-6:
                air_space_polys.append(simplified)

    if not air_space_polys:
        return

    # Centre any lopsided pocket: where a neighbour meets it with a near-straight
    # or reflex corner (which doesn't retreat under the corner-rounding, leaving a
    # flat wall), clip that corner into the pocket so every side retreats and the
    # pocket sits centered on the junction node.
    nb_polys = [c.polygon for c in all_tissue_cells if c.polygon is not None]
    if nb_polys:
        nb_tree = STRtree(nb_polys)
        air_space_polys = [organ._center_air_pocket(p, nb_tree, nb_polys)
                           for p in air_space_polys]

    air_union = GeometryProcessor.union_polygons(air_space_polys)
    seat_air_spaces(organ.all_cells, all_tissue_cells, air_union, air_space_polys,
                    protect_topology=organ.PROTECT_AIR_TOPOLOGY)
