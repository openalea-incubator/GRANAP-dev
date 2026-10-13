"""
The cell simplification and the aerenchyma merge, converted (devplan_metafspm.md, §34, B2a).

    simplify_cells(cells)                        GRANAP's CellGenerator.simplify_cells (generate_cell.py:712), the
                                                 polygons rebuilt and validated in bulk
    merge_intercellular_aerenchyma(organ)        GRANAP's Organ.merge_intercellular_aerenchyma (organ_class.py:1052),
                                                 with that simplification and the carving in bulk

Exact: shapely's vectorised functions (linearrings, polygons, is_valid, intersects, difference, area) run the same GEOS
operation on each geometry as the per-object calls, on the same coordinates and in the same order; the cells the
carving removes are removed at once instead of one by one, which leaves the same list. Every other line is GRANAP's.
"""
from collections import defaultdict

import numpy as np
import shapely
from shapely.ops import unary_union
from shapely.strtree import STRtree

from openalea.granap.cell_class import Cell
from openalea.granap.generate_cell import CellGenerator


def _set_polygons(cells, polygons) -> None:
    """cell.polygon = polygon for each, written straight into the MPG's column for the cells that are views."""
    for cell, polygon in zip(cells, polygons):
        if getattr(type(cell), "_mpg_view", False):
            cell._store._prop("polygon")[cell._vid] = polygon
        else:
            cell.polygon = polygon


def simplify_cells(grouped_cells):
    """Simplify cell boundaries by retaining only junction (and protected) vertices; the same list, in place."""
    polys = [c.polygon for c in grouped_cells]
    cell_ids = list(range(len(grouped_cells)))

    # Cells flagged ``protect_shape`` (the needle mesophyll air-space rhombi) keep
    # every boundary vertex here; ``protect_topology`` alone (e.g. the substomatal
    # chamber) only matters later for air-network wiring, not shape simplification.
    protect_ids = {
        idx for idx, cell in enumerate(grouped_cells)
        if getattr(cell, "protect_shape", False)
    }

    cell_vkeys, _, _, junction_set, protected_shape_set = CellGenerator._build_topology(
        polys, cell_ids, protect_ids=protect_ids or None
    )

    if not cell_vkeys:
        return grouped_cells

    # Phase 3 — rebuild each polygon keeping only junction and protected vertices (all rings at once)
    rows, coords, ring_of = [], [], []
    for idx in range(len(grouped_cells)):
        if idx not in cell_vkeys:
            continue
        vkeys = cell_vkeys[idx]
        simplified = [vk for vk in vkeys if (vk in junction_set or vk in protected_shape_set)]

        if len(simplified) < 3:
            simplified = vkeys

        ring_coords = list(simplified)
        if ring_coords[0] != ring_coords[-1]:
            ring_coords.append(ring_coords[0])
        coords.extend(ring_coords)
        ring_of.extend([len(rows)] * len(ring_coords))
        rows.append(idx)

    rings = shapely.linearrings(np.asarray(coords, dtype=float), indices=np.asarray(ring_of, dtype=np.int64))
    new_polys = shapely.polygons(rings)
    invalid = np.flatnonzero(~shapely.is_valid(new_polys))
    for i in invalid:
        new_polys[i] = new_polys[i].buffer(0)
    _set_polygons([grouped_cells[idx] for idx in rows], new_polys)
    return grouped_cells


def merge_intercellular_aerenchyma(organ):
    """Fuse touching air-space cells within the same angular sector, then carve tissue cells."""

    n_files = getattr(organ, '_aerenchyma_n_files', 1)
    start_angle = getattr(organ, '_aerenchyma_start_angle', 0.0)

    def cell_quadrant(cell):
        cell_angle = np.arctan2(cell.y, cell.x) % (2 * np.pi)
        rel = (cell_angle - start_angle) % (2 * np.pi)
        return int(rel / (2 * np.pi / n_files)) % n_files

    seen_ids: set = set()
    merge_pool = []
    for c in list(organ.all_cells.cells):
        if c.type == "air space" and c.polygon is not None:
            oid = id(c)
            if oid not in seen_ids:
                seen_ids.add(oid)
                merge_pool.append(c)

    if merge_pool:
        n_pool = len(merge_pool)
        parent = list(range(n_pool))
        cell_quadrants = [cell_quadrant(c) for c in merge_pool]

        def _find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def _union(i, j):
            ri, rj = _find(i), _find(j)
            if ri != rj:
                parent[ri] = rj

        # Only test pairs whose bounding boxes overlap (an STRtree query);
        pool_polys = [c.polygon for c in merge_pool]
        tree = STRtree(pool_polys)
        for i in range(n_pool):
            poly_i = pool_polys[i]
            quad_i = cell_quadrants[i]
            for j in tree.query(poly_i):
                if j <= i or cell_quadrants[j] != quad_i:
                    continue
                if poly_i.intersects(pool_polys[j]):
                    _union(i, j)

        groups: dict = defaultdict(list)
        for i, c in enumerate(merge_pool):
            groups[_find(i)].append(c)

        fused_cells = []
        for group in groups.values():
            if len(group) == 1:
                fused_cells.append(group[0])
                continue
            fused_polygon = unary_union([c.polygon for c in group])
            fused_cells.append(Cell(
                x=fused_polygon.centroid.x,
                y=fused_polygon.centroid.y,
                diameter=np.sqrt(fused_polygon.area / np.pi) * 2,
                id_cell=min(c.id_cell for c in group),
                id_layer=int(np.ceil(np.mean([c.id_layer for c in group]))),
                id_group=min(c.id_group for c in group),
                type="air space",
                polygon=fused_polygon,
                protect_topology=any(getattr(c, "protect_topology", False) for c in group),
                protect_shape=any(getattr(c, "protect_shape", False) for c in group),
            ))

        organ.all_cells.remove_cells(merge_pool)
        organ.all_cells.cells.extend(fused_cells)

    organ.all_cells.cells = simplify_cells(organ.all_cells.cells)

    tissue = organ.aerenchyma_params.get("tissue")
    air_spaces = organ.all_cells.get_cells_by_type("air space")
    tissue_cells = organ.all_cells.get_cells_by_type(tissue)
    tissue_cells.extend(a for a in air_spaces if a.id_layer == 0)

    air_union = unary_union([a.polygon for a in air_spaces if a.polygon is not None and a.id_layer != 0])

    # The carving, in bulk: a cell without polygon is dropped; a cell the air reaches is carved, and dropped if
    # nothing (or under 1e-6) is left; a cell the air never touches is left intact.
    polygons = np.array([cell.polygon for cell in tissue_cells], dtype=object)
    present = np.array([p is not None for p in polygons], dtype=bool)
    reached = np.zeros(len(tissue_cells), dtype=bool)
    if not air_union.is_empty and present.any():
        reached[present] = shapely.intersects(polygons[present], air_union)
    carved = shapely.difference(polygons[reached], air_union) if reached.any() else np.empty(0, dtype=object)
    kept = ~shapely.is_empty(carved) & (shapely.area(carved) > 1E-6) if carved.size else np.zeros(0, dtype=bool)
    reached_rows = np.flatnonzero(reached)
    _set_polygons([tissue_cells[i] for i in reached_rows[kept]], carved[kept])
    dropped = [tissue_cells[i] for i in np.flatnonzero(~present)] + [tissue_cells[i] for i in reached_rows[~kept]]
    if dropped:
        organ.all_cells.remove_cells(dropped)
