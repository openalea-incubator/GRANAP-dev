"""
The seeds, converted (devplan_metafspm.md, §44, B3): GRANAP's CellGenerator.generate_cells_info and
resolve_cell_border_overlaps (generate_cell.py:115, 283), copied, the groups' convex-hull footprints built in bulk
(shapely's multipoints, convex_hull and buffer: the same GEOS operations on each group, from the same coordinates) instead
of a MultiPoint, a hull and a buffer per group. Every other line is GRANAP's.
"""
from typing import Any, Dict, List

import numpy as np
import shapely as sp
from shapely.geometry import Point
from shapely.strtree import STRtree

from openalea.granap.cell_class import Cell
from openalea.granap.cell_manager import CellManager
from openalea.granap.generate_cell import CellGenerator
from openalea.granap.geometry_collection import GeometryProcessor


def generate_cells_info(layers_polygons: List[Dict[str, Any]],
                       center: Point, rng=None):
    """
    Generate cell information from layer polygons.

    Args:
        layers_polygons: List of layer polygon dictionaries
        center: Center point for angle/radius calculations

    Returns:
        pd.DataFrame of cells
    """
    all_cells = CellManager()
    id_cell = 1
    id_group = 1

    _rng = rng if rng is not None else np.random
    for i_layer, layer in enumerate(layers_polygons):
        cells_coords = CellGenerator.cells_on_layer(
            layer["polygon"],
            layer["cell_diameter"],
            layer["cell_width"],
            layer.get("shift", 0),
            rng=rng,
        )

        if layer.get("transfusion_type"):
            # Per-seed type assignment: each seed gets its own type and radius.
            # Orientation is computed from the global seed sequence so border
            # ellipses align with the ring, same as the standard path.
            # Border points are clipped to the layer polygon so that no
            # transfusion Voronoi seed escapes into the endodermis ring,
            # which would otherwise distort the endodermis shape.
            next_c = np.roll(cells_coords, -1, axis=0)
            prev_c = np.roll(cells_coords,  1, axis=0)
            seed_axes = np.arctan2(
                next_c[:, 1] - prev_c[:, 1],
                next_c[:, 0] - prev_c[:, 0],
            )
            tt_d = layer["tt_diameter"]
            tp_d = layer["tp_diameter"]
            p_tt = layer["p_tt"]
            layer_poly = layer["polygon"]
            sp.prepare(layer_poly)  # speeds the repeated covers() queries below

            for i, cell_coord in enumerate(cells_coords[1:]):
                seed_type = (
                    "transfusion tracheid"
                    if _rng.random() < p_tt
                    else "transfusion parenchyma"
                )
                d = tt_d if seed_type == "transfusion tracheid" else tp_d
                axis = seed_axes[i + 1]
                border_pts = GeometryProcessor.draw_ellipse(
                    cell_coord, axis, d * 0.7 / 2, d * 0.7 / 2, n_points=10
                )
                cell_angle = np.arctan2(
                    cell_coord[1] - center.y, cell_coord[0] - center.x
                )
                cell_radius = np.sqrt(
                    (cell_coord[0] - center.x) ** 2
                    + (cell_coord[1] - center.y) ** 2
                )
                # Vectorised covers(): one C call for the whole ring of border
                # points instead of constructing a shapely Point per point.
                ring = border_pts[1:]
                covered = sp.covers(layer_poly, sp.points(ring[:, 0], ring[:, 1]))
                for border_point, is_covered in zip(ring, covered):
                    if not is_covered:
                        continue
                    new_cell = Cell(
                        type=seed_type,
                        x=border_point[0],
                        y=border_point[1],
                        diameter=d,
                        id_cell=id_cell,
                        id_layer=i_layer,
                        id_group=id_group,
                        angle=cell_angle,
                        radius=cell_radius,
                        area=np.pi * (d / 2) ** 2,
                    )
                    all_cells.add_cell(new_cell)
                    id_cell += 1
                id_group += 1
            continue  # skip standard border generation for this layer

        if layer["cell_width"] != 0:
            layer_cell_borders = CellGenerator.cell_border(
                cells_coords,
                layer["cell_width"] * 0.7,
                layer["cell_diameter"] * 0.7,
                n_points=layer.get("n_points"),
            )
        else:
            layer_cell_borders = CellGenerator.cell_border(
                cells_coords,
                layer["cell_diameter"] * 0.7,
                layer["cell_width"] * 0.7,
                n_points=layer.get("n_points"),
            )

        # The next-inner layer polygon is loop-invariant — fetch and prepare
        # it once (prepared geometry speeds the per-group contains() queries).
        next_inner = (
            layers_polygons[i_layer + 1]["polygon"]
            if i_layer + 1 < len(layers_polygons)
            else None
        )
        if next_inner is not None:
            sp.prepare(next_inner)

        for i, cell_coord in enumerate(cells_coords[1:]):
            if layer["name"] == "parenchyma":
                new_cell = Cell(
                    type=layer["name"],
                    x=cell_coord[0],
                    y=cell_coord[1],
                    diameter=layer["cell_diameter"],
                    id_cell=id_cell,
                    id_layer=i_layer,
                    id_group=id_group,
                    angle=np.arctan2(cell_coord[1] - center.y,
                                      cell_coord[0] - center.x),
                    radius=np.sqrt((cell_coord[0] - center.x)**2 +
                                    (cell_coord[1] - center.y)**2),
                    area=np.pi * (layer["cell_diameter"] / 2)**2,
                    protect_shape=layer.get("protect_shape", False),
                )
                all_cells.add_cell(new_cell)
                id_cell += 1
                id_group += 1
            else:
                cell_border_points = layer_cell_borders[i]
                ring = cell_border_points[1:]
                # Discard border points that bleed into the next inner layer —
                # they would create seeds inside the wrong tissue zone.  One
                # vectorised contains_xy() call replaces a shapely Point per point.
                if next_inner is not None and len(ring):
                    bleeds = sp.contains_xy(next_inner, ring[:, 0], ring[:, 1])
                else:
                    bleeds = np.zeros(len(ring), dtype=bool)
                for border_point, bleed in zip(ring, bleeds):
                    if bleed:
                        continue
                    new_cell = Cell(
                        type=layer["name"],
                        x=border_point[0],
                        y=border_point[1],
                        diameter=layer["cell_diameter"],
                        id_cell=id_cell,
                        id_layer=i_layer,
                        id_group=id_group,
                        angle=np.arctan2(cell_coord[1] - center.y,
                                          cell_coord[0] - center.x),
                        radius=np.sqrt((cell_coord[0] - center.x)**2 +
                                        (cell_coord[1] - center.y)**2),
                        area=np.pi * (layer["cell_diameter"] / 2)**2,
                        protect_shape=layer.get("protect_shape", False),
                    )
                    all_cells.add_cell(new_cell)
                    id_cell += 1
                id_group += 1

    all_cells = resolve_cell_border_overlaps(all_cells)          # B3: the footprints in bulk
    return all_cells


def resolve_cell_border_overlaps(all_cells: CellManager) -> CellManager:
    """
    Remove cell_border points from lower-priority id_groups that overlap
    with higher-priority ones.

    Priority order:
      1. Higher id_layer wins (inner layers have precedence over outer).
      2. Within the same id_layer, higher id_group wins.

    A convex-hull footprint is built from each group's cell positions.
    Cells from a lower-priority group whose position falls inside a
    higher-priority footprint are removed.
    """

    if not all_cells.cells:
        return all_cells

    # --- build group metadata ----------------------------------------
    groups: dict = {}  # id_group -> {id_layer, indices, poly}
    for idx, cell in enumerate(all_cells.cells):
        g = cell.id_group
        if g not in groups:
            groups[g] = {
                "id_layer": cell.id_layer,
                "cell_diameter": cell.diameter,
                "id_group": g,
                "indices": [],
            }
        groups[g]["indices"].append(idx)

    # build a convex-hull footprint for each group (all at once: the same GEOS multipoint, hull and buffer per group)
    cells_list = all_cells.cells
    metas = list(groups.values())
    hulled = [m for m in metas if len(m["indices"]) >= 3]
    if hulled:
        index = np.concatenate([np.full(len(m["indices"]), k) for k, m in enumerate(hulled)])
        coords = np.array([(cells_list[i].x, cells_list[i].y) for m in hulled for i in m["indices"]], dtype=float)
        footprints = sp.buffer(sp.convex_hull(sp.multipoints(coords, indices=index)),
                               np.array([m["cell_diameter"] * 0.2 for m in hulled]), quad_segs=16)   # the method's default
        for m, footprint in zip(hulled, footprints):
            m["poly"] = footprint
    for meta in metas:
        if len(meta["indices"]) >= 3:
            continue
        pts = [(cells_list[i].x, cells_list[i].y) for i in meta["indices"]]
        if pts:
            r = cells_list[meta["indices"][0]].diameter / 2
            meta["poly"] = Point(pts[0]).buffer(r)
        else:
            meta["poly"] = None

    # --- sort groups from highest to lowest priority -----------------
    sorted_groups = sorted(
        groups.values(),
        key=lambda m: (m["id_layer"], m["id_group"]),
        reverse=True,
    )

    # --- remove overlapping lower-priority cells --------------------
    # Build STRtree over all group footprints so we only run the
    # expensive contains() check for geometrically nearby pairs.
    valid_groups = [m for m in sorted_groups if m["poly"] is not None]
    if not valid_groups:
        return all_cells

    polys = [m["poly"] for m in valid_groups]
    tree = STRtree(polys)

    ids_to_remove: set = set()
    cells = all_cells.cells
    for i, high in enumerate(valid_groups):
        high_poly = polys[i]
        sp.prepare(high_poly)  # repeated contains() queries below
        # query returns indices whose bounding boxes overlap high_poly
        for j in tree.query(high_poly):
            if j <= i:          # only lower-priority groups
                continue
            if not high_poly.intersects(polys[j]):
                continue
            idxs = [idx for idx in valid_groups[j]["indices"]
                    if idx not in ids_to_remove]
            if not idxs:
                continue
            # One vectorised contains_xy() for the whole low group instead of
            # constructing a shapely Point per candidate cell.
            xs = np.fromiter((cells[idx].x for idx in idxs), float, len(idxs))
            ys = np.fromiter((cells[idx].y for idx in idxs), float, len(idxs))
            inside = sp.contains_xy(high_poly, xs, ys)
            ids_to_remove.update(idx for idx, ins in zip(idxs, inside) if ins)

    if ids_to_remove:
        all_cells.cells = [
            c for i, c in enumerate(all_cells.cells)
            if i not in ids_to_remove
        ]

    return all_cells
