"""Special-tissue placements — the organ-agnostic half of the vocabulary.

Some anatomical features are *cell-relative* / post-fill: they are carved into or
seeded around cells that already exist (resin ducts, stomata, the metaxylem
sheath, companion cells, intercellular spaces).  They do not fit the shape-first
"region then fill" model in :mod:`tissue_builder` — there is no region to fill,
only existing cells to add to or replace.

This module collects those placements as **named, parameterised functions** so an
organ's recipe can call them via ``recipe.special(name, fn, ...)`` instead of
each organ re-implementing the carve / build / re-insert boilerplate.  The
geometry of *where* a feature goes stays with the organ (it is organ-specific);
these functions take that precomputed geometry and do the cell placement.

    carve_and_insert : remove cells under a mask, insert new cells, recompute
    place_resin_duct : parenchyma ring + inner lumen for each resin duct
    place_stomata    : guard cells + substomatal chamber + pore for each stoma
    consider_as_cell : collapse a region into a single cell

The cell-relative nature is why these stay below the tissue abstraction — see
``doc/tissue_refactor.md`` ("the cells-first engine").
"""

from typing import List, Optional

import numpy as np

from openalea.granap.cell_class import Cell
from openalea.granap.cell_manager import CellManager
from openalea.granap.generate_cell import CellGenerator
from openalea.granap.geometry_collection import GeometryProcessor

# Number of resampled points for the inner lumen (canal) polygon of a resin duct.
_CANAL_RESAMPLE_PTS: int = 15


def carve_and_insert(
    cell_manager: CellManager,
    carve_polygons,
    new_cells,
    *,
    buffer: float = 0.0,
    recalc: bool = True,
) -> None:
    """Remove cells under each carve polygon, insert ``new_cells``, recompute.

    The shared post-fill structural pattern: a feature carves space out of the
    surrounding tissue (``carve_polygons``, optionally each buffered outward by
    ``buffer``), then its own cells are added and the manager's derived
    properties are recomputed.

    Build ``new_cells`` *before* calling this (their ids are usually derived from
    the pre-removal cell counts).
    """
    for poly in carve_polygons:
        cell_manager.remove_cells_by_polygon(poly.buffer(buffer) if buffer else poly)
    cell_manager.extend_cells(new_cells)
    if recalc:
        cell_manager.recalculate_cell_properties()


def place_resin_duct(
    cell_manager: CellManager,
    duct_data,
    rdp: dict,
    layer_index: int,
) -> None:
    """Place resin ducts: a parenchyma ring + an inner lumen for each duct.

    ``duct_data`` is a list of per-duct dicts (``"outer"``/``"ring"``/``"canal"``/
    ``"ring_center"``) as computed by the organ; ``rdp`` is the resin-duct param
    dict; ``layer_index`` is the id_layer the duct cells belong to.
    """
    if not duct_data:
        return

    duct_cells: List[Cell] = []
    id_cell  = len(cell_manager.cells) + 1
    id_group = cell_manager.get_last_id_group() + 1

    for duct in duct_data:
        ring_center = duct["ring_center"]

        # resin-duct parenchyma cells along the ring
        x, y   = duct["ring"].exterior.coords.xy
        coords = np.column_stack((x, y))
        coords = GeometryProcessor.resample_coords(
            coords,
            target_n_points=np.round(duct["ring"].length / rdp["cell_diameter"]).astype(int),
        )
        for border in CellGenerator.cell_border(coords, rdp["cell_diameter"], rdp["cell_diameter"])[1:]:
            id_group += 1
            for cell_coord in border:
                duct_cells.append(Cell.radial(
                    "resin duct", cell_coord[0], cell_coord[1], rdp["cell_diameter"],
                    id_group, ring_center, id_cell=id_cell, id_layer=layer_index,
                ))
                id_cell += 1

        # inner lumen cells along the canal
        canal_center = duct["canal"].centroid
        x, y   = duct["canal"].exterior.coords.xy
        coords = GeometryProcessor.resample_coords(np.column_stack((x, y)), target_n_points=_CANAL_RESAMPLE_PTS)
        id_group += 1
        for coord in coords[1:]:
            duct_cells.append(Cell.radial(
                "duct", coord[0], coord[1], rdp["diameter"],
                id_group, canal_center, id_cell=id_cell, id_layer=layer_index,
            ))
            id_cell += 1

    carve_and_insert(cell_manager, [d["outer"] for d in duct_data], duct_cells)


def place_stomata(
    cell_manager: CellManager,
    stomata_geoms,
    sp: dict,
    cell_diam: float,
) -> None:
    """Place stomata: two guard cells, a substomatal chamber and a pore each.

    ``stomata_geoms`` is a list of ``(carve_poly, gc1, gc2, chamber, pore)`` as
    computed by the organ; ``sp`` is the stomata param dict; ``cell_diam`` the
    epidermis cell diameter (sets the carve buffer and inset).
    """
    organ_specific_cells = CellManager()
    stomata_carve_polys: list = []
    id_stomata = len(cell_manager.cells) + 1
    i_cell     = id_stomata

    for carve_poly, gc1, gc2, chamber, pore in stomata_geoms:
        stomata_carve_polys.append(carve_poly)

        for raw_poly, cell_type, n_pts in [
            (gc1,     "guard cell", 20),
            (gc2,     "guard cell", 20),
            (chamber, "air space",  10),
        ]:
            poly   = raw_poly.buffer(-cell_diam / 5)
            coords = GeometryProcessor.resample_coords(
                np.column_stack(poly.exterior.coords.xy), n_pts
            )
            id_stomata += 1
            for i_coord in coords:
                i_cell += 1
                organ_specific_cells.cells.append(Cell(
                    x=i_coord[0], y=i_coord[1],
                    diameter=np.sqrt(poly.area / np.pi) * 2,
                    id_cell=i_cell, id_group=id_stomata,
                    type=cell_type,
                ))

        poly   = pore.buffer(-sp["width"] / 4)
        coords = GeometryProcessor.resample_coords(
            np.column_stack(poly.exterior.coords.xy), 10
        )
        id_stomata += 1
        for i_coord in coords:
            i_cell += 1
            organ_specific_cells.cells.append(Cell(
                x=i_coord[0], y=i_coord[1],
                diameter=np.sqrt(poly.area / np.pi) * 2,
                id_cell=i_cell, id_group=id_stomata,
                type="pore",
            ))

    carve_and_insert(
        cell_manager, stomata_carve_polys, organ_specific_cells.cells,
        buffer=cell_diam / 5,
    )


def consider_as_cell(
    cell_manager: CellManager,
    region,
    tag: str,
    *,
    id_layer: int = 0,
    replace: bool = True,
) -> Cell:
    """Collapse a region into a single cell tagged ``tag``.

    The terminal "this whole region is one cell" verb: optionally remove any
    cells already inside ``region`` (``replace``), then insert one cell whose
    polygon *is* ``region`` (a fresh polygon, not Voronoi-derived).  Returns the
    inserted cell.
    """
    if replace:
        cell_manager.remove_cells_by_polygon(region)

    id_cell = len(cell_manager.cells) + 1
    cell = Cell(
        x=region.centroid.x, y=region.centroid.y,
        diameter=np.sqrt(region.area / np.pi) * 2,
        id_cell=id_cell, id_layer=id_layer, id_group=id_cell,
        type=tag, polygon=region,
    )
    cell_manager.cells.append(cell)
    return cell
