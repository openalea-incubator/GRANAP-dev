"""Special-tissue placements — the organ-agnostic half of the vocabulary.

Some anatomical features are *cell-relative* / post-fill: they are carved into or
seeded around cells that already exist (resin ducts, stomata, the metaxylem
sheath, companion cells, intercellular spaces).  They do not fit the shape-first
"region then fill" model in :mod:`tissue_class` — there is no region to fill,
only existing cells to add to or replace.

This module collects those placements as **named, parameterised functions** so an
organ's recipe can call them via ``recipe.special(name, fn, ...)`` instead of
each organ re-implementing the carve / build / re-insert boilerplate.  The
geometry of *where* a feature goes stays with the organ (it is organ-specific);
these functions take that precomputed geometry and do the cell placement.

    carve_and_insert : remove cells under a mask, insert new cells, recompute
    seat_air_spaces  : carve lacunae out of host cells, insert them as air spaces
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

# Minimum number of cells placed around each duct's sheath ring, regardless
# of how few its own tangential cell size would otherwise fit -- matches the
# real anatomy (resin_duct.png panel G shows ~10 sheath cells per duct).
_DUCT_SHEATH_MIN_CELLS: int = 10


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


def seat_air_spaces(
    cell_manager: CellManager,
    host_cells: List[Cell],
    air_union,
    air_faces,
    *,
    protect_topology: bool = False,
    min_area: float = 1e-6,
) -> None:
    """Carve intercellular lacunae out of their host cells and insert them.

    The shared post-fill pattern for intercellular air spaces: the lacunae have
    already been computed (``air_union`` is their union; ``air_faces`` are the
    individual polygons to become cells).  Each ``host_cells`` polygon is carved
    with ``air_union`` (so the lacuna boundary and the carved host boundary stay
    vertex-for-vertex identical), each face is inserted as an ``"air space"`` cell
    using the standard labelling (``id_layer=0``, ``id_group=id_cell``), and the
    cell set is re-simplified.

    ``air_faces`` and ``air_union`` are passed separately on purpose: callers may
    insert the *simplified* air polygons individually while carving with their
    union, or insert the union's connected components — the caller decides what a
    single lacuna cell is.

    When ``protect_topology`` is True the inserted lacunae are flagged so every
    vertex is kept as part of a crooked wall (see ``CellGenerator._build_topology``);
    used for small mid-wall lacunae (the needle mesophyll rhombi) whose off-wall
    tips would otherwise let a neighbour be straightened across the notch.
    """
    for cell in host_cells:
        if cell.polygon is None:
            continue
        carved = cell.polygon.difference(air_union)
        if not carved.is_empty and carved.area > min_area:
            cell.polygon = carved
        else:
            cell.polygon = None

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

    cell_manager.cells = CellGenerator.simplify_cells(cell_manager.cells)


def place_resin_duct(
    cell_manager: CellManager,
    duct_data,
    layer_index: int,
) -> None:
    """Place resin ducts: an outer sheath ring, an inner epithelium ring, the
    central lumen, and (where the organ's geometry calls for one) an outer
    transition ring, for each duct.

    Modeled inside-out: lumen (L) -> epithelium (Ep, 1 cell layer, thin-walled/secretory)
    -> sheath (Sh, 1 cell layer, thicker-walled), embedded in the mesophyll.
    The optional transition ring is a ring of ordinary host-tissue
    ("mesophyll") cells sized down to bridge the sheath and the coarse
    surrounding tissue -- see NeedleAnatomy._duct_zone_data, which decides
    per-duct whether one is needed and how big its cells are.

    Ring cells are elliptical, sized independently along the ring
    (tangential, "cell_width"/"sheath_cell_width") and across it (radial,
    "cell_diameter"/"sheath_cell_diameter") -- 0 for either width field means
    isotropic (falls back to the matching radial diameter), the same
    convention CellGenerator.cells_on_layer uses for ordinary ring layers.
    CellGenerator.cell_border's own parameter names are misleading here:
    what matters is argument *position* -- 1st = tangential (major axis,
    aligned along the boundary's local tangent), 2nd = radial (minor axis).
    The transition ring is isotropic (its own cell size, both axes).
    """
    if not duct_data:
        return

    duct_cells: List[Cell] = []
    id_cell  = len(cell_manager.cells) + 1
    id_group = cell_manager.get_last_id_group() + 1

    for duct in duct_data:
        center = duct["center"]

        # Outer-to-inner: transition ring (when present) first, then sheath
        # ring, then epithelium ring. Sizes come from this duct's own
        # (possibly scaled-to-fit) dict entries. The transition ring is
        # tagged as ordinary "mesophyll" -- the host tissue it's fitted
        # into -- rather than a new tissue type, exactly like
        # vascular_bundle.py's outer bundle sheath.
        ring_specs = []
        if duct.get("transition_ring") is not None:
            t = duct["transition_cell_size"]
            ring_specs.append(("mesophyll", "transition_ring", t, t))
        ring_specs += [
            ("resin duct sheath",     "sheath_ring",     duct["sheath_cell_width"], duct["sheath_cell_diameter"]),
            ("resin duct epithelium", "epithelium_ring", duct["cell_width"],        duct["cell_diameter"]),
        ]

        for cell_type, ring_key, tangential, radial in ring_specs:
            ring_poly = duct[ring_key]
            # Clamp the tangential cell size so at least
            # _DUCT_SHEATH_MIN_CELLS cells fit around this ring, regardless
            # of how few its own tangential size would otherwise produce
            # (matches resin_duct.png panel G, ~10 sheath cells per duct).
            # +1 because the resampled ring's first point duplicates its
            # last (closed boundary) and is dropped below ([1:]), so
            # target_n_points must be one more than the cell count wanted.
            tangential = min(tangential, ring_poly.length / (_DUCT_SHEATH_MIN_CELLS + 1))
            x, y   = ring_poly.exterior.coords.xy
            coords = np.column_stack((x, y))
            coords = GeometryProcessor.resample_coords(
                coords,
                target_n_points=np.round(ring_poly.length / tangential).astype(int),
            )
            # Anisotropic cell's metadata "diameter" = mean of its two axes
            # (matches the vascular grid's own precedent: xylem_cell_diameter
            # in vascular_elements_in_ellipses).
            mean_diameter = (tangential + radial) / 2
            for border in CellGenerator.cell_border(coords, tangential, radial)[1:]:
                id_group += 1
                for cell_coord in border:
                    duct_cells.append(Cell.radial(
                        cell_type, cell_coord[0], cell_coord[1], mean_diameter,
                        id_group, center, id_cell=id_cell, id_layer=layer_index,
                    ))
                    id_cell += 1

        # inner lumen cells along the canal
        canal_center = duct["canal"].centroid
        x, y   = duct["canal"].exterior.coords.xy
        coords = GeometryProcessor.resample_coords(np.column_stack((x, y)), target_n_points=_CANAL_RESAMPLE_PTS)
        id_group += 1
        for coord in coords[1:]:
            duct_cells.append(Cell.radial(
                "duct", coord[0], coord[1], duct["lumen_diameter"],
                id_group, canal_center, id_cell=id_cell, id_layer=layer_index,
            ))
            id_cell += 1

    carve_and_insert(cell_manager, [d["carve"] for d in duct_data], duct_cells)


def place_stomata(
    cell_manager: CellManager,
    stomata_geoms,
    sp: dict,
    cell_diam: float,
) -> None:
    """Place stomata: two guard cells, a substomatal chamber and a pore each.

    ``stomata_geoms`` is a list of ``(carve_poly, gc1, gc2, chamber, pore)`` --
    or, for sunken stomata, ``(carve_poly, gc1, gc2, chamber, pore, caps)``
    where ``caps`` are epidermis cells arching over the guard-cell pit -- as
    computed by the organ; ``sp`` is the stomata param dict; ``cell_diam`` the
    epidermis cell diameter (sets the carve buffer and inset).
    """
    organ_specific_cells = CellManager()
    stomata_carve_polys: list = []
    id_stomata = len(cell_manager.cells) + 1
    i_cell     = id_stomata

    for geom in stomata_geoms:
        carve_poly, gc1, gc2, chamber, pore = geom[:5]
        caps = geom[5] if len(geom) > 5 else []
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
                    protect_topology=(cell_type == "air space"),
                ))

        for cap_poly in caps:
            poly = cap_poly.buffer(-cell_diam / 5)
            if poly.is_empty or poly.area <= 0 or not hasattr(poly, "exterior") or poly.exterior is None:
                continue
            coords = GeometryProcessor.resample_coords(
                np.column_stack(poly.exterior.coords.xy), 20
            )
            id_stomata += 1
            for i_coord in coords:
                i_cell += 1
                organ_specific_cells.cells.append(Cell(
                    x=i_coord[0], y=i_coord[1],
                    diameter=np.sqrt(poly.area / np.pi) * 2,
                    id_cell=i_cell, id_group=id_stomata,
                    type="epidermis",
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
