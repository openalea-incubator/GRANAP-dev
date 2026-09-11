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
    carve_cells      : subtract a mask from host polygons, keep the survivors
    absorb_residual  : union the uncovered slivers of a region into its cells
    weld_points_onto_rings : insert foreign vertices into existing cell rings

The cell-relative nature is why these stay below the tissue abstraction — see
``doc/tissue_refactor.md`` ("the cells-first engine").
"""

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

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


def _largest_part(geom):
    """The biggest Polygon of a possibly-multipart geometry (None if there is none)."""
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, MultiPolygon):
        parts = [g for g in geom.geoms if not g.is_empty]
        return max(parts, key=lambda g: g.area) if parts else None
    return geom if isinstance(geom, Polygon) else None


def carve_cells(
    host_cells: Sequence[Cell],
    mask_union,
    *,
    min_area: float = 1e-6,
) -> List[Cell]:
    """Subtract ``mask_union`` from each host polygon; return the survivors.

    The load-bearing half of :func:`seat_air_spaces`, factored out for callers
    that already own the cells occupying the mask: carving with a *shared*
    ``mask_union`` is what keeps the mask boundary and the carved host boundary
    vertex-for-vertex identical, which is what lets
    ``CellGenerator._build_topology`` pair the two flanking cells' walls.  A
    ``difference`` that leaves nothing (or only a sliver below ``min_area``, or a
    non-polygonal remnant) drops the host from the returned list rather than
    leaving a ``None`` polygon behind.
    """
    survivors: List[Cell] = []
    for cell in host_cells:
        if cell.polygon is None or cell.polygon.is_empty:
            continue
        carved = _largest_part(cell.polygon.difference(mask_union))
        if carved is None or carved.area < min_area:
            continue
        if not carved.is_valid:
            carved = _largest_part(carved.buffer(0))
            if carved is None or carved.area < min_area:
                continue
        cell.polygon = carved
        survivors.append(cell)
    return survivors


#: A residual piece smaller than this fraction of the median host cell area is
#: boolean noise rather than a gap, and absorbing it does more harm than good
#: (see :func:`absorb_residual`).
_NOISE_AREA_RATIO = 1e-9


def absorb_residual(
    host_cells: Sequence[Cell],
    region,
    *,
    others: Sequence = (),
    min_area: float = 0.0,
) -> dict:
    """Union whatever of ``region`` no cell covers into the cell beside it.

    A tessellation grafted into an existing region never covers it perfectly:
    border seeds, sliver filtering and clipping leave thin uncovered strips.
    Every such strip is a stretch of boundary that only one cell owns, and a
    one-cell wall is not merely a missing connection -- MECHA's
    ``identify_border_walls_junctions`` files a single-reference wall whose owner
    is not epidermis under ``border_aerenchyma``, i.e. it becomes a **gas-space
    boundary in the middle of the tissue**.  So the region has to be tiled
    exactly, not approximately.

    Each uncovered piece is unioned into the ``host_cells`` polygon sharing the
    most boundary with it, but only when the union comes back as one valid,
    simple, hole-free ``Polygon`` -- an unguarded union readily yields a
    MultiPolygon (piece touching at a point) or a self-touching ring, and an
    invalid ring is worse than the gap it closed: it makes
    ``_build_topology`` emit walls with more than two flanking cells.

    ``others`` are polygons that also count as covering the region (e.g. real
    cells kept inside it) but must not be absorbed into.  Returns a report.

    A boolean ``difference`` of two tessellations always returns a crowd of
    zero-area slivers alongside any genuine gap, and those must be left alone:
    they close nothing, but merging one still inserts vertices into its host's
    ring, and three cells meeting along such a sliver then emit the *same* wall
    key -- a wall with three flanking cells, which is the very defect the
    validity guard below is trying to avoid.  Measured on the Arabidopsis
    section: every seed yields 16-21 pieces below 1e-14 (most exactly 0.0)
    against 0-5 real ones of 1e-8..1e-7, so the two populations are 13 orders of
    magnitude apart and a floor between them is unambiguous.  ``min_area`` is
    therefore raised to at least :data:`_NOISE_AREA_RATIO` of the median host
    cell area.
    """
    covered = [c.polygon for c in host_cells if c.polygon is not None]
    covered.extend(g for g in others if g is not None and not g.is_empty)
    if not covered:
        return {"n_pieces": 0, "absorbed_area": 0.0, "residual_area": region.area,
                "n_unabsorbed": 0}

    host_areas = [c.polygon.area for c in host_cells
                  if c.polygon is not None and c.polygon.area > 0]
    if host_areas:
        min_area = max(min_area, _NOISE_AREA_RATIO * float(np.median(host_areas)))

    residual = region.difference(unary_union(covered))
    if residual.is_empty:
        return {"n_pieces": 0, "absorbed_area": 0.0, "residual_area": 0.0,
                "n_unabsorbed": 0}

    pieces = list(residual.geoms) if isinstance(residual, MultiPolygon) else [residual]
    pieces = [p for p in pieces if isinstance(p, Polygon) and p.area > min_area]

    absorbed, unabsorbed = 0.0, 0
    for piece in pieces:
        probe = piece.buffer(0.0)
        ranked = sorted(
            (
                (cell.polygon.boundary.intersection(probe.boundary).length, i, cell)
                for i, cell in enumerate(host_cells)
                if cell.polygon is not None and not cell.polygon.disjoint(probe)
            ),
            key=lambda t: -t[0],
        )
        for _shared, _i, cell in ranked:
            merged = unary_union([cell.polygon, piece])
            if (
                isinstance(merged, Polygon)
                and merged.is_valid
                and not merged.interiors
                and merged.exterior.is_simple
            ):
                cell.polygon = merged
                absorbed += piece.area
                break
        else:
            unabsorbed += 1

    return {
        "n_pieces": len(pieces),
        "absorbed_area": absorbed,
        "residual_area": residual.area - absorbed,
        "n_unabsorbed": unabsorbed,
    }


#: An edge shorter than this fraction of the median edge is numerically
#: degenerate -- two vertices that are the "same" point as far as the anatomy is
#: concerned -- and must not be allowed to set a tolerance (see
#: :func:`_ring_edge_stats`).
_DEGENERATE_EDGE_RATIO = 1e-6


def _ring_edge_stats(rings: List[np.ndarray]) -> float:
    """Shortest *non-degenerate* edge across a set of rings (``inf`` if none).

    Not simply the shortest non-zero edge.  A Voronoi tessellation routinely
    produces a pair of near-coincident vertices, and its minimum edge length is
    therefore a random variable with a heavy tail toward zero: measured on the
    ``CellSetOrgan`` donor, the shortest edge is ~1e-5 at most seeds but 1e-19 at
    ``seed=1``.  Feeding that to :func:`conform_boundaries` collapses its
    tolerance below the floating-point noise floor and silently disables the
    whole weld/splice pass.  Edges that short are two copies of one point, so
    they carry no information about the geometry's real scale -- discard them and
    report the shortest edge that does.
    """
    shortest = np.inf
    for ring in rings:
        if len(ring) < 2:
            continue
        diffs = np.diff(ring, axis=0)
        lengths = np.hypot(diffs[:, 0], diffs[:, 1])
        lengths = lengths[lengths > 0]
        if not lengths.size:
            continue
        lengths = lengths[lengths > _DEGENERATE_EDGE_RATIO * np.median(lengths)]
        if lengths.size:
            shortest = min(shortest, float(lengths.min()))
    return shortest


def conform_boundaries(
    target_cells: Sequence[Cell],
    foreign_cells: Sequence[Cell],
    *,
    tol: Optional[float] = None,
) -> dict:
    """Make a grafted tessellation and the cells it was cut against agree exactly.

    :func:`weld_points_onto_rings` only helps when the graft's cut points already
    lie on the target's edges to floating-point precision.  That holds when the
    graft was clipped against those very edges and nothing moved afterwards --
    but not in general, and a cut point that is merely *near* an edge leaves the
    two sides keying different walls, which costs the interface its
    ``membrane``/``plasmodesmata`` edges and turns each orphaned wall into a
    gas-space boundary inside MECHA.

    This is the unconditional version.  For every foreign vertex ``P`` within
    ``tol`` of a target ring it either

    * **welds** -- ``P`` is within ``tol`` of an existing target vertex, so that
      vertex becomes the shared coordinate and the target ring is left alone; or
    * **splices** -- ``P`` projects onto the interior of a target edge, so the
      *projected foot* is inserted into the target ring.

    Either way the same coordinate is then written back into the foreign ring, so
    both sides carry a bit-identical vertex and ``CellGenerator._build_topology``
    clusters them at distance zero rather than relying on its snap tolerance.
    Splicing inserts a point that is by construction collinear with the edge it
    splits, so the target's geometry is unchanged -- the digitized cells keep
    their measured areas, which is asserted in the tests.

    ``tol`` defaults to a fifth of the shortest target edge: far enough above
    ``_build_topology``'s own ``snap_tol`` (1 % of the 5th-percentile edge) that a
    spliced point is never re-merged with its neighbour, and far enough below the
    shortest edge that a weld can never reach past a neighbouring vertex.  Both
    bounds are data-derived, so this works whatever the units.
    """
    targets = [
        c for c in target_cells
        if isinstance(c.polygon, Polygon) and not c.polygon.is_empty
    ]
    foreigns = [
        c for c in foreign_cells
        if isinstance(c.polygon, Polygon) and not c.polygon.is_empty
    ]
    if not targets or not foreigns:
        return {"welded": 0, "spliced": 0, "tol": 0.0, "moved_area": 0.0}

    rings = [np.asarray(c.polygon.exterior.coords, dtype=float) for c in targets]
    if tol is None:
        # Bound by the finest geometry on *either* side. Deriving this from the
        # targets alone is a trap: a graft that packs small cells can have edges
        # an order of magnitude shorter than the digitized outline's, and a
        # tolerance sized for the outline then swallows whole graft edges and
        # produces degenerate cells instead of a clean interface.
        shortest = min(
            _ring_edge_stats(rings),
            _ring_edge_stats(
                [np.asarray(c.polygon.exterior.coords, dtype=float) for c in foreigns]
            ),
        )
        tol = 0.2 * shortest if np.isfinite(shortest) else 1e-9

    points = sorted({
        (float(x), float(y))
        for c in foreigns for x, y in c.polygon.exterior.coords
    })

    # {target index: {segment index: [(t, foot), ...]}}
    inserts: Dict[int, Dict[int, list]] = {}
    subst: Dict[Tuple[float, float], Tuple[float, float]] = {}
    welded = spliced = 0

    for px, py in points:
        best = None                       # (distance, kind, target, seg, t, coord)
        # Every target this point must be spliced into -- not just the nearest.
        # A cut point routinely lies on the edge of *several* rings at once (two
        # cells flanking the same interface, or the cells around a residual patch
        # that ``absorb_residual`` merged away).  Splicing it into only one of
        # them leaves the others keying a longer wall, which is precisely the
        # single-reference wall this function exists to prevent.
        splices: List[Tuple[int, int, float, Tuple[float, float]]] = []
        for ti, ring in enumerate(rings):
            # nearest existing vertex
            d_vert = np.hypot(ring[:, 0] - px, ring[:, 1] - py)
            vi = int(np.argmin(d_vert))
            if d_vert[vi] <= tol and (best is None or d_vert[vi] < best[0]):
                best = (float(d_vert[vi]), "weld", ti, vi, 0.0,
                        (float(ring[vi, 0]), float(ring[vi, 1])))

            # nearest interior point of a segment
            a, b = ring[:-1], ring[1:]
            ab = b - a
            seg2 = (ab ** 2).sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                t = np.where(
                    seg2 > 0,
                    ((px - a[:, 0]) * ab[:, 0] + (py - a[:, 1]) * ab[:, 1]) / seg2,
                    -1.0,
                )
            foot_x = a[:, 0] + t * ab[:, 0]
            foot_y = a[:, 1] + t * ab[:, 1]
            d_seg = np.hypot(foot_x - px, foot_y - py)
            # strictly interior, and not so close to an endpoint that the weld
            # branch should have handled it
            margin = np.divide(tol, np.sqrt(seg2), out=np.full_like(seg2, np.inf),
                               where=seg2 > 0)
            ok = (t > margin) & (t < 1.0 - margin) & (d_seg <= tol)
            if ok.any():
                idx = int(np.flatnonzero(ok)[np.argmin(d_seg[ok])])
                coord = (float(foot_x[idx]), float(foot_y[idx]))
                splices.append((ti, idx, float(t[idx]), coord))
                if best is None or d_seg[idx] < best[0]:
                    best = (float(d_seg[idx]), "splice", ti, idx, float(t[idx]), coord)

        if best is None:
            continue
        _dist, kind, _ti, _idx, _t, coord = best
        subst[(px, py)] = coord
        if kind == "weld":
            welded += 1
        else:
            spliced += 1
        # The foot is collinear with the edge it splits, so each insertion leaves
        # that target's area unchanged however many rings the point lands on.
        for ti, idx, t_at, foot in splices:
            inserts.setdefault(ti, {}).setdefault(idx, []).append((t_at, foot))

    # Rebuild the target rings that gained vertices.
    moved_area = 0.0
    for ti, per_segment in inserts.items():
        ring = rings[ti]
        coords: List[Tuple[float, float]] = [(float(ring[0, 0]), float(ring[0, 1]))]
        for i in range(len(ring) - 1):
            for _t, coord in sorted(per_segment.get(i, []), key=lambda item: item[0]):
                if coord != coords[-1]:
                    coords.append(coord)
            coords.append((float(ring[i + 1, 0]), float(ring[i + 1, 1])))
        cell = targets[ti]
        rebuilt = Polygon(coords, [list(r.coords) for r in cell.polygon.interiors])
        if not rebuilt.is_valid:
            rebuilt = _largest_part(rebuilt.buffer(0))
        if rebuilt is None:
            continue
        moved_area = max(moved_area, abs(rebuilt.area - cell.polygon.area))
        cell.polygon = rebuilt

    # Write the agreed coordinates back into the graft.
    for cell in foreigns:
        coords = [
            subst.get((float(x), float(y)), (float(x), float(y)))
            for x, y in cell.polygon.exterior.coords
        ]
        deduped = [coords[0]]
        for point in coords[1:]:
            if point != deduped[-1]:
                deduped.append(point)
        if len(deduped) < 3:
            continue
        if deduped[0] != deduped[-1]:
            deduped.append(deduped[0])
        rebuilt = Polygon(deduped)
        if not rebuilt.is_valid:
            rebuilt = _largest_part(rebuilt.buffer(0))
        if rebuilt is not None:
            cell.polygon = rebuilt

    return {"welded": welded, "spliced": spliced, "tol": tol, "moved_area": moved_area}


def _weld_ring(
    coords: List[Tuple[float, float]],
    points: np.ndarray,
    tol: float,
) -> Tuple[List[Tuple[float, float]], int]:
    """Insert every ``points`` row that lies strictly inside an edge of ``coords``.

    ``coords`` is a closed ring.  Returns the new ring and how many vertices were
    inserted.  A point within ``tol`` of an existing vertex is skipped -- it is
    already shared.
    """
    if len(coords) < 2 or points.size == 0:
        return coords, 0

    out: List[Tuple[float, float]] = [coords[0]]
    inserted = 0
    tol2 = tol * tol

    for i in range(len(coords) - 1):
        ax, ay = coords[i]
        bx, by = coords[i + 1]
        dx, dy = bx - ax, by - ay
        seg2 = dx * dx + dy * dy
        if seg2 > 0.0:
            # Projection parameter of every candidate point onto this edge.
            t = ((points[:, 0] - ax) * dx + (points[:, 1] - ay) * dy) / seg2
            px = ax + t * dx
            py = ay + t * dy
            d2 = (points[:, 0] - px) ** 2 + (points[:, 1] - py) ** 2
            # Strictly interior: far enough from both endpoints that the point is
            # a genuinely new vertex rather than one already shared.
            margin = tol / np.sqrt(seg2)
            hit = (d2 <= tol2) & (t > margin) & (t < 1.0 - margin)
            if hit.any():
                order = np.argsort(t[hit])
                for row in points[hit][order]:
                    out.append((float(row[0]), float(row[1])))
                    inserted += 1
        out.append((bx, by))

    return out, inserted


def weld_points_onto_rings(
    cells: Iterable[Cell],
    points: Sequence[Tuple[float, float]],
    *,
    tol: float = 1e-9,
) -> int:
    """Insert foreign vertices into the rings of cells whose edges pass through them.

    Needed whenever a fresh tessellation is grafted against an *existing* cell
    boundary: the graft's cut points land in the middle of a neighbour's edge, so
    the neighbour has no vertex there.  ``CellGenerator._build_topology`` then
    keys the neighbour's wall between one pair of junctions and the graft's walls
    between different pairs, and ``NetworkExporter``/``AnatomyWriter`` never pair
    them -- the interface silently loses its ``membrane``/``plasmodesmata``
    edges.  Inserting the cut points explicitly is deliberate: GEOS overlay
    noding would do it as a side effect, but this repo pins GEOS precisely
    because overlay behaviour is load-bearing, so relying on that side effect
    would be fragile across builds.

    ``tol`` is an **exactness** threshold, not a snapping radius: the point is
    inserted verbatim, so a tolerance large enough to catch a point that is
    merely *near* the edge moves the target's boundary by that much and leaves
    the inserted vertex not quite equal to the graft's own -- which breaks the
    pairing it was meant to create.  Keep it at floating-point scale.  Cut points
    produced by intersecting against the target's own boundary satisfy that by
    construction.  (Welding points that are only approximately on the edge would
    need the projected foot inserted on both sides instead; this function
    deliberately does not do that.)

    Returns the number of vertices inserted -- ``0`` means nothing needed welding
    (or, if the caller expected otherwise, that the interface is not shared).
    """
    pts = np.asarray([(float(x), float(y)) for x, y in points], dtype=float)
    if pts.size == 0:
        return 0

    targets = [
        cell for cell in cells
        if isinstance(cell.polygon, Polygon) and not cell.polygon.is_empty
    ]
    total = 0
    for cell in targets:
        poly = cell.polygon
        shell, n = _weld_ring(list(poly.exterior.coords), pts, tol)
        holes = []
        for interior in poly.interiors:
            hole, k = _weld_ring(list(interior.coords), pts, tol)
            holes.append(hole)
            n += k
        if not n:
            continue
        welded = Polygon(shell, holes)
        if not welded.is_valid:
            welded = _largest_part(welded.buffer(0))
            if welded is None:
                continue
        cell.polygon = welded
        total += n
    return total
