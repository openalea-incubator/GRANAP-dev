"""Hybrid anatomy: real CellSet segmentation + a GRANAP-generated stele.

Why this exists: a CellSet-digitized root section carries everything MECHA
needs *except* a resolved stele.  Segmenting
individual pericycle / phloem / companion cells by hand is not worth the effort,
so the interior of the endodermis is usually drawn as **one annular cell** with
the few visible vessels punched out as holes.  MECHA cannot handle a cell with
holes.

:class:`CellSetOrgan` keeps every real cell verbatim and rebuilds only the
unresolved region, by running a small "donor" :class:`~openalea.granap.root_class.RootAnatomy`
whose base shape *is* the real region outline — so the generated stele inherits
the measured position, size and shape of the real one instead of an idealised
disc at the organ centre — and then grafting the donor's cells in.

The graft is the delicate part.  Three properties have to hold or the export is
quietly wrong rather than visibly broken:

1. **Real outlines must stay exact.**  Imported cells are never smoothed or
   resampled (unlike ``roi_organ.RoiOrgan``), because CellSet stores a
   shared-wall mesh: two adjacent cells list the *same* wall, so their rings
   share vertices verbatim and ``CellGenerator._build_topology`` can pair their
   walls.  They also carry ``protect_shape=True``, or
   ``CellGenerator.simplify_cells`` would rebuild each ring from its junction
   vertices alone and straighten every measured wall into a chord.
2. **The graft interface must be welded.**  Clipping donor cells to the region
   puts cut points in the middle of a real neighbour's edge; without a matching
   vertex on the neighbour the two sides key different walls and the
   endodermis-stele interface loses its ``membrane``/``plasmodesmata`` edges —
   the single most important interface in a root hydraulic model.  See
   :func:`~openalea.granap.special_tissues.weld_points_onto_rings`.
3. **``protect_topology`` must stay False on imported cells.**  ``NetworkExporter``
   folds protected vertices into the wall key as an *ordered* tuple, and the two
   cells flanking a wall traverse it in opposite directions, so a non-empty
   signature would reverse and stop the halves pairing.

Units: CellSet writes micrometres, GRANAP works in millimetres and MECHA's
``GeometryData.im_scale`` defaults to 1000, so the reader's default
``scale=0.001`` makes the whole round trip
(CellSet µm -> GRANAP mm -> ``write_to_xml`` -> MECHA µm) consistent.

Typical use::

    organ = CellSetOrgan("partial_CellSet_brasicaceae.xml", seed=0)
    organ.generate_cells()
    AnatomyWriter(organ).write_to_xml("hybrid.xml")
    # then, in MECHA:  InData(cellset_file="hybrid.xml")
"""

from __future__ import annotations

import copy
import math
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
from shapely.affinity import rotate as _rotate
from shapely.geometry import Polygon
from shapely.ops import unary_union

from openalea.granap.cell_class import Cell
from openalea.granap.cellset_reader import CellSetCell, CellSetSection, read_cellset
from openalea.granap.input_data import OrganInputData
from openalea.granap.layer_class import LayerPolygon
from openalea.granap.organ_class import Organ
from openalea.granap.special_tissues import (
    _largest_part,
    absorb_residual,
    carve_cells,
    conform_boundaries,
)

# Param blocks a stele-only donor must not have: the outer layers would peel
# rings off the region before the stele ever got built, and air spaces inside a
# ~16 um stele are simply wrong anatomy (the root presets do generate them --
# see the "air space" entries in test/test_vascular_regression.py's census).
_DONOR_DROP_PARAMS: Tuple[str, ...] = (
    "epidermis",
    "exodermis",
    "cortex",
    "endodermis",
    "inter_cellular_spaces",
    "aerenchyma",
)


# Fraction of the region radius the vascular core is assumed to occupy before
# it has been measured.  Only used for the first of _build_donor's two passes;
# the second pass uses the real value.
_CORE_RADIUS_GUESS: float = 0.62

# How far past the region rim the donor is grown, as a fraction of the region
# radius, before being clipped back.  A tessellation's outer ring is straightened
# by CellGenerator.simplify_cells into chords between junctions, so it cuts the
# corners of a real (wiggly, digitized) outline and falls short of it -- measured
# 97.2 % coverage with no overshoot, in 13 rim gaps, every one of which would
# become a one-cell wall and hence a gas-space boundary inside MECHA.  Growing
# the donor past the rim and clipping back makes the coverage exact by
# construction; _OVERSHOOT is added to the pericycle thickness as well, so the
# band that survives the clip is the thickness that was asked for.
#
# How badly overshoot=0 fails is platform-dependent, so don't write a test that
# pins it: on x86 it measures coverage 0.9941 with 3 interior border walls at
# seed 0, while on macOS/arm64 the same build tiles exactly (coverage 1.0) and
# reports no defect at all.  Where the straightened chords fall depends on which
# outer-ring vertices become junctions, which is a discrete decision.  Growing
# the donor makes the result exact by construction on every platform, which is
# the point of doing it that way rather than tuning a tolerance.
_OVERSHOOT: float = 0.19

# Nominal pericycle thickness, as a fraction of the region radius.
_PERICYCLE_FRAC: float = 0.26

# Every tag the root recipes use for a vessel.
_XYLEM_TAGS: Tuple[str, ...] = ("xylem", "metaxylem", "protoxylem")

#: Area below which a carved remainder is a boolean-noise fragment rather than a
#: cell.  Used as ``carve_cells``' floor in :meth:`CellSetOrgan._graft`, where
#: dropping a *real* remainder opens a gap in an otherwise exact tiling.
_DEGENERATE_AREA: float = 1e-14


def default_stele_input(
    radius: float,
    *,
    core_radius: Optional[float] = None,
    overshoot: float = 0.0,
) -> OrganInputData:
    """A dicot-root donor configuration scaled to a stele of radius ``radius`` (mm).

    Built from :meth:`OrganInputData.for_dicot_root` — so every param block the
    dicot recipe indexes is present and pydantic-validated — then shrunk to the
    measured region and stripped of everything outside the stele.  A dicot with
    ``n_vascular_peak=2`` gives a xylem plate with phloem in the two valleys and
    a primary cambium ring, which is a recognisable diarch (Arabidopsis-like)
    stele.

    Two radii, because the pipeline uses two:

    * ``radius`` sizes the tissues that ``Organ._build_layer_polygons`` peels off
      the region — the pericycle ring and the stele size gradient.
    * ``core_radius`` sizes the vascular star (xylem / phloem / cambium), whose
      radii are measured from the centre of the polygon the *vascular recipe*
      receives.  That polygon is not the region: ``RootAnatomy._create_central_layers``
      has already peeled the pericycle and one stele ring off it, so it is
      substantially smaller.  Passing the wrong reference here is how the phloem
      valleys end up outside the polygon and silently produce no cells, so
      :meth:`CellSetOrgan._build_donor` measures it and passes it in; the default
      is a rough guess for standalone use.

    ``overshoot`` (mm) is added to the pericycle thickness to compensate for the
    donor being grown past the region rim and clipped back — see
    :data:`_OVERSHOOT`.

    Pass your own :class:`OrganInputData` (or a raw ``List[Dict]``) to
    :class:`CellSetOrgan` as ``donor_input`` to override any of it.
    """
    r = float(radius)
    rs = float(core_radius) if core_radius else _CORE_RADIUS_GUESS * r
    over = float(overshoot)
    data = OrganInputData.for_dicot_root()

    for name in _DONOR_DROP_PARAMS:
        data.remove_param(name)

    data.set_values(
        "stele",
        thickness=2.0 * r,
        cell_diameter=0.22 * r,
        cell_diameter_center=0.30 * r,
        size_gradient_inflection=0.3,
    )
    data.set_values(
        "pericycle",
        cell_diameter=_PERICYCLE_FRAC * r + over,
        cell_width=0.24 * r,
        n_layers=1,
    )
    # The recipe clamps every radius below to the Chebyshev radius of the polygon
    # it is handed (RootAnatomy._xylem_star_region), so overshooting is safe;
    # undershooting is what leaves tissues out.
    data.set_values(
        "xylem",
        n_vascular_peak=2,
        radius_peak_side=1.00 * rs,
        radius_valley_side=0.28 * rs,
        arc_peak_side=0.55 * rs,
        arc_valley_side=0.55 * rs,
        vessel_diameter=0.50 * rs,
        vessel_diameter_min=0.30 * rs,
        vessel_diameter_sd=0.05 * rs,
    )
    data.set_values(
        "phloem",
        sieve_diameter=0.34 * rs,
        cluster_width=0.70 * rs,
        cluster_height=0.45 * rs,
        relative_distance=0.55,
    )
    data.set_values(
        "cambium",
        cell_diameter=0.22 * rs,
        cell_width=0.26 * rs,
        n_layers=1,
        visible_distance=1.00 * rs,
        radius_valley_side=0.30 * rs,
        radius_peak_side=0.95 * rs,
        arc_peak_side=0.35 * rs,
        arc_valley_side=0.40 * rs,
    )
    return data


class CellSetOrgan(Organ):
    """An :class:`Organ` whose cells come from a CellSet file, with unresolved
    regions regenerated by GRANAP.

    Args:
        xml_path:   CellSet ``<cellsetdata>`` file.  GRANAP's own
                    ``write_to_xml`` output (root element ``<granardata>``) is
                    also accepted — the schema is the same.
        scale:      Coordinate multiplier; the default 0.001 converts CellSet
                    micrometres to GRANAP millimetres.
        regenerate: Tags whose cells are replaced by a generated stele.  A cell
                    is *also* regenerated whenever it is annular, whatever its
                    tag — MECHA cannot consume one, so ``()`` does **not** mean
                    "import unchanged": an annular cell is always rebuilt.
        donor_input: Configuration for the donor stele — an
                    :class:`OrganInputData`, a raw ``List[Dict]``, or ``None``
                    to use :func:`default_stele_input` sized from the region.
        donor_overrides: Per-param tweaks applied *on top of* the auto-sized
                    default, as ``{param_name: {field: value}}`` — e.g.
                    ``{"xylem": {"n_vascular_peak": 4}}`` for a tetrarch root.
                    This is the knob to reach for when adapting to a new species:
                    it keeps the region-derived sizing (which you do not want to
                    hand-tune per section) and changes only what is genuinely
                    species-specific.  Ignored when ``donor_input`` is given,
                    since that already replaces the whole configuration.
        keep_inner_cells: Keep real cells found inside a regenerated region
                    (the digitized vessels) and carve the generated cells around
                    them.  ``False`` deletes them and lets the donor place its
                    own vessels.
        star_orientation: ``"auto"`` aligns the donor's xylem plate with the axis
                    through the two most widely separated kept vessels;
                    a number is an explicit angle in degrees; ``None`` means 0.
                    ``GeometryProcessor.oriented_star_polygon`` puts its arms at
                    fixed multiples of ``2*pi/n`` from angle 0, so the region is
                    rotated for generation and the cells rotated back.
        cambium_tag: Tag applied to the donor's primary-cambium cells.  It
                    defaults to ``"stele"`` because ``CGROUP_MAP`` sends
                    ``"cambium"`` to 12, which MECHA reads back as *companion*
                    cell — a procambial ring in a primary root is stele
                    parenchyma, not companion cells.  Pass ``"cambium"`` to keep
                    the recipe's own tag.
        donor_xylem_tag: What to do with the vessels the donor recipe places
                    when the digitized ones are being kept — the xylem is real
                    data, so GRANAP should not add vessels of its own.  The
                    default ``"stele"`` *retags* them as stele parenchyma, which
                    is what surrounds a vessel in a young root and leaves the
                    tessellation intact.  ``None`` deletes them instead and lets
                    :func:`~openalea.granap.special_tissues.absorb_residual`
                    hand the vacated space to the neighbours — anatomically the
                    same result, but the space merges into whichever cell
                    borders it most, so a few cells come out oversized.  Ignored
                    when ``keep_inner_cells`` is False or nothing was kept, since
                    then the donor's vessels are the only xylem there is.
        overshoot:  How far (mm) past the region rim to grow the donor before
                    clipping it back, so the region is tiled exactly.  ``None``
                    uses :data:`_OVERSHOOT` times the region radius.
        recenter:   Move the section to the origin.  ``Cell.radius`` is measured
                    from the world origin and MECHA ranks cells by distance from
                    a gravity centre, so an un-recentred section gives
                    meaningless radii.
        seed:       Fixed seed, also handed to the donor.  ``Organ`` keeps only
                    ``self.rng`` and never stores the seed, so it is kept here.
    """

    # The imported outlines are the data; never fuse the concave gaps between
    # real cells (the intercellular spaces of the real section) into them.
    AUTO_FUSE_GAPS: bool = False

    def __init__(
        self,
        xml_path: str,
        *,
        scale: float = 0.001,
        regenerate: Sequence[str] = ("stele",),
        donor_input: Any = None,
        donor_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
        keep_inner_cells: bool = True,
        star_orientation: Any = "auto",
        cambium_tag: str = "stele",
        donor_xylem_tag: Optional[str] = "stele",
        overshoot: Optional[float] = None,
        recenter: bool = True,
        seed: Optional[int] = 0,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        self.xml_path = xml_path
        self.scale = float(scale)
        self.regenerate = tuple(regenerate)
        self.donor_input = donor_input
        self.donor_overrides = donor_overrides or {}
        self.keep_inner_cells = bool(keep_inner_cells)
        self.star_orientation = star_orientation
        self.cambium_tag = cambium_tag
        self.donor_xylem_tag = donor_xylem_tag
        self.overshoot = overshoot
        self.recenter = bool(recenter)
        self._seed = seed

        # Populated by generate_cells(); inspectable afterwards.
        self.section: Optional[CellSetSection] = None
        self.donors: List[Organ] = []
        self.graft_report: List[Dict[str, Any]] = []
        # id(Cell) -> the CellSetCell it was imported from, so the ring data
        # (notably is_annular) stays reachable without abusing a Cell field.
        self._source_of: Dict[int, CellSetCell] = {}

    # ------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------

    def _load_cellset(self) -> None:
        """Read the file and turn every digitized cell into a :class:`Cell`."""
        self.section = read_cellset(self.xml_path, scale=self.scale)

        cells: List[Cell] = []
        self._source_of = {}
        for i, src in enumerate(self.section.cells):
            if src.polygon is None or src.polygon.is_empty:
                continue
            poly = src.polygon
            cells.append(
                Cell(
                    x=poly.centroid.x,
                    y=poly.centroid.y,
                    diameter=2.0 * math.sqrt(poly.area / math.pi),
                    type=src.tag,
                    id_cell=i,
                    id_layer=0,
                    id_group=i,
                    # CellSet's group integer *is* MECHA's cgroup; carrying it
                    # through means a tissue GRANAP has no tag for still reaches
                    # MECHA correctly (anatomy_writer resolves cgroup first).
                    cgroup=src.group,
                    polygon=poly,
                )
            )
            # Keep every measured vertex through CellGenerator.simplify_cells.
            cells[-1].protect_shape = True
            # Must stay False: see the module docstring, point 3.
            cells[-1].protect_topology = False
            self._source_of[id(cells[-1])] = src

        self.all_cells.cells = cells
        if self.recenter and cells:
            self.all_cells.recenter_cells()

    # ------------------------------------------------------------------
    # Regeneration
    # ------------------------------------------------------------------

    def _regeneration_targets(self) -> List[Tuple[Cell, Polygon, List[Cell]]]:
        """``(annular_cell, region, inner_cells)`` for every region to rebuild.

        ``region`` is the cell's outer boundary with its holes filled in;
        ``inner_cells`` are the real cells sitting inside it (the digitized
        vessels that punched those holes).
        """
        targets: List[Tuple[Cell, Polygon, List[Cell]]] = []
        for cell in list(self.all_cells.cells):
            src = self._source_of.get(id(cell))
            annular = bool(src.is_annular) if src is not None else False
            if not annular and cell.type not in self.regenerate:
                continue
            if cell.polygon is None or cell.polygon.is_empty:
                continue
            # Holes are filled in by taking the exterior ring alone; after
            # recentring we cannot reuse the reader's polygon, so rebuild from
            # the (already translated) cell polygon.
            region = Polygon(cell.polygon.exterior)
            if not region.is_valid:
                region = _largest_part(region.buffer(0))
            if region is None or region.is_empty:
                continue
            inner = [
                other
                for other in self.all_cells.cells
                if other is not cell
                and other.polygon is not None
                and region.contains(other.polygon.centroid)
            ]
            targets.append((cell, region, inner))
        return targets

    def _star_angle(self, inner_cells: Sequence[Cell]) -> float:
        """Orientation (degrees) for the donor's xylem plate."""
        if self.star_orientation != "auto":
            return 0.0 if self.star_orientation is None else float(self.star_orientation)
        pts = [(c.polygon.centroid.x, c.polygon.centroid.y) for c in inner_cells
               if c.polygon is not None]
        if len(pts) < 2:
            return 0.0
        best, angle = -1.0, 0.0
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d = math.dist(pts[i], pts[j])
                if d > best:
                    best = d
                    angle = math.degrees(
                        math.atan2(pts[j][1] - pts[i][1], pts[j][0] - pts[i][0])
                    )
        return angle

    @staticmethod
    def _core_radius(donor: Organ) -> Optional[float]:
        """Inscribed radius of the polygon the donor's vascular recipe will get.

        ``Organ.allocate_vascular_tissue`` hands ``_vascular_recipe`` whatever
        ``_which_layer_for_vascular`` picks -- for a root, the first layer polygon
        named ``"stele"`` (``RootAnatomy._which_layer_for_vascular``).  Building
        the layer polygons places no cells, so this is cheap to measure up front.
        """
        from openalea.granap.geometry_collection import GeometryProcessor

        try:
            layers = donor.generate_layer_polygons()
        except Exception:
            return None
        stele = next((l["polygon"] for l in layers if l["name"] == "stele"), None)
        if stele is None or stele.is_empty:
            return None
        return GeometryProcessor._chebyshev_center(stele)[2]

    def _build_donor(self, region: Polygon, angle: float) -> Tuple[Organ, List[Cell]]:
        """Generate a stele on ``region``, rotated so its xylem plate follows ``angle``."""
        from openalea.granap.root_class import RootAnatomy

        from openalea.granap.geometry_collection import GeometryProcessor

        centre = region.centroid
        region_rot = _rotate(region, -angle, origin=centre) if angle else region

        radius = math.sqrt(region.area / math.pi)
        overshoot = self.overshoot if self.overshoot is not None else _OVERSHOOT * radius
        base = (
            GeometryProcessor.buffer_polygon(region_rot, overshoot, smooth_factor=0.0)
            if overshoot > 0.0
            else region_rot
        )

        def _make(spec) -> Organ:
            organ = RootAnatomy(
                spec.to_dict_list() if isinstance(spec, OrganInputData) else spec,
                seed=self._seed,
            )
            # generate_base_shape() returns the cached _base_polygon when it is
            # set, so this substitutes the real outline for the parametric
            # contour without subclassing (RootAnatomy.__new__ is a transparent
            # monocot/dicot factory).
            organ._base_polygon = base
            return organ

        if self.donor_input is not None:
            donor = _make(self.donor_input)
        else:
            # Two passes.  The vascular recipe is handed the innermost "stele"
            # layer polygon, not the region, and its radii are measured from that
            # polygon's centre -- so measure it on a throwaway donor (building
            # layer polygons places no cells) and size the star to it.  Sizing
            # against the region instead puts the phloem valleys outside the
            # polygon and they silently produce nothing.
            def spec(core=None):
                data = default_stele_input(
                    radius, core_radius=core, overshoot=overshoot
                )
                for name, fields in self.donor_overrides.items():
                    data.set_values(name, **fields)
                return data
            probe = _make(spec())
            core = self._core_radius(probe) or _CORE_RADIUS_GUESS * radius
            donor = _make(spec(core))
        donor.generate_cells()

        cells: List[Cell] = []
        for src in donor.all_cells.cells:
            if src.polygon is None or src.polygon.is_empty:
                continue
            clone = copy.copy(src)
            clone.polygon = (
                _rotate(src.polygon, angle, origin=centre) if angle else src.polygon
            )
            if clone.type == "cambium" and self.cambium_tag != "cambium":
                clone.type = self.cambium_tag
            cells.append(clone)
        return donor, cells

    def _graft(self, region: Polygon, donor_cells: List[Cell],
               inner_cells: List[Cell]) -> Dict[str, Any]:
        """Clip the donor to ``region``, weld the interface, seat the real vessels."""
        # Clip against the real outline: every donor cell touching the boundary
        # now carries the region's own vertices, plus the cut points where two
        # donor cells meet the boundary.
        clipped: List[Cell] = []
        for cell in donor_cells:
            part = _largest_part(cell.polygon.intersection(region))
            if part is None or part.is_empty:
                continue
            cell.polygon = part
            clipped.append(cell)

        areas = [c.polygon.area for c in clipped]
        min_area = 0.02 * float(np.median(areas)) if areas else 0.0
        clipped = [c for c in clipped if c.polygon.area >= min_area]

        # The xylem is real data. When the digitized vessels are kept, the
        # donor's own vessels are duplicates of a tissue we already measured, so
        # they go -- either retagged as the parenchyma that surrounds a vessel
        # (default, tessellation untouched) or deleted, in which case the
        # residual step below hands their space to the neighbours.
        n_donor_xylem = 0
        if inner_cells:
            keep_cells = []
            for cell in clipped:
                if cell.type not in _XYLEM_TAGS:
                    keep_cells.append(cell)
                    continue
                n_donor_xylem += 1
                if self.donor_xylem_tag is not None:
                    cell.type = self.donor_xylem_tag
                    keep_cells.append(cell)
            clipped = keep_cells

        # Seat the digitized vessels: carving the donor cells with their union
        # keeps both sides of that boundary vertex-for-vertex identical.
        inner_polys = [c.polygon for c in inner_cells if c.polygon is not None]
        if inner_polys:
            mask = unary_union(inner_polys)
            if not mask.is_empty:
                # Carve with a *degeneracy* floor, not the sliver floor used for
                # clipping.  Dropping a carved remainder does not merely lose a
                # small cell: it opens a gap that ``absorb_residual`` below then
                # has to hand to a neighbour, and that merge is what puts three
                # cells on one wall.  Measured on the Arabidopsis section, the
                # clipped donor tiles the region *exactly* and carve is the only
                # stage that opens a gap -- 0 at seed 0 but 1e-7..7e-7 at seeds
                # 1/2/4, matching the residual absorb saw. Keeping the small
                # remainder as a cell is strictly better than gapping the region.
                clipped = carve_cells(clipped, mask, min_area=_DEGENERATE_AREA)

        # Tile the region exactly. Whatever the donor leaves uncovered is
        # boundary only one cell owns, and MECHA reads a one-cell wall inside the
        # tissue as a gas-space boundary -- so this is a correctness step, not a
        # cosmetic one. Absorbing into donor cells only keeps every cut point
        # pointing donor -> real, which is what makes the one-directional weld
        # below sufficient.
        absorbed = absorb_residual(clipped, region, others=inner_polys)

        # Weld last, once the donor boundary has stopped moving.  Every real cell
        # the graft was cut against needs the donor's cut points in its own ring,
        # and that goes in *both* directions:
        #   - outward, to the neighbours across the region rim (the endodermis);
        #   - inward, to the cells kept inside it (the digitized vessels), which
        #     is the more surprising half -- a vessel drawn with 9 vertices ends
        #     up abutting 7 donor cells, so most of the donor junctions along it
        #     fall mid-edge on the vessel's own coarse ring.
        # conform_boundaries writes the *same* coordinate into both rings, so the
        # match does not depend on the graft's cut points having landed exactly
        # on an edge -- which they only do for one particular configuration.
        #
        # The donor cells are conformed against *each other* as well as against
        # the real neighbours.  ``absorb_residual`` above merges each uncovered
        # patch into one host, which re-nodes that host's boundary along the walls
        # it shares with the *other donor cells* around the patch -- and they do
        # not gain the same vertices.  Conforming only donor -> real leaves that
        # asymmetry in place, and each un-shared vertex splits one wall into
        # three single-reference ones.  Measured on the Arabidopsis section: with
        # donor->real alone, seed 0 happened to need no absorbing and passed while
        # seeds 1-5 produced 22-49 interior border walls.
        neighbours = [
            c
            for c in self.all_cells.cells
            if c.polygon is not None and not c.polygon.disjoint(region)
        ]
        conformed = conform_boundaries(neighbours + clipped, clipped)
        n_welded = conformed["welded"] + conformed["spliced"]

        # Append with explicit ids. CellManager.extend_cells re-bases id_cell and
        # id_group by max_id + cell.id + 1, which collides for donor groups that
        # start at 0 and would fuse unrelated groups; offsetting id_group by
        # next_group_id() keeps the donor's fused Voronoi groups intact.
        group_offset = self.all_cells.next_group_id()
        next_id = len(self.all_cells.cells)
        for k, cell in enumerate(clipped):
            cell.id_cell = next_id + k
            cell.id_group = int(cell.id_group) + group_offset
            cell.id_layer = 0
            self.all_cells.cells.append(cell)

        covered = sum(c.polygon.area for c in clipped) + sum(p.area for p in inner_polys)
        return {
            "region_area": region.area,
            "n_donor_cells": len(clipped),
            "n_inner_kept": len(inner_cells),
            "n_neighbours": len(neighbours),
            "n_vertices_welded": n_welded,
            "n_donor_xylem_removed": n_donor_xylem,
            "min_area": min_area,
            "coverage": covered / region.area if region.area else 0.0,
            **{f"absorb_{k}": v for k, v in absorbed.items()},
        }

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def generate_cells(self) -> gpd.GeoDataFrame:
        """Import the real cells, regenerate the unresolved regions, export a gdf.

        Replaces ``Organ.generate_cells`` wholesale: there is no seeding, no
        global Voronoi and no layer peeling here — the real cells are the data,
        and the only tessellation is the donor's, which runs inside
        :meth:`_build_donor` through the ordinary pipeline.
        """
        if self._cells_gdf is not None:
            return self._cells_gdf

        self._load_cellset()
        self.donors = []
        self.graft_report = []

        for annular, region, inner in self._regeneration_targets():
            keep = inner if self.keep_inner_cells else []
            if not self.keep_inner_cells and inner:
                self.all_cells.remove_cells(inner)
            # Drop the unresolved cell before grafting so it cannot be picked up
            # as one of its own replacement's neighbours.
            self.all_cells.remove_cells([annular])

            angle = self._star_angle(keep)
            donor, donor_cells = self._build_donor(region, angle)
            self.donors.append(donor)
            report = self._graft(region, donor_cells, keep)
            report["tag"] = annular.type
            report["angle"] = angle
            self.graft_report.append(report)

        self.all_cells.recalculate_cell_properties()

        # Every cell must export. Both writers filter the GeoDataFrame on valid
        # geometry and then use its *index* as the MECHA cell id, while MECHA
        # computes cell node ids arithmetically (n_walls + n_junctions + i) over
        # range(n_cells) -- so one dropped row shifts every cell node after it.
        # An interior ring is just as bad: CellGenerator._build_topology reads
        # only ``poly.exterior``, so a hole would silently vanish from the
        # topology while still being absent from the polygon's area.
        bad = [
            (c.id_cell, c.type)
            for c in self.all_cells.cells
            if c.polygon is None
            or c.polygon.is_empty
            or c.polygon.geom_type != "Polygon"
            or c.polygon.interiors
        ]
        if bad:
            raise ValueError(
                "CellSetOrgan produced cells that cannot be exported "
                f"(empty, multipart or holed): {bad[:10]}"
                f"{' ...' if len(bad) > 10 else ''}"
            )

        cell_dicts = [c.cell_to_dict() for c in self.all_cells.cells]
        for i, c in enumerate(self.all_cells.cells):
            cell_dicts[i]["geometry"] = c.polygon
        self._cells_gdf = (
            gpd.GeoDataFrame(cell_dicts) if cell_dicts else gpd.GeoDataFrame()
        )
        return self._cells_gdf

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def topology_report(self) -> Dict[str, Any]:
        """Wall-sharing summary of the exported network — the graft's acceptance test.

        Call after :meth:`export_to_adjencymatrix`.  A wall with one flanking
        cell is not just a missing connection: MECHA's
        ``NetworkBuilder.identify_border_walls_junctions`` files a
        single-reference wall whose owner is not epidermis under
        ``border_aerenchyma``, turning it into a gas-space boundary.  So a
        correct graft has border walls **only** on the section's real outer
        surface, and ``interior_border_walls`` must be empty.
        """
        tags = {
            i: c.type for i, c in enumerate(self.all_cells.cells)
        }
        border: Dict[str, int] = {}
        multi = 0
        for wall_id, cell_nodes in self._wall_to_cells.items():
            if len(cell_nodes) > 2:
                multi += 1
            elif len(cell_nodes) == 1:
                idx = cell_nodes[0] - self.n_walls - self.n_junctions
                tag = tags.get(idx, "?")
                border[tag] = border.get(tag, 0) + 1

        pd_pairs: Dict[Tuple[str, str], int] = {}
        cell_tag = {
            nd: d.get("cell_type")
            for nd, d in self.graph.nodes(data=True)
            if d.get("type") == "cell"
        }
        for u, v, data in self.graph.edges(data=True):
            if data.get("path") != "plasmodesmata":
                continue
            a, b = cell_tag.get(u), cell_tag.get(v)
            if a and b:
                key = (a, b) if a <= b else (b, a)
                pd_pairs[key] = pd_pairs.get(key, 0) + 1

        surface = {"epidermis", "exodermis", "cortex"}
        interior = {t: n for t, n in border.items() if t not in surface}
        if interior or multi:
            # Report whichever defect actually occurred.  They are different
            # faults with different causes -- single-reference walls mean the
            # interface is not shared, walls with 3+ flanking cells mean a ring
            # came back invalid -- and a message that always leads with the
            # single-reference count reads as "0 walls are wrong" when only the
            # latter is present.
            faults = []
            if interior:
                faults.append(
                    f"{sum(interior.values())} wall(s) inside the tissue have a "
                    f"single flanking cell {interior}; MECHA reads those as "
                    "gas-space boundaries"
                )
            if multi:
                faults.append(
                    f"{multi} wall(s) have more than two flanking cells, which "
                    "means a cell ring came back invalid"
                )
            warnings.warn(
                "CellSetOrgan: the graft interface is not fully shared — "
                + "; ".join(faults)
                + ". This network is not physically sound. Measured causes, at "
                "seed 0 on x86: overshoot=0 (leaves real rim gaps no welding "
                "can close), donor_xylem_tag=None and n_vascular_peak>=3 (both "
                "pack the donor's vessels tighter than the stele can hold). "
                "Treat that as examples, not an exhaustive set: which seeds and "
                "options trip this varies by platform — overshoot=0 tiles "
                "cleanly on macOS/arm64 — so the report, not the list, is what "
                "tells you whether *your* section is sound.",
                stacklevel=2,
            )
        return {
            "n_walls": self.n_walls,
            "n_junctions": self.n_junctions,
            "n_cells": self.n_cells,
            "border_walls": border,
            "interior_border_walls": interior,
            "multi_cell_walls": multi,
            "plasmodesmata_pairs": pd_pairs,
        }

    def _create_base_shape(self) -> Polygon:
        """The section outline: the union of every real cell.

        Deliberately *not* ``RoiOrgan``'s convex hull — a real section is concave,
        and a hull would make ``find_gaps`` report the whole surround as a gap.
        """
        polys = [c.polygon for c in self.all_cells.cells if c.polygon is not None]
        if not polys:
            return Polygon()
        return _largest_part(unary_union(polys)) or Polygon()

    # ---------- Stubs for the abstract methods in Organ ----------

    def _which_layer_for_vascular(self, layers_polygons: List[LayerPolygon]):
        pass

    def _create_vascular_tissue(self, polygon: Polygon):
        pass

    def _organ_specific_tissues(self):
        pass

    def add_intercellular_spaces(self):
        pass

    def _create_central_layers(self, current_polygon: Polygon,
                               params: List[Dict[str, Any]]) -> List[LayerPolygon]:
        return []

    def reshape_layers(self, layers_polygons: List[LayerPolygon]) -> List[LayerPolygon]:
        return layers_polygons
