"""Reusable tissue-building vocabulary.

This module extracts the low-level, organ-agnostic primitives that were
previously duplicated across the ``fit_*`` methods of the anatomy classes.
It is the *generative* half of a tag-based, compositional model:

    generative primitives  -- create cells inside a zone
        place_packed_group   : seed every circle of a pre-computed packing
        fill_by_packing      : pack a zone with circles, then seed each
        fill_along           : seed cells along a polygon edge / line
        fill_by_rings        : seed concentric inward rings inside a zone

    Tissue (shape-first)    -- a tagged anatomical *region* (a shapely shape)
        .rotate / .translate : pure-geometry transforms of the region
        .smooth              : smooth the region boundary (pre-fill)
        .difference / .intersection / .union : region algebra

    edit verbs              -- the only verb valid on a filled cell
        retag_tissue         : rename a tag

    composition             -- order primitives into an explicit recipe
        TissueStep           : one named build step (+ the tags it produces)
        TissueRecipe         : an ordered, inspectable list of steps

The model is **shape-first**: a tissue is its region, manipulated as geometry,
and *then* filled with cells by a ``fill_*`` primitive.  Cells are the terminal
product of filling (retag-only).  The fill primitives operate on a
:class:`~openalea.granap.cell_manager.CellManager`, so the same vocabulary builds
vascular tissue, layer tissue, or any future organ.  Higher-level "recipes"
(monocot, dicot, needle) compose these pieces instead of re-implementing loops.
"""

from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import shapely as sp
from shapely.geometry import Point, Polygon
from shapely.affinity import rotate as _shapely_rotate, translate as _shapely_translate

from openalea.granap.cell_class import Cell
from openalea.granap.cell_manager import CellManager
from openalea.granap.geometry_collection import GeometryProcessor
from openalea.granap.generate_cell import CellGenerator


# ---------------------------------------------------------------------------
# Generative primitives
# ---------------------------------------------------------------------------

def place_packed_group(
    target: CellManager,
    packed,
    cell_type: str,
    *,
    n_border: int = 25,
    id_base: int = 0,
    angle_center: Optional[Tuple[float, float]] = None,
    min_diameter: Optional[float] = None,
    alt_type: Optional[str] = None,
) -> List[Tuple[Polygon, str, int]]:
    """Place border-point seeds for every circle of a circle-packing.

    This is the common pattern behind metaxylem/protoxylem/phloem packing:
    for each packed circle ``(cx, cy, r)`` a slightly shrunk ring of border
    points is resampled and added as a single Voronoi group.

    Args:
        target: CellManager the new cells are appended to.
        packed: iterable of ``(cx, cy, r)`` tuples (output of ``pack_circles``).
        cell_type: tag assigned to the cells.  When ``min_diameter`` is given,
            this is the tag for circles whose diameter is >= ``min_diameter``.
        n_border: number of border points resampled per circle (24 or 25 in the
            legacy code).
        id_base: ``id_group`` of circle *i* is ``id_base + i``.  Each circle is
            therefore its own Voronoi group, with disjoint ranges between calls.
        angle_center: ``(cx, cy)`` reference used for the per-seed ``angle`` /
            ``radius`` attributes.  ``None`` uses each circle's own centroid.
        min_diameter: optional diameter threshold to split a single packing into
            two tags (e.g. wide vessels -> "xylem", narrow -> "stele").
        alt_type: tag assigned to circles below ``min_diameter``.

    Returns:
        List of ``(placed_polygon, resolved_type, id_group)`` for every circle
        actually placed, so the caller can record vascular polygons selectively.
    """
    placed_out: List[Tuple[Polygon, str, int]] = []

    for i, rec in enumerate(packed):
        # A record is a circle (cx, cy, r) or an ellipse (cx, cy, a, b, angle);
        # see GeometryProcessor.pack_circles(allow_ellipse=...).
        if len(rec) == 3:
            pcx, pcy, r = rec
            placed = Point(pcx, pcy).buffer(r, resolution=32)
            actual_diam = r * 2
            inset = r * 0.15
        else:
            pcx, pcy, a_ax, b_ax, ang = rec
            placed = GeometryProcessor.ellipse_to_polygon(pcx, pcy, a_ax, b_ax, ang)
            actual_diam = 2.0 * np.sqrt(a_ax * b_ax)   # area-equivalent diameter
            inset = min(a_ax, b_ax) * 0.15
        placed_buff = placed.buffer(-inset)
        if placed_buff.is_empty:
            continue

        bx, by = placed_buff.exterior.coords.xy
        border_coords = GeometryProcessor.resample_coords(
            np.column_stack((bx, by)), target_n_points=n_border
        )

        if angle_center is None:
            acx, acy = placed.centroid.x, placed.centroid.y
        else:
            acx, acy = angle_center

        if min_diameter is not None:
            rtype = cell_type if actual_diam >= min_diameter else alt_type
        else:
            rtype = cell_type

        gid = id_base + i
        for border_pt in border_coords[1:]:
            target.add_cell(Cell.radial(
                rtype, border_pt[0], border_pt[1], actual_diam, gid, (acx, acy),
            ))
        placed_out.append((placed, rtype, gid))

    return placed_out


def fill_along(
    target: CellManager,
    geometry,
    cell_type: str,
    cell_diam: float,
    cell_width: float,
    cx: float,
    cy: float,
    xylem_union=None,
    keep_union=None,
) -> None:
    """Place cell seeds along a geometry edge (Polygon exterior or line).

    Each resampled position along the edge becomes one Voronoi group, oriented
    by its local tangent (via ``CellGenerator.cell_border``).  Positions falling
    inside ``xylem_union`` are skipped; if ``keep_union`` is given, positions
    *outside* it are skipped too (so the seeds are confined to it).  A skipped
    position still consumes an id_group, matching the legacy behaviour.
    """
    if isinstance(geometry, Polygon):
        line_segs = [geometry.exterior]
    elif hasattr(geometry, "geoms"):
        line_segs = list(geometry.geoms)
    else:
        line_segs = [geometry]

    next_id_group = (target.get_last_id_group() + 1) if target.cells else 0

    for line_seg in line_segs:
        raw_coords = np.array(line_seg.coords)
        seg_length = line_seg.length
        n_cells = max(2, int(np.ceil(seg_length / (cell_width or cell_diam))))
        cells_coords = GeometryProcessor.resample_coords(raw_coords, n_cells)
        if len(cells_coords) < 2:
            continue

        cell_borders = CellGenerator.cell_border(
            cells_coords,
            cell_width * 0.7 if cell_width else cell_diam * 0.7,
            cell_diam * 0.7 if cell_width else 0,
        )

        for i, _coord in enumerate(cells_coords[1:]):
            pt = Point(_coord[0], _coord[1])
            if (xylem_union and xylem_union.contains(pt)) or \
               (keep_union is not None and not keep_union.contains(pt)):
                next_id_group += 1
                continue
            id_group = next_id_group
            next_id_group += 1
            for border_pt in cell_borders[i][1:]:
                target.add_cell(Cell.radial(
                    cell_type, border_pt[0], border_pt[1], cell_diam, id_group, (cx, cy),
                ))


def fill_by_rings(
    target: CellManager,
    fill_zone,
    cell_diameter: float,
    cell_width: float,
    cell_type: str,
    cx: float,
    cy: float,
    start_id: int,
    erosion_polygon=None,
    initial_space: Optional[float] = None,
) -> int:
    """Fill a polygon zone with seeds on concentric inward rings.

    Buffers ``fill_zone`` (or ``erosion_polygon`` if given) inward one cell at a
    time, seeding each ring.  When ``erosion_polygon`` is supplied the rings are
    eroded from it but filtered to stay inside ``fill_zone``.

    ``initial_space`` sets how far the *first* ring is inset from the boundary:
    the first erosion is ``initial_space + cell_diameter/2`` (default
    ``cell_diameter/2`` -> a full ``cell_diameter``).  Pass ``0.0`` to hug the
    boundary (first ring half a cell in), so seeds sit right against a holed
    zone's inner edges instead of leaving a cell-wide moat around each hole.

    Returns the next free ``id_group`` so callers can chain multiple zones.
    """
    if fill_zone is None or fill_zone.is_empty:
        return start_id
    if fill_zone.area < np.pi * (cell_diameter / 2) ** 2:
        return start_id

    next_id  = start_id
    space    = cell_diameter / 2 if initial_space is None else initial_space
    tang     = cell_width if cell_width else cell_diameter
    current  = erosion_polygon if erosion_polygon is not None else fill_zone
    # When eroding from ``erosion_polygon`` the rings are filtered to stay inside
    # ``fill_zone``; batch those point-in-polygon tests with ``contains_xy`` (one
    # array call per ring) instead of a shapely ``Point`` + prepared ``.contains``
    # per candidate
    filter_active = erosion_polygon is not None

    # Reject a seed that coincides with an already-placed one.
    placed_x: List[float] = []
    placed_y: List[float] = []
    min_spacing2 = (0.7 * min(cell_diameter, tang)) ** 2

    while not current.is_empty and current.area > (cell_diameter / 2) ** 2 * np.pi:
        current = current.buffer(-space - cell_diameter / 2, resolution=16)
        if current.is_empty:
            break
        space = cell_diameter / 2

        geoms = list(current.geoms) if hasattr(current, "geoms") else [current]
        for geom in geoms:
            if geom.is_empty or geom.geom_type != "Polygon":
                continue
            # Seed the exterior contour AND every interior hole boundary.
            # ``cells_on_layer`` only traces a polygon's exterior, so without the
            # interior rings a holed fill zone (e.g. parenchyma packed around
            # carved-out sieve elements) leaves an unseeded moat around each hole
            # that the hole's own Voronoi cell then balloons into. Tracing the
            # interiors packs a ring of seeds hugging each hole so it renders at
            # its true size.
            coord_arrays = [CellGenerator.cells_on_layer(geom, cell_diameter, cell_width)]
            for interior in geom.interiors:
                ring_poly = Polygon(interior)
                if ring_poly.is_empty:
                    continue
                coord_arrays.append(
                    CellGenerator.cells_on_layer(ring_poly, cell_diameter, cell_width)
                )
            coord_arrays = [c for c in coord_arrays if len(c)]
            if not coord_arrays:
                continue
            seed_coords  = np.vstack(coord_arrays)
            border_rings = CellGenerator.cell_border(seed_coords, tang * 0.7, cell_diameter * 0.7)
            seeds = seed_coords[1:]
            rings = border_rings[1:]
            seed_ok = None
            if filter_active and len(seeds):
                sx = np.fromiter((p[0] for p in seeds), float, len(seeds))
                sy = np.fromiter((p[1] for p in seeds), float, len(seeds))
                seed_ok = sp.contains_xy(fill_zone, sx, sy)
            for si, (pt, border_pts) in enumerate(zip(seeds, rings)):
                if filter_active and not seed_ok[si]:
                    continue
                if placed_x:
                    dx = np.asarray(placed_x) - pt[0]
                    dy = np.asarray(placed_y) - pt[1]
                    if np.min(dx * dx + dy * dy) < min_spacing2:
                        continue
                placed_x.append(pt[0])
                placed_y.append(pt[1])
                id_group    = next_id
                next_id    += 1
                cell_angle  = np.arctan2(pt[1] - cy, pt[0] - cx)
                cell_radius = np.sqrt((pt[0] - cx) ** 2 + (pt[1] - cy) ** 2)
                border_pts = border_pts[1:]
                border_ok = None
                if filter_active and len(border_pts):
                    bx = np.fromiter((b[0] for b in border_pts), float, len(border_pts))
                    by = np.fromiter((b[1] for b in border_pts), float, len(border_pts))
                    border_ok = sp.contains_xy(fill_zone, bx, by)
                for bi, border_pt in enumerate(border_pts):
                    if filter_active and not border_ok[bi]:
                        continue
                    target.add_cell(Cell(
                        type=cell_type,
                        x=border_pt[0], y=border_pt[1],
                        diameter=cell_diameter,
                        id_cell=id_group, id_group=id_group,
                        angle=cell_angle, radius=cell_radius,
                    ))
    return next_id


def fill_by_packing(
    target: CellManager,
    zone,
    cell_type: str,
    *,
    rng,
    n_border: int = 25,
    id_base: Optional[int] = None,
    angle_center: Optional[Tuple[float, float]] = None,
    min_diameter: Optional[float] = None,
    alt_type: Optional[str] = None,
    **pack_kwargs,
) -> List[Tuple[Polygon, str, int]]:
    """Fill a zone by circle-packing, then seed each circle (zone-level verb).

    Combines ``GeometryProcessor.pack_circles`` with :func:`place_packed_group`
    so a recipe can express "fill this zone by packing" as a single step,
    symmetric with :func:`fill_along` and :func:`fill_by_rings`.

    ``id_base`` defaults to one past the highest ``id_group`` already in
    ``target``, so successive calls land in disjoint id ranges (each packed
    circle is its own Voronoi group).  ``**pack_kwargs`` is forwarded verbatim
    to ``pack_circles`` (``proportion``, ``diameter_max``, ``gradient_*`` ...).

    Returns the :func:`place_packed_group` output.
    """
    if zone is None or zone.is_empty:
        return []
    if id_base is None:
        id_base = (target.get_last_id_group() + 1) if target.cells else 0
    packed = GeometryProcessor.pack_circles(zone, rng=rng, **pack_kwargs)
    return place_packed_group(
        target, packed, cell_type,
        n_border=n_border, id_base=id_base, angle_center=angle_center,
        min_diameter=min_diameter, alt_type=alt_type,
    )


# ---------------------------------------------------------------------------
# Tissue -- a tagged anatomical region (shape-first); cells are a fill product
# ---------------------------------------------------------------------------

def _as_shape(other):
    """Accept a :class:`Tissue` or a raw shapely geometry; return the geometry."""
    return other.shape if isinstance(other, Tissue) else other


class Tissue:
    """A tagged anatomical region: a shapely shape plus the tag its cells take.

    Shape-first: a tissue *is* its region, manipulated as pure geometry
    (rotate / translate / smooth / boolean-combine) entirely before any cell
    exists.  Filling the region with cells (the ``fill_*`` primitives, given
    ``tissue.shape`` and ``tissue.tag``) is a separate, terminal step; the
    resulting cells are retag-only (:func:`retag_tissue`).

    Every transform mutates ``self.shape`` in place and returns ``self`` so they
    can be chained, e.g. ``Tissue("phloem", e).intersection(stele).difference(xylem)``.
    """

    def __init__(self, tag: str, shape):
        self.tag = tag
        self.shape = shape

    # -- pure-geometry transforms -------------------------------------------
    def rotate(self, angle: float, origin: Tuple[float, float] = (0.0, 0.0)) -> "Tissue":
        """Rotate the region by ``angle`` degrees about ``origin``."""
        self.shape = _shapely_rotate(self.shape, angle, origin=origin)
        return self

    def translate(self, dx: float, dy: float) -> "Tissue":
        """Translate the region by ``(dx, dy)``."""
        self.shape = _shapely_translate(self.shape, xoff=dx, yoff=dy)
        return self

    def smooth(self, smoothness: float) -> "Tissue":
        """Smooth the region boundary (pre-fill, pure geometry)."""
        if smoothness:
            self.shape = GeometryProcessor.buffer_polygon(
                self.shape, 0, smooth_factor=smoothness
            )
        return self

    # -- region algebra ------------------------------------------------------
    def difference(self, other) -> "Tissue":
        """Subtract another region (Tissue or shapely geometry) from this one."""
        self.shape = self.shape.difference(_as_shape(other))
        return self

    def intersection(self, other) -> "Tissue":
        """Clip this region to another (Tissue or shapely geometry)."""
        self.shape = self.shape.intersection(_as_shape(other))
        return self

    def union(self, other) -> "Tissue":
        """Merge another region (Tissue or shapely geometry) into this one."""
        self.shape = self.shape.union(_as_shape(other))
        return self

    # -- queries -------------------------------------------------------------
    @property
    def is_empty(self) -> bool:
        return self.shape is None or self.shape.is_empty

    @property
    def area(self) -> float:
        return 0.0 if self.is_empty else self.shape.area

    def __repr__(self) -> str:
        return f"Tissue({self.tag!r}, area={self.area:.4g})"


# ---------------------------------------------------------------------------
# Edit verbs -- the only verb valid on a filled cell is retag
# ---------------------------------------------------------------------------

def retag_tissue(target: CellManager, old_tag: str, new_tag: str) -> int:
    """Rename every cell tagged ``old_tag`` to ``new_tag``.

    The terminal cell-level edit: once a region is filled, cells are inert
    except for their tag.  Returns the number of cells retagged.
    """
    n = 0
    for c in target.cells:
        if c.type == old_tag:
            c.type = new_tag
            n += 1
    return n


# ---------------------------------------------------------------------------
# Composition -- order primitives into an explicit recipe
# ---------------------------------------------------------------------------

class TissueStep:
    """One named step of a tissue-build recipe.

    A step is just a label, the tags it is expected to produce, and a callable
    that performs the work (typically a closure over one or more of the
    primitives / edit verbs above, or an existing ``fit_*`` method).  Keeping
    the build order as data — rather than buried in control flow — lets a recipe
    be inspected (:meth:`TissueRecipe.describe`), reordered, or extended with
    edit-verb steps without touching the generators.
    """

    def __init__(
        self,
        name: str,
        fn: Callable[[], None],
        *,
        produces: Tuple[str, ...] = (),
        kind: str = "step",
    ):
        self.name = name
        self.fn = fn
        self.produces = tuple(produces)
        self.kind = kind          # "fill" | "fill_each" | "cleanup" | "special" | "add"

    def run(self) -> None:
        self.fn()

    def __repr__(self) -> str:
        tags = ", ".join(self.produces)
        return f"TissueStep({self.name!r}, kind={self.kind!r}, produces=[{tags}])"


def _dispatch_fill(target: CellManager, tissue: "Tissue", strategy: str, rng, **kw):
    """Fill ``tissue.shape`` with ``tissue.tag`` cells using a named strategy.

    Maps a recipe's ``strategy=`` onto the underlying fill primitives so a recipe
    step can declare *how* a region is filled without naming the function:

    * ``"packing"`` -> :func:`fill_by_packing` (circle-pack the region, seed each)
    * ``"rings"``   -> :func:`fill_by_rings`   (concentric inward rings)
    * ``"line"``    -> :func:`fill_along`      (seeds along the region exterior)

    ``**kw`` is forwarded verbatim to the chosen primitive, so each strategy
    keeps its own parameters.  The strategies have different return types
    (``packing`` returns the placed-circle list; ``rings`` the next free id;
    ``line`` ``None``) — whatever the primitive returns is handed to the step's
    ``record`` hook so the caller can do its own bookkeeping.
    """
    if strategy == "packing":
        return fill_by_packing(target, tissue.shape, tissue.tag, rng=rng, **kw)
    if strategy == "rings":
        return fill_by_rings(target, tissue.shape, cell_type=tissue.tag, **kw)
    if strategy == "line":
        return fill_along(target, tissue.shape, tissue.tag, **kw)
    raise ValueError(f"unknown fill strategy {strategy!r}")


class TissueRecipe:
    """An ordered, inspectable sequence of :class:`TissueStep` objects.

    Beyond the low-level :meth:`add` (wrap an arbitrary callable), a recipe can
    be expressed *declaratively* — a step names a region (:class:`Tissue`) and a
    fill strategy, rather than wrapping a bespoke ``fit_*`` method:

        recipe = TissueRecipe(cells=organ.vascular_cells, rng=organ.rng)
        recipe.fill("xylem star", xylem_region, strategy="packing", **pack_kw)
        recipe.cleanup("clear stele under xylem", organ.clear_stele)
        recipe.fill_each("phloem valleys", valley_regions, strategy="packing", **pack_kw)

    The bound ``cells`` / ``rng`` are read lazily at :meth:`build` time, so an
    organ can replace ``organ.vascular_cells`` between steps and later fills still
    land in the current manager (pass ``cells`` as the organ's attribute holder
    via :meth:`bind`, or re-bind).
    """

    def __init__(
        self,
        steps: Optional[List[TissueStep]] = None,
        *,
        cells: Optional[CellManager] = None,
        rng=None,
    ):
        self.steps: List[TissueStep] = list(steps) if steps else []
        self._cells = cells
        self._rng = rng

    def bind(self, cells, rng) -> "TissueRecipe":
        """Set the target CellManager + rng used by :meth:`fill` / :meth:`fill_each`.

        ``cells`` may be a :class:`CellManager` or a zero-arg callable returning
        one — pass a callable when the organ replaces its manager between steps,
        so each fill resolves the *current* manager at build time.
        """
        self._cells = cells
        self._rng = rng
        return self

    def _target(self) -> CellManager:
        return self._cells() if callable(self._cells) else self._cells

    def add(self, name: str, fn: Callable[[], None], *, produces: Tuple[str, ...] = ()) -> "TissueRecipe":
        self.steps.append(TissueStep(name, fn, produces=produces, kind="add"))
        return self

    # -- declarative steps ---------------------------------------------------
    def fill(
        self,
        name: str,
        tissue: "Tissue",
        *,
        strategy: str = "packing",
        produces: Optional[Tuple[str, ...]] = None,
        record: Optional[Callable[["Tissue", object], None]] = None,
        **kw,
    ) -> "TissueRecipe":
        """Add a step that fills one region ``tissue`` by ``strategy``.

        ``record(tissue, result)`` (optional) runs after the fill with whatever
        the strategy returned, for mask / polygon bookkeeping.  ``produces``
        defaults to ``(tissue.tag,)``.
        """
        def _run(tissue=tissue, record=record, strategy=strategy, kw=kw):
            result = _dispatch_fill(self._target(), tissue, strategy, self._rng, **kw)
            if record is not None:
                record(tissue, result)

        self.steps.append(TissueStep(
            name, _run, produces=produces if produces is not None else (tissue.tag,),
            kind="fill",
        ))
        return self

    def fill_each(
        self,
        name: str,
        tissues,
        *,
        strategy: str = "packing",
        produces: Optional[Tuple[str, ...]] = None,
        record: Optional[Callable[["Tissue", object], None]] = None,
        **kw,
    ) -> "TissueRecipe":
        """Add a step that fills several regions ``tissues`` by ``strategy``.

        ``tissues`` may be an iterable or a zero-arg callable returning one
        (deferred until build time, e.g. when the regions depend on an earlier
        step).  ``produces`` defaults to the tags of the regions.
        """
        def _run(tissues=tissues, record=record, strategy=strategy, kw=kw):
            seq = tissues() if callable(tissues) else tissues
            for tissue in seq:
                result = _dispatch_fill(self._target(), tissue, strategy, self._rng, **kw)
                if record is not None:
                    record(tissue, result)

        if produces is None and not callable(tissues):
            produces = tuple(dict.fromkeys(t.tag for t in tissues))
        self.steps.append(TissueStep(name, _run, produces=produces or (), kind="fill_each"))
        return self

    def cleanup(self, name: str, fn: Callable[[], None]) -> "TissueRecipe":
        """Add a cell/group-level cleanup step (produces nothing new)."""
        self.steps.append(TissueStep(name, fn, kind="cleanup"))
        return self

    def special(self, name: str, fn: Callable[[], None], *, produces: Tuple[str, ...] = ()) -> "TissueRecipe":
        """Add a bespoke placement step (sheath, bundles, ...) that isn't a plain fill."""
        self.steps.append(TissueStep(name, fn, produces=produces, kind="special"))
        return self

    def build(self) -> None:
        """Run every step in order."""
        for step in self.steps:
            step.run()

    def describe(self) -> List[Tuple[str, Tuple[str, ...]]]:
        """Return ``(name, produces)`` for each step, for inspection / preview."""
        return [(s.name, s.produces) for s in self.steps]

    def plan(self) -> List[Tuple[str, str, Tuple[str, ...]]]:
        """Return ``(name, kind, produces)`` for each step — the full plan.

        Richer than :meth:`describe` (which is kept name+produces for
        back-compat); ``kind`` distinguishes fill / fill_each / cleanup / special
        / add so a preview can render *how* each tissue is produced.
        """
        return [(s.name, s.kind, s.produces) for s in self.steps]

    def format_plan(self) -> str:
        """Render the plan as an indented, human-readable block."""
        lines = []
        for s in self.steps:
            tags = (" -> " + ", ".join(s.produces)) if s.produces else ""
            lines.append(f"  [{s.kind}] {s.name}{tags}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"TissueRecipe({len(self.steps)} steps)"

    def __iter__(self):
        return iter(self.steps)

    def __len__(self):
        return len(self.steps)
