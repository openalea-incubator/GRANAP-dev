"""Vascular bundle construction (shape-first, organ-agnostic).

A vascular bundle is built as three things:

1. an **envelope** — an oriented footprint at ``(cx, cy)`` with radial orientation ``theta`` (local +y = radial, pointing toward the organ surface);
2. an **internal partition** into tissue sub-zones — *the bundle type is the topology of this partition*;
3. a **per-zone fill** with the right cell type via the existing ``fill_*`` primitives.

Two partition families cover the four botanical types:

* **banded** (``collateral`` / ``bicollateral``): parallel cuts perpendicular to
  the radial axis -> stacked bands (xylem / cambium / phloem, inner->outer).
* **concentric** (``concentric`` + ``amphivasal`` / ``amphicribral``): a core +
  a ring (one tissue surrounding the other).

The monocot xylem "face" (``xylem_layout="face"``) is a bespoke fill over the
whole bundle region: metaxylem vessel(s) at the radial middle, a protoxylem
bundle (+ optional lacuna void) in the inner half, and the phloem cluster in the
outer half — all positioned relative to the bundle centre.  Metaxylem and
protoxylem are both tagged ``xylem`` (no distinction).

Placement (how many bundles and where — a eustele ring vs a scattered
atactostele) is the caller's job; this module builds one bundle at a time.

Convention for cavities (protoxylem lacuna, and the pith cavity elsewhere): a
cavity is a polygon with **no cells**; the whole bundle **envelope** is returned
for the removal mask, so a lacuna inside it is already cleared of ground seeds.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from shapely.affinity import translate as _shapely_translate
from shapely.geometry import LineString, Point, Polygon, box
from shapely.prepared import prep

from openalea.granap.cell_class import Cell
from openalea.granap.cell_manager import CellManager
from openalea.granap.geometry_collection import GeometryProcessor
from shapely.ops import unary_union

from openalea.granap.tissue_class import fill_by_packing, fill_by_rings, fill_along


@dataclass
class BundleResult:
    """What one built bundle exposes to the caller for mask / view / bookkeeping."""
    envelope: Optional[Polygon] = None          # whole footprint -> removal mask
    vessel_polygons: List[Polygon] = field(default_factory=list)   # xylem vessels
    cavity_polygons: List[Polygon] = field(default_factory=list)   # lacuna voids
    # (role, region) per filled sub-zone — for the tissue-polygon view. Each is a
    # subset of the envelope, so registering them never enlarges the removal mask.
    zone_polygons: List[Tuple[str, Polygon]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _largest(geom):
    """Largest Polygon piece of a (possibly Multi)Polygon, or None."""
    if geom is None or geom.is_empty:
        return None
    pieces = [g for g in (geom.geoms if hasattr(geom, "geoms") else [geom])
              if g.geom_type == "Polygon" and not g.is_empty]
    return max(pieces, key=lambda g: g.area) if pieces else None


def _local_envelope(width: float, height: float, shape: str,
                    focus_exponent: float = 4.0, egg_waist: float = 0.6) -> Polygon:
    """Envelope in the local frame: centred at the origin, radial axis = +y.

    ``ellipse`` / ``circle`` as before; ``focus_ellipse`` is a superellipse
    (``focus_exponent`` tunes flank fullness at a fixed bounding box); ``egg`` is
    a teardrop whose wider lobe is offset toward the outer (+y) pole by
    ``egg_waist`` (the outer lobe's share of the radial extent)."""
    if shape == "circle":
        d = min(width, height)
        return Point(0, 0).buffer(d / 2, resolution=64)
    if shape == "focus_ellipse":
        # Superellipse: width along x (tangential), height along y (radial).
        return GeometryProcessor.focus_ellipse_polygon(
            0.0, 0.0, width / 2, height / 2, 0.0, exponent=focus_exponent)
    if shape == "egg":
        # Teardrop: the a_out lobe (outer share = egg_waist) points to +y (radial
        # surface pole); b = semi-minor (tangential).
        a_out = height * egg_waist
        a_in = height * (1.0 - egg_waist)
        return GeometryProcessor.egg_polygon(0.0, 0.0, a_out, a_in, width / 2, 90.0)
    return GeometryProcessor.oriented_ellipse(0.0, 0.0, width, height, 90.0)


def _tissue_size_fn(ground_cell_size):
    """Normalise ``ground_cell_size`` (scalar, callable ``f(x,y)``, or None) into a
    ``f(x, y) -> size or None`` used to read the local tissue cell size."""
    if ground_cell_size is None:
        return None
    if callable(ground_cell_size):
        def _f(x, y):
            try:
                v = float(ground_cell_size(x, y))
            except Exception:
                return None
            return v if v > 0 else None
        return _f
    val = float(ground_cell_size)
    return (lambda x, y: val) if val > 0 else None


def _tissue_name_fn(ground_tissue_name):
    """Normalise ``ground_tissue_name`` (callable ``f(x, y) -> name`` or None) into a
    ``f(x, y) -> name or None`` used to name an outer-sheath cell after the tissue it
    sits in.  Anything that isn't callable turns the naming off (the cells keep the
    generic ``bundle sheath`` tag)."""
    if ground_tissue_name is None or not callable(ground_tissue_name):
        return None

    def _f(x, y):
        try:
            name = ground_tissue_name(x, y)
        except Exception:
            return None
        return name or None
    return _f


def _majority_tissue_name(name_fn, px: float, py: float, r: float) -> Optional[str]:
    """Tissue name occupying most of a sheath cell centred at ``(px, py)`` of radius
    ``r``.  The cell is a small disk that can straddle a tissue boundary, so the name
    is voted over its centre plus a ring of samples inside it (the majority wins;
    ties break toward the nearer/earlier sample)."""
    names = [name_fn(px, py)]
    for a in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
        names.append(name_fn(px + r * np.cos(a), py + r * np.sin(a)))
    names = [n for n in names if n]
    if not names:
        return None
    return Counter(names).most_common(1)[0][0]


def outer_sheath_mask_pad(bp: dict, ground_cell_size: Optional[float]) -> float:
    """Conservative outward reservation for a *candidate* bundle placement.

    The real outer bundle sheath is grown cell by cell against the local tissue
    (see :func:`_grow_bundle_sheath`) and its exact reach isn't known until the
    bundle is built, so placement/overlap tests reserve a fixed, modest pad —
    about one local tissue cell — around the footprint.  This keeps bundles a
    tissue-cell apart without the old mean-sized ring's over-reservation.  ``0``
    when the outer sheath is off.  ``ground_cell_size`` may be a scalar (a callable
    is treated as off here — placement passes the local scalar explicitly).
    """
    if not bp.get("outer_sheath", True):
        return 0.0
    if not ground_cell_size or callable(ground_cell_size) or float(ground_cell_size) <= 0:
        return 0.0
    return float(ground_cell_size)


#: A bundle-sheath cell is inserted only where the bundle cell and the tissue cell
#: differ by more than this size ratio; below it the neighbours already match.
_SHEATH_MIN_RATIO = 4.0


def _place_ring(cells: CellManager, pcx: float, pcy: float, r_draw: float,
                tag: str, angle_center, n_border: int = 12) -> None:
    """Seed one round cell as a ring of ``n_border`` (>= 8) border points — one
    Voronoi group.  Analytic (no shapely buffering), so it is cheap for the many
    small cells of a bundle sheath while still reading as a circle."""
    if r_draw <= 0:
        return
    br = r_draw * 0.85
    gid = cells.next_group_id()
    for a in np.linspace(0.0, 2.0 * np.pi, max(8, n_border), endpoint=False):
        cells.add_cell(Cell.radial(
            tag, pcx + br * np.cos(a), pcy + br * np.sin(a), r_draw * 2.0, gid, angle_center))


def _grow_bundle_sheath(cells: CellManager, foot: Polygon, bp: dict,
                        ground_cell_size, outline, cx: float, cy: float,
                        env=None, ground_tissue_name=None) -> Polygon:
    """Place a single outer bundle-sheath ring and return the removal mask
    (footprint ∪ placed sheath cells, clipped to ``outline``).

    One cell thick, sized *per position* to smooth the size jump between the bundle
    and the tissue it sits in.  For each position around the footprint, ``x`` is the
    bundle's outer cell there — the fibre ``sclerenchyma_cell_diameter`` where the
    boundary point sits on a fibre cap (outside the envelope ``env``), else the
    ``parenchyma_diameter`` — and ``y`` is the tissue cell size at that cell's own
    location (``ground_cell_size(x, y)``):

    * if ``max(x, y) / min(x, y) <= _SHEATH_MIN_RATIO`` the neighbours already match
      — place nothing there;
    * otherwise place one cell of size ``sqrt(x * y)`` — the *geometric* mean, so the
      size ratio to each neighbour is the same ``sqrt(y / x)`` (a balanced step,
      unlike the arithmetic mean which skews toward the coarser cell).

    Sizing ``x`` from the fibres at a cap keeps the sheath cell small and tight
    against them, so the edge fibres have a neighbour to bound their Voronoi region
    instead of stretching across an empty band into the (coarser) parenchyma-sized
    sheath.  Cells are spaced ~their own size tangentially, so a coarse-tissue side
    carries a few large sheath cells and a fine side many small ones — the ring is
    asymmetric.  Positions outside ``outline`` (the organ's last layer polygon) are
    skipped, so the sheath never leaves the stem.

    Each placed cell is *named after the tissue it is fitted into* rather than a
    generic ``bundle sheath``: ``ground_tissue_name(x, y)`` (when the organ supplies
    it) is voted over the cell's footprint and the majority tissue wins (see
    :func:`_majority_tissue_name`).  With no name function the cells keep the
    ``bundle sheath`` tag.
    """
    tissue = _tissue_size_fn(ground_cell_size)
    if tissue is None:
        return foot
    name_fn = _tissue_name_fn(ground_tissue_name)
    parench = float(bp.get("parenchyma_diameter", 0.012))    # bundle interior cell
    fibre = float(bp.get("sclerenchyma_cell_diameter", 0.008))
    # A cap sticks out past the envelope, so a boundary point outside ``env`` sits on
    # fibres; size the sheath from the fibre there, else from the parenchyma.
    in_env = prep(env).contains if env is not None else (lambda p: True)
    inside = prep(outline).contains if outline is not None else (lambda p: True)

    # Sample the footprint boundary very finely with outward normals, then *march*
    # along it placing cells.  The resolution is tied to the smallest cell that can
    # appear (a fibre), not the coarser parenchyma: a grid tied to the coarse cell
    # cannot reach the tight spacing a fine-tissue side needs, so the spacing floors
    # out and reads as uniform everywhere — the very thing to avoid here.
    ring = foot.exterior
    step_ref = max(0.25 * min(parench, fibre), 1e-4)
    n = max(16, int(np.ceil(ring.length / step_ref)))
    coords = GeometryProcessor.resample_coords(np.array(ring.coords), target_n_points=n)
    pts = coords[:-1] if np.allclose(coords[0], coords[-1]) else coords
    cen = np.array([foot.centroid.x, foot.centroid.y])
    m = len(pts)

    # Walk the boundary accumulating *arc length* since the last placed cell and drop
    # the next cell once that arc reaches this cell's own diameter — each cell reserves
    # a gap equal to its own size, so a big cell (coarse tissue) leaves a wide gap and a
    # small cell (fine tissue) a tight one, sized and spaced from its own local tissue
    # rather than blended with its neighbour.  The target is recomputed at every step
    # from the local cell size, so the spacing follows the tissue precisely instead of
    # the near-constant spacing a fixed Euclidean grid produced.
    placed = []                                             # (px, py, size)
    arc = np.inf                                            # inf -> place the first eligible cell
    for i in range(m):
        if i > 0:
            arc += float(np.hypot(*(pts[i] - pts[i - 1])))

        b = pts[i]
        t = pts[(i + 1) % m] - pts[i - 1]
        nrm = np.array([t[1], -t[0]])
        if np.dot(nrm, b - cen) < 0:
            nrm = -nrm
        nrm = nrm / (np.hypot(*nrm) or 1.0)

        # bundle-side neighbour: fibre on a cap (boundary point outside the
        # envelope), else the interior parenchyma.
        x = fibre if not in_env(Point(b)) else parench
        probe = b + nrm * (0.6 * x)                         # just outside the footprint
        if not inside(Point(probe)):
            continue
        y = tissue(probe[0], probe[1])
        if y is None or min(x, y) <= 0:
            continue
        if max(x, y) / min(x, y) <= _SHEATH_MIN_RATIO:      # neighbours already match
            continue
        s = float(np.sqrt(x * y))                           # geometric mean
        if arc < s:                                         # not this cell's own diameter yet
            continue
        c = b + nrm * (0.5 * s)
        if not inside(Point(c)):
            continue
        placed.append((c[0], c[1], s))
        arc = 0.0

    for px, py, s in placed:
        # Name the cell after the tissue it is fitted into (majority over its
        # footprint); fall back to the generic tag when no name function is given.
        tag = "bundle sheath"
        if name_fn is not None:
            tag = _majority_tissue_name(name_fn, px, py, s * 0.5) or tag
        _place_ring(cells, px, py, s * 0.5, tag, (cx, cy))

    mask = unary_union([foot] + [Point(px, py).buffer(s * 0.6) for px, py, s in placed])
    if outline is not None:
        mask = mask.intersection(outline)
    return _largest(mask) or foot


# ---------------------------------------------------------------------------
# Layout — the topology per bundle type
# ---------------------------------------------------------------------------

def bundle_layout(bp: dict):
    """Resolve bundle params into a partition spec.

    Returns ``("banded", [(role, fraction), ...])`` inner->outer, or
    ``("concentric", (core_role, ring_role))``.
    """
    bt = bp.get("bundle_type", "collateral")

    if bt == "concentric":
        if bp.get("concentric_type", "amphivasal") == "amphivasal":
            return "concentric", ("phloem", "xylem")      # xylem rings a phloem core
        return "concentric", ("xylem", "phloem")          # phloem rings a xylem core

    xf = bp.get("xylem_fraction", 0.5)
    pf = bp.get("phloem_fraction", 0.35)
    cf = bp.get("cambium_fraction", 0.08)
    ipf = bp.get("inner_phloem_fraction", 0.0)

    has_cambium = bp.get("has_cambium", True)
    bands: List[Tuple[str, float]] = []
    if bt == "bicollateral" and ipf > 0.0:
        bands.append(("phloem", ipf))                     # inner phloem
        # Bicollateral bundles carry a cambium strip on the inner phloem side too
        # by default (both faces of the xylem), unless inner_cambium is disabled.
        if has_cambium and bp.get("inner_cambium", True):
            bands.append(("cambium", cf))
    bands.append(("xylem", xf))                           # xylem (inner of outer phloem)
    if has_cambium:
        bands.append(("cambium", cf))
    bands.append(("phloem", pf))                          # outer phloem

    # Local +y is radial-outward; inner->outer = bottom->top. Flip for phloem_inward.
    if not bp.get("phloem_outward", True):
        bands = list(reversed(bands))
    return "banded", bands


def partition_banded(env: Polygon, bands) -> List[Tuple[str, Polygon]]:
    """Cut ``env`` into horizontal strips (inner->outer along local +y)."""
    minx, miny, maxx, maxy = env.bounds
    span = (maxy - miny) or 1.0
    total = sum(f for _, f in bands) or 1.0
    out, y = [], miny
    for role, frac in bands:
        y1 = y + span * frac / total
        strip = env.intersection(box(minx - 1, y, maxx + 1, y1))
        out.append((role, strip))
        y = y1
    return out


def partition_concentric(env: Polygon, core_role: str, ring_role: str,
                         core_w: float, core_h: float) -> List[Tuple[str, Polygon]]:
    """Split ``env`` into a central core ellipse and the surrounding ring."""
    core = GeometryProcessor.oriented_ellipse(0.0, 0.0, core_w, core_h, 90.0).intersection(env)
    ring = env.difference(core)
    return [(ring_role, ring), (core_role, core)]


# ---------------------------------------------------------------------------
# Cell-level placement
# ---------------------------------------------------------------------------

def _place_circle(cells: CellManager, pcx: float, pcy: float, r_draw: float,
                  tag: str, angle_center, n_border: int = 22) -> Optional[Polygon]:
    """Seed one conducting cell: a ring of border points = one Voronoi group.

    Returns the drawn circle polygon (for the parenchyma to hug), or None."""
    placed = Point(pcx, pcy).buffer(r_draw, resolution=32)
    buff = _largest(placed.buffer(-r_draw * 0.15))
    if buff is None:
        return None
    xs, ys = buff.exterior.coords.xy
    coords = GeometryProcessor.resample_coords(np.column_stack((xs, ys)), target_n_points=n_border)
    if len(coords) < 2:
        return None
    gid = cells.next_group_id()
    for pt in coords[1:]:
        cells.add_cell(Cell.radial(tag, pt[0], pt[1], r_draw * 2, gid, angle_center))
    return placed


def _place_region_cell(cells: CellManager, region, tag: str, angle_center,
                       n_border: int = 18) -> Optional[Polygon]:
    """Seed one cell whose shape follows ``region`` (a ring of border points = one
    Voronoi group).  Used to drop a lacuna in as an ordinary ``air space`` cell.

    Returns the drawn polygon (for the parenchyma to hug), or None."""
    region = _largest(region)
    if region is None or region.is_empty:
        return None
    r = np.sqrt(region.area / np.pi)
    buff = _largest(region.buffer(-r * 0.15))
    if buff is None:
        return None
    xs, ys = buff.exterior.coords.xy
    coords = GeometryProcessor.resample_coords(np.column_stack((xs, ys)), target_n_points=n_border)
    if len(coords) < 2:
        return None
    gid = cells.next_group_id()
    for pt in coords[1:]:
        cells.add_cell(Cell.radial(tag, pt[0], pt[1], r * 2, gid, angle_center))
    return region


def _pack_place(cells, rng, zone, tag, cx, cy, *, voronoi_grow, r_floor,
                n_border=22, **pack_kw):
    """Pack circles in ``zone`` and seed each as a slightly-shrunk conducting cell.

    Each circle is drawn smaller than packed by the Voronoi growth a neighbouring
    parenchyma cell adds on render (floored at ``r_floor``), so it renders at its
    true size instead of ballooning into the surrounding parenchyma.  Returns the
    drawn circle polygons and their ``(cx, cy, r_draw)`` centres.
    """
    zone = _largest(zone)
    if zone is None:
        return [], []
    packed = GeometryProcessor.pack_circles(zone, rng=rng, **pack_kw)
    polys, centers = [], []
    for pcx, pcy, r in packed:
        r_draw = min(r, max(r - voronoi_grow / 3, r_floor))
        placed = _place_circle(cells, pcx, pcy, r_draw, tag, (cx, cy), n_border)
        if placed is not None:
            polys.append(placed)
            centers.append((pcx, pcy, r_draw))
    return polys, centers


def _place_companions(cells, rng, zone, sieve_centers, comp_d, comp_w, voronoi_grow, cx, cy):
    """One companion cell beside each sieve element (rejecting sieve overlaps).

    Mirrors the dicot-root secondary phloem: a companion is tried on random sides
    of its sieve and kept on the first free one, so every sieve element is paired.
    Each companion is an oriented cell — radial extent ``comp_d`` x tangential
    extent ``comp_w`` — aligned with the radial direction at its location.
    """
    zone = _largest(zone)
    if zone is None or not sieve_centers:
        return []
    sieve_union = unary_union([Point(c[0], c[1]).buffer(c[2]) for c in sieve_centers])
    comp_d = max(comp_d - voronoi_grow, comp_d * 0.5)
    comp_w = max(comp_w - voronoi_grow, comp_w * 0.5)
    off = max(comp_d, comp_w) / 2      # centre offset from the sieve edge
    comps = []
    for pcx, pcy, r in sieve_centers:
        theta0 = rng.uniform(0.0, 2 * np.pi)
        for k in range(8):
            a = theta0 + 2 * np.pi * k / 8
            ccx, ccy = pcx + (r + off * 1.05) * np.cos(a), pcy + (r + off * 1.05) * np.sin(a)
            if not zone.contains(Point(ccx, ccy)):
                continue
            # Radial axis at this cell = away from the organ centre (cx, cy).
            ang = np.degrees(np.arctan2(ccy - cy, ccx - cx))
            cell_poly = GeometryProcessor.oriented_ellipse(ccx, ccy, comp_w, comp_d, ang)
            if cell_poly.intersects(sieve_union) or any(cell_poly.intersects(c) for c in comps):
                continue
            placed = _place_region_cell(cells, cell_poly.intersection(zone),
                                        "companion cell", (cx, cy), n_border=14)
            if placed is not None:
                comps.append(cell_poly)
            break
    return comps


def _fill_parenchyma(cells, zone, occupied, tag, cx, cy, p_diam, p_width):
    """Ring-fill ``zone`` minus the conducting-cell footprints, hugging the holes.

    Hugging (``initial_space=0``) is what keeps the vessels / sieves / companions
    at their true size: the default erosion leaves a cell-wide moat around each
    that its Voronoi cell would then balloon into.
    """
    zone = _largest(zone)
    if zone is None:
        return
    fill_zone = zone.difference(occupied) if (occupied is not None and not occupied.is_empty) else zone
    if fill_zone.is_empty:
        return
    fill_by_rings(cells, fill_zone, p_diam, p_width, tag, cx, cy,
                  cells.next_group_id(), erosion_polygon=fill_zone, initial_space=0.0)


def _fill_xylem_packed(cells, rng, zone, cx, cy, xylem, bp, result) -> None:
    """Dicot / concentric xylem: size-graded vessels + xylem parenchyma around them."""
    p_diam = bp.get("parenchyma_diameter", 0.012)
    p_w = bp.get("parenchyma_width", 0.012)
    voronoi_grow = 0.25 * (p_diam + p_w)
    vessels, _ = _pack_place(
        cells, rng, zone, "xylem", cx, cy,
        voronoi_grow=voronoi_grow, r_floor=p_diam * 0.4, n_border=25,
        proportion=bp.get("prop_vessel", 0.55),
        direction=xylem.get("direction", "center"),
        diameter_max=xylem.get("vessel_diameter", 0.06),
        diameter_min=xylem.get("vessel_diameter_min", 0.02),
        diameter_sd=xylem.get("vessel_diameter_sd", 0.003),
        gradient_function=xylem.get("gradient_function", "five_pl"),
        gradient_inflection=xylem.get("gradient_inflection", 0.5),
        gradient_steepness=xylem.get("gradient_steepness", 3.0),
        gradient_asymmetry=xylem.get("gradient_asymmetry", 1.0),
    )
    result.vessel_polygons.extend(vessels)
    _fill_parenchyma(cells, zone, unary_union(vessels) if vessels else None,
                     "parenchyma", cx, cy, p_diam, p_w)


def _xylem_file_strips(zone, theta, n_files, half_width, rng, jitter):
    """Thin radial parenchyma-separator strips that cut the xylem zone into
    ``n_files`` tangential compartments (files), lightly jittered.

    Each strip runs along the bundle's radial axis (``theta``, pith->cambium) at an
    evenly-spaced tangential offset, overshooting the zone radially so it cuts all
    the way across.  ``jitter`` (0 = a rigid grid) perturbs each strip's tangential
    position and angle a little so the files don't read as mechanical rows.  Built
    *before* the vessels — the medullar-ray idea — so the vessel packer fills the
    compartments between them instead of one open blob.
    """
    if n_files <= 1:
        return []
    gx, gy = zone.centroid.x, zone.centroid.y
    outer = np.array([np.cos(theta), np.sin(theta)])            # radial (pith->cambium)
    tang = np.array([-np.sin(theta), np.cos(theta)])            # tangential
    w = _radial_half(zone, gx, gy, tang)                        # tangential half-width
    reach = _radial_half(zone, gx, gy, outer) + 2.0 * half_width
    step = 2.0 * w / n_files
    strips = []
    for i in range(1, n_files):
        t = -w + step * i
        t += rng.uniform(-1.0, 1.0) * jitter * 0.5 * step       # nudge the position
        dang = rng.uniform(-1.0, 1.0) * jitter * np.radians(12.0)  # tilt the file a touch
        d = np.array([np.cos(theta + dang), np.sin(theta + dang)])
        center = np.array([gx, gy]) + tang * t
        strip = LineString([tuple(center - d * reach), tuple(center + d * reach)]).buffer(
            half_width, cap_style=2)
        piece = strip.intersection(zone)
        if not piece.is_empty:
            strips.append(piece)
    return strips


def _fill_xylem_files(cells, rng, zone, cx, cy, theta, xylem, bp, result) -> None:
    """Dicot xylem as endarch radial files (the medullar-ray idea, parenchyma-first).

    Thin radial parenchyma strips are placed *before* the vessels and cut the xylem
    zone into ``n_xylem_files`` tangential compartments.  Each compartment is packed
    with size-graded vessels referenced to the organ centre and graded **endarch** —
    small protoxylem at the inner (pith) tip, large metaxylem toward the cambium.
    Parenchyma then fills the strips and the space around the vessels, so the
    vessels sit in natural rows separated by parenchyma instead of one dense clump
    of touching circles.  The files are lightly jittered (``xylem_file_jitter``).
    """
    zone = _largest(zone)
    if zone is None:
        return
    p_diam = bp.get("parenchyma_diameter", 0.012)
    p_w = bp.get("parenchyma_width", 0.012)
    voronoi_grow = 0.25 * (p_diam + p_w)

    n_files = max(int(bp.get("n_xylem_files", 3)), 1)
    jitter = float(bp.get("xylem_file_jitter", 0.3))
    strips = _xylem_file_strips(zone, theta, n_files, 0.6 * p_w, rng, jitter)
    strip_union = unary_union(strips) if strips else None
    compartments = zone.difference(strip_union) if strip_union is not None else zone
    pieces = [g for g in (compartments.geoms if hasattr(compartments, "geoms") else [compartments])
              if g.geom_type == "Polygon" and not g.is_empty]

    # Endarch size gradient measured *along the radial axis* (pith->cambium).  The
    # bundle's ``(cx, cy)`` is the bundle centre, not the organ centre, so grade
    # against a proxy centre far down the -radial (pith-ward) direction: distance
    # from it grows monotonically toward the cambium.  ``radial_range`` normalises
    # the gradient over the xylem band so it actually bites (the band is thin
    # relative to that distance), and ``direction="edge"`` puts the small
    # protoxylem at the inner tip and the large metaxylem toward the cambium.
    outer = np.array([np.cos(theta), np.sin(theta)])
    gcenter = (cx - 100.0 * outer[0], cy - 100.0 * outer[1])
    dc = np.hypot(*(np.asarray(zone.exterior.coords).T - np.array([[gcenter[0]], [gcenter[1]]])))
    radial_range = (float(dc.min()), float(dc.max()))

    vessels = []
    for piece in pieces:
        vs, _ = _pack_place(
            cells, rng, piece, "xylem", cx, cy,
            voronoi_grow=voronoi_grow, r_floor=p_diam * 0.4, n_border=25,
            proportion=bp.get("prop_vessel", 0.55),
            direction="edge",                       # endarch: large toward the cambium
            gradient_center=gcenter,                # grade along the radial (pith->cambium) axis
            gradient_radial_range=radial_range,     # ...over the xylem band, so the gradient bites
            diameter_max=xylem.get("vessel_diameter", 0.045),
            diameter_min=xylem.get("vessel_diameter_min", 0.012),
            diameter_sd=xylem.get("vessel_diameter_sd", 0.003),
            gradient_function=xylem.get("gradient_function", "five_pl"),
            gradient_inflection=xylem.get("gradient_inflection", 0.5),
            gradient_steepness=xylem.get("gradient_steepness", 3.0),
            gradient_asymmetry=xylem.get("gradient_asymmetry", 1.0),
        )
        vessels.extend(vs)
    result.vessel_polygons.extend(vessels)
    _fill_parenchyma(cells, zone, unary_union(vessels) if vessels else None,
                     "parenchyma", cx, cy, p_diam, p_w)


def _fill_xylem_face(cells, rng, zone, cx, cy, theta, bp, phloem, result) -> None:
    """The monocot 'face' bundle, arranged about the bundle centre along +radial.

    * **metaxylem** — ``n_metaxylem`` single vessels at the radial *middle*
      (placed first), spread tangentially; each sized ``metaxylem_diameter`` (+-
      ``_sd``, clipped to ``_diameter_min``) and tagged ``xylem``;
    * **protoxylem** — ``n_protoxylem`` small bundle region(s) in the *inner*
      (centre-facing) half at ``protoxylem_relative_distance``, each an ellipse
      (``protoxylem_width`` x ``protoxylem_height``) packed with protoxylem
      vessels, also tagged ``xylem``;
    * an optional **lacuna** just inner of each protoxylem bundle (a void seeded
      as one ``air space`` cell), kept with the protoxylem inside the inner half;
    * the **phloem** cluster (sieve elements + one companion cell beside each) in
      the *outer* (surface-facing) half at ``phloem_relative_distance``.

    All radial offsets are measured from the bundle centre, so the inner and
    outer halves never collide.  Vessels are seeded a touch undersized so the
    surrounding parenchyma's Voronoi growth brings them back to size.
    """
    zone = _largest(zone)
    if zone is None:
        return
    gx, gy = zone.centroid.x, zone.centroid.y
    theta_deg = np.degrees(theta)
    outer = np.array([np.cos(theta), np.sin(theta)])       # radial, toward surface
    tang = np.array([-np.sin(theta), np.cos(theta)])
    h = _radial_half(zone, gx, gy, outer)                  # radial half-height of the bundle
    # phloem_inward flips which pole faces the surface (keeps the banded convention).
    if not bp.get("phloem_outward", True):
        outer = -outer

    p_diam = bp.get("parenchyma_diameter", 0.012)
    p_w = bp.get("parenchyma_width", 0.012)
    vgrow = 0.25 * (p_diam + p_w)                          # Voronoi over-grow to pre-shrink for
    margin = 1.2 * p_diam

    def at(s, t):                                          # radial offset s, tangential offset t
        return np.array([gx, gy]) + outer * s + tang * t

    def seed(pcx, pcy, r_eff, tag, n_border):
        r_draw = max(r_eff - vgrow / 2.0, r_eff * 0.55)    # undersize so it doesn't balloon
        return _place_circle(cells, pcx, pcy, r_draw, tag, (cx, cy), n_border)

    placed_polys, occupied = [], []

    # 1) Metaxylem — single vessel(s) at the radial middle, spread tangentially.
    dm = float(bp.get("metaxylem_diameter", 0.04))
    dm_sd = float(bp.get("metaxylem_diameter_sd", 0.003))
    dm_min = float(bp.get("metaxylem_diameter_min", dm * 0.5))
    n_meta = int(bp.get("n_metaxylem", 2))
    m_gap = float(bp.get("metaxylem_gap", 0.04))
    for k in range(n_meta):
        d = float(np.clip(rng.normal(dm, dm_sd), dm_min, np.inf))
        poly = _largest(Point(*at(0.0, (k - (n_meta - 1) / 2.0) * (dm + m_gap)))
                        .buffer(d / 2, resolution=32).intersection(zone))
        if poly is None:
            continue
        r_eff = np.sqrt(poly.area / np.pi)
        placed = seed(poly.centroid.x, poly.centroid.y, r_eff, "xylem", 24)
        if placed is not None:
            placed_polys.append(placed)
            occupied.append(placed)

    # 2) Protoxylem bundle(s) (+ optional inner lacuna) in the inner half.
    px_w = float(bp.get("protoxylem_width", 0.03))
    px_h = float(bp.get("protoxylem_height", 0.03))
    px_rel = float(bp.get("protoxylem_relative_distance", 0.6))
    n_proto = int(bp.get("n_protoxylem", 1))
    add_lacuna = bool(bp.get("lacuna", False))
    lac_w = float(bp.get("lacuna_width", 0.03))
    lac_h = float(bp.get("lacuna_height", 0.02)) if add_lacuna else 0.0

    # Keep the protoxylem region (and its inner lacuna) within the inner half
    # [-h + margin, 0]: pull the centre in until region + lacuna fit.
    # The lacuna sits flush against the protoxylem's inner edge (no gap), so the
    # inner-half budget is exactly the region plus the lacuna.
    inner_span = max(h - margin, px_h / 2.0)
    lac_reach = lac_h if add_lacuna else 0.0
    s_px_max = max(inner_span - px_h / 2.0 - lac_reach, px_h / 2.0)
    s_px = min(px_rel * inner_span, s_px_max)
    for k in range(n_proto):
        t = (k - (n_proto - 1) / 2.0) * (px_w + margin)
        pc = at(-s_px, t)
        region = _largest(GeometryProcessor.oriented_ellipse(pc[0], pc[1], px_w, px_h, theta_deg)
                          .intersection(zone))
        if region is not None and not region.is_empty:
            vessels, _ = _pack_place(
                cells, rng, region, "xylem", cx, cy,
                voronoi_grow=vgrow, r_floor=p_diam * 0.4, n_border=18,
                proportion=bp.get("prop_vessel", 0.55), direction=None,
                diameter_max=float(bp.get("protoxylem_diameter", 0.012)),
                diameter_min=float(bp.get("protoxylem_diameter_min", 0.006)),
                diameter_sd=float(bp.get("protoxylem_diameter_sd", 0.0015)),
                gradient_function="normal",
            )
            if not vessels:
                # Region too small for the packer to fit a vessel — a small
                # protoxylem bundle is still a bundle: seed one vessel on its
                # inscribed circle so it always renders.
                ix, iy, ir = GeometryProcessor.get_inscribed_circle(region)
                placed = seed(ix, iy, ir, "xylem", 18) if ir > 0 else None
                if placed is not None:
                    vessels = [placed]
            placed_polys.extend(vessels)
            occupied.extend(vessels)
        if add_lacuna:
            # Flush against the protoxylem region's inner edge.
            lc = at(-(s_px + px_h / 2.0 + lac_h / 2.0), t)
            lac = _largest(GeometryProcessor.oriented_ellipse(lc[0], lc[1], lac_w, lac_h, theta_deg)
                           .intersection(zone))
            placed_lac = (_place_region_cell(cells, lac, "air space", (cx, cy), n_border=18)
                          if lac is not None and not lac.is_empty else None)
            if placed_lac is not None:
                occupied.append(placed_lac)

    # 3) Phloem cluster in the outer half, at phloem_relative_distance.
    ph_w = float(bp.get("phloem_width", phloem.get("cluster_width", 0.05)))
    ph_h = float(bp.get("phloem_height", phloem.get("cluster_height", 0.04)))
    ph_rel = float(bp.get("phloem_relative_distance", 0.5))
    outer_span = max(h - margin, ph_h / 2.0)
    # ``phloem_relative_distance`` (0..1) slides the cluster through the outer half,
    # from just outside the bundle centre (0) to the envelope edge (1).
    s_ph = min(max(ph_rel * outer_span, ph_h / 2.0), outer_span - ph_h / 2.0)
    pc = at(s_ph, 0.0)
    ell = _largest(GeometryProcessor.oriented_ellipse(pc[0], pc[1], ph_w, ph_h, theta_deg)
                   .intersection(zone))
    # The phloem is placed freely, but its cells must not land inside the xylem:
    # carve the already-placed vessels / lacunae out of the cluster region so no
    # sieve element or companion cell is ever seeded over a metaxylem/protoxylem.
    if ell is not None and not ell.is_empty and occupied:
        ell = _largest(ell.difference(unary_union(occupied)))
    if ell is not None and not ell.is_empty:
        ph_occ = _place_phloem_cells(cells, rng, ell, cx, cy, phloem, bp)
        if ph_occ is not None:
            occupied.append(ph_occ)

    result.vessel_polygons.extend(placed_polys)
    _fill_parenchyma(cells, zone, unary_union(occupied) if occupied else None,
                     "parenchyma", cx, cy, p_diam, p_w)


def _radial_half(zone: Polygon, cx0: float, cy0: float, axis) -> float:
    """Half-extent of ``zone`` projected onto ``axis`` (about ``(cx0, cy0)``)."""
    coords = np.column_stack(zone.exterior.coords.xy) - np.array([cx0, cy0])
    proj = coords @ np.asarray(axis)
    return 0.5 * float(proj.max() - proj.min())


def _place_phloem_cells(cells, rng, sieve_zone, cx, cy, phloem, bp):
    """Pack sieve elements + a companion cell beside each into ``sieve_zone``.

    Returns the union of their footprints (or None) so the caller can fill
    parenchyma around them.  This is the phloem *cluster* — sieve elements +
    companion cells; the parenchyma is a separate fill.
    """
    p_diam = bp.get("parenchyma_diameter", 0.012)
    p_w = bp.get("parenchyma_width", 0.012)
    voronoi_grow = 0.25 * (p_diam + p_w)
    sieve_d = phloem.get("sieve_diameter", 0.012)
    sieve_min = bp.get("sieve_diameter_min", 0.006)
    comp_d = bp.get("companion_cell_diameter", 0.004)
    comp_w = bp.get("companion_cell_width", 0.004)
    prop_sieve = bp.get("prop_sieve", 0.45)
    # Fraction to pack with sieves so sieve+companion together ~= prop_sieve.
    proportion = prop_sieve * sieve_d ** 2 / (sieve_d ** 2 + comp_d * comp_w)
    sieves, centers = _pack_place(
        cells, rng, sieve_zone, "sieve element", cx, cy,
        voronoi_grow=voronoi_grow, r_floor=min(sieve_d / 2, p_diam * 0.4), n_border=16,
        proportion=proportion, direction=None,
        diameter_max=sieve_d, diameter_min=sieve_min,
        diameter_sd=phloem.get("sieve_diameter_sd", 0.001), gradient_function="normal",
    )
    comps = _place_companions(cells, rng, sieve_zone, centers, comp_d, comp_w, voronoi_grow, cx, cy)
    return unary_union(sieves + comps) if (sieves or comps) else None


def _phloem_ellipse(zone, cx0, cy0, axis, theta, bp, phloem, s_lo, s_hi):
    """Phloem sieve-cluster ellipse, placed along ``axis`` between radial offsets
    ``s_lo`` (relative distance 0) and ``s_hi`` (relative distance 1)."""
    w = float(bp.get("phloem_width", phloem.get("cluster_width", 0.05)))
    h = float(bp.get("phloem_height", phloem.get("cluster_height", 0.04)))
    rel = float(bp.get("phloem_relative_distance", 0.5))
    s = s_lo + rel * (s_hi - s_lo)
    ec = np.array([cx0, cy0]) + np.asarray(axis) * s
    ell = GeometryProcessor.oriented_ellipse(ec[0], ec[1], w, h, np.degrees(theta))
    return _largest(ell.intersection(zone))


def _fill_phloem(cells, rng, zone, cx, cy, theta, phloem, bp, result, cluster=True) -> None:
    """Phloem tissue: a phloem ellipse (sieve-element + companion-cell cluster)
    with ground parenchyma packed around it.

    Banded bundle (``cluster=True``): sieve elements + companion cells pack into an
    ellipse (``phloem_width`` x ``phloem_height``) centred along the bundle's radial
    axis at ``phloem_relative_distance`` within the phloem region; the rest of the
    region is parenchyma.  Concentric bundle (``cluster=False``): the whole zone
    (the phloem core or ring) is the sieve region.
    """
    p_diam = bp.get("parenchyma_diameter", 0.012)
    p_w = bp.get("parenchyma_width", 0.012)

    sieve_zone = zone
    if cluster:
        gx, gy = zone.centroid.x, zone.centroid.y
        outer = np.array([np.cos(theta), np.sin(theta)])
        h = float(bp.get("phloem_height", phloem.get("cluster_height", 0.04)))
        hh = _radial_half(zone, gx, gy, outer)
        lim = max(hh - h / 2.0, 0.0)
        sieve_zone = _phloem_ellipse(zone, gx, gy, outer, theta, bp, phloem, -lim, lim) or zone

    occupied = _place_phloem_cells(cells, rng, sieve_zone, cx, cy, phloem, bp)
    _fill_parenchyma(cells, zone, occupied, "parenchyma", cx, cy, p_diam, p_w)


def _fill_cambium(cells, rng, zone, cx, cy, cambium) -> None:
    """Thin band of small cambium cells (packed to fill the strip)."""
    d = cambium.get("cell_diameter", 0.01)
    fill_by_packing(cells, zone, "cambium", rng=rng, n_border=14, angle_center=(cx, cy),
                    proportion=1.0, direction=None, diameter_max=d, diameter_min=d,
                    diameter_sd=0.0, gradient_function="normal")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _sheath_zones(working, bp):
    """Peel the sheath off the working envelope (local frame).

    Every bundle gets a sheath: sclerenchyma fibres when requested, otherwise a
    thin parenchyma bundle-sheath ring.  Returns the (possibly shrunk) working
    envelope and a list of ``(tag, region, cell_diameter, cell_width)`` sheath
    zones.
    """
    t = bp.get("sheath_thickness", 0.012)
    sheath = bp.get("sheath", "none")
    p_diam, p_w = bp.get("parenchyma_diameter", 0.012), bp.get("parenchyma_width", 0.012)
    scl = bp.get("sclerenchyma_cell_diameter", 0.008)
    scl_w = bp.get("sclerenchyma_cell_width", scl)
    zones = []

    if sheath == "none":
        # No fibres -> a thin parenchyma sheath (one ring of ground cells).
        inner = working.buffer(-t)
        if not inner.is_empty:
            zones.append(("parenchyma", working.difference(inner), p_diam, p_w))
            working = inner
        return working, zones

    if sheath in ("ring", "both"):
        inner = working.buffer(-t)
        if not inner.is_empty:
            zones.append(("sclerenchyma", working.difference(inner), scl, scl_w))
            working = inner
    if sheath in ("caps", "both"):
        minx, miny, maxx, maxy = working.bounds
        cap_in = working.intersection(box(minx - 1, miny, maxx + 1, miny + t))
        cap_out = working.intersection(box(minx - 1, maxy - t, maxx + 1, maxy + 1))
        if not cap_in.is_empty:
            zones.append(("sclerenchyma", cap_in, scl, scl_w))
        if not cap_out.is_empty:
            zones.append(("sclerenchyma", cap_out, scl, scl_w))
        working = working.intersection(box(minx - 1, miny + t, maxx + 1, maxy - t))
    return working, zones


def _outward_caps(env_local: Polygon, bp: dict):
    """Asymmetric sclerenchyma fibre caps at the radial pole(s), outside the envelope.

    A cap is ``n_layers`` concentric fibre **files** hugging one pole.  File ``i``
    (``i = 0 .. n-1``) is the pole-side arc of the envelope buffered outward by
    ``i × sclerenchyma_cell_diameter``: file 0 sits *on* the envelope edge (so the
    fibres abut the bundle's own tissue — no gap), and each further file stacks one
    cell outward.  The outward pole is the ``+y`` (surface-facing) hemisphere, the
    inward pole the ``−y`` (centre-facing) hemisphere (``env_local`` is centred at
    the origin, radial axis +y).  Tracing files (rather than ring-filling a tapering
    crescent) keeps every fibre bounded by its neighbours, so cells stay round and no
    gaps open — and a single-layer cap still renders one clean file.

    ``n_caps_layers_outward`` / ``n_caps_layers_inward`` set the per-pole layer count;
    independent counts give the asymmetry.  Both default 0 → no caps.  Returns
    ``(files, region)``: ``files`` a list of ``(file_line, cell_diameter, cell_width)``
    (innermost first) that the fibres are seeded along, and ``region`` the clean
    contour-following cap polygon(s) — envelope buffered outward by ``n × scl`` at
    each pole — so the caller can unify it with the envelope for the bundle-sheath
    wrap / removal mask.  ``region`` is ``None`` when there are no caps.
    """
    scl = bp.get("sclerenchyma_cell_diameter", 0.008)
    scl_w = bp.get("sclerenchyma_cell_width", scl)
    minx, miny, maxx, maxy = env_local.bounds
    n_out = int(bp.get("n_caps_layers_outward", 0))
    n_in = int(bp.get("n_caps_layers_inward", 0))
    span = max(maxx - minx, maxy - miny) + (max(n_out, n_in) + 1) * scl + 1.0

    files, regions = [], []
    # (layer count, hemisphere the pole occupies) — +y outward, −y inward.
    poles = []
    if n_out > 0:
        poles.append((n_out, box(minx - span, 0.0, maxx + span, maxy + span)))
    if n_in > 0:
        poles.append((n_in, box(minx - span, miny - span, maxx + span, 0.0)))
    for n, hemisphere in poles:
        # clean cap region: the pole-side band of the envelope grown out by n cells.
        band = _largest(env_local.buffer(n * scl).difference(env_local).intersection(hemisphere))
        if band is not None and not band.is_empty:
            regions.append(band)
        # one fibre file per layer: concentric pole arcs, file 0 on the envelope edge.
        for i in range(n):
            arc = env_local.buffer(i * scl).exterior.intersection(hemisphere)
            if arc is not None and not arc.is_empty:
                files.append((arc, scl, scl_w))
    region = unary_union(regions) if regions else None
    return files, region


def bundle_cambium_anchor(bp: dict) -> float:
    """Local +y (radial) offset of the (outer) cambium band's centre.

    Placement curves (the eustele ring) put the *bundle centre* on the curve by
    default; anchoring on this value instead puts the bundle's **cambium** on the
    curve, so every fascicular cambium lines up into one continuous ring (the
    contour that becomes the secondary-growth vascular cambium).  Returns 0.0 when
    the bundle carries no cambium band — a concentric bundle, or ``has_cambium``
    False — so an un-anchored bundle stays centred on the curve as before.
    """
    env = _local_envelope(bp["width"], bp["height"], bp.get("shape", "ellipse"),
                          focus_exponent=bp.get("focus_exponent", 4.0),
                          egg_waist=bp.get("egg_waist", 0.6))
    working, _ = _sheath_zones(env, bp)
    mode, spec = bundle_layout(bp)
    if mode != "banded":
        return 0.0
    cambia = [g for role, g in partition_banded(working, spec)
              if role == "cambium" and g is not None and not g.is_empty]
    if not cambia:
        return 0.0
    return max(g.centroid.y for g in cambia)      # the outermost cambium band


def _anchor_shift(geoms, anchor: float):
    """Translate local-frame geometries by ``-anchor`` along +y (radial).

    Applied before :meth:`GeometryProcessor.place_local` so the local point
    ``(0, anchor)`` — the cambium band centre — lands on the placement point.
    """
    if not anchor:
        return geoms
    return [(_shapely_translate(g, 0.0, -anchor) if (g is not None and not g.is_empty) else g)
            for g in geoms]


def build_bundle(cells: CellManager, rng, cx: float, cy: float, theta: float,
                 bp: dict, xylem: dict, phloem: dict, cambium: dict,
                 ground_cell_size=None,
                 anchor: float = 0.0,
                 fill_cambium: bool = True,
                 sheath_outline=None,
                 ground_tissue_name=None) -> BundleResult:
    """Build one vascular bundle at ``(cx, cy)`` oriented radially at ``theta`` (rad).

    ``bp`` is the ``vascular_bundle`` param dict; ``xylem``/``phloem``/``cambium``
    are the reused cell-level param dicts.  Each tissue zone is filled with its
    proper component cells — vessels + xylem parenchyma; sieve elements +
    companion cells + phloem parenchyma; cambium; a sclerenchyma or parenchyma
    bundle sheath — so no cell is tagged with a bare tissue name.  Cells are
    appended to ``cells`` (disjoint Voronoi-group ids).  Returns a
    :class:`BundleResult` (envelope for the mask, vessel + cavity + zone polygons).

    ``ground_cell_size`` (the surrounding ground-tissue cell diameter, which only
    the organ knows) turns on an *outer bundle sheath*: one extra file of
    ``bundle sheath`` cells wrapping the whole bundle, sized at the mean of the
    inner sheath cell and that ground cell (see :func:`outer_sheath_ring_diameter`).
    It sits just outside the envelope, so the returned envelope — hence the removal
    mask — grows to cover it.  Left ``None`` (the organ-agnostic default), no outer
    sheath is added and the bundle is unchanged.

    ``ground_tissue_name`` (a callable ``f(x, y) -> tissue name``, again organ-only
    knowledge) names each outer-sheath cell after the tissue it is fitted into — the
    majority tissue over the cell's footprint — instead of the generic
    ``bundle sheath`` tag; left ``None`` the cells stay ``bundle sheath``.

    ``anchor`` (local +y, radial) shifts the whole bundle so that the local point
    ``(0, anchor)`` lands on ``(cx, cy)`` instead of the envelope centre.  Passing
    :func:`bundle_cambium_anchor` puts the bundle's cambium — not its centre — on
    the placement point, so a ring of bundles shares one cambium contour (default
    0.0 keeps the envelope centred, as before).

    ``fill_cambium`` False leaves the cambium band *unfilled* (the zone is still
    partitioned and registered, so xylem and phloem stay separated by the gap): the
    caller then lays the cambium down itself — e.g. the dicot stem draws the
    fascicular and interfascicular cambium in one pass along a shared contour so
    they share the same number of cell files.
    """
    result = BundleResult()
    theta_deg = np.degrees(theta)
    env_local = _local_envelope(bp["width"], bp["height"], bp.get("shape", "ellipse"),
                                focus_exponent=bp.get("focus_exponent", 4.0),
                                egg_waist=bp.get("egg_waist", 0.6))

    working, sheath_local = _sheath_zones(env_local, bp)

    # Asymmetric fibre caps extend the bundle *outside* the envelope as concentric
    # files (seeded further below).  They grow the footprint that drives the outer
    # bundle-sheath wrap and the ground-removal mask so the surrounding ground is
    # cleared for them and the wrap bounds their outermost file.  No caps ->
    # footprint is the bare envelope, so a bundle without caps is byte-identical.
    cap_files_local, cap_region_local = _outward_caps(env_local, bp)
    footprint_local = (_largest(unary_union([env_local, cap_region_local])) or env_local
                       if cap_region_local is not None else env_local)

    mode, spec = bundle_layout(bp)
    is_face = mode == "banded" and bp.get("xylem_layout", "packed") == "face"
    if is_face:
        # Monocot face bundle: the whole envelope is one region — metaxylem,
        # protoxylem+lacunae and the phloem ellipse are placed explicitly inside it
        # (so the phloem's relative distance spans the bundle, not a thin band).
        zones_local = [("xylem", working)]
    elif mode == "banded":
        zones_local = partition_banded(working, spec)
    else:
        core_role, ring_role = spec
        zones_local = partition_concentric(working, core_role, ring_role,
                                           bp.get("core_width", 0.05), bp.get("core_height", 0.05))

    # Footprint (envelope + fibre caps) and the bare envelope in the world frame —
    # the outer bundle sheath is grown against the local tissue from the footprint at
    # the end of the build; the envelope tells it which boundary points sit on caps.
    foot_world, env_world = GeometryProcessor.place_local(
        _anchor_shift([footprint_local, env_local], anchor), cx, cy, theta_deg)

    sheath_geoms = GeometryProcessor.place_local(
        _anchor_shift([r for _, r, _, _ in sheath_local], anchor), cx, cy, theta_deg)
    zone_geoms = GeometryProcessor.place_local(
        _anchor_shift([g for _, g in zones_local], anchor), cx, cy, theta_deg)
    result.envelope = foot_world           # provisional; grown into the mask below

    # Sheath first (fibres or a parenchyma bundle sheath).
    for (tag, _r, cd, cw), geom in zip(sheath_local, sheath_geoms):
        if geom is None or geom.is_empty:
            continue
        result.zone_polygons.append((tag, geom))
        if tag == "sclerenchyma":
            # Small fibres pack into 1-2 layers by ring-fill.
            _fill_parenchyma(cells, geom, None, tag, cx, cy, cd, cw)
        else:
            # A parenchyma bundle sheath is a single file hugging the envelope
            # (the ring is too thin for ring-fill's erosion to seed).
            fill_along(cells, geom, tag, cd, cw, cx, cy)

    # Then the tissue zones, each filled with its component cells.
    for (role, _g), geom in zip(zones_local, zone_geoms):
        if geom is None or geom.is_empty:
            continue
        result.zone_polygons.append((role, geom))
        if role == "xylem":
            layout = bp.get("xylem_layout", "packed")
            if layout == "face":
                _fill_xylem_face(cells, rng, geom, cx, cy, theta, bp, phloem, result)
            elif layout == "files":
                _fill_xylem_files(cells, rng, geom, cx, cy, theta, xylem, bp, result)
            else:
                _fill_xylem_packed(cells, rng, geom, cx, cy, xylem, bp, result)
        elif role == "phloem":
            _fill_phloem(cells, rng, geom, cx, cy, theta, phloem, bp, result,
                         cluster=(mode == "banded"))
        elif role == "cambium":
            if fill_cambium:
                _fill_cambium(cells, rng, geom, cx, cy, cambium)

    # Asymmetric fibre caps: one traced file of sclerenchyma per layer, hugging the
    # pole contour from the envelope edge outward.  Seeded before the outer sheath so
    # the wrap (highest id_group) bounds the outermost file.  The clean cap region
    # (already folded into footprint_local) is registered once for the tissue view.
    if cap_files_local:
        cap_geoms = GeometryProcessor.place_local(
            _anchor_shift([ln for ln, _, _ in cap_files_local], anchor), cx, cy, theta_deg)
        for (_ln, cd, cw), gline in zip(cap_files_local, cap_geoms):
            if gline is None or gline.is_empty:
                continue
            fill_along(cells, gline, "sclerenchyma", cd, cw, cx, cy)
    if cap_region_local is not None and not cap_region_local.is_empty:
        placed = GeometryProcessor.place_local(
            _anchor_shift([cap_region_local], anchor), cx, cy, theta_deg)[0]
        if placed is not None and not placed.is_empty:
            result.zone_polygons.append(("sclerenchyma", placed))

    # Finally, grow the outer bundle sheath cell by cell against the local tissue
    # (each cell sized from its own neighbours), clipped to the organ outline so it
    # never leaves the stem.  Sets the removal mask (result.envelope).
    result.envelope = _grow_bundle_sheath(
        cells, foot_world, bp, ground_cell_size, sheath_outline, cx, cy, env=env_world,
        ground_tissue_name=ground_tissue_name)

    return result


# ---------------------------------------------------------------------------
# Arc bundle — a pie slice of a continuous vascular cylinder
# ---------------------------------------------------------------------------

def _radius_range(zone: Polygon, ox: float, oy: float) -> Tuple[float, float]:
    """Min / max distance of ``zone``'s vertices from the curvature centre ``(ox, oy)``."""
    ext = np.asarray(zone.exterior.coords)
    d = np.hypot(ext[:, 0] - ox, ext[:, 1] - oy)
    r_in, r_out = float(d.min()), float(d.max())
    for ring in zone.interiors:
        ic = np.asarray(ring.coords)
        r_in = min(r_in, float(np.hypot(ic[:, 0] - ox, ic[:, 1] - oy).min()))
    return r_in, r_out


def build_arc_bundle(cells: CellManager, rng, cx: float, cy: float, theta: float,
                     bp: dict, xylem: dict, phloem: dict, cambium: dict,
                     ground_cell_size=None, sheath_outline=None,
                     ground_tissue_name=None) -> BundleResult:
    """Build one vein as a **slice of a vascular cylinder** — concentric xylem /
    cambium / phloem *arcs* spanning ``arc_degrees`` of a circle of radius
    ``arc_radius`` — the continuous dicot-stem cylinder kept to one pie slice.

    The slice is oriented like every other leaf vein: ``theta`` points to the abaxial
    (lower) surface, so the curvature centre sits on the adaxial side and the arcs run,
    inner -> outer, **xylem (adaxial) -> cambium -> phloem (abaxial)**.  The xylem is
    endarch-graded (small protoxylem on the inner/adaxial face -> large metaxylem
    toward the cambium) and, with ``xylem_layout == "files"``, cut into
    ``n_xylem_files`` radial files by thin parenchyma strips (the cambium and phloem
    stay continuous), exactly like :class:`ContinuousDicotStemAnatomy`.

    Returns a :class:`BundleResult` (envelope for the removal mask, vessel + zone
    polygons) so the caller registers it like any other bundle.
    """
    result = BundleResult()
    r0 = float(bp.get("arc_radius", 0.25))
    span = np.radians(float(bp.get("arc_degrees", 70.0)))
    if r0 <= 0.0 or span <= 0.0:
        return result
    half_cam = float(bp.get("arc_cambium_thickness", 0.015)) / 2.0
    xt = float(bp.get("arc_xylem_thickness", 0.05))
    pt = float(bp.get("arc_phloem_thickness", 0.035))
    if not bp.get("phloem_outward", True):
        theta = theta + np.pi        # phloem faces the adaxial side instead

    # Curvature centre on the adaxial side; the outward (+radial) direction is abaxial.
    u = np.array([np.cos(theta), np.sin(theta)])         # outward = abaxial
    r_far = (r0 + half_cam + max(xt, pt)) * 2.0

    def _build(anchor_x, anchor_y):
        """The three concentric arcs + wedge for an anchor point on the cambium
        contour.  Returns (ox, oy, a0, a1, xylem, phloem, cambium)."""
        ox = float(anchor_x - r0 * u[0])
        oy = float(anchor_y - r0 * u[1])
        cang = float(np.arctan2(anchor_y - oy, anchor_x - ox))   # O -> bundle direction
        cont = Point(ox, oy).buffer(r0, resolution=128)
        xa = cont.buffer(-half_cam).difference(cont.buffer(-half_cam - xt))
        pa = cont.buffer(half_cam + pt).difference(cont.buffer(half_cam))
        ca = cont.buffer(half_cam).difference(cont.buffer(-half_cam))
        b0, b1 = cang - span / 2.0, cang + span / 2.0
        wdg = Polygon([(ox, oy)] + [(ox + r_far * np.cos(a), oy + r_far * np.sin(a))
                                    for a in np.linspace(b0, b1, 32)])
        return ox, oy, b0, b1, xa.intersection(wdg), pa.intersection(wdg), ca.intersection(wdg)

    # Anchoring the arc by its abaxial (cambium-bottom) edge puts the whole bundle —
    # its thick xylem especially — above the placement point, pushing the vein into the
    # top (adaxial) side of the leaf.  Instead, build the (unclipped) bundle once, then
    # slide it along the radial axis so its centre of mass lands on (cx, cy): the vein
    # is then vertically centred on its placement point regardless of the arc span or
    # the xylem/phloem thickness ratio.
    _, _, _, _, xa0, pa0, ca0 = _build(cx, cy)
    env0 = unary_union([g for g in (xa0, pa0, ca0) if not g.is_empty])
    if not env0.is_empty:
        d = np.array([cx - env0.centroid.x, cy - env0.centroid.y])
        s = float(np.dot(d, u))                              # component along the radial axis
        anchor_x, anchor_y = cx + s * u[0], cy + s * u[1]
    else:
        anchor_x, anchor_y = cx, cy
    ox, oy, a0, a1, xylem_annulus, phloem_annulus, cambium_band = _build(anchor_x, anchor_y)

    # The zones are already the pie slice (wedge-clipped in _build); clip to the outline.
    def clip(zone):
        z = zone.intersection(sheath_outline) if sheath_outline is not None else zone
        return _largest(z)

    xylem_zone = clip(xylem_annulus)
    phloem_zone = clip(phloem_annulus)
    cambium_zone = clip(cambium_band)

    p_diam = float(bp.get("parenchyma_diameter", 0.012))
    p_w = float(bp.get("parenchyma_width", 0.012))
    vgrow = 0.25 * (p_diam + p_w)

    # --- xylem: endarch-graded vessels, optionally cut into radial files ------
    if xylem_zone is not None and not xylem_zone.is_empty:
        grr = _radius_range(xylem_zone, ox, oy)
        pieces = [xylem_zone]
        n_files = int(bp.get("n_xylem_files", 0))
        if bp.get("xylem_layout", "packed") == "files" and n_files >= 2:
            # Split the arc into n_files angular sub-wedges (radial files); the
            # parenchyma pass over the whole annulus fills the seams between them.
            pieces = []
            for k in range(n_files):
                b0 = a0 + (a1 - a0) * k / n_files
                b1 = a0 + (a1 - a0) * (k + 1) / n_files
                sub = Polygon([(ox, oy)] + [(ox + r_far * np.cos(a), oy + r_far * np.sin(a))
                                            for a in np.linspace(b0, b1, 6)])
                pieces.extend(p for p in [_largest(xylem_zone.intersection(sub))]
                              if p is not None and not p.is_empty)
        vessels = []
        for piece in pieces:
            vs, _ = _pack_place(
                cells, rng, piece, "xylem", cx, cy,
                voronoi_grow=vgrow, r_floor=p_diam * 0.4, n_border=25,
                proportion=float(bp.get("prop_vessel", 0.55)),
                direction="edge",                    # endarch: large toward the cambium
                gradient_center=(ox, oy), gradient_radial_range=grr,
                diameter_max=xylem.get("vessel_diameter", 0.045),
                diameter_min=xylem.get("vessel_diameter_min", 0.012),
                diameter_sd=xylem.get("vessel_diameter_sd", 0.003),
                gradient_function=xylem.get("gradient_function", "five_pl"),
                gradient_inflection=xylem.get("gradient_inflection", 0.5),
                gradient_steepness=xylem.get("gradient_steepness", 3.0),
                gradient_asymmetry=xylem.get("gradient_asymmetry", 1.0),
            )
            vessels.extend(vs)
        result.vessel_polygons.extend(vessels)
        _fill_parenchyma(cells, xylem_zone, unary_union(vessels) if vessels else None,
                         "parenchyma", cx, cy, p_diam, p_w)
        result.zone_polygons.append(("xylem", xylem_zone))

    # --- phloem: sieve + companion cells, parenchyma around them --------------
    # Packed in angular sectors: pack_circles is superlinear in the cell count, so a
    # wide arc packed as one region is far slower than the same cells packed
    # sector-by-sector.  The seams are invisible — filled by the parenchyma pass.
    if phloem_zone is not None and not phloem_zone.is_empty:
        ph_arc = (r0 + half_cam + pt) * span
        sieve_d = float(phloem.get("sieve_diameter", 0.012))
        n_sec = max(1, int(round(ph_arc / (10.0 * sieve_d))))
        occupied = []
        for k in range(n_sec):
            b0 = a0 + (a1 - a0) * k / n_sec
            b1 = a0 + (a1 - a0) * (k + 1) / n_sec
            sub = Polygon([(ox, oy)] + [(ox + r_far * np.cos(a), oy + r_far * np.sin(a))
                                        for a in np.linspace(b0, b1, 6)])
            sec = _largest(phloem_zone.intersection(sub))
            if sec is None or sec.is_empty:
                continue
            occ = _place_phloem_cells(cells, rng, sec, cx, cy, phloem, bp)
            if occ is not None and not occ.is_empty:
                occupied.append(occ)
        _fill_parenchyma(cells, phloem_zone,
                         unary_union(occupied) if occupied else None,
                         "parenchyma", cx, cy, p_diam, p_w)
        result.zone_polygons.append(("phloem", phloem_zone))

    # --- cambium arc ---------------------------------------------------------
    if cambium_zone is not None and not cambium_zone.is_empty:
        _fill_cambium(cells, rng, cambium_zone, cx, cy, cambium)
        result.zone_polygons.append(("cambium", cambium_zone))

    envelope = unary_union([z for z in (xylem_zone, cambium_zone, phloem_zone)
                            if z is not None and not z.is_empty])
    result.envelope = _largest(envelope) or envelope

    # Optional outer bundle sheath, grown against the surrounding mesophyll.
    if ground_cell_size is not None and result.envelope is not None \
            and not result.envelope.is_empty:
        result.envelope = _grow_bundle_sheath(
            cells, result.envelope, bp, ground_cell_size, sheath_outline, cx, cy,
            env=result.envelope, ground_tissue_name=ground_tissue_name)
    return result
