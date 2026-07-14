"""Vascular bundle construction (shape-first, organ-agnostic).

A vascular bundle is built as three things:

1. an **envelope** — an oriented footprint at ``(cx, cy)`` with radial orientation
   ``theta`` (local +y = radial, pointing toward the organ surface);
2. an **internal partition** into tissue sub-zones — *the bundle type is the
   topology of this partition*;
3. a **per-zone fill** with the right cell type via the existing ``fill_*``
   primitives.

Two partition families cover the four botanical types:

* **banded** (``collateral`` / ``bicollateral``): parallel cuts perpendicular to
  the radial axis → stacked bands (xylem / cambium / phloem, inner→outer).
* **concentric** (``concentric`` + ``amphivasal`` / ``amphicribral``): a core +
  a ring (one tissue surrounding the other).

The monocot xylem "face" (``xylem_layout="face"``) is a bespoke fill *within* the
xylem zone: a few discrete metaxylem "eyes" + protoxylem, with an optional
protoxylem lacuna (a void).

Placement (how many bundles and where — a eustele ring vs a scattered
atactostele) is the caller's job; this module builds one bundle at a time.

Convention for cavities (protoxylem lacuna, and the pith cavity elsewhere): a
cavity is a polygon with **no cells**; the whole bundle **envelope** is returned
for the removal mask, so a lacuna inside it is already cleared of ground seeds.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from shapely.geometry import Point, Polygon, box

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


def _local_envelope(width: float, height: float, shape: str) -> Polygon:
    """Envelope in the local frame: centred at the origin, radial axis = +y."""
    if shape == "circle":
        d = min(width, height)
        return Point(0, 0).buffer(d / 2, resolution=64)
    return GeometryProcessor.oriented_ellipse(0.0, 0.0, width, height, 90.0)


# ---------------------------------------------------------------------------
# Layout — the topology per bundle type
# ---------------------------------------------------------------------------

def bundle_layout(bp: dict):
    """Resolve bundle params into a partition spec.

    Returns ``("banded", [(role, fraction), ...])`` inner→outer, or
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

    bands: List[Tuple[str, float]] = []
    if bt == "bicollateral" and ipf > 0.0:
        bands.append(("phloem", ipf))                     # inner phloem
        if bp.get("inner_cambium", False):
            bands.append(("cambium", cf))
    bands.append(("xylem", xf))                           # xylem (inner of outer phloem)
    if bp.get("has_cambium", True):
        bands.append(("cambium", cf))
    bands.append(("phloem", pf))                          # outer phloem

    # Local +y is radial-outward; inner→outer = bottom→top. Flip for phloem_inward.
    if not bp.get("phloem_outward", True):
        bands = list(reversed(bands))
    return "banded", bands


def partition_banded(env: Polygon, bands) -> List[Tuple[str, Polygon]]:
    """Cut ``env`` into horizontal strips (inner→outer along local +y)."""
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


def _place_companions(cells, rng, zone, sieve_centers, comp_d, voronoi_grow, cx, cy):
    """One companion cell beside each sieve element (rejecting sieve overlaps).

    Mirrors the dicot-root secondary phloem: a companion is tried on random sides
    of its sieve and kept on the first free one, so every sieve element is paired.
    """
    zone = _largest(zone)
    if zone is None or not sieve_centers:
        return []
    sieve_union = unary_union([Point(c[0], c[1]).buffer(c[2]) for c in sieve_centers])
    comp_d = max(comp_d - voronoi_grow, comp_d * 0.5)
    comp_r = comp_d / 2
    comps = []
    for pcx, pcy, r in sieve_centers:
        theta0 = rng.uniform(0.0, 2 * np.pi)
        for k in range(8):
            a = theta0 + 2 * np.pi * k / 8
            ccx, ccy = pcx + (r + comp_r * 1.05) * np.cos(a), pcy + (r + comp_r * 1.05) * np.sin(a)
            if not zone.contains(Point(ccx, ccy)):
                continue
            circle = Point(ccx, ccy).buffer(comp_r)
            if circle.intersects(sieve_union) or any(circle.intersects(c) for c in comps):
                continue
            placed = _place_circle(cells, ccx, ccy, comp_r, "companion cell", (cx, cy), n_border=14)
            if placed is not None:
                comps.append(circle)
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
        cells, rng, zone, "vessel", cx, cy,
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
                     "xylem parenchyma", cx, cy, p_diam, p_w)


def _fill_xylem_face(cells, rng, zone, cx, cy, theta, bp, result) -> None:
    """The monocot mask: discrete metaxylem + protoxylem + optional lacuna, then
    xylem parenchyma packed around them.

    Endarch (stem): protoxylem toward the centre, metaxylem toward the phloem
    pole; exarch flips the two poles.
    """
    zone = _largest(zone)
    if zone is None:
        return
    gx, gy = zone.centroid.x, zone.centroid.y
    R = np.sqrt(zone.area / np.pi)
    outer = np.array([np.cos(theta), np.sin(theta)])       # radial, toward surface
    tang = np.array([-np.sin(theta), np.cos(theta)])

    meta_dir = outer if bp.get("xylem_maturation", "endarch") == "endarch" else -outer
    proto_dir = -meta_dir
    placed_polys = []

    # Metaxylem "eyes": n vessels spread tangentially, offset toward the meta pole.
    n_meta = int(bp.get("n_metaxylem", 2))
    dm = float(bp.get("metaxylem_diameter", 0.045))
    gap = float(bp.get("metaxylem_gap", 0.02))
    meta_base = np.array([gx, gy]) + meta_dir * (0.25 * R)
    for k in range(n_meta):
        d = float(np.clip(rng.normal(dm, bp.get("metaxylem_diameter_sd", 0.004)), dm * 0.3, np.inf))
        tk = (k - (n_meta - 1) / 2.0) * (dm + gap)
        c = meta_base + tang * tk
        poly = _largest(Point(c[0], c[1]).buffer(d / 2, resolution=32).intersection(zone))
        if poly is None:
            continue
        r_eff = np.sqrt(poly.area / np.pi)
        placed = _place_circle(cells, poly.centroid.x, poly.centroid.y, r_eff, "metaxylem", (cx, cy))
        if placed is not None:
            placed_polys.append(placed)

    # Protoxylem: small vessels toward the proto pole.
    n_proto = int(bp.get("n_protoxylem", 3))
    dp = float(bp.get("protoxylem_diameter", 0.012))
    proto_base = np.array([gx, gy]) + proto_dir * (0.35 * R)
    for k in range(n_proto):
        d = float(np.clip(rng.normal(dp, bp.get("protoxylem_diameter_sd", 0.002)), dp * 0.3, np.inf))
        tk = (k - (n_proto - 1) / 2.0) * (dp * 1.4)
        c = proto_base + tang * tk
        poly = _largest(Point(c[0], c[1]).buffer(d / 2, resolution=24).intersection(zone))
        if poly is None:
            continue
        r_eff = np.sqrt(poly.area / np.pi)
        placed = _place_circle(cells, poly.centroid.x, poly.centroid.y, r_eff, "protoxylem", (cx, cy), n_border=16)
        if placed is not None:
            placed_polys.append(placed)

    # Protoxylem lacuna: an air cavity at the very inner pole (no cells).
    lacuna = None
    if bp.get("lacuna", False):
        lc = np.array([gx, gy]) + proto_dir * (0.6 * R)
        lacuna = _largest(GeometryProcessor.oriented_ellipse(
            lc[0], lc[1], bp.get("lacuna_width", 0.03), bp.get("lacuna_height", 0.025),
            np.degrees(theta)).intersection(zone))
        if lacuna is not None and not lacuna.is_empty:
            result.cavity_polygons.append(lacuna)

    result.vessel_polygons.extend(placed_polys)
    occupied = placed_polys + ([lacuna] if lacuna is not None else [])
    p_diam = bp.get("parenchyma_diameter", 0.012)
    _fill_parenchyma(cells, zone, unary_union(occupied) if occupied else None,
                     "xylem parenchyma", cx, cy, p_diam, bp.get("parenchyma_width", 0.012))


def _fill_phloem(cells, rng, zone, cx, cy, phloem, bp, result) -> None:
    """Phloem tissue: small sieve elements, a companion cell beside each, and
    phloem parenchyma packed around them (no cell is tagged 'phloem')."""
    p_diam = bp.get("parenchyma_diameter", 0.012)
    p_w = bp.get("parenchyma_width", 0.012)
    voronoi_grow = 0.25 * (p_diam + p_w)
    sieve_d = phloem.get("sieve_diameter", 0.012)
    sieve_min = bp.get("sieve_diameter_min", 0.006)
    comp_d = bp.get("companion_diameter", 0.007)
    prop_sieve = bp.get("prop_sieve", 0.45)
    # Fraction of the zone to pack with sieves so sieve+companion ~= prop_sieve.
    proportion = prop_sieve * sieve_d ** 2 / (sieve_d ** 2 + comp_d * comp_d)
    sieves, centers = _pack_place(
        cells, rng, zone, "sieve element", cx, cy,
        voronoi_grow=voronoi_grow, r_floor=min(sieve_d / 2, p_diam * 0.4), n_border=16,
        proportion=proportion, direction=None,
        diameter_max=sieve_d, diameter_min=sieve_min,
        diameter_sd=phloem.get("sieve_diameter_sd", 0.001), gradient_function="normal",
    )
    comps = _place_companions(cells, rng, zone, centers, comp_d, voronoi_grow, cx, cy)
    occupied = unary_union(sieves + comps) if (sieves or comps) else None
    _fill_parenchyma(cells, zone, occupied, "phloem parenchyma", cx, cy, p_diam, p_w)


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
    zones = []

    if sheath == "none":
        # No fibres -> a thin parenchyma bundle sheath (one ring of cells).
        inner = working.buffer(-t)
        if not inner.is_empty:
            zones.append(("bundle sheath", working.difference(inner), p_diam, p_w))
            working = inner
        return working, zones

    if sheath in ("ring", "both"):
        inner = working.buffer(-t)
        if not inner.is_empty:
            zones.append(("sclerenchyma", working.difference(inner), scl, scl))
            working = inner
    if sheath in ("caps", "both"):
        minx, miny, maxx, maxy = working.bounds
        cap_in = working.intersection(box(minx - 1, miny, maxx + 1, miny + t))
        cap_out = working.intersection(box(minx - 1, maxy - t, maxx + 1, maxy + 1))
        if not cap_in.is_empty:
            zones.append(("sclerenchyma", cap_in, scl, scl))
        if not cap_out.is_empty:
            zones.append(("sclerenchyma", cap_out, scl, scl))
        working = working.intersection(box(minx - 1, miny + t, maxx + 1, maxy - t))
    return working, zones


def build_bundle(cells: CellManager, rng, cx: float, cy: float, theta: float,
                 bp: dict, xylem: dict, phloem: dict, cambium: dict) -> BundleResult:
    """Build one vascular bundle at ``(cx, cy)`` oriented radially at ``theta`` (rad).

    ``bp`` is the ``vascular_bundle`` param dict; ``xylem``/``phloem``/``cambium``
    are the reused cell-level param dicts.  Each tissue zone is filled with its
    proper component cells — vessels + xylem parenchyma; sieve elements +
    companion cells + phloem parenchyma; cambium; a sclerenchyma or parenchyma
    bundle sheath — so no cell is tagged with a bare tissue name.  Cells are
    appended to ``cells`` (disjoint Voronoi-group ids).  Returns a
    :class:`BundleResult` (envelope for the mask, vessel + cavity + zone polygons).
    """
    result = BundleResult()
    theta_deg = np.degrees(theta)
    env_local = _local_envelope(bp["width"], bp["height"], bp.get("shape", "ellipse"))

    working, sheath_local = _sheath_zones(env_local, bp)

    mode, spec = bundle_layout(bp)
    if mode == "banded":
        zones_local = partition_banded(working, spec)
    else:
        core_role, ring_role = spec
        zones_local = partition_concentric(working, core_role, ring_role,
                                           bp.get("core_width", 0.05), bp.get("core_height", 0.05))

    sheath_geoms = GeometryProcessor.place_local([r for _, r, _, _ in sheath_local], cx, cy, theta_deg)
    zone_geoms = GeometryProcessor.place_local([g for _, g in zones_local], cx, cy, theta_deg)
    result.envelope = GeometryProcessor.place_local([env_local], cx, cy, theta_deg)[0]

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
            if bp.get("xylem_layout", "packed") == "face":
                _fill_xylem_face(cells, rng, geom, cx, cy, theta, bp, result)
            else:
                _fill_xylem_packed(cells, rng, geom, cx, cy, xylem, bp, result)
        elif role == "phloem":
            _fill_phloem(cells, rng, geom, cx, cy, phloem, bp, result)
        elif role == "cambium":
            _fill_cambium(cells, rng, geom, cx, cy, cambium)

    # Lacuna cavities show as voids in the tissue view too.
    for cav in result.cavity_polygons:
        result.zone_polygons.append(("medullary cavity", cav))

    return result
