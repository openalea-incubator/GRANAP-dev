"""Shared secondary-growth fills (organ-agnostic).

Free-function ports of the dicot-root secondary-growth algorithms so the dicot
*stem* can reuse them without forking the root: rendering the cambium as
concentric cell files, packing secondary-xylem vessels + axial parenchyma into a
zone, and the secondary phloem (a band standing on the cambium, split radially
into an alive sub-zone — sieve tubes + companion cells + parenchyma — and a dead
sub-zone — sieve tubes + parenchyma).

Every function takes an explicit :class:`CellManager` (+ an rng where packing is
involved), so they are decoupled from the organ classes.  Geometry is in absolute
mm about the organ centre ``(cx, cy)``.  The root keeps its own copies for now;
this module is the reusable surface the stem builds on (a later pass can migrate
the root onto it).
"""

import numpy as np
import shapely as sp
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union
from shapely.affinity import rotate, translate

from openalea.granap.cell_class import Cell
from openalea.granap.geometry_collection import GeometryProcessor
from openalea.granap.tissue_class import place_packed_group, fill_by_rings, fill_along


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def angular_wedge(cx: float, cy: float, theta_c: float, half_angle: float,
                  r_outer: float, n_arc: int = 50) -> Polygon:
    """Pie-wedge polygon: apex at ``(cx, cy)`` spanning ``theta_c ± half_angle``
    out to ``r_outer``.  Intersect with an annulus to get a xylem sector."""
    arc = np.linspace(theta_c - half_angle, theta_c + half_angle, n_arc)
    return Polygon([(cx, cy)] + [
        (cx + r_outer * np.cos(a), cy + r_outer * np.sin(a)) for a in arc
    ])


def flared_wedge(cx: float, cy: float, theta_c: float, r_inner: float,
                 r_outer: float, base_half_width: float, flare_angle: float,
                 cap_half_angle: float, n_arc: int = 48) -> Polygon:
    """Radially-flaring sector centred on ``theta_c`` between ``r_inner`` and
    ``r_outer``.

    Unlike :func:`angular_wedge` (a constant angular half-width), this starts at a
    tangential half-width ``base_half_width`` at ``r_inner`` and widens outward with
    its side edges tilted ``flare_angle`` (radians) from the radial direction, the
    angular half-width capped at ``cap_half_angle`` (radians).  Secondary-xylem
    sectors therefore start at the vascular-bundle width against the primary xylem
    and only merge into a continuous ring once their capped edges meet further out
    (a full ring at ``cap_half_angle`` = π / n_bundles).  ``flare_angle`` = 0 gives
    straight radial sides that never widen.
    """
    rs = np.linspace(r_inner, r_outer, max(int(n_arc), 2))
    tan_f = np.tan(flare_angle)
    plus, minus = [], []
    for r in rs:
        t = base_half_width + tan_f * (r - r_inner)      # tangential half-width
        ha = min(t / r, cap_half_angle) if r > 0 else cap_half_angle
        plus.append((cx + r * np.cos(theta_c + ha), cy + r * np.sin(theta_c + ha)))
        minus.append((cx + r * np.cos(theta_c - ha), cy + r * np.sin(theta_c - ha)))
    return Polygon(minus + plus[::-1]).buffer(0)


def _polys(geom):
    """List of non-empty Polygon pieces of a (possibly Multi)geometry."""
    if geom is None or geom.is_empty:
        return []
    parts = geom.geoms if hasattr(geom, "geoms") else [geom]
    return [g for g in parts if g.geom_type == "Polygon" and not g.is_empty]


# ---------------------------------------------------------------------------
# Cambium
# ---------------------------------------------------------------------------

def render_cambium_files(cells, contour: Polygon, n_layers: int,
                         cell_d: float, cell_w: float, cx: float, cy: float) -> None:
    """Render the cambium as ``n_layers`` concentric files along ``contour``,
    each buffered inward by one cell diameter (mirrors the root)."""
    for k in range(max(int(n_layers), 1)):
        ring = contour if k == 0 else contour.buffer(-k * cell_d)
        if ring.is_empty:
            break
        for g in _polys(ring):
            fill_along(cells, g.exterior, "cambium", cell_d, cell_w, cx, cy)


# ---------------------------------------------------------------------------
# Secondary xylem
# ---------------------------------------------------------------------------

def pack_xylem_vessels(cells, rng, zone: Polygon, sx: dict, cx: float, cy: float,
                       id_base: int, grr=None):
    """Pack graded secondary-xylem vessels into ``zone``; seed one per circle.

    ``grr`` is an optional ``(r_inner, r_outer)`` gradient radial range (annual
    rings).  Returns ``(vessel_polygons, next_id)``.
    """
    packed = GeometryProcessor.pack_circles(
        zone,
        proportion=sx["prop_vessel_ring"], direction="center",
        diameter_max=sx["vessel_diameter"], diameter_min=sx["vessel_diameter_min"],
        diameter_sd=sx["vessel_diameter_sd"], gradient_function=sx["gradient_function"],
        gradient_inflection=sx["gradient_inflection"], gradient_steepness=sx["gradient_steepness"],
        gradient_asymmetry=sx["gradient_asymmetry"], enforce_gradient_min=sx["enforce_gradient_min"],
        allow_ellipse=sx["allow_ellipse"], ellipse_max_aspect=sx["ellipse_max_aspect"],
        pack_strategy=sx["packing_strategy"], adjacent=sx["must_be_adjacent"],
        gradient_center=(cx, cy), gradient_radial_range=grr, rng=rng,
    )
    placed = place_packed_group(cells, packed, "xylem", n_border=25,
                                id_base=id_base, angle_center=None)
    return [p for p, _t, _g in placed], id_base + len(packed)


def fill_secondary_xylem_sector(cells, rng, sector: Polygon, sx: dict,
                                cx: float, cy: float, start_id: int,
                                annual_bands=None, medullar_union=None) -> tuple:
    """Fill one secondary-xylem sector: vessels + axial parenchyma around them.

    Returns ``(vessel_polygons, next_id)``.  The axial parenchyma is tagged
    ``parenchyma`` (the stem ground-tissue tag) and hugs the vessels.

    Mirrors the dicot root's ``fit_secondary_xylem`` inner loop: the sector is first
    cut by the medullar-ray corridors (``medullar_union``) into fragments, then each
    fragment is split into the ``annual_bands`` (growth rings) — a graded vessel pack
    per band, referenced to the band's ``(r_inner, r_outer)`` so the size gradient
    restarts each ring (small at the ring start, large toward its outer edge).  With
    ``annual_bands=None`` (``n_ring`` = 1) the fragment is packed whole; with
    ``medullar_union=None`` no ray cut is made.
    """
    next_id = start_id
    vessels = []

    for base in _polys(sector):
        if medullar_union is not None and not medullar_union.is_empty:
            remaining = base.difference(medullar_union)
        else:
            remaining = base
        for zone in _polys(remaining):
            if annual_bands is None:
                band_pieces = [(zone, None)]
            else:
                band_pieces = []
                for band_poly, r_in, r_out in annual_bands:
                    for g in _polys(zone.intersection(band_poly)):
                        band_pieces.append((g, (r_in, r_out)))

            for piece, grr in band_pieces:
                vs, next_id = pack_xylem_vessels(cells, rng, piece, sx, cx, cy,
                                                 next_id, grr=grr)
                vessels.extend(vs)
                if vs:
                    # Keep the axial fill clear of the vessels' Voronoi over-grow
                    # (they seed as an inset ring), so no parenchyma seed lands where
                    # a vessel will expand.  Mirrors the dicot root exactly.
                    clearance = sx["cell_diameter"] * 0.5
                    axial = piece.difference(unary_union(vs).buffer(clearance))
                else:
                    axial = piece
                if not axial.is_empty:
                    # Fill the whole axial zone at once, eroding against the *full*
                    # piece (not each fragment) — the root's erosion_polygon — so the
                    # concentric rings seat correctly and the parenchyma actually fills
                    # the non-vessel area instead of leaving gaps the Voronoi inflates.
                    next_id = fill_by_rings(cells, axial, sx["cell_diameter"],
                                            sx["cell_width"], "parenchyma", cx, cy,
                                            next_id, erosion_polygon=piece)
    return vessels, next_id


def fill_ray_parenchyma(cells, ray_zone: Polygon, sx: dict,
                        cx: float, cy: float, start_id: int) -> int:
    """Fill the interfascicular gap zones with radially-oriented ray parenchyma.

    A simple concentric-ring fill tagged ``parenchyma``; the strips between the
    xylem sectors read as the parenchyma rays.  Returns the next id.
    """
    next_id = start_id
    for g in _polys(ray_zone):
        next_id = fill_by_rings(cells, g, sx["parenchyma_diameter"], sx["parenchyma_width"],
                                "parenchyma", cx, cy, next_id, erosion_polygon=g)
    return next_id


# ---------------------------------------------------------------------------
# Annual growth rings (ported from DicotRootAnatomy._build_annual_bands)
# ---------------------------------------------------------------------------

def build_annual_bands(secondary_contour: Polygon, primary_contour: Polygon,
                       n_ring: int):
    """Radial growth-ring bands of the secondary-xylem annulus, following the
    secondary-cambium contour buffered inward in equal steps.  Returns a list of
    ``(band_polygon, r_inner, r_outer)`` inner→outer, or ``None`` when ``n_ring<=1``.
    """
    if n_ring <= 1:
        return None
    _, _, sc_r = GeometryProcessor._chebyshev_center(secondary_contour)
    _, _, pc_r = GeometryProcessor._chebyshev_center(primary_contour)
    step = max(sc_r - pc_r, 0.0) / n_ring
    if step <= 0:
        return None
    bands = []
    prev = secondary_contour
    for k in range(1, n_ring):
        inner = secondary_contour.buffer(-k * step)
        bands.append((prev.difference(inner), sc_r - k * step, sc_r - (k - 1) * step))
        prev = inner
    bands.append((prev, pc_r, sc_r - (n_ring - 1) * step))
    return bands


# ---------------------------------------------------------------------------
# Medullar rays (ported from DicotRootAnatomy, adapted to sector centres)
# ---------------------------------------------------------------------------

def radial_strip(cx: float, cy: float, theta: float, width: float,
                 r_outer: float, r_inner: float = 0.0) -> Polygon:
    """Constant-tangential-width radial strip centred on ``theta`` (a rectangle from
    ``r_inner`` to ``r_outer``), so a ray keeps the same physical width at every
    radius (unlike an angular wedge, which would taper)."""
    strip = box(r_inner, -width / 2.0, r_outer, width / 2.0)
    strip = rotate(strip, theta, origin=(0.0, 0.0), use_radians=True)
    return translate(strip, cx, cy)


def _bisect_circular(occupied: list) -> float:
    """Midpoint of the widest gap around a circle of thetas in ``[0, 2π)``."""
    m = len(occupied)
    if m == 0:
        return 0.0
    if m == 1:
        return (occupied[0] + np.pi) % (2.0 * np.pi)
    best_mid, best_gap = 0.0, -1.0
    for i in range(m):
        lo = occupied[i]
        hi = occupied[(i + 1) % m] + (2.0 * np.pi if i + 1 == m else 0.0)
        if hi - lo > best_gap:
            best_gap = hi - lo
            best_mid = (lo + hi) / 2.0
    return best_mid % (2.0 * np.pi)


def _bisect_bounded(bounds: list) -> float:
    """Midpoint of the widest interior gap in a sorted, bounded list."""
    best_mid, best_gap = bounds[0], -1.0
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if hi - lo > best_gap:
            best_gap = hi - lo
            best_mid = (lo + hi) / 2.0
    return best_mid


def _new_ray_start_radii(rng, n_new: int, span_base: float, span: float,
                         sd_mm: float, pc_r: float, r_outer: float) -> list:
    """Start radii for the ``n_new`` rate-driven rays, spread over
    ``[span_base, r_outer]`` then jittered by ``sd_mm`` and sorted ascending."""
    radii = []
    for j in range(n_new):
        r = span_base + (j + 0.5) / n_new * span
        if sd_mm > 0.0:
            r += rng.normal(0.0, sd_mm)
        radii.append(max(pc_r, min(r_outer * 0.999, r)))
    radii.sort()
    return radii


def build_medullar_ray_polygons(rng, annular_zone, vessel_zones, primary_contour,
                                cx, cy, r_outer_wedge, r_outer, sector_thetas,
                                half, prop_stele, mr, r_inner_min: float = 0.0) -> list:
    """Constant-width radial-strip polygons for each medullar ray, adapted to the
    stem's secondary-xylem *sectors* (centred on ``sector_thetas`` with angular
    half-width ``half``).

    ``n_medullar`` initial rays start at the primary ring; ``n_medullar_rate`` adds
    rays that start further out (density grows toward the periphery).  With
    ``allow_non_vascular`` False the rays live inside the vessel sectors (initial
    rays even per sector, rate-driven rays round-robin + bisected within a sector);
    with True (or ``prop_stele`` >= 1) they span the full circle.  Returns a list of
    ``(polygon, theta_c)``.
    """
    n_medullar = int(mr.get("n_medullar", 0))
    rate = float(mr.get("n_medullar_rate", 0.0))
    if n_medullar <= 0 and rate <= 0.0:
        return []

    base_width = float(mr.get("base_width", 0.005))
    allow_non_vascular = bool(mr.get("allow_non_vascular", False))
    n_sectors = max(len(sector_thetas), 1)

    _, _, pc_r = GeometryProcessor._chebyshev_center(primary_contour)
    annulus = max(r_outer - pc_r, 0.0)
    span_base = pc_r + float(mr.get("start_radius", 0.0)) * annulus
    span = max(r_outer - span_base, 0.0)
    n_new = int(round(rate * span)) if rate > 0.0 else 0
    sd_mm = float(mr.get("start_radius_sd", 0.0)) * annulus

    if allow_non_vascular:
        clip_zone = annular_zone
    else:
        valid = [z for z in vessel_zones if z is not None and not z.is_empty]
        if not valid:
            return []
        clip_zone = unary_union(valid)

    rays = []
    if allow_non_vascular or prop_stele >= 1.0:
        initial = ([2.0 * np.pi * k / n_medullar for k in range(n_medullar)]
                   if n_medullar > 0 else [])
        rays.extend((th, 0.0) for th in initial)
        occupied = sorted(t % (2.0 * np.pi) for t in initial)
        for r_j in _new_ray_start_radii(rng, n_new, span_base, span, sd_mm, pc_r, r_outer):
            th = _bisect_circular(occupied)
            occupied.append(th)
            occupied.sort()
            rays.append((th, r_j))
    else:
        rays_pp = n_medullar // n_sectors
        extra = n_medullar % n_sectors
        slice_bounds = []
        for pk in range(n_sectors):
            theta_zone = sector_thetas[pk]
            n_r = rays_pp + (1 if pk < extra else 0)
            slice_initial = []
            for j in range(n_r):
                offset = (2.0 * (j + 1) / (n_r + 1) - 1.0) * half
                th = theta_zone + offset
                slice_initial.append(th)
                rays.append((th, 0.0))
            slice_bounds.append([theta_zone - half, *sorted(slice_initial), theta_zone + half])
        for i, r_j in enumerate(
                _new_ray_start_radii(rng, n_new, span_base, span, sd_mm, pc_r, r_outer)):
            bounds = slice_bounds[i % n_sectors]
            th = _bisect_bounded(bounds)
            bounds.append(th)
            bounds.sort()
            rays.append((th, r_j))

    result = []
    for theta_c, r_inner in rays:
        poly = radial_strip(cx, cy, theta_c, base_width, r_outer_wedge,
                            max(r_inner, r_inner_min)).intersection(clip_zone)
        if not poly.is_empty:
            result.append((poly, theta_c))
    return result


def prepare_medullar_rays(cells, rng, annular_zone, vessel_zones, primary_contour,
                          cx, cy, r_outer_wedge, r_outer, sector_thetas, half,
                          prop_stele, cambium_cell_diameter, mr, r_inner_min: float = 0.0):
    """Build the medullar-ray corridors (before vessel packing so they can cut the
    vessel sectors) and clear cambium seeds inside them.  Returns
    ``(medullar_ray_polys, medullar_union)`` — ``([], None)`` when none requested.

    ``r_inner_min`` holds the ray corridors off the primary xylem: their inner tips
    start at this radius (a short margin outside the primary ring) so those cells are
    surrounded by dense secondary-xylem parenchyma instead of ballooning into the
    sparse xylem interface (a Voronoi artifact)."""
    if int(mr.get("n_medullar", 0)) <= 0 and float(mr.get("n_medullar_rate", 0.0)) <= 0.0:
        return [], None
    polys = build_medullar_ray_polygons(
        rng, annular_zone, vessel_zones, primary_contour, cx, cy,
        r_outer_wedge, r_outer, sector_thetas, half, prop_stele, mr,
        r_inner_min=r_inner_min)
    if not polys:
        return polys, None
    geoms = []
    for poly, _ in polys:
        geoms.extend(_polys(poly))
    if not geoms:
        return polys, None
    medullar_union = unary_union(geoms)
    # Remove cambium seeds inside the ray corridors (they become ray, not cambium).
    zone = medullar_union.buffer(cambium_cell_diameter)
    cambium = [c for c in cells.cells if c.type == "cambium"]
    if cambium:
        xs = np.fromiter((c.x for c in cambium), float, len(cambium))
        ys = np.fromiter((c.y for c in cambium), float, len(cambium))
        inside = sp.contains_xy(zone, xs, ys)
        drop = {id(c) for c, h in zip(cambium, inside) if h}
        cells.cells = [c for c in cells.cells if id(c) not in drop]
    return polys, medullar_union


def seed_radial_cell(cells, cell_type, px, py, theta_mid, r, d_cell, lane_width,
                     id_group, border_cos, border_sin, border_scale: float = 0.7):
    """Seed one radially-oriented elliptical cell as ``len(border_cos)`` border
    points sharing ``id_group`` (radial semi-axis ``d_cell/2``, tangential
    ``lane_width/2``, both scaled by ``border_scale`` and rotated to ``theta_mid``)."""
    a_rad = d_cell * 0.5 * border_scale
    b_tan = lane_width * 0.5 * border_scale
    cos_t, sin_t = np.cos(theta_mid), np.sin(theta_mid)
    for j in range(len(border_cos)):
        er = a_rad * border_cos[j]
        et = b_tan * border_sin[j]
        cells.add_cell(Cell(
            type=cell_type,
            x=px + er * cos_t - et * sin_t,
            y=py + er * sin_t + et * cos_t,
            diameter=d_cell, id_cell=id_group, id_group=id_group,
            angle=theta_mid, radius=r, area=np.pi * a_rad * b_tan,
        ))


def fill_medullar_rays(cells, medullar_poly, theta_c, cx, cy, mr, start_id) -> int:
    """Fill a medullar-ray corridor with ``medullar_ray`` cells, holding the
    tangential width constant at ``base_width`` at every radius (angular half-extent
    recomputed as ``base_width / (2 r)``)."""
    if medullar_poly is None or medullar_poly.is_empty:
        return start_id
    d_cell = float(mr.get("cell_diameter", 0.025))
    w_cell = float(mr.get("cell_width", 0.005))
    base_width = float(mr.get("base_width", 0.005))
    n_lanes = max(1, int(np.ceil(base_width / max(w_cell, 1e-9))))
    lane_width = base_width / n_lanes

    n_border = 25
    phi = np.linspace(0.0, 2.0 * np.pi, n_border, endpoint=False)
    border_cos, border_sin = np.cos(phi), np.sin(phi)
    next_id = start_id

    for geom in _polys(medullar_poly):
        radii = [np.hypot(x - cx, y - cy) for x, y in geom.exterior.coords]
        r_inner, r_outer = min(radii), max(radii)
        lanes = np.arange(n_lanes)
        r = max(r_inner - d_cell / 2.0, d_cell / 2.0)
        while r <= r_outer:
            half_angle_r = base_width / (2.0 * r)
            theta_lo_r = theta_c - half_angle_r
            theta_hi_r = theta_c + half_angle_r
            theta_mid = theta_lo_r + (lanes + 0.5) * (theta_hi_r - theta_lo_r) / n_lanes
            px = cx + r * np.cos(theta_mid)
            py = cy + r * np.sin(theta_mid)
            inside = sp.contains_xy(geom, px, py)
            for lane in range(n_lanes):
                if not inside[lane]:
                    continue
                id_group = next_id
                next_id += 1
                seed_radial_cell(cells, "medullar_ray", px[lane], py[lane],
                                 theta_mid[lane], r, d_cell, lane_width, id_group,
                                 border_cos, border_sin)
            r += d_cell
    return next_id


def fill_ray_parenchyma_split(cells, rng, vessel_zones, annular_zone, cx, cy, sx,
                              r_outer, gap_thetas, gap_half, r_start, start_id) -> int:
    """Ray parenchyma in the interfascicular gaps, filled as radial lanes that split
    (lane count doubles) as the arc widens outward — the root's ``_fill_ray_parenchyma``,
    keyed on the gap centres ``gap_thetas`` (midway between adjacent xylem sectors)
    each spanning ``± gap_half``.  Returns the next id."""
    valid = [z for z in vessel_zones if z is not None and not z.is_empty]
    if annular_zone is None or annular_zone.is_empty or r_outer <= 0.0 or gap_half <= 0.0:
        return start_id
    d_cell = float(sx["parenchyma_diameter"])
    w_cell = float(sx["parenchyma_width"])
    sd_cell = float(sx.get("parenchyma_diameter_sd", 0.0))
    split_threshold = w_cell + 3.0 * sd_cell

    zones_union = unary_union(valid) if valid else None
    ray_zone = annular_zone.difference(zones_union) if zones_union is not None else annular_zone
    if ray_zone.is_empty:
        return start_id

    n_border = 15
    phi = np.linspace(0.0, 2.0 * np.pi, n_border, endpoint=False)
    border_cos, border_sin = np.cos(phi), np.sin(phi)
    next_id = start_id

    for theta_c in gap_thetas:
        theta_lo = theta_c - gap_half
        theta_hi = theta_c + gap_half
        r_start_k = max(r_start, d_cell)
        init_spacing = w_cell / r_start_k
        n_init = max(1, int(np.ceil((theta_hi - theta_lo) / init_spacing)))
        lines = list(np.linspace(theta_lo, theta_hi, n_init + 1))
        thresholds = [
            float(np.clip(rng.uniform(0.7, 1.3) * split_threshold,
                          0.5 * split_threshold, 1.5 * split_threshold))
            for _ in range(len(lines) - 1)
        ]
        r = r_start_k + d_cell / 2.0
        while r <= r_outer:
            new_lines = [lines[0]]
            new_thresholds = []
            noise_scale = 0.1 * split_threshold
            for i in range(len(lines) - 1):
                a1, a2 = lines[i], lines[i + 1]
                thr = thresholds[i]
                if (a2 - a1) * r > thr:
                    new_lines.append((a1 + a2) / 2.0)
                    t_left = float(np.clip(thr + rng.normal(0, noise_scale),
                                           0.5 * split_threshold, 1.5 * split_threshold))
                    t_right = float(np.clip(thr + rng.normal(0, noise_scale),
                                            0.5 * split_threshold, 1.5 * split_threshold))
                    new_thresholds.extend([t_left, t_right])
                else:
                    new_thresholds.append(thr)
                new_lines.append(a2)
            lines = sorted(new_lines)
            thresholds = new_thresholds

            la = np.asarray(lines[:-1])
            lb = np.asarray(lines[1:])
            theta_mid_arr = (la + lb) / 2.0
            px_arr = cx + r * np.cos(theta_mid_arr)
            py_arr = cy + r * np.sin(theta_mid_arr)
            inside = sp.contains_xy(ray_zone, px_arr, py_arr)
            for i in range(len(lines) - 1):
                if not inside[i]:
                    continue
                lane_arc_width = (lines[i + 1] - lines[i]) * r
                id_group = next_id
                next_id += 1
                seed_radial_cell(cells, "parenchyma", px_arr[i], py_arr[i],
                                 theta_mid_arr[i], r, d_cell, lane_arc_width, id_group,
                                 border_cos, border_sin)
            r += d_cell
    return next_id


# ---------------------------------------------------------------------------
# Secondary phloem (ported from DicotRootAnatomy)
# ---------------------------------------------------------------------------

def cambium_local_frame(cam_ext, cx: float, cy: float, theta: float, r_far: float):
    """Anchor point on the cambium exterior at ``theta`` + local (tangent, normal).

    A ray from the centre at ``theta`` meets the cambium exterior at ``P``; the
    tangent is read either side of ``P`` and the outward normal is its
    perpendicular.  Returns ``(P, (tx, ty), (nx, ny))`` or ``None``.
    """
    from shapely.geometry import LineString
    ray = LineString([(cx, cy), (cx + r_far * np.cos(theta), cy + r_far * np.sin(theta))])
    inter = ray.intersection(cam_ext)
    if inter.is_empty:
        return None
    pts = [inter] if inter.geom_type == "Point" else [g for g in inter.geoms if g.geom_type == "Point"]
    if not pts:
        return None
    pt = max(pts, key=lambda p: (p.x - cx) ** 2 + (p.y - cy) ** 2)
    L = cam_ext.length
    s = cam_ext.project(pt)
    eps = max(L * 1e-3, 1e-5)
    a = cam_ext.interpolate((s - eps) % L)
    b = cam_ext.interpolate((s + eps) % L)
    tx, ty = b.x - a.x, b.y - a.y
    tn = np.hypot(tx, ty) or 1.0
    tx, ty = tx / tn, ty / tn
    nx, ny = ty, -tx
    if nx * (pt.x - cx) + ny * (pt.y - cy) < 0:
        nx, ny = -nx, -ny
    return (pt.x, pt.y), (tx, ty), (nx, ny)


def phloem_trapeze_curved(cam_ext, cx: float, cy: float, P, tangent, normal,
                          base_arc_half_width: float, top_width: float,
                          height: float, n: int = 40) -> Polygon:
    """Trapeze standing on the cambium at ``P``, base following the cambium curve,
    tapering to ``top_width`` at ``height`` along the outward normal."""
    (px, py), (tx, ty), (nx, ny) = P, tangent, normal
    L = cam_ext.length
    s0 = cam_ext.project(Point(px, py))
    inset = height * 0.3
    base = []
    for si in np.linspace(s0 - base_arc_half_width, s0 + base_arc_half_width, n):
        q = cam_ext.interpolate(si % L)
        dx, dy = q.x - cx, q.y - cy
        d = np.hypot(dx, dy) or 1.0
        base.append((q.x - dx / d * inset, q.y - dy / d * inset))
    hw = top_width / 2.0
    tcx, tcy = px + nx * (height + inset), py + ny * (height + inset)
    top_r = (tcx + tx * hw, tcy + ty * hw)
    top_l = (tcx - tx * hw, tcy - ty * hw)
    return Polygon(base + [top_r, top_l]).buffer(0)


def fill_phloem_zone(cells, rng, zone, alive: bool, cx: float, cy: float,
                     sp: dict, start_id: int) -> int:
    """Place sieve tubes (+ companion cells when ``alive``) then parenchyma around
    them, hugging their footprints.  Ported from the dicot root.

    Sieve tubes are tagged ``sieve element``, companions ``companion cell`` and
    the ground tissue ``parenchyma`` (the stem palette).
    """
    if zone is None or zone.is_empty:
        return start_id

    sub_zones = _polys(zone)
    min_area = np.pi * (sp["sieve_diameter_min"] / 2) ** 2
    next_id = start_id
    voronoi_grow = 0.25 * (sp["parenchyma_diameter"] + sp["parenchyma_width"])
    sieve_r_floor = sp["parenchyma_diameter"] * 0.4

    for arm_zone in sub_zones:
        if arm_zone.area < min_area:
            continue
        proportion = sp["prop_sieve"] * sp["sieve_diameter"] ** 2 / (
            sp["sieve_diameter"] ** 2 + (sp["companion_diameter"] * sp["companion_width"]))
        packed = GeometryProcessor.pack_circles(
            arm_zone, proportion=proportion, direction=None,
            diameter_max=sp["sieve_diameter"], diameter_min=sp["sieve_diameter_min"],
            diameter_sd=sp["sieve_diameter_sd"], gradient_function="normal", rng=rng,
        )
        sieve_polys, sieve_centers, companion_polys = [], [], []
        combined_budget = sp["prop_sieve"] * arm_zone.area
        comp_area_each = (np.pi * (sp["companion_diameter"] / 2) * (sp["companion_width"] / 2)) if alive else 0.0
        combined_area = 0.0

        for pcx, pcy, r in packed:
            increment = np.pi * r ** 2 + comp_area_each
            if sieve_polys and combined_area + increment > combined_budget:
                break
            r_draw = min(r, max(r - voronoi_grow / 3, sieve_r_floor))
            placed = Point(pcx, pcy).buffer(r_draw, resolution=32)
            placed_buff = placed.buffer(-r_draw * 0.15)
            if placed_buff.is_empty:
                continue
            bx, by = placed_buff.exterior.coords.xy
            border = GeometryProcessor.resample_coords(np.column_stack((bx, by)), target_n_points=25)
            gid = next_id
            next_id += 1
            for pt in border[1:]:
                cells.add_cell(Cell.radial("sieve element", pt[0], pt[1], r_draw * 2, gid, (cx, cy)))
            sieve_polys.append(placed)
            sieve_centers.append((pcx, pcy, r_draw))
            combined_area += increment

        if alive and sieve_polys:
            sieve_union = unary_union(sieve_polys)
            comp_d = max(sp["companion_diameter"] - voronoi_grow, sp["companion_width"])
            for pcx, pcy, r in sieve_centers:
                comp_r = comp_d / 2
                theta0 = rng.uniform(0.0, 2 * np.pi)
                for k in range(8):
                    angle = theta0 + 2 * np.pi * k / 8
                    ccx = pcx + (r + comp_r * 1.05) * np.cos(angle)
                    ccy = pcy + (r + comp_r * 1.05) * np.sin(angle)
                    cpt = Point(ccx, ccy)
                    if not arm_zone.contains(cpt):
                        continue
                    circle = cpt.buffer(comp_r)
                    if circle.intersects(sieve_union) or any(circle.intersects(c) for c in companion_polys):
                        continue
                    cbuff = circle.buffer(-comp_r * 0.15)
                    if cbuff.is_empty:
                        continue
                    bx, by = cbuff.exterior.coords.xy
                    border = GeometryProcessor.resample_coords(np.column_stack((bx, by)), target_n_points=16)
                    gid = next_id
                    next_id += 1
                    for pt in border[1:]:
                        cells.add_cell(Cell.radial("companion cell", pt[0], pt[1],
                                                   sp["companion_diameter"], gid, (cx, cy)))
                    companion_polys.append(circle)
                    break

        placed_union = (unary_union(sieve_polys + companion_polys)
                        if (sieve_polys or companion_polys) else Polygon())
        fill_zone = arm_zone.difference(placed_union)
        if not fill_zone.is_empty:
            next_id = fill_by_rings(cells, fill_zone, sp["parenchyma_diameter"],
                                    sp["parenchyma_width"], "parenchyma", cx, cy, next_id,
                                    erosion_polygon=fill_zone, initial_space=0.0)
    return next_id
