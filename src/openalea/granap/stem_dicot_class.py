"""Dicot stem anatomy (eustele).

``DicotStemAnatomy`` builds a dicot stem: a central pith ringed by a single ring
of discrete *collateral* vascular bundles — xylem on the inner (pith) face,
phloem on the outer (cortex) face, with a strip of fascicular cambium between —
then a cortex and epidermis outside.  Instantiate via ``StemAnatomy(input_data)``
— the factory in :mod:`openalea.granap.stem_class` dispatches here when
``planttype == 2``.

The bundles are placed on an evenly-spaced ring at the pith/cortex boundary
(``_bundle_ring_positions``).  The requested ``n_bundles`` is clamped to the
count that fits without adjacent bundle envelopes overlapping (a warning is
issued when it has to be reduced), which keeps the ring symmetric.
"""

import logging
import warnings
from typing import List, Tuple

import numpy as np
from shapely.affinity import translate
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from openalea.granap.geometry_collection import GeometryProcessor
from openalea.granap.tissue_class import TissueRecipe, fill_along
from openalea.granap.stem_class import StemAnatomy
from openalea.granap.vascular_bundle import build_bundle, bundle_cambium_anchor
from openalea.granap import secondary_growth as sg

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dicot stem subclass
# ---------------------------------------------------------------------------

class DicotStemAnatomy(StemAnatomy):
    """Dicot stem: a ring of discrete collateral bundles (xylem in / phloem out /
    cambium between) around a central pith (eustele)."""

    # ------------------------------------------------------------------
    # Vascular tissue
    # ------------------------------------------------------------------
    #
    # No _parse_vascular_params override: build_bundle reads the raw xylem /
    # phloem / cambium param dicts directly, and the bundle count is the
    # vascular_bundle.n_bundles field — so there is nothing to pre-parse.

    def _vascular_recipe(self, polygon: Polygon) -> TissueRecipe:
        """Declarative description of how the eustele bundle ring is assembled.

        Built and run by the shared ``Organ._create_vascular_tissue`` scaffold;
        the remove-mask + extend step runs later in ``Organ.generate_cells``.
        The build order is data, inspectable via ``recipe.describe()``.

        The single ``collateral bundle ring`` step defers to
        :meth:`_build_bundle_ring`, which builds one collateral bundle per ring
        slot.
        """
        recipe = TissueRecipe().bind(lambda: self.vascular_cells, self.rng)
        bp = self._get_param("vascular_bundle")
        if not bp or int(bp.get("n_bundles", 0)) == 0:
            return recipe                       # no bundles -> empty
        recipe.special(
            "collateral bundle ring",
            lambda: self._build_bundle_ring(polygon),
            produces=("xylem", "cambium", "phloem", "bundle sheath"),
        )
        return recipe

    # ------------------------------------------------------------------
    # Cambium-ring contour (the shape the bundles are placed on)
    # ------------------------------------------------------------------

    def _secondary_growth(self) -> bool:
        """True when secondary growth is requested."""
        sec = self._get_param("secondary_growth")
        return bool(sec.get("value", False)) if sec else False

    def _secondary_annulus_thickness(self) -> float:
        """Radial thickness the stem gains under secondary growth: the secondary
        xylem annulus (``secondary_cambium.growth``) + the secondary phloem band
        (``secondary_phloem.height``) + the displaced primary-phloem remnant pushed
        just outside it.  Zero under primary growth."""
        if not self._secondary_growth():
            return 0.0
        sc = self._get_param("secondary_cambium") or {}
        sp = self._get_param("secondary_phloem") or {}
        return (float(sc.get("growth", 0.0)) + float(sp.get("height", 0.0))
                + self._primary_phloem_thickness())

    def _primary_phloem_thickness(self) -> float:
        """Radial thickness reserved outside the secondary phloem for the displaced
        primary-phloem remnant — the bundle's phloem-fraction share of its height.

        Zero unless ``secondary_phloem.keep_primary`` is set: secondary growth
        crushes the primary phloem, so it is not rendered (nor room reserved for it)
        by default."""
        sp = self._get_param("secondary_phloem") or {}
        if not sp.get("keep_primary", False):
            return 0.0
        bp = self._get_param("vascular_bundle") or {}
        return float(bp.get("height", 0.0)) * float(bp.get("phloem_fraction", 0.35))

    def _calculate_radius(self) -> float:
        """Stem outer radius, grown outward to make room for secondary growth."""
        return super()._calculate_radius() + self._secondary_annulus_thickness()

    def _primary_ring_radius(self, polygon: Polygon) -> float:
        """Radius of the primary bundle ring (where the bundles / fascicular cambium
        sit).

        Under secondary growth the central region is enlarged to make room for the
        secondary-xylem annulus, so the primary ring is pinned to the pith radius
        (``thickness/2``) rather than the enlarged region's radius; primary growth
        keeps the historical ``sqrt(area/pi)``.
        """
        if self._secondary_growth():
            return float(self.vascular_params["thickness"]) / 2.0
        return float(np.sqrt(polygon.area / np.pi))

    def _ring_contour_at(self, cx: float, cy: float, r: float, shape: str,
                         ratio: float = 0.75, branches: int = 5,
                         amp: float = 0.12) -> Polygon:
        """A ring contour of radius ``r`` in the eustele ``ring_shape`` family.

        Shared by the primary bundle ring and the secondary cambium: a circle by
        default; an ``ellipse`` flattened by ``ratio``; or a lobed ``star`` of
        ``branches`` arms with valley depth ``amp`` (fraction of ``r``).
        """
        if shape == "ellipse":
            return GeometryProcessor.ellipse_to_polygon(cx, cy, r, r * float(ratio), 0.0)
        if shape == "star":
            n = max(int(branches), 2)
            a = min(max(float(amp), 0.0), 0.9)
            r_min = r * (1.0 - a)
            star = GeometryProcessor.star_polygon(
                n_branches=n, r_min=r_min, r_max=r,
                arc_base=0.5 * np.pi * r_min / n, arc_top=0.35 * np.pi * r / n,
            )
            return translate(star, cx, cy)
        return translate(GeometryProcessor.circle_polygon(r), cx, cy)

    def _cambium_ring_contour(self, polygon: Polygon) -> Polygon:
        """The drawn contour the eustele bundles are placed on — the (primary)
        cambium ring — in the ``ring_shape`` family at the primary ring radius."""
        bp = self._get_param("vascular_bundle")
        cx, cy = polygon.centroid.x, polygon.centroid.y
        return self._ring_contour_at(
            cx, cy, self._primary_ring_radius(polygon),
            bp.get("ring_shape", "circle"), bp.get("ring_ellipse_ratio", 0.75),
            bp.get("ring_star_branches", 5), bp.get("ring_star_amplitude", 0.12),
        )

    def _contour_slots(self, contour: Polygon, cx: float, cy: float,
                       n: int) -> List[Tuple[float, float, float]]:
        """``n`` evenly-spaced ``(px, py, theta)`` slots along ``contour``.

        Points are sampled by arc length around the contour; ``theta`` is the
        outward normal at each (from the local tangent of neighbouring samples),
        so every bundle points away from the organ centre wherever it sits on a
        circle, ellipse or star arm.
        """
        if n <= 0:
            return []
        coords = np.asarray(contour.exterior.coords)
        pts = GeometryProcessor.resample_coords(coords, target_n_points=n + 1)
        slots = []
        for i in range(n):
            px, py = pts[i]
            nx_pt, ny_pt = pts[(i + 1) % n]
            # (i - 1) % n, not i - 1: the resampled ring is closed (pts[n] == pts[0]),
            # so a bare pts[-1] is pts[0] itself — the current point — which flattens
            # the tangent for slot 0 and mis-orients that bundle.
            px_pt, py_pt = pts[(i - 1) % n]
            tx, ty = nx_pt - px_pt, ny_pt - py_pt          # local tangent
            nx, ny = ty, -tx                                # rotate -90 -> a normal
            if nx * (px - cx) + ny * (py - cy) < 0:         # make it point outward
                nx, ny = -nx, -ny
            slots.append((float(px), float(py), float(np.arctan2(ny, nx))))
        return slots

    def _ring_slots(self, polygon: Polygon) -> List[Tuple[float, float, float]]:
        """Bundle slots along the cambium-ring contour, clamped to non-overlap.

        The requested ``n_bundles`` is reduced until no two adjacent bundle
        envelopes touch (a warning is issued when it must drop); because
        resampling re-spaces the slots, the whole slot set is recomputed at each
        candidate count so the ring stays evenly spaced.
        """
        bp = self._bundle_params()
        n_req = int(bp.get("n_bundles", 0))
        if n_req <= 0:
            return []
        contour = self._cambium_ring_contour(polygon)
        cx, cy = polygon.centroid.x, polygon.centroid.y
        r_pith = np.sqrt(polygon.area / np.pi)
        anchor = bundle_cambium_anchor(bp)
        gap = self._bundle_clearance(bp)
        ground = self._pith_cell_diameter_at(r_pith, r_pith)

        for n in range(n_req, 0, -1):
            slots = self._contour_slots(contour, cx, cy, n)
            envs = [self._placed_bundle_envelope(px, py, th, bp, ground, anchor)
                    for px, py, th in slots]
            if n == 1 or all(
                not self._bundle_overlaps(envs[i], [envs[(i + 1) % n]], gap)
                for i in range(n)
            ):
                if n < n_req:
                    warnings.warn(
                        f"DicotStemAnatomy: {n_req} bundles overlap on the "
                        f"{bp.get('ring_shape', 'circle')} ring; placing {n} "
                        f"evenly-spaced non-overlapping bundles instead (reduce "
                        f"vascular_bundle.width/height or n_bundles to fit more).",
                        stacklevel=2,
                    )
                return slots
        return []

    def _cambium_band_thickness(self, cambium: dict) -> float:
        """Radial thickness of the cambium band = ``n_layers`` cell files."""
        n = max(int(cambium.get("n_layers", 2)), 1)
        return n * float(cambium.get("cell_diameter", 0.01))

    def _bundle_params(self) -> dict:
        """Bundle params with the fascicular cambium gap sized to the ring.

        The cambium — the fascicular strip *and* the secondary ring — is one band
        of ``n_layers`` cell files laid on the ring contour.  For the fascicular
        strip to hold the same ``n_layers`` as the interfascicular ring, the bundle
        must reserve a gap at least that thick between xylem and phloem; so this
        returns a *copy* of the ``vascular_bundle`` param with ``cambium_fraction``
        raised to fit (banded open bundles only), leaving everything else untouched.
        """
        bp = dict(self._get_param("vascular_bundle") or {})
        if not bp:
            return bp
        if bp.get("has_cambium", True) and bp.get("bundle_type", "collateral") != "concentric":
            cambium = self._get_param("cambium") or {}
            t = self._cambium_band_thickness(cambium)
            h = float(bp.get("height", 0.18)) or 0.18
            bp["cambium_fraction"] = max(float(bp.get("cambium_fraction", 0.08)),
                                         1.2 * t / h)
        return bp

    def _build_bundle_ring(self, polygon: Polygon) -> None:
        """Build the eustele: bundles on a shared cambium contour, then the cambium.

        The cambium *shape* is defined once — the ring contour
        (:meth:`_cambium_ring_contour`) and its ``n_layers`` cell files — and each
        bundle is placed on it, anchored on its cambium band
        (``bundle_cambium_anchor``) so every fascicular strip sits on the contour.
        Bundles are built *without* their cambium (``fill_cambium=False``); the
        cambium is then laid down in one pass by :meth:`_build_cambium`, which
        materialises either just the in-bundle arcs (primary growth) or the entire
        ring (secondary growth).  Building it in one place is what makes the
        fascicular and interfascicular cambium share the same number of layers.

        Each bundle envelope is registered in ``vascular_tissue_polygons`` so
        ``generate_cells`` clears the pith/cortex seeds underneath it.
        """
        bp = self._bundle_params()
        xylem = self._get_param("xylem")
        phloem = self._get_param("phloem")
        cambium = self._get_param("cambium")
        if not bp:
            return
        if self._secondary_growth():
            # Full secondary growth replaces the discrete-bundle path.
            self._build_secondary_growth(polygon)
            return
        cx0, cy0 = polygon.centroid.x, polygon.centroid.y
        r_pith = np.sqrt(polygon.area / np.pi)
        anchor = bundle_cambium_anchor(bp)
        contour = self._cambium_ring_contour(polygon)
        slots = self._ring_slots(polygon)

        conducting: List[Polygon] = []      # xylem / phloem zones — the ring avoids these
        fascicular: List[Polygon] = []      # per-bundle cambium zones — the primary clip region
        sheath: List[Polygon] = []          # bundle-sheath zones — the ring must not eat these
        for cx, cy, theta in slots:
            ground = self._pith_cell_diameter_at(np.hypot(cx - cx0, cy - cy0), r_pith)
            res = build_bundle(self.vascular_cells, self.rng, cx, cy, theta,
                               bp, xylem, phloem, cambium,
                               ground_cell_size=ground, anchor=anchor,
                               fill_cambium=False)
            self._register_bundle(res)
            for role, g in res.zone_polygons:
                if g is None or g.is_empty:
                    continue
                if role in ("xylem", "phloem"):
                    conducting.append(g)
                elif role == "cambium":
                    fascicular.append(g)
                elif role in ("bundle sheath", "sclerenchyma", "parenchyma"):
                    sheath.append(g)

        # Primary growth: fascicular cambium only (secondary growth branched off
        # earlier into _build_secondary_growth).
        self._build_cambium(contour, fascicular, conducting, cambium,
                            secondary=False, sheath=sheath)

    def _build_cambium(self, contour: Polygon, fascicular: List[Polygon],
                       conducting: List[Polygon], cambium: dict,
                       secondary: bool, sheath: List[Polygon] = None) -> None:
        """Lay the vascular cambium as ``n_layers`` cell files along the contour.

        The cambium shape is fixed (the contour and its ``n_layers`` files); what
        gets *materialised* depends on the growth stage:

        * **primary** (``secondary`` False): only the arcs that fall **inside a
          bundle** — the fascicular cambium strips (``keep_union`` = the bundle
          cambium zones).  These zones are already registered per bundle.
        * **secondary** (``secondary`` True): the **entire ring**.  The bundle
          sheaths (and the parenchyma bundle-sheath file) lying on the band are
          removed first, so the ring replaces them and runs unbroken through every
          bundle; the whole band is then registered as ``cambium``.

        Either way the files are blocked from the conducting tissues (xylem /
        phloem), so the cambium never overwrites a vessel or sieve element.  Because
        both stages use the *same* file loop, the fascicular and interfascicular
        cambium always carry the same number of layers.
        """
        cx, cy = contour.centroid.x, contour.centroid.y
        n_layers = max(int(cambium.get("n_layers", 2)), 1)
        cell_d = float(cambium.get("cell_diameter", 0.01))
        cell_w = float(cambium.get("cell_width", cell_d))
        conducting_union = unary_union(conducting) if conducting else None
        block_union = conducting_union

        if secondary:
            half = max(n_layers * cell_d / 2.0, cell_d / 2.0)
            band = contour.buffer(half).difference(contour.buffer(-half))
            if conducting_union is not None:
                band = band.difference(conducting_union)
            if band.is_empty:
                return
            # Remove the bundle sheaths on the band so the ring replaces them
            # instead of overlapping surviving sheath cells.
            self._clear_vascular_cells(band, ("bundle sheath", "parenchyma", "sclerenchyma"))
            keep_union = None
            register = band
        else:
            if not fascicular:
                return
            # Confine the cambium to the bundle cambium zones (buffered a touch so
            # points on the strip edge are not dropped by the point-in test), but
            # hold it a cell-width clear of the bundle sheath: a seed centre may pass
            # the sheath block yet still have its border cells reach into the sheath,
            # so keep the whole seed a margin away from it.
            sheath_union = unary_union(sheath) if sheath else None
            keep_union = unary_union(fascicular).buffer(0.25 * cell_d)
            if sheath_union is not None:
                keep_union = keep_union.difference(sheath_union.buffer(0.5 * cell_d))
            register = None      # the fascicular zones are registered per bundle
            # Also block the fascicular files from the sheath (and the conducting
            # tissues) so their files never seed on the sheath ring.
            blockers = list(conducting) + list(sheath or [])
            block_union = unary_union(blockers) if blockers else None

        for k in range(n_layers):
            off = (k - (n_layers - 1) / 2.0) * cell_d
            ring = contour.buffer(off) if off else contour
            if ring.is_empty:
                continue
            fill_along(self.vascular_cells, ring.exterior, "cambium",
                       cell_d, cell_w, cx, cy,
                       xylem_union=block_union, keep_union=keep_union)

        # Primary: a band-edge seed can pass the sheath block yet still have a border
        # cell reach into the sheath.  Drop any cambium group that touches the sheath
        # so the sheath keeps its ring unbroken (the secondary path clears the sheath
        # the other way — the ring replaces it).
        if not secondary and sheath:
            self._clear_vascular_cells(unary_union(sheath), ("cambium",))

        if register is not None and not register.is_empty:
            self.vascular_tissue_polygons.setdefault("cambium", []).append(register)

    def _clear_vascular_cells(self, region: Polygon, types: Tuple[str, ...]) -> None:
        """Drop whole vascular-cell groups of ``types`` whose seed lies in ``region``.

        Seed cells are grouped by ``id_group`` (one Voronoi cell per group); a group
        is removed entirely if any of its seeds of a matching type falls inside
        ``region``, so no half-cells are left behind.
        """
        if region is None or region.is_empty:
            return
        from shapely.geometry import Point
        drop = {c.id_group for c in self.vascular_cells.cells
                if c.type in types and region.contains(Point(c.x, c.y))}
        if not drop:
            return
        self.vascular_cells.cells = [
            c for c in self.vascular_cells.cells
            if not (c.id_group in drop and c.type in types)
        ]

    # ------------------------------------------------------------------
    # Secondary growth (the cambium has closed and moved outward)
    # ------------------------------------------------------------------

    def _build_secondary_growth(self, polygon: Polygon) -> None:
        """Secondary growth that *grows* the primary bundle instead of replacing it.

        The transition is only partially destructive: the **primary xylem** is
        endarch and cannot move, so it stays pinned at the pith edge and the
        secondary xylem is built radially on top of it; the **primary phloem** is
        crushed outward ahead of the new secondary phloem.  Radial stack, pith→cortex:

        * **primary xylem** — a thin remnant per bundle pinned just inside the primary
          ring (:meth:`_build_primary_xylem`), reusing ``build_bundle``;
        * **secondary xylem** — the annulus between the primary ring and the secondary
          cambium (minus the cambial band shell), split into a graded-vessel sector
          behind each bundle (``prop_stele`` wide, so ``prop_stele`` → 1 merges the
          sectors into a continuous cylinder) with **parenchyma rays** in the gaps;
        * **secondary cambium** — a closed ring at the primary ring grown outward by
          ``secondary_cambium.growth`` (same ``ring_shape`` family);
        * **secondary phloem** — a band just outside the cambium (root-style trapezes,
          alive/dead split), one arm behind each bundle;
        * **primary phloem** — a thin remnant per bundle pushed just outside the
          secondary phloem band (:meth:`_build_primary_phloem`).
        """
        cx, cy = polygon.centroid.x, polygon.centroid.y
        bp = self._get_param("vascular_bundle")
        sc = self._get_param("secondary_cambium")
        sx = self._get_param("secondary_xylem")
        xylem = self._get_param("xylem")
        phloem = self._get_param("phloem")
        cambium = self._get_param("cambium")
        n = int(bp.get("n_bundles", 0)) if bp else 0
        if n <= 0 or not sc or not sx:
            return

        r_prim = self._primary_ring_radius(polygon)
        r_sec = r_prim + float(sc["growth"])
        primary_contour = self._ring_contour_at(
            cx, cy, r_prim, bp.get("ring_shape", "circle"),
            bp.get("ring_ellipse_ratio", 0.75), bp.get("ring_star_branches", 5),
            bp.get("ring_star_amplitude", 0.12))
        secondary_contour = self._ring_contour_at(
            cx, cy, r_sec, sc.get("shape", "circle"),
            sc.get("ring_ellipse_ratio", 0.75), sc.get("ring_star_branches", 5),
            sc.get("ring_star_amplitude", 0.12))

        band_depth = int(sc["n_layers"]) * float(sc["cell_diameter"])
        xylem_boundary = secondary_contour.buffer(-band_depth)
        if xylem_boundary.is_empty:
            xylem_boundary = secondary_contour
        annulus = xylem_boundary.difference(primary_contour)
        if annulus.is_empty:
            return

        angles = [2.0 * np.pi * k / n for k in range(n)]
        half = (np.pi / n) * float(sx["prop_stele"])
        base_half_width = float(bp.get("width", 0.1)) / 2.0
        flare = np.radians(float(sx.get("flare_angle", 30.0)))
        r_outer_wedge = r_sec * 2.0

        # --- preserved primary xylem (pinned at the pith edge) -------------
        self._build_primary_xylem(primary_contour, angles, bp, xylem, phloem,
                                   cambium, cx, cy, r_outer_wedge)

        # --- secondary cambium ring ----------------------------------------
        sg.render_cambium_files(self.vascular_cells, secondary_contour,
                                int(sc["n_layers"]), float(sc["cell_diameter"]),
                                float(sc["cell_width"]), cx, cy)
        cambial_shell = secondary_contour.difference(secondary_contour.buffer(-band_depth))
        if not cambial_shell.is_empty:
            self.vascular_tissue_polygons.setdefault("cambium", []).append(cambial_shell)

        # --- secondary xylem sectors (behind each bundle) ------------------
        # Flared sectors: start at the bundle width against the primary xylem and
        # widen outward (see flared_wedge), so they only merge once their capped
        # edges meet further out.
        sectors = []
        for th in angles:
            zone = sg.flared_wedge(cx, cy, th, r_prim, r_outer_wedge,
                                   base_half_width, flare, half).intersection(annulus)
            if not zone.is_empty:
                sectors.append(zone)

        # Real outer xylem radius (inner edge of the cambium band) — the span over
        # which rate-driven medullar rays are initiated and vessels graded.
        r_outer = r_sec - band_depth

        # Medullar rays are built BEFORE vessel packing so they cut the sectors, and
        # the growth-ring bands re-grade the vessels per ring (n_ring).
        mr = self._get_param("medullar_rays") or {}
        # Start the rays just clear of the primary xylem (half a cell) so they run
        # through the whole secondary xylem yet their inner tips still sit in dense
        # parenchyma rather than ballooning against the sparse primary-xylem edge.
        ray_inner_min = r_prim + 0.5 * float(sx["cell_diameter"])
        medullar_polys, medullar_union = sg.prepare_medullar_rays(
            self.vascular_cells, self.rng, annulus, sectors, primary_contour,
            cx, cy, r_outer_wedge, r_outer, angles, half, float(sx["prop_stele"]),
            float(sc["cell_diameter"]), mr, r_inner_min=ray_inner_min)
        annual_bands = sg.build_annual_bands(secondary_contour, primary_contour,
                                             int(sx.get("n_ring", 1)))

        next_id = self.vascular_cells.next_group_id()
        vessels = []
        for zone in sectors:
            vs, next_id = sg.fill_secondary_xylem_sector(
                self.vascular_cells, self.rng, zone, sx, cx, cy, next_id,
                annual_bands=annual_bands, medullar_union=medullar_union)
            vessels.extend(vs)
        if sectors:
            self.vascular_tissue_polygons.setdefault("xylem", []).append(unary_union(sectors))
        self.vascular_polygons.extend(vessels)

        # --- parenchyma rays (the interfascicular gaps between bundles) -----
        # Lane-splitting radial fill in a fixed ±gap_half wedge midway between
        # adjacent sectors.
        gap_thetas = [th + np.pi / n for th in angles]
        gap_half = max(np.pi / n - half, 0.0)
        next_id = sg.fill_ray_parenchyma_split(
            self.vascular_cells, self.rng, sectors, annulus, cx, cy, sx,
            r_outer, gap_thetas, gap_half, r_prim, next_id)
        # Register ONLY the wedge footprint actually filled with ray parenchyma, not
        # the whole interfascicular gap: the sectors flare from the bundle width, so
        # near the pith the gap is far wider than the fixed ±gap_half wedge.  Clearing
        # that whole gap of pith seeds while the ray fill reaches only the wedge would
        # leave an empty band the neighbouring Voronoi cells balloon into — so the
        # pith is left in place there (as it grades into the interfascicular rays).
        if sectors and gap_half > 0.0:
            wedges = unary_union([sg.angular_wedge(cx, cy, gt, gap_half, r_outer_wedge)
                                  for gt in gap_thetas])
            ray_zone = annulus.intersection(wedges).difference(unary_union(sectors))
            if not ray_zone.is_empty:
                self.vascular_tissue_polygons.setdefault("parenchyma", []).append(ray_zone)

        # --- medullar rays --------------------------------------------------
        for poly, theta_c in medullar_polys:
            next_id = sg.fill_medullar_rays(self.vascular_cells, poly, theta_c,
                                            cx, cy, mr, next_id)
        if medullar_union is not None and not medullar_union.is_empty:
            self.vascular_tissue_polygons.setdefault("medullar_ray", []).append(medullar_union)

        # --- secondary phloem ----------------------------------------------
        self._build_secondary_phloem(secondary_contour, angles, half, cx, cy)

        # --- displaced primary phloem (pushed outside the secondary phloem) -
        self._build_primary_phloem(secondary_contour, angles, bp, xylem, phloem,
                                   cambium, cx, cy, r_outer_wedge)

    def _primary_tissue_bp(self, bp: dict, role: str, thickness: float) -> dict:
        """A copy of the ``vascular_bundle`` params reduced to a single-tissue banded
        bundle — xylem-only or phloem-only, no cambium, no sheath — of radial
        ``thickness``.

        Lets :func:`~openalea.granap.vascular_bundle.build_bundle` lay just the
        preserved primary xylem or the displaced primary phloem remnant under
        secondary growth, reusing the same conducting-cell fills as a primary bundle.
        The bundle sheath (for the primary xylem) is laid separately as a *partial*
        cup by :meth:`_build_primary_xylem`, so it is disabled here."""
        out = dict(bp)
        out["bundle_type"] = "collateral"
        out["has_cambium"] = False
        out["inner_phloem_fraction"] = 0.0
        out["outer_sheath"] = False
        out["sheath"] = "none"
        out["sheath_thickness"] = 0.0
        out["height"] = float(thickness)
        out["xylem_fraction"] = 1.0 if role == "xylem" else 0.0
        out["phloem_fraction"] = 1.0 if role == "phloem" else 0.0
        return out

    def _place_primary_remnant(self, contour: Polygon, angles, remnant_bp: dict,
                               xylem: dict, phloem: dict, cambium: dict,
                               cx: float, cy: float, r_far: float,
                               anchor: float) -> list:
        """Build one single-tissue primary remnant bundle at each ``angle`` on
        ``contour`` and return their :class:`BundleResult`\\ s.

        A ray from the centre at each angle meets ``contour`` at ``P``; the bundle is
        placed there oriented along the local outward normal and shifted by ``anchor``
        so the requested edge (outer for xylem, inner for phloem) lands on ``P``.
        """
        results = []
        for th in angles:
            frame = sg.cambium_local_frame(contour.exterior, cx, cy, th, r_far)
            if frame is None:
                continue
            (px, py), _tangent, (nx, ny) = frame
            theta = float(np.arctan2(ny, nx))
            res = build_bundle(self.vascular_cells, self.rng, px, py, theta,
                               remnant_bp, xylem, phloem, cambium,
                               ground_cell_size=None, anchor=anchor,
                               fill_cambium=False)
            self._register_bundle(res)
            results.append(res)
        return results

    def _build_primary_xylem(self, primary_contour: Polygon, angles, bp: dict,
                             xylem: dict, phloem: dict, cambium: dict,
                             cx: float, cy: float, r_far: float) -> None:
        """Lay the preserved primary xylem as a thin remnant pinned at the pith edge,
        one per bundle position, then cup each with a **partial bundle sheath**.

        The primary xylem is endarch and immovable, so it stays put while the
        secondary xylem is built radially on top of it.  A xylem-only remnant bundle
        (:meth:`_primary_tissue_bp`) is anchored so its **outer** edge sits on the
        primary ring — i.e. it occupies the outer pith, flush against where the
        secondary-xylem annulus begins.  A single file of ``bundle sheath`` cells is
        then laid along the **pith-facing** part of each envelope only (the inner
        half, radius below the envelope centre), leaving the outer side open so the
        primary xylem joins straight onto the secondary xylem instead of being
        wrapped all the way round.
        """
        h_px = float(bp.get("height", 0.0)) * float(bp.get("xylem_fraction", 0.5))
        if h_px <= 0.0:
            return
        px_bp = self._primary_tissue_bp(bp, "xylem", h_px)
        results = self._place_primary_remnant(primary_contour, angles, px_bp, xylem,
                                              phloem, cambium, cx, cy, r_far,
                                              anchor=h_px / 2.0)
        # Partial sheath: a file of cells hugging each envelope's inner (pith-facing)
        # arc, cut off at the envelope centre radius so it never crosses onto the
        # outer (secondary-xylem) side.
        scd = float(bp.get("parenchyma_diameter", 0.012))
        scw = float(bp.get("parenchyma_width", 0.012))
        for res in results:
            env = res.envelope
            if env is None or env.is_empty:
                continue
            r_center = np.hypot(env.centroid.x - cx, env.centroid.y - cy)
            inner_disc = Point(cx, cy).buffer(max(r_center, 0.0))
            inner_arc = env.exterior.intersection(inner_disc)
            geoms = inner_arc.geoms if hasattr(inner_arc, "geoms") else [inner_arc]
            for line in geoms:
                if line.geom_type in ("LineString", "LinearRing") and line.length > 0:
                    fill_along(self.vascular_cells, line, "bundle sheath",
                               scd, scw, cx, cy)

    def _build_primary_phloem(self, secondary_contour: Polygon, angles, bp: dict,
                              xylem: dict, phloem: dict, cambium: dict,
                              cx: float, cy: float, r_far: float) -> None:
        """Lay the displaced primary phloem as a thin remnant pushed just outside the
        secondary phloem band, one per bundle position.

        The transition crushes the primary phloem outward ahead of the new secondary
        phloem.  A phloem-only remnant bundle (:meth:`_primary_tissue_bp`) is anchored
        so its **inner** edge sits on the outer edge of the secondary phloem band
        (the secondary cambium contour buffered out by ``secondary_phloem.height``).
        """
        h_pp = self._primary_phloem_thickness()
        if h_pp <= 0.0:
            return
        sp = self._get_param("secondary_phloem")
        sp_height = float(sp.get("height", 0.0)) if sp else 0.0
        contour = secondary_contour.buffer(sp_height) if sp_height > 0.0 else secondary_contour
        pp_bp = self._primary_tissue_bp(bp, "phloem", h_pp)
        self._place_primary_remnant(contour, angles, pp_bp, xylem, phloem,
                                    cambium, cx, cy, r_far, anchor=-h_pp / 2.0)

    def _build_secondary_phloem(self, secondary_contour: Polygon, angles, half: float,
                                cx: float, cy: float) -> None:
        """Root-style secondary phloem: a band just outside the secondary cambium,
        carved into one tapering trapeze per bundle and split radially into an alive
        (sieve + companion + parenchyma) and a dead (sieve + parenchyma) sub-zone."""
        sp = self._get_param("secondary_phloem")
        if not sp:
            return
        band = secondary_contour.buffer(float(sp["height"])).difference(secondary_contour)
        if band.is_empty:
            return
        minx, miny, maxx, maxy = band.bounds
        r_outer = max(maxx - cx, cx - minx, maxy - cy, cy - miny) * 1.5
        cam_ext = secondary_contour.exterior

        masks = []
        for th in angles:
            frame = sg.cambium_local_frame(cam_ext, cx, cy, th, r_outer)
            if frame is None:
                continue
            P, tangent, normal = frame
            r_P = np.hypot(P[0] - cx, P[1] - cy)
            base_hw = half * r_P
            top_w = min(float(sp["top_width"]), 2.0 * base_hw)
            masks.append(sg.phloem_trapeze_curved(
                cam_ext, cx, cy, P, tangent, normal, base_hw, top_w, float(sp["height"])))
        arms = band.intersection(unary_union(masks)) if masks else band
        if arms.is_empty:
            return
        self.vascular_tissue_polygons.setdefault("phloem", []).append(arms)

        alive_annulus = secondary_contour.buffer(float(sp["alive_distance"]))
        alive_zone = arms.intersection(alive_annulus)
        dead_zone = arms.difference(alive_annulus)
        next_id = self.vascular_cells.next_group_id()
        next_id = sg.fill_phloem_zone(self.vascular_cells, self.rng, alive_zone, True, cx, cy, sp, next_id)
        next_id = sg.fill_phloem_zone(self.vascular_cells, self.rng, dead_zone, False, cx, cy, sp, next_id)
