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
from shapely.geometry import Polygon
from shapely.ops import unary_union

from openalea.granap.geometry_collection import GeometryProcessor
from openalea.granap.tissue_class import TissueRecipe, fill_along
from openalea.granap.stem_class import StemAnatomy
from openalea.granap.vascular_bundle import build_bundle, bundle_cambium_anchor

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
        """True when secondary growth is requested (continuous cambium ring)."""
        sg = self._get_param("secondary_growth")
        return bool(sg.get("value", False)) if sg else False

    def _cambium_ring_contour(self, polygon: Polygon) -> Polygon:
        """The drawn contour the eustele bundles are placed on — the cambium ring.

        A circle at the pith/cortex boundary by default; an ``ellipse`` (flattened
        by ``ring_ellipse_ratio``) or a lobed ``star`` (``ring_star_branches`` arms
        of depth ``ring_star_amplitude``) when requested.  Each bundle sits on this
        contour with its cambium on the line, so under secondary growth the
        fascicular cambia join into one continuous ring along it.
        """
        bp = self._get_param("vascular_bundle")
        cx, cy = polygon.centroid.x, polygon.centroid.y
        r = np.sqrt(polygon.area / np.pi)
        shape = bp.get("ring_shape", "circle")

        if shape == "ellipse":
            ratio = float(bp.get("ring_ellipse_ratio", 0.75))
            return GeometryProcessor.ellipse_to_polygon(cx, cy, r, r * ratio, 0.0)
        if shape == "star":
            n = max(int(bp.get("ring_star_branches", 5)), 2)
            amp = min(max(float(bp.get("ring_star_amplitude", 0.12)), 0.0), 0.9)
            r_min = r * (1.0 - amp)
            # Arms about half the inter-arm pitch wide at the base, tapering to a
            # rounded tip; the exact widths only shape the lobes, not the count.
            star = GeometryProcessor.star_polygon(
                n_branches=n, r_min=r_min, r_max=r,
                arc_base=0.5 * np.pi * r_min / n,
                arc_top=0.35 * np.pi * r / n,
            )
            return translate(star, cx, cy)
        return translate(GeometryProcessor.circle_polygon(r), cx, cy)

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
        cx0, cy0 = polygon.centroid.x, polygon.centroid.y
        r_pith = np.sqrt(polygon.area / np.pi)
        anchor = bundle_cambium_anchor(bp)
        contour = self._cambium_ring_contour(polygon)
        slots = self._ring_slots(polygon)

        conducting: List[Polygon] = []      # xylem / phloem zones — the ring avoids these
        fascicular: List[Polygon] = []      # per-bundle cambium zones — the primary clip region
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

        self._build_cambium(contour, fascicular, conducting, cambium,
                            secondary=self._secondary_growth())

    def _build_cambium(self, contour: Polygon, fascicular: List[Polygon],
                       conducting: List[Polygon], cambium: dict,
                       secondary: bool) -> None:
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
            # points on the strip edge are not dropped by the point-in test).
            keep_union = unary_union(fascicular).buffer(0.25 * cell_d)
            register = None      # the fascicular zones are registered per bundle

        for k in range(n_layers):
            off = (k - (n_layers - 1) / 2.0) * cell_d
            ring = contour.buffer(off) if off else contour
            if ring.is_empty:
                continue
            fill_along(self.vascular_cells, ring.exterior, "cambium",
                       cell_d, cell_w, cx, cy,
                       xylem_union=conducting_union, keep_union=keep_union)

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
