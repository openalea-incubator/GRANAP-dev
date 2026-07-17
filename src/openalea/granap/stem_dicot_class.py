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
            px_pt, py_pt = pts[i - 1]
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
        bp = self._get_param("vascular_bundle")
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

    def _build_bundle_ring(self, polygon: Polygon) -> None:
        """Build the eustele: one collateral bundle per contour slot, then — under
        secondary growth — close the fascicular cambia into a continuous ring.

        Each bundle is placed on the cambium-ring contour anchored on its cambium
        (``bundle_cambium_anchor``), so the fascicular cambia all sit on one line;
        its envelope is registered in ``vascular_tissue_polygons`` so
        ``generate_cells`` clears the pith/cortex seeds underneath it, and its own
        cells were appended to ``self.vascular_cells`` by ``build_bundle``.

        Primary growth stops there — cambium is visible only inside each bundle.
        With ``secondary_growth`` on, :meth:`_build_cambium_ring` fills the
        interfascicular gaps along the contour, so the cambium reads as one ring.
        """
        bp = self._get_param("vascular_bundle")
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

        envelopes = []
        for cx, cy, theta in slots:
            ground = self._pith_cell_diameter_at(np.hypot(cx - cx0, cy - cy0), r_pith)
            res = build_bundle(self.vascular_cells, self.rng, cx, cy, theta,
                               bp, xylem, phloem, cambium,
                               ground_cell_size=ground, anchor=anchor)
            self._register_bundle(res)
            if res.envelope is not None and not res.envelope.is_empty:
                envelopes.append(res.envelope)

        if self._secondary_growth():
            self._build_cambium_ring(contour, envelopes, bp, cambium)

    def _build_cambium_ring(self, contour: Polygon, envelopes: List[Polygon],
                            bp: dict, cambium: dict) -> None:
        """Close the vascular cambium into a continuous ring (secondary growth).

        The bundles already carry their fascicular cambium on the contour; here we
        add the *interfascicular* cambium in the gaps between them, so the whole
        contour reads as one meristematic ring.  ``n_layers`` concentric cambium
        files are laid along the contour (offset in/out by a cell diameter each),
        skipping the spans already occupied by a bundle.  The band region is
        registered as ``cambium`` so the pith/cortex seeds beneath it are cleared
        and the tissue view draws the full ring.
        """
        cx, cy = contour.centroid.x, contour.centroid.y
        n_layers = max(int(cambium.get("n_layers", 2)), 1)
        cell_d = float(cambium.get("cell_diameter", 0.01))
        cell_w = float(cambium.get("cell_width", cell_d))
        skip = unary_union(envelopes) if envelopes else None

        for k in range(n_layers):
            off = (k - (n_layers - 1) / 2.0) * cell_d
            ring = contour.buffer(off) if off else contour
            if ring.is_empty:
                continue
            fill_along(self.vascular_cells, ring.exterior, "cambium",
                       cell_d, cell_w, cx, cy, xylem_union=skip)

        # Register the whole annular band (minus the bundles) as cambium: it clears
        # the ground seeds under the interfascicular arcs and completes the ring in
        # the tissue view (the fascicular strips are registered by the bundles).
        half = max(n_layers * cell_d / 2.0, cell_d / 2.0)
        band = contour.buffer(half).difference(contour.buffer(-half))
        if skip is not None:
            band = band.difference(skip)
        if not band.is_empty:
            self.vascular_tissue_polygons.setdefault("cambium", []).append(band)
