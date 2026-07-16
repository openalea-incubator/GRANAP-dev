"""Monocot stem anatomy (atactostele).

``MonocotStemAnatomy`` builds a monocot stem: an epidermis and a narrow cortex
around a large ground tissue (``pith``) in which the vascular bundles are
*scattered* — no organised ring, no cambium.  Each bundle is a collateral
'face' bundle: xylem (metaxylem at the middle, protoxylem toward the centre)
with the phloem toward the surface.  Instantiate
via ``StemAnatomy(input_data)`` — the factory in
:mod:`openalea.granap.stem_class` dispatches here when ``planttype == 1``.

Bundle placement (``_scattered_bundle_positions``) supports more than one
``vascular_bundle`` spec, each owning a radial band (``radius_min`` ..
``radius_max`` mm) so the bundle *kind* can vary with radius.  Within its band a
spec is laid out one of two ways (its ``placement`` field):

* ``random`` — bundles scattered by rejection sampling, kept clear of the
  medullary cavity and min-distance spaced, with each candidate's oriented
  envelope required to stay clear of every bundle already placed (no clumping,
  no overlap);
* ``even`` — bundles equally spaced on a ring at the band midpoint radius
  (circumference / ``n_bundles`` apart), rotated by the spec's ``angle`` phase so
  two close-radius bands can interleave.
"""

import logging
from typing import List, Tuple

import numpy as np
from shapely.geometry import Polygon, Point
from shapely.prepared import prep

from openalea.granap.tissue_class import TissueRecipe
from openalea.granap.stem_class import StemAnatomy
from openalea.granap.vascular_bundle import build_bundle

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Monocot stem subclass
# ---------------------------------------------------------------------------

class MonocotStemAnatomy(StemAnatomy):
    """Monocot stem: vascular bundles scattered through the ground tissue
    (atactostele), collateral, with no cambium."""

    # ------------------------------------------------------------------
    # Vascular tissue
    # ------------------------------------------------------------------
    #
    # No _parse_vascular_params override: build_bundle reads the raw xylem /
    # phloem / cambium param dicts directly, and the bundle count is the
    # vascular_bundle.n_bundles field — so there is nothing to pre-parse.

    def _vascular_recipe(self, polygon: Polygon) -> TissueRecipe:
        """Declarative description of how the scattered bundles are assembled.

        Built and run by the shared ``Organ._create_vascular_tissue`` scaffold;
        the remove-mask + extend step runs later in ``Organ.generate_cells``.
        The build order is data, inspectable via ``recipe.describe()``.

        The single ``scattered vascular bundles`` step defers to
        :meth:`_build_scattered_bundles`, which builds one 'face' bundle per
        sampled centre.
        """
        recipe = TissueRecipe().bind(lambda: self.vascular_cells, self.rng)
        if not any(int(s.get("n_bundles", 0)) > 0 for s in self._bundle_specs()):
            return recipe                       # no bundles -> empty
        recipe.special(
            "scattered vascular bundles",
            lambda: self._build_scattered_bundles(polygon),
            produces=("xylem", "phloem", "companion cell", "sclerenchyma", "air space"),
        )
        return recipe

    def _scattered_bundle_positions(
        self, polygon: Polygon
    ) -> List[Tuple[float, float, float, dict]]:
        """Place every band's bundles across the ground tissue (atactostele).

        Each ``vascular_bundle`` spec owns a radial band and its own ``n_bundles``;
        the specs are laid out inner-band-first (``_bundle_specs``) so a placed
        bundle's envelope is tested against all bands already placed.  Returns
        ``(x, y, theta, bp)`` per bundle: ``theta`` the radial angle (phloem faces
        outward) and ``bp`` the spec that governs that bundle's build.
        """
        specs = [s for s in self._bundle_specs() if int(s.get("n_bundles", 0)) > 0]
        if not specs or polygon.is_empty:
            return []

        cx0, cy0 = polygon.centroid.x, polygon.centroid.y
        R = np.sqrt(polygon.area / np.pi)
        prepared = prep(polygon)

        placed: List[Tuple[float, float, float, dict]] = []
        envelopes: List[Polygon] = []
        for bp in specs:
            if bp.get("placement", "random") == "even":
                self._place_even_bundles(bp, cx0, cy0, R, prepared, placed, envelopes)
            else:
                self._place_random_bundles(bp, cx0, cy0, R, prepared, placed, envelopes)
        return placed

    def _place_random_bundles(self, bp, cx0, cy0, R, prepared, placed, envelopes) -> None:
        """Scatter one band's bundles by rejection sampling (no clumping/overlap).

        A full-span band (no ``radius_min``/``radius_max`` set) keeps the historical
        peripheral-bias radial sampling so the default single-kind atactostele is
        unchanged; an explicit sub-band samples area-uniform within its annulus.
        A candidate is rejected unless it is inside the ground tissue, clear of the
        medullary cavity, at least ``min_dist`` from every placed centre, and its
        oriented envelope stays clear of every placed envelope (across all bands).
        """
        n = int(bp.get("n_bundles", 0))
        r_lo, r_hi = self._spec_band(bp, R)
        full_span = (float(bp.get("radius_min", 0.0)) <= 0.0
                     and float(bp.get("radius_max", 0.0)) <= 0.0)
        # Cheap centre pre-filter (skips obvious clashes before the exact test);
        # the envelope-intersection test below is what actually forbids overlap.
        min_dist = 0.8 * max(bp.get("width", 0.09), bp.get("height", 0.13))
        gap = self._bundle_clearance(bp)
        cavity_keepout = self.cavity_radius + 0.5 * bp.get("height", 0.13)

        n_here = 0
        for _ in range(n * 400):
            if n_here >= n:
                break
            if full_span:
                # Peripheral bias: exponent < 1 pushes the radius toward R.
                r = R * self.rng.uniform() ** 0.35
            else:
                # Area-uniform within the annulus (even spatial density in the band).
                u = self.rng.uniform()
                r = np.sqrt(u * (r_hi ** 2 - r_lo ** 2) + r_lo ** 2)
            ang = self.rng.uniform(0.0, 2.0 * np.pi)
            x, y = cx0 + r * np.cos(ang), cy0 + r * np.sin(ang)
            if r < r_lo or r >= r_hi:
                continue
            if np.hypot(x - cx0, y - cy0) < cavity_keepout:
                continue
            if not prepared.contains(Point(x, y)):
                continue
            if any(np.hypot(x - px, y - py) < min_dist for px, py, *_ in placed):
                continue
            theta = np.arctan2(y - cy0, x - cx0)
            # Definitive non-overlap test: reject unless this bundle's actual
            # oriented envelope stays clear of every one already placed.
            env = self._placed_bundle_envelope(x, y, theta, bp)
            if self._bundle_overlaps(env, envelopes, gap):
                continue
            placed.append((x, y, theta, bp))
            envelopes.append(env)
            n_here += 1

        if n_here < n:
            log.info("MonocotStemAnatomy: placed %d/%d random bundles in band "
                     "[%.3g, %.3g] mm (ground tissue too tight for the rest).",
                     n_here, n, r_lo, r_hi)

    def _place_even_bundles(self, bp, cx0, cy0, R, prepared, placed, envelopes) -> None:
        """Place one band's bundles equally spaced on a ring at the band midpoint.

        The ring holds ``n_bundles`` slots a full turn apart (circumference /
        ``n_bundles``), rotated by the spec's ``angle`` phase (degrees) — set two
        close-radius bands half a step apart to interleave them.  A slot is skipped
        (with a log) if it falls outside the ground tissue or its envelope would
        overlap a bundle already placed, so ``even`` never produces overlap either.
        """
        n = int(bp.get("n_bundles", 0))
        r_lo, r_hi = self._spec_band(bp, R)
        r_ring = 0.5 * (r_lo + r_hi)             # band midpoint (full-span -> R/2)
        gap = self._bundle_clearance(bp)
        angle0 = np.radians(float(bp.get("angle", 0.0)))

        n_here = 0
        for k in range(n):
            theta = angle0 + 2.0 * np.pi * k / n
            x, y = cx0 + r_ring * np.cos(theta), cy0 + r_ring * np.sin(theta)
            if not prepared.contains(Point(x, y)):
                continue
            env = self._placed_bundle_envelope(x, y, theta, bp)
            if self._bundle_overlaps(env, envelopes, gap):
                continue
            placed.append((x, y, theta, bp))
            envelopes.append(env)
            n_here += 1

        if n_here < n:
            log.info("MonocotStemAnatomy: placed %d/%d even bundles on the "
                     "r=%.3g mm ring (some slots fell outside the ground tissue "
                     "or overlapped another band).", n_here, n, r_ring)

    def _build_scattered_bundles(self, polygon: Polygon) -> None:
        """Build the atactostele: one collateral 'face' bundle per sampled centre.

        Each bundle is built with the ``bp`` spec that governs its band, so bundle
        kind varies with radius.  Its envelope is registered in
        ``vascular_tissue_polygons`` so ``generate_cells`` clears the ground-tissue
        seeds underneath it; the bundle's own cells were appended to
        ``self.vascular_cells`` by ``build_bundle``.
        """
        xylem = self._get_param("xylem")
        phloem = self._get_param("phloem")
        cambium = self._get_param("cambium")
        for cx, cy, theta, bp in self._scattered_bundle_positions(polygon):
            res = build_bundle(self.vascular_cells, self.rng, cx, cy, theta,
                               bp, xylem, phloem, cambium)
            self._register_bundle(res)
