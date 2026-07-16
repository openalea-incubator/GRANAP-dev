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
from shapely.geometry import Polygon

from openalea.granap.tissue_class import TissueRecipe
from openalea.granap.stem_class import StemAnatomy
from openalea.granap.vascular_bundle import build_bundle

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

    def _bundle_ring_positions(self, polygon: Polygon) -> List[Tuple[float, float, float]]:
        """Evenly spaced ``(cx, cy, theta)`` slots on the pith/cortex boundary.

        Bundles straddle the ring so their inner (xylem) half sits in the pith and
        their outer (phloem) half toward the cortex.  ``theta`` is each slot's
        polar angle (radial orientation).

        The requested count is clamped to the number that actually fit around the
        ring without their envelopes touching (a warning is issued if it had to be
        reduced): reducing the count keeps the ring evenly spaced and symmetric,
        which dropping individual slots would not.
        """
        bp = self._get_param("vascular_bundle")
        n = int(bp.get("n_bundles", 0))
        if n <= 0:
            return []
        cx0, cy0 = polygon.centroid.x, polygon.centroid.y
        r_ring = np.sqrt(polygon.area / np.pi)     # outer pith radius

        n_fit = self._max_ring_bundles(cx0, cy0, r_ring, bp, n)
        if n_fit < n:
            warnings.warn(
                f"DicotStemAnatomy: {n} bundles overlap on the ring; placing "
                f"{n_fit} evenly-spaced non-overlapping bundles instead "
                f"(reduce vascular_bundle.width/height or n_bundles to fit more).",
                stacklevel=2,
            )
            n = n_fit

        out = []
        for k in range(n):
            theta = 2.0 * np.pi * k / n
            out.append((cx0 + r_ring * np.cos(theta), cy0 + r_ring * np.sin(theta), theta))
        return out

    def _max_ring_bundles(self, cx0: float, cy0: float, r_ring: float,
                          bp: dict, n_req: int) -> int:
        """Largest bundle count (<= ``n_req``) whose adjacent envelopes stay clear.

        On an evenly-spaced ring every adjacent pair is congruent, so testing one
        pair (slots 0 and 1) settles the whole ring.
        """
        gap = self._bundle_clearance(bp)
        # Bundles ring the pith edge, so their outer sheath is sized against the
        # pith-edge cell (r_norm = 1); include it in the clearance footprint.
        ground = self._pith_cell_diameter_at(r_ring, r_ring)
        for n in range(n_req, 1, -1):
            env0 = self._placed_bundle_envelope(
                cx0 + r_ring, cy0, 0.0, bp, ground)
            theta1 = 2.0 * np.pi / n
            env1 = self._placed_bundle_envelope(
                cx0 + r_ring * np.cos(theta1), cy0 + r_ring * np.sin(theta1), theta1, bp, ground)
            if not self._bundle_overlaps(env0, [env1], gap):
                return n
        return 1

    def _build_bundle_ring(self, polygon: Polygon) -> None:
        """Build the eustele: one collateral bundle per ring slot.

        Each bundle's envelope is registered in ``vascular_tissue_polygons`` so
        ``generate_cells`` clears the pith/cortex seeds underneath it; the bundle's
        own cells were appended to ``self.vascular_cells`` by ``build_bundle``.

        (Secondary growth — an interfascicular cambium closing the ring into
        continuous cylinders — is a later extension, mirroring the dicot-root
        secondary path.)
        """
        bp = self._get_param("vascular_bundle")
        xylem = self._get_param("xylem")
        phloem = self._get_param("phloem")
        cambium = self._get_param("cambium")
        if not bp:
            return
        cx0, cy0 = polygon.centroid.x, polygon.centroid.y
        r_pith = np.sqrt(polygon.area / np.pi)
        for cx, cy, theta in self._bundle_ring_positions(polygon):
            ground = self._pith_cell_diameter_at(np.hypot(cx - cx0, cy - cy0), r_pith)
            res = build_bundle(self.vascular_cells, self.rng, cx, cy, theta,
                               bp, xylem, phloem, cambium, ground_cell_size=ground)
            self._register_bundle(res)
