"""Monocot stem anatomy (atactostele).

``MonocotStemAnatomy`` builds a monocot stem: an epidermis and a narrow cortex
around a large ground tissue (``pith``) in which the vascular bundles are
*scattered* — no organised ring, no cambium.  Each bundle is collateral
(metaxylem + protoxylem on the inner face, phloem on the outer).  Instantiate
via ``StemAnatomy(input_data)`` — the factory in
:mod:`openalea.granap.stem_class` dispatches here when ``planttype == 1``.

Scaffold status: parameter parsing, layer setup and the vascular *hook* are in
place; the scattered-bundle placement itself (``_build_scattered_bundles``) is a
documented stub, so ``generate_cells()`` currently renders the ground tissue +
cortex + epidermis without vascular cells.
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

        SCAFFOLD: the single ``scattered vascular bundles`` step currently calls
        a stub (:meth:`_build_scattered_bundles`) that places no cells yet.
        """
        recipe = TissueRecipe().bind(lambda: self.vascular_cells, self.rng)
        bp = self._get_param("vascular_bundle")
        if not bp or int(bp.get("n_bundles", 0)) == 0:
            return recipe                       # no bundles -> empty
        recipe.special(
            "scattered vascular bundles",
            lambda: self._build_scattered_bundles(polygon),
            produces=("metaxylem", "protoxylem", "phloem", "sclerenchyma"),
        )
        return recipe

    def _scattered_bundle_positions(self, polygon: Polygon) -> List[Tuple[float, float, float]]:
        """Sample bundle centres across the ground tissue (atactostele).

        Rejection sampling inside ``polygon`` with a min-distance rule, biased
        toward the periphery (denser bundles near the surface, as in a real
        monocot stem), and staying clear of the medullary cavity when the pith is
        hollow.  ``theta`` is each centre's radial angle (phloem faces outward).
        """
        bp = self._get_param("vascular_bundle")
        n = int(bp.get("n_bundles", 0))
        if n <= 0 or polygon.is_empty:
            return []
        cx0, cy0 = polygon.centroid.x, polygon.centroid.y
        R = np.sqrt(polygon.area / np.pi)
        min_dist = 0.8 * max(bp.get("width", 0.09), bp.get("height", 0.13))
        cavity_keepout = self.cavity_radius + 0.5 * bp.get("height", 0.13)
        prepared = prep(polygon)

        placed: List[Tuple[float, float, float]] = []
        for _ in range(n * 400):
            if len(placed) >= n:
                break
            # Peripheral bias: exponent < 1 pushes the radius toward R.
            r = R * self.rng.uniform() ** 0.35
            ang = self.rng.uniform(0.0, 2.0 * np.pi)
            x, y = cx0 + r * np.cos(ang), cy0 + r * np.sin(ang)
            if np.hypot(x - cx0, y - cy0) < cavity_keepout:
                continue
            if not prepared.contains(Point(x, y)):
                continue
            if any(np.hypot(x - px, y - py) < min_dist for px, py, _ in placed):
                continue
            placed.append((x, y, np.arctan2(y - cy0, x - cx0)))

        if len(placed) < n:
            log.info("MonocotStemAnatomy: placed %d/%d scattered bundles "
                     "(ground tissue too tight for the rest).", len(placed), n)
        return placed

    def _build_scattered_bundles(self, polygon: Polygon) -> None:
        """Build the atactostele: one collateral 'face' bundle per sampled centre.

        Each bundle's envelope is registered in ``vascular_tissue_polygons`` so
        ``generate_cells`` clears the ground-tissue seeds underneath it; the
        bundle's own cells were appended to ``self.vascular_cells`` by
        ``build_bundle``.
        """
        bp = self._get_param("vascular_bundle")
        xylem = self._get_param("xylem")
        phloem = self._get_param("phloem")
        cambium = self._get_param("cambium")
        for cx, cy, theta in self._scattered_bundle_positions(polygon):
            res = build_bundle(self.vascular_cells, self.rng, cx, cy, theta,
                               bp, xylem, phloem, cambium)
            self._register_bundle(res)
