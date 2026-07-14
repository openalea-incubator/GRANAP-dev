"""Dicot stem anatomy (eustele).

``DicotStemAnatomy`` builds a dicot stem: a central pith ringed by a single ring
of discrete *collateral* vascular bundles — xylem on the inner (pith) face,
phloem on the outer (cortex) face, with a strip of fascicular cambium between —
then a cortex and epidermis outside.  Instantiate via ``StemAnatomy(input_data)``
— the factory in :mod:`openalea.granap.stem_class` dispatches here when
``planttype == 2``.

Scaffold status: parameter parsing, layer setup and the vascular *hook* are in
place; the bundle-ring placement itself (``_build_bundle_ring``) is a documented
stub, so ``generate_cells()`` currently renders the pith + cortex + epidermis
without vascular cells.
"""

import logging
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

    def _parse_vascular_params(self) -> None:
        xylem = self._get_param("xylem")
        phloem = self._get_param("phloem")
        cambium = self._get_param("cambium")

        self.vascular_params.update({
            # Number of bundles evenly spaced around the eustele ring.
            "n_bundles":              int(xylem.get("n_vascular_peak",          8)),
            # Xylem vessels of one bundle (centre-to-tip size gradient).
            "xylem_diameter_max":     float(xylem.get("vessel_diameter",        0.08)),
            "xylem_diameter_min":     float(xylem.get("vessel_diameter_min",    0.02)),
            "xylem_diameter_sd":      float(xylem.get("vessel_diameter_sd",     0.002)),
            # Phloem cap of one bundle.
            "phloem_diameter":        float(phloem.get("sieve_diameter",        0.012)),
            "phloem_diameter_sd":     float(phloem.get("sieve_diameter_sd",     0.001)),
            "phloem_width":           float(phloem.get("cluster_width",         0.1)),
            "phloem_height":          float(phloem.get("cluster_height",        0.05)),
            # Fascicular cambium strip between xylem and phloem.
            "cambium_cell_diameter":  float(cambium.get("cell_diameter",        0.01)),
            "cambium_cell_width":     float(cambium.get("cell_width",           0.02)),
        })

        # Primary phloem / cambium are built only when their param entries are
        # present (opt-out).
        self.has_primary_phloem = bool(phloem)
        self.has_cambium = bool(cambium)

    # ------------------------------------------------------------------
    # Vascular tissue
    # ------------------------------------------------------------------

    def _vascular_recipe(self, polygon: Polygon) -> TissueRecipe:
        """Declarative description of how the eustele bundle ring is assembled.

        Built and run by the shared ``Organ._create_vascular_tissue`` scaffold;
        the remove-mask + extend step runs later in ``Organ.generate_cells``.
        The build order is data, inspectable via ``recipe.describe()``.

        SCAFFOLD: the steps below currently call a stub
        (:meth:`_build_bundle_ring`) that places no cells yet.
        """
        recipe = TissueRecipe().bind(lambda: self.vascular_cells, self.rng)
        if self.vascular_params.get("n_bundles", 0) == 0:
            return recipe                       # no bundles -> empty
        recipe.special(
            "collateral bundle ring",
            lambda: self._build_bundle_ring(polygon),
            produces=("xylem", "cambium", "phloem"),
        )
        return recipe

    def _bundle_ring_positions(self, polygon: Polygon) -> List[Tuple[float, float, float]]:
        """Evenly spaced ``(cx, cy, theta)`` slots on the pith/cortex boundary.

        Bundles straddle the ring so their inner (xylem) half sits in the pith and
        their outer (phloem) half toward the cortex.  ``theta`` is each slot's
        polar angle (radial orientation).
        """
        n = int(self.vascular_params.get("n_bundles", 0))
        if n <= 0:
            return []
        cx0, cy0 = polygon.centroid.x, polygon.centroid.y
        r_ring = np.sqrt(polygon.area / np.pi)     # outer pith radius
        out = []
        for k in range(n):
            theta = 2.0 * np.pi * k / n
            out.append((cx0 + r_ring * np.cos(theta), cy0 + r_ring * np.sin(theta), theta))
        return out

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
        for cx, cy, theta in self._bundle_ring_positions(polygon):
            res = build_bundle(self.vascular_cells, self.rng, cx, cy, theta,
                               bp, xylem, phloem, cambium)
            self._register_bundle(res)
