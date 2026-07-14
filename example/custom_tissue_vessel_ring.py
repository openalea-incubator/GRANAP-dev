"""Worked example — designing a NON-TRIVIAL custom tissue.

Goal: replace a monocot root's vascular tissue with a **packed vessel ring**:
a central pith left free of vessels, surrounded by an annulus that is filled by
circle-packing with a centre→edge size gradient (big vessels inside, small
outside). This is the canonical "build a region, fill it, record its mask"
pattern that a new organ's ``_vascular_recipe`` follows.

It demonstrates every idea from the tutorial's "designing a new tissue" section:

  1. Build the region with ``GeometryProcessor`` + ``Tissue`` boolean algebra
     (an annulus = outer disc MINUS inner pith disc).
  2. Fill it declaratively with ``recipe.fill(..., strategy="packing", ...)``.
  3. **Record** the placed vessel polygons into ``self.vascular_polygons`` so the
     shared pipeline's unified mask removes the layer seeds sitting underneath —
     you never remove them yourself.
  4. Preview the zones with ``plot_tissues`` (no Voronoi), then the real cells
     with ``plot_cells``.

Run:  ``python custom_tissue_vessel_ring.py``   (from the example/ directory)
"""

import os
import sys

import matplotlib.pyplot as plt
from shapely.geometry import Point

sys.path.append(os.path.abspath(".."))

from openalea.granap.root_monocot_class import MonocotRootAnatomy
from openalea.granap.input_data import OrganInputData
from openalea.granap.tissue_class import Tissue, TissueRecipe

SEED = 0

# Geometry of the custom vascular zone (fractions of the stele polygon radius).
PITH_FRACTION = 0.30    # inner pith left free of vessels
VESSEL_MAX_D = 0.06     # largest vessel diameter (at the ring's inner edge)
VESSEL_MIN_D = 0.012    # smallest vessel diameter (at the outer edge)
FILL_PROPORTION = 0.85  # stop packing at 85% area coverage


class VesselRingRoot(MonocotRootAnatomy):
    """A monocot root whose stele is a packed vessel ring around a clear pith.

    Only ``_vascular_recipe`` changes; everything else (layers, base shape,
    Voronoi, intercellular spaces) is inherited unchanged. The base
    ``Organ._create_vascular_tissue`` builds and runs this recipe, and
    ``Organ.generate_cells`` applies the unified vascular mask afterwards.
    """

    def _vascular_recipe(self, polygon) -> TissueRecipe:
        # Bind the recipe to the organ's vascular CellManager (a callable, so it
        # always resolves the *current* manager) and its seeded rng.
        recipe = TissueRecipe().bind(lambda: self.vascular_cells, self.rng)

        # (1) Region algebra — annulus = stele disc  MINUS  central pith disc.
        centre = polygon.centroid
        radius = (polygon.area / 3.141592653589793) ** 0.5
        pith = Point(centre.x, centre.y).buffer(radius * PITH_FRACTION)
        ring = Tissue("xylem", polygon).difference(pith).smooth(0.2)

        # (2)+(3) Fill the ring by packing, and record each placed vessel polygon
        # into vascular_polygons for the unified mask. ``result`` is the list of
        # (polygon, tag, id_group) tuples returned by the packing fill.
        def record_vessels(tissue, result):
            self.vascular_polygons.extend(poly for poly, _tag, _gid in result)

        recipe.fill(
            "packed vessel ring", ring, strategy="packing",
            record=record_vessels,
            # forwarded verbatim to GeometryProcessor.pack_circles:
            proportion=FILL_PROPORTION,
            direction="center",             # large vessels near the centre
            diameter_max=VESSEL_MAX_D,
            diameter_min=VESSEL_MIN_D,
            gradient_function="five_pl",
            gradient_steepness=3.0,
        )
        return recipe


def build() -> VesselRingRoot:
    data = OrganInputData.for_root()          # default monocot layers
    data.set_value("stele", "thickness", 0.30)
    root = VesselRingRoot(data, seed=SEED)
    root.generate_cells()
    return root


def main(show=True):
    root = build()

    counts = {}
    for c in root.all_cells.cells:
        counts[c.type] = counts.get(c.type, 0) + 1
    print("Cell-type census:", counts)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    root.plot_tissues(ax=axes[0], show=False, labels=True, fuse=True)
    axes[0].set_title("Tissue zones (dry run)")
    root.plot_cells(ax=axes[1], show=False, title="Packed vessel ring — cells")
    leg = axes[1].get_legend()
    if leg:
        leg.remove()
    plt.tight_layout()
    if show:
        plt.show()


if __name__ == "__main__":
    main()
