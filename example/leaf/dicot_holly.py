"""Holly (*Ilex aquifolium*) leaf cross-section — a measured dicot leaf.

All lengths are in mm.  Built from measurements of a real section:

* length (cross-section width) 6.4; a ~0.45 mm lamina thickening to 0.6 at the
  midrib (0.48 at 0.25 mm, 0.45 at 0.3 mm from the centre).  The extra thickness
  is a bump on the **abaxial** side only (an abaxial keel); the adaxial face stays
  flat;
* a **semicircular arc midrib**, 0.5 mm wide, spanning 170 degrees:
  - 38 radial xylem files, 0.08 mm deep, vessels 0.012 on the inner (adaxial)
    face grading to 0.006 against the cambium;
  - 2 cambium layers, 0.008 mm in total (0.004 cells);
  - a 0.025 mm phloem arc of ~5 layers of 0.004 cells;
  - a 0.05 mm arc of 5-6 layers of thick-walled (suberised) cells outside the
    phloem, tagged ``sclerenchyma``;
  - undifferentiated ground cells (0.019) around the midrib instead of
    palisade / spongy;
* 1 epidermis (0.015 deep x 0.032 wide), 1 hypodermis (0.020 x 0.025),
  2 palisade layers of columnar cells (0.040 x 0.015), and a spongy mesophyll
  with heavy aerenchyma below;
* small round minor veins (0.04 mm), the first 0.6 mm from the centre on each
  side, then one every 0.2-0.6 mm (irregular, as in a reticulate dicot venation);
* hypostomatous: no adaxial stomata, abaxial ones about every 0.25 mm.

Notes on how the measurements map onto the model are inline below.
"""

import os
import sys
import time
from math import radians, sin

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(".."))

from openalea.granap.leaf_class import LeafAnatomy

SEED = 0

LENGTH = 6.4
HALF = LENGTH / 2.0

# -- midrib geometry --------------------------------------------------------
# A circular arc can't be both 0.5 wide and 0.36 tall at 170 degrees, so the
# measured width is kept: the outermost (sclerenchyma) arc's chord is 0.5 mm, and
# the cambium radius follows from the layer thicknesses outside it.
ARC_DEGREES = 170.0
MIDRIB_WIDTH = 0.5
XYLEM_T, CAMBIUM_T, PHLOEM_T, SCL_T = 0.08, 0.008, 0.025, 0.05
OUTER_RADIUS = MIDRIB_WIDTH / (2.0 * sin(radians(ARC_DEGREES / 2.0)))
ARC_RADIUS = OUTER_RADIUS - CAMBIUM_T / 2.0 - PHLOEM_T - SCL_T      # cambium radius

# -- lamina thickness -------------------------------------------------------
# A flat 0.45 mm lamina plus a raised-cosine abaxial keel on the midrib: height
# 0.15 (-> 0.60 at the centre) and full width 0.71 (-> +0.03 = 0.48 at 0.25 mm,
# +0.01 = 0.46 at 0.3 mm, 0 beyond 0.355 mm).
LAMINA = 0.45
KEEL_HEIGHT, KEEL_WIDTH = 0.15, 0.71

MINOR_DIAMETER = 0.04
STOMATA_SPACING = 0.25


def minor_vein_positions(rng, first=0.6, gap=(0.2, 0.6), edge_margin=0.2):
    """x positions of the minor veins: the first at ``first`` mm from the centre
    on each side, then one every ``gap`` mm (uniform random), up to the margin.
    The two halves are drawn independently, so the pattern is not mirrored."""
    xs = []
    for side in (-1.0, 1.0):
        x = first
        while x < HALF - edge_margin:
            xs.append(side * x)
            x += rng.uniform(*gap)
    return sorted(xs)


def build_holly(seed=SEED):
    midrib = dict(
        name="vascular_bundle", placement="center", span_fraction=0.0, n_bundles=1,
        width=MIDRIB_WIDTH,
        # semicircular arc midrib: xylem (inner, adaxial) -> cambium -> phloem ->
        # sclerenchyma (outer, abaxial)
        arc_degrees=ARC_DEGREES, arc_radius=ARC_RADIUS,
        arc_xylem_thickness=XYLEM_T, arc_cambium_thickness=CAMBIUM_T,
        arc_phloem_thickness=PHLOEM_T,
        arc_sclerenchyma_thickness=SCL_T,
        arc_sclerenchyma_cell_diameter=0.009, arc_sclerenchyma_cell_width=0.009,
        xylem_layout="files", n_xylem_files=38,
        # big vessels on the inner face, small against the cambium
        arc_xylem_large_side="inner",
        sieve_diameter_min=0.003,
        # abaxial keel only; the adaxial face stays flat over the midrib
        rib_abaxial_height=KEEL_HEIGHT, rib_abaxial_width=KEEL_WIDTH,
        rib_adaxial_height=0.0,
        # undifferentiated ground cells around the midrib, over the keel's width
        mesophyll_region_width=KEEL_WIDTH,
        mesophyll_cell_diameter=0.019, mesophyll_cell_width=0.019,
    )
    # Small round minor veins: a tiny 'face' bundle (one protoxylem cluster + a
    # little phloem), as for the Nerium minors.
    minor = dict(
        name="vascular_bundle", placement="explicit",
        x_positions=minor_vein_positions(np.random.default_rng(seed)),
        shape="circle", width=MINOR_DIAMETER, height=MINOR_DIAMETER,
        xylem_layout="face", n_metaxylem=0, n_protoxylem=1,
        protoxylem_diameter=0.006, protoxylem_diameter_min=0.005,
        protoxylem_width=0.016, protoxylem_height=0.014,
        phloem_width=0.018, phloem_height=0.010, relative_distance=0.5,
        sheath="none", lacuna=False,
        rib_adaxial_height=0.0, rib_abaxial_height=0.0,
    )

    return [
        {"name": "planttype", "value": 2, "organ": "leaf", "width": LENGTH,
         # flat lamina; the midrib's extra thickness is the abaxial keel (above)
         "thickness_profile": [[0.0, LAMINA], [HALF - 0.4, LAMINA], [HALF, 0.0]],
         "edge_radius": 0.14},
        {"name": "epidermis", "cell_diameter": 0.015, "cell_width": 0.032,
         "n_layers": 1, "shift": 0.3, "order": 4},
        {"name": "hypodermis", "cell_diameter": 0.020, "cell_width": 0.025,
         "n_layers": 1, "shift": 0.3, "order": 3},
        # 2 columnar palisade layers under the adaxial epidermis
        {"name": "palisade", "cell_diameter": 0.040, "cell_width": 0.015, "n_layers": 2},
        # spongy fills the rest (cell size not measured — 0.02 chosen)
        {"name": "spongy", "cell_diameter": 0.02, "cell_width": 0.02},
        midrib, minor,
        # midrib vessels: 0.012 (inner face) grading to 0.006 (cambium)
        {"name": "xylem", "vessel_diameter": 0.012, "vessel_diameter_min": 0.006,
         "vessel_diameter_sd": 0.001},
        # ~5 layers of 0.004 phloem cells in the 0.025 mm arc
        {"name": "phloem", "sieve_diameter": 0.004, "sieve_diameter_sd": 0.0004},
        # 2 cambium layers in 0.008 mm
        {"name": "cambium", "cell_diameter": 0.004},
        # high aerenchyma in the spongy: fine intercellular air + random lacunae
        # (same settings as the Nerium spongy)
        {"name": "inter_cellular_spaces", "tissue": ["spongy"], "smoothness": 0.45},
        {"name": "aerenchyma", "tissue": "spongy", "aerenchyma_proportion": 0.3},
        # hypostomatous: abaxial stomata only, about every 0.25 mm
        {"name": "stomata", "n_adaxial": 0, "n_abaxial": int(round(LENGTH / STOMATA_SPACING)),
         "width": 0.03, "depth": 0.03, "sub_chamber": 0.05},
    ]


def main(show=True):
    print("=== holly (Ilex aquifolium) leaf ===")
    t0 = time.time()
    leaf = LeafAnatomy(build_holly(), seed=SEED)
    leaf.generate_cells()
    counts = {}
    for c in leaf.all_cells.cells:
        counts[c.type] = counts.get(c.type, 0) + 1
    n_b = len(leaf.vascular_tissue_polygons.get("bundle", []))
    print(f"  Time: {time.time() - t0:.2f}s   cells: {len(leaf.all_cells.cells)}   bundles: {n_b}")
    for t in sorted(counts):
        print(f"    {t:14s} {counts[t]}")

    fig, ax = plt.subplots(figsize=(15, 4.0))
    leaf.plot_cells(show=False, ax=ax,
                    title="Holly (Ilex aquifolium) leaf — measured dicot")
    leg = ax.get_legend()
    if leg is not None:
        leg.set_title("tissue")
        for txt in leg.get_texts():
            txt.set_fontsize(7)
    ax.set_aspect("equal")
    plt.tight_layout()
    if show:
        plt.show()


if __name__ == "__main__":
    main()
