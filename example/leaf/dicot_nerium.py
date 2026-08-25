"""Oleander (*Nerium oleander*) leaf cross-section — a measured dicot leaf.

All lengths are in mm.  Built from measurements of a real section:

* length (cross-section width) 6.3; a ~constant ~0.47 mm lamina.  The measured
  thickness rise toward the midrib (1.01 at the centre, 0.56 at 0.635, 0.47 at
  1.76) is an **abaxial keel** — a wide positive rib bulging the lower surface —
  not a thicker lamina;
* a narrow **adaxial groove** over the midrib (a negative rib: 0.07 mm deep at
  the centre, ~0.028 at 0.1 mm — a raised-cosine channel);
* a big **continuous-cylinder midrib**: a 165-degree arc, xylem arc 0.11 thick
  organised as 34 radial files of ~7 vessels each (vessel diameter 0.018 grading
  to 0.012), a phloem arc 0.05 thick with 0.012 mm sieve elements;
* a **multiple (3-layer) epidermis** of tall cells (0.018 x 0.021) — the oleander
  water-storage epidermis;
* **4 palisade layers** of columnar cells (0.041 x 0.012) under the adaxial
  epidermis; a loose spongy mesophyll (0.015) below, riddled with dense, irregular
  aerenchyma lacunae (dicot-style scattered air, not the monocot's per-vein lacuna);
* 16 small minor veins (height 0.07, width 0.043; xylem ~0.006, phloem ~0.005);
* hypostomatous (stomata on the abaxial face only — oleander bears them in sunken
  crypts, idealised here as a plain abaxial row).

A few notes on how the measurements map onto the model are inline below.
"""

import os
import sys
import time
from math import radians, sin

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(".."))

from openalea.granap.leaf_class import LeafAnatomy

SEED = 0

LENGTH = 6.3
HALF = LENGTH / 2.0               

# The midrib is a 165-degree arc; its tangential extent (~chord) is reused as the
# vein ``width`` so the placement rule drops the minor veins that fall under it.
ARC_RADIUS = 0.30
ARC_DEGREES = 165.0
MIDRIB_WIDTH = 2.0 * ARC_RADIUS * sin(radians(ARC_DEGREES / 2.0))


def build_nerium():
    major = dict(
        name="vascular_bundle", placement="center", span_fraction=0.0, n_bundles=1,
        width=MIDRIB_WIDTH,
        # continuous-cylinder (arc) midrib
        arc_degrees=ARC_DEGREES, arc_radius=ARC_RADIUS,
        arc_xylem_thickness=0.11, arc_phloem_thickness=0.05, arc_cambium_thickness=0.012,
        xylem_layout="files", n_xylem_files=34, xylem_file_jitter=0.0,
        # Lower vessel proportion so the vessels sit further apart (more xylem
        # parenchyma between them) — the packed look was too dense.
        prop_vessel=0.42,
        # The lamina is a ~constant ~0.47 mm slab; the thickness increase toward the
        # midrib is an ABAXIAL KEEL (a wide positive rib below), *not* a thicker
        # profile: +0.54 at the centre tapering to 0 by ~0.87 mm reproduces the
        # measured 1.01 (centre) / 0.56 (0.635) / 0.47 (1.76) thicknesses.  On top, a
        # narrow adaxial GROOVE (negative rib) dips the upper surface (0.07 at the
        # centre, ~0.028 at 0.1 mm).
        rib_abaxial_height=0.54, rib_abaxial_width=1.73,
        rib_adaxial_height=-0.07, rib_adaxial_width=0.35,
        # Around the major vein the palisade/spongy differentiation gives way to plain
        # (undifferentiated) mesophyll: a full-thickness band a little wider than the
        # arc, filled with mesophyll cells instead of palisade + spongy.
        mesophyll_region_width=MIDRIB_WIDTH + 0.12,
        mesophyll_cell_diameter=0.02, mesophyll_cell_width=0.02,
    )
    # Small minor veins: a tiny 'face' bundle (one protoxylem cluster + a little
    # phloem, no metaxylem).  The 'face' layout carries its own protoxylem_diameter,
    # which is how the minors get 0.006 mm vessels while the arc keeps the 0.018 mm
    # ones from the shared xylem block (see the note at the bottom).
    minor = dict(
        name="vascular_bundle", placement="scatter", n_bundles=16, span_fraction=0.88,
        width=0.043, height=0.07,
        xylem_layout="face", n_metaxylem=0, n_protoxylem=1,
        protoxylem_diameter=0.006, protoxylem_diameter_min=0.005,
        protoxylem_width=0.022, protoxylem_height=0.03,
        phloem_width=0.022, phloem_height=0.018, relative_distance=0.5,
        sheath="none", lacuna=False,
        rib_adaxial_height=0.0, rib_abaxial_height=0.012, rib_abaxial_width=0.10,
    )

    return [
        {"name": "planttype", "value": 2, "organ": "leaf", "width": LENGTH,
         # ~constant lamina thickness; the midrib's extra thickness is the abaxial
         # keel rib on the major vein (above), not a thicker profile here.
         "thickness_profile": [[0.0, 0.47], [1.76, 0.47], [2.72, 0.45], [HALF, 0.0]],
         "edge_radius": 0.14},
        {"name": "epidermis", "cell_diameter": 0.018, "cell_width": 0.021,
         "n_layers": 1, "shift": 0.3, "order": 4},
        {"name": "hypodermis", "cell_diameter": 0.018, "cell_width": 0.021,
                  "n_layers": 2, "shift": 0.3, "order": 3},
        # 4 columnar palisade layers under the adaxial epidermis
        {"name": "palisade", "cell_diameter": 0.041, "cell_width": 0.012, "n_layers": 4},
        # spongy fills the rest; high intercellular air added below.  Cells are 50%
        # bigger than the measured 0.01 to give a looser, airier spongy mesophyll.
        {"name": "spongy", "cell_diameter": 0.015, "cell_width": 0.015},
        major, minor,
        # arc-vein vessels: 0.018 grading to 0.012 mm
        {"name": "xylem", "vessel_diameter": 0.018, "vessel_diameter_min": 0.012,
         "vessel_diameter_sd": 0.0015},
        {"name": "phloem", "sieve_diameter": 0.0096, "sieve_diameter_sd": 0.0008},
        {"name": "cambium", "cell_diameter": 0.008},
        # high level of fine intercellular air between the spongy cells.  (Pushed no
        # higher than ~0.45 on purpose: above that the fine air bridges every gap and
        # the aerenchyma below fuses into one solid void instead of distinct lacunae.)
        {"name": "inter_cellular_spaces", "tissue": ["spongy"], "smoothness": 0.45},
        # aerenchyma: random spongy cells are turned into air
        {"name": "aerenchyma", "tissue": "spongy", "aerenchyma_proportion": 0.25},
        # hypostomatous: stomata on the abaxial face only
        {"name": "stomata", "n_adaxial": 0, "n_abaxial": 14,
         "width": 0.03, "depth": 0.03, "sub_chamber": 0.05},
    ]


def main(show=True):
    print("=== oleander (Nerium oleander) leaf ===")
    t0 = time.time()
    leaf = LeafAnatomy(build_nerium(), seed=SEED)
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
                    title="Oleander (Nerium oleander) leaf — measured dicot")
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
