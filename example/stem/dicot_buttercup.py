"""Build a buttercup stem (*Ranunculus*, dicot eustele) and plot it.

Measured on ``CT tige dico renoncule2.scn``.  A 3 mm-diameter stem: one epidermis
then directly the ground tissue (no separate cortex), carrying a ring of **8 big +
8 small** open collateral bundles, alternating.  Primary growth: fascicular cambium
only (no interfascicular cambium).

* **big** — 0.24 mm high x 0.17 mm wide (fibre caps not included), centred
  0.320 mm under the epidermis; fibre caps outside that envelope: 2 layers toward
  the centre, and toward the epidermis a cap thinning from 5 layers over the pole
  (90°) through 4 (70°) and 3 (30°) to 2 at the flanks (0°);
  xylem in 2 poles with a few vessels between, vessel diameter 0.035 (near the
  cambium) -> 0.018 (inner edge); phloem 0.09 x 0.07 (w x h), sieve 0.011,
  companion cells 0.010 x 0.02 (d x w).
* **small** — 0.15 x 0.125, centred 0.270 mm under the epidermis; the same cells,
  but scaled-down regions (fewer vessels).

Ground-tissue cell diameter: 0.017 under the epidermis -> 0.035 at 0.18 mm depth
-> 0.076 at 0.46 mm -> 0.09 at the centre.

Both kinds are anchored on their fascicular cambium, so the cambium ring sets the
bundle depth.  The small bundles' cambium is shallower than the big ones', so the
ring is an 8-arm **star**: arms (outer radius) under the small bundles, valleys
(inner radius) under the big ones.
"""

import os
import sys
import time

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(".."))

from openalea.granap.stem_class import StemAnatomy
from openalea.granap.input_data import (
    OrganInputData, VascularBundleParams, BundlePatternParams,
)

SEED = 0

# Sizes in mm.
STEM_RADIUS = 1.5                                   # 3 mm diameter
EPIDERMIS = 0.015
PITH_RADIUS = STEM_RADIUS - EPIDERMIS               # ground tissue runs to the epidermis

FIBRE = 0.010                                       # sclerenchyma (cap) cell size
N_CAP_IN = 2                                        # fibre layers toward the centre
N_CAP_OUT, N_CAP_OUT_FLANK = 5, 2                   # toward the epidermis: pole / flanks

# Measured envelope heights exclude the fibre caps, which extend outside them.
BIG_HEIGHT = 0.24
SMALL_HEIGHT = 0.15

# Cambium depth under the stem surface = bundle-centre depth - the cambium offset
# from the envelope centre (``bundle_cambium_anchor`` for these specs, with the
# xylem / phloem / cambium shares 0.60 / 0.35 / 0.05):
#   big:   0.320 - 0.0286 = 0.291
#   small: 0.270 - 0.0174 = 0.253
RING = dict(
    ring_shape="star", n_peaks=8,
    radius_peak_side=STEM_RADIUS - 0.253,           # arms   -> small bundles
    radius_valley_side=STEM_RADIUS - 0.291,         # valleys -> big bundles
    # flat arm tips (~0.30 mm) and flat valleys (~0.42 mm), so each bundle
    # sits on a level stretch of the ring and its cambium stays tangential.
    arc_peak_side=0.15, arc_valley_side=0.25,
)

BASE = dict(
    bundle_type="collateral", has_cambium=True, shape="ellipse",
    xylem_layout="files", n_xylem_files=2, xylem_file_jitter=0.5,  # 2 poles, a bit irregular
    xylem_fraction=0.60, phloem_fraction=0.35, cambium_fraction=0.05,
    prop_vessel=0.6, prop_sieve=0.5,
    sieve_diameter_min=0.008,
    companion_cell_diameter=0.010, companion_cell_width=0.02,
    parenchyma_diameter=0.01, parenchyma_width=0.01,
    sheath="none",
    n_caps_layers_inward=N_CAP_IN, n_caps_layers_outward=N_CAP_OUT,
    n_caps_layers_outward_flank=N_CAP_OUT_FLANK,
    sclerenchyma_cell_diameter=FIBRE, sclerenchyma_cell_width=FIBRE,
    **RING,
)

BUNDLES = [
    VascularBundleParams(**{**BASE, **dict(
        kind="big", width=0.17, height=BIG_HEIGHT,
        phloem_width=0.09, phloem_height=0.07, phloem_relative_distance=0.5,
    )}),
    # Same cells, scaled-down regions: the xylem band holds only a few vessels.
    VascularBundleParams(**{**BASE, **dict(
        kind="small", width=0.125, height=SMALL_HEIGHT,
        phloem_width=0.066, phloem_height=0.044, phloem_relative_distance=0.5,
    )}),
    # small first so it lands on the star arms (align_to_arms); equal-angle
    # spacing puts each big bundle at the centre of a valley.
    BundlePatternParams(sequence=["small", "big"], repeats=8,
                        spacing="angle", align_to_arms=True),
]


def build_buttercup() -> OrganInputData:
    """Assemble the buttercup-stem ``OrganInputData`` (dicot preset + 8 big / 8 small)."""
    data = OrganInputData.for_dicot_stem()

    # -- Ground tissue, epidermis to centre.  The 5PL shape is fitted to the four
    #    measured points (edge ~0.017, 0.18 mm -> ~0.036, 0.46 mm -> ~0.074,
    #    centre 0.09); ``cell_diameter`` is the 5PL lower asymptote. ------------
    data.set_value("pith", "thickness",                2 * PITH_RADIUS)
    data.set_value("pith", "cell_diameter",            0.010)
    data.set_value("pith", "cell_diameter_center",     0.09)
    data.set_value("pith", "size_gradient_function",   "five_pl")
    data.set_value("pith", "size_gradient_inflection", 1.0)
    data.set_value("pith", "size_gradient_steepness",  7.33)
    data.set_value("pith", "size_gradient_asymmetry",  3.45)
    data.set_value("pith", "cavity_radius",            0.0)

    # -- No cortex: the epidermis sits directly on the ground tissue ----------
    for name in ("cortex", "aerenchyma"):
        data.remove_param(name)
    # Intercellular air spaces in the ground tissue 
    data.set_values("inter_cellular_spaces", tissue=["parenchyma"], smoothness=0.2)
    data.set_values("epidermis", cell_diameter=EPIDERMIS, cell_width=EPIDERMIS)

    # -- Bundle cells: graded vessels (big near the cambium), phloem sieve ------
    data.set_values("xylem", vessel_diameter=0.035, vessel_diameter_min=0.018,
                    vessel_diameter_sd=0.002)
    data.set_values("phloem", sieve_diameter=0.011, sieve_diameter_sd=0.001)
    # Fascicular cambium: small, flat initials (preset 0.01 x 0.02, divided by 4).
    data.set_values("cambium", cell_diameter=0.0025, cell_width=0.005)

    # -- Vasculature: replace the preset bundle with the big/small pattern -----
    data.remove_param("vascular_bundle")
    data.params += BUNDLES
    return data


def main(show=True):
    data = build_buttercup()
    print("=== buttercup dicot stem ===")
    t0 = time.time()
    stem = StemAnatomy(data, seed=SEED)
    stem.generate_cells()
    dt = time.time() - t0
    counts = {}
    for c in stem.all_cells.cells:
        counts[c.type] = counts.get(c.type, 0) + 1
    n_got = len(stem.vascular_tissue_polygons.get("bundle", []))
    print(f"  Time: {dt:.2f}s   bundles placed: {n_got}   cells: {len(stem.all_cells.cells)}")
    for t in sorted(counts):
        print(f"    {t:16s} {counts[t]}")

    fig, ax = plt.subplots(figsize=(10, 10))
    stem.plot_cells(show=False, ax=ax, title="Buttercup dicot stem — 8 big + 8 small bundles")
    leg = ax.get_legend()
    if leg is not None:
        leg.set_title("tissue")
        for txt in leg.get_texts():
            txt.set_fontsize(7)
    plt.tight_layout()
    if show:
        plt.show()


if __name__ == "__main__":
    main()
