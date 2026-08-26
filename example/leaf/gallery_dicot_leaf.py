"""Dicot leaf cross-section demo (dorsiventral) — midrib bundle designs.

Palisade mesophyll under the adaxial (upper) epidermis, spongy mesophyll above
the abaxial (lower) one, and one big central midrib.  The **same leaf** is shown
with the central midrib built two ways:

1. **arc / cylinder-slice** — concentric xylem / cambium / phloem arcs (a pie
   slice of the continuous dicot-stem cylinder), xylem adaxial -> phloem abaxial,
   with the xylem in radial files (lines);
2. **collateral bundle** — the ordinary compact collateral vein used before the
   arc bundle existed (xylem / cambium / phloem in radial files, packed into one
   envelope).

Everything else is identical between the two panels: outline, mesophyll, and the
minor veins that sit too close to the wide midrib are dropped in both, purely
through the midrib's ``width`` (its tangential extent), which makes
``_vein_layout`` clear any narrower vein within that span.
"""

import os
import sys
import time
from math import radians

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(".."))

from openalea.granap.input_data import OrganInputData
from openalea.granap.leaf_class import LeafAnatomy

SEED = 0

WIDTH = 4.0
THICKNESS = 0.45

# The one big midrib: a 70-degree arc of a 0.28 mm-radius cylinder.  Its tangential
# extent (arc length) is reused as the collateral-bundle width so the two panels are
# the same leaf and drop the same minor veins.
ARC_RADIUS = 0.35
ARC_DEGREES = 160
MIDRIB_WIDTH = ARC_RADIUS * radians(ARC_DEGREES)


def build_dicot_leaf(midrib_kind="arc"):
    """A near-constant-thickness dicot lamina with one big central midrib.

    An ellipse tapers to paper-thin pointed margins that read poorly, so the outline
    is a flat ``thickness_profile`` (full thickness out to near the edge, then a short
    rounded taper).  ``midrib_kind`` picks how the central vein is built:

    * ``"arc"``        — a slice of a vascular cylinder (concentric xylem / cambium /
      phloem arcs, xylem in radial files);
    * ``"collateral"`` — the ordinary compact collateral bundle used before the arc
      bundle existed.

    Both use the same ``width`` (``MIDRIB_WIDTH``), so the two leaves are identical
    apart from the midrib design and drop the same minor veins.
    """
    params = OrganInputData.for_dicot_leaf().to_dict_list()
    plant = next(p for p in params if p["name"] == "planttype")
    plant["width"] = WIDTH
    # Flat to 90% of the half-width, then a quick rounded taper to the margin.
    half = WIDTH / 2.0
    plant["thickness_profile"] = [[0.0, THICKNESS], [0.9 * half, THICKNESS],
                                  [half, 0.0]]
    plant["edge_radius"] = THICKNESS / 2.0     # round the blunt margin

    midrib = next(p for p in params
                  if p["name"] == "vascular_bundle" and p.get("placement") == "center")
    midrib["width"] = MIDRIB_WIDTH             # drops the minor veins within its span
    midrib["xylem_layout"] = "files"           # xylem vessels in radial lines
    # Straight, well-separated files: many narrow files (~one vessel wide) and no
    # jitter, so the xylem vessels read as tidy radial lines instead of a packed blob.
    midrib["n_xylem_files"] = 6
    midrib["xylem_file_jitter"] = 0.0
    # Around a major vein the palisade/spongy differentiation usually gives way to
    # plain mesophyll; replace it in a full-thickness band a little wider than the
    # vein (see mesophyll_region_width).
    midrib["mesophyll_cell_diameter"] = 0.03
    midrib["mesophyll_cell_width"] = 0.03

    # Sunken, keeled midrib: the adaxial (upper) surface dips into a groove (a NEGATIVE
    # rib) while the abaxial (lower) surface bulges into a keel (a positive rib), so the
    # whole midrib sits a bit below the lamina.  The lamina mid-line drops with it, and
    # the bundle is centred on that (shifted-down) mid-line — i.e. it stays centred
    # between the two epidermes — automatically (see LeafAnatomy._vein_y).
    if midrib_kind == "arc":
        midrib["arc_degrees"] = ARC_DEGREES    # a slice of the vascular cylinder
        midrib["arc_radius"] = ARC_RADIUS
        midrib["arc_xylem_thickness"] = 0.15
        midrib["arc_phloem_thickness"] = 0.02
        midrib["arc_cambium_thickness"] = 0.01
        midrib["relative_distance"] = 0.7
        midrib["mesophyll_region_width"] = 0.85   # covers the wide arc + a margin
        midrib["rib_adaxial_height"] = -0.2      # adaxial groove (negative = sunken)
        midrib["rib_adaxial_width"] = 1.1
        midrib["rib_abaxial_height"] = 0.3      # abaxial keel
        midrib["rib_abaxial_width"] = 1.2
    else:                                       # "collateral": the old compact bundle
        midrib["arc_degrees"] = 0.0            # 0 -> ordinary build_bundle (collateral)
        # A bit less wide than the arc's span, but still well bigger than a minor
        # vein (width 0.08).  Kept above ~0.31 so it still clears the innermost minor.
        midrib["relative_distance"] = 0.7
        midrib["width"] = 0.31
        midrib["height"] = 0.24                # compact envelope depth
        midrib["mesophyll_region_width"] = 0.5    # covers the compact bundle + a margin
        midrib["rib_adaxial_height"] = -0.2      # adaxial groove (negative = sunken)
        midrib["rib_adaxial_width"] = 1.1
        midrib["rib_abaxial_height"] = 0.3      # abaxial keel
        midrib["rib_abaxial_width"] = 1.2
    return params


SCENARIOS = [
    ("arc / cylinder-slice midrib", "arc"),
    ("collateral midrib (compact bundle)", "collateral"),
]


def main(show=True):
    fig, axs = plt.subplots(len(SCENARIOS), 1, figsize=(13, 6.5))
    for ax, (label, kind) in zip(axs.ravel(), SCENARIOS):
        print(f"=== dicot leaf — {label} ===")
        t0 = time.time()
        leaf = LeafAnatomy(build_dicot_leaf(kind), seed=SEED)
        leaf.generate_cells()
        print(f"  Time: {time.time() - t0:.2f}s   cells: {len(leaf.all_cells.cells)}")
        leaf.plot_cells(show=False, ax=ax, title=label)
        leg = ax.get_legend()
        if leg is not None:
            leg.set_title("tissue")
            for txt in leg.get_texts():
                txt.set_fontsize(6)
        ax.set_aspect("equal")

    plt.suptitle("Dicot leaf — same lamina, midrib as an arc vs a collateral bundle",
                 fontsize=14)
    plt.tight_layout()
    if show:
        plt.show()


if __name__ == "__main__":
    main()
