"""Pinus (*Pinus pinaster*) needle cross-section — a measured pine needle.

All lengths are in mm.  Built from measurements of a real section:

Global dimension:
- length (cross-section width) 2.232; max thickness 1.29 mm.
- lenght of the central cylinder 1.47 (endodermis included); max thickness 0.63 mm.

*1 layer of epidermis*, *2 hypodermis layers* (up to 5 hypodermis cell in the corner), 

*1 palisade layer* (height 0.09, width 0.04) with a central anticlinal infoldoing on the adaxial and abaxial side. 

*1 loose spongy mesophyll layer* (height 0.1, width 0.06) on the adaxial side (under the central
cylinder, this model's convention) with up to 2 infolding per cell.
*2 loose spongy mesophyll layers* on the abaxial side (height 0.08, width 0.065) with up to 2
infolding per cell.
(the abaxial-layer height was recorded as "0.8" and both widths as "0.6"/"0.65" — all three
treated as decimal-point typos (0.08, 0.06, 0.065), consistent with every other cell dimension
here being <=0.09 mm; a literal 0.6 mm cell width left the mesophyll ring with only ~6 cells total)

*1 endodermis layer* (height 0.035, width 0.045)

*2 transfusion tissue layers* (height 0.1) ~3.36 mm²:
  - transfusion parenchyma (ellipse of major 0.05, 0.04 minor) the dominant, larger cells, packed
    first into the full zone
  - transfusion tracheid, smaller and more numerous, packed second into the residual gaps around
    the parenchyma
  the two share the 85% occupancy target evenly ("in same quantities"), but because the second
  pass packs into a gap-constrained residue it under-fills, so the realized split is ~63/37 by
  area in the parenchyma's favour


2 vascular ellipses (major 0.32, minor 0.2; Angle 30°) ~0.055mm²; the 30° orientation is set
explicitly via central_cylinder.vascular_angle -- left unset, the engine auto-detects orientation
from the shape of the narrow sliver each bundle is fit into, which for this needle reads as ~90°
and makes the intended wide (major) axis appear vertical instead ("permuted" major/minor).
GeometryProcessor.two_ellipses mirrors the given angle across the vertical midline for the
right-hand bundle (180-angle), so the pair reads as a symmetric "V" tilted outward rather than
both bundles leaning the same direction.
 - 1 cambium layers in middle
 - clusters of 3 cells separated by 0.015 mm width cell
 - small phloem cell of 0.008 mm height; 0.013 mm xylem, 3 Strasburger cells of 0.02 at the edge of each vascuar ellipse.
 - 0.02 vascular parenchyma

11 stomata on the abaxial side, 6 on the adaxial side; none in corners
2 resin ducts, modeled inside-out from direct measurements: an open lumen
(canal) 0.037mm across, a single epithelium cell layer directly bordering it
(radial 0.006mm, tangential 0.013mm), and an outer sheath cell layer
(radial 0.018mm, tangential 0.023mm) additive beyond the epithelium -- true
total footprint (~0.085mm) is derived from these, not a separate figure.

abaxial
  _____
 /     \"
/_______\"
adaxial

A few notes on how the measurements map onto the model are inline below.
"""

import os
import sys

SEED = 0

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(".."))

from openalea.granap.needle_class import NeedleAnatomy


def build_pinaster():
    WIDTH, THICKNESS = 2.232, 1.29
    _, ABAXIAL_POLE, CORNER_POS, CORNER_NEG = NeedleAnatomy.pole_and_corner_angles(WIDTH, THICKNESS)

    # thickness_profile floors stay well above 0 (never fully flush with the
    # outer neighbor's boundary) so CellGenerator.generate_cells_info's
    # next-layer "bleed" clip doesn't drop that neighbor's border cells --
    # each floor is sized to clear ~0.7 * the *outer* neighbor's own
    # cell_diameter/cell_width against this ring's own floor-depth
    # (cell_diameter * floor). zone_angles then restricts actual cell
    # *presence* to the real zone (a corner wedge, or one half of the
    # cross-section), so the profile's floor only has to satisfy the
    # bleed-clip margin, not also read as "no extra layer here" on its own.

    # Extra abaxial-only mesophyll ring: 2 layers abaxial (rounded/domed
    # side), 1 adaxial (flat side, under the central cylinder). Floor must
    # clear *palisade*'s border extension (cell_diameter=0.09, the next
    # layer out): ~0.7*0.09=0.063 needs to stay under floor*0.08 (this
    # ring's own cell_diameter) => floor >= ~0.79; 0.9 keeps margin.
    MESOPHYLL_FLOOR = 0.9
    ABAXIAL_MESOPHYLL_PROFILE = sorted([
        (ABAXIAL_POLE, 1.0),
        (CORNER_NEG, MESOPHYLL_FLOOR),
        (CORNER_POS, MESOPHYLL_FLOOR),
    ])

    # Hypodermis corner thickening: 2 large deposits at the corners (up to 5
    # total hypodermis cell layers there), none elsewhere. Floor must clear
    # *epidermis*'s own border-point bleed-clip margin (its outer neighbor,
    # cell_diameter=0.02): ~0.7*0.02=0.014 needs to stay under
    # floor*hypodermis_corner's own cell_diameter (0.0225) => floor >=
    # ~0.622; 0.7 left only ~11% margin (too thin -- was cropping epidermis
    # at the adaxial pole, where this ring sits right at the floor with no
    # corner nodule to widen it). 0.85 keeps margin, same as MESOPHYLL_FLOOR.
    HYPODERMIS_FLOOR = 0.85
    NODULE_PROFILE = NeedleAnatomy.corner_bump_profile(
        [(CORNER_POS, 10.0, 1.0), (CORNER_NEG, 10.0, 1.0)],
        floor=HYPODERMIS_FLOOR,
    )
    # Kept a couple degrees narrower than the profile bump above, so the
    # visible zone stays inside the bump's near-peak region rather than
    # tapering off toward its own edges.
    CORNER_ZONE_HALF_WIDTH = 4.0

    return [
        {"name": "planttype", "value": 3, "organ": "needle",
         "width": 2.232, "thickness": 1.29},
        # Half-ellipse central cylinder (schema default; "ellipse" would
        # morph it via reshape_layers, not wanted here). vascular_width/
        # height are the two vascular bundles' major/minor axes; overall
        # central-cylinder extent isn't directly settable with this shape --
        # it emerges from the sum of layer thicknesses peeled inward.
        # vascular_angle is the docstring's "Angle 30°": left unset, the
        # engine auto-detects orientation from the narrow sliver each bundle
        # is fit into (~90° here), making the wide axis appear vertical.
        {"name": "central_cylinder",
         "vascular_width": 0.33, "vascular_height": 0.25, "vascular_angle": 30,
         "cell_diameter": 0.02},  # "0.02 vascular parenchyma"
        {"name": "endodermis", "cell_diameter": 0.035, "cell_width": 0.045, "order": 3},
        # Spongy mesophyll: 1 layer everywhere, plus a second ring peaking at
        # the abaxial pole (ABAXIAL_MESOPHYLL_PROFILE/zone_angles) for 2
        # layers abaxially, 1 adaxially. A distinct name is required
        # (LayerManager rejects duplicate names; several algorithms look up
        # "mesophyll" assuming a single ring), so it's its own cell type.
        {"name": "mesophyll", "cell_diameter": 0.1, "cell_width": 0.06, "order": 4},
        {"name": "mesophyll_abaxial", "cell_diameter": 0.08, "cell_width": 0.065, "order": 4.1,
         "thickness_profile": ABAXIAL_MESOPHYLL_PROFILE,
         "zone_angles": {"mode": "half", "pole": ABAXIAL_POLE}},
        # Palisade: single uniform ring. Anticlinal wall infolding is a
        # cell-wall-shape detail the engine doesn't model.
        {"name": "palisade", "cell_diameter": 0.09, "cell_width": 0.04, "order": 4.5},
        # Hypodermis: 2 layers everywhere, plus up to 3 genuine extra rows
        # (n_layers=3, not bigger cells -- cells_on_layer seeds one row per
        # ring regardless of depth) confined to the two corners via
        # NODULE_PROFILE (structural floor) + zone_angles (actual presence).
        {"name": "hypodermis", "cell_diameter": 0.02, "cell_width":0.0225, "n_layers": 2, "order": 5},
        {"name": "hypodermis_corner", "cell_diameter": 0.0225, "n_layers": 3, "order": 5.1,
         "thickness_profile": NODULE_PROFILE,
         "zone_angles": {"mode": "wedge", "centers": [CORNER_POS, CORNER_NEG],
                         "half_width": CORNER_ZONE_HALF_WIDTH}},
        {"name": "epidermis", "cell_diameter": 0.015, "cell_width":0.02, "order": 6},
        # Transfusion tissue: circle-packed (NeedleAnatomy.
        # add_transfusion_tissue) at 85% occupancy, in two structural passes
        # -- parenchyma (large ellipses, 0.045 diameter) into the full zone
        # first, then tracheids (small, 0.022 diameter, more numerous) into
        # what's left -- rather than the ring seeder. transfusion_tracheids
        # _ratio=1.0 is the docstring's "in same quantities" (a 50/50 split of
        # the packed occupancy target). The *realized* areas are not 50/50
        # though: the first pass nearly reaches its target while the second
        # packs into a gap-constrained residue and under-fills, so 1.0 lands
        # at ~63% parenchyma / ~37% tracheid -- parenchyma reads as the
        # dominant element with tracheids as a fine matrix between them,
        # matching Transfusion_tissue.png. (Measured sweep: ratio 0.6 -> 78%
        # parenchyma, too dominant, tracheids reduced to slivers; 1.5 -> 49%,
        # back to parenchyma not occupying enough.)
        {"name": "transfusion_tissue", "n_layers": 2, "pack_circles": True,
         "diameter_max": 0.05, "proportion": 0.85,
         "parenchyma_diameter": 0.045, "tracheids_diameter": 0.022,
         "transfusion_tracheids_ratio": 1.0},
        # Vascular grid: n_per_cluster=3/n_clusters=4 approximates "clusters
        # of 3 cells separated by a 0.015mm interstitial cell"; n_files is a
        # density knob, tuned down to fit the small vascular ellipses.
        {"name": "xylem", "cell_diameter": 0.013, "n_files": 3, "n_clusters": 4, "n_per_cluster": 3},
        {"name": "phloem", "cell_diameter": 0.008, "n_files": 3},
        {"name": "cambium", "cell_diameter": 0.01},
        # Sizes the interstitial-lineage "Str. Interstitial cell" grid
        # columns; the corner "Strasburger cell" cluster is retagged from
        # parenchyma at its own (unrelated) size -- see
        # NeedleAnatomy.retag_corner_parenchyma.
        {"name": "Strasburger cells", "cell_diameter": 0.02},
        # Resin duct, inside-out from direct measurements: lumen 0.037 ->
        # epithelium (radial/height 0.006, tangential/width 0.013) -> sheath
        # (radial/height 0.018, tangential/width 0.023), each layer strictly
        # additive. True total outer diameter is derived, not itself a
        # parameter: 0.037 + 2*0.006 + 2*0.018 ~= 0.085mm.
        {"name": "resin_duct", "n_files": 2, "lumen_diameter": 0.037,
         "cell_diameter": 0.006, "cell_width": 0.013,
         "sheath_cell_diameter": 0.018, "sheath_cell_width": 0.023},
        # Directional, corner-excluded stomata: 11 abaxial (domed/top), 6
        # adaxial (flat/bottom) -- see docstring diagram above.
        # chamber_clearance=2.0: delete the hypodermis cell(s) directly inward of
        # each sub-stomatal chamber (2 hypodermis-cell-diameters of reach, via an
        # oriented column under the chamber -- see NeedleAnatomy._inward_column)
        # so palisade mesophyll can extend up to the chamber, matching the
        # sunken-stoma anatomy. Trade-off (tune_clearance.py ray-cast sweep): this
        # opens 14/18 stomata at ~3.6 hypodermis cells removed per stoma on
        # average (449 -> 384 total); the remaining ~4 sit against the corner
        # hypodermis_corner nodules (up to 5 layers there vs. 2 elsewhere) and
        # would need disproportionately more reach -- costly everywhere else --
        # to also clear, so 16/18 was not reachable within a ~3/stoma budget.
        {"name": "stomata", "n_adaxial": 8, "n_abaxial": 11, "edge_margin": 0.1,
        # sunken=True splits each guard cell into its outer rectangle and the
        # ellipse below it: the ellipse stays the guard cell, sunk into a pit,
        # and the rectangle becomes an epidermal cell arching over it -- the
        # sunken stoma of "Pine needle stoma cuticle hypodermis.jpg".
         "width": 0.021, "depth": 0.08, "sub_chamber": 0.06, "chamber_clearance": 2.0,
        # guard_cell_diameter/guard_cell_aspect flatten the guard-cell ellipse
        # from the previous round default (cell.width, aspect 0.5) into a
        # wider, lower lens shape -- closer to a real guard cell's outline
        # than a round blob.
         "guard_cell_diameter": 0.02, "guard_cell_aspect": 0.7,
         "sunken": True},
        # Rhombic wall-centred air spaces for the "loose" spongy mesophyll
        # (NeedleAnatomy._apply_mesophyll_wall_rhombi covers both mesophyll
        # rings), plus thin full-height air slits carved every two palisade
        # cells (NeedleAnatomy._apply_palisade_wall_slits).
        {"name": "inter_cellular_spaces", "tissue": ["mesophyll", "palisade"],
         "smoothness": [0.3, 0.0], "slit_width": [0, 0.002], "slit_every": [0, 2]},
    ]

def main(show=True):
    print("=== Pine (*Pinus pinaster*) needle leaf ===")
    leaf = NeedleAnatomy(build_pinaster(), seed=SEED)
    leaf.generate_cells()
    leaf.plot_layers(show=show, title=f"Needle Layers")
    counts = {}
    for c in leaf.all_cells.cells:
        counts[c.type] = counts.get(c.type, 0) + 1
    for t in sorted(counts):
        print(f"    {t:14s} {counts[t]}")

    fig, ax = plt.subplots(figsize=(15, 4.0))
    leaf.plot_cells(show=False, ax=ax,
                    title="Pine (*Pinus pinaster*) needle leaf")
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