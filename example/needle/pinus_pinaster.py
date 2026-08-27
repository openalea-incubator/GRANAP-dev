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
  - transfusion parenchyma (ellipse of major 0.05, 0.04 minor) 60% of occupancy filled via circle-packing
  - transusion tracheid (in same quantities)


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
2 resin ducts (diameter 0.1 including their parenchyma cells 0.012)

abaxial
  _____
 /     \"
/_______\"
adaxial

A few notes on how the measurements map onto the model are inline below.
"""

import os
import sys
import math

SEED = 0

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(".."))

from openalea.granap.needle_class import NeedleAnatomy

def _bump_profile(peaks, floor=0.15):
    """Build a circular (angle_deg, multiplier) thickness_profile from a list
    of (center_angle_deg, half_width_deg, peak_multiplier) nodules.

    Each nodule is a rise/peak/fall triangle over a `floor` baseline
    elsewhere (angles wrap circularly, see NeedleAnatomy._angular_multiplier);
    keep nodules well separated so adjacent triangles don't overlap.

    `floor` must stay nonzero: at floor=0 a ring's boundary coincides with
    its outer neighbor's, producing near-coincident Voronoi seeds and wildly
    oversized (numerically unstable) cell polygons at the valley nodules.
    """
    pts = [(0.0, floor), (360.0, floor)]
    for center, half_width, mult in peaks:
        pts.append((center - half_width, floor))
        pts.append((center, mult))
        pts.append((center + half_width, floor))
    return sorted(pts)


def _pole_and_corner_angles(width, thickness):
    """Locate the needle cross-section's adaxial pole, abaxial pole, and two
    corners as polar angles (degrees) around the base shape's centroid --
    the convention "thickness_profile"/"zone_angles" entries use (see
    NeedleAnatomy._angular_multiplier / _offset_layer_polygon).

    The base half-ellipse (GeometryProcessor.half_ellipse_polygon) is flat
    at y=0 (adaxial edge, x in [-width/2, width/2]) and domed up to
    (0, thickness) (abaxial peak). Treated as a uniform lamina its centroid
    sits at y_c = 4*thickness/(3*pi) above the flat edge (half-disk-centroid
    formula) -- a naive "0=adaxial, 180=abaxial, +-90=corners" guess is wrong
    for this shape's aspect ratio, putting the poles at the corners instead.
    """
    a = width / 2.0
    y_c = 3.5 * thickness / (3.0 * math.pi)
    adaxial_pole = math.degrees(math.atan2(-y_c, 0.0)) % 360.0
    abaxial_pole = math.degrees(math.atan2(thickness - y_c, 0.0)) % 360.0
    corner_pos = math.degrees(math.atan2(-y_c, a)) % 360.0    # near 0/360 side
    corner_neg = math.degrees(math.atan2(-y_c, -a)) % 360.0   # near 180 side
    return adaxial_pole, abaxial_pole, corner_pos, corner_neg


def build_pinaster():
    WIDTH, THICKNESS = 2.232, 1.29
    _, ABAXIAL_POLE, CORNER_POS, CORNER_NEG = _pole_and_corner_angles(WIDTH, THICKNESS)

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
    # total hypodermis cell layers there), none elsewhere.
    HYPODERMIS_FLOOR = 0.7
    NODULE_PROFILE = _bump_profile(
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
        {"name": "hypodermis", "cell_diameter": 0.0225, "n_layers": 2, "order": 5},
        {"name": "hypodermis_corner", "cell_diameter": 0.0225, "n_layers": 3, "order": 5.1,
         "thickness_profile": NODULE_PROFILE,
         "zone_angles": {"mode": "wedge", "centers": [CORNER_POS, CORNER_NEG],
                         "half_width": CORNER_ZONE_HALF_WIDTH}},
        {"name": "epidermis", "cell_diameter": 0.02, "order": 6},
        # Transfusion tissue: circle-packed (NeedleAnatomy.
        # add_transfusion_tissue) at 60% occupancy, tracheids and parenchyma
        # in two structural passes (tracheids into the full zone first,
        # parenchyma into what's left) rather than the ring seeder.
        {"name": "transfusion_tissue", "n_layers": 2, "pack_circles": True,
         "diameter_max": 0.05, "proportion": 0.6,
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
        # Resin ducts: "diameter 0.1 including their parenchyma cells 0.012"
        # -> the inner-canal packing diameter plus the ring cell size.
        {"name": "resin_duct", "n_files": 2, "diameter": 0.1, "cell_diameter": 0.012},
        # Directional, corner-excluded stomata: 11 abaxial (domed/top), 6
        # adaxial (flat/bottom) -- see docstring diagram above.
        {"name": "stomata", "n_adaxial": 8, "n_abaxial": 11, "edge_margin": 0.08,
         "width": 0.025, "depth": 0.08, "sub_chamber": 0.04},
        # Rhombic wall-centred air spaces for the "loose" spongy mesophyll
        # (NeedleAnatomy._apply_mesophyll_wall_rhombi covers both mesophyll
        # rings).
        {"name": "inter_cellular_spaces", "tissue": ["mesophyll"], "smoothness": [0.3]},
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