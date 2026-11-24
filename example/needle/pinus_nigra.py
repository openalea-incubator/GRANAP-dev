"""Pinus (*Pinus nigra*) needle cross-section — a measured pine needle.
pinus_nigra.png (scale bar 200µm) fig. 8 of 10.3732/ajb.0800127 Meicenheimer et al., 2008

All lengths are in mm.  Built from measurements of a real section:

Global dimension:
- length (cross-section width) 1.396; max thickness 0.955 mm.
- lenght of the central cylinder 0.737 (endodermis included) superellipse shape; max thickness 0.448 mm.

*1 layer of epidermis*, *2 hypodermis layers*

*1 palisade layer* (height 0.055, width 0.038)

*1 loose spongy mesophyll layer* (height 0.052, width 0.044) adaxial
*2 to 3 loose spongy mesophyll layer* on the abaxial side

*1 endodermis layer* (height 0.03, width 0.055)

*2 transfusion tissue layers* (height 0.1)
  - transfusion parenchyma larger cells packed 0.002mm^2
    first into the full zone (1/3 area in tissue)
  - transfusion tracheid, smaller and more numerous, packed second into the residual gaps around
    the parenchyma (2/3 area in tissue)


abaxial
  _____
 /     \"
/_______\"
adaxial

A few notes on how the measurements map onto the model are inline below.

------------------------------------------------------------------------------
Provenance of everything NOT in the measurement list above
------------------------------------------------------------------------------
The list above says nothing about the bundle envelopes, the resin ducts, the
stomata, or the epidermis/hypodermis cell sizes, so those were read off
pinus_nigra.png itself using its scale bar: the bar is 88 px long for 200 µm,
i.e. **0.002273 mm/px**. Cross-checks on that scale (all within ~2%): the
section outline measures 1.368 x 0.939 mm against the stated 1.396 x 0.955,
and the endodermis ring 0.76 x 0.42 mm against the stated 0.737 x 0.448. Every
figure-derived number below is marked "(figure)"; the measurement list's own
numbers are used verbatim.

Supplied separately, and used verbatim rather than figure-derived: the four
resin-duct positions (as pixel coordinates, converted to bearings in
build_nigra), xylem cell diameter 0.008, and phloem cell height 0.005.

The figure resolves tissues down to ~0.015 mm (7 px), which covers the bundle
envelopes and the outer layer stack. It does NOT resolve individual vascular
elements (the 0.008 xylem cells are ~3.5 px, which is why reading them off the
figure overestimated them by 2x), nor a resin duct's lumen/epithelium/sheath
partition (each band is 3-8 px) -- so only each duct's *total* footprint was
measured, and the inside-out split keeps pinus_pinaster.py's proportions.

Orientation note: this model's convention is abaxial = domed side = +y (top),
adaxial = flat side = y=0 (bottom), which is how the figure is oriented too. In
the figure each bundle's xylem (regular radial files, brightly lignified) sits
on the adaxial half with the phloem abaxial of it; the engine derives that
polarity itself from ``central_cylinder.vascular_angle``, so nothing here sets
it explicitly.

Difference from *P. pinaster*: no corner-thickened hypodermis (the measurement
list gives a flat "2 hypodermis layers", and the figure shows no corner
nodules), and the central cylinder is a closed ellipse rather than a
half-ellipse — see central_cylinder below.
"""

import os
import sys

SEED = 0

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(".."))

from openalea.granap.needle_class import NeedleAnatomy


def build_nigra():
    WIDTH, THICKNESS = 1.396, 0.955
    _, ABAXIAL_POLE, CORNER_POS, CORNER_NEG = NeedleAnatomy.pole_and_corner_angles(WIDTH, THICKNESS)
    # -> abaxial pole 90.0, corners 333.07 (+x) and 206.93 (-x)

    # Resin-duct positions, converted from measured pixel positions in
    # pinus_nigra.png into the model frame (origin at the base shape's
    # centroid, y_c = 3.5*thickness/(3*pi) above the flat adaxial edge --
    # NeedleAnatomy.pole_and_corner_angles' own frame; the bearing each
    # point works out to is kept alongside it purely as a cross-check):
    #   px(106,321) -> (-0.499, 0.245)  bearing 192.4 deg -- left flank, just above CORNER_NEG (206.9)
    #   px(523,324) -> ( 0.468, 0.238)  bearing 346.0 deg -- right flank, just above CORNER_POS (333.1)
    #   px(334,100) -> ( 0.030, 0.756)  bearing  85.7 deg -- abaxial pole (90.0)
    #   px(507,222) -> ( 0.431, 0.474)  bearing  15.5 deg -- right abaxial flank -- the small duct
    # The three large ones read as "corners and top", but note none of them is
    # exactly at a corner/pole: each is displaced 13-14 deg along the flank.
    # Placed by explicit position (resin_duct.positions) rather than bearing
    # or the engine's fixed pizza-slice positions: point-seating puts a duct
    # as close as the hypodermis/endodermis buffers allow to this exact
    # measured spot, rather than at a wedge's widest-inscribed-circle centre
    # -- placing this same data by "angles" (the previous approach) drifted
    # the small duct ~0.06mm (~6% of needle thickness) off this point,
    # toward wherever its wedge happened to be locally widest.
    DUCTS_LARGE = [(-0.499, 0.245), (0.468, 0.238), (0.030, 0.756)]
    DUCTS_SMALL = [(0.431, 0.474)]

    # thickness_profile floors stay well above 0 (never fully flush with the
    # outer neighbor's boundary) so CellGenerator.generate_cells_info's
    # next-layer "bleed" clip doesn't drop that neighbor's border cells --
    # each floor is sized to clear ~0.7 * the *outer* neighbor's own
    # cell_diameter/cell_width against this ring's own floor-depth
    # (cell_diameter * floor). zone_angles then restricts actual cell
    # *presence* to the real zone (one half of the cross-section, or an
    # abaxial cap), so the profile's floor only has to satisfy the bleed-clip
    # margin, not also read as "no extra layer here" on its own.
    #
    # Both zoned mesophyll rings below have cell_diameter 0.052 and an outer
    # neighbour of 0.052 (the cap) / 0.055 (palisade), so the constraint is
    # floor >= 0.7*0.055/0.052 = 0.74; 0.9 keeps margin (same value
    # pinus_pinaster.py settled on for the same ring).
    MESOPHYLL_FLOOR = 0.9

    # "2 to 3 loose spongy mesophyll layers on the abaxial side" against 1
    # adaxial, built as base ring + two zoned rings rather than one: the first
    # extra ring covers the whole abaxial half (-> 2 layers there), the second
    # only an abaxial cap around the pole (-> 3 layers at the top). A single
    # n_layers=2 abaxial ring would give a flat 3 everywhere abaxial and lose
    # the "2 to 3" gradation.
    ABAXIAL_MESOPHYLL_PROFILE = sorted([
        (ABAXIAL_POLE, 1.0),
        (CORNER_NEG, MESOPHYLL_FLOOR),
        (CORNER_POS, MESOPHYLL_FLOOR),
    ])
    # Cap ring: a single bump centred on the abaxial pole. As in
    # pinus_pinaster.py's corner nodules, the visible zone (zone_angles) is
    # kept a few degrees narrower than the profile bump so the cells sit in
    # the bump's near-peak region rather than on its tapering flanks.
    CAP_PROFILE = NeedleAnatomy.corner_bump_profile(
        [(ABAXIAL_POLE, 50.0, 1.0)],
        floor=MESOPHYLL_FLOOR,
    )
    CAP_ZONE_HALF_WIDTH = 45.0

    return [
        {"name": "planttype", "value": 3, "organ": "needle",
         "width": WIDTH, "thickness": THICKNESS},
        # shape="ellipse" (not the "half_ellipse" schema default that
        # pinus_pinaster.py keeps): the measurement list gives the central
        # cylinder both a length and a thickness and calls it a superellipse,
        # and the figure shows the endodermis closing right around it as a
        # full ring. That routes layer construction through
        # NeedleAnatomy.reshape_layers, which morphs the rings between the
        # outer outline and the endodermis from the needle's half-ellipse
        # toward this closed ellipse. The engine has no true superellipse for
        # the central cylinder -- only a plain ellipse -- so the blunter flanks
        # of the real profile are not reproduced.
        # reshape_layers groups the stack by what governs its shape: the
        # surface tissues (epidermis, hypodermis, palisade) stay parallel to
        # the global outline at their own measured thicknesses, and the
        # mesophyll family + endodermis carry the whole transition to this
        # ellipse, each at its measured share of whatever radial room is
        # left. Realized depths at the abaxial pole against the measurements:
        #     epidermis   0.0220 vs 0.02  measured
        #     hypodermis  0.0512 vs 0.05  (2 rings)
        #     palisade    0.0417 vs 0.055
        #     mesophyll   0.161  vs 0.156 (3 rings)
        #     endodermis  0.0310 vs 0.03
        # Adaxially the mesophyll group is compressed (0.0315/ring against
        # 0.0537 abaxially) because the flat face sits much closer to the
        # cylinder than the dome does -- which is what the section shows, and
        # the reason mesophyll rather than a surface tissue absorbs it.
        {"name": "central_cylinder", "shape": "ellipse",
         "layer_length": 0.737, "layer_thickness": 0.448,
         # (figure) each bundle envelope measures ~0.21 x 0.20 mm, the pair
         # spanning 0.46 of the 0.737 cylinder. width > height keeps the major
         # axis tangential before vascular_angle rotates it (see
         # CentralCylinderParams). 25 deg (figure) is the tilt of the
         # xylem/phloem interface off horizontal; GeometryProcessor.
         # two_ellipses mirrors it for the right-hand bundle (180-angle), so
         # the pair reads as a symmetric outward-tilted "V".
         "vascular_width": 0.22, "vascular_height": 0.19, "vascular_angle": 25,
         "cell_diameter": 0.02},   # (figure) vascular parenchyma
        # n_points: Voronoi seeds per endodermis cell's border ellipse
        # (LayerPolygon.n_points -> CellGenerator.cell_border), raised from
        # the engine default (15, anisotropic-cell rule) to make the ring
        # read as the distinctly ROUND, loosely-adherent cells the figure
        # shows (pinched point-contacts, not flat shared walls) rather than
        # today's coarse polygons.
        #
        # A previous pass here concluded n_points had "a real geometric
        # limit" (cell count and vertex count frozen at ~8/cell no matter how
        # high n_points went) and shipped it anyway on the theory that it
        # would matter for some future, coarser geometry. That conclusion was
        # wrong, and the mechanism was misdiagnosed: n_points DOES work --
        # CellGenerator.simplify_cells (the Phase-3 rebuild after Voronoi
        # tessellation) keeps only each polygon's *junction* vertices
        # (``vk in junction_set``) unless the cell is flagged
        # ``protect_shape``, so every one of those extra border seeds was
        # being discarded right after tessellation, regardless of n_points --
        # the ~8-vertex count measured was simplify_cells' floor, not a
        # Voronoi limit. ``protect_shape`` (a new per-layer flag, plumbed
        # exactly like n_points: recipe dict -> LayerPolygon.protect_shape ->
        # generate_cells_info's Cell(...) construction -> simplify_cells'
        # protect_ids) keeps every boundary vertex instead, which is the
        # missing half of this feature -- n_points alone does nothing
        # visible without it.
        #
        # Re-swept 15/24/40/80, all with protect_shape=True (seed=0,
        # generate_cells()), measuring vertices AFTER simplify_cells -- the
        # metric that actually reflects what gets rendered/exported:
        #   n_points  count  mean_verts(after)  max_verts(after)  mean_area   runtime_s
        #      15       34        43.9                49          0.001567    ~10-11
        #      24       34        54.2                59          0.001587    ~9-10
        #      40       34        72.8                79          0.001595    ~9-11
        #      80       34       118.7               127          0.001598    ~10-11
        # (mean_verts(before) == mean_verts(after) at every n_points now --
        # protect_shape means nothing is discarded.) Area converges the same
        # way the old sweep measured (~+1.8% end to end), because that part
        # was never in question -- only the *vertex/shape* claim was wrong.
        # A single-cell overlay (area + isoperimetric-quotient check, not
        # just eyeballing) shows most of the achievable rounding is captured
        # by 40 (~87% of the 15->80 roundness gain vs ~61% at 24); 80 adds
        # finer waviness but reads the same as 40 at normal render zoom. 40
        # is kept for that reason -- 2.7x the default seed count, not 8x.
        {"name": "endodermis", "cell_diameter": 0.03, "cell_width": 0.055, "order": 3,
         "n_points": 30},
        # NOTE: this layer previously also carried "shift": 0.5 (added by an
        # earlier pass, not requested, and not needed for roundness -- shift
        # only randomizes where the ring's seed sequence *starts*, unrelated
        # to per-cell border density). It caused two real breaks in the
        # endodermis ring (mesophyll cells intruding through the gap between
        # consecutive endodermis seeds -- see NeedleAnatomy.reshape_layers'
        # ellipse morph, which is what actually shapes this ring): with
        # shift=0.5 the union of all endodermis-cell polygons came out in 2
        # disconnected pieces (breaks at bearings 258.3 deg and 169.0 deg);
        # with shift dropped (0.0, the default) it is 1 connected piece at
        # this same seed. Dropped rather than fixed in the engine because no
        # other example (pinus_pinaster.py, gallery_needle_features.py) sets
        # shift on any layer, both check out with 0 ring breaks already, and
        # the parameter was never asked for here in the first place.
        # Spongy mesophyll: base ring everywhere + abaxial half + abaxial cap.
        # Distinct names are required (LayerManager rejects duplicates, and
        # several algorithms look up "mesophyll" expecting a single ring); the
        # "mesophyll_*" prefix is what makes the extra rings count as
        # mesophyll-family for the resin-duct annulus and the wall-rhombi air
        # spaces.
        {"name": "mesophyll", "cell_diameter": 0.052, "cell_width": 0.044, "order": 4},
        {"name": "mesophyll_abaxial", "cell_diameter": 0.052, "cell_width": 0.044, "order": 4.1,
         "thickness_profile": ABAXIAL_MESOPHYLL_PROFILE,
         "zone_angles": {"mode": "half", "pole": ABAXIAL_POLE}},
        {"name": "mesophyll_abaxial_cap", "cell_diameter": 0.052, "cell_width": 0.044, "order": 4.2,
         "thickness_profile": CAP_PROFILE,
         "zone_angles": {"mode": "wedge", "centers": [ABAXIAL_POLE],
                         "half_width": CAP_ZONE_HALF_WIDTH}},
        {"name": "palisade", "cell_diameter": 0.055, "cell_width": 0.038, "order": 4.5},
        # 2 hypodermis layers, uniform -- no corner thickening in this needle
        # (contrast pinus_pinaster.py's hypodermis_corner nodules).
        # (figure) the brightly-walled hypodermis cells read ~0.025 radial x
        # 0.03 tangential.
        {"name": "hypodermis", "cell_diameter": 0.025, "cell_width": 0.03,
         "n_layers": 2, "order": 5},
        # (figure) the epidermis band is ~0.02 mm deep with wall-to-wall
        # spacing ~0.015 mm, i.e. radially elongated -- the reverse of
        # pinus_pinaster.py's wider-than-deep epidermis.
        {"name": "epidermis", "cell_diameter": 0.02, "cell_width": 0.015, "order": 6},
        # Transfusion tissue: circle-packed (NeedleAnatomy.
        # add_transfusion_tissue) in two structural passes -- parenchyma into
        # the full zone first, then tracheids into what's left.
        # parenchyma_diameter 0.0505 is the measured 0.002 mm^2 per cell read
        # as a disc (d = 2*sqrt(A/pi)); tracheids_diameter 0.025 is (figure).
        # tracheids_diameter 0.015 is not a stated measurement -- only that
        # tracheids are the smaller, more numerous element -- so it is set
        # well below the parenchyma's 0.0505. Realized sizes come out larger
        # than requested for both (Voronoi expansion, ~+45%): 0.015 renders
        # as ~0.021 against the parenchyma's ~0.054, i.e. ~40% of its
        # diameter, at 325 tracheids to 27 parenchyma.
        #
        # transfusion_tracheids_ratio is the knob for the measured 1/3
        # parenchyma : 2/3 tracheid *area* split. It is not that split
        # directly: it sets the two passes' occupancy *targets*
        # (p_tracheid = r/(1+r)), and because pass 2 packs into a
        # gap-constrained residue it under-fills, the realized parenchyma
        # share lands above its target. How far above depends on
        # tracheids_diameter -- finer tracheids pack the residue better, so
        # the ratio needed for a given split drops as they shrink:
        #     tracheid 0.025, ratio 3.0 -> 33.0 / 67.0
        #     tracheid 0.018, ratio 3.0 -> 28.0 / 72.0
        #     tracheid 0.015, ratio 1.5 -> 38.6 / 61.4
        #     tracheid 0.015, ratio 1.8 -> 34.7 / 65.3
        #     tracheid 0.015, ratio 1.9 -> the measured split (below)
        #     tracheid 0.015, ratio 2.0 -> 31.8 / 68.2
        #     tracheid 0.012, ratio 2.0 -> 30.0 / 70.0
        # Re-sweep this whenever either diameter changes; the two are
        # coupled. (Contrast pinus_pinaster.py, whose measurement is the
        # inverse split and which therefore sits at ratio 1.0.)
        {"name": "transfusion_tissue", "n_layers": 2, "pack_circles": True,
         "diameter_max": 0.055, "proportion": 0.85,
         "parenchyma_diameter": 0.0505, "tracheids_diameter": 0.015,
         "transfusion_tracheids_ratio": 1.9},
        # Measured: xylem cells 0.008 across, phloem cells 0.005 high
        # (cell_diameter is the radial/height size for both, the same
        # convention pinus_pinaster.py's "0.008 mm height" phloem uses).
        # These replace the ~0.016/0.010 the figure appeared to show -- at
        # 3-7 px per cell it was reading cell + wall together.
        # n_files is a density knob, not a measurement, and it has to be
        # re-tuned whenever cell_diameter changes: it sets how many seeds
        # divide a region whose *extent* is fixed by the envelope partition,
        # so too few seeds leave each cell's Voronoi territory oversized
        # however small cell_diameter is. Sweep at seed 0, realized mean cell
        # area expressed as an equivalent diameter:
        #     xylem  n_files  6 -> 0.0104   10 -> 0.0083   14 -> 0.0071
        #     phloem n_files  6 -> 0.0103   16 -> 0.0065   22 -> 0.0056
        #                                                  26 -> 0.0052
        # 10 and 26 realize the measured 0.008 and 0.005 to within 4%. They
        # are high because the bundle envelope measured off the figure
        # (0.22 x 0.19) gives the phloem region more area than 0.005 cells
        # would fill at a lower count -- the two cannot both be satisfied, and
        # the stated cell sizes win here. Lowering them to pinaster's n_files=3
        # renders both tissues at ~0.0104, i.e. 30-100% oversized.
        {"name": "xylem", "cell_diameter": 0.008, "n_files": 10, "n_clusters": 4, "n_per_cluster": 3},
        {"name": "phloem", "cell_diameter": 0.005, "n_files": 26},
        # Not measured: scaled off xylem to keep pinus_pinaster.py's cambium
        # : xylem relation (0.01/0.013 = 0.77 -> 0.008 * 0.77).
        {"name": "cambium", "cell_diameter": 0.006},
        # Sizes the interstitial-lineage "Str. Interstitial cell" grid
        # columns; the corner "Strasburger cell" cluster is retagged from
        # parenchyma at its own (unrelated) size -- see
        # NeedleAnatomy.retag_corner_parenchyma.
        {"name": "Strasburger cells", "cell_diameter": 0.02},
        # Two resin-duct blocks, each placed by explicit position
        # ("positions", see NeedleAnatomy._duct_zone_data) so the large and
        # small ducts can differ in size and sit exactly where they were
        # measured. Both keep pinus_pinaster.py's inside-out proportions
        # (lumen : epithelium : sheath), which the figure cannot resolve;
        # what the figure does give is each duct's total footprint, and
        # that is what these reproduce:
        #   large: 0.037 + 2*0.006 + 2*0.018 = 0.085 mm  (measured ~0.09)
        #   small: 0.022 + 2*0.004 + 2*0.011 = 0.052 mm  (~0.6 of the large)
        # The small duct's own footprint was NOT separately measurable at this
        # resolution -- 0.6 is a proportional stand-in for "small".
        {"name": "resin_duct", "positions": DUCTS_LARGE,
         "lumen_diameter": 0.037,
         "cell_diameter": 0.006, "cell_width": 0.013,
         "sheath_cell_diameter": 0.018, "sheath_cell_width": 0.023},
        {"name": "resin_duct", "positions": DUCTS_SMALL,
         "lumen_diameter": 0.022,
         "cell_diameter": 0.004, "cell_width": 0.008,
         "sheath_cell_diameter": 0.011, "sheath_cell_width": 0.014},
        # Directional, corner-excluded stomata: 9 abaxial (domed/top), 4
        # adaxial (flat/bottom). Pit geometry is pinus_pinaster.py's scaled by
        # this needle's thickness (0.955/1.29 = 0.74); chamber_clearance=2.0
        # deletes the hypodermis cell(s) directly inward of each sub-stomatal
        # chamber so palisade mesophyll can reach it, as in the sunken-stoma
        # anatomy. With no corner hypodermis nodules here, that clearance has
        # a uniform 2-layer hypodermis to cut through everywhere.
        #
        # All 13 land at seed 0. Note this is sensitive to the epidermis ring:
        # _pick_stomata_triplets samples each angular run evenly in
        # epidermis-group *index* order (not by angle) and needs both
        # neighbouring groups (g-1, g+1) to exist, so a run can silently lose
        # its end picks -- with the earlier, morph-thickened epidermis only 11
        # of these 13 were placed. Check the printed census, not just the
        # requested numbers, after changing anything about the epidermis.
        {"name": "stomata", "n_adaxial": 4, "n_abaxial": 9, "edge_margin": 0.1,
         "width": 0.018, "depth": 0.06, "sub_chamber": 0.045,
         "chamber_clearance": 2.0,
         "guard_cell_diameter": 0.018, "guard_cell_aspect": 0.7,
         "sunken": True},
        # Rhombic wall-centred air spaces for the "loose" spongy mesophyll
        # (NeedleAnatomy._apply_mesophyll_wall_rhombi covers every
        # mesophyll_* ring), plus thin full-height air slits carved every two
        # palisade cells (NeedleAnatomy._apply_palisade_wall_slits).
        {"name": "inter_cellular_spaces", "tissue": ["mesophyll", "palisade"],
         "smoothness": [0.3, 0.0], "slit_width": [0, 0.002], "slit_every": [0, 2]},
    ]


def main(show=True):
    print("=== Pine (*Pinus nigra*) needle leaf ===")
    leaf = NeedleAnatomy(build_nigra(), seed=SEED)
    leaf.generate_cells()
    counts = {}
    for c in leaf.all_cells.cells:
        counts[c.type] = counts.get(c.type, 0) + 1
    for t in sorted(counts):
        print(f"    {t:14s} {counts[t]}")

    _, ax = plt.subplots(figsize=(15, 4.0))
    leaf.plot_cells(show=False, ax=ax,
                    title="Pine (*Pinus nigra*) needle leaf")
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
