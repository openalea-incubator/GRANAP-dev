"""Checks for the leaf cross-section (LeafAnatomy: monocot slab + dicot)."""

import numpy as np
from shapely.geometry import LineString

from openalea.granap.input_data import OrganInputData
from openalea.granap.leaf_class import (
    LeafAnatomy, MonocotLeafAnatomy, DicotLeafAnatomy,
)


def _dicot_input():
    return OrganInputData.for_dicot_leaf()


def _build():
    leaf = LeafAnatomy(seed=0)
    leaf.generate_cells()
    return leaf


def test_monocot_is_uniform_mesophyll():
    """Monocot leaf: epidermis + one uniform mesophyll tissue (no palisade/spongy)."""
    leaf = _build()
    cells = [c for c in leaf.all_cells.cells if c.polygon is not None]
    assert len(cells) > 100
    types = {c.type for c in cells}
    assert {"epidermis", "mesophyll"} <= types
    assert "palisade" not in types and "spongy" not in types


def test_bands_stack_epidermis_out_mesophyll_in():
    """Down the central column the epidermis (outer) wraps the mesophyll (inner),
    i.e. |y| is larger for epidermis than mesophyll."""
    leaf = _build()
    # A central column, away from the tapered tips where epidermis wraps around.
    central = [c for c in leaf.all_cells.cells
               if c.polygon is not None and abs(c.x) < 0.5]

    def med_absy(t):
        v = [abs(c.y) for c in central if c.type == t]
        return float(np.median(v)) if v else -1.0

    assert med_absy("epidermis") > med_absy("mesophyll") >= 0.0


def test_slab_is_gap_free_after_fuse():
    leaf = _build()
    total_gap = sum(g.area for g in leaf.find_gaps())
    assert total_gap < 0.01


def test_vein_row_is_transverse_xylem_adaxial():
    """An even row of veins, each xylem (adaxial/up) above phloem (abaxial/down)."""
    leaf = _build()
    assert len(leaf.vascular_tissue_polygons.get("bundle", [])) == 7
    cells = leaf.all_cells.cells
    xylem = [c.y for c in cells if c.type == "xylem"]
    sieve = [c.y for c in cells if c.type == "sieve element"]
    assert xylem and sieve
    assert np.median(xylem) > np.median(sieve)      # xylem sits adaxial of the phloem


def test_stomata_on_both_epidermes():
    """Monocot leaf is amphistomatous — stomata on both faces."""
    leaf = _build()
    guard = [c.y for c in leaf.all_cells.cells if c.type == "guard cell"]
    pores = [c for c in leaf.all_cells.cells if c.type == "pore"]
    assert guard and pores
    assert any(y > 0 for y in guard)                 # adaxial (upper)
    assert any(y < 0 for y in guard)                 # abaxial (lower)


def test_veins_rib_thickens_the_leaf():
    """Each vein thickens the outline into a rib: the lamina is thicker over a vein
    than between two veins."""
    leaf = LeafAnatomy(seed=0)
    outline = leaf.generate_base_shape()
    xs = leaf._vein_x_positions()
    miny, maxy = outline.bounds[1], outline.bounds[3]

    def thickness(x):
        seg = LineString([(x, miny - 1), (x, maxy + 1)]).intersection(outline)
        return (seg.bounds[3] - seg.bounds[1]) if not seg.is_empty else 0.0

    k = len(xs) // 2
    at_vein = thickness(xs[k])
    between = thickness(0.5 * (xs[k] + xs[k + 1]))
    assert at_vein > between + 0.03          # a visible rib


def test_veins_have_a_lacuna_air_space():
    """The 'face' veins carry a protoxylem lacuna (air space) near the mid-plane."""
    leaf = _build()
    # Air spaces near y~0 are vein lacunae (substomatal chambers sit near the faces).
    mid_air = [c for c in leaf.all_cells.cells
               if c.type == "air space" and abs(c.y) < 0.1]
    assert len(mid_air) >= 1


# ---------------------------------------------------------------------------
# Dicot leaf (dorsiventral)
# ---------------------------------------------------------------------------

def _build_dicot():
    leaf = LeafAnatomy(_dicot_input(), seed=0)
    leaf.generate_cells()
    return leaf


def test_factory_dispatches_on_planttype():
    assert isinstance(LeafAnatomy(seed=0), MonocotLeafAnatomy)          # default = monocot
    assert isinstance(LeafAnatomy(_dicot_input(), seed=0), DicotLeafAnatomy)


def test_dicot_is_dorsiventral():
    """Palisade under the adaxial (upper) epidermis, spongy above the abaxial one."""
    leaf = _build_dicot()
    pal = [c.y for c in leaf.all_cells.cells if c.type == "palisade"]
    spo = [c.y for c in leaf.all_cells.cells if c.type == "spongy"]
    assert pal and spo
    assert np.median(pal) > 0.0 > np.median(spo)


def test_dicot_stomata_denser_abaxial():
    leaf = _build_dicot()
    guard = [c.y for c in leaf.all_cells.cells if c.type == "guard cell"]
    adaxial = sum(1 for y in guard if y > 0)
    abaxial = sum(1 for y in guard if y < 0)
    assert adaxial > 0                 # still some on the upper face
    assert abaxial > adaxial           # denser below


def test_dicot_gap_free():
    leaf = _build_dicot()
    assert sum(g.area for g in leaf.find_gaps()) < 0.01


def _blunt_margin_input():
    """Dicot leaf ending in a blunt rounded margin instead of a thin taper.

    Full thickness out to 90 % of the half-width, then a short rounded edge, so the
    surface curves sharply where it wraps the margin — the shape that leaves a fat
    uncovered ribbon there.
    """
    params = OrganInputData.for_dicot_leaf().to_dict_list()
    plant = next(p for p in params if p["name"] == "planttype")
    half, thickness = plant["width"] / 2.0, 0.45
    plant["thickness_profile"] = [[0.0, thickness], [0.9 * half, thickness],
                                  [half, 0.0]]
    plant["edge_radius"] = thickness / 2.0
    return params


def test_fusion_fills_enclosed_holes_but_not_the_surface_ribbon():
    """``fuse_gaps`` absorbs the holes the tissue closes around and leaves the rest.

    The tessellation always leaves a thin uncovered ribbon between the outermost
    Voronoi edges and the organ outline, which runs unbroken past many cells wherever
    the surface curves sharply.  Fusing that ribbon stretches a single epidermis cell
    right across its neighbours, so ``FUSE_ENCLOSED_GAPS_ONLY`` keeps fusion to the
    enclosed holes — which must still all be filled.
    """
    def longest_epidermis(leaf):
        return max(_longest_extent(c.polygon) for c in leaf.all_cells.cells
                   if c.type == "epidermis" and c.polygon is not None)

    fused = LeafAnatomy(_blunt_margin_input(), seed=0)
    fused.generate_cells()
    assert fused.find_gaps(enclosed_only=True) == []

    raw = LeafAnatomy(_blunt_margin_input(), seed=0)
    raw.AUTO_FUSE_GAPS = False           # the tessellation, ribbon and all
    raw.generate_cells()

    # Growing into an enclosed hole stretches a bordering cell a little; growing
    # along the ribbon would stretch one across a dozen neighbours (it used to reach
    # 2.2x here, and 12x on a leaf with a blunter margin).
    assert longest_epidermis(fused) < 1.5 * longest_epidermis(raw)


def _longest_extent(poly):
    """Length of the long side of a cell's minimum rotated rectangle."""
    xs, ys = poly.minimum_rotated_rectangle.exterior.coords.xy
    return max(np.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i]) for i in range(4))


def test_dicot_vein_is_collateral_with_cambium():
    """A dicot leaf vein is the collateral (xylem/cambium/phloem) bundle, not the
    monocot 'face' bundle — so it has a cambium and no protoxylem lacuna."""
    leaf = _build_dicot()
    types = {c.type for c in leaf.all_cells.cells}
    assert "cambium" in types
    # No lacuna inside the vein: the midrib column (x~0) is solid xylem/cambium, so
    # there is no air space there (the spongy intercellular air sits between veins).
    vein_air = [c for c in leaf.all_cells.cells
                if c.type == "air space" and abs(c.x) < 0.08 and abs(c.y) < 0.10]
    assert len(vein_air) == 0


def test_dicot_has_vein_size_classes():
    """The dicot leaf carries >1 vein size-class, with the widest (midrib) centred."""
    leaf = LeafAnatomy(_dicot_input(), seed=0)
    layout = leaf._vein_layout()
    widths = sorted({round(float(s.get("width", 0.12)), 3) for _, s in layout})
    assert len(widths) >= 2
    x_widest, _ = max(layout, key=lambda t: float(t[1].get("width", 0.12)))
    assert abs(x_widest) < 0.05                # midrib at the centre


# ---------------------------------------------------------------------------
# Presets + create_from_input routing (Phase 5)
# ---------------------------------------------------------------------------

def test_leaf_presets_and_routing():
    from openalea.granap.input_data import OrganInputData
    from openalea.granap.organ_class import Organ

    mono = OrganInputData.for_monocot_leaf()
    dico = OrganInputData.for_dicot_leaf()
    # presets dispatch through the LeafAnatomy factory
    assert isinstance(LeafAnatomy(mono), MonocotLeafAnatomy)
    assert isinstance(LeafAnatomy(dico), DicotLeafAnatomy)
    # Organ.create_from_input routes the 'leaf' organ tag to LeafAnatomy
    assert isinstance(Organ.create_from_input(mono), MonocotLeafAnatomy)
    assert isinstance(Organ.create_from_input(dico), DicotLeafAnatomy)


def test_dicot_palisade_is_elongated():
    """Palisade cells are columnar — taller (y) than wide (x) by default."""
    leaf = _build_dicot()
    ratios = []
    for c in leaf.all_cells.cells:
        if c.type == "palisade" and c.polygon is not None:
            mnx, mny, mxx, mxy = c.polygon.bounds
            w = mxx - mnx
            if w > 0:
                ratios.append((mxy - mny) / w)
    assert ratios and np.median(ratios) > 1.8      # clearly elongated vertically


def test_dicot_spongy_has_intercellular_air():
    """Spongy mesophyll carries lots of intercellular air (ICS module)."""
    leaf = _build_dicot()
    spongy = [c for c in leaf.all_cells.cells if c.type == "spongy"]
    air = [c for c in leaf.all_cells.cells if c.type == "air space"]
    assert spongy and len(air) > 20


def test_monocot_inter_bundle_aerenchyma():
    """Monocot mesophyll carries an aerenchyma lacuna between each pair of veins."""
    leaf = _build()
    xs = sorted(leaf._vein_x_positions())
    mids = [0.5 * (a + b) for a, b in zip(xs[:-1], xs[1:])]
    # Inter-bundle lacunae are the big air spaces on the mid-plane (substomatal
    # chambers are smaller and sit near the faces).
    big = [c for c in leaf.all_cells.cells
           if c.type == "air space" and c.polygon is not None
           and c.polygon.area > 0.005 and abs(c.y) < 0.06]
    assert len(big) >= len(mids) - 1               # ~one lacuna per inter-vein gap
    for c in big:                                   # each sits between two veins
        assert min(abs(c.x - xm) for xm in mids) < 0.12
