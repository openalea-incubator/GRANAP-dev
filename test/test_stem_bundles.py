"""Tests for stem vascular bundles and the hollow (fistular) pith.

Topology is asserted at the ``build_bundle`` level (fast, deterministic): a
bundle is built at ``(1, 0)`` oriented radially at ``theta=0``, so the *outer*
(surface-ward) direction is +x and "farther out" means a larger x.  This pins
the arrangement that defines each bundle type:

    collateral    -> phloem outer of xylem
    bicollateral  -> phloem on both radial sides of the xylem
    amphivasal    -> xylem rings a phloem core (xylem farther from centre)
    amphicribral  -> phloem rings a xylem core (phloem farther from centre)
    face (monocot)-> metaxylem outer of protoxylem, + a lacuna void

Two whole-organ tests then confirm the monocot atactostele preset generates and
that a hollow pith leaves the centre empty.  (The dicot eustele is not re-checked
here: ``test_vascular_regression.test_dicot_stem_golden`` builds the same preset at
the same seed and pins its exact census.)
"""

import os
import sys

import numpy as np

sys.path.append(os.path.abspath(".."))

from openalea.granap.cell_manager import CellManager
from openalea.granap.vascular_bundle import build_bundle
from openalea.granap.stem_class import StemAnatomy
from openalea.granap.input_data import OrganInputData

SEED = 0
CX, CY = 1.0, 0.0          # bundle centre; theta=0 -> outer direction is +x


def _params(**over):
    """Bundle + cell-level param dicts from the dicot preset, with overrides."""
    by_name = {p["name"]: dict(p) for p in OrganInputData.for_dicot_stem().to_dict_list()}
    bp = by_name["vascular_bundle"]
    bp.update(over)
    return bp, by_name["xylem"], by_name["phloem"], by_name.get("cambium", {})


def _build(**over):
    bp, xylem, phloem, cambium = _params(**over)
    cells = CellManager()
    res = build_bundle(cells, np.random.default_rng(SEED), CX, CY, 0.0, bp, xylem, phloem, cambium)
    return cells, res


def _mean_x(cells, tag):
    xs = [c.x for c in cells.cells if c.type == tag]
    return float(np.mean(xs)) if xs else None


def _mean_dist(cells, tag):
    d = [np.hypot(c.x - CX, c.y - CY) for c in cells.cells if c.type == tag]
    return float(np.mean(d)) if d else None


# -- bundle topology ---------------------------------------------------------

def test_collateral_phloem_outer_of_xylem():
    cells, _ = _build(bundle_type="collateral", has_cambium=True, xylem_layout="packed")
    assert _mean_x(cells, "sieve element") > _mean_x(cells, "xylem"), "phloem must sit outer of xylem"
    assert any(c.type == "cambium" for c in cells.cells), "open collateral has a cambium strip"


def test_bicollateral_phloem_both_sides():
    cells, _ = _build(bundle_type="bicollateral", inner_phloem_fraction=0.2, xylem_layout="packed")
    xc = _mean_x(cells, "xylem")
    ph_x = [c.x for c in cells.cells if c.type == "sieve element"]
    assert any(x > xc for x in ph_x) and any(x < xc for x in ph_x), \
        "bicollateral must have phloem on both radial sides of the xylem"


def test_amphivasal_xylem_rings_phloem():
    cells, _ = _build(bundle_type="concentric", concentric_type="amphivasal",
                      shape="circle", width=0.16, height=0.16)
    assert _mean_dist(cells, "xylem") > _mean_dist(cells, "sieve element"), \
        "amphivasal: xylem ring is farther from the bundle centre than the phloem core"


def test_amphicribral_phloem_rings_xylem():
    cells, _ = _build(bundle_type="concentric", concentric_type="amphicribral",
                      shape="circle", width=0.16, height=0.16)
    assert _mean_dist(cells, "sieve element") > _mean_dist(cells, "xylem"), \
        "amphicribral: phloem ring is farther from the bundle centre than the xylem core"


def test_face_bundle_has_xylem_and_lacuna():
    # The face bundle tags metaxylem + protoxylem alike as 'xylem'; its larger
    # (metaxylem) vessels sit outer of the smaller (protoxylem) ones, and lacuna=True
    # drops an air-space void in the inner half.
    cells, res = _build(bundle_type="collateral", has_cambium=False, xylem_layout="face",
                        lacuna=True)
    xyl = [c for c in cells.cells if c.type == "xylem"]
    assert xyl, "face bundle must place xylem vessels"
    big = [c.x for c in xyl if c.diameter >= 0.025]     # metaxylem-scale
    small = [c.x for c in xyl if c.diameter < 0.025]    # protoxylem-scale
    if big and small:
        assert np.mean(big) > np.mean(small), "larger (metaxylem) vessels sit outer"
    assert any(c.type == "air space" for c in cells.cells), \
        "lacuna=True must place an air-space lacuna cell"


def test_sheath_produces_sclerenchyma():
    cells, _ = _build(sheath="both", xylem_layout="face", has_cambium=False)
    assert any(c.type == "sclerenchyma" for c in cells.cells), "sheath must place sclerenchyma cells"


# -- asymmetric fibre caps (n_caps_layers_outward / _inward) -----------------
# The caps extend the bundle *outside* the envelope; outer = +x at theta=0.

def test_no_caps_place_no_fibres():
    cells, _ = _build(sheath="none", n_caps_layers_outward=0, n_caps_layers_inward=0)
    assert not [c for c in cells.cells if c.type == "sclerenchyma"], \
        "sheath 'none' + no caps must place no sclerenchyma fibres"


def test_outward_cap_fibres_sit_outer():
    cells, res = _build(sheath="none", n_caps_layers_outward=3, n_caps_layers_inward=0)
    fib = [c for c in cells.cells if c.type == "sclerenchyma"]
    assert fib, "an outward cap must place sclerenchyma fibres"
    assert _mean_x(cells, "sclerenchyma") > CX, "outward-pole cap fibres sit outer of the bundle centre (+x)"
    assert res.envelope.bounds[2] > CX, "the cap extends the envelope (mask) outward"


def test_inward_cap_fibres_sit_inner():
    cells, _ = _build(sheath="none", n_caps_layers_outward=0, n_caps_layers_inward=3)
    fib = [c for c in cells.cells if c.type == "sclerenchyma"]
    assert fib, "an inward cap must place sclerenchyma fibres"
    assert _mean_x(cells, "sclerenchyma") < CX, "inward-pole cap fibres sit inner of the bundle centre (-x)"


def test_caps_are_asymmetric_and_scale_with_count():
    # 4 outward vs 2 inward layers: both poles seed fibres, and the footprint
    # (removal mask) extends farther out than in — depth scales with the count.
    cells, res = _build(sheath="none", n_caps_layers_outward=4, n_caps_layers_inward=2)
    fib = [c for c in cells.cells if c.type == "sclerenchyma"]
    assert any(c.x > CX for c in fib) and any(c.x < CX for c in fib), "both poles must be capped"
    x0, _, x1, _ = res.envelope.bounds
    assert (x1 - CX) > (CX - x0), \
        "the 4-layer outward cap must extend the bundle farther than the 2-layer inward cap"


# -- tapered outward cap (n_caps_layers_outward_flank) ------------------------
# At theta=0 the outward pole is +x and the flanks are the +/-y sides.

def _fibre_xy(cells):
    return np.array([(c.x, c.y) for c in cells.cells if c.type == "sclerenchyma"])


def test_outward_flank_unset_or_equal_is_the_uniform_cap():
    # None (default) and a flank count equal to the pole count are both "no taper":
    # the same fibres in the same places as the plain uniform cap.
    ref = _fibre_xy(_build(sheath="none", n_caps_layers_outward=4)[0])
    for flank in (None, 4):
        got = _fibre_xy(_build(sheath="none", n_caps_layers_outward=4,
                               n_caps_layers_outward_flank=flank)[0])
        assert got.shape == ref.shape and np.allclose(got, ref), \
            f"flank={flank} must reproduce the uniform 4-layer cap"


def test_outward_flank_above_pole_is_clamped():
    ref = _fibre_xy(_build(sheath="none", n_caps_layers_outward=3)[0])
    got = _fibre_xy(_build(sheath="none", n_caps_layers_outward=3,
                           n_caps_layers_outward_flank=7)[0])
    assert got.shape == ref.shape and np.allclose(got, ref), \
        "a flank count above the pole count is clamped to it (uniform cap)"


def test_tapered_cap_keeps_pole_depth_and_thins_at_flanks():
    # 5 layers over the pole, 1 at the flanks: the cap reaches as far out at the
    # pole as the uniform 5-layer cap, but holds fewer fibres and stays thin on the
    # tangential sides.
    _, uni = _build(sheath="none", n_caps_layers_outward=5)
    _, tap = _build(sheath="none", n_caps_layers_outward=5, n_caps_layers_outward_flank=1)
    uni_cells, _ = _build(sheath="none", n_caps_layers_outward=5)
    tap_cells, _ = _build(sheath="none", n_caps_layers_outward=5, n_caps_layers_outward_flank=1)
    n_uni, n_tap = len(_fibre_xy(uni_cells)), len(_fibre_xy(tap_cells))
    assert 0 < n_tap < n_uni, "the tapered cap must hold fewer fibres than the uniform one"

    _, uy0, ux1, uy1 = uni.envelope.bounds
    _, ty0, tx1, ty1 = tap.envelope.bounds
    scl = _params()[0]["sclerenchyma_cell_diameter"]
    assert abs(tx1 - ux1) < 0.5 * scl, "the pole (+x) keeps the full cap depth"
    assert (ty1 - ty0) < (uy1 - uy0) - scl, "the flanks (+/-y) carry a thinner cap"


def test_tapered_cap_layer_count_follows_the_angle():
    # Layer i survives where the linearly interpolated count rounds above it:
    # flank 2, pole 5 -> 2 / 3 / 4 / 5 layers at 0 / 30 / 60 / 90 degrees.
    from openalea.granap.vascular_bundle import _cap_layer_threshold
    def count(deg):
        a = np.radians(deg)
        return sum(1 for i in range(5) if a >= _cap_layer_threshold(i, 5, 2))
    assert [count(d) for d in (0, 30, 60, 90)] == [2, 3, 4, 5]


# -- whole-organ smoke -------------------------------------------------------

def _census(organ):
    organ.generate_cells()
    c = {}
    for cell in organ.all_cells.cells:
        c[cell.type] = c.get(cell.type, 0) + 1
    return c


def test_monocot_atactostele_generates():
    data = OrganInputData.for_monocot_stem()
    data.set_value("vascular_bundle", "n_bundles", 6)   # fewer -> faster test
    c = _census(StemAnatomy(data, seed=SEED))
    for t in ("xylem", "sclerenchyma", "parenchyma", "epidermis"):
        assert c.get(t, 0) > 0, f"monocot stem missing {t}"
    assert c.get("cambium", 0) == 0, "monocot bundles are closed (no cambium)"


def test_hollow_pith_leaves_cavity_empty():
    from shapely.geometry import Point
    data = OrganInputData.for_monocot_stem()
    data.set_value("vascular_bundle", "n_bundles", 6)
    data.set_value("pith", "cavity_radius", 0.12)
    organ = StemAnatomy(data, seed=SEED)
    organ.generate_cells()
    # The hollow (fistular) cavity stays a true void: the pith aerenchyma zone is
    # an annulus with a hole there, so no cell polygon covers the centre.
    center = Point(0.0, 0.0)
    covering = [c for c in organ.all_cells.cells
                if c.polygon is not None and c.polygon.contains(center)]
    assert not covering, "no cell should fill the hollow medullary cavity"


def test_outline_ring_follows_a_square_stem():
    # ring_shape='outline' lays the bundle ring on the pith/cortex boundary itself,
    # so on a square stem the bundles follow the square into its corners instead of
    # sitting on a circle of the same area, and every bundle cell stays inside.
    from shapely.geometry import Point
    data = OrganInputData.for_dicot_stem()
    data.set_value("vascular_bundle", "ring_shape", "outline")
    data.params.append({"name": "base_shape", "shape": "square", "width": 1.4})
    organ = StemAnatomy(data, seed=SEED)
    organ.generate_cells()
    outline = organ.generate_base_shape()
    vascular = [c for c in organ.all_cells.cells
                if c.type in ("xylem", "sieve element", "cambium")]
    assert vascular, "the square stem should carry bundles"
    assert all(outline.contains(Point(c.x, c.y)) for c in vascular)
    # The corners are farther from the centre than the side middles: the outline
    # ring reaches ~0.85 here, the default circle ring stops at ~0.69.
    r = max(np.hypot(c.x - outline.centroid.x, c.y - outline.centroid.y) for c in vascular)
    assert r > 0.7, f"bundles should follow the square into its corners (max r = {r:.2f})"
