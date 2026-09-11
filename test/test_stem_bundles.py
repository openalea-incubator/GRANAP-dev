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

Two whole-organ smoke tests then confirm the dicot eustele and monocot
atactostele presets generate, and that a hollow pith leaves the centre empty.
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


# -- whole-organ smoke -------------------------------------------------------

def _census(organ):
    organ.generate_cells()
    c = {}
    for cell in organ.all_cells.cells:
        c[cell.type] = c.get(cell.type, 0) + 1
    return c


def test_dicot_eustele_generates():
    c = _census(StemAnatomy(OrganInputData.for_dicot_stem(), seed=SEED))
    # Central ground tissue is tagged 'parenchyma' (the stem pith); xylem is a single
    # 'xylem' tag (no metaxylem/protoxylem split).
    for t in ("xylem", "sieve element", "cambium", "parenchyma", "cortex", "epidermis"):
        assert c.get(t, 0) > 0, f"dicot stem missing {t}"
    assert c.get("aerenchyma", 0) == 0, "a plain dicot stem must not default to aerenchyma"


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
