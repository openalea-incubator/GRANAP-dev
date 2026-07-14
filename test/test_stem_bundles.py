"""Tests for stem vascular bundles and the hollow (fistular) pith.

Topology is asserted at the ``build_bundle`` level (fast, deterministic): a
bundle is built at ``(1, 0)`` oriented radially at ``theta=0``, so the *outer*
(surface-ward) direction is +x and "farther out" means a larger x.  This pins
the arrangement that defines each bundle type:

    collateral    -> phloem outer of xylem
    bicollateral  -> phloem on both radial sides of the xylem
    amphivasal    -> xylem rings a phloem core (xylem farther from centre)
    amphicribral  -> phloem rings a xylem core (phloem farther from centre)
    face (monocot)-> metaxylem outer of protoxylem (endarch), + a lacuna void

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
    assert _mean_x(cells, "phloem") > _mean_x(cells, "xylem"), "phloem must sit outer of xylem"
    assert any(c.type == "cambium" for c in cells.cells), "open collateral has a cambium strip"


def test_bicollateral_phloem_both_sides():
    cells, _ = _build(bundle_type="bicollateral", inner_phloem_fraction=0.2, xylem_layout="packed")
    xc = _mean_x(cells, "xylem")
    ph_x = [c.x for c in cells.cells if c.type == "phloem"]
    assert any(x > xc for x in ph_x) and any(x < xc for x in ph_x), \
        "bicollateral must have phloem on both radial sides of the xylem"


def test_amphivasal_xylem_rings_phloem():
    cells, _ = _build(bundle_type="concentric", concentric_type="amphivasal",
                      shape="circle", width=0.16, height=0.16)
    assert _mean_dist(cells, "xylem") > _mean_dist(cells, "phloem"), \
        "amphivasal: xylem ring is farther from the bundle centre than the phloem core"


def test_amphicribral_phloem_rings_xylem():
    cells, _ = _build(bundle_type="concentric", concentric_type="amphicribral",
                      shape="circle", width=0.16, height=0.16)
    assert _mean_dist(cells, "phloem") > _mean_dist(cells, "xylem"), \
        "amphicribral: phloem ring is farther from the bundle centre than the xylem core"


def test_face_metaxylem_outer_of_protoxylem_with_lacuna():
    cells, res = _build(bundle_type="collateral", has_cambium=False, xylem_layout="face",
                        lacuna=True, xylem_maturation="endarch")
    assert _mean_x(cells, "metaxylem") > _mean_x(cells, "protoxylem"), \
        "endarch face: metaxylem sits outer of protoxylem"
    assert len(res.cavity_polygons) == 1, "lacuna=True must carve one protoxylem lacuna void"


def test_sheath_produces_sclerenchyma():
    cells, _ = _build(sheath="both", xylem_layout="face", has_cambium=False)
    assert any(c.type == "sclerenchyma" for c in cells.cells), "sheath must place sclerenchyma cells"


# -- whole-organ smoke -------------------------------------------------------

def _census(organ):
    organ.generate_cells()
    c = {}
    for cell in organ.all_cells.cells:
        c[cell.type] = c.get(cell.type, 0) + 1
    return c


def test_dicot_eustele_generates():
    c = _census(StemAnatomy(OrganInputData.for_dicot_stem(), seed=SEED))
    for t in ("xylem", "phloem", "cambium", "pith", "cortex", "epidermis"):
        assert c.get(t, 0) > 0, f"dicot stem missing {t}"


def test_monocot_atactostele_generates():
    data = OrganInputData.for_monocot_stem()
    data.set_value("vascular_bundle", "n_bundles", 6)   # fewer -> faster test
    c = _census(StemAnatomy(data, seed=SEED))
    for t in ("metaxylem", "protoxylem", "sclerenchyma", "pith", "epidermis"):
        assert c.get(t, 0) > 0, f"monocot stem missing {t}"
    assert c.get("cambium", 0) == 0, "monocot bundles are closed (no cambium)"


def test_hollow_pith_leaves_cavity_empty():
    data = OrganInputData.for_monocot_stem()
    data.set_value("vascular_bundle", "n_bundles", 6)
    data.set_value("pith", "cavity_radius", 0.12)
    organ = StemAnatomy(data, seed=SEED)
    organ.generate_cells()
    pith_r = [np.hypot(c.x, c.y) for c in organ.all_cells.cells if c.type == "pith"]
    assert pith_r, "expected pith cells outside the cavity"
    assert min(pith_r) >= 0.12 * 0.9, "no pith cells should fall inside the medullary cavity"
