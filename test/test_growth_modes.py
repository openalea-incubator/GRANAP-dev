"""Smoke + structure tests for the four canonical organ growth cases.

Covers the presets exposed by ``OrganInputData``:

    * monocot root            (``for_root``)
    * dicot root, primary     (``for_dicot_root``)
    * dicot root, secondary   (``for_dicot_secondary``)
    * dicot root, annual ring (``for_dicot_annual``)

Each case is built, generated, and checked for the tissues that *define* it
(rather than exact counts, which the golden suite pins).  Run as a script to
also render the four cross-sections side by side.
"""

import os
import sys

sys.path.append(os.path.abspath(".."))

from openalea.granap.root_class import RootAnatomy
from openalea.granap.input_data import OrganInputData

SEED = 0


# -- builders ----------------------------------------------------------------

def monocot() -> RootAnatomy:
    return RootAnatomy(OrganInputData.for_root(), seed=SEED)


def dicot_primary() -> RootAnatomy:
    return RootAnatomy(OrganInputData.for_dicot_root(), seed=SEED)


def dicot_secondary() -> RootAnatomy:
    return RootAnatomy(OrganInputData.for_dicot_secondary(), seed=SEED)


def dicot_annual() -> RootAnatomy:
    return RootAnatomy(OrganInputData.for_dicot_annual(), seed=SEED)


CASES = {
    "monocot":         monocot,
    "dicot_primary":   dicot_primary,
    "dicot_secondary": dicot_secondary,
    "dicot_annual":    dicot_annual,
}


# -- helpers -----------------------------------------------------------------

def _census(make_organ) -> dict:
    organ = make_organ()
    organ.generate_cells()
    counts: dict = {}
    for cell in organ.all_cells.cells:
        counts[cell.type] = counts.get(cell.type, 0) + 1
    return counts


# -- reproducibility ---------------------------------------------------------

def test_all_cases_reproducible():
    """Same preset + seed must yield an identical cell census."""
    for name, make in CASES.items():
        assert _census(make) == _census(make), f"{name} not reproducible at seed={SEED}"


# -- per-case structure ------------------------------------------------------

def test_monocot_structure():
    c = _census(monocot)
    # Common root layers.
    for t in ("epidermis", "exodermis", "cortex", "endodermis", "pericycle", "stele"):
        assert c.get(t, 0) > 0, f"monocot missing {t}"
    # Monocot vasculature (ring of metaxylem + protoxylem + phloem), no cambium.
    assert c.get("metaxylem", 0) > 0, "monocot missing metaxylem"
    assert c.get("phloem", 0) > 0, "monocot missing phloem"
    assert c.get("cambium", 0) == 0, "monocot should have no cambium"
    assert c.get("medullar_ray", 0) == 0, "monocot should have no medullar rays"


def test_dicot_primary_structure():
    c = _census(dicot_primary)
    for t in ("epidermis", "cortex", "endodermis", "pericycle", "stele"):
        assert c.get(t, 0) > 0, f"dicot_primary missing {t}"
    # Star xylem + primary phloem + primary cambium.
    assert c.get("xylem", 0) > 0, "dicot_primary missing xylem"
    assert c.get("phloem", 0) > 0, "dicot_primary missing phloem"
    assert c.get("cambium", 0) > 0, "dicot_primary missing cambium"

def test_dicot_secondary_structure():
    c = _census(dicot_secondary)
    # Secondary-growth signatures.
    for t in ("xylem", "phloem", "cambium"):
        assert c.get(t, 0) > 0, f"dicot_secondary missing {t}"


def test_dicot_annual_has_more_vessels_than_secondary():
    """N annual rings repeat the vessel packing, so annual has more xylem cells."""
    sec = _census(dicot_secondary)
    ann = _census(dicot_annual)
    assert ann.get("cambium", 0) > 0, "dicot_annual missing cambium"
    assert ann.get("xylem", 0) > sec.get("xylem", 0), (
        f"annual xylem ({ann.get('xylem', 0)}) should exceed "
        f"secondary xylem ({sec.get('xylem', 0)})"
    )
