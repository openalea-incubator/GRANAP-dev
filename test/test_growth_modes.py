"""Structure tests for the growth cases the golden suite does not pin.

``test_vascular_regression`` covers the monocot and dicot-primary presets by exact
census.  What is left here are the two presets it does not carry — ``for_dicot_secondary``
and ``for_woody_dicot`` — and the relation between them (woody repeats vessel packing,
so it must out-vessel the single-ring secondary case).
"""

import os
import sys

sys.path.append(os.path.abspath(".."))

from openalea.granap.root_class import RootAnatomy
from openalea.granap.input_data import OrganInputData

SEED = 0


# -- builders ----------------------------------------------------------------

def dicot_secondary() -> RootAnatomy:
    return RootAnatomy(OrganInputData.for_dicot_secondary(), seed=SEED)


def dicot_woody() -> RootAnatomy:
    return RootAnatomy(OrganInputData.for_woody_dicot(), seed=SEED)


# -- helpers -----------------------------------------------------------------

def _census(make_organ) -> dict:
    organ = make_organ()
    organ.generate_cells()
    counts: dict = {}
    for cell in organ.all_cells.cells:
        counts[cell.type] = counts.get(cell.type, 0) + 1
    return counts


# -- per-case structure ------------------------------------------------------
#
# Presence-of-tissue checks for the monocot and dicot-primary presets used to live
# here; ``test_vascular_regression`` pins their exact census (absences included), so
# they said nothing this file did not already get for free.  Reproducibility is
# likewise a property of the RNG plumbing, not of a preset — ``test_seed0_reproducible``
# covers it on a representative few.


_SECONDARY = None


def _secondary_census() -> dict:
    """The ``for_dicot_secondary()`` census, built once (both tests below read it)."""
    global _SECONDARY
    if _SECONDARY is None:
        _SECONDARY = _census(dicot_secondary)
    return _SECONDARY


def test_dicot_secondary_structure():
    c = _secondary_census()
    # Secondary-growth signatures.
    for t in ("xylem", "phloem", "cambium"):
        assert c.get(t, 0) > 0, f"dicot_secondary missing {t}"


def test_dicot_woody_has_more_vessels_than_secondary():
    """Woody growth thickens the stele and repeats vessel packing, so it has
    more xylem cells than the single-ring secondary case."""
    sec = _secondary_census()
    woody = _census(dicot_woody)
    assert woody.get("cambium", 0) > 0, "dicot_woody missing cambium"
    assert woody.get("xylem", 0) > sec.get("xylem", 0), (
        f"woody xylem ({woody.get('xylem', 0)}) should exceed "
        f"secondary xylem ({sec.get('xylem', 0)})"
    )
