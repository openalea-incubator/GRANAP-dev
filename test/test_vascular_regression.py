"""Golden regression tests for the root anatomy pipeline.

These pin the exact ``seed=0`` cell-type census for the canonical root
configurations.  They are the safety net the tissue refactor relied on: any
change that alters the produced anatomy (intended or not) will trip these, so an
intended change is a deliberate golden update rather than a silent drift.

Reproducibility note: full ``seed=0`` determinism depends on the Voronoi jitter
drawing from the organ's seeded ``self.rng`` (see ``Cell.jitter`` /
``CellGenerator.voronoi_diagram``).  Before that fix, dicot secondary growth
drifted run-to-run because the jitter used the global ``np.random``.
"""

import os
import sys

sys.path.append(os.path.abspath(".."))

from openalea.granap.root_class import RootAnatomy
from openalea.granap.input_data import OrganInputData

SEED = 0


# -- canonical configurations ------------------------------------------------

def monocot_default() -> OrganInputData:
    return OrganInputData.for_root()


def monocot_star() -> OrganInputData:
    data = OrganInputData.for_root()
    data.set_value("xylem", "xylem_shape", "star")
    return data


def monocot_star_pith() -> OrganInputData:
    data = monocot_star()
    data.set_value("xylem", "pith_radius", 0.05)
    return data


def dicot_primary() -> OrganInputData:
    return OrganInputData.for_dicot_root()


def dicot_secondary() -> OrganInputData:
    data = OrganInputData.for_dicot_root()
    data.set_value("secondary_growth", "value", True)
    return data


# -- golden census (seed=0) --------------------------------------------------

GOLDEN = {
    "monocot_default": (monocot_default, {
        "air space": 359, "cortex": 202, "endodermis": 31, "epidermis": 168,
        "exodermis": 79, "metaxylem": 5, "pericycle": 93, "phloem": 10,
        "protoxylem": 10, "stele": 341,
    }),
    "monocot_star": (monocot_star, {
        "air space": 359, "cortex": 202, "endodermis": 31, "epidermis": 168,
        "exodermis": 79, "pericycle": 93, "phloem": 5, "stele": 217, "xylem": 25,
    }),
    "monocot_star_pith": (monocot_star_pith, {
        "air space": 359, "cortex": 202, "endodermis": 31, "epidermis": 168,
        "exodermis": 79, "pericycle": 93, "phloem": 5, "stele": 249, "xylem": 23,
    }),
    "dicot_primary": (dicot_primary, {
        "air space": 601, "cambium": 81, "cortex": 349, "endodermis": 70,
        "epidermis": 247, "exodermis": 118, "pericycle": 223, "phloem": 51,
        "stele": 860, "xylem": 22,
    }),
    "dicot_secondary": (dicot_secondary, {
        "air space": 537, "cambium": 89, "cortex": 338, "endodermis": 70,
        "epidermis": 247, "exodermis": 118, "medullar_ray": 48, "pericycle": 223,
        "phloem": 143, "phloem_parenchyma": 133, "stele": 425, "xylem": 48,
    }),
}


def _census(make_input) -> dict:
    root = RootAnatomy(make_input(), seed=SEED)
    root.generate_cells()
    counts: dict = {}
    for cell in root.all_cells.cells:
        counts[cell.type] = counts.get(cell.type, 0) + 1
    return counts


def _check(name: str) -> None:
    make_input, expected = GOLDEN[name]
    got = _census(make_input)
    assert got == expected, (
        f"{name}: anatomy census drifted from golden.\n"
        f"  expected: {dict(sorted(expected.items()))}\n"
        f"  got:      {dict(sorted(got.items()))}"
    )


def test_seed0_reproducible():
    """Two builds of the same config must be identical (no global-RNG leakage)."""
    for name, (make_input, _) in GOLDEN.items():
        assert _census(make_input) == _census(make_input), f"{name} not reproducible"


def test_monocot_default_golden():
    _check("monocot_default")


def test_monocot_star_golden():
    _check("monocot_star")


def test_monocot_star_pith_golden():
    _check("monocot_star_pith")


def test_dicot_primary_golden():
    _check("dicot_primary")


def test_dicot_secondary_golden():
    _check("dicot_secondary")


if __name__ == "__main__":
    test_seed0_reproducible()
    for key in GOLDEN:
        _check(key)
        print("OK", key)
    print("ALL VASCULAR REGRESSION GOLDENS PASSED")
