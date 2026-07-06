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
from openalea.granap.needle_class import NeedleAnatomy
from openalea.granap.input_data import OrganInputData

SEED = 0


# -- canonical configurations ------------------------------------------------
#
# Each builder returns a *constructed organ* (seeded), so the suite spans both
# RootAnatomy and NeedleAnatomy.  Build a fresh organ every call — generation
# mutates it.

def monocot_default() -> RootAnatomy:
    return RootAnatomy(OrganInputData.for_root(), seed=SEED)


def monocot_arch() -> RootAnatomy:
    data = OrganInputData.for_root()
    data.set_value("xylem", "xylem_shape", "arch")
    return RootAnatomy(data, seed=SEED)


def dicot_primary() -> RootAnatomy:
    return RootAnatomy(OrganInputData.for_dicot_root(), seed=SEED)


def dicot_secondary() -> RootAnatomy:
    data = OrganInputData.for_dicot_root()
    data.set_value("secondary_growth", "value", True)
    return RootAnatomy(data, seed=SEED)


def needle_default() -> NeedleAnatomy:
    return NeedleAnatomy(OrganInputData.for_needle(), seed=SEED)


def needle_features() -> NeedleAnatomy:
    """Needle with extra resin ducts and stomata (matches ``test_needle``)."""
    data = OrganInputData.for_needle()
    data.set_value("resin_duct", "n_files", 2)
    data.set_value("stomata", "n_files", 10)
    return NeedleAnatomy(data, seed=SEED)


# -- golden census (seed=0) --------------------------------------------------

GOLDEN = {
    "monocot_default": (monocot_default, {
        "air space": 367, "cortex": 206, "endodermis": 32, "epidermis": 168,
        "exodermis": 79, "metaxylem": 5, "pericycle": 97, "phloem": 10,
        "protoxylem": 10, "stele": 410,
    }),
    "dicot_primary": (dicot_primary, {
        "air space": 621, "cambium": 81, "cortex": 355, "endodermis": 72,
        "epidermis": 248, "exodermis": 119, "pericycle": 230, "phloem": 51,
        "stele": 1026, "xylem": 31,
    }),
    "dicot_secondary": (dicot_secondary, {
        "air space": 573, "cambium": 99, "companion_cell": 43, "cortex": 332,
        "endodermis": 51, "epidermis": 248, "exodermis": 119, "pericycle": 139,
        "phloem": 73, "stele": 1034, "xylem": 47,
    }),
    "needle_default": (needle_default, {
        "Strasburger cell": 38, "air space": 312, "cambium": 58, "duct": 3,
        "endodermis": 49, "epidermis": 231, "guard cell": 8, "hypodermis": 387,
        "mesophyll": 228, "parenchyma": 244, "phloem": 310, "pore": 4,
        "resin duct": 42, "transfusion": 103, "xylem": 270,
    }),
    "needle_features": (needle_features, {
        "Strasburger cell": 38, "air space": 327, "cambium": 58, "duct": 2,
        "endodermis": 49, "epidermis": 219, "guard cell": 20, "hypodermis": 366,
        "mesophyll": 230, "parenchyma": 244, "phloem": 310, "pore": 10,
        "resin duct": 28, "transfusion": 103, "xylem": 270,
    }),
}


def _census(make_organ) -> dict:
    organ = make_organ()
    organ.generate_cells()
    counts: dict = {}
    for cell in organ.all_cells.cells:
        counts[cell.type] = counts.get(cell.type, 0) + 1
    return counts


def _check(name: str) -> None:
    make_organ, expected = GOLDEN[name]
    got = _census(make_organ)
    assert got == expected, (
        f"{name}: anatomy census drifted from golden.\n"
        f"  expected: {dict(sorted(expected.items()))}\n"
        f"  got:      {dict(sorted(got.items()))}"
    )


def test_seed0_reproducible():
    """Two builds of the same config must be identical (no global-RNG leakage)."""
    for name, (make_organ, _) in GOLDEN.items():
        assert _census(make_organ) == _census(make_organ), f"{name} not reproducible"


def test_monocot_default_golden():
    _check("monocot_default")


def test_monocot_arch_reproducible():
    """Arch mode builds vessels and is reproducible (no global-RNG leakage)."""
    a = _census(monocot_arch)
    b = _census(monocot_arch)
    assert a == b, "monocot_arch not reproducible"
    assert a.get("metaxylem", 0) > 0 and a.get("protoxylem", 0) > 0


def test_dicot_primary_golden():
    _check("dicot_primary")


def test_dicot_secondary_golden():
    _check("dicot_secondary")


def test_needle_default_golden():
    _check("needle_default")


def test_needle_features_golden():
    _check("needle_features")
