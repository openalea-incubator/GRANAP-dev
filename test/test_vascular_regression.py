"""Golden regression tests for the root anatomy pipeline.

These pin the exact ``seed=0`` cell-type census for the canonical root
configurations.  They are the safety net the tissue refactor relied on: any
change that alters the produced anatomy (intended or not) will trip these, so an
intended change is a deliberate golden update rather than a silent drift.

Reproducibility note: full ``seed=0`` determinism depends on the Voronoi jitter
drawing from the organ's seeded ``self.rng`` (see ``Cell.jitter`` /
``CellGenerator.voronoi_diagram``).  Before that fix, dicot secondary growth
drifted run-to-run because the jitter used the global ``np.random``.

Cross-platform note: ``dicot_stem`` used to differ on macOS/Windows because the
xylem file-separator strips were clipped to the zone before being used to cut it
(``_xylem_file_strips``), which left severance to a collinear-boundary decision
GEOS resolves at the last bit — a 1e-9 nudge flipped 2 of 8 bundles from 3 xylem
files to 2.  With the cutter left unclipped all 8 bundles split as parameterised
and the census is stable to ~1e-4.  Counts here are stable across GEOS 3.13/3.14
and py3.13/3.14; the geometry stack is pinned in ``pyproject.toml``.
"""

import os
import sys

sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "example", "needle"))

from openalea.granap.root_class import RootAnatomy
from openalea.granap.needle_class import NeedleAnatomy
from openalea.granap.stem_class import StemAnatomy
from openalea.granap.input_data import OrganInputData
from gallery_needle_features import build_gallery_needle_data

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
    """The feature-showcase needle from ``example/needle/gallery_needle_features.py``.

    Calls that gallery's own ``build_gallery_needle_data`` directly (rather
    than a hand-copied duplicate of its config) so this golden fixture can
    never silently drift from what the gallery actually demonstrates -- see
    that function's docstring for why.
    """
    data = build_gallery_needle_data()
    return NeedleAnatomy(data, seed=SEED)


def dicot_stem() -> StemAnatomy:
    """Dicot stem eustele: a ring of open collateral bundles around a pith."""
    return StemAnatomy(OrganInputData.for_dicot_stem(), seed=SEED)


def monocot_stem() -> StemAnatomy:
    """Monocot stem atactostele: scattered 'face' bundles + sclerenchyma."""
    return StemAnatomy(OrganInputData.for_monocot_stem(), seed=SEED)


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
        "air space": 574, "cambium": 99, "companion_cell": 55, "cortex": 332,
        "endodermis": 51, "epidermis": 248, "exodermis": 119, "pericycle": 139,
        "phloem": 108, "stele": 1056, "xylem": 50,
    }),
    "needle_default": (needle_default, {
        "Strasburger cell": 38, "air space": 476, "cambium": 58, "duct": 3,
        "endodermis": 49, "epidermis": 231, "guard cell": 8, "hypodermis": 387,
        "mesophyll": 228, "parenchyma": 244, "phloem": 310, "pore": 4,
        "resin duct": 42, "transfusion": 103, "xylem": 270,
    }),
    "needle_features": (needle_features, {
        "Str. Interstitial cell": 90, "Strasburger cell": 26, "air space": 503,
        "cambium": 32, "duct": 2, "endodermis": 45, "epidermis": 239,
        "guard cell": 20, "hypodermis": 355, "hypodermis_corner": 29,
        "mesophyll": 231, "parenchyma": 76, "phloem": 164, "pore": 10,
        "resin duct epithelium": 20, "resin duct sheath": 36,
        "transfusion parenchyma": 43, "transfusion tracheid": 125, "xylem": 202,
    }),
    "dicot_stem": (dicot_stem, {
        "air space": 150, "cambium": 66, "companion cell": 71, "cortex": 188,
        "epidermis": 220, "parenchyma": 3963, "sieve element": 71, "xylem": 72,
    }),
    "monocot_stem": (monocot_stem, {
        "air space": 543, "companion cell": 159, "cortex": 470, "epidermis": 261,
        "parenchyma": 3853, "sclerenchyma": 2137, "sieve element": 159, "xylem": 39,
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


def test_dicot_stem_golden():
    _check("dicot_stem")


def test_monocot_stem_golden():
    _check("monocot_stem")
