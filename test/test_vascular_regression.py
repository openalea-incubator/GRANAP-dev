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

Three more instances of the same family were traced and fixed, which is why the
stem and needle entries here moved.  All were **exact ties** decided by the last
bits of coordinates, which glibc, UCRT and Apple/arm64 ``sin``/``cos``/
``arctan2`` compute differently (arm64 also contracts ``a*b+c`` into a single
FMA with one rounding, so it diverges from x86 even given identical libm):

* ``GeometryProcessor._chebyshev_center`` fed an unsnapped polygon to
  ``shapely.maximum_inscribed_circle``.  A zone symmetric about an axis has two
  tied optimal centres — ``dicot_stem``'s 3-o'clock bundle sits exactly on the
  x-axis — and the noise chose between them, so the whole Apollonian packing
  followed a different branch.  The polygon is now snapped first.
* ``_grow_bundle_sheath`` asked ``env.contains(b)`` for points ``b`` on the
  bundle *footprint*'s boundary, to tell "this point sits on a fibre cap" from
  "this point is on the envelope".  But the footprint is the envelope plus the
  caps, from the same coordinates through the same transform, so a non-cap point
  lies exactly *on* ``env``'s boundary (and in the arc-bundle path ``foot`` *is*
  ``env``) — a coin flip that sized each sheath cell from either the fibre or the
  parenchyma and cascaded through the march.  The test is now tolerant.
* ``NeedleAnatomy.retag_corner_parenchyma`` asked
  ``zone.contains(Point(c.x, c.y))`` where the zone is cut out of the very layer
  polygon the cells were seeded in, so it inherits that outline — and the
  layer's *border* seeds sit exactly on it.  44 of 250 candidates here (30 of
  102 for ``needle_features``) were boundary cases, i.e. half the Strasburger
  cells were tagged by arithmetic noise, and macOS/arm64 resolved 4 of them the
  other way.  Boundary seeds now count as inside, which is both the geometric
  reading (their Voronoi body lies in the region) and the biological one (they
  are the rim that contacts the transfusion tissue).  This is why
  ``Strasburger cell``/``parenchyma`` moved 53/197 -> 71/179 and 26/76 -> 40/62.

With the first two in place ``dicot_stem`` and ``monocot_stem`` are
**bit-identical** between a native Windows build and a WSL Ubuntu build of the
same pinned ``geos==3.14.1``/``shapely==2.1.2``: equal seed counts in every
``(tissue, id_layer)`` bucket, and every cell centroid matching to <1e-14.

What makes all three fixable rather than merely re-tuned is that the tie cells
are *separated* from the genuine decisions by many orders of magnitude — for the
needle, ties within 5.3e-17 against a nearest true interior cell beyond 1e-6 —
so the tolerance is not a fitted value.
``test_needle_retag_has_no_knife_edge_decisions`` pins that gap.
"""

import os
import sys

sys.path.append(os.path.abspath(".."))

import pytest
from shapely.geometry import Point

from openalea.granap.root_class import RootAnatomy
from openalea.granap.needle_class import NeedleAnatomy, _POCKET_ON_EDGE_TOL
from openalea.granap.stem_class import StemAnatomy
from openalea.granap.input_data import OrganInputData
from needle_configs import build_gallery_needle_data

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
        "phloem": 108, "stele": 1057, "xylem": 50,
    }),
    # Strasburger/parenchyma refrozen 53/197 -> 71/179 when
    # retag_corner_parenchyma stopped deciding boundary seeds by arithmetic
    # noise; see that method's docstring.  Nothing else in either census moved.
    "needle_default": (needle_default, {
        "Str. Interstitial cell": 96, "Strasburger cell": 71, "air space": 538,
        "cambium": 58, "duct": 3, "endodermis": 49, "epidermis": 239,
        "guard cell": 8, "hypodermis": 387, "mesophyll": 252, "parenchyma": 179,
        "phloem": 316, "pore": 4, "resin duct epithelium": 30,
        "resin duct sheath": 42, "transfusion": 103, "xylem": 394,
    }),
    "needle_features": (needle_features, {      # 26/76 -> 40/62, same fix
        "Str. Interstitial cell": 90, "Strasburger cell": 40, "air space": 499,
        "cambium": 32, "duct": 2, "endodermis": 45, "epidermis": 239,
        "guard cell": 20, "hypodermis": 355, "hypodermis_corner": 27,
        "mesophyll": 231, "parenchyma": 62, "phloem": 164, "pore": 10,
        "resin duct epithelium": 20, "resin duct sheath": 36,
        "transfusion parenchyma": 37, "transfusion tracheid": 112, "xylem": 202,
    }),
    "dicot_stem": (dicot_stem, {
        "air space": 150, "cambium": 66, "companion cell": 71, "cortex": 188,
        "epidermis": 220, "parenchyma": 3962, "sieve element": 71, "xylem": 72,
    }),
    "monocot_stem": (monocot_stem, {
        "air space": 549, "companion cell": 159, "cortex": 455, "epidermis": 261,
        "parenchyma": 3830, "sclerenchyma": 2137, "sieve element": 159, "xylem": 39,
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


@pytest.mark.parametrize("name", ["needle_default", "needle_features"])
def test_needle_retag_has_no_knife_edge_decisions(name, monkeypatch):
    """No corner-parenchyma cell may sit *near* the pocket edge tolerance.

    ``retag_corner_parenchyma`` counts a seed within ``_POCKET_ON_EDGE_TOL`` of
    the pocket edge as inside it, because the layer's border seeds land exactly
    on that edge and a bare ``contains`` made their tag a coin flip on the last
    bit (see that method's docstring — it drifted the census on macOS/arm64).

    That tolerance is only safe because of a *gap*: every candidate is either a
    true boundary case, within ~5e-17, or a clear decision beyond 1e-6 — six
    orders of magnitude clear of the 1e-9 threshold, and eight clear of the
    ~1e-15 platform noise that could push a cell across it.  Nothing lands in
    between, so the exact tolerance cannot matter.

    This test pins that gap rather than the tolerance.  If a change to layer
    seeding or ellipse placement ever parks a cell mid-band, the tolerance
    silently becomes load-bearing again and the goldens become platform-
    dependent once more — so fail here, where the cause is legible, instead of
    in a golden census on one CI runner.
    """
    seen = []
    original = NeedleAnatomy.retag_corner_parenchyma

    def spy(self):
        zone = getattr(self, "_parenchyma_pocket_zone", None)
        if zone is not None and not zone.is_empty:
            edge = zone.boundary
            seen.extend(
                edge.distance(Point(c.x, c.y))
                for c in self.all_cells.get_cells_by_type("parenchyma")
            )
        return original(self)

    monkeypatch.setattr(NeedleAnatomy, "retag_corner_parenchyma", spy)

    organ = GOLDEN[name][0]()
    organ.generate_cells()

    assert seen, "the retag never ran, so this test proves nothing"
    ties = [d for d in seen if d <= _POCKET_ON_EDGE_TOL]
    assert ties, "expected boundary seeds; has layer seeding changed?"

    ambiguous = [d for d in seen if _POCKET_ON_EDGE_TOL < d < 1e-6]
    assert not ambiguous, (
        f"{len(ambiguous)} cell(s) sit between the pocket-edge tolerance "
        f"({_POCKET_ON_EDGE_TOL:.0e}) and 1e-6, closest {min(ambiguous):.3e}. "
        "The tolerance is now load-bearing: these cells' tags depend on its "
        "exact value, so the needle census will drift across platforms again. "
        "Re-measure the distance distribution before touching the tolerance."
    )
    assert max(ties) < 1e-13, (
        f"widest 'on the edge' seed is {max(ties):.3e}, far above the ~5e-17 "
        "these cases measure; that is no longer floating-point noise"
    )


def test_dicot_stem_golden():
    _check("dicot_stem")


def test_monocot_stem_golden():
    _check("monocot_stem")
