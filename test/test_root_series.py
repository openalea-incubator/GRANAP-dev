"""Root developmental series (ROOT_SERIES_PLAN) — Phase 1: prescription plumbing.

Phase 1 = a persistent ``track_id`` on cells that survives the pipeline to the gdf,
plus a RootAnatomy branch that places an *explicit* xylem vessel set (positions +
radii + ids) instead of packing random ones — the mechanism the apex->collet series
is built on.
"""

import numpy as np

from openalea.granap.input_data import OrganInputData
from openalea.granap.root_class import RootAnatomy
from openalea.granap.root_series import DicotRootSeries


def test_normal_root_has_track_id_column_all_null():
    """The new track_id column exists and is null for an ordinary (untracked) root."""
    r = RootAnatomy(OrganInputData.for_root(), seed=0)
    gdf = r.generate_cells()
    assert "track_id" in gdf.columns
    assert gdf["track_id"].notna().sum() == 0


def test_prescribed_vessels_land_where_told_with_ids():
    """Prescribed vessels appear as metaxylem cells at their positions, each carrying its
    track_id through the whole pipeline to the gdf."""
    vessels = [(-0.06, 0.0, 0.03, 10), (0.06, 0.0, 0.025, 20), (0.0, 0.07, 0.02, 30)]
    r = RootAnatomy(OrganInputData.for_root(), seed=0).prescribe_vessels(vessels)
    gdf = r.generate_cells()

    tracked = gdf[gdf["track_id"].notna()]
    assert sorted(int(t) for t in tracked["track_id"].unique()) == [10, 20, 30]
    for (vx, vy, vr, tid) in vessels:
        row = tracked[tracked["track_id"] == tid]
        assert len(row) == 1, f"vessel {tid} should be one tracked cell"
        assert set(row["type"]) == {"metaxylem"}
        c = row.geometry.iloc[0].centroid
        assert np.hypot(c.x - vx, c.y - vy) < 0.02   # lands where prescribed


def test_prescription_is_deterministic():
    """Same prescribed set -> same tracked geometry (identity is stable)."""
    vessels = [(-0.05, 0.0, 0.03, 1), (0.05, 0.0, 0.03, 2)]
    def centroids():
        r = RootAnatomy(OrganInputData.for_root(), seed=0).prescribe_vessels(vessels)
        g = r.generate_cells()
        t = g[g["track_id"].notna()].sort_values("track_id")
        return [(round(p.centroid.x, 6), round(p.centroid.y, 6)) for p in t.geometry]
    assert centroids() == centroids()


# ---------------------------------------------------------------------------
# Dicot developmental series (ROOT_SERIES_PLAN Phase 3) — primary growth.
#
# The pith is a central front that recedes from the apex toward the collet;
# a vessel appears once the pith clears it (outer protoxylem first, inner
# metaxylem last) and grows to its 5PL target.  Positions/targets are extracted
# once from the smallest-pith (collet) section, so identity is stable.
# ---------------------------------------------------------------------------

def _dicot_series(**kw):
    base = OrganInputData.for_dicot_root()
    base.set_value("xylem", "n_vascular_peak", 4)
    return DicotRootSeries(base, seed=0, **kw)


def test_dicot_extraction_is_stable_regardless_of_span():
    """The primordial set (positions + 5PL targets) is extracted from the smallest-pith
    section, so it does not depend on how the lengths are spanned/sampled."""
    a = _dicot_series(lengths=[0, 25, 50, 75, 100],
                      stele_radius=(0.30, 0.30), pith_radius=(0.24, 0.0))
    b = _dicot_series(lengths=[50, 100],
                      stele_radius=(0.30, 0.30), pith_radius=(0.24, 0.0))
    assert len(a._extract()) == len(b._extract()) > 0


def test_dicot_pith_recedes_present_count_is_monotone():
    """As the pith recedes (apex -> collet) the number of differentiated vessels only grows,
    and the collet (pith gone) has every primordial while the apex (pith full) has none."""
    s = _dicot_series(start=0.0, end=100.0, samples=6,
                      stele_radius=(0.30, 0.30), pith_radius=(0.26, 0.0))
    counts = [len(s._active_vessels(L)) for L in s.lengths]
    assert counts[0] == 0                       # apex: pith fills the star
    assert counts[-1] == len(s._extract())      # collet: every vessel present
    assert counts == sorted(counts)             # monotone non-decreasing


def test_dicot_outer_vessels_appear_before_inner():
    """Centripetal maturation: at a mid pith the present vessels are the outer ones (large
    fractional radius); the innermost (central metaxylem) is still inside the pith."""
    s = _dicot_series(start=0.0, end=100.0, samples=6,
                      stele_radius=(0.30, 0.30), pith_radius=(0.26, 0.0))
    prim = {p["tid"]: p for p in s._extract()}
    innermost = min(prim.values(), key=lambda p: p["fd"])["tid"]
    mid = s.lengths[len(s.lengths) // 2]
    present_ids = {v[3] for v in s._active_vessels(mid)}
    assert present_ids                          # some vessels have differentiated
    assert innermost not in present_ids         # ...but not the central one yet


def test_dicot_series_generates_tracked_metaxylem():
    """A dicot section with a receded pith places its prescribed vessels as tracked
    metaxylem cells carrying their track_id through to the gdf."""
    s = _dicot_series(lengths=[100.0], stele_radius=(0.30, 0.30), pith_radius=(0.0, 0.0))
    res = s.generate()
    gdf = res.sections[0]["gdf"]
    tracked = gdf[gdf["track_id"].notna()]
    assert len(tracked) > 0
    assert set(tracked["type"]) == {"metaxylem"}
    assert sorted(int(t) for t in tracked["track_id"].unique()) == \
        sorted(p["tid"] for p in s._extract())
