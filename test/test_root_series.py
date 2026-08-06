"""Root developmental series (ROOT_SERIES_PLAN) — Phase 1: prescription plumbing.

Phase 1 = a persistent ``track_id`` on cells that survives the pipeline to the gdf,
plus a RootAnatomy branch that places an *explicit* xylem vessel set (positions +
radii + ids) instead of packing random ones — the mechanism the apex->collet series
is built on.
"""

import numpy as np

from openalea.granap.input_data import OrganInputData
from openalea.granap.root_class import RootAnatomy


def test_normal_root_has_track_id_column_all_null():
    """The new track_id column exists and is null for an ordinary (untracked) root."""
    r = RootAnatomy(OrganInputData.for_root(), seed=0)
    gdf = r.generate_cells()
    assert "track_id" in gdf.columns
    assert gdf["track_id"].notna().sum() == 0


def test_prescribed_vessels_land_where_told_with_ids():
    """Prescribed vessels appear as xylem cells at their positions, each carrying its
    track_id through the whole pipeline to the gdf."""
    vessels = [(-0.06, 0.0, 0.03, 10), (0.06, 0.0, 0.025, 20), (0.0, 0.07, 0.02, 30)]
    r = RootAnatomy(OrganInputData.for_root(), seed=0).prescribe_vessels(vessels)
    gdf = r.generate_cells()

    tracked = gdf[gdf["track_id"].notna()]
    assert sorted(int(t) for t in tracked["track_id"].unique()) == [10, 20, 30]
    for (vx, vy, vr, tid) in vessels:
        row = tracked[tracked["track_id"] == tid]
        assert len(row) == 1, f"vessel {tid} should be one tracked cell"
        assert set(row["type"]) == {"xylem"}
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
