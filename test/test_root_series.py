"""Root developmental series (ROOT_SERIES_PLAN) — Phase 1: prescription plumbing.

Phase 1 = a persistent ``track_id`` on cells that survives the pipeline to the gdf,
plus a RootAnatomy branch that places an *explicit* xylem vessel set (positions +
radii + ids) instead of packing random ones — the mechanism the apex->collet series
is built on.
"""

import numpy as np
import pytest

from openalea.granap.input_data import OrganInputData
from openalea.granap.root_class import RootAnatomy
from openalea.granap.root_series import DicotRootSeries, RootSeries


def test_normal_root_has_track_id_column_all_null():
    """The new track_id column exists and is null for an ordinary (untracked) root."""
    r = RootAnatomy(OrganInputData.for_root(), seed=0)
    gdf = r.generate_cells()
    assert "track_id" in gdf.columns
    assert gdf["track_id"].notna().sum() == 0


def test_prescribed_vessels_land_where_told_and_stably():
    """Prescribed vessels appear as metaxylem cells at their positions, each carrying its
    track_id through the whole pipeline to the gdf — and the same prescribed set gives the
    same geometry twice, so identity is stable across sections of a series."""
    vessels = [(-0.06, 0.0, 0.03, 10), (0.06, 0.0, 0.025, 20), (0.0, 0.07, 0.02, 30)]

    def build():
        r = RootAnatomy(OrganInputData.for_root(), seed=0).prescribe_vessels(vessels)
        g = r.generate_cells()
        return g[g["track_id"].notna()]

    tracked = build()
    assert sorted(int(t) for t in tracked["track_id"].unique()) == [10, 20, 30]
    for (vx, vy, vr, tid) in vessels:
        row = tracked[tracked["track_id"] == tid]
        assert len(row) == 1, f"vessel {tid} should be one tracked cell"
        assert set(row["type"]) == {"metaxylem"}
        c = row.geometry.iloc[0].centroid
        assert np.hypot(c.x - vx, c.y - vy) < 0.02   # lands where prescribed

    def centroids(t):
        t = t.sort_values("track_id")
        return [(round(p.centroid.x, 6), round(p.centroid.y, 6)) for p in t.geometry]
    assert centroids(tracked) == centroids(build())


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


def test_prescribed_metaxylem_keeps_requested_size():
    """Prescribed (series) vessels are seeded against the same metaxylem sheath as
    the default ring, so they get the same sheath-gap inset: each realised vessel
    matches its prescribed radius to within a few percent, small ones included."""
    vessels = [(-0.06, 0.0, 0.03, 10), (0.06, 0.0, 0.0175, 20), (0.0, 0.07, 0.01, 30)]
    g = RootAnatomy(OrganInputData.for_root(), seed=0).prescribe_vessels(vessels).generate_cells()
    ratios = []
    for (_x, _y, r, tid) in vessels:
        area = g[g["track_id"] == tid].geometry.area.sum()
        ratios.append(np.sqrt(area / (np.pi * r * r)))   # realised / prescribed diameter
    assert 0.97 < min(ratios) and max(ratios) < 1.08, ratios
    assert max(ratios) - min(ratios) < 0.04, f"size-dependent inflation: {ratios}"


def test_dicot_extraction_is_stable_regardless_of_span():
    """The primordial set (positions + 5PL targets) is extracted from the smallest-pith
    section, so it does not depend on how the lengths are spanned/sampled."""
    a = _dicot_series(lengths=[0, 25, 50, 75, 100],
                      stele_radius=(0.30, 0.30), pith_radius=(0.24, 0.0))
    b = _dicot_series(lengths=[50, 100],
                      stele_radius=(0.30, 0.30), pith_radius=(0.24, 0.0))
    assert len(a._extract()) == len(b._extract()) > 0


_RECEDING = []


def _receding_series():
    """The one receding-pith series the next two tests both want; ``_extract()`` runs a
    full dicot anatomy build, so it is done once and memoised on the instance."""
    if not _RECEDING:
        _RECEDING.append(_dicot_series(start=0.0, end=100.0, samples=6,
                                       stele_radius=(0.30, 0.30), pith_radius=(0.26, 0.0)))
    return _RECEDING[0]


def test_dicot_pith_recedes_present_count_is_monotone():
    """As the pith recedes (apex -> collet) the number of differentiated vessels only grows,
    and the collet (pith gone) has every primordial while the apex (pith full) has none."""
    s = _receding_series()
    counts = [len(s._active_vessels(L)) for L in s.lengths]
    assert counts[0] == 0                       # apex: pith fills the star
    assert counts[-1] == len(s._extract())      # collet: every vessel present
    assert counts == sorted(counts)             # monotone non-decreasing


def test_dicot_outer_vessels_appear_before_inner():
    """Centripetal maturation: at a mid pith the present vessels are the outer ones (large
    fractional radius); the innermost (central metaxylem) is still inside the pith."""
    s = _receding_series()
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


# ---------------------------------------------------------------------------
# Monocot developmental series (ROOT_SERIES_PLAN Phase 1+2) — fusion / migration.
#
# Unit tests on the identity + placement machinery (``_active_vessels`` /
# ``_migrated_vessels``), which is cheap: it never builds an anatomy.  One
# single-section ``generate()`` at the end covers the plumbing to the gdf.
# ---------------------------------------------------------------------------

def _series(**kw):
    """A minimal monocot series; every schedule flat unless a test overrides it."""
    kw.setdefault("lengths", [0.0, 25.0, 50.0, 75.0, 100.0])
    kw.setdefault("n_fused", (2, 2))
    kw.setdefault("vessel_radius", (0.03, 0.03))
    kw.setdefault("stele_radius", (0.12, 0.12))
    return RootSeries(OrganInputData.for_root(), seed=0, **kw)


def _fuse_2_to_1(length):
    """2 fused metaxylem from 50 mm up, 1 below — a single fusion between the two."""
    return 2 if length >= 50.0 else 1


def _fuse_4_to_1(length):
    """4 fused metaxylem at the collet down to 1 at the apex, one fusion per step."""
    return int(np.clip(round(length / 25.0), 1, 4))


def test_series_samples_the_requested_lengths():
    """start/end/samples and an explicit list give the same sampled positions."""
    a = _series(lengths=None, start=0.0, end=100.0, samples=5)
    assert a.lengths == [0.0, 25.0, 50.0, 75.0, 100.0]


def test_ramp_schedule_hits_its_endpoints():
    """A ``(start, end)`` tuple ramps linearly from the first to the last sampled length."""
    s = _series(vessel_radius=(0.04, 0.02))
    assert s.vessel_radius(0.0) == 0.04          # value at lengths[0]
    assert s.vessel_radius(100.0) == 0.02        # value at lengths[-1]
    assert s.vessel_radius(50.0) == 0.03         # halfway


def test_callable_schedule_is_used_as_is():
    s = _series(n_fused=_fuse_2_to_1)
    assert len(s._active_vessels(100.0)) == 2
    assert len(s._active_vessels(0.0)) == 1


def test_vessel_count_follows_n_fused():
    """The number of active vessels is exactly what the ``n_fused`` schedule asks for."""
    s = _series(n_fused=_fuse_2_to_1)
    assert [len(s._active_vessels(L)) for L in s.lengths] == [1, 1, 2, 2, 2]


def test_every_primordial_belongs_to_exactly_one_vessel():
    """At any length the fused groups partition the primordials — none lost, none doubled."""
    s = _series(n_fused=_fuse_2_to_1)
    for L in s.lengths:
        members = [m for v in s._active_vessels(L) for m in v["members"]]
        assert sorted(members) == [0, 1], f"not a partition at {L} mm"


def test_fusion_is_hierarchical():
    """Groups only ever merge going apex-ward: a group at a lower count is the union of
    groups at a higher one, which is what makes an identity followable."""
    s = _series(n_fused=(4, 4))
    for k in range(2, s.N_fuse + 1):
        fine = [set(g) for g in s._parts[k]]
        for coarse in (set(g) for g in s._parts[k - 1]):
            parts = [f for f in fine if f <= coarse]
            assert set().union(*parts) == coarse, f"{coarse} is not a union of {k}-groups"


def test_fused_vessel_keeps_a_member_id_as_its_track_id():
    """The survivor of a fusion is identified by the smallest primordial it contains, so
    its colour/id persists across the merge."""
    s = _series(n_fused=_fuse_2_to_1)
    for L in s.lengths:
        for v in s._active_vessels(L):
            assert v["tid"] == min(v["members"])


def test_fused_vessel_is_bigger_than_a_single_one():
    """``area_retention`` grows a fused vessel: 0 = no growth, 1 = area-conserving."""
    single = _series(n_fused=(1, 1), area_retention=0.0)._active_vessels(0.0)[0]["r"]
    none = _series(n_fused=(2, 2), area_retention=0.0)._fused_radius(2, 0.0)
    full = _series(n_fused=(2, 2), area_retention=1.0)._fused_radius(2, 0.0)
    assert none == single                                # 0 -> unchanged
    assert full == pytest.approx(single * np.sqrt(2))    # 1 -> area of two vessels


def test_terminating_vessel_is_present_only_above_its_stop():
    """A terminator is an extra singleton present for ``length >= stop``, then gone."""
    s = _series(terminations=[60.0])
    present = {L: {v["members"] for v in s._active_vessels(L)} for L in s.lengths}
    term = s._term_ids[0]
    assert all((term,) in present[L] for L in (75.0, 100.0))
    assert all((term,) not in present[L] for L in (0.0, 25.0, 50.0))


def test_created_vessel_is_present_only_below_its_appear():
    """A creator is the mirror case: an extra singleton present for ``length <= appear``."""
    s = _series(creations=[40.0])
    present = {L: {v["members"] for v in s._active_vessels(L)} for L in s.lengths}
    creat = s._creat_ids[0]
    assert all((creat,) in present[L] for L in (0.0, 25.0))
    assert all((creat,) not in present[L] for L in (50.0, 75.0, 100.0))


def test_vessels_stay_inside_the_stele():
    """Every placed vessel (centre + radius) fits within the stele radius at its length."""
    s = _series(n_fused=_fuse_4_to_1, stele_radius=(0.10, 0.20), terminations=[60.0],
                migration_length=30.0, fusion_length=50.0)
    for L, vessels in s._migrated_vessels().items():
        R = s.stele_radius(L)
        for (x, y, r, _tid, _m) in vessels:
            assert np.hypot(x, y) + r <= R + 1e-9, f"vessel outside the stele at {L} mm"


def test_approach_never_drives_a_vessel_into_a_bystander():
    """A converging pair must not sweep across a vessel it is *not* fusing with (here a
    terminator sits between them).  Placed on the class slots (``migration_length=0``) the
    arrangement is collision-free, so any overlap can only come from the approach."""
    s = _series(n_fused=_fuse_4_to_1, terminations=[60.0],
                migration_length=0.0, fusion_length=50.0)
    for L, vessels in s._migrated_vessels().items():
        for i in range(len(vessels)):
            for j in range(i + 1, len(vessels)):
                xi, yi, ri = vessels[i][:3]
                xj, yj, rj = vessels[j][:3]
                assert np.hypot(xi - xj, yi - yj) >= ri + rj - 1e-9, f"overlap at {L} mm"


def test_slots_are_spaced_for_the_biggest_vessel_present():
    """A fused vessel is wider than ``vessel_radius`` by ``area_retention``, and now that
    vessels migrate all the way onto their slots the arrangement has to fit it.  A tight
    stele (0.09 mm) holding 0.035 mm vessels is where the class arrangement stops being
    usable and the even-ring fallback has to take over."""
    s = _series(n_fused=_fuse_4_to_1, vessel_radius=(0.035, 0.035),
                stele_radius=(0.09, 0.09), area_retention=0.4,
                migration_length=0.0, fusion_length=0.0)
    for L, vessels in s._migrated_vessels().items():
        for i in range(len(vessels)):
            for j in range(i + 1, len(vessels)):
                xi, yi, ri = vessels[i][:3]
                xj, yj, rj = vessels[j][:3]
                assert np.hypot(xi - xj, yi - yj) >= ri + rj - 1e-9, \
                    f"{vessels[i][4]} overlaps {vessels[j][4]} at {L} mm"


def test_migration_zero_snaps_to_the_class_slots():
    """``migration_length=0`` places vessels straight on the class arrangement."""
    s = _series(n_fused=(3, 3), migration_length=0.0)
    for L, vessels in s._migrated_vessels().items():
        slots = s._class_slots(len(vessels), s.stele_radius(L), 2 * s.vessel_radius(L))
        for (x, y, _r, _tid, _m) in vessels:
            assert min(np.hypot(x - sx, y - sy) for sx, sy in slots) < 1e-9


def test_fusion_length_brings_the_pair_into_contact_before_it_merges():
    """The point of ``fusion_length``: at the last section where the two parents are still
    separate they are already touching, so the merge below is continuous rather than a jump
    across a gap."""
    def gap_at(fusion_length):
        s = _series(n_fused=_fuse_2_to_1, migration_length=40.0, fusion_length=fusion_length)
        a, b = s._migrated_vessels()[50.0]     # 50 mm = last length with 2 vessels
        return np.hypot(a[0] - b[0], a[1] - b[1]) - (a[2] + b[2])

    assert gap_at(50.0) == pytest.approx(0.0, abs=1e-9)   # touching
    assert gap_at(0.0) > 1e-3                             # ...and it did something


def test_fusion_approach_is_gradual():
    """The parents close the gap progressively over ``fusion_length`` mm, not in one step."""
    s = _series(n_fused=_fuse_2_to_1, migration_length=40.0, fusion_length=50.0)
    mig = s._migrated_vessels()
    gaps = []
    for L in (100.0, 75.0, 50.0):
        a, b = mig[L]
        gaps.append(np.hypot(a[0] - b[0], a[1] - b[1]) - (a[2] + b[2]))
    assert gaps[0] > gaps[1] > gaps[2]         # shrinking collet -> apex
    assert gaps[2] == pytest.approx(0.0, abs=1e-9)


def test_placement_does_not_depend_on_how_densely_you_sample():
    """The anatomy is a function of depth, not of the sampling grid: sections asked for at
    the same lengths must come out identical however many other lengths were requested
    alongside them.  Migration is timed from the event, and the walk steps through the
    exact event boundaries, so refining ``lengths`` only interpolates."""
    probes = [0.0, 33.0, 66.0, 99.0, 132.0]
    kw = dict(n_fused=lambda x: int(np.clip(round(1 + 3 * x / 150.0), 1, 4)),
              terminations=[90.0], creations=[40.0],
              vessel_radius=(0.035, 0.020), stele_radius=(0.09, 0.18),
              migration_length=40.0, fusion_length=40.0)

    def placed(extra):
        s = RootSeries(OrganInputData.for_root(), seed=0,
                       lengths=sorted(set(probes) | set(extra)), **kw)
        mig = s._migrated_vessels()
        return {L: {v[4]: (v[0], v[1]) for v in mig[L]} for L in probes}

    coarse = placed(np.linspace(0.0, 150.0, 8))
    fine = placed(np.linspace(0.0, 150.0, 61))
    irregular = placed([0.0, 7.0, 19.0, 41.0, 55.0, 88.0, 91.0, 120.0, 143.0, 150.0])
    for other in (fine, irregular):
        for L in probes:
            assert set(other[L]) == set(coarse[L]), f"different vessels at {L} mm"
            for members, (x, y) in coarse[L].items():
                ox, oy = other[L][members]
                # Not bit-exact: the class arrangement goes through ``fitEllipse``, which
                # is unstable on a near-degenerate pizza slice and moves the centre by
                # ~1e-16.  1e-9 mm is ten million times finer than a vessel.
                assert np.hypot(ox - x, oy - y) < 1e-9, f"{members} moved at {L} mm"


def test_migration_is_timed_from_the_event_not_from_the_series_start():
    """``migration_length`` is a distance measured from the fusion, so the survivor moves
    linearly in depth over the 40 mm below it and arrives exactly there — not already
    parked because the series happened to start further up, and not still drifting after."""
    s = RootSeries(OrganInputData.for_root(), seed=0,
                   lengths=[100.0, 75.0, 50.0, 40.0, 30.0, 20.0, 10.0, 0.0],
                   n_fused=_fuse_2_to_1, vessel_radius=(0.03, 0.03),
                   stele_radius=(0.12, 0.12), migration_length=40.0)
    mig = s._migrated_vessels()                     # the fusion is at 50 mm
    p = {L: np.array(mig[L][0][:2]) for L in (40.0, 30.0, 20.0, 10.0, 0.0)}
    assert not np.allclose(p[40.0], p[30.0])        # still migrating below the fusion
    assert np.allclose(p[30.0] - p[40.0], p[20.0] - p[30.0], atol=1e-12)   # linear in depth
    assert not np.allclose(p[20.0], p[10.0])        # has not arrived early...
    assert np.allclose(p[10.0], p[0.0], atol=1e-12)  # ...and has at 50 - 40 = 10 mm


def test_fusion_length_zero_changes_nothing():
    """The approach is opt-in: 0 (the default) leaves placement exactly as it was."""
    kw = dict(n_fused=_fuse_2_to_1, migration_length=40.0)
    assert _series(**kw)._migrated_vessels() == _series(fusion_length=0.0, **kw)._migrated_vessels()


def test_monocot_series_generates_tracked_metaxylem():
    """One section end-to-end: the prescribed vessels reach the gdf as tracked metaxylem
    carrying their track_id, and the track table agrees with them."""
    s = _series(lengths=[100.0], n_fused=(2, 2))
    res = s.generate()
    gdf = res.sections[0]["gdf"]
    tracked = gdf[gdf["track_id"].notna()]
    assert set(tracked["type"]) == {"metaxylem"}
    assert sorted(int(t) for t in tracked["track_id"].unique()) == [0, 1]
    assert sorted(r["track_id"] for r in res.track_rows) == [0, 1]
    assert res.lengths == [100.0]
