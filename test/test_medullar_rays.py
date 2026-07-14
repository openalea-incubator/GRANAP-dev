"""Tests for dicot secondary-growth medullar rays.

Covers the constant-tangential-width corridors and the radius-dependent
initiation rate (``n_medullar_rate``): a fixed ``n_medullar`` set from the primary
cambium plus extra rays that appear further out, so ray density grows toward the
periphery.  Visual demo: ``example/medullar_rays.py``.
"""

import os
import sys

import numpy as np

sys.path.append(os.path.abspath(".."))

from openalea.granap.root_class import RootAnatomy
from openalea.granap.input_data import OrganInputData, DicotMedularRaysParams


SEED = 0


def make_secondary_root(**mr_kwargs) -> RootAnatomy:
    """Dicot secondary-growth root with a medullar-ray param entry."""
    data = OrganInputData.for_dicot_root()
    data.set_value("secondary_growth", "value", True)
    data.set_value("stele", "thickness", 1.2)
    data.params.append(DicotMedularRaysParams(**mr_kwargs))
    root = RootAnatomy(data, seed=SEED)
    root.generate_cells()
    return root


def _center(root):
    xs = [c.x for c in root.vascular_cells.cells]
    ys = [c.y for c in root.vascular_cells.cells]
    return np.mean(xs), np.mean(ys)


def _ray_cells(root):
    return [c for c in root.vascular_cells.cells if c.type == "medullar_ray"]


def distinct_rays(root, r_lo_frac=0.0, r_hi_frac=1.0, gap_deg=3.0) -> int:
    """Number of distinct medullar rays whose cells fall in the radial band
    ``[r_lo_frac, r_hi_frac]`` of the ray extent, by clustering cell angles with a
    ``gap_deg`` separation (wrap-aware)."""
    cells = _ray_cells(root)
    if not cells:
        return 0
    cx, cy = _center(root)
    radii = [np.hypot(c.x - cx, c.y - cy) for c in cells]
    rmin, rmax = min(radii), max(radii)
    span = rmax - rmin or 1.0
    lo, hi = rmin + r_lo_frac * span, rmin + r_hi_frac * span

    angles = sorted(
        np.arctan2(c.y - cy, c.x - cx) % (2 * np.pi)
        for c, r in zip(cells, radii) if lo <= r <= hi
    )
    if not angles:
        return 0
    gap = np.radians(gap_deg)
    clusters = 1
    for a, b in zip(angles[:-1], angles[1:]):
        if b - a > gap:
            clusters += 1
    # merge the wrap-around cluster if the first and last are adjacent
    if clusters > 1 and (angles[0] + 2 * np.pi - angles[-1]) <= gap:
        clusters -= 1
    return clusters


def test_rate_zero_matches_n_medullar():
    """With rate=0, exactly n_medullar rays are present (no rate-driven extras).

    (Inner bands can show slightly fewer because the primary cambium is
    star-shaped — rays over its peaks start at a larger radius than in its
    valleys — so we assert the full-extent total, and that the outer band does
    not exceed n_medullar.)"""
    root = make_secondary_root(n_medullar=6, n_medullar_rate=0.0, allow_non_vascular=True)
    assert distinct_rays(root) == 6, "Expected exactly n_medullar rays at rate=0"
    assert distinct_rays(root, 0.55, 1.0) <= 6, "rate=0 must not add rays outward"


def test_rate_adds_rays_outward():
    """With rate>0 the outer wood holds more distinct rays than the inner wood."""
    root = make_secondary_root(
        n_medullar=6, n_medullar_rate=150.0, start_radius=0.0,
        start_radius_sd=0.1, allow_non_vascular=True,
    )
    inner = distinct_rays(root, 0.0, 0.45)
    outer = distinct_rays(root, 0.55, 1.0)
    assert outer > inner, f"Expected denser outer rays (inner={inner}, outer={outer})"


def test_rate_increases_total_rays():
    """A higher rate yields more medullar-ray cells overall (monotonic)."""
    n0 = len(_ray_cells(make_secondary_root(n_medullar=6, n_medullar_rate=0.0, allow_non_vascular=True)))
    n1 = len(_ray_cells(make_secondary_root(n_medullar=6, n_medullar_rate=50.0, allow_non_vascular=True)))
    n2 = len(_ray_cells(make_secondary_root(n_medullar=6, n_medullar_rate=150.0, allow_non_vascular=True)))
    assert n0 < n1 < n2, f"Ray cell count should grow with rate ({n0}, {n1}, {n2})"


def test_rate_driven_rays_still_reach_phloem():
    """Rays reach the outer edge, so the secondary phloem is still produced."""
    root = make_secondary_root(
        n_medullar=6, n_medullar_rate=100.0, start_radius=0.2,
        start_radius_sd=0.05, allow_non_vascular=True,
    )
    counts = {}
    for c in root.all_cells.cells:
        counts[c.type] = counts.get(c.type, 0) + 1
    assert counts.get("phloem", 0) > 0, "Expected secondary phloem with rate-driven rays"
    assert counts.get("medullar_ray", 0) > 0, "Expected medullar-ray cells"
