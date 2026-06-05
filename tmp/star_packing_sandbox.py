"""
Sandbox for investigating circle packing inside a star-shaped xylem polygon.
Option B: geometric trapezoid branches — no solver, always valid.
"""

import numpy as np
import shapely as sp
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.ops import unary_union
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize

# ── Parameters ────────────────────────────────────────────────────────────

N_BRANCHES = 5
INNER_R    = 0.04  # inner_radius_xylem  — base of branches
OUTER_R    = 0.20   # outer_radius_xylem  — tip of branches
ARC_BASE   = 0.05   # arc length at INNER_R (base width)
ARC_TOP    = 0.02   # arc length at OUTER_R (tip width)

SMOOTH_FACTOR = 0.5
SMOOTH_ITER   = 10

# ── Xylem vessel parameters (for future packing) ──────────────────────────

DIAM_MAX = 0.07
DIAM_MIN = 0.01
DIAM_SD  = 0.002
DENSITY  = 1


# ── Star polygon (Option B: geometric trapezoid) ───────────────────────────

def star_polygon(n_branches, r_min, r_max, arc_base, arc_top, n_arc=200):
    """
    Each branch is a trapezoid: inner arc at r_min (half-width arc_base/r_min)
    connected by straight sides to outer arc at r_max (half-width arc_top/r_max).
    A central disk at r_min fills the space between branches.
    arc_base and arc_top are fully independent — no solver, never crashes.
    """
    w_base = arc_base / r_min
    w_top  = arc_top  / r_max

    branches = []
    for k in range(n_branches):
        tk = 2 * np.pi * k / n_branches
        inner = np.linspace(tk - w_base, tk + w_base, n_arc)
        outer = np.linspace(tk - w_top,  tk + w_top,  n_arc)
        pts = np.vstack([
            np.column_stack([r_min * np.cos(inner), r_min * np.sin(inner)]),
            np.column_stack([r_max * np.cos(outer[::-1]), r_max * np.sin(outer[::-1])]),
        ])
        branches.append(sp.Polygon(pts))

    return unary_union([sp.Point(0, 0).buffer(r_min, resolution=64)] + branches)


# ── Smoothing ─────────────────────────────────────────────────────────────
def resample_coords(coords: np.ndarray, target_n_points: int = 200, 
                        shift_distance: float = 0) -> np.ndarray:
        """
        Resample coordinates to have uniform spacing.
        
        Args:
            coords: Array of (x, y) coordinates
            target_n_points: Desired number of points
            shift_distance: Distance to shift the starting point
        
        Returns:
            Resampled coordinate array
        """
        coords = np.array(coords)
        if len(coords) < 2:
            return coords
        
        dists = np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1))
        cum_dist = np.concatenate(([0], np.cumsum(dists)))
        total_len = cum_dist[-1]
        
        new_dists = np.linspace(0, total_len, target_n_points)
        
        if shift_distance != 0:
            new_dists = (new_dists + shift_distance) % total_len
            # Ensure they are sorted for interpolation if not a close
        
        new_x = np.interp(new_dists, cum_dist, coords[:, 0])
        new_y = np.interp(new_dists, cum_dist, coords[:, 1])
        
        return np.column_stack((new_x, new_y))

def smoothing_polygon(coords: np.ndarray, smooth_factor: float, 
                         iterations: int = 10) -> np.ndarray:
        """
        Smooth coordinates using Laplacian smoothing.
        
        Args:
            coords: Array of (x, y) coordinates
            smooth_factor: Smoothing strength (0-1)
            iterations: Number of smoothing passes
        
        Returns:
            Smoothed coordinate array
        """
        coords = resample_coords(coords, target_n_points=200)
        
        for _ in range(iterations):
            is_closed = np.allclose(coords[0], coords[-1])
            
            if is_closed:
                pts = coords[:-1]
            else:
                pts = coords
            
            pts = pts.astype(float)
            
            if len(pts) < 3:
                return coords
            
            prev_pts = np.roll(pts, 1, axis=0)
            next_pts = np.roll(pts, -1, axis=0)
            
            smoothed_pts = (1 - smooth_factor) * pts + \
                          smooth_factor * (prev_pts + next_pts) / 2.0
            
            if is_closed:
                coords = np.vstack([smoothed_pts, smoothed_pts[0]])
            else:
                coords = smoothed_pts
        
        return coords
    

# ── Build ──────────────────────────────────────────────────────────────────

star        = star_polygon(N_BRANCHES, INNER_R, OUTER_R, ARC_BASE, ARC_TOP)
star_smooth = sp.Polygon(
    smoothing_polygon(np.column_stack(star.exterior.xy),
                      smooth_factor=SMOOTH_FACTOR, iterations=SMOOTH_ITER)
)

print(f"Raw    — area: {star.area:.6f},  valid: {star.is_valid}")
print(f"Smooth — area: {star_smooth.area:.6f},  valid: {star_smooth.is_valid}")

# ── Circle packing ────────────────────────────────────────────────────────

def chebyshev_center(polygon, grid_n=15):
    """Pole of inaccessibility: maximise min-distance to all boundaries."""
    minx, miny, maxx, maxy = polygon.bounds
    best_r, best_xy = 0.0, (polygon.centroid.x, polygon.centroid.y)

    for x in np.linspace(minx, maxx, grid_n):
        for y in np.linspace(miny, maxy, grid_n):
            p = sp.Point(x, y)
            if polygon.contains(p):
                r = p.distance(polygon.boundary)
                if r > best_r:
                    best_r, best_xy = r, (x, y)

    def neg_r(xy):
        p = sp.Point(xy)
        return -p.distance(polygon.boundary) if polygon.contains(p) else 0.0

    res = minimize(neg_r, best_xy, method='Nelder-Mead',
                   options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 300})
    cx, cy = res.x
    p = sp.Point(cx, cy)
    if polygon.contains(p):
        r = p.distance(polygon.boundary)
        if r >= best_r:
            return cx, cy, r
    return best_xy[0], best_xy[1], best_r


def apollonian_pack(polygon, diam_max, diam_min):
    """
    Iterative Apollonian packing with center-to-edge size gradient:
      - Circles near the polygon centroid get up to diam_max.
      - Circles near the boundary get down to diam_min.
      - Target diameter interpolated linearly with normalised distance.
    """
    placed = []
    stack = [polygon]

    poly_cx, poly_cy = polygon.centroid.x, polygon.centroid.y
    minx, miny, maxx, maxy = polygon.bounds
    max_dist = max(maxx - poly_cx, poly_cx - minx, maxy - poly_cy, poly_cy - miny)

    while stack:
        region = stack.pop()

        if region.area < np.pi * (diam_min / 2) ** 2:
            continue

        cx, cy, r_ins = chebyshev_center(region)

        # Distance-based target: 0 at centroid → diam_max, 1 at edge → diam_min
        t = min(np.hypot(cx - poly_cx, cy - poly_cy) / max_dist, 1.0)
        target_diam = diam_max + (diam_min - diam_max) * t
        r = min(r_ins, target_diam / 2)

        if r * 2 < diam_min:
            continue

        circle = sp.Point(cx, cy).buffer(r, resolution=32)
        placed.append((cx, cy, r))

        remaining = region.difference(circle)
        if remaining.is_empty:
            continue

        geoms = list(remaining.geoms) if hasattr(remaining, 'geoms') else [remaining]
        stack.extend(g for g in geoms if not g.is_empty)

    return placed


_cx, _cy, _ins_r = chebyshev_center(star_smooth)
_ins_diam = _ins_r * 2
print(f"Star inscribed diameter: {_ins_diam:.4f}")
print(f"  DIAM_MAX={DIAM_MAX}  →  effective max = {min(DIAM_MAX, _ins_diam):.4f}  "
      f"({'geometry-limited' if DIAM_MAX > _ins_diam else 'param-limited'})")
print(f"  DIAM_MIN={DIAM_MIN}  →  {'OK' if DIAM_MIN < _ins_diam else 'TOO LARGE — 0 circles'}")

packed = apollonian_pack(star_smooth, DIAM_MAX, DIAM_MIN)
print(f"Packed {len(packed)} circles  "
      f"(diam {DIAM_MAX:.4f} → {DIAM_MIN:.4f})")

placed_circles = [sp.Point(x, y).buffer(r, resolution=32) for x, y, r in packed]

# ── Visualisation ─────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

for ax, poly, title in [
    (axes[0], star,        "Raw (trapezoid)"),
    (axes[1], star_smooth, f"Smoothed  (factor={SMOOTH_FACTOR}, iter={SMOOTH_ITER})"),
]:
    sx, sy = poly.exterior.xy
    ax.fill(sx, sy, alpha=0.15, color="steelblue")
    ax.plot(sx, sy, color="steelblue", linewidth=1.5)

    for r, ls in [(INNER_R, "--"), (OUTER_R, ":")]:
        theta = np.linspace(0, 2 * np.pi, 300)
        ax.plot(r * np.cos(theta), r * np.sin(theta),
                color="gray", linestyle=ls, linewidth=0.8)

    for circ in placed_circles:
        if circ.is_empty:
            continue
        cx2, cy2 = circ.exterior.xy
        ax.fill(cx2, cy2, alpha=0.4, color="tomato")
        ax.plot(cx2, cy2, color="darkred", linewidth=0.5)

    ax.set_aspect("equal")
    ax.set_title(title)

plt.suptitle(
    f"{N_BRANCHES} branches  |  inner={INNER_R}  outer={OUTER_R}  "
    f"arc_base={ARC_BASE}  arc_top={ARC_TOP}"
)
plt.tight_layout()

# ── Packing plot (separate window) ────────────────────────────────────────

fig2, ax2 = plt.subplots(figsize=(7, 7))

sx, sy = star_smooth.exterior.xy
ax2.fill(sx, sy, alpha=0.08, color="steelblue")
ax2.plot(sx, sy, color="steelblue", linewidth=1.0, linestyle="--", label="star (smooth)")

norm = plt.Normalize(DIAM_MIN, DIAM_MAX)
cmap = plt.cm.plasma
for circ, (px, py, r) in zip(placed_circles, packed):
    ex, ey = circ.exterior.xy
    ax2.fill(ex, ey, alpha=0.75, color=cmap(norm(r * 2)))
    ax2.plot(ex, ey, color="k", linewidth=0.3)
    ax2.plot(px, py, "k+", markersize=3)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
plt.colorbar(sm, ax=ax2, label="diameter")
ax2.set_aspect("equal")
ax2.set_title(f"Apollonian packing — {len(packed)} circles")
ax2.legend(fontsize=8)
plt.tight_layout()
plt.show()
