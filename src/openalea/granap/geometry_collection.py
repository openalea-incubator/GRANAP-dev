"""
Geometry processor module for handling polygon operations.
"""

import numpy as np
import shapely as sp
from typing import Tuple, List, Optional
from shapely.geometry import Point, Polygon, MultiPolygon, GeometryCollection
from cv2 import fitEllipse
from scipy.optimize import minimize
from scipy.spatial import Delaunay, ConvexHull
from shapely.ops import unary_union

from openalea.granap.math_functions import GRADIENT_FUNCTIONS, rescale


class GeometryProcessor:
    """
    Handles all geometric operations for anatomy generation.
    
    Provides methods for creating base shapes, buffering, smoothing,
    and coordinate manipulation.
    """
    
    @staticmethod
    def half_ellipse_polygon(width: float, height: float, n_points: int = 1000) -> Polygon:
        """
        Generate a polygon representing the upper half of an ellipse.
        
        Args:
            width: Width of the ellipse
            height: Height of the ellipse
            n_points: Number of points for discretization
        
        Returns:
            Shapely Polygon representing half ellipse
        """
        x = np.linspace(-width/2, width/2, n_points)
        y = height * np.sqrt(1 - (x / (width/2))**2)
        polygon = np.column_stack((x, y))
        return sp.Polygon(polygon)
    
    @staticmethod
    def circle_polygon(radius: float, n_points: int = 1000) -> Polygon:
        """
        Generate a circular polygon.
        
        Args:
            radius: Radius of the circle
            n_points: Number of points for discretization
        
        Returns:
            Shapely Polygon representing circle
        """
        theta = np.linspace(0, 2*np.pi, n_points)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        return sp.Polygon(np.column_stack((x, y)))
    
    @staticmethod
    def star_polygon(
        n_branches: int,
        r_min: float,
        r_max: float,
        arc_base: float,
        arc_top: float,
        n_arc: int = 200,
    ) -> Polygon:
        """
        Generate a star-shaped polygon as a single continuous boundary.

        Alternates outer arcs at r_max (peaks) with inner arcs at r_min (valleys),
        connected by straight sides. No boolean operations.

        Args:
            n_branches: Number of branches (>= 2)
            r_min:      Inner radius between branches
            r_max:      Outer radius at branch tips (> r_min)
            arc_base:   Half arc length at r_min (base width of each branch)
            arc_top:    Half arc length at r_max (tip width of each branch)
            n_arc:      Number of points per arc segment
        """
        w_base = arc_base / r_min
        w_top  = arc_top  / r_max

        segments = []
        for k in range(n_branches):
            tk     = 2 * np.pi * k / n_branches
            t_next = 2 * np.pi * (k + 1) / n_branches
            outer  = np.linspace(tk - w_top,  tk + w_top,      n_arc)
            valley = np.linspace(tk + w_base, t_next - w_base, n_arc)
            segments.append([[r_min * np.cos(tk - w_base), r_min * np.sin(tk - w_base)]])
            segments.append(np.column_stack([r_max * np.cos(outer),  r_max * np.sin(outer)]))
            segments.append([[r_min * np.cos(tk + w_base), r_min * np.sin(tk + w_base)]])
            segments.append(np.column_stack([r_min * np.cos(valley), r_min * np.sin(valley)]))

        return sp.Polygon(np.vstack(segments))


    @staticmethod
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
    
    @staticmethod
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
        coords = GeometryProcessor.resample_coords(coords, target_n_points=200)
        
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
    
    @staticmethod
    def buffer_polygon(polygon: Polygon, distance: float, 
                      smooth_factor: float = 0) -> Polygon:
        """
        Buffer a polygon with optional smoothing.
        
        Args:
            polygon: Input polygon
            distance: Buffer distance (positive = expand, negative = shrink)
            smooth_factor: Smoothing strength (0 = no smoothing)
        
        Returns:
            Buffered (and optionally smoothed) polygon
        """
        polygon_buffered = polygon.buffer(distance, resolution=16)
        
        if smooth_factor > 0:
            if hasattr(polygon_buffered, 'exterior'):
                x, y = np.array(polygon_buffered.exterior.coords.xy)
                coords = np.column_stack((x, y))
            elif hasattr(polygon_buffered, 'geoms'):
                coords = np.array(polygon_buffered.geoms[0].exterior.coords.xy)
            else:
                coords = np.array(polygon_buffered.exterior.coords.xy)
            if coords.size == 0:
                return polygon_buffered
            else:
                coords_smooth = GeometryProcessor.smoothing_polygon(
                    coords, smooth_factor
                )
                return sp.Polygon(coords_smooth)
        else:
            return polygon_buffered

    @staticmethod
    def union_polygons(polygons: List[Polygon]) -> Polygon:
        """
        Union a list of polygons.
        
        Args:
            polygons: List of polygons
        
        Returns:
            Union of all polygons
        """
        return sp.ops.unary_union(polygons)
    
    @staticmethod
    def difference_polygons(polygon1: Polygon, polygon2: Polygon):
        """
        Difference two polygons.
        
        Args:
            polygon1: First polygon
            polygon2: Second polygon
        
        Returns:
            Difference of the two polygons
        """
        return polygon1.difference(polygon2)
    
    @staticmethod
    def draw_ellipse(center: Tuple[float, float], axis: float, 
                    major_axis: float, minor_axis: float, 
                    n_points: int = 5) -> np.ndarray:
        """
        Generate points along an ellipse boundary.
        
        Args:
            center: (x, y) center coordinates
            axis: Rotation angle in radians
            major_axis: Major axis length
            minor_axis: Minor axis length
            n_points: Number of points to generate
        
        Returns:
            Array of (x, y) coordinates
        """
        t = np.linspace(0, 2*np.pi, n_points)
        x = center[0] + major_axis * np.cos(t) * np.cos(axis) - \
            minor_axis * np.sin(t) * np.sin(axis)
        y = center[1] + major_axis * np.cos(t) * np.sin(axis) + \
            minor_axis * np.sin(t) * np.cos(axis)
        return np.column_stack((x, y))

    @staticmethod
    def ellipse_to_polygon(cx, cy, rx, ry, angle):
        """
        Create a polygon for an ellipse 
        """
        circle = sp.Point(0, 0).buffer(1)
        ellipse = sp.affinity.scale(circle, rx, ry, origin=(0, 0))   
        ellipse = sp.affinity.rotate(ellipse, angle, origin=(0, 0))
        ellipse = sp.affinity.translate(ellipse, cx, cy)
        
        return ellipse

    @staticmethod
    def get_chebyshev_center(polygon):
        """
        Finds the approximate center of the Maximum Inscribed Circle (Pole of Inaccessibility).
        """
        cx, cy, _ = GeometryProcessor.get_inscribed_circle(polygon)
        return cx, cy

    @staticmethod
    def get_inscribed_circle(polygon) -> Tuple[float, float, float]:
        """
        Return (cx, cy, radius) of the largest circle that fits inside *polygon*.

        Uses the same binary-search erosion as get_chebyshev_center but also
        exposes the inscribed-circle radius so callers can reason about available
        space without running the search twice.
        """
        try:
            min_x, min_y, max_x, max_y = polygon.bounds
            lb = 0.0
            ub = min(max_x - min_x, max_y - min_y) / 2.0

            for _ in range(20):
                mid = (lb + ub) / 2.0
                if polygon.buffer(-mid).is_empty:
                    ub = mid
                else:
                    lb = mid

            deepest = polygon.buffer(-lb * 0.99)
            if deepest.is_empty:
                return polygon.centroid.x, polygon.centroid.y, lb
            return deepest.centroid.x, deepest.centroid.y, lb
        except Exception:
            return polygon.centroid.x, polygon.centroid.y, 0.0

    @staticmethod
    def _chebyshev_center(polygon, grid_n: int = 15) -> Tuple[float, float, float]:
        """
        Pole of inaccessibility (largest inscribed circle): ``(cx, cy, radius)``.

        Computed by GEOS via :func:`shapely.maximum_inscribed_circle` — a single
        C call returning the centre→boundary segment.  This replaced a Python
        grid search + scipy Nelder-Mead refinement that ran once per packed
        circle and dominated ``pack_circles``.  The result differs from the old
        approximation at the tolerance level, so it is a deliberate, golden-
        rebaselined change (not byte-identical).  ``grid_n`` is accepted for
        backward compatibility but unused.
        """
        if polygon.is_empty or polygon.area <= 0.0:
            c = polygon.centroid
            return c.x, c.y, 0.0

        minx, miny, maxx, maxy = polygon.bounds
        tolerance = max(maxx - minx, maxy - miny) * 1e-4 or 1e-6
        try:
            line = sp.maximum_inscribed_circle(polygon, tolerance=tolerance)
            (cx, cy), _boundary_pt = line.coords
            return cx, cy, line.length
        except Exception:
            c = polygon.centroid
            return c.x, c.y, 0.0

    @staticmethod
    def pack_circles(
        polygon,
        proportion:          float                       = 1.0,
        direction:           Optional[str]               = "center",
        diameter_max:        float                       = 0.1,
        diameter_min:        Optional[float]             = None,
        diameter_sd:         float                       = 0.0,
        gradient_function:   str                         = "five_pl",
        gradient_inflection: float                       = 0.5,
        gradient_steepness:  float                       = 3.0,
        gradient_asymmetry:  float                       = 1.0,
        first_circle_shift:  float                       = 0.0,
        adjacent:            bool                        = False,
        gradient_center:     Optional[Tuple[float, float]] = None,
        rng                                              = None,
    ) -> List[Tuple[float, float, float]]:
        """
        Unified Apollonian circle packing with proportion stop, directional gradient,
        optional diameter noise, and adjacency constraint.

        Args:
            polygon:             Shapely polygon to fill.
            proportion:          Stop when filled_area / polygon_area >= proportion.
            direction:           Size gradient: "center" (large→small outward),
                                 "edge" (large→small inward), "middle" (large at mid-radius),
                                 None (no spatial gradient; size drawn randomly per circle).
            diameter_max:        Maximum circle diameter.
            diameter_min:        Minimum circle diameter.  Defaults to
                                 max(diameter_max - 3*diameter_sd, diameter_max * 0.01).
            diameter_sd:         Per-circle diameter noise std-dev.  For direction=None with
                                 gradient_function="normal" this is the sampling std-dev.
            gradient_function:   "five_pl" | "linear" (spatial) or "normal" | "uniform" (random).
                                 "gaussian" is accepted as an alias for "normal".
            gradient_inflection: Inflection point in [0, 1] for five_pl.
            gradient_steepness:  Hill coefficient for five_pl.
            gradient_asymmetry:  Asymmetry exponent for five_pl.
            first_circle_shift:  Max random shift of the first circle centre as a fraction
                                 of its inscribed radius.  0.0 = deterministic.
            adjacent:            If True, every circle after the first must be tangent to at
                                 least one already-placed circle.
            gradient_center:     Optional (x, y) reference point for computing the gradient
                                 position t.  When None (default) the polygon centroid is
                                 used.  Pass the stele/organ centre when the polygon is a
                                 sub-region (e.g. a pizza-slice zone) so that t reflects
                                 radial distance from the true anatomical centre rather than
                                 from the zone centroid.

        Returns:
            List of (cx, cy, radius) tuples for each placed circle.
        """
        if gradient_function == "gaussian":
            gradient_function = "normal"

        if diameter_min is None:
            diameter_min = max(diameter_max - 3.0 * diameter_sd, diameter_max * 0.01)

        if direction in ("center", "edge", "middle") and gradient_function in GRADIENT_FUNCTIONS:
            base_fn = rescale(
                GRADIENT_FUNCTIONS[gradient_function],
                lo=diameter_min,
                hi=diameter_max,
                c=gradient_inflection,
                b=gradient_steepness,
                m=gradient_asymmetry,
            )
        else:
            base_fn = None

        poly_cx, poly_cy = polygon.centroid.x, polygon.centroid.y
        minx, miny, maxx, maxy = polygon.bounds

        if gradient_center is not None:
            ref_cx, ref_cy = gradient_center
            max_dist = max(
                np.hypot(maxx - ref_cx, maxy - ref_cy),
                np.hypot(maxx - ref_cx, miny - ref_cy),
                np.hypot(minx - ref_cx, maxy - ref_cy),
                np.hypot(minx - ref_cx, miny - ref_cy),
            )
        else:
            ref_cx, ref_cy = poly_cx, poly_cy
            max_dist = max(maxx - ref_cx, ref_cx - minx, maxy - ref_cy, ref_cy - miny)

        if max_dist < 1e-12:
            max_dist = 1.0
        poly_area = polygon.area
        _rng = rng if rng is not None else np.random

        placed: List[Tuple[float, float, float]] = []
        total_area = 0.0
        stack = [polygon]

        while stack:
            region = stack.pop()
            if region.is_empty or region.area < np.pi * (diameter_min / 2) ** 2 * (1 - 0.001):
                continue

            cx, cy, r_ins = GeometryProcessor._chebyshev_center(region)

            if not placed and first_circle_shift > 0.0:
                angle     = _rng.uniform(0.0, 2.0 * np.pi)
                magnitude = _rng.uniform(0.0, first_circle_shift * r_ins)
                new_cx    = cx + magnitude * np.cos(angle)
                new_cy    = cy + magnitude * np.sin(angle)
                if polygon.contains(Point(new_cx, new_cy)):
                    new_r_ins = polygon.exterior.distance(Point(new_cx, new_cy))
                    if new_r_ins >= diameter_min / 2:
                        cx, cy, r_ins = new_cx, new_cy, new_r_ins

            if direction is None or base_fn is None:
                if gradient_function == "normal":
                    mean_diam   = (diameter_max + diameter_min) / 2.0
                    target_diam = float(np.clip(
                        _rng.normal(mean_diam, diameter_sd), diameter_min, diameter_max
                    ))
                elif gradient_function == "uniform":
                    target_diam = float(_rng.uniform(diameter_min, diameter_max))
                else:
                    target_diam = diameter_max
            else:
                t = min(np.hypot(cx - ref_cx, cy - ref_cy) / max_dist, 1.0)
                if direction == "edge":
                    t = 1.0 - t
                elif direction == "middle":
                    t = 2.0 * abs(t - 0.5)
                target_diam = base_fn(t)
                if diameter_sd > 0.0:
                    target_diam = float(np.clip(
                        _rng.normal(target_diam, diameter_sd), diameter_min, np.inf
                    ))

            r = min(r_ins, target_diam / 2)
            if r * 2 < diameter_min * (1 - 0.001):
                continue

            if adjacent and placed:
                tol = diameter_min * 0.1
                if not any(np.hypot(cx - px, cy - py) <= r + pr + tol for px, py, pr in placed):
                    remaining = region.difference(sp.Point(cx, cy).buffer(r, resolution=32))
                    if not remaining.is_empty:
                        geoms = list(remaining.geoms) if hasattr(remaining, 'geoms') else [remaining]
                        stack.extend(g for g in geoms if not g.is_empty)
                    continue

            # Budget check: if placing r would overshoot the proportion target, shrink it
            # to exactly consume the remaining budget. If even diameter_min won't fit in
            # the budget, stop entirely.
            budget = proportion * poly_area - total_area
            if budget <= 0.0:
                break
            at_limit = False
            if np.pi * r ** 2 > budget:
                r_budget = np.sqrt(budget / np.pi)
                if r_budget >= diameter_min / 2:
                    r = r_budget
                    at_limit = True
                else:
                    break

            placed.append((cx, cy, r))
            total_area += np.pi * r ** 2

            remaining = region.difference(sp.Point(cx, cy).buffer(r, resolution=32))
            if remaining.is_empty or at_limit:
                if at_limit:
                    break
                continue
            geoms = list(remaining.geoms) if hasattr(remaining, 'geoms') else [remaining]
            stack.extend(g for g in geoms if not g.is_empty)

        return placed

    @staticmethod
    def fit_inner_ellipse(polygon, rx: Optional[float] = None, ry: Optional[float] = None, shrink_step=0.98, min_scale=0.2, debug=False):
        """
        Fit an inner ellipse to a polygon
        """
        # convert to numpy array of points
        points = np.array(polygon.exterior.coords.xy).T
        points = points.reshape(-1, 1, 2).astype(np.float32)
    
        # fit ellipse to get orientation and aspect ratio
        (cx_fit, cy_fit), (major, minor), angle = fitEllipse(points)
        
        # Use Chebyshev Center (deepest point inside) instead of fitEllipse center or Centroid
        cx, cy = GeometryProcessor.get_chebyshev_center(polygon)
    
        if rx is not None and ry is None:
            ry = rx
        rx = major / 2 if rx is None else rx
        ry = minor / 2 if ry is None else ry
        
        scale_factor_x = 1.0 
        scale_factor_y = 1.0 
        
        result_ellipse = None
    
        # Try to shrink until it fits
        while scale_factor_x > min_scale:
            ell = GeometryProcessor.ellipse_to_polygon(
                cx, cy,
                rx * scale_factor_x,
                ry * scale_factor_y,
                angle
            )
    
            if polygon.contains(ell):
                result_ellipse = {
                    "center": [cx, cy],
                    "axes": [rx * scale_factor_x, ry * scale_factor_y],
                    "angle": angle,
                    "polygon": ell
                }
                break
    
            scale_factor_x *= shrink_step
            scale_factor_y *= shrink_step*0.95
        
        if result_ellipse is None:
            # Fallback
            result_ellipse = {
                "center": [cx, cy],
                "axes": [rx * scale_factor_x, ry * scale_factor_y],
                "angle": angle,
                "polygon": ell
            }
    
        if debug:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot(*polygon.exterior.xy, label='Polygon', color='blue')
            ax.plot(*result_ellipse["polygon"].exterior.xy, label='Ellipse', color='red')
            ax.set_aspect('equal')
            plt.legend()
            plt.show()
    
        return result_ellipse

    @staticmethod
    def pizza_slice(polygon, n_slices):
        """
        Split a polygon into n slices using radial lines from the center.
        """
        cx, cy = polygon.centroid.x, polygon.centroid.y
        slices = []
        
        # Determine a large enough radius to cover the polygon
        minx, miny, maxx, maxy = polygon.bounds
        radius = max(maxx - minx, maxy - miny) * 2
        
        for i in range(n_slices):
            angle_start = 2 * np.pi * i / n_slices
            angle_end = 2 * np.pi * (i + 1) / n_slices
            
            # Create a wedge polygon
            # Points: center, point at angle_start, point at angle_end
            p1 = (cx + radius * np.cos(angle_start), cy + radius * np.sin(angle_start))
            p2 = (cx + radius * np.cos(angle_end), cy + radius * np.sin(angle_end))
            
            wedge = sp.Polygon([(cx, cy), p1, p2])
            
            slice_polygon = polygon.intersection(wedge)
            if not slice_polygon.is_empty:
                slices.append(slice_polygon)
                
        return slices
    

    @staticmethod
    def two_ellipses(polygon, rx, ry):
        # vertical splitting line (make it long enough to fully cross the polygon)
        center = polygon.centroid
    
        # Define the splitting rectangle
        split_rect = sp.box(
            center.x + 0.1*polygon.bounds[0],          # minx
            polygon.bounds[1] - 10, # miny
            center.x + 0.1*polygon.bounds[2],          # maxx
            polygon.bounds[3] + 10  # maxy
        )
    
        # Get the parts of the polygon outside the rectangle
        outside_polygon = polygon.difference(split_rect)
    
        # Split the outside polygon into left and right parts
        if outside_polygon.geom_type == "MultiPolygon":
            parts = list(outside_polygon.geoms)
        else:
            parts = [outside_polygon]
    
        if isinstance(parts, GeometryCollection):
            parts = list(parts.geoms)
    
        if len(parts) != 2:
            raise ValueError("Polygon was not split into two parts")
    
        # assign left / right based on centroid x
        left_poly, right_poly = sorted(
            parts,
            key=lambda p: p.centroid.x
        )
    
        ellipses = []
    

        ellipses.append(GeometryProcessor.fit_inner_ellipse(left_poly.buffer(-0.002), rx, ry))
        ellipses.append(GeometryProcessor.fit_inner_ellipse(right_poly.buffer(-0.002), rx, ry))
    
        return ellipses

