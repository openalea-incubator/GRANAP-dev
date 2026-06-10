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
        Generate a star-shaped polygon using geometric trapezoid branches.

        Each branch is a trapezoid: inner arc at r_min (half-angle arc_base/r_min)
        connected by straight sides to outer arc at r_max (half-angle arc_top/r_max).
        A central disk at r_min fills the space between branches.
        arc_base and arc_top are fully independent — no solver, never crashes.

        Args:
            n_branches: Number of branches (>= 2)
            r_min:      Inner radius at the base of the branches
            r_max:      Outer radius at the tip of the branches (> r_min)
            arc_base:   Arc length at r_min (base width)
            arc_top:    Arc length at r_max (tip width)
            n_arc:      Number of points per arc edge

        Returns:
            Shapely Polygon representing the star
        """
        w_base = arc_base / r_min
        w_top  = arc_top  / r_max

        # Branches extend slightly inside the disk so they genuinely overlap it.
        # Exact tangency (inner arc == disk edge) triggers a GEOS TopologyException.
        r_inner = r_min * (1.0 - 1e-4)

        branches = []
        for k in range(n_branches):
            tk = 2 * np.pi * k / n_branches
            inner = np.linspace(tk - w_base, tk + w_base, n_arc)
            outer = np.linspace(tk - w_top,  tk + w_top,  n_arc)
            pts = np.vstack([
                np.column_stack([r_inner * np.cos(inner), r_inner * np.sin(inner)]),
                np.column_stack([r_max * np.cos(outer[::-1]), r_max * np.sin(outer[::-1])]),
            ])
            branches.append(sp.Polygon(pts))

        return unary_union([sp.Point(0, 0).buffer(r_min, resolution=64)] + branches)


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
        Pole of inaccessibility via grid search + Nelder-Mead refinement.
        Returns (cx, cy, radius) of the largest inscribed circle.
        """
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

    @staticmethod
    def apollonian_pack(
        polygon,
        diam_max: float,
        diam_min: float,
        gradient_inflection: float = 0.5,
        gradient_steepness: float = 3.0,
        gradient_asymmetry: float = 1.0,
        first_vessel_shift: float = 0.0,
    ) -> List[Tuple[float, float, float]]:
        """
        Iterative Apollonian packing with a centre-to-edge size gradient.

        Circles near the polygon centroid get up to diam_max; circles near the
        boundary get down to diam_min. Target diameter follows a 5-parameter
        logistic (5PL) curve of normalised distance t from the centroid:

            f(t) = diam_min + (diam_max - diam_min) / (1 + (t / c)^b)^m

        where c = gradient_inflection, b = gradient_steepness, m = gradient_asymmetry.

        Args:
            polygon:              Shapely polygon to fill
            diam_max:             Maximum circle diameter (at centroid, t=0)
            diam_min:             Minimum circle diameter (at boundary, t=1)
            gradient_inflection:  5PL inflection point on [0, 1] (c parameter)
            gradient_steepness:   5PL Hill coefficient — sharpness of transition (b)
            gradient_asymmetry:   5PL asymmetry exponent (m)
            first_vessel_shift:   Maximum random shift of the first vessel centre,
                                  expressed as a fraction of the local inscribed radius.
                                  0.0 = deterministic (always at the Chebyshev centre).

        Returns:
            List of (cx, cy, radius) tuples for each placed circle
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

            cx, cy, r_ins = GeometryProcessor._chebyshev_center(region)

            # Random shift for the first vessel only, bounded so the centre
            # stays inside the polygon and the inscribed radius is recomputed.
            # The shifted position is accepted only when r_ins at the new location
            # is still >= diam_min/2; otherwise fall back to the Chebyshev centre
            # to avoid skipping the entire region.
            if not placed and first_vessel_shift > 0.0:
                angle = np.random.uniform(0.0, 2.0 * np.pi)
                magnitude = np.random.uniform(0.0, first_vessel_shift * r_ins)
                new_cx = cx + magnitude * np.cos(angle)
                new_cy = cy + magnitude * np.sin(angle)
                if polygon.contains(Point(new_cx, new_cy)):
                    new_r_ins = polygon.exterior.distance(Point(new_cx, new_cy))
                    if new_r_ins >= diam_min / 2:
                        cx, cy = new_cx, new_cy
                        r_ins = new_r_ins

            t = min(np.hypot(cx - poly_cx, cy - poly_cy) / max_dist, 1.0)
            if t <= 0.0:
                target_diam = float(diam_max)
            else:
                target_diam = float(
                    diam_min + (diam_max - diam_min)
                    / (1.0 + (t / gradient_inflection) ** gradient_steepness) ** gradient_asymmetry
                )
            r = min(r_ins, target_diam / 2)

            if r * 2 < diam_min:
                continue

            placed.append((cx, cy, r))

            remaining = region.difference(sp.Point(cx, cy).buffer(r, resolution=32))
            if remaining.is_empty:
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
    def _front_chain_positions(r: float, n: int) -> List[np.ndarray]:
        """
        Compute centres of n equal circles of radius r using a greedy front-chain
        algorithm (similar to the circlify / D3 circle-pack approach).

        Each new circle is placed tangent to the adjacent pair in the current front
        chain that minimises the resulting enclosing-circle radius.  The returned
        centres are re-centred on their mean.
        """
        D = 2.0 * r  # distance between touching circle centres

        if n == 1:
            return [np.zeros(2)]
        if n == 2:
            return [np.array([-r, 0.0]), np.array([r, 0.0])]

        cx_list = [-r, r]
        cy_list = [0.0, 0.0]
        # Circular doubly-linked list for the front chain
        nxt = [1, 0]
        prv = [1, 0]

        def _place(a, b):
            """Outer-tangent position for a new circle touching circles a and b."""
            ax, ay = cx_list[a], cy_list[a]
            bx, by = cx_list[b], cy_list[b]
            dx, dy = bx - ax, by - ay
            d = np.hypot(dx, dy)
            if d < 1e-12 or d > 2 * D + 1e-9:
                return None
            theta = np.arctan2(dy, dx)
            # Law of cosines: angle at a in triangle (a, b, new) with all sides = D
            alpha = np.arccos(np.clip(d / (2 * D), -1.0, 1.0))
            return ax + D * np.cos(theta + alpha), ay + D * np.sin(theta + alpha)

        def _ok(px, py):
            return all(np.hypot(px - cx_list[i], py - cy_list[i]) >= D - 1e-6
                       for i in range(len(cx_list)))

        for _ in range(2, n):
            best_pos = None
            best_R   = float('inf')
            best_a   = 0

            a = 0
            for _step in range(len(cx_list) + 2):
                b = nxt[a]
                p = _place(a, b)
                if p is not None and _ok(*p):
                    R = max(
                        max(np.hypot(cx_list[i], cy_list[i]) for i in range(len(cx_list))),
                        np.hypot(p[0], p[1]),
                    ) + r
                    if R < best_R:
                        best_R, best_pos, best_a = R, p, a
                a = b
                if a == 0:
                    break

            if best_pos is None:
                # Fallback: spiral outward from the last placed circle
                ang = np.arctan2(cy_list[-1], cx_list[-1]) if np.hypot(cx_list[-1], cy_list[-1]) > 1e-9 else 0.0
                best_pos = (cx_list[-1] + D * np.cos(ang), cy_list[-1] + D * np.sin(ang))
                best_a = len(cx_list) - 1

            k = len(cx_list)
            cx_list.append(best_pos[0])
            cy_list.append(best_pos[1])
            nxt.append(0)
            prv.append(0)

            # Insert k between best_a and its successor in the front chain
            b = nxt[best_a]
            nxt[best_a] = k;  prv[k] = best_a
            nxt[k] = b;       prv[b] = k

            # Prune: if the new circle and nxt[b] are already touching, b is
            # enclosed and can be removed from the front chain.
            c_ = nxt[b]
            if c_ != best_a and np.hypot(cx_list[k] - cx_list[c_], cy_list[k] - cy_list[c_]) < D + 1e-6:
                nxt[k] = c_
                prv[c_] = k

        # Re-centre on mean
        mx = sum(cx_list) / len(cx_list)
        my = sum(cy_list) / len(cy_list)
        return [np.array([x - mx, y - my]) for x, y in zip(cx_list, cy_list)]

    @staticmethod
    def _front_chain_positions_variable(radii: List[float]) -> List[np.ndarray]:
        """
        Compute centres of n circles with potentially different radii using a greedy
        front-chain algorithm.  Circle k is placed tangent to the pair (a, b) on the
        current front that minimises the resulting enclosing-circle radius.

        The returned centres are re-centred on their mean position.
        """
        n = len(radii)
        if n == 1:
            return [np.zeros(2)]

        r0, r1 = radii[0], radii[1]
        # Place circle 0 at x = -r0 and circle 1 at x = r1 so they are tangent at the origin
        cx_list = [-r0, r1]
        cy_list = [0.0, 0.0]
        nxt = [1, 0]
        prv = [1, 0]

        def _place(a, b, r_new):
            """Outer-tangent position for a circle of radius r_new touching circles a and b."""
            ax, ay = cx_list[a], cy_list[a]
            bx, by = cx_list[b], cy_list[b]
            dx, dy = bx - ax, by - ay
            d = np.hypot(dx, dy)
            da = radii[a] + r_new   # required distance from centre a to new centre
            db = radii[b] + r_new   # required distance from centre b to new centre
            if d < 1e-12 or d > da + db + 1e-9:
                return None
            # Law of cosines: cos(alpha) at vertex a in triangle (a, new, b)
            cos_alpha = np.clip((d * d + da * da - db * db) / (2.0 * d * da), -1.0, 1.0)
            alpha = np.arccos(cos_alpha)
            theta = np.arctan2(dy, dx)
            return ax + da * np.cos(theta + alpha), ay + da * np.sin(theta + alpha)

        def _ok(px, py, r_new):
            return all(
                np.hypot(px - cx_list[i], py - cy_list[i]) >= radii[i] + r_new - 1e-6
                for i in range(len(cx_list))
            )

        for k_new in range(2, n):
            r_new = radii[k_new]
            best_pos = None
            best_R   = float('inf')
            best_a   = 0

            a = 0
            for _step in range(len(cx_list) + 2):
                b = nxt[a]
                p = _place(a, b, r_new)
                if p is not None and _ok(*p, r_new):
                    R = max(
                        max(np.hypot(cx_list[i], cy_list[i]) + radii[i] for i in range(len(cx_list))),
                        np.hypot(p[0], p[1]) + r_new,
                    )
                    if R < best_R:
                        best_R, best_pos, best_a = R, p, a
                a = b
                if a == 0:
                    break

            if best_pos is None:
                # Fallback: place the new circle outward from the last placed one
                ang = np.arctan2(cy_list[-1], cx_list[-1]) if np.hypot(cx_list[-1], cy_list[-1]) > 1e-9 else 0.0
                d_fallback = radii[len(cx_list) - 1] + r_new
                best_pos = (cx_list[-1] + d_fallback * np.cos(ang), cy_list[-1] + d_fallback * np.sin(ang))
                best_a = len(cx_list) - 1

            k = len(cx_list)
            cx_list.append(best_pos[0])
            cy_list.append(best_pos[1])
            nxt.append(0)
            prv.append(0)

            b = nxt[best_a]
            nxt[best_a] = k;  prv[k] = best_a
            nxt[k] = b;       prv[b] = k

            c_ = nxt[b]
            if c_ != best_a and np.hypot(cx_list[k] - cx_list[c_], cy_list[k] - cy_list[c_]) < radii[k] + radii[c_] + 1e-6:
                nxt[k] = c_
                prv[c_] = k

        mx = sum(cx_list) / len(cx_list)
        my = sum(cy_list) / len(cy_list)
        return [np.array([x - mx, y - my]) for x, y in zip(cx_list, cy_list)]

    @staticmethod
    def pack_circles_variable(cell_diameters: List[float],
                              n_points: int = 64) -> Tuple[List[Polygon], Polygon]:
        """
        Pack circles with individually sampled diameters using the variable-radius
        front-chain algorithm.

        Args:
            cell_diameters: Diameter of each individual circle (one value per cell)
            n_points: Number of polygon vertices per circle

        Returns:
            (small_circles, parent_circle)
            - small_circles: list of Polygon, one per cell, centred at origin
            - parent_circle: Polygon of the minimum enclosing circle (at origin)
        """
        radii = [d / 2.0 for d in cell_diameters]
        positions = GeometryProcessor._front_chain_positions_variable(radii)

        parent_r = max(np.hypot(p[0], p[1]) + r for p, r in zip(positions, radii))

        theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        ux, uy = np.cos(theta), np.sin(theta)

        small_circles = [
            sp.Polygon(np.column_stack((p[0] + r * ux, p[1] + r * uy)))
            for p, r in zip(positions, radii)
        ]
        parent_circle = sp.Polygon(np.column_stack((parent_r * ux, parent_r * uy)))

        return small_circles, parent_circle

    @staticmethod
    def pack_circles(cell_diameter: float, n_cells: int,
                     n_points: int = 64) -> Tuple[List[Polygon], Polygon]:
        """
        Pack n_cells equal circles of cell_diameter using a front-chain algorithm.

        The parent (enclosing) circle size is derived from the packing — it is the
        minimum circle that contains all placed cells.  All geometry is centred at
        the origin; translate to the desired bundle position after calling.

        Args:
            cell_diameter: Diameter of each individual circle to pack
            n_cells: Number of circles
            n_points: Number of polygon vertices per circle

        Returns:
            (small_circles, parent_circle)
            - small_circles: list of Polygon, one per cell, centred at origin
            - parent_circle: Polygon of the minimum enclosing circle (at origin)
        """
        r = cell_diameter / 2.0
        positions = GeometryProcessor._front_chain_positions(r, n_cells)

        parent_r = max(np.hypot(p[0], p[1]) for p in positions) + r

        theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        ux, uy = np.cos(theta), np.sin(theta)

        small_circles = [
            sp.Polygon(np.column_stack((p[0] + r * ux, p[1] + r * uy)))
            for p in positions
        ]
        parent_circle = sp.Polygon(np.column_stack((parent_r * ux, parent_r * uy)))

        return small_circles, parent_circle

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

