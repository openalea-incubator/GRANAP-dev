"""
Geometry processor module for handling polygon operations.
"""

import numpy as np
import shapely as sp
from typing import Tuple, List, Optional
from shapely.geometry import Point, Polygon, MultiPolygon, GeometryCollection
from cv2 import fitEllipse


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
            x, y = np.array(polygon_buffered.exterior.coords.xy)
            coords = np.column_stack((x, y))
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
        try:
            # Initial bounds for binary search
            min_x, min_y, max_x, max_y = polygon.bounds
            lb = 0.0
            ub = min(max_x - min_x, max_y - min_y) / 2.0
            
            # Binary search for the largest buffer distance that isn't empty
            for _ in range(15):
                mid = (lb + ub) / 2.0
                if polygon.buffer(-mid).is_empty:
                    ub = mid
                else:
                    lb = mid
            
            # Get the centroid of the deepest valid erosion
            deepest = polygon.buffer(-lb * 0.99)
            if deepest.is_empty:
                return polygon.centroid.x, polygon.centroid.y
            else:
                return deepest.centroid.x, deepest.centroid.y
        except:
            return polygon.centroid.x, polygon.centroid.y

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




