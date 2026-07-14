"""
Geometry processor module for handling polygon operations.
"""

import numpy as np
import shapely as sp
from typing import Tuple, List, Optional
from shapely.geometry import Point, Polygon, MultiPolygon, GeometryCollection
from shapely.affinity import translate as _shapely_translate, rotate as _shapely_rotate, scale as _shapely_scale
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
    def oriented_ellipse(tx: float, ty: float, width: float, height: float,
                         angle_deg: float, resolution: int = 64) -> Polygon:
        """Axis-aligned unit disc scaled to ``width`` x ``height``, rotated so its
        major (``height``) axis points along ``angle_deg`` (minus the 90° that maps
        the +y major axis to the radial direction), then translated to ``(tx, ty)``.

        The one source for every oriented vascular cluster ellipse (phloem
        valleys, arch phloem, proto/phloem bundles, whole vascular-bundle
        envelopes).  See :meth:`place_local` for the matching transform applied to
        a *set* of local-frame geometries.
        """
        raw = Point(0, 0).buffer(1, resolution=resolution)
        raw = _shapely_scale(raw, width / 2, height / 2)
        raw = _shapely_rotate(raw, angle_deg - 90, origin=(0, 0))
        return _shapely_translate(raw, tx, ty)

    @staticmethod
    def place_local(geoms, cx: float, cy: float, angle_deg: float):
        """Map local-frame geometries (radial axis = local +y) to their place.

        Applies the same transform as :meth:`oriented_ellipse` — ``rotate(angle_deg
        - 90, origin=0)`` then ``translate(cx, cy)`` — to every geometry in
        ``geoms``, so an envelope built at the origin with its radial axis along
        +y and all of its interior sub-zones move together to ``(cx, cy)`` at
        orientation ``angle_deg``.  Returns a list of transformed geometries.
        """
        out = []
        for g in geoms:
            g = _shapely_rotate(g, angle_deg - 90, origin=(0, 0))
            g = _shapely_translate(g, cx, cy)
            out.append(g)
        return out

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
    def oriented_star_polygon(
        n_branches: int,
        radius_peak_side: float,
        radius_valley_side: float,
        arc_peak_side: float,
        arc_valley_side: float,
        n_arc: int = 200,
    ) -> Polygon:
        """Star built from *peak-side* / *valley-side* radii and arcs.

        The star's inner radius is ``min(radius_peak_side, radius_valley_side)``
        and its outer radius (the arm tips) the ``max`` of the two. The arm
        therefore points to whichever side owns the larger radius:

        - ``radius_peak_side >= radius_valley_side``: the arm points along the
          reference (peak) direction — branch tips at ``2*pi*k/n``, no offset.
        - ``radius_valley_side > radius_peak_side``: the star is offset half a
          period so the arm falls on the valley direction instead.

        Each side's arc travels with its own radius. This keeps the peak/valley
        naming meaningful regardless of which radius is larger.

        Args:
            n_branches:         Number of branches (>= 2)
            radius_peak_side:   Radius on the peak (reference-direction) side
            radius_valley_side: Radius on the valley (half-period) side
            arc_peak_side:      Half arc length on the peak side
            arc_valley_side:    Half arc length on the valley side
            n_arc:              Number of points per arc segment
        """
        r_min = min(radius_peak_side, radius_valley_side)
        r_max = max(radius_peak_side, radius_valley_side)
        peak_is_outer = radius_peak_side >= radius_valley_side
        arc_top  = arc_peak_side   if peak_is_outer else arc_valley_side  # arc at r_max
        arc_base = arc_valley_side if peak_is_outer else arc_peak_side    # arc at r_min

        star = GeometryProcessor.star_polygon(
            n_branches=n_branches, r_min=r_min, r_max=r_max,
            arc_base=arc_base, arc_top=arc_top, n_arc=n_arc,
        )
        if not peak_is_outer:
            star = sp.affinity.rotate(star, np.pi / n_branches, origin=(0, 0), use_radians=True)
        return star

    @staticmethod
    def rectangle_polygon(width: float, height: float) -> Polygon:
        """
        Generate an axis-aligned rectangle centred on the origin.

        Args:
            width:  Total width  (x extent)
            height: Total height (y extent)

        Returns:
            Shapely Polygon for the rectangle (a square when width == height)
        """
        w, h = width / 2.0, height / 2.0
        return sp.Polygon([(-w, -h), (w, -h), (w, h), (-w, h)])

    @staticmethod
    def triangle_polygon(width: float, height: float) -> Polygon:
        """
        Generate an upward-pointing isosceles triangle centred on the origin.

        Args:
            width:  Base length (x extent)
            height: Apex height (y extent)

        Returns:
            Shapely Polygon for the triangle
        """
        w, h = width / 2.0, height / 2.0
        return sp.Polygon([(-w, -h), (w, -h), (0.0, h)])


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

        # Closedness is invariant across passes (a closed ring is re-closed each
        # pass, an open one stays open), so test it once instead of per iteration.
        is_closed = np.allclose(coords[0], coords[-1])

        for _ in range(iterations):
            pts = coords[:-1] if is_closed else coords
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
    def focus_ellipse_polygon(cx, cy, rx, ry, angle, exponent=4.0, n=200):
        """A superellipse / Lamé curve ``|x/rx|**e + |y/ry|**e = 1``.

        Named ``focus_ellipse`` because ``exponent`` acts as a "latus rectum" knob
        on top of a classic ellipse.  Same bounding box as
        :meth:`ellipse_to_polygon` (half-axes ``rx``, ``ry``), but ``exponent``
        controls how *full* the outline is — a way to enlarge the shape (bigger
        latus rectum, more area, blunter flanks) **without** changing the width or
        height:

        - ``exponent == 2`` -> a classic ellipse (area ``pi*rx*ry``, latus rectum
          ``2*ry**2/rx``);
        - ``exponent > 2``  -> fuller/blunter flanks toward a rounded rectangle
          (area grows toward ``4*rx*ry``, latus rectum increases);
        - ``exponent < 2``  -> pointier toward a diamond (latus rectum shrinks).

        The curve keeps the axis endpoints (``±rx`` on the major axis, ``±ry`` on
        the minor) fixed; only the fullness between them changes.  ``angle`` is in
        degrees, applied about the shape centre before translating to ``(cx, cy)``.
        """
        t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        p = 2.0 / float(exponent)
        x = rx * np.sign(np.cos(t)) * np.abs(np.cos(t)) ** p
        y = ry * np.sign(np.sin(t)) * np.abs(np.sin(t)) ** p
        poly = sp.Polygon(np.column_stack([x, y]))
        poly = sp.affinity.rotate(poly, angle, origin=(0, 0))
        poly = sp.affinity.translate(poly, cx, cy)
        return poly

    @staticmethod
    def fit_focus_ellipse(profile):
        """Best-fit :meth:`focus_ellipse_polygon` to a measured contour profile.

        ``profile`` is a list of ``(major_pos, minor_width)`` measurements: the
        distance from the centre along the major axis (mm) and the FULL width in
        the minor direction there (mm).  The widest point sets the semi-minor
        axis, the farthest point (the tip, width -> 0) sets the semi-major axis,
        and the single superellipse ``exponent`` is least-squares fitted to the
        interior points.

        Returns ``(semi_major, semi_minor, exponent)``.  With one interior point
        the fit is exact; with several it is a best fit (a superellipse has only
        the exponent free once the axes are fixed), and it falls back to a classic
        ellipse (exponent 2) when no interior point is given.
        """
        pts = [(float(x), float(w)) for x, w in profile]
        if len(pts) < 2:
            raise ValueError(
                "focus-ellipse profile needs >= 2 (major_pos, minor_width) points"
            )
        semi_minor = max(w for _, w in pts) / 2.0
        semi_major = max(x for x, _ in pts)
        if semi_major <= 0.0 or semi_minor <= 0.0:
            raise ValueError(
                "focus-ellipse profile needs a positive centre width and tip position"
            )

        interior = [(x, w) for x, w in pts
                    if 0.0 < x < semi_major - 1e-9 and w > 1e-9]
        if not interior:
            return semi_major, semi_minor, 2.0

        u = np.array([x / semi_major for x, _ in interior])
        v = np.array([(w / 2.0) / semi_minor for _, w in interior])
        exponents = np.linspace(0.5, 12.0, 2301)
        residuals = np.array([np.sum((u ** e + v ** e - 1.0) ** 2) for e in exponents])
        return semi_major, semi_minor, float(exponents[int(np.argmin(residuals))])

    @staticmethod
    def egg_polygon(cx, cy, a_out, a_in, b, angle, n=28):
        """Teardrop / 'violin' shape: an asymmetric oval whose widest point (half
        width ``b``) is NOT at the middle of the major axis.  Two half-ellipses
        share that waist — an ``a_out`` lobe on the +major side and an ``a_in``
        lobe on the -major side.  Its area is ``pi*b*(a_out+a_in)/2`` (independent
        of the split), i.e. the same as a symmetric ellipse of semi-major
        ``(a_out+a_in)/2`` — so the waist can be offset without changing area.

        ``angle`` (degrees) orients the +major (the ``a_out`` lobe) direction."""
        xo = np.linspace(0.0, a_out, n)
        yo = b * np.sqrt(np.clip(1.0 - (xo / max(a_out, 1e-9)) ** 2, 0.0, 1.0))
        xi = np.linspace(-a_in, 0.0, n)[:-1]
        yi = b * np.sqrt(np.clip(1.0 - (xi / max(a_in, 1e-9)) ** 2, 0.0, 1.0))
        ux = np.concatenate([xi, xo]); uy = np.concatenate([yi, yo])
        X = np.concatenate([ux, ux[::-1]])
        Y = np.concatenate([uy, -uy[::-1]])
        th = np.radians(angle)
        c, s = np.cos(th), np.sin(th)
        return sp.Polygon(np.column_stack([
            cx + c * X - s * Y, cy + s * X + c * Y]))

    @staticmethod
    def _fit_target_ellipse(region, cx, cy, r_ins, target_r, max_aspect):
        """Largest ellipse (up to the target circle's area, aspect-capped) that
        fits inside *region*, centred at ``(cx, cy)`` and elongated along the
        region's long axis.

        Used as a fallback when the inscribed circle (radius ``r_ins``) is too
        narrow to reach ``target_r`` in a tight, elongated region (e.g. a star
        arm): the semi-minor axis fills the available width (``r_ins``) while the
        semi-major grows along the region until the area matches the target
        circle ``pi*target_r**2`` or the aspect cap ``max_aspect`` is hit, then
        shrinks until it fits.

        Returns ``(semi_major, semi_minor, angle_deg)`` or ``None`` when an
        ellipse offers no meaningful gain over the inscribed circle.
        """
        if r_ins <= 0.0:
            return None
        b_ax = r_ins                                   # semi-minor fills the width
        target_area = np.pi * target_r * target_r
        a_ax = min(target_area / (np.pi * b_ax), max_aspect * b_ax)
        if a_ax <= b_ax * 1.02:                        # no meaningful elongation
            return None

        # Orientation: the long edge of the region's minimum rotated rectangle.
        try:
            xs, ys = region.minimum_rotated_rectangle.exterior.coords.xy
            p0 = np.array([xs[0], ys[0]]); p1 = np.array([xs[1], ys[1]]); p2 = np.array([xs[2], ys[2]])
            e1, e2 = p1 - p0, p2 - p1
            long_edge = e1 if e1.dot(e1) >= e2.dot(e2) else e2
            angle = float(np.degrees(np.arctan2(long_edge[1], long_edge[0])))
        except Exception:
            return None

        # Shrink the major axis until the ellipse fits inside the region.
        for _ in range(8):
            ell = GeometryProcessor.ellipse_to_polygon(cx, cy, a_ax, b_ax, angle)
            if region.contains(ell):
                return (a_ax, b_ax, angle)
            a_ax *= 0.9
            if a_ax <= b_ax * 1.02:
                break
        return None

    @staticmethod
    def _fit_radial_ellipse(region, px, py, target_r, theta_rad, max_aspect):
        """Ellipse of the target circle's area, oriented **radially** (major axis
        along ``theta_rad`` from the gradient centre), that fits inside *region*.

        Trades tangential width for radial length while preserving the area
        ``pi*target_r**2``: the semi-minor (tangential) shrinks from ``target_r``
        and the semi-major (radial) grows to compensate, up to ``max_aspect``,
        until the ellipse fits.  Returns ``(semi_major, semi_minor, angle_deg)``
        or ``None``.
        """
        ang = float(np.degrees(theta_rad))
        # Semi-minor (tangential) = local clearance to the boundary, so the
        # ellipse just fits across the arm; the major then grows radially.
        clear = region.boundary.distance(sp.Point(px, py)) * 0.98
        if clear <= 0.0:
            return None
        b_ax = min(clear, target_r)
        target_area = np.pi * target_r * target_r
        a_ax = min(target_area / (np.pi * b_ax), max_aspect * b_ax)
        if a_ax <= b_ax * 1.02:                 # no meaningful elongation
            return None
        # Shrink the radial major until the ellipse fits (e.g. it would otherwise
        # overshoot the pith / arm ends).
        for _ in range(8):
            ell = GeometryProcessor.ellipse_to_polygon(px, py, a_ax, b_ax, ang)
            if region.contains(ell):
                return (a_ax, b_ax, ang)
            a_ax *= 0.9
            if a_ax <= b_ax * 1.02:
                break
        return None

    @staticmethod
    def _pack_targets_radial(region, base_fn, cx, cy, radial_range, direction,
                             diameter_min, diameter_max, allow_ellipse,
                             ellipse_max_aspect, proportion, poly_area, rng,
                             enforce_gradient_min=0.0):
        """Size-first, gradient-driven radial packing.

        Marches outward in radius over ``radial_range = (r0, r1)``; at each radius
        the gradient prescribes the vessel diameter ``D``, and a ring of D-circles
        is placed where it fits (one per arm / spaced ~D around the circumference).
        Where a circle is too wide for the local (tangential) space, a radial
        ellipse of the same area is used instead (when ``allow_ellipse``).  This
        is the "fit the big vessel first, ellipse if needed, then go on (to
        smaller radii)" behaviour.

        ``enforce_gradient_min``: within the enforced band (gradient position
        ``tt <= enforce_gradient_min``) the vessel diameter is the gradient target
        itself (never clamped up to ``diameter_min``), so a spot that cannot hold
        a full-target vessel (as a circle or radial ellipse) is left empty rather
        than filled with a shrunken circle — "big vessels only, gaps empty".

        Returns ``(placed_records, total_area, remaining_pieces)`` — the leftover
        pieces are handed to the space-first packer for the small interstitial
        fill (hybrid).
        """
        r0, r1 = radial_range
        placed: List[Tuple] = []
        total_area = 0.0
        if r1 <= r0:
            return placed, total_area, [region]
        _rng = rng if rng is not None else np.random
        budget = proportion * poly_area
        remaining = region

        r = r0
        guard = 0
        while r < r1 and total_area < budget and guard < 10000:
            guard += 1
            t = (r - r0) / (r1 - r0)
            if direction == "edge":
                tt = 1.0 - t
            elif direction == "middle":
                tt = 2.0 * abs(t - 0.5)
            else:
                tt = t
            # In the enforced band the vessel keeps the gradient target size (no
            # clamp up to diameter_min): a spot too tight for the full target is
            # left empty below, not filled with a shrunken circle.
            enforced = tt <= enforce_gradient_min
            D = float(base_fn(tt)) if enforced else max(float(base_fn(tt)), diameter_min)
            if D <= 0.0:
                r += diameter_min
                continue

            circumference = 2.0 * np.pi * r
            # Oversample angularly so arms at arbitrary angles are hit; the
            # carve-and-recheck below dedups any overlapping candidates.
            n_pos = max(48, int(np.ceil(circumference / (0.3 * D))))
            theta0 = _rng.uniform(0.0, 2.0 * np.pi / n_pos)
            placed_here = False
            for k in range(n_pos):
                if total_area >= budget:
                    break
                theta = theta0 + 2.0 * np.pi * k / n_pos
                px = cx + r * np.cos(theta)
                py = cy + r * np.sin(theta)
                if not remaining.contains(sp.Point(px, py)):
                    continue
                circ = sp.Point(px, py).buffer(D / 2, resolution=32)
                if remaining.contains(circ):
                    geom = circ
                    rec = (px, py, D / 2)
                    area = np.pi * (D / 2) ** 2
                elif allow_ellipse:
                    fit = GeometryProcessor._fit_radial_ellipse(
                        remaining, px, py, D / 2, theta, ellipse_max_aspect)
                    if fit is None:
                        continue
                    a_ax, b_ax, ang = fit
                    geom = GeometryProcessor.ellipse_to_polygon(px, py, a_ax, b_ax, ang)
                    rec = (px, py, a_ax, b_ax, ang)
                    area = np.pi * a_ax * b_ax
                else:
                    continue
                placed.append(rec)
                placed_here = True
                total_area += area
                remaining = remaining.difference(geom)
            # A full ring of D-vessels was laid down at this radius -> the next
            # non-overlapping ring is a whole diameter out.  But when nothing fit
            # (e.g. this radius is jammed against the pith or a zone edge), step
            # out by a small increment instead of a full D so the march doesn't
            # leap over the radii where the big vessels *would* fit — which
            # otherwise leaves the whole ring to the interstitial space-fill and
            # no radial ellipse ever gets a chance.  The carve-and-recheck above
            # dedups any overlap the finer stepping might revisit.
            r += D if placed_here else max(diameter_min, 0.25 * D)

        if remaining.is_empty:
            pieces: List = []
        else:
            pieces = (list(remaining.geoms) if hasattr(remaining, "geoms") else [remaining])
            pieces = [g for g in pieces if not g.is_empty]
        return placed, total_area, pieces

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
        gradient_radial_range: Optional[Tuple[float, float]] = None,
        enforce_gradient_min: float                      = 0.0,
        allow_ellipse:       bool                        = False,
        ellipse_max_aspect:  float                       = 2.0,
        pack_strategy:       str                         = "space",
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
            enforce_gradient_min: Radial extent in [0, 1] (gradient-position space, same
                                 axis as ``gradient_inflection``) over which the gradient
                                 minimum is enforced.  Where the local gradient position
                                 ``t <= enforce_gradient_min``, no circle smaller than the
                                 gradient-prescribed target is placed — a spot too tight
                                 for the local target is left empty instead of being filled
                                 with a shrunken circle.  Beyond it (or with the default
                                 ``0.0``) the global ``diameter_min`` applies as usual, so
                                 ``0.0`` disables the constraint and ``1.0`` enforces it
                                 everywhere.  Only active for spatial gradients
                                 (``direction`` in "center"/"edge"/"middle").
            allow_ellipse:       If True, when the inscribed circle is too narrow to reach
                                 the target diameter in an elongated region, fit an
                                 area-matched ellipse (elongated along the region) instead
                                 of shrinking the circle.  Such placements are returned as
                                 5-tuples ``(cx, cy, semi_major, semi_minor, angle_deg)``;
                                 circles remain 3-tuples ``(cx, cy, r)``.
            ellipse_max_aspect:  Maximum major/minor axis ratio for ellipse fallbacks, so
                                 vessels don't become slivers (default 2.0).
            pack_strategy:       Packing order.  "space" (default) is the space-first
                                 Apollonian fill (largest inscribed circle first, sized by
                                 the gradient at that spot).  "target" is a size-first,
                                 gradient-driven *radial* pass — it places the big vessels
                                 first at the radius the gradient assigns (ellipse if the
                                 arm is too narrow), then fills the leftover space with the
                                 space-first packer for the small interstitial cells
                                 (hybrid).  "target" requires a spatial gradient.

        Returns:
            List of records, each a circle ``(cx, cy, r)`` or, for ellipse
            placements, ``(cx, cy, semi_major, semi_minor, angle_deg)``.
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

        # Each record is a circle ``(cx, cy, r)`` or, when allow_ellipse fits one,
        # an ellipse ``(cx, cy, semi_major, semi_minor, angle_deg)``.
        placed: List[Tuple] = []
        total_area = 0.0

        # Size-first pass (gradient-driven radial): place the big vessels first,
        # then hand the leftover pieces to the space-first packer below (hybrid).
        if pack_strategy == "target" and base_fn is not None:
            radial_range = gradient_radial_range or (0.0, max_dist)
            placed, total_area, stack = GeometryProcessor._pack_targets_radial(
                polygon, base_fn, ref_cx, ref_cy, radial_range, direction,
                diameter_min, diameter_max, allow_ellipse, ellipse_max_aspect,
                proportion, poly_area, _rng, enforce_gradient_min,
            )
        else:
            stack = [polygon]

        def _center_r(rec):
            """(x, y, area-equivalent radius) of a placed circle or ellipse."""
            if len(rec) == 3:
                return rec[0], rec[1], rec[2]
            return rec[0], rec[1], np.sqrt(rec[2] * rec[3])

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

            gradient_t = None
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
                dist = np.hypot(cx - ref_cx, cy - ref_cy)
                if gradient_radial_range is not None:
                    # Gradient measured within a band [r0, r1] (e.g. one annual
                    # ring) rather than across the whole region, so the size
                    # gradient resets per band: large at r0, small at r1.
                    r0, r1 = gradient_radial_range
                    t = (dist - r0) / (r1 - r0) if r1 > r0 else 0.0
                    t = min(max(t, 0.0), 1.0)
                else:
                    t = min(dist / max_dist, 1.0)
                if direction == "edge":
                    t = 1.0 - t
                elif direction == "middle":
                    t = 2.0 * abs(t - 0.5)
                gradient_t = t
                target_diam = base_fn(t)
                if diameter_sd > 0.0:
                    target_diam = float(np.clip(
                        _rng.normal(target_diam, diameter_sd), diameter_min, np.inf
                    ))

            r = min(r_ins, target_diam / 2)

            # Ellipse fallback: when the inscribed circle cannot reach the target
            # diameter because the region is narrow/elongated, fit an area-matched
            # (aspect-capped) ellipse along the region instead of shrinking.
            ell = None
            if allow_ellipse and r_ins < (target_diam / 2) * (1 - 0.001):
                ell = GeometryProcessor._fit_target_ellipse(
                    region, cx, cy, r_ins, target_diam / 2, ellipse_max_aspect)

            if ell is not None:
                a_ax, b_ax, ang = ell
                geom = GeometryProcessor.ellipse_to_polygon(cx, cy, a_ax, b_ax, ang)
                geom_area = np.pi * a_ax * b_ax
                eff_r = np.sqrt(a_ax * b_ax)            # area-equivalent radius
            else:
                geom = sp.Point(cx, cy).buffer(r, resolution=32)
                geom_area = np.pi * r ** 2
                eff_r = r

            # Within the inner gradient zone (t <= enforce_gradient_min) the local
            # floor is the gradient target, so a spot too tight for the prescribed
            # diameter is left empty rather than filled with a shrunken circle.
            # Outside that zone (or with enforce_gradient_min <= 0) the global
            # diameter_min applies as usual.  (An ellipse already meets the target
            # area, so it passes this check.)
            if (enforce_gradient_min > 0.0 and gradient_t is not None
                    and gradient_t <= enforce_gradient_min):
                local_min = target_diam
            else:
                local_min = diameter_min
            if eff_r * 2 < local_min * (1 - 0.001):
                # Refused by the gradient-min constraint (the spot could hold a
                # circle, just not one as large as the local target): leave this
                # spot empty but carve it out and keep exploring the rest of the
                # region, so zones outside the enforced band still get packed.
                # (Without this, dropping the region would also discard the outer
                # part of a region that straddles the enforced band.)
                if local_min > diameter_min and eff_r * 2 >= diameter_min * (1 - 0.001):
                    remaining = region.difference(sp.Point(cx, cy).buffer(r_ins, resolution=32))
                    if not remaining.is_empty:
                        geoms = list(remaining.geoms) if hasattr(remaining, 'geoms') else [remaining]
                        stack.extend(g for g in geoms if not g.is_empty)
                continue

            if adjacent and placed:
                tol = diameter_min * 0.1
                near = any(
                    np.hypot(cx - px, cy - py) <= eff_r + pr + tol
                    for px, py, pr in (_center_r(rec) for rec in placed)
                )
                if not near:
                    remaining = region.difference(geom)
                    if not remaining.is_empty:
                        geoms = list(remaining.geoms) if hasattr(remaining, 'geoms') else [remaining]
                        stack.extend(g for g in geoms if not g.is_empty)
                    continue

            # Budget check: if placing this shape would overshoot the proportion
            # target, scale it down to exactly consume the remaining budget. If
            # even diameter_min won't fit in the budget, stop entirely.
            budget = proportion * poly_area - total_area
            if budget <= 0.0:
                break
            at_limit = False
            if geom_area > budget:
                s = np.sqrt(budget / geom_area)
                if eff_r * s >= diameter_min / 2:
                    geom = sp.affinity.scale(geom, s, s, origin=(cx, cy))
                    eff_r *= s
                    geom_area = budget
                    if ell is not None:
                        a_ax, b_ax = a_ax * s, b_ax * s
                    else:
                        r *= s
                    at_limit = True
                else:
                    break

            if ell is not None:
                placed.append((cx, cy, a_ax, b_ax, ang))
            else:
                placed.append((cx, cy, eff_r))
            total_area += geom_area

            remaining = region.difference(geom)
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
        # fitEllipse needs >= 5 points; low-vertex polygons (e.g. a triangular
        # or square stele slice) are densified along their boundary first.
        if len(points) < 6:
            points = GeometryProcessor.resample_coords(points, target_n_points=50)
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

