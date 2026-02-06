
import numpy as np
from shapely.geometry import Point, Polygon
import shapely.affinity as affinity
from cv2 import fitEllipse
import matplotlib.pyplot as plt

def ellipse_to_polygon(cx, cy, rx, ry, angle):
    """
    Create a polygon for an ellipse 
    """
    circle = Point(0, 0).buffer(1)
    ellipse = affinity.scale(circle, rx, ry, origin=(0, 0))   
    ellipse = affinity.rotate(ellipse, angle, origin=(0, 0))
    ellipse = affinity.translate(ellipse, cx, cy)
    
    return ellipse

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

def fit_inner_ellipse(polygon, shrink_step=0.98, min_scale=0.2, debug=True):
    """
    Fit an inner ellipse to a polygon
    """
    # convert to numpy array of points
    points = np.array(polygon.exterior.coords.xy).T
    points = points.reshape(-1, 1, 2).astype(np.float32)

    # fit ellipse to get orientation and aspect ratio
    (cx_fit, cy_fit), (major, minor), angle = fitEllipse(points)
    
    # Use Chebyshev Center (deepest point inside) instead of fitEllipse center or Centroid
    cx, cy = get_chebyshev_center(polygon)
    
    rx = major / 2
    ry = minor / 2
    
    scale_factor_x = 1.0 
    scale_factor_y = 1.0 
    
    result_ellipse = None

    # Try to shrink until it fits
    while scale_factor_x > min_scale:
        ell = ellipse_to_polygon(
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
        # plot major axis
        ax.plot([result_ellipse["center"][0] + result_ellipse["axes"][0] * np.sin(result_ellipse["angle"]), result_ellipse["center"][0] - result_ellipse["axes"][0] * np.sin(result_ellipse["angle"])],
                [result_ellipse["center"][1] - result_ellipse["axes"][1] * np.cos(result_ellipse["angle"]), result_ellipse["center"][1] + result_ellipse["axes"][1] * np.cos(result_ellipse["angle"])],
                label='Major Axis', color='green')
        ax.set_aspect('equal')
        plt.legend()
        plt.show()

    return result_ellipse

if __name__ == "__main__":
    # Test D-shape
    angles = np.linspace(-np.pi/2, np.pi/2, 50)
    radius = 0.04
    center_y = 0.2
    arc_x = radius * np.cos(angles)
    arc_y = center_y + radius * np.sin(angles)

    arc_y = [y if y > 0.2 else 0.2 for y in arc_y]
    pts = np.column_stack((arc_x, arc_y))
    pts = np.vstack([pts, pts[0]])
    poly = Polygon(pts)
    
    fit_inner_ellipse(poly, debug=True)
