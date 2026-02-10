
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import shapely as sp
import shapely.affinity as affinity
from shapely.ops import split, unary_union
from shapely.geometry import Point, Polygon, MultiPolygon, LineString, GeometryCollection, LinearRing
from cv2 import fitEllipse

from scipy.spatial import Voronoi, voronoi_plot_2d
import geopandas as gpd

params_data = [
    # P. pinaster
    {"name": "planttype", "value": 3, "organ": "needle"}, # 1 = Monocot, 2 = Dicot, 3 = Gymnosperm
    {"name": "randomness", "value": 1.0, "smoothness": 0.3}, # 0 = No randomness, 3 = Maximum randomness; smoothness is the smoothing factor (0 = no smoothing, 1 = maximum smoothing)
    {"name": "secondarygrowth", "value": 0},
    {"name": "central_cylinder", "cell_diameter": 0.02, "layer_thickness": 0.43, "layer_length": 1.05, "transfusion_layers": 2, "transfusion_tracheids_ratio": 0.5}, # Cell diameter in millimeters
    {"name": "transfusion_tracheids", "cell_diameter": 0.05, "cell_width": 0.03},
    {"name": "transfusion_parenchyma", "cell_diameter": 0.05, "cell_width": 0.04},
    {"name": "endodermis", "cell_diameter": 0.02, "cell_width": 0.05, "n_layers": 1, "order": 3},
    {"name": "mesophyll", "cell_diameter": 0.08, "cell_width": 0.045, "n_layers": 3, "order": 4},
    {"name": "hypodermis", "cell_diameter": 0.0225, "n_layers": 2, "order": 5},
    {"name": "epidermis", "cell_diameter": 0.02, "n_layers": 1, "order": 6},
    {"name": "xylem", "n_files": 10, "cell_diameter": 0.007, "n_clusters": 4, "n_per_cluster": 3}, # Number of files
    {"name": "phloem", "n_files": 8, "cell_diameter": 0.003}, 
    {"name": "cambium", "cell_diameter": 0.003}, 
    {"name": "resin_ducts", "diameter": 0.5, "n_files": 17},
    {"name": "inter_cellular_space", "ratio": 0.5},
    {"name": "stomata", "n_files": 22, "width": 0.07},
    {"name": "Strasburger cells", "layer_diameter": 0.002, "cell_diameter": 0.05}
]


def get_needle_width(params):
    # get needle width from parameters
    needle_width = 0
    for param in params:
        if param["name"] == "central_cylinder":
            needle_width += param.get("layer_length")
        elif param["name"] == "endodermis":
            needle_width += param.get("cell_diameter") * param.get("n_layers")*2
        elif param["name"] == "mesophyll":
            needle_width += param.get("cell_diameter") * param.get("n_layers")*2
        elif param["name"] == "hypodermis":
            needle_width += param.get("cell_diameter") * param.get("n_layers")*2
        elif param["name"] == "epidermis":
            needle_width += param.get("cell_diameter")*2

    return needle_width

def get_needle_thickness(params):
    # get needle height from parameters
    needle_thickness = 0
    for param in params:
        if param["name"] == "central_cylinder":
            needle_thickness += param.get("layer_thickness")
        elif param["name"] == "endodermis":
            needle_thickness += param.get("cell_diameter") * param.get("n_layers")*2
        elif param["name"] == "mesophyll":
            needle_thickness += param.get("cell_diameter") * param.get("n_layers")*2
        elif param["name"] == "hypodermis":
            needle_thickness += param.get("cell_diameter") * param.get("n_layers")*2
        elif param["name"] == "epidermis":
            needle_thickness += param.get("cell_diameter")*2
    return needle_thickness

def half_ellipse_polygon(width, height, n_points=1000):
    """
    Generate a polygon representing the upper half of an ellipse.
    """
    # Generate points along the x-axis
    x = np.linspace(-width/2, width/2, n_points)
    
    # Calculate corresponding y values for the ellipse equation: (x/a)^2 + (y/b)^2 = 1
    # where a = width/2 and b = height
    y = height * np.sqrt(1 - (x / (width/2))**2)
    
    # Combine x and y coordinates
    polygon = np.column_stack((x, y))
    polygon = sp.Polygon(polygon)

    return polygon

def make_generic_needle(params):
    needle_width = get_needle_width(params)
    needle_thickness = get_needle_thickness(params)

    width_vascular = [p for p in params if p["name"] == "central_cylinder"][0]["layer_length"]
    thickness_vascular = [p for p in params if p["name"] == "central_cylinder"][0]["layer_thickness"]
    needle_layer = needle_width/2 - width_vascular/2
    
    needle_width = 2 * np.sqrt((width_vascular/2 + needle_layer)**2 /
                            (1-(needle_layer/needle_thickness)**2))

    polygon = half_ellipse_polygon(needle_width, needle_thickness)
    return polygon

def order_layers(params):
    # remove layers without order
    params_ordered = [param for param in params if "order" in param]
    # order layers by order
    params_ordered.sort(key=lambda x: x["order"], reverse=True)
    return params_ordered

def layer_array(params_ordered):
    # create array of layers
    layer_array = []
    for param in params_ordered:
        for i in range(param["n_layers"]):
            layer_array.append({"name": param["name"], "cell_diameter": param["cell_diameter"], })
    return layer_array

def ellipse_to_polygon(cx, cy, rx, ry, angle):
    """
    Create a polygon for an ellipse 
    """
    circle = Point(0, 0).buffer(1)
    ellipse = affinity.scale(circle, rx, ry, origin=(0, 0))   
    ellipse = affinity.rotate(ellipse, angle, origin=(0, 0))
    ellipse = affinity.translate(ellipse, cx, cy)
    
    return ellipse


def resample_coords(coords, target_n_points=200):
    # Ensure coords is a numpy array
    coords = np.array(coords)
    if len(coords) < 2:
        return coords
    # Calculate cumulative distance along the path
    dists = np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1))
    cum_dist = np.concatenate(([0], np.cumsum(dists)))
    total_len = cum_dist[-1]
    
    # Generate value space for interpolation
    # evenly spaced points
    # add a small random noise to the points to avoid having the same points
    new_dists = np.linspace(0, total_len, target_n_points)

    # Interpolate x and y
    new_x = np.interp(new_dists, cum_dist, coords[:,0])
    new_y = np.interp(new_dists, cum_dist, coords[:,1])
    
    return np.column_stack((new_x, new_y))

def smoothing_polygon(coords, smooth_factor, iterations=10):
    """
    Smooths coordinates using a periodic Laplacian smoothing (moving average).
    Resamples the polygon to ensure uniform vertex distribution.
    iterations: Number of smoothing passes.
    """
    # Resample first to ensure uniform point distribution
    coords = resample_coords(coords, target_n_points=200)

    for _ in range(iterations):
        # Identify if the polygon is closed
        is_closed = np.allclose(coords[0], coords[-1])
        
        if is_closed:
            pts = coords[:-1]
        else:
            pts = coords
    
        pts = pts.astype(float)
        
        if len(pts) < 3:
            return coords
    
        prev_pts = np.roll(pts, 3, axis=0)
        next_pts = np.roll(pts, -3, axis=0)
    
        smoothed_pts = (1 - smooth_factor) * pts + \
                       smooth_factor * (prev_pts + next_pts) / 2.0
    
        if is_closed:
            coords = np.vstack([smoothed_pts, smoothed_pts[0]])
        else:
            coords = smoothed_pts
            
    return coords

def buffer_polygon(polygon, distance, smooth_factor):
    polygon_buffered = polygon.buffer(distance, resolution=16)
    
    if smooth_factor > 0:
        # Extract coordinates
        x,y = np.array(polygon_buffered.exterior.coords.xy)
        coords = np.column_stack((x, y))
        if coords.size == 0:
            return polygon_buffered
        else:
            coords_smooth = smoothing_polygon(coords, smooth_factor)
            # Create new polygon
            polygon_smoothed = sp.Polygon(coords_smooth)
            return polygon_smoothed
    else:
        return polygon_buffered
    


def sort_vertices_by_angle(polygon):
    """
    Trie les sommets d'un polygone par angle polaire (atan2) par rapport à son centroïde.
    Retourne les coordonnées triées (sans fermer le polygone).
    """
    # Calculer le centroïde
    centroid = polygon.centroid
    cx, cy = centroid.x, centroid.y

    # Extraire les coordonnées et calculer les angles polaires
    coords = np.array(polygon.exterior.coords[:-1])  # Exclure le dernier point (doublon)
    angles = [np.atan2(y - cy, x - cx) for x, y in coords]

    # Trier les sommets par angle (ordre anti-horaire)
    sorted_indices = np.argsort(angles)
    sorted_coords = coords[sorted_indices]

    return sorted_coords

def resample_coords(coords, target_n_points=200):
    # Ensure coords is a numpy array
    coords = np.array(coords)
    if len(coords) < 2:
        return coords
    # Calculate cumulative distance along the path
    dists = np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1))
    cum_dist = np.concatenate(([0], np.cumsum(dists)))
    total_len = cum_dist[-1]
    
    # Generate value space for interpolation
    # evenly spaced points
    # add a small random noise to the points to avoid having the same points
    new_dists = np.linspace(0, total_len, target_n_points)

    # Interpolate x and y
    new_x = np.interp(new_dists, cum_dist, coords[:,0])
    new_y = np.interp(new_dists, cum_dist, coords[:,1])
    
    return np.column_stack((new_x, new_y))

def interpolate_polygons(polygon_start_coords, polygon_end_coords, alpha):
    """
    Interpole entre deux ensembles de coordonnées triées par angle polaire.
    Les deux ensembles doivent avoir le même nombre de points.
    """
    # Interpolation linéaire
    coords_interp = (1 - alpha) * polygon_start_coords + alpha * polygon_end_coords

    # Fermer le polygone en ajoutant le premier point à la fin
    closed_coords = np.vstack([coords_interp, coords_interp[0]])

    return Polygon(closed_coords)

def make_layers_polygons(layer_array, polygon_start, params, polygon_end=None):
    layers_polygons = []
    smooth_factor = [p["smoothness"] for p in params if p["name"] == "randomness"][0]

    # Si aucun polygone cible n'est fourni, utiliser le polygone de départ
    if polygon_end is None:
        polygon_end = polygon_start

    # Trier et rééchantillonner les polygones
    target_n_points = 200  # Nombre de points pour l'interpolation
    polygon_start_coords = sort_vertices_by_angle(polygon_start)
    polygon_end_coords = sort_vertices_by_angle(polygon_end)

    polygon_start_coords = resample_coords(polygon_start_coords, target_n_points)
    polygon_end_coords = resample_coords(polygon_end_coords, target_n_points)

    for i_layer, layer in enumerate(layer_array):

        alpha = i_layer / (len(layer_array) - 1) if len(layer_array) > 1 else 0
        polygon_interp = interpolate_polygons(polygon_start_coords, polygon_end_coords, alpha)

        layers_polygons.append({
            "name": layer["name"],
            "polygon": polygon_interp,
            "cell_diameter": layer["cell_diameter"],
            "id_layer": i_layer + 1
        })

    return layers_polygons


def make_needle_polygons(params):
    # 1. Order layers
    params_ordered = order_layers(params)
    
    # 3. Generate generic needle shape
    polygon_end = make_generic_needle(params)
    center_end = polygon_end.centroid
    # central cylinder
    width_vascular = [p for p in params if p["name"] == "central_cylinder"][0]["layer_length"]
    thickness_vascular = [p for p in params if p["name"] == "central_cylinder"][0]["layer_thickness"]
    polygon_start = ellipse_to_polygon(center_end.x, center_end.y, width_vascular/2, thickness_vascular/2, 0)
    
    # 4. Generate layers polygons
    layers_polygons = make_layers_polygons(layer_array(params_ordered), polygon_start, params, polygon_end)
    
    return layers_polygons



def plot_polygons(layers_polygons):
    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='box')
    for layer in layers_polygons:
        ax.plot(*layer["polygon"].exterior.xy)
    plt.show()

layers_polygons = make_needle_polygons(params_data)
plot_polygons(layers_polygons)