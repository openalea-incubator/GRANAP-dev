
# Pine needle test 1
# New class: PineNeedleAnatomy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import shapely as sp
from CellVoronoi import cell_voro
from scipy.spatial import Voronoi
import geopandas as gpd

params_data = [
    {"name": "planttype", "value": 3}, # 1 = Monocot, 2 = Dicot, 3 = Gymnosperm
    {"name": "randomness", "value": 1.0}, # 0 = No randomness, 3 = Maximum randomness
    {"name": "secondarygrowth", "value": 0},
    {"name": "central_cylinder", "cell_diameter": 0.0063, "layer_thickness": 0.1, "layer_length": 0.3}, # Cell diameter in millimeters
    {"name": "transfusion_tracheids", "cell_diameter": 0.009},
    {"name": "transfusion_parenchyma", "cell_diameter": 0.015},
    {"name": "endodermis", "cell_diameter": 0.017, "n_layers": 1, "order": 3},
    {"name": "mesophyll", "cell_diameter": 0.05, "n_layers": 3, "order": 4},
    {"name": "hypodermis", "cell_diameter": 0.025, "n_layers": 3, "order": 5},
    {"name": "epidermis", "cell_diameter": 0.018, "n_layers": 1, "order": 6},
    {"name": "xylem", "n_files": 4, "cell_diameter": 0.005, "n_clusters": 3, "n_per_clusters": 3}, # Number of files
    {"name": "phloem", "n_files": 3, "cell_diameter": 0.01}, # Number of files
    {"name": "resin_ducts", "diameter": 0.5, "n_files": 2},
    {"name": "inter_cellular_space", "ratio": 0, "size": 0},
    {"name": "stomata", "n_files": 5, "width": 0.07},
    {"name": "Strasburger cells", "layer_diameter": 0.002, "cell_diameter": 0.05}
]

def get_needle_width(params):
    # get needle width from parameters
    needle_width = 0
    for param in params:
        if param["name"] == "central_cylinder":
            needle_width += param.get("layer_length")
        elif param["name"] == "endodermis":
            needle_width += param.get("cell_diameter") * param.get("n_layers")
        elif param["name"] == "mesophyll":
            needle_width += param.get("cell_diameter") * param.get("n_layers")
        elif param["name"] == "hypodermis":
            needle_width += param.get("cell_diameter") * param.get("n_layers")
        elif param["name"] == "epidermis":
            needle_width += param.get("cell_diameter")
    return needle_width

def get_needle_thickness(params):
    # get needle height from parameters
    needle_thickness = 0
    for param in params:
        if param["name"] == "central_cylinder":
            needle_thickness += param.get("layer_thickness")
        elif param["name"] == "endodermis":
            needle_thickness += param.get("cell_diameter") * param.get("n_layers")
        elif param["name"] == "mesophyll":
            needle_thickness += param.get("cell_diameter") * param.get("n_layers")
        elif param["name"] == "hypodermis":
            needle_thickness += param.get("cell_diameter") * param.get("n_layers")
        elif param["name"] == "epidermis":
            needle_thickness += param.get("cell_diameter")
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
    
        prev_pts = np.roll(pts, 1, axis=0)
        next_pts = np.roll(pts, -1, axis=0)
    
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

def make_layers_polygons(layer_array, polygon):
    layers_polygons = []
    for i_layer, layer in enumerate(layer_array):
        
        if i_layer == 0: # add a oustide layer very close to the first polygon
            space_increment = layer["cell_diameter"] /2
            polygon = buffer_polygon(polygon, space_increment, smooth_factor=0.01)
            layers_polygons.append({"name": "outside", "polygon": polygon, "cell_diameter": layer["cell_diameter"]/3, "id_layer": i_layer})
        # then add the layer polygon
        polygon = buffer_polygon(polygon, -space_increment/2 - layer["cell_diameter"] / 4, smooth_factor=0.5)
        space_increment = layer["cell_diameter"] / 2
        layers_polygons.append({"name": layer["name"], "polygon": polygon, "cell_diameter": layer["cell_diameter"], "id_layer": i_layer+1})
    return layers_polygons

def cells_on_layer(layer_polygon, cell_diameter):
    # get the exterior coordinates of the polygon
    x,y = np.array(layer_polygon.exterior.coords.xy)
    perimeter = layer_polygon.length
    n_cells = int(np.round(perimeter / cell_diameter))
    # resample the coordinates to have n_cells points
    cells_coords = resample_coords(np.column_stack((x, y)), n_cells)
    return cells_coords

def cells_info(layers_polygons):
    all_cells = []
    id_cell = 1
    center = layers_polygons[0]["polygon"].centroid
    for i_layer, layer in enumerate(layers_polygons):
        cells_coords = cells_on_layer(layer["polygon"], layer["cell_diameter"])
        for cell_coord in cells_coords[1:]: # ingore the first cell
            all_cells.append({"type": layer["name"], "x": cell_coord[0], "y": cell_coord[1], "cell_diameter": layer["cell_diameter"],
            "id_cell": id_cell,
            "id_layer": i_layer,
            "id_group": int(0),
            "angle": np.arctan2(cell_coord[1]-center.y, cell_coord[0]-center.x),
            "radius": np.sqrt((cell_coord[0]-center.x)**2 + (cell_coord[1]-center.y)**2),
            "area": np.pi * (layer["cell_diameter"]/2)**2,
            })
            id_cell += 1
    all_cells = pd.DataFrame(all_cells)
    vor = Voronoi(all_cells[["x", "y"]])
    return all_cells, vor, center

def plot_section(section_gdf: gpd.GeoDataFrame):
    """Display the root section as polygons using GeoPandas and Matplotlib."""
    if section_gdf.empty:
        print("GeoDataFrame is empty, cannot plot.")
        return

    # GeoPandas handles the figure creation and geometry plotting
    fig, ax = plt.subplots(figsize=(8, 8))

    section_gdf.plot(
        ax=ax,
        column='type',           # Color polygons by the 'type' column
        cmap='viridis',          # Use a nice color map
        edgecolor='black',       # Outline the cells
        linewidth=0.5,           # Line width for the outline
        alpha=0.5,               # Transparency
        legend=True,             # Display the legend
        legend_kwds={'title': 'Cell Type', 'loc': 'best'}
    )
    ax.set_aspect("equal", "box")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Cross Section Preview")
    plt.tight_layout()
    plt.show()
    

def plot_generic_needle(params):
    # kill any previous plot
    plt.close('all')
    polygons = make_layers_polygons(layer_array(order_layers(params)), make_generic_needle(params))

    fig, ax = plt.subplots()
    colors = plt.cm.viridis(np.linspace(0, 1, len(polygons)))
    for polygon, color in zip(polygons, colors):
        ax.plot(*polygon["polygon"].exterior.xy, color=color)
        cells_coords = cells_on_layer(polygon["polygon"], polygon["cell_diameter"])
        ax.scatter(cells_coords[:,0], cells_coords[:,1], s=10, color=color)
    ax.set_aspect('equal')
    ax.legend([polygon["name"] for polygon in polygons])
    plt.show()

def test_voro(params):
    polygons = make_layers_polygons(layer_array(order_layers(params)), make_generic_needle(params))
    # create voronoi diagram
    all_cells, vor, center = cells_info(polygons)
    voronoi = cell_voro(all_cells, vor, center)
    all_cells = voronoi["all_cells"]
    plot_section(all_cells)
    
    
    
    rs2 = voronoi["rs2"]
    # plot_section(rs2)
    

    
# plot_generic_needle(params_data)
test_voro(params_data)
