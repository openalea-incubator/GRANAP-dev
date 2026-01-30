
# Pine needle test 1
# New class: PineNeedleAnatomy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import shapely as sp

from scipy.spatial import Voronoi, voronoi_plot_2d
import geopandas as gpd

params_data = [
    {"name": "planttype", "value": 3}, # 1 = Monocot, 2 = Dicot, 3 = Gymnosperm
    {"name": "randomness", "value": 1.0}, # 0 = No randomness, 3 = Maximum randomness
    {"name": "secondarygrowth", "value": 0},
    {"name": "central_cylinder", "cell_diameter": 0.0063, "layer_thickness": 0.15, "layer_length": 0.35, "transfusion_layers": 3, "transfusion_tracheids_ratio": 0.5}, # Cell diameter in millimeters
    {"name": "transfusion_tracheids", "cell_diameter": 0.02},
    {"name": "transfusion_parenchyma", "cell_diameter": 0.03},
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
    

def make_layers_polygons(layer_array, polygon, params):
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

    # add parenchyma cells until the polygon is filled
    params_cc = [p for p in params if p["name"] == "central_cylinder"]
    params_tp = [p for p in params if p["name"] == "transfusion_parenchyma"]
    params_tt = [p for p in params if p["name"] == "transfusion_tracheids"]
    transfusion_layers = params_cc[0]["transfusion_layers"]
    transfusion_tracheids_ratio = params_cc[0]["transfusion_tracheids_ratio"]
    tt_cell_diameter = params_tt[0]["cell_diameter"]
    tp_cell_diameter = params_tp[0]["cell_diameter"]
    parenchyma_cell_diameter = params_cc[0]["cell_diameter"]

    while polygon.area > (params_cc[0]["cell_diameter"]/2)**2 * np.pi:
        # Transfusion parenchyma and tracheids
        if transfusion_layers > 0:
            parenchyma_cell_diameter = (tp_cell_diameter + tt_cell_diameter)/2
            transfusion_layers -= 1
            polygon = buffer_polygon(polygon, -space_increment/2 - parenchyma_cell_diameter / 4, smooth_factor=0.6)
            space_increment = parenchyma_cell_diameter / 2
            layers_polygons.append({"name": "transfusion", "polygon": polygon, "cell_diameter": parenchyma_cell_diameter, "id_layer": i_layer+1})
        # Parenchyma
        else:
            parenchyma_cell_diameter = params_cc[0]["cell_diameter"]
            polygon = buffer_polygon(polygon, -space_increment/2 - parenchyma_cell_diameter / 4, smooth_factor=0.7)
            space_increment = parenchyma_cell_diameter / 2
            layers_polygons.append({"name": "parenchyma", "polygon": polygon, "cell_diameter": parenchyma_cell_diameter, "id_layer": i_layer+1})
        
    return layers_polygons

def cells_on_layer(layer_polygon, cell_diameter):
    # get the exterior coordinates of the polygon
    x,y = np.array(layer_polygon.exterior.coords.xy)
    perimeter = layer_polygon.length
    n_cells = int(np.round(perimeter / cell_diameter))*2
    # resample the coordinates to have n_cells points
    cells_coords = resample_coords(np.column_stack((x, y)), n_cells)
    return cells_coords

def cell_border(cell_coords, cell_height, cell_width = 0):
    # place 5 points on the border of the elliptical cells
    if len(cell_coords) == 0:
        return []
    major_axis = cell_height
    if cell_width == 0:
        minor_axis = cell_height
    else:
        minor_axis = cell_width
    cells_border = []
    prev_cell_coord = cell_coords[-1] # IndexError: index 0 is out of bounds for axis 0 with size 0
    for i, cell_coord in enumerate(cell_coords):
        if i == len(cell_coords)-1:
            next_cell_coord = cell_coords[0]
        else:
            prev_cell_coord = cell_coords[i-1]
            next_cell_coord = cell_coords[i+1]
        axis = np.arctan2(next_cell_coord[1]-prev_cell_coord[1], next_cell_coord[0]-prev_cell_coord[0])
        cells_border.append(draw_ellipse(cell_coord, axis, major_axis/4, minor_axis/4, n_points=10))
    return cells_border   
    
def draw_ellipse(center, axis, major_axis, minor_axis, n_points=5):
    t = np.linspace(0, 2*np.pi, n_points)
    x = center[0] + major_axis * np.cos(t) * np.cos(axis) - minor_axis * np.sin(t) * np.sin(axis)
    y = center[1] + major_axis * np.cos(t) * np.sin(axis) + minor_axis * np.sin(t) * np.cos(axis)
    return np.column_stack((x, y)) 
    

def cells_info(layers_polygons):
    all_cells = []
    id_cell = 1
    id_group = 1
    center = layers_polygons[0]["polygon"].centroid
    for i_layer, layer in enumerate(layers_polygons):
        cells_coords = cells_on_layer(layer["polygon"], layer["cell_diameter"])
        layer_cell_borders = cell_border(cells_coords, layer["cell_diameter"]*0.7)

        for i, cell_coord in enumerate(cells_coords[1:]): # ingore the first cell
            if layer["name"] == "parenchyma":
                i_cell = {"type": layer["name"], "x": cell_coord[0], "y": cell_coord[1], "cell_diameter": layer["cell_diameter"],
                          "id_cell": id_cell,
                          "id_layer": i_layer,
                          "id_group": id_group,
                          "angle": np.arctan2(cell_coord[1]-center.y, cell_coord[0]-center.x),
                          "radius": np.sqrt((cell_coord[0]-center.x)**2 + (cell_coord[1]-center.y)**2),
                          "area": np.pi * (layer["cell_diameter"]/2)**2,
                }
                all_cells.append(i_cell) # center of the cell
                id_cell += 1
                id_group += 1
            else:
                cell_border_points = layer_cell_borders[i]
                for border_point in cell_border_points[1:]: # 5 coordinates for each cell border
                    all_cells.append({"type": layer["name"], "x": border_point[0], "y": border_point[1], "cell_diameter": layer["cell_diameter"],
                    "id_cell": id_cell,
                    "id_layer": i_layer,
                    "id_group": id_group,
                    "angle": np.arctan2(cell_coord[1]-center.y, cell_coord[0]-center.x),
                    "radius": np.sqrt((cell_coord[0]-center.x)**2 + (cell_coord[1]-center.y)**2),
                    "area": np.pi * (layer["cell_diameter"]/2)**2,
                    })
                    id_cell += 1
            
                id_group += 1
    all_cells = pd.DataFrame(all_cells)
    vor = Voronoi(all_cells[["x", "y"]])
    # fig = voronoi_plot_2d(vor)
    # plt.show()
    # 
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
    polygons = make_layers_polygons(layer_array(order_layers(params)), make_generic_needle(params), params)
    all_cells, _, _ = cells_info(polygons)
    fig, ax = plt.subplots()
    colors = plt.cm.viridis(np.linspace(0, 1, len(polygons)))
    for polygon, color in zip(polygons, colors):
        ax.plot(*polygon["polygon"].exterior.xy, color=color)
        cells_coords = all_cells[all_cells["type"] == polygon["name"]][["x", "y"]]
        ax.scatter(cells_coords["x"], cells_coords["y"], s=10, color=color)
    ax.set_aspect('equal')
    ax.legend([polygon["name"] for polygon in polygons])
    plt.show()

def process_voronoi_groups(all_cells, vor):
    geometries = []
    for i in range(len(all_cells)):
        region_idx = vor.point_region[i]
        region_vertices_indices = vor.regions[region_idx]
        
        if -1 in region_vertices_indices or len(region_vertices_indices) == 0:
            geometries.append(None)
        else:
            vertices = vor.vertices[region_vertices_indices]
            poly = sp.Polygon(vertices)
            if not poly.is_valid: # Check for validity
                poly = poly.buffer(0)
            geometries.append(poly)
    
    gdf = gpd.GeoDataFrame(all_cells, geometry=geometries)
    
    # Remove regions with "type" == "outside"
    gdf = gdf[gdf["type"] != "outside"]
    gdf = gdf.dropna(subset=["geometry"])
    
    # Union all polygon with the same id_group
    # dissolve aggregates by default using 'first', which preserves 'type' if consistent within group
    grouped_gdf = gdf.dissolve(by="id_group", as_index=False)
    
    # Calculate the region group area
    grouped_gdf["area"] = grouped_gdf.geometry.area
    
    return grouped_gdf

def test_voro(params):
    polygons = make_layers_polygons(layer_array(order_layers(params)), make_generic_needle(params), params)
    # create voronoi diagram
    all_cells, vor, center = cells_info(polygons)
    
    # Process attributes and merge
    grouped_cells = process_voronoi_groups(all_cells, vor)
    
    plot_section(grouped_cells)
    

# plot_generic_needle(params_data)
test_voro(params_data)
