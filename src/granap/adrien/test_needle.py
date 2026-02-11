
# Pine needle test no class just functions
# New class: PineNeedleAnatomy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import shapely as sp
import shapely.affinity as affinity
from shapely.ops import split, unary_union
from shapely.geometry import Point, Polygon, MultiPolygon, LineString, GeometryCollection
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
    

def make_layers_polygons(layer_array, polygon, params):
    layers_polygons = []
    smooth_factor = [p["smoothness"] for p in params if p["name"] == "randomness"][0]
    for i_layer, layer in enumerate(layer_array):
        
        if i_layer == 0: # add a oustide layer very close to the first polygon
            space_increment = layer["cell_diameter"] /2
            polygon = buffer_polygon(polygon, space_increment, smooth_factor=0.01)
            layers_polygons.append({"name": "outside", "polygon": polygon, "cell_diameter": layer["cell_diameter"]/3, "id_layer": i_layer})

        polygon = buffer_polygon(polygon, -space_increment - layer["cell_diameter"] / 2, smooth_factor=smooth_factor)
        # do we need to adjust the polygon to make it more circular as we go closer to the central cylinder?
        # adjust = 0.08 * smooth_factor * ((len(layer_array)-i_layer)/len(layer_array))
        # polygon = affinity.scale(polygon, xfact=1+adjust, yfact=1, origin='center')
        space_increment = (layer["cell_diameter"] / 2) 
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
            polygon = buffer_polygon(polygon, -space_increment - parenchyma_cell_diameter / 2, smooth_factor=smooth_factor/2)
            space_increment = parenchyma_cell_diameter / 2 
            # polygon = affinity.scale(polygon, xfact=1+adjust/2, yfact=1, origin='center')
            layers_polygons.append({"name": "transfusion_parenchyma", "polygon": polygon, "cell_diameter": parenchyma_cell_diameter, "id_layer": i_layer+1})
        # Parenchyma
        else:
            parenchyma_cell_diameter = params_cc[0]["cell_diameter"]
            polygon = buffer_polygon(polygon, -space_increment - parenchyma_cell_diameter / 2, smooth_factor=smooth_factor/2)
            
            space_increment = parenchyma_cell_diameter / 2 - smooth_factor * parenchyma_cell_diameter / 4
            # polygon = affinity.scale(polygon, xfact=1+adjust/2, yfact=1, origin='center')
            layers_polygons.append({"name": "parenchyma", "polygon": polygon, "cell_diameter": parenchyma_cell_diameter, "id_layer": i_layer+1})
    
    # add cell_width for all layers
    for layer in layers_polygons:
        # if in param there is a cell_width for the layer, use it
        param_match = next((p for p in params if p["name"] == layer["name"]), None)
        if param_match and "cell_width" in param_match:
            layer["cell_width"] = param_match["cell_width"]/2
        else:
            layer["cell_width"] = 0
    return layers_polygons

def cells_on_layer(layer_polygon, cell_diameter, cell_width = 0):
    # get the exterior coordinates of the polygon
    x,y = np.array(layer_polygon.exterior.coords.xy)
    perimeter = layer_polygon.length
    if cell_width == 0:
        cell_width = cell_diameter
    else:
        cell_width = cell_width*2
    n_cells = int(np.round(perimeter / cell_width))
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

    if cell_height == cell_width:
        n_points = 10
    else:
        n_points = 15

    cells_border = []
    prev_cell_coord = cell_coords[-1] # IndexError: index 0 is out of bounds for axis 0 with size 0
    for i, cell_coord in enumerate(cell_coords):
        if i == len(cell_coords)-1:
            next_cell_coord = cell_coords[0]
        else:
            prev_cell_coord = cell_coords[i-1]
            next_cell_coord = cell_coords[i+1]
        axis = np.arctan2(next_cell_coord[1]-prev_cell_coord[1], next_cell_coord[0]-prev_cell_coord[0])
        
        cells_border.append(draw_ellipse(cell_coord, axis, major_axis/2, minor_axis/2, n_points=n_points))
    return cells_border   
    
def draw_ellipse(center, axis, major_axis, minor_axis, n_points=5):
    t = np.linspace(0, 2*np.pi, n_points)
    x = center[0] + major_axis * np.cos(t) * np.cos(axis) - minor_axis * np.sin(t) * np.sin(axis)
    y = center[1] + major_axis * np.cos(t) * np.sin(axis) + minor_axis * np.sin(t) * np.cos(axis)
    return np.column_stack((x, y)) 
    

def cells_info(layers_polygons, params):
    all_cells = []
    id_cell = 1
    id_group = 1
    center = layers_polygons[0]["polygon"].centroid
    for i_layer, layer in enumerate(layers_polygons):
        cells_coords = cells_on_layer(layer["polygon"], layer["cell_diameter"], layer["cell_width"])
        layer["cell_width"] = layer["cell_width"]*2
        if layer["cell_width"] != 0 and layer["cell_width"] < layer["cell_diameter"]:
            layer_cell_borders = cell_border(cells_coords, layer["cell_width"]*0.7, layer["cell_diameter"]*0.7)
        elif layer["cell_width"] != 0 and layer["cell_width"] > layer["cell_diameter"]:
            print(layer["name"])
            layer_cell_borders = cell_border(cells_coords, layer["cell_width"]*0.7, layer["cell_diameter"]*0.7)
        else:
            layer_cell_borders = cell_border(cells_coords, layer["cell_diameter"]*0.7, layer["cell_width"]*0.7)

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

    layer_for_vascular = [l["name"] for l in layers_polygons].index("parenchyma")

    # add vascular tissue
    all_cells = allocate_vascular_tissue(layers_polygons[layer_for_vascular]["polygon"], all_cells, params) # remove parenchyma cells in the vascular tissue and add vascular cells instead

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
    # flip y axis
    ax.invert_yaxis()

    section_gdf.plot(
        ax=ax,
        column='type',           # Color polygons by the 'type' column
        cmap='viridis',          # Use a nice color map
        edgecolor='black',       # Outline the cells
        linewidth=0.5,           # Line width for the outline
        alpha=0.5,               # Transparency
        legend=False,             # Display the legend
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
    all_cells, _, _ = cells_info(polygons, params)
    fig, ax = plt.subplots()
    colors = plt.cm.viridis(np.linspace(0, 1, len(polygons)))
    for polygon, color in zip(polygons, colors):
        ax.plot(*polygon["polygon"].exterior.xy, color=color)
    
    cells_coords = all_cells[["x", "y"]]
    ax.scatter(cells_coords["x"], cells_coords["y"], s=10, color="black")
    
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

def smooth_cells(grouped_gdf, debug = False):
    """
    Smooths the boundaries between adjacent cells by simplifying shared edges to straight lines.
    Removes vertices that are intermediate points on a boundary shared with a single neighbor.
    This effectively straightens the interface between any two cell groups.
    """
    from shapely.geometry import Polygon

    geoms = grouped_gdf.geometry.tolist()
    # We use indices to identify polygons
    
    # 1. Quantize and build Edge Map
    point_map = {}
    next_pt_id = 0
    coords_list = [] # Map pt_id -> (x,y)
    
    def get_pt_id_mem(x, y):
        nonlocal next_pt_id
        # Round 6 decimal places for robustness
        k = (round(x, 6), round(y, 6))
        if k not in point_map:
            point_map[k] = next_pt_id
            coords_list.append(k)
            next_pt_id += 1
        return point_map[k]
    
    # Store edges to identify neighbors
    # edge_to_polys: (min_id, max_id) -> list of poly_indices
    edge_to_polys = {}
    poly_rings_ids = [] # Store the vertex IDs for each polygon to avoid re-parsing
    
    for idx, poly in enumerate(geoms):
        if poly is None or poly.is_empty:
            poly_rings_ids.append([])
            continue
        
        if isinstance(poly, MultiPolygon):
            if debug:
                print(poly.geoms)
                # plot the multipolygon
                fig, ax = plt.subplots()
                for p in poly.geoms:
                    ax.plot(*p.exterior.xy)
                plt.show()
                # find a polygon that is not empty
            for p in poly.geoms:
                if not p.is_empty:
                    poly = p
                    break
        # Handle exterior ring only (assuming cells are simple polygons)
        # If needed, can extend to interiors
        rings = [poly.exterior]
        # rings.extend(poly.interiors) # Uncomment to handle holes
        
        rings_pt_ids = []
        for ring in rings:
            pts = list(ring.coords)
            if pts[0] == pts[-1]:
                pts = pts[:-1]
                
            if len(pts) < 3:
                rings_pt_ids.append([])
                continue
            
            p_ids = [get_pt_id_mem(x, y) for x, y in pts]
            rings_pt_ids.append(p_ids)
            
            n_pts = len(p_ids)
            for i in range(n_pts):
                u = p_ids[i]
                v = p_ids[(i+1)%n_pts]
                if u == v: continue
                
                edge_key = tuple(sorted((u, v)))
                if edge_key not in edge_to_polys:
                    edge_to_polys[edge_key] = []
                edge_to_polys[edge_key].append(idx)
        
        poly_rings_ids.append(rings_pt_ids)
        
    # 2. Reconstruct Polygons with Straightened Boundaries
    new_geoms = []
    
    for idx in range(len(geoms)):
        rings_ids = poly_rings_ids[idx]
        if not rings_ids:
            new_geoms.append(geoms[idx])
            continue
            
        new_rings_coords = []
        
        for ring_pt_ids in rings_ids:
            if not ring_pt_ids: continue
            
            # Identify neighbor for each edge leaving a vertex
            # ring_pt_ids[i] -> ring_pt_ids[i+1]
            n_pts = len(ring_pt_ids)
            edge_neighbors = []
            
            for i in range(n_pts):
                u = ring_pt_ids[i]
                v = ring_pt_ids[(i+1)%n_pts]
                edge_key = tuple(sorted((u, v)))
                
                neighbors = edge_to_polys.get(edge_key, [])
                
                # Find the neighbor that is NOT idx
                # If shared by multiple others, effectively just "shared"
                # If only shared by idx (boundary), neighbor is None
                other = None
                for n_idx in neighbors:
                    if n_idx != idx:
                        other = n_idx
                        break
                edge_neighbors.append(other)
            
            # Filter vertices
            optimized_ring = []
            for k in range(n_pts):
                u = ring_pt_ids[k]
                
                # Look at incoming edge and outgoing edge neighbors
                prev_edge_idx = (k - 1) % n_pts
                curr_edge_idx = k
                
                n_prev = edge_neighbors[prev_edge_idx]
                n_curr = edge_neighbors[curr_edge_idx]
                
                # Keep vertex if it is a transition point or on mesh boundary
                # Logic:
                # - If neighbors differ, it's a triple junction -> Keep
                # - If neighbors are same but None (exterior), it's mesh boundary -> Keep
                # - If neighbors are same and not None, it's intermediate on shared edge -> Drop
                
                if n_prev != n_curr:
                    optimized_ring.append(u)
                elif n_prev is None:
                    optimized_ring.append(u)
                # else: drop
            
            # Safety check: if we removed too many vertices, fallback
            if len(optimized_ring) < 3:
                optimized_ring = ring_pt_ids

            # Reconstruct coords
            ring_coords = [coords_list[pid] for pid in optimized_ring]
            new_rings_coords.append(ring_coords)
        
        if not new_rings_coords:
             new_geoms.append(geoms[idx])
        else:
             # Create Polygon
             ext = new_rings_coords[0]
             if ext[0] != ext[-1]: ext.append(ext[0])
             
             new_poly = Polygon(ext)
             new_geoms.append(new_poly)

    grouped_gdf.geometry = new_geoms
    return grouped_gdf

def fit_needle_vascular_tissue(polygon, params):
    # from polygon, fit two ellipses
    vascular_elements = []
    
    ellipses = two_ellipses(polygon)
    vascular_elements = vascular_elements_in_ellipses(ellipses, params)
    
    return vascular_elements

def two_ellipses(polygon):
    # vertical splitting line (make it long enough to fully cross the polygon)
    center = polygon.centroid
    split_line = LineString([
        (center.x, polygon.bounds[1] - 10),
        (center.x, polygon.bounds[3] + 10),
    ])

    # Define the splitting rectangle
    # Format: box(minx, miny, maxx, maxy)
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

    # split polygon
    # parts = split(polygon, split_line)

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

    # now you can fit one ellipse per side
    ellipses.append(fit_inner_ellipse(left_poly.buffer(-0.002)))
    ellipses.append(fit_inner_ellipse(right_poly.buffer(-0.002)))

    return ellipses

def vascular_elements_in_ellipses(ellipses, params, debug = False):
    # vascular elements dictionary with multiPolygon geometry and cell data
    vascular_elements = []
    # create a list of polygons for each ellipse
    list_ellipses_polygons = []
    # create a list of cells in all ellipses
    cells_in_ellipses = []
    
    id_cell = 0
    id_layer = 0
    for ellipse in ellipses:
        # get ellipse parameters
        center = ellipse["polygon"].centroid
        rx, ry = ellipse["axes"]
        angle = np.deg2rad(ellipse["angle"])-np.pi/2

        print(np.rad2deg(angle))
        # add rows of xylem cells in upper part of ellipse
        params_xylem = [p for p in params if p["name"] == "xylem"]
        xylem_rows = params_xylem[0]["n_files"] # cell files
        xylem_cell_width = params_xylem[0]["cell_diameter"] # cell width

        # add rows of phloem cells in lower part of ellipse
        params_phloem = [p for p in params if p["name"] == "phloem"]
        phloem_rows = params_phloem[0]["n_files"]
        phloem_cell_diameter = params_phloem[0]["cell_diameter"]
        # add cambium cells between xylem and phloem
        params_cambium = [p for p in params if p["name"] == "cambium"]

        xylem_cell_height = (rx-params_cambium[0]["cell_diameter"])/xylem_rows
        phloem_cell_height = (rx-params_cambium[0]["cell_diameter"])/phloem_rows


        n_xylem_width = int(np.ceil(ry*2/xylem_cell_width)) # number of cells in width
        xylem_cells = []
        
        xylem_cluster_n = int(params_xylem[0]["n_clusters"]) # number of clusters
        xylem_cluster_size = int(params_xylem[0]["n_per_cluster"]) # number of cells per cluster in width

        # verify if there are enough cells for the clusters
        cluster_width = xylem_cluster_size*xylem_cell_width
        xylem_cluster_size = int(np.ceil((ry*2 - xylem_cell_width*(xylem_cluster_n-1))/(xylem_cell_width*xylem_cluster_n)))
        
        temp_cluster_id = xylem_cluster_size

        for i in range(n_xylem_width+1):
            id_layer += 1
            for j_xlm in range(xylem_rows+1):
                id_cell += 1
                xyl_coord = [i*xylem_cell_width - ry + xylem_cell_width/2,  # starting from left to right
                             j_xlm*xylem_cell_height - ry + xylem_cell_height/2] # starting from middle to top
                # tilt the cells
                xyl_coord = [xyl_coord[0]*np.cos(angle) - xyl_coord[1]*np.sin(angle), xyl_coord[0]*np.sin(angle) + xyl_coord[1]*np.cos(angle)] 
                # translate the cells
                xyl_coord = [xyl_coord[0] + center.x, xyl_coord[1] + center.y]
                
                if temp_cluster_id == 0:
                    cell_type = "Strasburger cell"
                else:
                    cell_type = "xylem"
                xylem_cell_diameter = (xylem_cell_width + xylem_cell_height)/2
                    
                i_xylem = {"type": cell_type, "x": xyl_coord[0], "y": xyl_coord[1], "cell_diameter": xylem_cell_diameter,
                          "id_cell": id_cell,
                          "id_layer": id_layer,
                          "id_group": id_cell,
                          "angle": np.arctan2(xyl_coord[1]-center.y, xyl_coord[0]-center.x),
                          "radius": np.sqrt((xyl_coord[0]-center.x)**2 + (xyl_coord[1]-center.y)**2),
                          "area": np.pi * (xylem_cell_diameter/2)**2,
                }

                # is the point in ellipse
                if ellipse["polygon"].contains(Point(xyl_coord)):
                    cells_in_ellipses.append(i_xylem)   
            
            for j_phl in range(1, phloem_rows+1):
                id_cell += 1
                phlo_coord = [i*xylem_cell_width - ry + xylem_cell_width/2,  # starting from left to right
                             j_phl*phloem_cell_height + phloem_cell_height/2] # starting from middle to top
                # tilt the cells
                phlo_coord = [phlo_coord[0]*np.cos(angle) - phlo_coord[1]*np.sin(angle), phlo_coord[0]*np.sin(angle) + phlo_coord[1]*np.cos(angle)]
                phlo_coord = [phlo_coord[0] + center.x, phlo_coord[1] + center.y]
                phloem_cell_diameter = (xylem_cell_width + phloem_cell_height)/2

                i_phloem = {
                    "type": "phloem",
                    "x": phlo_coord[0],
                    "y": phlo_coord[1],
                    "cell_diameter": phloem_cell_diameter,
                    "id_cell": id_cell,
                    "id_layer": id_layer,
                    "id_group": id_cell,
                    "angle": np.arctan2(phlo_coord[1]-center.y, phlo_coord[0]-center.x),
                    "radius": np.sqrt((phlo_coord[0]-center.x)**2 + (phlo_coord[1]-center.y)**2),
                    "area": np.pi * (phloem_cell_diameter/2)**2,
                }
                # is the point in ellipse
                if ellipse["polygon"].contains(Point(phlo_coord)):
                    cells_in_ellipses.append(i_phloem)

            if temp_cluster_id == 0:
                temp_cluster_id = xylem_cluster_size+1
            temp_cluster_id -= 1

            # cambium cell
            id_cell += 1

            xyl_coord = [i*xylem_cell_width - ry + xylem_cell_width/2,  # starting from left to right
                        0] 
            # tilt the cells
            xyl_coord = [xyl_coord[0]*np.cos(angle) - xyl_coord[1]*np.sin(angle), xyl_coord[0]*np.sin(angle) + xyl_coord[1]*np.cos(angle)]  
            xyl_coord = [xyl_coord[0] + center.x, xyl_coord[1] + center.y]
            i_cambium = {
                "type": "cambium",
                "x": xyl_coord[0],
                "y": xyl_coord[1],
                "cell_diameter": xylem_cell_diameter,
                "id_cell": id_cell,
                "id_layer": id_layer,
                "id_group": id_cell,
                "angle": np.arctan2(xyl_coord[1]-center.y, xyl_coord[0]-center.x),
                "radius": np.sqrt((xyl_coord[0]-center.x)**2 + (xyl_coord[1]-center.y)**2),
                "area": np.pi * (xylem_cell_diameter/2)**2,
            }
            # is the point in ellipse
            if ellipse["polygon"].contains(Point(xyl_coord)):
                cells_in_ellipses.append(i_cambium)
    
        # create a list of polygons for each ellipse
        list_ellipses_polygons.append(ellipse["polygon"])

        if debug:
            # plot the ellipse
            color_map = {"Strasburger cell": "red", "xylem": "blue", "phloem": "green", "cambium": "yellow"}
            plt.plot(ellipse["polygon"].exterior.xy[0], ellipse["polygon"].exterior.xy[1])
            # plot the cells
            for cell in cells_in_ellipses:
                plt.plot(cell["x"], cell["y"], "o", color = color_map[cell["type"]])
            plt.show()
        
    vascular_elements = {"cells": cells_in_ellipses,
                         "polygons": list_ellipses_polygons}
    
    return vascular_elements

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

def fit_inner_ellipse(polygon, shrink_step=0.98, min_scale=0.2, debug=False):
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
        ax.set_aspect('equal')
        plt.legend()
        plt.show()

    return result_ellipse

def allocate_vascular_tissue(center_polygon: Polygon, all_cells, params):

    # what type of vascular tissue to allocate
    params_type = [p for p in params if p["name"] == "planttype"]
    if params_type[0]["value"] == 3 and params_type[0]["organ"] == "needle":
        # from polygon, fit two ellipses
        vascular_elements = fit_needle_vascular_tissue(center_polygon, params)
    if params_type[0]["value"] == 1 and params_type[0]["organ"] == "root":
        # monocot root
        # from polygon, fit one circle
        vascular_elements = fit_root_monocot_vascular_tissue(center_polygon, params)
    if params_type[0]["value"] == 2 and params_type[0]["organ"] == "root":
        # dicot root
        # from polygon, pack vascular elements
        vascular_elements = fit_root_dicot_vascular_tissue(center_polygon, params)

    # remove the cells in the vascular elements
    vascular_polygons = unary_union(vascular_elements["polygons"])
    all_cells = [
        cell for cell in all_cells
        if not Point(cell["x"], cell["y"]).intersects(vascular_polygons.buffer(0.001))
    ]
        
    # add vascular cells
    for cell in vascular_elements["cells"]:
        max_id_layer = max([c["id_layer"] for c in all_cells])
        cell["id_layer"] = max_id_layer + cell["id_layer"]
        cell["id_cell"] = len(all_cells) + cell["id_cell"]
        cell["id_group"] = len(all_cells) + cell["id_cell"]
        all_cells.append(cell)
    
    return all_cells

def fit_root_monocot_vascular_tissue(polygon, params):
    # from polygon, fit vascular tissue
    vascular_elements = []
    
    for param in params:
        if param["name"] == "central_cylinder":
            vascular_elements.append(param)
    
    return vascular_elements

def fit_root_dicot_vascular_tissue(polygon, params):
    # from polygon, fit vascular tissue
    vascular_elements = []
    
    for param in params:
        if param["name"] == "central_cylinder":
            vascular_elements.append(param)
    
    return vascular_elements

def last_transfusion_polygon(layer_polygon):
    last_transfusion_polygon = None
    last_transfusion_area = np.inf
    for layer in layer_polygon:
        if layer["name"] == "transfusion":
            # last one, smaller than the previous one
            if last_transfusion_polygon is not None:
                if layer["polygon"].area < last_transfusion_area:
                    last_transfusion_polygon = layer["polygon"]
                    last_transfusion_area = layer["polygon"].area
    return last_transfusion_polygon

def test_voro(params, plot=False):
    polygons = make_layers_polygons(layer_array(order_layers(params)), make_generic_needle(params), params)
    # create voronoi diagram
    all_cells, vor, center = cells_info(polygons, params)
    
    # Process attributes and merge
    grouped_cells = process_voronoi_groups(all_cells, vor)
    
    grouped_cells = smooth_cells(grouped_cells)

    if plot:
        plot_section(grouped_cells)

    # calculate metrics
    metrics = calculate_metrics(grouped_cells)
    return grouped_cells, metrics
    

def calculate_metrics(grouped_gdf):
    metrics = {}
    for layer in grouped_gdf["type"].unique():
        metrics[layer] = {}
        metrics[layer]["area"] = grouped_gdf[grouped_gdf["type"] == layer]["area"].sum()
        metrics[layer]["width"] = max(grouped_gdf[grouped_gdf["type"] == layer]["x"])-min(grouped_gdf[grouped_gdf["type"] == layer]["x"])
        metrics[layer]["thickness"] = max(grouped_gdf[grouped_gdf["type"] == layer]["y"])-min(grouped_gdf[grouped_gdf["type"] == layer]["y"])
    return metrics

# test smoothing factor on macro metrics
# range smoothness from 0 to 1, plot metrics vs smoothness
def test_smoothing_factor():
    smoothness_range = np.linspace(0, 1, 10)
    metrics = []
    for smoothness in smoothness_range:
        params_data[1]["smoothness"] = smoothness
        grouped_cells, metrics_i = test_voro(params_data, plot=True)
        metrics.append(metrics_i)
    
    # plot metrics layers vs smoothness
    for layer in metrics[0].keys():
        if layer in ["epidermis", "hypodermis", "endodermis"]:
            plt.plot(smoothness_range, [m[layer]["width"] for m in metrics], label=layer)
            plt.plot(smoothness_range, [m[layer]["thickness"] for m in metrics], label=layer)
    plt.xlabel("Smoothness")
    plt.ylabel("Length")
    plt.legend()
    plt.show()

plot_generic_needle(params_data)
# test_voro(params_data, plot=True)

# test_smoothing_factor()
