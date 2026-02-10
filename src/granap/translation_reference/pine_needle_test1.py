
# Pine needle test 1
# New class: PineNeedleAnatomy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import shapely as sp
from shapely.ops import split
from shapely.geometry import Polygon, LineString
from cv2 import fitEllipse

from scipy.spatial import Voronoi, voronoi_plot_2d
import geopandas as gpd

params_data = [
    {"name": "planttype", "value": 3, "organ": "needle"}, # 1 = Monocot, 2 = Dicot, 3 = Gymnosperm
    {"name": "randomness", "value": 1.0}, # 0 = No randomness, 3 = Maximum randomness
    {"name": "secondarygrowth", "value": 0},
    {"name": "central_cylinder", "cell_diameter": 0.0063, "layer_thickness": 0.15, "layer_length": 0.35, "transfusion_layers": 3, "transfusion_tracheids_ratio": 0.5}, # Cell diameter in millimeters
    {"name": "transfusion_tracheids", "cell_diameter": 0.02},
    {"name": "transfusion_parenchyma", "cell_diameter": 0.03},
    {"name": "endodermis", "cell_diameter": 0.017, "cell_width": 0.05, "n_layers": 1, "order": 3},
    {"name": "mesophyll", "cell_diameter": 0.05, "cell_width": 0.03, "n_layers": 3, "order": 4},
    {"name": "hypodermis", "cell_diameter": 0.025, "n_layers": 3, "order": 5},
    {"name": "epidermis", "cell_diameter": 0.018, "n_layers": 1, "order": 6},
    {"name": "xylem", "n_files": 4, "cell_diameter": 0.005, "n_clusters": 3, "n_per_cluster": 3}, # Number of files
    {"name": "phloem", "n_files": 3, "cell_diameter": 0.01}, # Number of files
    {"name": "resin_ducts", "diameter": 0.5, "n_files": 2},
    {"name": "inter_cellular_space", "ratio": 0, "size": 0},
    {"name": "stomata", "n_files": 5, "width": 0.07},
    {"name": "Strasburger cells", "layer_diameter": 0.002, "cell_diameter": 0.05}
]

class Cell:
    def __init__(self, x: float, y: float, diameter: float, width: float=0, height: float=0, 
                type: str="", id_cell: int=-1, id_layer: int=-1, id_group: int=-1,
                angle: float=None, radius: float=None, area: float=None, polygon: Polygon=None):
        self.x = x # cell center x-coordinate
        self.y = y # cell center y-coordinate
        self.diameter = diameter # cell diameter
        self.width = width if width != 0 else diameter # cell width
        self.height = height if height != 0 else diameter # cell height
        self.type = type # cell type
        self.id_cell = id_cell # cell id
        self.id_layer = id_layer # layer id
        self.id_group = id_group # group id
        self.angle = angle if angle != None else np.arctan2(y, x) # angle of the cell center from the center of the organ
        self.radius = radius if radius != None else np.sqrt(x**2 + y**2) # distance of the cell center from the center of the organ
        self.area = area if area != None else np.pi * (diameter/2)**2 # approximate area of the cell
        self.polygon = polygon if polygon != None else None # polygon of the cell

    def cell_to_dict(self):
        return {"type": self.type, "x": self.x, "y": self.y, "cell_diameter": self.diameter,
                          "id_cell": self.id_cell,
                          "id_layer": self.id_layer,
                          "id_group": self.id_group,
                          "angle": self.angle,
                          "radius": self.radius,
                          "area": self.area,
                }

class AllCells:
    def __init__(self):
        self.cells = []

    def add_cell(self, cell: Cell):
        self.cells.append(cell)

    def get_cells(self):
        return self.cells

    def get_cell_by_id(self, id_cell: int):
        for cell in self.cells:
            if cell.id_cell == id_cell:
                return cell
        return None

    def get_cells_by_type(self, type: str):
        return [cell for cell in self.cells if cell.type == type]

    def get_cells_by_layer(self, id_layer: int):
        return [cell for cell in self.cells if cell.id_layer == id_layer]

    def get_cells_by_group(self, id_group: int):
        return [cell for cell in self.cells if cell.id_group == id_group]

    def get_cells_by_polygon(self, polygon: Polygon):
        return [cell for cell in self.cells if cell.polygon.intersects(polygon)]
    
    def remove_cells_by_polygon(self, polygon: Polygon):
        # if self.cells have polygon
        if self.cells[0].polygon is not None:
            self.cells = [cell for cell in self.cells if not cell.polygon.intersects(polygon)]
        else:
            for cell in self.cells:
                point = Point(cell.x, cell.y)
                if point.intersects(polygon):
                    self.cells.remove(cell)
    
    def recalculate_cell_properties(self):
        """Recalculate the properties of all cells in the list."""
        for i, cell in enumerate(self.cells):
            cell.angle = np.arctan2(cell.y, cell.x)
            cell.radius = cell.polygon.centroid.distance(Point(0, 0))
            cell.area = cell.polygon.area
            cell.id_cell = i

class Layer:
    def __init__(self, name: str, cell_diameter: float, id_layer: int, polygon: Polygon):
        self.name = name
        self.cell_diameter = cell_diameter
        self.id_layer = id_layer
        self.polygon = polygon
        self.coords = None

    def layer_to_dict(self):
        return {"name": self.name, "cell_diameter": self.cell_diameter, "id_layer": self.id_layer, "polygon": self.polygon}

    def resample_coords(self, target_n_points=200):
        # Ensure coords is a numpy array
        coords = np.array(self.coords)
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
    
    def smoothing_polygon(self, smooth_factor, iterations=10):
        """
        Smooths coordinates using a periodic Laplacian smoothing (moving average).
        Resamples the polygon to ensure uniform vertex distribution.
        iterations: Number of smoothing passes.
        """
        # Resample first to ensure uniform point distribution
        coords = self.resample_coords(target_n_points=200)

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
            
        self.coords = coords
        return coords


class Layers:
    def __init__(self):
        self.layers = []
        self.layer_array = []

    def add_layer(self, layer: Layer):
        self.layers.append(layer)

    def layer_array(self, params):
        # create array of layers
        for param in params:
            for i in range(param["n_layers"]):
                self.layer_array.append({"name": param["name"], "cell_diameter": param["cell_diameter"], "id_layer": i})
        return self.layer_array

class Organ:
    def __init__(self):
        self.cells = AllCells()
        self.layers = Layers()
        self.params = []
        self.bounding_polygon = None
    
    def set_params(self, params):
        for param in params:
            self.params.append(param)

    def order_layers(self):
        # remove layers without order
        params_ordered = [param for param in self.params if "order" in param]
        # order layers by order
        params_ordered.sort(key=lambda x: x["order"], reverse=True)
        return params_ordered

    def make_layers_polygons(self):
        self.layer_array = self.order_layers()
        for i_layer, layer in enumerate(self.layer_array):
            if i_layer == 0: # add an outside layer
                space_increment = layer["cell_diameter"] /2
                polygon = self.buffer_polygon(self.bounding_polygon, space_increment, smooth_factor=0.01)
                self.layers.add_layer(Layer("outside", polygon, layer["cell_diameter"]/3, i_layer))
            # then add the layer polygon
            polygon = self.buffer_polygon(polygon, -space_increment/2 - layer["cell_diameter"] / 4, smooth_factor=0.5)
            space_increment = layer["cell_diameter"] / 2
            self.layers.add_layer(Layer(layer["name"], polygon, layer["cell_diameter"], i_layer+1))
        return self.layers

    @staticmethod
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

class Needle:
    def __init__(self):
        self.organ = Organ()
        self.needle_width = 0
        self.needle_thickness = 0    

    def set_cells(self, cells):
        self.organ.cells = cells

    def set_layers(self, layers):
        self.organ.layers = layers

    def get_needle_width(self):
        # get needle width from organ parameters
        for param in self.organ.params:
            if param["name"] == "central_cylinder":
                self.needle_width += param.get("layer_length")
            elif param["name"] == "endodermis":
                self.needle_width += param.get("cell_diameter") * param.get("n_layers")
            elif param["name"] == "mesophyll":
                self.needle_width += param.get("cell_diameter") * param.get("n_layers")
            elif param["name"] == "hypodermis":
                self.needle_width += param.get("cell_diameter") * param.get("n_layers")
            elif param["name"] == "epidermis":
                self.needle_width += param.get("cell_diameter")
        return self.needle_width

    def get_needle_thickness(self):
        # get needle height from organ parameters
        for param in self.organ.params:
            if param["name"] == "central_cylinder":
                self.needle_thickness += param.get("layer_thickness")
            elif param["name"] == "endodermis":
                self.needle_thickness += param.get("cell_diameter") * param.get("n_layers")
            elif param["name"] == "mesophyll":
                self.needle_thickness += param.get("cell_diameter") * param.get("n_layers")
            elif param["name"] == "hypodermis":
                self.needle_thickness += param.get("cell_diameter") * param.get("n_layers")
            elif param["name"] == "epidermis":
                self.needle_thickness += param.get("cell_diameter")
        return self.needle_thickness

    def half_ellipse_polygon(self, width, height, n_points=1000):
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

    def make_generic_needle(self):
        needle_width = self.get_needle_width(self.organ.params)
        needle_thickness = self.get_needle_thickness(self.organ.params)
        self.organ.bounding_polygon = self.half_ellipse_polygon(needle_width, needle_thickness)
        return self.organ.bounding_polygon

    def make_needle(self):
        self.make_generic_needle()
        self.organ.make_layers_polygons(self.organ.layer_array, self.organ.bounding_polygon, self.organ.params)
        
        # add parenchyma cells until the polygon is filled
        params_cc = [p for p in self.organ.params if p["name"] == "central_cylinder"]
        params_tp = [p for p in self.organ.params if p["name"] == "transfusion_parenchyma"]
        params_tt = [p for p in self.organ.params if p["name"] == "transfusion_tracheids"]
        transfusion_layers = params_cc[0]["transfusion_layers"]
        transfusion_tracheids_ratio = params_cc[0]["transfusion_tracheids_ratio"]
        tt_cell_diameter = params_tt[0]["cell_diameter"]
        tp_cell_diameter = params_tp[0]["cell_diameter"]
        parenchyma_cell_diameter = params_cc[0]["cell_diameter"]

        while self.organ.bounding_polygon.area > (params_cc[0]["cell_diameter"]/2)**2 * np.pi:
            # Transfusion parenchyma and tracheids
            if transfusion_layers > 0:
                parenchyma_cell_diameter = (tp_cell_diameter + tt_cell_diameter)/2
                transfusion_layers -= 1
                self.organ.bounding_polygon = self.buffer_polygon(self.organ.bounding_polygon, -space_increment/2 - parenchyma_cell_diameter / 4, smooth_factor=0.6)
                space_increment = parenchyma_cell_diameter / 2
                self.organ.layers.add_layer(Layer("transfusion", self.organ.bounding_polygon, parenchyma_cell_diameter, i_layer+1))
            # Parenchyma
            else:
                self.organ.bounding_polygon = self.buffer_polygon(self.organ.bounding_polygon, -space_increment/2 - parenchyma_cell_diameter / 4, smooth_factor=0.7)
                space_increment = parenchyma_cell_diameter / 2
                self.organ.layers.add_layer(Layer("parenchyma", self.organ.bounding_polygon, parenchyma_cell_diameter, i_layer+1))
    
    # add cell_width for all layers
    for layer in self.organ.layers.layers:
        # if in param there is a cell_width for the layer, use it
        param_match = next((p for p in params if p["name"] == layer.name), None)
        if param_match and "cell_width" in param_match:
            layer.cell_width = param_match["cell_width"]/4
        else:
            layer.cell_width = 0
    return self.organ.layers


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
    
    def fit_needle_vascular_tissue(polygon, params):
        # from polygon, fit two ellipses
        vascular_elements = []
        
        ellipses = two_ellipses(polygon)
        vascular_elements = vascular_elements_in_ellipses(ellipses, params)
    
        return vascular_elements

def fit_root_monocot_vascular_tissue(polygon, params):
    # from polygon, fit one circle
    vascular_elements = []
    
    return vascular_elements

def fit_root_dicot_vascular_tissue(polygon, params):
    # from polygon, pack vascular elements
    vascular_elements = []
    
    return vascular_elements

def two_ellipses(polygon):
    # vertical splitting line (make it long enough to fully cross the polygon)
    center = polygon.centroid
    split_line = LineString([
        (center.x, polygon.bounds[1] - 10),
        (center.x, polygon.bounds[3] + 10),
    ])

    # split polygon
    parts = split(polygon, split_line)

    if len(parts) != 2:
        raise ValueError("Polygon was not split into two parts")

    # assign left / right based on centroid x
    left_poly, right_poly = sorted(
        parts,
        key=lambda p: p.centroid.x
    )

    ellipses = []

    # now you can fit one ellipse per side
    ellipses.append(fit_inner_ellipse(left_poly))
    ellipses.append(fit_inner_ellipse(right_poly))

    return ellipses

def vascular_elements_in_ellipses(ellipses, params):
    vascular_elements = []

    cells_in_ellipses = []
    id_cell = 0
    id_layer = 0
    for ellipse in ellipses:
        # get ellipse parameters
        center = ellipse["center"]
        rx, ry = ellipse["axes"]

        angle = ellipse["angle"]
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

        xylem_cell_height = (ry-params_cambium[0]["cell_diameter"])/xylem_rows
        phloem_cell_height = (ry-params_cambium[0]["cell_diameter"])/phloem_rows


        n_xylem_width = rx*2/xylem_cell_width
        xylem_cells = []
        
        xylem_cluster_n = params_xylem[0]["n_clusters"] # number of clusters
        xylem_cluster_size = params_xylem[0]["n_per_cluster"] # number of cells per cluster in width

        # verify if there are enough cells for the clusters
        cluster_width = xylem_cluster_size*xylem_cell_width
        if rx*2 < xylem_cluster_n * cluster_width + xylem_cell_width*(xylem_cluster_n-1):
            xylem_cluster_size = int(np.ceil((rx*2 - xylem_cell_width*(xylem_cluster_n-1))/(xylem_cell_width*xylem_cluster_n)))
            print("Not enough cells for the clusters, new cluster size: ", xylem_cluster_size)
        else:
            xylem_cluster_size = int(np.floor((rx*2 - xylem_cell_width*(xylem_cluster_n-1))/(xylem_cell_width*xylem_cluster_n)))
            print("Too many cells for the clusters, new cluster size: ", xylem_cluster_size)
        
        temp_cluster_id = xylem_cluster_size
        for i in range(n_xylem_width):
            id_layer += 1
            for j_xlm in range(xylem_rows):
                id_cell += 1
                xyl_coord = [center.x + i*xylem_cell_width - rx + xylem_cell_width/2,  # starting from left to right
                             center.y + j_xlm*xylem_cell_height + xylem_cell_height/2] # starting from middle to top
                # tilt the cells
                xyl_coord = [xyl_coord[0]*np.cos(angle) - xyl_coord[1]*np.sin(angle), xyl_coord[0]*np.sin(angle) + xyl_coord[1]*np.cos(angle)]  
                
                if temp_cluster_id == 0:
                    cell_type = "Strasburger cell"
                else:
                    cell_type = "xylem"
                    
                i_xylem = {"type": cell_type, "x": xyl_coord[0], "y": xyl_coord[1], "cell_diameter": xylem_cell_diameter,
                          "id_cell": id_cell,
                          "id_layer": id_layer,
                          "id_group": 0,
                          "angle": np.arctan2(xyl_coord[1]-center.y, xyl_coord[0]-center.x),
                          "radius": np.sqrt((xyl_coord[0]-center.x)**2 + (xyl_coord[1]-center.y)**2),
                          "area": np.pi * (xylem_cell_diameter/2)**2,
                }
                cells_in_ellipses.append(i_xylem)
            
            for j_phl in range(phloem_rows):
                id_cell += 1
                phlo_coord = [center.x + i*xylem_cell_width - rx + xylem_cell_width/2,  # starting from left to right
                             center.y - j_phl*phloem_cell_height - phloem_cell_height/2] # starting from middle to top
                # tilt the cells
                phlo_coord = [phlo_coord[0]*np.cos(angle) - phlo_coord[1]*np.sin(angle), phlo_coord[0]*np.sin(angle) + phlo_coord[1]*np.cos(angle)]

                i_phloem = {
                    "type": "phloem",
                    "x": phlo_coord[0],
                    "y": phlo_coord[1],
                    "cell_diameter": phloem_cell_diameter,
                    "id_cell": id_cell,
                    "id_layer": id_layer,
                    "id_group": 0,
                    "angle": np.arctan2(phlo_coord[1]-center.y, phlo_coord[0]-center.x),
                    "radius": np.sqrt((phlo_coord[0]-center.x)**2 + (phlo_coord[1]-center.y)**2),
                    "area": np.pi * (phloem_cell_diameter/2)**2,
                }
                cells_in_ellipses.append(i_phloem)

            if temp_cluster_id == 0:
                temp_cluster_id = xylem_cluster_size
            temp_cluster_id -= 1

            # cambium cell
            id_cell += 1

            xyl_coord = [center.x + i*xylem_cell_width - rx + xylem_cell_width/2,  # starting from left to right
                         center.y ] # major axis
            # tilt the cells
            xyl_coord = [xyl_coord[0]*np.cos(angle) - xyl_coord[1]*np.sin(angle), xyl_coord[0]*np.sin(angle) + xyl_coord[1]*np.cos(angle)]  
            i_cambium = {
                "type": "cambium",
                "x": xyl_coord[0],
                "y": xyl_coord[1],
                "cell_diameter": xylem_cell_diameter,
                "id_cell": id_cell,
                "id_layer": id_layer,
                "id_group": 0,
                "angle": np.arctan2(xyl_coord[1]-center.y, xyl_coord[0]-center.x),
                "radius": np.sqrt((xyl_coord[0]-center.x)**2 + (xyl_coord[1]-center.y)**2),
                "area": np.pi * (xylem_cell_diameter/2)**2,
            }
            cells_in_ellipses.append(i_cambium)
    
        # create a list of polygons for each ellipse
        ellipses_polygons = []
        for ellipse in ellipses:
            ellipses_polygons.append(ellipse_to_polygon(ellipse["center"], ellipse["axes"], ellipse["angle"]))
        
        vascular_elements.append({"cells": cells_in_ellipse,
                                  "polygon": MultiPolygon(ellipses_polygons)
        })
    return vascular_elements

def ellipse_to_polygon(cx, cy, rx, ry, angle):
    # create ellipse polygon
    ellipse = Ellipse((cx, cy), rx, ry, angle)
    return ellipse

def fit_inner_ellipse(polygon, shrink_step=0.98, min_scale=0.2):
    # convert to numpy array of points
    points = np.array(polygon.exterior.coords.xy).T

    # OpenCV fitEllipse expects (N, 1, 2)
    points = points.reshape(-1, 1, 2).astype(np.float32)

    # fit ellipse
    (cx, cy), (major, minor), angle = cv2.fitEllipse(points)
    rx = major / 2
    ry = minor / 2

    scale_factor = 0.98

    while scale_factor > min_scale:
        ell = ellipse_to_polygon(
            cx, cy,
            rx * scale_factor,
            ry * scale_factor,
            angle
        )

        if polygon.contains(ell):
            return {
                "center": (cx, cy),
                "axes": (rx * scale_factor, ry * scale_factor),
                "angle": angle,
            }

        scale_factor *= shrink_step

def allocate_vascular_tissue(center_polygon, all_cells, params):

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
    for cell in all_cells:
        point = Point(cell["x"], cell["y"])
        if point.intersects(MultiPolygon(vascular_elements["polygon"])):
            all_cells.remove(cell)
        
    # add vascular cells
    for cell in vascular_elements["cells"]:
        max_id_layer = max([c["id_layer"] for c in all_cells])
        cell["id_layer"] = max_id_layer + cell["id_layer"]
        cell["id_cell"] = len(all_cells) + cell["id_cell"]
        point = Point(cell["x"], cell["y"])
        if point.intersects(MultiPolygon(vascular_elements["polygon"])):
            all_cells.append(cell)
    
    return all_cells


def cells_on_layer(layer_polygon, cell_diameter, cell_width = 0):
    # get the exterior coordinates of the polygon
    x,y = np.array(layer_polygon.exterior.coords.xy)
    perimeter = layer_polygon.length
    if cell_width == 0:
        cell_width = cell_diameter
    else:
        cell_width = cell_width*4
    n_cells = int(np.round(perimeter / cell_width))*2
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
        
        cells_border.append(draw_ellipse(cell_coord, axis, major_axis/4, minor_axis/4, n_points=n_points))
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
        layer["cell_width"] = layer["cell_width"]*4
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
    
    layer_for_vascular = [l["name"] for l in layers_polygons].index("parenchyma")[0]

    # add vascular tissue
    all_cells = allocate_vascular_tissue(layers_polygons[layer_for_vascular], all_cells, params) # remove parenchyma cells in the vascular tissue and add vascular cells instead

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

def smooth_cells(grouped_gdf):
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

def test_voro(params):
    polygons = make_layers_polygons(layer_array(order_layers(params)), make_generic_needle(params), params)
    # create voronoi diagram
    all_cells, vor, center = cells_info(polygons, params)
    
    # Process attributes and merge
    grouped_cells = process_voronoi_groups(all_cells, vor)
    
    grouped_cells = smooth_cells(grouped_cells)

    plot_section(grouped_cells)
    

# plot_generic_needle(params_data)
test_voro(params_data)
