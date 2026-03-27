"""
Needle anatomy implementation.
"""

import numpy as np
from typing import List, Dict, Any
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.ops import unary_union

from granap.organ_class import Organ
from granap.cell_class import Cell
from granap.cell_manager import CellManager
from granap.generate_cell import CellGenerator
from granap.layer_class import Layer
from granap.geometry_collection import GeometryProcessor
from granap.shapes import PolygonInterpolator
import matplotlib.pyplot as plt


class NeedleAnatomy(Organ):
    """
    Needle cross-sectional anatomy.
    
    Implements the specific structure of gymnosperm needle leaves,
    including transfusion tissue and resin ducts.
    """
    
    from granap.input_data import OrganInputData

    def __init__(self, input_data: Any = None):
        """
        Initialize needle anatomy.
        """
        super().__init__()
        # Initialize parameters from input_data or default
        if hasattr(input_data, 'params'):
            self.params = input_data.params
        elif isinstance(input_data, list):
            self.params = input_data
        else:
            self._initialize_default_params()
        
        self._initialize_params()
        self._initialize_default_layers()

            
    def _initialize_default_params(self) -> Dict[str, Any]:
        """Initialize default parameters."""

        self.params = [
            # P. pinaster
            {"name": "planttype", "value": 3, "organ": "needle", "width": 1.8, "thickness": 1.1}, # global parameters
            {"name": "randomness", "value": 1.0, "smoothness": 0.3}, # 0 = No randomness, 3 = Maximum randomness; smoothness is the smoothing factor (0 = no smoothing, 1 = maximum smoothing)
            {"name": "central_cylinder", "shape": "half_ellipse", "cell_diameter": 0.02, "layer_thickness": 0.43, "layer_length": 1.05, "vascular_width": 0.15, "vascular_height": 0.2}, # Cell diameter in millimeters
            {"name": "transfusion_tissue", "tracheids_diameter": 0.05, "parenchyma_diameter": 0.03, "transfusion_tracheids_ratio": 0.5, "n_layers":2},
            {"name": "endodermis", "cell_diameter": 0.02, "cell_width": 0.05, "n_layers": 1, "order": 3, "shift": 5},
            {"name": "mesophyll", "cell_diameter": 0.08, "cell_width": 0.045, "n_layers": 3, "order": 4, "shift":10},
            {"name": "hypodermis", "cell_diameter": 0.0225, "n_layers": 2, "order": 5},
            {"name": "epidermis", "cell_diameter": 0.02, "n_layers": 1, "order": 6},
            {"name": "xylem", "n_files": 10, "cell_diameter": 0.007, "n_clusters": 4, "n_per_cluster": 3}, # Number of files
            {"name": "phloem", "n_files": 8, "cell_diameter": 0.003}, 
            {"name": "cambium", "cell_diameter": 0.002}, 
            {"name": "resin_duct", "diameter": 0.1, "n_files": 3, "cell_diameter": 0.02},
            {"name": "inter_cellular_spaces", "mesophyll": 0.01},
            {"name": "stomata", "n_files": 4, "width": 0.025, "depth": 0.06, "sub_chamber": 0.04},
            {"name": "Strasburger cells", "layer_diameter": 0.002, "cell_diameter": 0.05}
        ]

    def _initialize_params(self) -> None:
        """Initialize central layers."""
        # get the central cylinder parameters
        self.central_cylinder_params = [param for param in self.params if param["name"] == "central_cylinder"][0]
        # get the transfusion tissue parameters
        self.transfusion_params = [param for param in self.params if param["name"] == "transfusion_tissue"][0]
        # get the global parameters
        self.global_params = [param for param in self.params if param["name"] == "planttype"][0]

        self.layers = [param for param in self.params if "order" in param]
        self.layers = sorted(self.layers, key=lambda x: x["order"])        
    
    def _initialize_default_layers(self) -> None:
        """Initialize default needle layers."""
        layer_array = []
        for param in self.layers:
            self.layer_manager.add_layer(Layer(
                name=param["name"],
                cell_diameter=param["cell_diameter"],
                cell_width=param.get("cell_width", param["cell_diameter"]),
                shift=param.get("shift", 0.0),
                n_layers=param.get("n_layers", 1),
                order=param.get("order", 0)
            ))
    
    def _create_base_shape(self) -> Polygon:
        """
        Create the half-ellipse shape of a needle cross-section.
        
        Returns:
            Half-ellipse polygon
        """
        # check if width and thickness are provided
        if self.global_params.get("width") is None: #key error
            self.global_params["width"] = 0
        if self.global_params.get("thickness") is None: #key error
            self.global_params["thickness"] = 0
        # if width and thickness are not provided, calculate them from the layers     
        if self.global_params["width"] == 0 and self.global_params["thickness"] == 0:
            width = self._calculate_needle_width()
            thickness = self._calculate_needle_thickness()
        # if width or thickness is provided, calculate the other
        elif self.global_params["width"] == 0:
            width = self._calculate_needle_width()
            thickness = self.global_params["thickness"]
        elif self.global_params["thickness"] == 0:
            width = self.global_params["width"]
            thickness = self._calculate_needle_thickness()
        # if both width and thickness are provided, use them
        else:
            width = self.global_params["width"]
            thickness = self.global_params["thickness"]
        
        return GeometryProcessor.half_ellipse_polygon(width, thickness)

    def reshape_layers(self, layers_polygons: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        When "central_cylinder" has shape="ellipse", interpolate each layer
        polygon between the outer half-ellipse (t=0) and a full ellipse
        aligned with the endodermis layer (t=1).

        Layers from the outside down to the endodermis are gradually morphed.
        Layers inward from the endodermis (transfusion, parenchyma …) are
        fully changed to fit inside the ellipse.
        """
        if self.central_cylinder_params.get("shape") != "ellipse":
            return layers_polygons

        if not layers_polygons:
            return layers_polygons

        # --- build the target ellipse ----------------------------------------
        # Use the layer_thickness and layer_length of the central cylinder as
        # the semi-axes of the target full ellipse.
        rx = self.central_cylinder_params["layer_length"] / 2
        ry = self.central_cylinder_params["layer_thickness"] / 2 
        n_pts = 120
        angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        ellipse_coords = [(rx * np.cos(a), ry * np.sin(a)) for a in angles]
        ellipse_coords = [(x, y + self.global_params["thickness"] / 2.2) for x, y in ellipse_coords]
        target_ellipse = GeometryProcessor.buffer_polygon(
            Polygon(ellipse_coords),
            0, smooth_factor=0.0
        )

        # --- find the index of the endodermis layer --------------------------
        layer_names = [lp["name"] for lp in layers_polygons]
        
        endo_idx = layer_names.index("endodermis")

        # outside polygon (index 0) is the reference half-ellipse shape; we
        # keep it as-is (t=0) and warp everything inward up to endo_idx (t=1).
        outer_poly = layers_polygons[0]["polygon"]

        # Pre-compute one interpolator between the outer shape and the ellipse.
        try:
            interp = PolygonInterpolator(outer_poly, target_ellipse)
        except Exception:
            # If PolygonInterpolator fails (degenerate geometry), skip reshape.
            return layers_polygons

        n_to_morph = endo_idx + 1  # indices 0 … endo_idx inclusive
        
        for i in range(1, n_to_morph):          # skip index 0 (outside)
            t = i / max(n_to_morph - 1, 1)     # 0 < t <= 1
            print(t)
            try:
                new_poly = interp.fast_interpolate(t)
                if not new_poly.is_empty and new_poly.is_valid:
                    layers_polygons[i] = dict(layers_polygons[i])
                    layers_polygons[i]["polygon"] = new_poly
            except Exception:
                pass  # leave this layer polygon unchanged on error

        layers_polygons = layers_polygons[:endo_idx+1]  # remove layers after endodermis

        layers_polygons.extend(self._create_central_layers(target_ellipse, params= self.params))  # add new central layers

        return layers_polygons
    
    def _calculate_needle_width(self) -> float:
        """Calculate total needle width from layers."""
        # width of vascular cylinder
        width_vascular = self.central_cylinder_params["layer_length"]
        # width of all supplementary layers
        width_layer = 0
        for layer in self.layer_manager.get_layers():
            if hasattr(layer, 'n_layers'):
                width_layer += layer.get_total_thickness()
            elif hasattr(layer, 'cell_diameter'):
                width_layer += layer.cell_diameter
        # thickness of vascular cylinder
        thickness_vascular = self.central_cylinder_params["layer_thickness"]
        # thickness of all supplementary layers which is equal to width_layer
        self.thickness_layer = width_layer
        thickness_total = (2 * self.thickness_layer) + thickness_vascular
        
        width = 2 * np.sqrt((width_vascular/2 + self.thickness_layer)**2 / 
                            (1-(self.thickness_layer/thickness_total)**2))

        return width
    
    def _calculate_needle_thickness(self) -> float:
        """Calculate total needle thickness from layers."""
        thickness = self.central_cylinder_params["layer_thickness"]
        
        for layer in self.layer_manager.get_layers():
            if hasattr(layer, 'n_layers'):
                thickness += 2 * layer.get_total_thickness()
            elif hasattr(layer, 'cell_diameter'):
                thickness += 2 * layer.cell_diameter
        
        return thickness
    
    def _create_central_layers(self, current_polygon: Polygon,
                               params: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create transfusion tissue and parenchyma layers.
        
        Args:
            current_polygon: Current inner polygon boundary
            params: Parameter dictionaries
        
        Returns:
            List of central layer polygon dictionaries
        """
        central_layers = []
        space_increment = self.central_cylinder_params["cell_diameter"] / 2
        transfusion_layers_remaining = self.transfusion_params["n_layers"]
        
        tt_diameter = self.transfusion_params["tracheids_diameter"]
        tp_diameter = self.transfusion_params["parenchyma_diameter"]
        parenchyma_diameter = self.central_cylinder_params["cell_diameter"]
        
        i_layer = len(params)
        
        while current_polygon.area > (parenchyma_diameter / 2)**2 * np.pi:
            if transfusion_layers_remaining > 0:
                # Transfusion tissue
                avg_diameter = (tp_diameter + tt_diameter) / 2 
                transfusion_layers_remaining -= 1
                
                current_polygon = GeometryProcessor.buffer_polygon(
                    current_polygon,
                    -space_increment - avg_diameter / 2,
                    smooth_factor=0.6
                )
                
                space_increment = avg_diameter / 2
                
                central_layers.append({
                    "name": "transfusion",
                    "polygon": current_polygon,
                    "cell_diameter": avg_diameter,
                    "id_layer": i_layer + 1,
                    "cell_width": 0
                })
            else:
                # Parenchyma
                current_polygon = GeometryProcessor.buffer_polygon(
                    current_polygon,
                    -space_increment - parenchyma_diameter / 2,
                    smooth_factor=0.7
                )
                
                space_increment = parenchyma_diameter / 2
                
                central_layers.append({
                    "name": "parenchyma",
                    "polygon": current_polygon,
                    "cell_diameter": parenchyma_diameter,
                    "id_layer": i_layer + 1,
                    "cell_width": 0
                })
            
            i_layer += 1
        
        return central_layers
    
    def set_central_cylinder_params(self, **kwargs) -> None:
        """
        Update central cylinder parameters.
        
        Args:
            **kwargs: Parameter names and values to update
        """
        self.central_cylinder_params.update(kwargs)
        self._invalidate_geometry()
    
    def set_transfusion_params(self, **kwargs) -> None:
        """
        Update transfusion tissue parameters.
        
        Args:
            **kwargs: Parameter names and values to update
        """
        self.transfusion_params.update(kwargs)
        self._invalidate_geometry()

    def _which_layer_for_vascular(self, layers_polygons: List[Dict[str, Any]]):
        """
        Find the layer where vascular tissue will be allocated.
        
        Args:
            layers_polygons: List of layer polygon dictionaries
        """
        layer_for_vascular = [l["name"] for l in layers_polygons].index("parenchyma")
        polygon_for_vascular = layers_polygons[layer_for_vascular]["polygon"]
        return polygon_for_vascular
    
    def _create_vascular_tissue(self, polygon: Polygon):
        """
        Create vascular tissue.
        
        Args:
            polygon: Polygon boundary
        """
        self.fit_vascular_elements(polygon)
        # remove the cells in the vascular elements
        vascular_polygons = unary_union(self.vascular_polygons)
        self.all_cells.remove_cells_in_polygon(vascular_polygons)

        # add vascular cells to all_cells
        self.all_cells.extend_cells(self.vascular_cells)
        self.all_cells.recalculate_cell_properties()

    def fit_vascular_elements(self, polygon):
        # from polygon, fit two ellipses
        rx = self.central_cylinder_params["vascular_width"]/2
        ry = self.central_cylinder_params["vascular_height"]/2
        ellipses = GeometryProcessor.two_ellipses(polygon, rx, ry)
        cells_in_ellipses, list_ellipses_polygons = self.vascular_elements_in_ellipses(ellipses)
        self.vascular_cells = cells_in_ellipses
        self.vascular_polygons = list_ellipses_polygons
        

    def vascular_elements_in_ellipses(self, ellipses, debug = False):

        # create a list of polygons for each ellipse
        list_ellipses_polygons: List[Polygon] = []
        # create a list of cells in all ellipses
        cells_in_ellipses: List[Cell] = []
        
        id_cell = 0
        id_layer = 0
        for ellipse in ellipses:
            # get ellipse parameters
            center = ellipse["polygon"].centroid
            rx, ry = ellipse["axes"]
            angle = np.deg2rad(ellipse["angle"])-np.pi/2

            # add rows of xylem cells in upper part of ellipse
            params_xylem = [p for p in self.params if p["name"] == "xylem"]
            xylem_rows = params_xylem[0]["n_files"] # cell files
            xylem_cell_width = params_xylem[0]["cell_diameter"] # cell width
    
            # add rows of phloem cells in lower part of ellipse
            params_phloem = [p for p in self.params if p["name"] == "phloem"]
            phloem_rows = params_phloem[0]["n_files"]
            phloem_cell_diameter = params_phloem[0]["cell_diameter"]
            # add cambium cells between xylem and phloem
            params_cambium = [p for p in self.params if p["name"] == "cambium"]
    
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
                    
                    xylem_cell = Cell(
                        id_cell=id_cell,
                        id_layer=id_layer,
                        id_group=id_cell,
                        type=cell_type,
                        x=xyl_coord[0],
                        y=xyl_coord[1],
                        diameter=xylem_cell_diameter,
                        angle=np.arctan2(xyl_coord[1]-center.y, xyl_coord[0]-center.x),
                        radius=np.sqrt((xyl_coord[0]-center.x)**2 + (xyl_coord[1]-center.y)**2),
                        area=np.pi * (xylem_cell_diameter/2)**2,
                    )
    
                    # is the point in ellipse
                    if ellipse["polygon"].contains(Point(xyl_coord)):
                        cells_in_ellipses.append(xylem_cell)   
                
                for j_phl in range(1, phloem_rows+1):
                    id_cell += 1
                    phlo_coord = [i*xylem_cell_width - ry + xylem_cell_width/2,  # starting from left to right
                                 j_phl*phloem_cell_height + phloem_cell_height/2] # starting from middle to top
                    # tilt the cells
                    phlo_coord = [phlo_coord[0]*np.cos(angle) - phlo_coord[1]*np.sin(angle), phlo_coord[0]*np.sin(angle) + phlo_coord[1]*np.cos(angle)]
                    phlo_coord = [phlo_coord[0] + center.x, phlo_coord[1] + center.y]
                    phloem_cell_diameter = (xylem_cell_width + phloem_cell_height)/2

                    phloem_cell = Cell(
                        id_cell=id_cell,
                        id_layer=id_layer,
                        id_group=id_cell,
                        type="phloem",
                        x=phlo_coord[0],
                        y=phlo_coord[1],
                        diameter=phloem_cell_diameter,
                        angle=np.arctan2(phlo_coord[1]-center.y, phlo_coord[0]-center.x),
                        radius=np.sqrt((phlo_coord[0]-center.x)**2 + (phlo_coord[1]-center.y)**2),
                        area=np.pi * (phloem_cell_diameter/2)**2,
                    )

                    # is the point in ellipse
                    if ellipse["polygon"].contains(Point(phlo_coord)):
                        cells_in_ellipses.append(phloem_cell)
    
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
                cambium_cell = Cell(
                    id_cell=id_cell,
                    id_layer=id_layer,
                    id_group=id_cell,
                    type="cambium",
                    x=xyl_coord[0],
                    y=xyl_coord[1],
                    diameter=xylem_cell_diameter,
                    angle=np.arctan2(xyl_coord[1]-center.y, xyl_coord[0]-center.x),
                    radius=np.sqrt((xyl_coord[0]-center.x)**2 + (xyl_coord[1]-center.y)**2),
                    area=np.pi * (xylem_cell_diameter/2)**2,
                )

                # is the point in ellipse
                if ellipse["polygon"].contains(Point(xyl_coord)):
                    cells_in_ellipses.append(cambium_cell)
        
            # create a list of polygons for each ellipse
            list_ellipses_polygons.append(ellipse["polygon"])
    
            if debug:
                # plot the ellipse
                color_map = {"Strasburger cell": "red", "xylem": "blue", "phloem": "green", "cambium": "yellow"}
                plt.plot(ellipse["polygon"].exterior.xy[0], ellipse["polygon"].exterior.xy[1])
                # plot the cells
                for cell in cells_in_ellipses:
                    plt.plot(cell.x, cell.y, "o", color = color_map[cell.type])
                plt.show()
            
        return cells_in_ellipses, list_ellipses_polygons

    def _organ_specific_tissues(self):
        """
        Add organ specific tissues.

        For needles, it adds resin ducts and stomata.
        """
        self.add_canal()

        # add stomata
        self.add_stomata()

    def add_canal(self):
        """
        Add resin ducts.
        Selection of portion of mesophyll layer. The two first ducts are located at the edges of the needle.

        diameter of the inner part of the duct full diameter - two parenchyma cells.
        """

        resin_duct_params = [
            p for p in self.params if p["name"] == "resin_duct"
        ]
        if not resin_duct_params:
            return []

        layer_for_duct = [l["name"] for l in self._layers_polygons].index("mesophyll")
        polygon_for_duct = self._layers_polygons[layer_for_duct]["polygon"]

        polygon_for_duct = polygon_for_duct.difference(GeometryProcessor.buffer_polygon(polygon_for_duct, -resin_duct_params[0]["diameter"]*1.2, 0))

        duct_cells = []
        id_cell = len(self.all_cells.cells)+1
        id_group = self.all_cells.get_last_id_group() + 1
        
        add_duct = []
        n_canal = resin_duct_params[0]["n_files"]
        if n_canal < 7:
            n_regions = 7
            if n_canal > 0:
                add_duct.append(3)
            if n_canal > 1:
                add_duct.append(6)

            remaining_places = [i for i in range(n_regions) if i not in add_duct]
            add_duct += list(np.random.choice(remaining_places, n_canal-len(add_duct), replace=False))
        else:
            n_regions = n_canal
            add_duct = range(n_regions)

        polygons_for_duct = GeometryProcessor.pizza_slice(polygon_for_duct, n_regions)
        ducts = []
        for slice_id, slice_polygon in enumerate(polygons_for_duct):

            if slice_id not in add_duct:
                continue

            # create the bounding polygon of the duct
            duct_poly = GeometryProcessor.fit_inner_ellipse(slice_polygon, resin_duct_params[0]["diameter"]/2)
            duct_poly_buffered = GeometryProcessor.buffer_polygon(duct_poly["polygon"], resin_duct_params[0]["cell_diameter"]/2, 0)
            # create the duct polygon
            ducts.append(duct_poly_buffered)
            # create the parenchyma cells polygon
            duct_polygon_buff = GeometryProcessor.buffer_polygon(duct_poly["polygon"], -(resin_duct_params[0]["cell_diameter"]/2)*0.15)
            # create the inner canal polygon
            canal_polygon = GeometryProcessor.buffer_polygon(duct_polygon_buff, -(resin_duct_params[0]["cell_diameter"]))
            # get the centroid of the parenchyma cells 
            x, y = duct_polygon_buff.exterior.coords.xy
            center = duct_poly["polygon"].centroid
            coords = np.column_stack((x, y))
            duct_perim = duct_polygon_buff.length
            coords = GeometryProcessor.resample_coords(coords, target_n_points=np.round(duct_perim/resin_duct_params[0]["cell_diameter"]).astype(int))
            cell_borders = CellGenerator.cell_border(coords, 
                    resin_duct_params[0]["cell_diameter"], 
                    resin_duct_params[0]["cell_diameter"])        


            for i_border, border in enumerate(cell_borders[1:]):
                id_group += 1
                for i_cell, cell_coord in enumerate(border):
                    duct_cells.append(Cell(
                        id_cell=id_cell,
                        id_layer=layer_for_duct,
                        id_group=id_group,
                        type="resin_duct",
                        x=cell_coord[0],
                        y=cell_coord[1],
                        diameter=resin_duct_params[0]["cell_diameter"],
                        angle=np.arctan2(cell_coord[1]-center.y, cell_coord[0]-center.x),
                        radius=np.sqrt((cell_coord[0]-center.x)**2 + (cell_coord[1]-center.y)**2),
                        area=np.pi * (resin_duct_params[0]["cell_diameter"]/2)**2,
                    ))
                    id_cell += 1

            x, y = canal_polygon.exterior.coords.xy
            center = canal_polygon.centroid
            coords = np.column_stack((x, y))
            coords = GeometryProcessor.resample_coords(coords, target_n_points=15)
            id_group += 1
            for i_cell, coord in enumerate(coords[1:]):
                duct_cells.append(Cell(
                    id_cell=id_cell,
                    id_layer=layer_for_duct,
                    id_group=id_group,
                    type="duct",
                    x=coord[0],
                    y=coord[1],
                    diameter=resin_duct_params[0]["diameter"],
                    angle=np.arctan2(coord[1]-center.y, coord[0]-center.x),
                    radius=np.sqrt((coord[0]-center.x)**2 + (coord[1]-center.y)**2),
                    area=np.pi * (resin_duct_params[0]["diameter"]/2)**2,
                ))
                id_cell += 1

        # remove cells that are in the ducts
        for duct in ducts:
            self.all_cells.remove_cells_by_polygon(duct)

        # add the resin duct cells to the list of cells
        self.all_cells.extend_cells(duct_cells)
        self.all_cells.recalculate_cell_properties()


    def add_intercellular_spaces(self):
        """
        Compute intercellular (air space) for the mesophyll layer.

        Mesophyll cell polygons are smoothed, and the difference between the
        original union and the smoothed union yields the gap regions that represent air spaces.  Each air
        space polygon is then simplified to reduce its vertex count.

        """
        intercellular_spaces_params = [
            p for p in self.params if p["name"] == "inter_cellular_spaces"
        ]
        if not intercellular_spaces_params:
            return []

        # Collect mesophyll polygons (cells are *not* touched)
        mesophyll_cells = self.all_cells.get_cells_by_type("mesophyll")
        mesophyll_polys = [
            c.polygon for c in mesophyll_cells if c.polygon is not None
        ]
        if len(mesophyll_polys) < 2:
            return []

        full_union = GeometryProcessor.union_polygons(mesophyll_polys)
        full_union_buffed = full_union.buffer(-mesophyll_cells[0].diameter*0.5)
  
        # smooth the polygons
        smoothed = []
        for poly in mesophyll_polys:
            shrunk = GeometryProcessor.buffer_polygon(poly, 0, smooth_factor=intercellular_spaces_params[0]["mesophyll"])
            if not shrunk.is_empty:
                smoothed.append(shrunk)

        if not smoothed:
            return []

        smoothed_union = GeometryProcessor.union_polygons(smoothed)
        air_region = full_union.difference(smoothed_union)

        # Decompose into individual polygons
        if isinstance(air_region, MultiPolygon):
            raw_air_polys = list(air_region.geoms)
        elif air_region.is_empty:
            return []
        else:
            raw_air_polys = [air_region]

        # Simplify each air space polygon (reduce vertex count)
        # Tolerance ~ 5 % of the median equivalent radius of mesophyll cells
        r_values = [np.sqrt(p.area / np.pi) for p in mesophyll_polys]
        tol = float(np.median(r_values)) * 0.05

        air_space_polys = []
        for poly in raw_air_polys:
            if poly.intersects(full_union_buffed):
                simplified = poly.simplify(tol, preserve_topology=True)
                if not simplified.is_empty and simplified.area > 1E-6:
                    air_space_polys.append(simplified)

        air_union = GeometryProcessor.union_polygons(air_space_polys)

        for cell in mesophyll_cells:
            carved = cell.polygon.difference(air_union)
            if not carved.is_empty:
                cell.polygon = carved

        # create cells for the air spaces
        air_spaces_cells = CellManager()
        id_cell = len(self.all_cells.cells)
        for air_space_polygon in air_space_polys:
            id_cell += 1
            air_space_cell = Cell(
                x = air_space_polygon.centroid.x,
                y = air_space_polygon.centroid.y,
                diameter = np.sqrt(air_space_polygon.area/np.pi)*2,
                id_cell=id_cell,
                id_layer=0,
                id_group=id_cell,
                type="air space",
                polygon=air_space_polygon,
            )
            air_spaces_cells.cells.append(air_space_cell)
        # add the air spaces cells to the all_cells
        
        self.all_cells.cells.extend(air_spaces_cells.cells)
        self.all_cells.cells = CellGenerator.simplify_cells(self.all_cells.cells)


    def add_stomata(self):
        """
        Add stomata to the needle.

        {"name": "stomata", "n_files": 5, "width": 0.07, "depth": 0.01, "sub_chamber": 0.01}

        """
        self.all_cells.recenter_cells()
        stomata_params = [
            p for p in self.params if p["name"] == "stomata"
        ]
        if stomata_params:
            organ_specific_cells = CellManager()
            stomata_params = stomata_params[0]
            n_stomata = stomata_params["n_files"]
            # select "n_files" points on the epidermis

            # Get epidermis cells
            epidermis_cells = self.all_cells.get_cells_by_type("epidermis")
            
            if not epidermis_cells:
                return organ_specific_cells
                
            # Sample `n_stomata` evenly spaced cells, avoiding the very ends
            indices = np.linspace(300, len(epidermis_cells)-np.round(len(epidermis_cells)/n_stomata), n_stomata, dtype=int)
            located_cells = []

            # makes the stomata
            stomata_carve_polys = []
            id_stomata = len(self.all_cells.cells) + 1
            i_cell = id_stomata
            
            for i in indices:
                # get cell triplet with different id_group
                i_group_triplet  = epidermis_cells[i].id_group

                epidermis_cell_triplet = self.all_cells.get_cells_by_groups([i_group_triplet-1, i_group_triplet, i_group_triplet+1])
                located_cell = epidermis_cells[i]
                
                carve_poly, guard_cell_1_poly, guard_cell_2_poly, sub_stomatal_chamber, spacing_poly = CellGenerator.create_stomata(epidermis_cell_triplet, stomata_setting = stomata_params)
                stomata_carve_polys.append(carve_poly)
                
                # guard cell 1
                poly = guard_cell_1_poly.buffer(-located_cell.diameter/5)
                x, y = poly.exterior.coords.xy

                coords = np.column_stack((x, y))

                resampled_coords = GeometryProcessor.resample_coords(coords, 20)
                id_stomata += 1
                for i_coord in resampled_coords:
                    i_cell += 1
                    gc1_cell = Cell(
                        x=i_coord[0], y=i_coord[1],
                        diameter=np.sqrt(poly.area/np.pi)*2,
                        id_cell=i_cell, id_layer=0, id_group=id_stomata,
                        type="guard cell")

                    organ_specific_cells.cells.append(gc1_cell)
                
                # guard cell 2
                poly = guard_cell_2_poly.buffer(-located_cell.diameter/5)
                x, y = poly.exterior.coords.xy

                coords = np.column_stack((x, y))

                resampled_coords = GeometryProcessor.resample_coords(coords, 20)
                id_stomata += 1
                for i_coord in resampled_coords:
                    i_cell += 1
                    gc2_cell = Cell(
                        x=i_coord[0], y=i_coord[1],
                        diameter=np.sqrt(poly.area/np.pi)*2,
                        id_cell=i_cell, id_layer=0, id_group=id_stomata,
                        type="guard cell")
                    organ_specific_cells.cells.append(gc2_cell)

                # chamber
                poly = sub_stomatal_chamber.buffer(-located_cell.diameter/5)
                x, y = poly.exterior.coords.xy

                coords = np.column_stack((x, y))

                resampled_coords = GeometryProcessor.resample_coords(coords, 10)
                id_stomata += 1
                for i_coord in resampled_coords:
                    i_cell += 1
                    chamber_cell = Cell(
                        x=i_coord[0], y=i_coord[1],
                        diameter=np.sqrt(poly.area/np.pi)*2,
                        id_cell=i_cell, id_layer=0, id_group=id_stomata,
                        type="air space")
                    organ_specific_cells.cells.append(chamber_cell)

                # spacing
                poly = spacing_poly.buffer(-stomata_params["width"]/4)
                x, y = poly.exterior.coords.xy

                coords = np.column_stack((x, y))

                resampled_coords = GeometryProcessor.resample_coords(coords, 10)
                id_stomata += 1
                for i_coord in resampled_coords:
                    i_cell += 1
                    spacing_cell = Cell(
                        x=i_coord[0], y=i_coord[1],
                        diameter=np.sqrt(poly.area/np.pi)*2,
                        id_cell=i_cell, id_layer=0, id_group=id_stomata,
                    type="pore")
                    organ_specific_cells.cells.append(spacing_cell)


            # remove cells that are in the stomata
            for stomata in stomata_carve_polys:
                self.all_cells.remove_cells_by_polygon(stomata.buffer(located_cell.diameter/5))
    
            # add the stomata cells to the list of cells
            self.all_cells.extend_cells(organ_specific_cells.cells)
            self.all_cells.recalculate_cell_properties()

        




