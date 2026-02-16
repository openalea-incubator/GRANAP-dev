"""
Needle anatomy implementation.
"""

import numpy as np
from typing import List, Dict, Any
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union

from granap.organ_class import Organ
from granap.cell_class import Cell
from granap.layer_class import Layer
from granap.geometry_collection import GeometryProcessor


class NeedleAnatomy(Organ):
    """
    Needle cross-sectional anatomy.
    
    Implements the specific structure of gymnosperm needle leaves,
    including transfusion tissue and resin ducts.
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize needle anatomy.
        """
        super().__init__()
        # Initialize parameters
        if params is None:
            self._initialize_default_params()
        else:
            self.params = params
        
        self._initialize_params()
        self._initialize_default_layers()

            
    def _initialize_default_params(self) -> Dict[str, Any]:
        """Initialize default parameters."""

        self.params = [
            # P. pinaster
            {"name": "planttype", "value": 3, "organ": "needle", "width": 1.8, "thickness": 1.1}, # global parameters
            {"name": "randomness", "value": 1.0, "smoothness": 0.3}, # 0 = No randomness, 3 = Maximum randomness; smoothness is the smoothing factor (0 = no smoothing, 1 = maximum smoothing)
            {"name": "central_cylinder", "cell_diameter": 0.02, "layer_thickness": 0.43, "layer_length": 1.05, "vascular_width": 0.15, "vascular_height": 0.2}, # Cell diameter in millimeters
            {"name": "transfusion_tissue", "tracheids_diameter": 0.05, "parenchyma_diameter": 0.03, "transfusion_tracheids_ratio": 0.5, "n_layers":2},
            {"name": "endodermis", "cell_diameter": 0.02, "cell_width": 0.05, "n_layers": 1, "order": 3, "shift": 5},
            {"name": "mesophyll", "cell_diameter": 0.08, "cell_width": 0.045, "n_layers": 3, "order": 4, "shift":3},
            {"name": "hypodermis", "cell_diameter": 0.0225, "n_layers": 2, "order": 5},
            {"name": "epidermis", "cell_diameter": 0.02, "n_layers": 1, "order": 6},
            {"name": "xylem", "n_files": 10, "cell_diameter": 0.007, "n_clusters": 4, "n_per_cluster": 3}, # Number of files
            {"name": "phloem", "n_files": 8, "cell_diameter": 0.003}, 
            {"name": "cambium", "cell_diameter": 0.002}, 
            {"name": "resin_ducts", "diameter": 0.5, "n_files": 17},
            {"name": "inter_cellular_space", "ratio": 0.5},
            {"name": "stomata", "n_files": 22, "width": 0.07},
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
        if self.global_params["width"] is None:
            self.global_params["width"] = 0
        if self.global_params["thickness"] is None:
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
        thickness_layer = width_layer
        thickness_total = (2 * thickness_layer) + thickness_vascular
        
        width = 2 * np.sqrt((width_vascular/2 + width_layer)**2 / 
                            (1-(thickness_layer/thickness_total)**2))

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
    
            print(np.rad2deg(angle))
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

# needle = NeedleAnatomy()
# needle.plot_cells()
