"""
Needle anatomy implementation.
"""

import numpy as np
from typing import List, Dict, Any
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.ops import unary_union

from openalea.granap.organ_class import Organ
from openalea.granap.cell_class import Cell
from openalea.granap.cell_manager import CellManager
from openalea.granap.generate_cell import CellGenerator
from openalea.granap.layer_class import Layer
from openalea.granap.geometry_collection import GeometryProcessor
from openalea.granap.shapes import PolygonInterpolator
from openalea.granap.input_data import OrganInputData
import matplotlib.pyplot as plt


class NeedleAnatomy(Organ):
    """
    Needle cross-sectional anatomy.

    Implements the specific structure of gymnosperm needle leaves,
    including transfusion tissue and resin ducts.
    """

    def __init__(self, input_data: Any = None):
        """
        Initialize needle anatomy.
        """
        super().__init__()
        if isinstance(input_data, OrganInputData):
            self.params = input_data.to_dict_list()
        elif isinstance(input_data, list):
            self.params = input_data
        else:
            self.params = OrganInputData.for_needle().to_dict_list()

        self._initialize_params()
        self._initialize_default_layers()

    def _initialize_params(self) -> None:
        """Initialize central layers."""
        # 1. Global params
        self.global_params = next(p for p in self.params if p["name"] == "planttype")
        # 2. Central cylinder params
        self.central_cylinder_params = next(p for p in self.params if p["name"] == "central_cylinder")
        # 3. Transfusion tissue params
        self.transfusion_params = next(p for p in self.params if p["name"] == "transfusion_tissue")

        # 3. Intercellular spaces / aerenchyma — store raw config dicts directly
        self.intercellular_spaces_params = [p for p in self.params if p["name"] == "inter_cellular_spaces"]
        self.aerenchyma_params = next((p for p in self.params if p["name"] == "aerenchyma"), {})

        # 4. Extract layer definitions (any param with 'order' that is not a vascular zone)
        self.layers = [param for param in self.params if "order" in param]
        self.layers = sorted(self.layers, key=lambda x: x["order"])        
    
    def _initialize_default_layers(self) -> None:
        """Initialize default needle layers."""
        for param in self.layers:
            self.layer_manager.add_layer(Layer.from_dict(param))
    
    def _create_base_shape(self) -> Polygon:
        """
        Create the half-ellipse shape of a needle cross-section.
        
        Returns:
            Half-ellipse polygon
        """
        # if width and thickness are not provided (set to 0), calculate them from the layers
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
        transfusion_type = self.transfusion_params.get("transfusion_type", False)
        tt_ratio = self.transfusion_params.get("transfusion_tracheids_ratio", 0.5)
        p_tt = tt_ratio / (1.0 + tt_ratio) if tt_ratio > 0 else 0.0

        i_layer = len(params)

        while current_polygon.area > (parenchyma_diameter / 2)**2 * np.pi:
            if transfusion_layers_remaining > 0:
                avg_diameter = (tp_diameter + tt_diameter) / 2
                transfusion_layers_remaining -= 1

                current_polygon = GeometryProcessor.buffer_polygon(
                    current_polygon,
                    -space_increment - avg_diameter / 2,
                    smooth_factor=0.6
                )

                space_increment = avg_diameter / 2

                layer_dict = {
                    "name": "transfusion",
                    "polygon": current_polygon,
                    "cell_diameter": avg_diameter,
                    "id_layer": i_layer + 1,
                    "cell_width": 0
                }
                if transfusion_type:
                    layer_dict.update({
                        "transfusion_type": True,
                        "tt_diameter": tt_diameter,
                        "tp_diameter": tp_diameter,
                        "p_tt": p_tt,
                    })
                central_layers.append(layer_dict)
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
                        type="resin duct",
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
        """Orchestrate intercellular space and aerenchyma generation."""
        self.add_intercellular()
        self.add_aerenchyma()
        self.merge_intercellular_aerenchyma()

    def add_intercellular(self):
        """Compute air spaces for each inter_cellular_spaces entry.

        Each entry may list one or more tissues. When multiple tissues are given,
        cells from all of them are processed together so that intercellular spaces
        are also generated at the boundary between adjacent tissues.
        Smoothness can be a single float (applied to every tissue) or a list with
        one value per tissue.
        """
        for ics in self.intercellular_spaces_params:
            self._apply_intercellular(ics)

    def _apply_intercellular(self, ics: dict) -> None:
        """Apply one inter_cellular_spaces entry to the relevant tissue cells."""
        tissues = ics.get("tissue", [])
        if isinstance(tissues, str):
            tissues = [tissues]
        if not tissues:
            return

        smoothness = ics.get("smoothness", 0)
        if isinstance(smoothness, (int, float)):
            smoothness_per_tissue = [float(smoothness)] * len(tissues)
        else:
            smoothness_per_tissue = [float(s) for s in smoothness]

        if not any(smoothness_per_tissue):
            return

        # Collect cells from all tissues, tracking the smoothness for each cell
        all_tissue_cells = []
        cell_smoothness: dict = {}
        for tissue_name, s in zip(tissues, smoothness_per_tissue):
            cells = self.all_cells.get_cells_by_type(tissue_name)
            for c in cells:
                cell_smoothness[id(c)] = s
            all_tissue_cells.extend(cells)

        tissue_polys = [c.polygon for c in all_tissue_cells if c.polygon is not None]
        if len(tissue_polys) < 2:
            return

        full_union = GeometryProcessor.union_polygons(tissue_polys)
        min_diameter = min(c.diameter for c in all_tissue_cells)
        full_union_buffed = full_union.buffer(-min_diameter * 0.5)

        smoothed = []
        for cell in all_tissue_cells:
            if cell.polygon is None:
                continue
            s = cell_smoothness[id(cell)]
            shrunk = GeometryProcessor.buffer_polygon(cell.polygon, 0, smooth_factor=s)
            if not shrunk.is_empty:
                smoothed.append(shrunk)

        if not smoothed:
            return

        smoothed_union = GeometryProcessor.union_polygons(smoothed)
        air_region = full_union.difference(smoothed_union)

        if isinstance(air_region, MultiPolygon):
            raw_air_polys = list(air_region.geoms)
        elif air_region.is_empty:
            return
        else:
            raw_air_polys = [air_region]

        r_values = [np.sqrt(p.area / np.pi) for p in tissue_polys]
        tol = float(np.median(r_values)) * 0.05

        air_space_polys = []
        for poly in raw_air_polys:
            if poly.intersects(full_union_buffed):
                simplified = poly.simplify(tol, preserve_topology=True)
                if not simplified.is_empty and simplified.area > 1E-6:
                    air_space_polys.append(simplified)

        if not air_space_polys:
            return

        air_union = GeometryProcessor.union_polygons(air_space_polys)

        for cell in all_tissue_cells:
            if cell.polygon is None:
                continue
            carved = cell.polygon.difference(air_union)
            if not carved.is_empty and carved.area > 1E-6:
                cell.polygon = carved
            else:
                cell.polygon = None

        id_cell = len(self.all_cells.cells)
        for air_space_polygon in air_space_polys:
            id_cell += 1
            self.all_cells.cells.append(Cell(
                x=air_space_polygon.centroid.x,
                y=air_space_polygon.centroid.y,
                diameter=np.sqrt(air_space_polygon.area / np.pi) * 2,
                id_cell=id_cell,
                id_layer=0,
                id_group=id_cell,
                type="air space",
                polygon=air_space_polygon,
            ))

        self.all_cells.cells = CellGenerator.simplify_cells(self.all_cells.cells)

    def add_aerenchyma(self):
        """Generate aerenchyma in the tissue defined in aerenchyma_params."""
        aerenchyma_prop = self.aerenchyma_params.get("aerenchyma_proportion", 0)
        if not aerenchyma_prop:
            return

        tissue = self.aerenchyma_params.get("tissue")
        n_files = int(self.aerenchyma_params.get("n_files", 1))
        aerenchyma_type = int(self.aerenchyma_params.get("aerenchyma_type", 1))

        self._aerenchyma_n_files = n_files
        self._aerenchyma_start_angle = np.random.uniform(0, 2 * np.pi)
        start_angle = self._aerenchyma_start_angle

        def cell_quadrant(cell):
            cell_angle = np.arctan2(cell.y, cell.x) % (2 * np.pi)
            rel = (cell_angle - start_angle) % (2 * np.pi)
            return int(rel / (2 * np.pi / n_files)) % n_files

        if aerenchyma_prop > 1:
            print("Aerenchyma proportion is greater than 1, setting it to 1")
            aerenchyma_prop = 1

        tissue_cells = self.all_cells.get_cells_by_type(tissue)
        if not tissue_cells:
            return

        max_tissue_layer = max(c.id_layer for c in tissue_cells)
        candidates = [c for c in tissue_cells if c.id_layer < max_tissue_layer]
        candidates.extend(self.all_cells.get_cells_by_type("air space"))

        if not candidates:
            return

        total_tissue_area = sum(c.polygon.area for c in tissue_cells if c.polygon is not None)
        total_air_area = sum(c.polygon.area for c in self.all_cells.get_cells_by_type("air space") if c.polygon is not None)
        max_possible_area = sum(c.polygon.area for c in candidates if c.polygon is not None)

        target_aerenchyma_area = (total_tissue_area + total_air_area) * aerenchyma_prop

        if target_aerenchyma_area > max_possible_area:
            print(f"Warning: asked proportion ({aerenchyma_prop:.2f}) requires {target_aerenchyma_area:.2f} area, which is greater than available cells ({max_possible_area:.2f}). Lowering aerenchyma_proportion.")
            aerenchyma_prop = max_possible_area / (total_tissue_area + total_air_area)
            target_aerenchyma_area = max_possible_area

        print(f"Targeted aerenchyma prop: {(target_aerenchyma_area / (total_tissue_area + total_air_area)):.3f}")

        target_per_quadrant = (target_aerenchyma_area - total_air_area) / ((n_files) ** 1.12 + 1)

        quadrant_buckets = [[] for _ in range(n_files)]
        for c in candidates:
            quadrant_buckets[cell_quadrant(c)].append(c)

        if aerenchyma_type == 1:
            for q, bucket in enumerate(quadrant_buckets):
                central_angle = (start_angle + (q + 0.5) * 2 * np.pi / n_files) % (2 * np.pi)
                def _ang_dist(cell, ca=central_angle):
                    a = np.arctan2(cell.y, cell.x) % (2 * np.pi)
                    d = abs(a - ca)
                    return min(d, 2 * np.pi - d)
                bucket.sort(key=_ang_dist)
        elif aerenchyma_type == 2:
            for q, bucket in enumerate(quadrant_buckets):
                if not bucket:
                    continue
                central_angle = (start_angle + (q + 0.5) * 2 * np.pi / n_files) % (2 * np.pi)
                def _ang_dist_seed(cell, ca=central_angle):
                    a = np.arctan2(cell.y, cell.x) % (2 * np.pi)
                    d = abs(a - ca)
                    return min(d, 2 * np.pi - d)
                seed = min(bucket, key=_ang_dist_seed)
                bucket.sort(key=lambda c, s=seed: np.hypot(c.x - s.x, c.y - s.y))

        quadrant_area = [0.0] * n_files
        quadrant_idx = [0] * n_files

        changed = True
        while changed:
            changed = False
            for q in range(n_files):
                if quadrant_area[q] >= target_per_quadrant:
                    continue
                bucket = quadrant_buckets[q]
                while quadrant_idx[q] < len(bucket):
                    cell = bucket[quadrant_idx[q]]
                    quadrant_idx[q] += 1
                    if cell.type != "air space" and cell.polygon is not None:
                        cell.type = "air space"
                        quadrant_area[q] += cell.polygon.area
                        changed = True
                        break

        tissue = self.aerenchyma_params.get("tissue")
        total_tissue_area = sum(c.polygon.area for c in self.all_cells.get_cells_by_type(tissue) if c.polygon is not None)
        total_air_area = sum(c.polygon.area for c in self.all_cells.get_cells_by_type("air space") if c.polygon is not None)
        print(f"Actual aerenchyma prop: {(total_air_area / (total_tissue_area + total_air_area)):.3f}")

    def merge_intercellular_aerenchyma(self):
        """Fuse touching air-space cells within the same angular sector, then carve tissue cells."""
        from collections import defaultdict

        n_files = getattr(self, '_aerenchyma_n_files', 1)
        start_angle = getattr(self, '_aerenchyma_start_angle', 0.0)

        def cell_quadrant(cell):
            cell_angle = np.arctan2(cell.y, cell.x) % (2 * np.pi)
            rel = (cell_angle - start_angle) % (2 * np.pi)
            return int(rel / (2 * np.pi / n_files)) % n_files

        seen_ids: set = set()
        merge_pool = []
        for c in list(self.all_cells.cells):
            if c.type == "air space" and c.polygon is not None:
                oid = id(c)
                if oid not in seen_ids:
                    seen_ids.add(oid)
                    merge_pool.append(c)

        if merge_pool:
            n_pool = len(merge_pool)
            parent = list(range(n_pool))
            cell_quadrants = [cell_quadrant(c) for c in merge_pool]

            def _find(i):
                while parent[i] != i:
                    parent[i] = parent[parent[i]]
                    i = parent[i]
                return i

            def _union(i, j):
                ri, rj = _find(i), _find(j)
                if ri != rj:
                    parent[ri] = rj

            for i in range(n_pool):
                for j in range(i + 1, n_pool):
                    if cell_quadrants[i] != cell_quadrants[j]:
                        continue
                    if merge_pool[i].polygon.touches(merge_pool[j].polygon) or merge_pool[i].polygon.intersects(merge_pool[j].polygon):
                        _union(i, j)

            groups: dict = defaultdict(list)
            for i, c in enumerate(merge_pool):
                groups[_find(i)].append(c)

            fused_cells = []
            for group in groups.values():
                if len(group) == 1:
                    fused_cells.append(group[0])
                    continue
                fused_polygon = unary_union([c.polygon for c in group])
                fused_cells.append(Cell(
                    x=fused_polygon.centroid.x,
                    y=fused_polygon.centroid.y,
                    diameter=np.sqrt(fused_polygon.area / np.pi) * 2,
                    id_cell=min(c.id_cell for c in group),
                    id_layer=int(np.ceil(np.mean([c.id_layer for c in group]))),
                    id_group=min(c.id_group for c in group),
                    type="air space",
                    polygon=fused_polygon,
                ))

            self.all_cells.remove_cells_by_ids([c.id_cell for c in merge_pool])
            self.all_cells.cells.extend(fused_cells)

        self.all_cells.cells = CellGenerator.simplify_cells(self.all_cells.cells)

        # Carve tissue cells that are trapped inside air spaces
        tissue = self.aerenchyma_params.get("tissue")
        air_spaces = self.all_cells.get_cells_by_type("air space")
        tissue_cells = self.all_cells.get_cells_by_type(tissue)
        tissue_cells.extend(a for a in air_spaces if a.id_layer == 0)

        air_union = unary_union([a.polygon for a in air_spaces if a.polygon is not None and a.id_layer != 0])

        for cell in tissue_cells:
            carved = cell.polygon.difference(air_union)
            if not carved.is_empty and carved.area > 1E-6:
                cell.polygon = carved
            else:
                self.all_cells.remove_cells_by_ids([cell.id_cell])



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

        




