"""
Needle anatomy implementation.
"""

import dataclasses
import numpy as np
from typing import List, Dict, Any
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union

from openalea.granap.organ_class import Organ
from openalea.granap.cell_class import Cell
from openalea.granap.cell_manager import CellManager
from openalea.granap.generate_cell import CellGenerator
from openalea.granap.layer_class import Layer, LayerPolygon
from openalea.granap.geometry_collection import GeometryProcessor
from openalea.granap.shapes import PolygonInterpolator
from openalea.granap.input_data import OrganInputData
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Module-level constants — geometry tuning parameters
# ---------------------------------------------------------------------------
# Number of epidermis border-point cells to skip at the start of the boundary
_STOMATA_SKIP_BORDER_PTS: int = 300

# The mesophyll ring used for duct placement is the outer annulus whose inner
# edge is 1.2× duct diameters from the mesophyll boundary.
_DUCT_RING_BUFFER_FACTOR: float = 1.2

# The parenchyma-ring polygon is obtained by shrinking the fitted ellipse
# inward by this fraction of cell_diameter (prevents ring cells from sitting
# on the exact ellipse edge).
_DUCT_RING_INNER_SHRINK: float = 0.15

# Number of resampled points for the inner lumen (canal) polygon.
_CANAL_RESAMPLE_PTS: int = 15

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
            try:
                new_poly = interp.fast_interpolate(t)
                if not new_poly.is_empty and new_poly.is_valid:
                    layers_polygons[i] = dataclasses.replace(layers_polygons[i], polygon=new_poly)
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

                central_layers.append(LayerPolygon(
                    name="transfusion",
                    polygon=current_polygon,
                    cell_diameter=avg_diameter,
                    id_layer=i_layer + 1,
                    transfusion_type=transfusion_type,
                    tt_diameter=tt_diameter if transfusion_type else 0.0,
                    tp_diameter=tp_diameter if transfusion_type else 0.0,
                    p_tt=p_tt if transfusion_type else 0.0,
                ))
            else:
                # Parenchyma
                current_polygon = GeometryProcessor.buffer_polygon(
                    current_polygon,
                    -space_increment - parenchyma_diameter / 2,
                    smooth_factor=0.7
                )

                space_increment = parenchyma_diameter / 2

                central_layers.append(LayerPolygon(
                    name="parenchyma",
                    polygon=current_polygon,
                    cell_diameter=parenchyma_diameter,
                    id_layer=i_layer + 1,
                ))
            
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

        The remove-mask + extend step is done once in Organ.generate_cells()
        after this method returns, so we only need to populate vascular_cells
        and vascular_polygons here.
        """
        self.fit_vascular_elements(polygon)

    def fit_vascular_elements(self, polygon):
        # from polygon, fit two ellipses
        rx = self.central_cylinder_params["vascular_width"]/2
        ry = self.central_cylinder_params["vascular_height"]/2
        ellipses = GeometryProcessor.two_ellipses(polygon, rx, ry)
        cells_in_ellipses, list_ellipses_polygons = self.vascular_elements_in_ellipses(ellipses)
        vascular_cm = CellManager()
        vascular_cm.cells = cells_in_ellipses
        self.vascular_cells = vascular_cm
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
        self.add_stomata()

    # ------------------------------------------------------------------
    # Geometry helpers — pure computation, no cell placement
    # ------------------------------------------------------------------

    def _duct_zone_data(self, layers_polygons):
        """
        Compute resin duct geometry from layer polygons without placing cells.

        Returns (duct_data, rdp) where duct_data is a list of per-duct dicts:
          - "outer":        mask polygon used to remove existing cells
          - "ring":         parenchyma ring polygon (for resin-duct cell placement)
          - "canal":        inner lumen polygon (for duct cell placement)
          - "ring_center":  centroid of the fitted inner ellipse (angle reference)
        rdp is the resin_duct parameter dict.
        Returns ([], None) when there are no resin_duct params or no mesophyll layer.
        """
        rdp_list = [p for p in self.params if p["name"] == "resin_duct"]
        if not rdp_list:
            return [], None
        rdp = rdp_list[0]

        layer_names = [l["name"] for l in layers_polygons]
        if "mesophyll" not in layer_names:
            return [], None

        polygon_for_duct = layers_polygons[layer_names.index("mesophyll")]["polygon"]
        polygon_for_duct = polygon_for_duct.difference(
            GeometryProcessor.buffer_polygon(polygon_for_duct, -rdp["diameter"] * _DUCT_RING_BUFFER_FACTOR, 0)
        )

        n_canal = rdp["n_files"]
        add_duct = []
        if n_canal < 7:
            n_regions = 7
            if n_canal > 0:
                add_duct.append(3)
            if n_canal > 1:
                add_duct.append(6)
            remaining_places = [i for i in range(n_regions) if i not in add_duct]
            add_duct += list(np.random.choice(remaining_places, n_canal - len(add_duct), replace=False))
        else:
            n_regions = n_canal
            add_duct = list(range(n_regions))

        duct_data = []
        for slice_id, slice_polygon in enumerate(GeometryProcessor.pizza_slice(polygon_for_duct, n_regions)):
            if slice_id not in add_duct:
                continue
            duct_poly       = GeometryProcessor.fit_inner_ellipse(slice_polygon, rdp["diameter"] / 2)
            outer           = GeometryProcessor.buffer_polygon(duct_poly["polygon"],  rdp["cell_diameter"] / 2, 0)
            ring            = GeometryProcessor.buffer_polygon(duct_poly["polygon"], -(rdp["cell_diameter"] / 2) * _DUCT_RING_INNER_SHRINK)
            canal           = GeometryProcessor.buffer_polygon(ring,                 -rdp["cell_diameter"])
            duct_data.append({
                "outer":       outer,
                "ring":        ring,
                "canal":       canal,
                "ring_center": duct_poly["polygon"].centroid,
            })

        return duct_data, rdp

    @staticmethod
    def _stomata_carve_polygons(triplet_centers, sp, cell_diam):
        """
        Compute stomata geometry from triplet positions without placing cells.

        triplet_centers: list of ((px,py), (cx,cy), (nx,ny)) — prev/curr/next
                         epidermis seed positions for each stoma.
        sp:             stomata parameter dict.
        cell_diam:      epidermis cell diameter.

        Returns a list of (carve_poly, gc1, gc2, chamber, pore) tuples,
        one per successfully computed stoma.
        """
        results = []
        for k, (prev_xy, curr_xy, next_xy) in enumerate(triplet_centers):
            mock_cells = [
                Cell(x=prev_xy[0], y=prev_xy[1], diameter=cell_diam,
                     id_group=3 * k,     id_cell=3 * k,     id_layer=0, type="epidermis"),
                Cell(x=curr_xy[0], y=curr_xy[1], diameter=cell_diam,
                     id_group=3 * k + 1, id_cell=3 * k + 1, id_layer=0, type="epidermis"),
                Cell(x=next_xy[0], y=next_xy[1], diameter=cell_diam,
                     id_group=3 * k + 2, id_cell=3 * k + 2, id_layer=0, type="epidermis"),
            ]
            try:
                results.append(CellGenerator.create_stomata(mock_cells, stomata_setting=sp))
            except Exception:
                pass
        return results

    # ------------------------------------------------------------------
    # Cell-placement methods — call geometry helpers then place cells
    # ------------------------------------------------------------------

    def add_canal(self):
        """Add resin ducts (parenchyma ring + inner lumen) to the mesophyll."""
        duct_data, rdp = self._duct_zone_data(self._layers_polygons)
        if not duct_data:
            return

        layer_for_duct = [l["name"] for l in self._layers_polygons].index("mesophyll")
        duct_cells = []
        id_cell  = len(self.all_cells.cells) + 1
        id_group = self.all_cells.get_last_id_group() + 1

        for duct in duct_data:
            ring_center = duct["ring_center"]

            # resin-duct parenchyma cells along the ring
            x, y   = duct["ring"].exterior.coords.xy
            coords  = np.column_stack((x, y))
            coords  = GeometryProcessor.resample_coords(
                coords,
                target_n_points=np.round(duct["ring"].length / rdp["cell_diameter"]).astype(int)
            )
            for border in CellGenerator.cell_border(coords, rdp["cell_diameter"], rdp["cell_diameter"])[1:]:
                id_group += 1
                for cell_coord in border:
                    duct_cells.append(Cell(
                        id_cell=id_cell, id_layer=layer_for_duct, id_group=id_group,
                        type="resin duct",
                        x=cell_coord[0], y=cell_coord[1],
                        diameter=rdp["cell_diameter"],
                        angle=np.arctan2(cell_coord[1] - ring_center.y, cell_coord[0] - ring_center.x),
                        radius=np.sqrt((cell_coord[0] - ring_center.x)**2 + (cell_coord[1] - ring_center.y)**2),
                        area=np.pi * (rdp["cell_diameter"] / 2)**2,
                    ))
                    id_cell += 1

            # inner lumen cells along the canal
            canal_center = duct["canal"].centroid
            x, y  = duct["canal"].exterior.coords.xy
            coords = GeometryProcessor.resample_coords(np.column_stack((x, y)), target_n_points=_CANAL_RESAMPLE_PTS)
            id_group += 1
            for coord in coords[1:]:
                duct_cells.append(Cell(
                    id_cell=id_cell, id_layer=layer_for_duct, id_group=id_group,
                    type="duct",
                    x=coord[0], y=coord[1],
                    diameter=rdp["diameter"],
                    angle=np.arctan2(coord[1] - canal_center.y, coord[0] - canal_center.x),
                    radius=np.sqrt((coord[0] - canal_center.x)**2 + (coord[1] - canal_center.y)**2),
                    area=np.pi * (rdp["diameter"] / 2)**2,
                ))
                id_cell += 1

        for duct in duct_data:
            self.all_cells.remove_cells_by_polygon(duct["outer"])
        self.all_cells.extend_cells(duct_cells)
        self.all_cells.recalculate_cell_properties()

    def _aerenchyma_target_denominator(self, n_files: int) -> float:
        return float(n_files ** 1.12 + 1)

    def add_stomata(self):
        """Add stomata to the needle epidermis."""
        self.all_cells.recenter_cells()
        stomata_params_list = [p for p in self.params if p["name"] == "stomata"]
        if not stomata_params_list:
            return

        sp         = stomata_params_list[0]
        n_stomata  = sp["n_files"]
        epidermis_cells = self.all_cells.get_cells_by_type("epidermis")
        if not epidermis_cells:
            return

        cell_diam = epidermis_cells[0].diameter

        # Sample n_stomata evenly spaced groups, avoiding the very ends
        indices = np.linspace(
            _STOMATA_SKIP_BORDER_PTS,
            len(epidermis_cells) - np.round(len(epidermis_cells) / n_stomata),
            n_stomata, dtype=int
        )

        # Build triplet centroids from placed epidermis cell groups
        triplet_centers = []
        for i in indices:
            g = epidermis_cells[i].id_group
            try:
                triplet_centers.append((
                    self.all_cells.get_centroid_of_group(g - 1),
                    self.all_cells.get_centroid_of_group(g),
                    self.all_cells.get_centroid_of_group(g + 1),
                ))
            except KeyError:
                pass  # adjacent group was removed; skip this stomata position

        stomata_geoms = self._stomata_carve_polygons(triplet_centers, sp, cell_diam)

        organ_specific_cells = CellManager()
        stomata_carve_polys  = []
        id_stomata = len(self.all_cells.cells) + 1
        i_cell     = id_stomata

        for carve_poly, gc1, gc2, chamber, pore in stomata_geoms:
            stomata_carve_polys.append(carve_poly)

            for raw_poly, cell_type, n_pts in [
                (gc1,     "guard cell", 20),
                (gc2,     "guard cell", 20),
                (chamber, "air space",  10),
            ]:
                poly   = raw_poly.buffer(-cell_diam / 5)
                coords = GeometryProcessor.resample_coords(
                    np.column_stack(poly.exterior.coords.xy), n_pts
                )
                id_stomata += 1
                for i_coord in coords:
                    i_cell += 1
                    organ_specific_cells.cells.append(Cell(
                        x=i_coord[0], y=i_coord[1],
                        diameter=np.sqrt(poly.area / np.pi) * 2,
                        id_cell=i_cell, id_layer=0, id_group=id_stomata,
                        type=cell_type,
                    ))

            poly   = pore.buffer(-sp["width"] / 4)
            coords = GeometryProcessor.resample_coords(
                np.column_stack(poly.exterior.coords.xy), 10
            )
            id_stomata += 1
            for i_coord in coords:
                i_cell += 1
                organ_specific_cells.cells.append(Cell(
                    x=i_coord[0], y=i_coord[1],
                    diameter=np.sqrt(poly.area / np.pi) * 2,
                    id_cell=i_cell, id_layer=0, id_group=id_stomata,
                    type="pore",
                ))

        for carve in stomata_carve_polys:
            self.all_cells.remove_cells_by_polygon(carve.buffer(cell_diam / 5))
        self.all_cells.extend_cells(organ_specific_cells.cells)
        self.all_cells.recalculate_cell_properties()

    # ------------------------------------------------------------------
    # Visualization hook
    # ------------------------------------------------------------------

    def _extra_tissue_polygons(self, layers_polygons):
        """
        Return resin-duct and stomata polygons for plot_tissues visualization,
        without placing any cell.

        Resin ducts are exact (same geometry as add_canal).
        Stomata positions are approximated from epidermis seed positions using
        the same index formula as add_stomata, so count and placement are
        consistent with generate_cells output.
        """
        extra = {}

        # Resin ducts — delegate entirely to the shared geometry helper
        duct_data, _ = self._duct_zone_data(layers_polygons)
        if duct_data:
            extra["resin_duct"]  = [d["outer"] for d in duct_data]
            extra["resin_canal"] = [d["canal"] for d in duct_data]

        # Stomata — approximate seed positions from the epidermis layer polygon
        stomata_params_list = [p for p in self.params if p["name"] == "stomata"]
        if stomata_params_list and layers_polygons:
            sp        = stomata_params_list[0]
            n_stomata = sp["n_files"]
            layer_names = [l["name"] for l in layers_polygons]
            if "epidermis" in layer_names:
                epi_layer  = layers_polygons[layer_names.index("epidermis")]
                cell_diam  = epi_layer.get("cell_diameter", 0.015)
                cell_width = epi_layer.get("cell_width", 0)
                shift      = epi_layer.get("shift", 0)

                seeds   = CellGenerator.cells_on_layer(epi_layer["polygon"], cell_diam, cell_width, shift)
                # n_border matches cell_border: 14 pts if rectangular, 9 if circular
                n_border    = 14 if cell_width != 0 else 9
                n_epi_cells = max(1, (len(seeds) - 1) * n_border)

                # Mirror the index selection from add_stomata
                end_idx = n_epi_cells - int(np.round(n_epi_cells / n_stomata))
                indices = np.linspace(_STOMATA_SKIP_BORDER_PTS, end_idx, n_stomata, dtype=int)
                seed_indices = np.clip(indices // n_border, 0, len(seeds) - 2)

                triplet_centers = [
                    (
                        tuple(seeds[max(0, si - 1)]),
                        tuple(seeds[si]),
                        tuple(seeds[min(len(seeds) - 2, si + 1)]),
                    )
                    for si in seed_indices
                ]

                stomata_geoms = self._stomata_carve_polygons(triplet_centers, sp, cell_diam)
                extra["stomata"] = [geom[0] for geom in stomata_geoms]

        return extra

        




