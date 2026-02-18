"""
Root anatomy implementation.
"""

import numpy as np
from typing import List, Dict, Any

from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from shapely.affinity import translate

from granap.organ_class import Organ
from granap.layer_class import Layer
from granap.cell_class import Cell
from granap.cell_manager import CellManager
from granap.generate_cell import CellGenerator
from granap.geometry_collection import GeometryProcessor


class RootAnatomy(Organ):
    """
    Root cross-sectional anatomy.
    
    Implements the typical structure of plant roots with
    circular cross-section and vascular cylinder.
    """
    
    def __init__(self, randomness: float = 1.0):
        """
        Initialize root anatomy.
        
        Args:
            randomness: Degree of randomness in cell placement (0-3)
        """
        super().__init__(randomness)
        self._initialize_default_layers()

        # Root specific parameters
        self.vascular_params = {
            "thickness": 0.2,
            "cell_diameter": 0.006,
            "xylem_diameter": 0.05,
            "phloem_diameter": 0.012,
            "n_vascular_bundles": 4
        }
    
    def _initialize_default_layers(self) -> None:
        """Initialize default root layers."""
        # Outer to inner (order: higher = outer)
        self.layer_manager.add_layer(Layer(
            name="epidermis",
            cell_diameter=0.02,
            n_layers=1,
            shift=0.5,
            order=5
        ))
        
        self.layer_manager.add_layer(Layer(
            name="cortex",
            cell_diameter=0.03,
            n_layers=3,
            order=4
        ))
        
        self.layer_manager.add_layer(Layer(
            name="endodermis",
            cell_diameter=0.015,
            cell_width=0.035,
            n_layers=1,
            order=3
        ))
        
        self.layer_manager.add_layer(Layer(
            name="pericycle",
            cell_diameter=0.01,
            cell_width=0.005,
            n_layers=1,
            order=2
        ))
    
    def _create_base_shape(self) -> Polygon:
        """
        Create the circular shape of a root cross-section.
        
        Returns:
            Circular polygon
        """
        radius = self._calculate_root_radius()
        return GeometryProcessor.circle_polygon(radius)
    
    def _calculate_root_radius(self) -> float:
        """Calculate total root radius from layers."""
        radius = self.vascular_params["thickness"] / 2
        
        for layer in self.layer_manager.get_layers():
            if hasattr(layer, 'n_layers'):
                radius += layer.get_total_thickness()
            elif hasattr(layer, 'cell_diameter'):
                radius += layer.cell_diameter
        
        return radius
    
    def _create_central_layers(self, current_polygon: Polygon,
                               params: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create vascular cylinder (xylem and phloem).
        
        Args:
            current_polygon: Current inner polygon boundary
            params: Parameter dictionaries
        
        Returns:
            List of central layer polygon dictionaries
        """
        central_layers = []
        cell_diameter = self.vascular_params["cell_diameter"]
        # first space increment is the cell diameter of the layer with the smallest order
        min_order = min([l.order for l in self.layer_manager.get_layers() if l.order > 0])
        space_increment = self.layer_manager.get_layer_by_order(min_order).cell_diameter/2
        i_layer = len(params)
        
        # Create vascular parenchyma layers
        while current_polygon.area > (cell_diameter / 2)**2 * np.pi:
            current_polygon = GeometryProcessor.buffer_polygon(
                current_polygon,
                -space_increment - cell_diameter / 2,
                smooth_factor=0.6
            )
            
            space_increment = cell_diameter / 2
            
            central_layers.append({
                "name": "vascular_parenchyma",
                "polygon": current_polygon,
                "cell_diameter": cell_diameter,
                "id_layer": i_layer + 1,
                "cell_width": 0
            })
            
            i_layer += 1
        
        return central_layers
    
    def set_vascular_params(self, **kwargs) -> None:
        """
        Update vascular cylinder parameters.
        
        Args:
            **kwargs: Parameter names and values to update
        """
        self.vascular_params.update(kwargs)
        self._invalidate_geometry()
    
    def add_lateral_root_primordium(self, angle: float, distance: float) -> None:
        """
        Add a lateral root primordium (developmental structure).
        
        Args:
            angle: Angular position (radians)
            distance: Distance from center
        
        Note:
            This is a placeholder for future implementation
        """
        # This would require more complex geometry manipulation
        # Left as a placeholder for future enhancement
        pass

    def _create_vascular_tissue(self, polygon_for_vascular: Polygon, debug = False):
        """
        Create vascular tissue (xylem and phloem).
        """
        if self.vascular_params["n_vascular_bundles"] == 0:
            return
        
        self.fit_vascular_elements(polygon_for_vascular)
        # remove the cells in the vascular elements
        vascular_polygons = unary_union(self.vascular_polygons)
        self.all_cells.remove_cells_in_polygon(vascular_polygons)

        # add vascular cells to all_cells
        self.all_cells.extend_cells(self.vascular_cells.cells)
        self.all_cells.recalculate_cell_properties()
        if debug:
            self.all_cells.plot_cells()

    def fit_vascular_elements(self, polygon):
        # from polygon, fit two ellipses
        n_xylem_cells = self.vascular_params["n_vascular_bundles"]

        slices = GeometryProcessor.pizza_slice(polygon, n_xylem_cells)
        cells_in_slices, list_xylem_polygons = self.vascular_elements_in_slice(slices)
        self.vascular_cells = cells_in_slices
        self.vascular_polygons = list_xylem_polygons
    
    def vascular_elements_in_slice(self, slices: List[Polygon]):
        list_xylem_polygons = []
        cells_in_slices = CellManager()
        i_cell = 0
        for i_slice, slice in enumerate(slices):
            
            xylem_polygon = GeometryProcessor.fit_inner_ellipse(slice, self.vascular_params["xylem_diameter"]/2)
            xylem_polygon = xylem_polygon["polygon"]
            xylem_polygon_buff = GeometryProcessor.buffer_polygon(xylem_polygon, -(self.vascular_params["xylem_diameter"]/2)*0.15)
            x, y = xylem_polygon_buff.exterior.coords.xy
            center = xylem_polygon.centroid
            coords = np.column_stack((x, y))
            coords = GeometryProcessor.resample_coords(coords, target_n_points=25)

            # Iterate over centers and borders together
            # coords[1:] slices the centers, cell_borders[1:] slices the corresponding borders
            for cell_border_pts in coords[1:]:
                i_cell += 1
                new_cell = Cell(
                        type="xylem",
                        x=cell_border_pts[0],
                        y=cell_border_pts[1],
                        diameter=self.vascular_params["xylem_diameter"],
                        id_cell=i_slice,
                        id_layer=i_slice,
                        id_group=i_slice,
                        angle=np.arctan2(cell_border_pts[1] - center.y, 
                                          cell_border_pts[0] - center.x),
                        radius=np.sqrt((cell_border_pts[0] - center.x)**2 + 
                                        (cell_border_pts[1] - center.y)**2),
                        area=np.pi * (self.vascular_params["xylem_diameter"]/2)**2
                    )
                cells_in_slices.add_cell(new_cell)

            list_xylem_polygons.append(xylem_polygon)
        return cells_in_slices, list_xylem_polygons
        
    def _which_layer_for_vascular(self, layers_polygons: List[Dict[str, Any]]):
        """
        Find the layer where vascular tissue will be allocated.
        
        Args:
            layers_polygons: List of layer polygon dictionaries
        """
        layer_for_vascular = [l["name"] for l in layers_polygons].index("vascular_parenchyma")
        polygon_for_vascular = layers_polygons[layer_for_vascular]["polygon"]
        return polygon_for_vascular
