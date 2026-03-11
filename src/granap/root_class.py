"""
Root anatomy implementation.
"""

import numpy as np
from typing import List, Dict, Any

from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.ops import unary_union
from shapely.affinity import translate

from granap.organ_class import Organ
from granap.layer_class import Layer
from granap.cell_class import Cell
from granap.cell_manager import CellManager
from granap.geometry_collection import GeometryProcessor


class RootAnatomy(Organ):
    """
    Root cross-sectional anatomy.
    
    Implements the typical structure of plant roots with
    circular cross-section and vascular cylinder.
    """
    
    def __init__(self):
        """
        Initialize root anatomy.
        """
        super().__init__()
        self._initialize_default_layers()

        # Root specific parameters
        self.vascular_params = {
            "thickness": 0.23,
            "cell_diameter": 0.012,
            "xylem_diameter": 0.06,
            "phloem_diameter": 0.012,
            "n_vascular_bundles": 5,
            "ratio_proto_meta": 2.5
        }
        self.intercellular_spaces_params = {
            "cortex": 0.1
        }
    
    def _initialize_default_layers(self) -> None:
        """Initialize default root layers."""
        # Outer to inner (order: higher = outer)
        self.layer_manager.add_layer(Layer(
            name="epidermis",
            cell_diameter=0.015,
            n_layers=1,
            shift=0.5,
            order=6
        ))

        self.layer_manager.add_layer(Layer(
            name="exodermis",
            cell_diameter=0.035,
            n_layers=1,
            order=5
        ))
        
        self.layer_manager.add_layer(Layer(
            name="cortex",
            cell_diameter=0.042,
            n_layers=5,
            order=4
        ))
        
        self.layer_manager.add_layer(Layer(
            name="endodermis",
            cell_diameter=0.015,
            cell_width=0.030,
            n_layers=1,
            order=3
        ))
        
        self.layer_manager.add_layer(Layer(
            name="pericycle",
            cell_diameter=0.01,
            cell_width=0.009,
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

    def _organ_specific_tissues(self):
        """
        Add organ specific tissues.
        """
        pass

    def add_intercellular_spaces(self):
        """
        Add intercellular spaces.
        """
        if self.intercellular_spaces_params["cortex"] == 0:
            return
        else:
                # Collect mesophyll polygons (cells are *not* touched)
            cortex_cells = self.all_cells.get_cells_by_type("cortex")
            cortex_polys = [
                c.polygon for c in cortex_cells if c.polygon is not None
            ]
            if len(cortex_polys) < 2:
                return []

            full_union = GeometryProcessor.union_polygons(cortex_polys)
            full_union_buffed = full_union.buffer(-cortex_cells[0].diameter*0.5)

            smoothed = []
            for poly in cortex_polys:
                shrunk = GeometryProcessor.buffer_polygon(poly, 0, smooth_factor=self.intercellular_spaces_params["cortex"])
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
            r_values = [np.sqrt(p.area / np.pi) for p in cortex_polys]
            tol = float(np.median(r_values)) * 0.05
    
            air_space_polys = []
            for poly in raw_air_polys:
                if poly.intersects(full_union_buffed):
                    simplified = poly.simplify(tol, preserve_topology=True)
                    if not simplified.is_empty and simplified.area > 1E-6:
                        air_space_polys.append(simplified)
    
            air_union = GeometryProcessor.union_polygons(air_space_polys)
    
            for cell in cortex_cells:
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
            return air_spaces_cells

    def _organ_specific_cells(self) -> CellManager:
        """
        Add organ specific cells.
        
        Returns:
            CellManager object with organ specific cells
        """

        organ_specific_cells = CellManager()

        # add root hairs
        organ_specific_cells = self.add_root_hairs(organ_specific_cells)

        return organ_specific_cells

    def add_root_hairs(self, organ_specific_cells: CellManager) -> CellManager:
        """
        Add root hairs to the root.
        
        Args:
            organ_specific_cells: CellManager object with organ specific cells
        
        Returns:
            CellManager object with root hairs
        """
        return organ_specific_cells

        
