"""
Plant anatomy base module providing abstract interface.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from scipy.sparse import lil_matrix

from granap.layer_class import Layer
from granap.layer_manager import LayerManager
from granap.geometry_collection import GeometryProcessor
from granap.generate_cell import CellGenerator
from granap.cell_class import Cell
from granap.cell_manager import CellManager
from granap.network_base import AbstractNetwork
from granap.input_data import OrganInputData

class Organ(AbstractNetwork, ABC):
    """
    Abstract base class for plant anatomical structures.
    
    Defines the interface and common functionality for generating
    cross-sectional anatomy of different plant types.
    Inherits from AbstractNetwork for hydraulic network construction.
    """
    
    def __init__(self, randomness: float = 1.0):
        """
        Initialize the anatomy structure.
        
        Args:
            randomness: Degree of randomness in cell placement (0-3)
        """
        AbstractNetwork.__init__(self)
        self.layer_manager = LayerManager()
        self.randomness = randomness
        self._base_polygon: Optional[Polygon] = None
        self._layers_polygons: List[Dict[str, Any]] = []
        self._cells_gdf: Optional[gpd.GeoDataFrame] = None
        self.all_cells = CellManager()

    @classmethod
    def create_from_input(cls, input_data: OrganInputData) -> "Organ":
        """
        Factory method to initialize the appropriate Organ subclass 
        (RootAnatomy or NeedleAnatomy) from an OrganInputData instance.
        """
        # Determine the organ type from the parameters
        ptype_param = next((p for p in input_data.params if p["name"] == "planttype"), None)
        organ_type = None

        if ptype_param:
            if ptype_param.get("organ") == "needle" or ptype_param.get("value") == 3:
                organ_type = "needle"
            elif ptype_param.get("organ") == "root" or ptype_param.get("value") in [1, 2, 1.0, 2.0]:
                organ_type = "root"

        # Fallback to duck-typing the input parameters if 'organ' isn't explicitly defined
        if not organ_type:
            names = {p["name"] for p in input_data.params}
            if "stele" in names or "cortex" in names:
                organ_type = "root"
            else:
                organ_type = "needle"

        if organ_type == "needle":
            from granap.needle_class import NeedleAnatomy
            return NeedleAnatomy(input_data)
        else:
            from granap.root_class import RootAnatomy
            return RootAnatomy(input_data)
    
    def add_layer(self, layer: Layer, position: Optional[int] = None) -> None:
        """
        Add a tissue layer to the anatomy.
        
        Args:
            layer: Layer object to add
            position: Optional position index (None = append)
        """
        self.layer_manager.add_layer(layer, position)
        self._invalidate_geometry()
    
    def update_params(self, param_name: str, attribute: str, value: Any) -> None:
        """
        Update a parameter of the organ.
    
        self.params = [{"name": "param_name_1", "attribute_1": 0.0, "attribute_2": 0.0, ...},
                       {"name": "param_name_2", "attribute_1": 0.0, "attribute_2": 0.0, ...},
                       ...]
    
        Args:
            param_name: Name of the parameter to update
            attribute: Name of the attribute to update
            value: New value of the parameter
        """
        for p in self.params:
            if p["name"] == param_name:
                p[attribute] = value
                self._invalidate_geometry()
                return
        raise ValueError(f"Parameter '{param_name}' not found in params.")

    
    def remove_layer(self, name: str) -> Layer:
        """
        Remove a tissue layer by name.
        
        Args:
            name: Name identifier of the layer
        
        Returns:
            The removed Layer object    
        """
        removed = self.layer_manager.remove_layer(name)
        self._invalidate_geometry()
        return removed
    
    def get_layer(self, name: str) -> Optional[Layer]:
        """Get a layer by name."""
        return self.layer_manager.get_layer(name)
    
    def list_layers(self) -> List[str]:
        """List all layer names."""
        return [layer.name for layer in self.layer_manager.get_layers()]
    
    def _invalidate_geometry(self) -> None:
        """Invalidate cached geometry after layer changes."""
        self._base_polygon = None
        self._layers_polygons = []
        self._cells_gdf = None
    
    def generate_base_shape(self) -> Polygon:
        """
        Generate or retrieve the base shape.
        
        Returns:
            Base polygon
        """
        if self._base_polygon is None:
            self._base_polygon = self._create_base_shape()
        return self._base_polygon
    
    def generate_layer_polygons(self) -> List[Dict[str, Any]]:
        """
        Generate polygons for all layers.
        
        Returns:
            List of layer polygon dictionaries
        """
        if not self._layers_polygons:
            self._layers_polygons = self._build_layer_polygons()
        return self._layers_polygons
    
    def _build_layer_polygons(self) -> List[Dict[str, Any]]:
        """Build layer polygons from current layer configuration."""
        layers_polygons = []
        layer_array = self.layer_manager.expand_layers()
        
        polygon = self.generate_base_shape()
        
        for i_layer, layer in enumerate(layer_array):
            if i_layer == 0:
                # Add outside layer
                space_increment = layer["cell_diameter"] / 2
                polygon = GeometryProcessor.buffer_polygon(
                    polygon, space_increment, smooth_factor=0.01
                )
                layers_polygons.append({
                    "name": "outside",
                    "polygon": polygon,
                    "cell_diameter": layer["cell_diameter"] / 3,
                    "id_layer": i_layer,
                    "cell_width": 0
                })
            
            # Add the layer polygon
            polygon = GeometryProcessor.buffer_polygon(
                polygon, 
                -space_increment - layer["cell_diameter"]/2,
                smooth_factor=0.5
            )
            
            space_increment = layer["cell_diameter"] / 2
            
            layers_polygons.append({
                "name": layer["name"],
                "polygon": polygon,
                "cell_diameter": layer["cell_diameter"],
                "id_layer": i_layer + 1,
                "cell_width": layer["cell_width"],
                "shift": layer["shift"]
            })
        
        # Add central layers (vascular, parenchyma, etc.)
        params = [l.to_dict() for l in self.layer_manager.get_layers()]
        central_layers = self._create_central_layers(polygon, params)
        layers_polygons.extend(central_layers)

        # Optional reshape: let subclasses morph layer polygons
        layers_polygons = self.reshape_layers(layers_polygons)
        
        return layers_polygons
    
    def generate_cells(self) -> gpd.GeoDataFrame:
        """
        Generate cell geometries using Voronoi tessellation.
        
        Returns:
            GeoDataFrame with cell geometries
        """
        if self._cells_gdf is None:
            layers_polygons = self.generate_layer_polygons()
            center = layers_polygons[0]["polygon"].centroid
            
            # Clear existing cells in layers
            for layer in self.layer_manager.get_layers():
                layer.cells = []
            
            self.all_cells = CellGenerator.generate_cells_info(
                layers_polygons, center
            )

            # add vascular tissue
            self.allocate_vascular_tissue(layers_polygons)

            # add organ specific tissues
            self._organ_specific_tissues()

            vor = CellGenerator.voronoi_diagram(self.all_cells)
            
            grouped_cells = CellGenerator.process_voronoi_groups(self.all_cells, vor).cells
            grouped_cells = CellGenerator.simplify_cells(grouped_cells)
            # repopulate all_cells with the grouped cells
            self.all_cells = CellManager()
            self.all_cells.cells = grouped_cells
            self.add_intercellular_spaces()
            
            for cell in self.all_cells.cells:
                # Find the layer name from layers_polygons using id_layer
                # id_layer is 0-indexed index of layers_polygons list
                if 0 <= cell.id_layer < len(layers_polygons):
                    layer_name = layers_polygons[cell.id_layer]["name"]
                    if layer_name != "outside":
                        layer = self.get_layer(layer_name)
                        if layer:
                            layer.cells.append(cell)

            
            # Convert to GeoDataFrame
            cell_dicts = [c.cell_to_dict() for c in self.all_cells.cells]
            for i, c in enumerate(self.all_cells.cells):
                cell_dicts[i]['geometry'] = c.polygon
                
            self._cells_gdf = gpd.GeoDataFrame(cell_dicts)
        
        return self._cells_gdf
    
    @abstractmethod
    def reshape_layers(self, layers_polygons: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optionally reshape layer polygons after they have been built.

        The default implementation is a no-op (returns the list unchanged).
        Subclasses can override this to morph each layer's polygon — for
        example, interpolating between the outer organ shape and an inner
        ellipse so that the central cylinder has a different cross-section.

        Args:
            layers_polygons: List of layer polygon dictionaries as produced
                by ``_build_layer_polygons``.

        Returns:
            The (potentially modified) list of layer polygon dictionaries.
        """
        return layers_polygons

    def allocate_vascular_tissue(self, layers_polygons: List[Dict[str, Any]]):
        """
        Allocate vascular tissue.
        Define the region where vascular tissue will be allocated.
        
        Args:
            layers_polygons: List of layer polygon dictionaries
        """
        # Find the layer where vascular tissue will be allocated
        polygon_for_vascular = self._which_layer_for_vascular(layers_polygons)
        # Create vascular tissue
        self._create_vascular_tissue(polygon_for_vascular)

    @abstractmethod
    def _which_layer_for_vascular(self, layers_polygons: List[Dict[str, Any]]):
        """
        Find the layer where vascular tissue will be allocated.
        
        Args:
            layers_polygons: List of layer polygon dictionaries
        """
        pass

    @abstractmethod
    def _create_vascular_tissue(self, polygon: Polygon):
        """
        Create vascular tissue.
        
        Args:
            polygon: Polygon boundary
        """
        pass

    @abstractmethod
    def _organ_specific_tissues(self):
        """
        Add organ specific tissues.
        
        Returns:
        """
        pass

    @abstractmethod
    def add_intercellular_spaces(self):
        """
        Compute and return intercellular (air space) polygons.

        Returns
        -------
        CellManager
            CellManager object with air space cells.
            Return an empty CellManager when there are no air spaces.
        """
        pass
        
    
    def plot_layers(self, show: bool = True, **kwargs) -> Optional[plt.Figure]:
        """
        Plot layer boundaries.
        
        Args:
            show: Whether to display the plot
        
        Returns:
            Matplotlib figure
        """
        
        layers_polygons = self.generate_layer_polygons()
        
        ax = kwargs.get('ax')
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        colors = plt.cm.viridis(np.linspace(0, 1, len(layers_polygons)))
        
        for polygon_data, color in zip(layers_polygons, colors):
            ax.plot(*polygon_data["polygon"].exterior.xy, 
                   color=color, label=polygon_data["name"])
        
        ax.set_aspect('equal')
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title(kwargs.get('title', f"{self.__class__.__name__} - Layer Boundaries"))
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        if fig is not None:
            plt.tight_layout()
            if show:
                plt.show()
            return fig
        return None

    
    def plot_cells(self, show: bool = True, **kwargs) -> Optional[plt.Figure]:
        """
        Plot cell geometries.
        
        Args:
            show: Whether to display the plot
        
        Returns:
            Matplotlib figure
        """
        cells_gdf = self.generate_cells()
        
        ax = kwargs.get('ax')
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        cells_gdf.plot(
            ax=ax,
            column='type',
            cmap='viridis',
            edgecolor='black',
            linewidth=0.5,
            alpha=0.5,
            legend=True,
            legend_kwds={'title': 'Cell Type', 'loc': 'best'}
        )
        
        ax.set_aspect("equal", "box")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title(kwargs.get('title', f"{self.__class__.__name__} - Cross Section"))
        
        if fig is not None:
            plt.tight_layout()
            if show:
                plt.show()
            return fig
        return None
    
    def export_to_geopandas(self) -> gpd.GeoDataFrame:
        """
        Export cell geometries as GeoDataFrame.
        
        Returns:
            GeoDataFrame with cell data
        """
        return self.generate_cells()
    
    def export_to_csv(self, filepath: str) -> None:
        """
        Export cell data to CSV file.
        
        Args:
            filepath: Output file path
        """
        cells_gdf = self.generate_cells()
        # Drop geometry column for CSV export
        cells_df = cells_gdf.drop(columns=['geometry'])
        cells_df.to_csv(filepath, index=False)

    def write_to_xml(self, path: str, **kwargs):
        """Write anatomy cross section as .xml file."""
        from granap.anatomy_writer import AnatomyWriter
        AnatomyWriter(self).write_to_xml(path, **kwargs)
        
    def write_to_obj(self, path: str, **kwargs):
        """Write anatomy cross section as .obj file."""
        from granap.anatomy_writer import AnatomyWriter
        AnatomyWriter(self).write_to_obj(path, **kwargs)

    def write_to_svg(self, path: str, **kwargs):
        """Write anatomy cross section as .svg file."""
        from granap.anatomy_writer import AnatomyWriter
        AnatomyWriter(self).write_to_svg(path, **kwargs)
        
    def write_to_geo(self, path: str, **kwargs):
        """Write anatomy cross section as .geo file for GMSH."""
        from granap.anatomy_writer import AnatomyWriter
        AnatomyWriter(self).write_to_geo(path, **kwargs)


    def export_to_adjencymatrix(self) -> lil_matrix:
        """
        Build the hydraulic network from cell geometry and return
        the sparse adjacency matrix.

        Returns
        -------
        lil_matrix
            Sparse adjacency matrix (n_total x n_total).
        """
        # Ensure cells are generated before building the network
        self.generate_cells()
        return super().export_to_adjencymatrix()

    # ------------------------------------------------------------------
    # Network construction from Voronoi cell geometry
    # ------------------------------------------------------------------
    def _build_anatnetwork(self) -> None:
        """
        Populate ``self.graph`` from the cell GeoDataFrame.
        Delegated to AnatomyWriter's NetworkExporter.
        """
        from granap.anatomy_writer import NetworkExporter
        NetworkExporter(self).export(self)

    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Calculate anatomical statistics.
        
        Returns:
            Dictionary with statistics
        """
        cells_gdf = self.generate_cells()
        
        stats = {
            "total_cells": len(cells_gdf),
            "cell_types": cells_gdf['type'].unique().tolist(),
            "cells_per_type": cells_gdf['type'].value_counts().to_dict(),
            "total_area": cells_gdf.geometry.area.sum(),
            "mean_cell_area": cells_gdf['area'].mean(),
            "n_layers": len(self.layer_manager)
        }
        
        return stats
    
    @abstractmethod
    def _create_base_shape(self) -> Polygon:
        """
        Create the base shape for the organ.
        
        This method must be implemented by subclasses to define
        the characteristic shape of each organ type.
        
        Returns:
            Base polygon shape
        """
        pass
    
    @abstractmethod
    def _create_central_layers(self, current_polygon: Polygon,
                               params: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create central tissue layers (vascular, parenchyma, etc.).
        
        This method must be implemented by subclasses to define
        organ-specific central structures.
        
        Args:
            current_polygon: Current inner polygon boundary
            params: Parameter dictionaries
        
        Returns:
            List of central layer polygon dictionaries
        """
        pass
