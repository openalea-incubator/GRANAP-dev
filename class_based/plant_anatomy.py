"""
Plant anatomy base module providing abstract interface.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point

from layer import Layer
from layer_manager import LayerManager
from geometry_processor import GeometryProcessor
from cell_generator import CellGenerator


class PlantAnatomy(ABC):
    """
    Abstract base class for plant anatomical structures.
    
    Defines the interface and common functionality for generating
    cross-sectional anatomy of different plant types.
    """
    
    def __init__(self, randomness: float = 1.0):
        """
        Initialize the anatomy structure.
        
        Args:
            randomness: Degree of randomness in cell placement (0-3)
        """
        self.layer_manager = LayerManager()
        self.randomness = randomness
        self._base_polygon: Optional[Polygon] = None
        self._layers_polygons: List[Dict[str, Any]] = []
        self._cells_gdf: Optional[gpd.GeoDataFrame] = None
    
    def add_layer(self, layer: Layer, position: Optional[int] = None) -> None:
        """
        Add a tissue layer to the anatomy.
        
        Args:
            layer: Layer object to add
            position: Optional position index (None = append)
        """
        self.layer_manager.add_layer(layer, position)
        self._invalidate_geometry()
    
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
                -space_increment / 2 - layer["cell_diameter"] / 4,
                smooth_factor=0.5
            )
            space_increment = layer["cell_diameter"] / 2
            
            cell_width = layer["cell_width"]
            param_match = self.layer_manager.get_layer(layer["name"])
            if param_match and param_match.cell_width:
                cell_width = param_match.cell_width / 4
            
            layers_polygons.append({
                "name": layer["name"],
                "polygon": polygon,
                "cell_diameter": layer["cell_diameter"],
                "id_layer": i_layer + 1,
                "cell_width": cell_width
            })
        
        # Add central layers (vascular, parenchyma, etc.)
        params = [l.to_dict() for l in self.layer_manager.get_layers()]
        central_layers = self._create_central_layers(polygon, params)
        layers_polygons.extend(central_layers)
        
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
            
            all_cells, vor = CellGenerator.generate_cells_info(
                layers_polygons, center
            )
            
            grouped_cells = CellGenerator.process_voronoi_groups(all_cells, vor)
            grouped_cells = CellGenerator.smooth_cells(grouped_cells)
            
            self._cells_gdf = grouped_cells
        
        return self._cells_gdf
    
    def plot_layers(self, show: bool = True) -> plt.Figure:
        """
        Plot layer boundaries.
        
        Args:
            show: Whether to display the plot
        
        Returns:
            Matplotlib figure
        """
        plt.close('all')
        layers_polygons = self.generate_layer_polygons()
        
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, len(layers_polygons)))
        
        for polygon_data, color in zip(layers_polygons, colors):
            ax.plot(*polygon_data["polygon"].exterior.xy, 
                   color=color, label=polygon_data["name"])
        
        ax.set_aspect('equal')
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title(f"{self.__class__.__name__} - Layer Boundaries")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        if show:
            plt.show()
        
        return fig
    
    def plot_cells(self, show: bool = True) -> plt.Figure:
        """
        Plot cell geometries.
        
        Args:
            show: Whether to display the plot
        
        Returns:
            Matplotlib figure
        """
        cells_gdf = self.generate_cells()
        
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
        ax.set_title(f"{self.__class__.__name__} - Cross Section")
        plt.tight_layout()
        
        if show:
            plt.show()
        
        return fig
    
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


import numpy as np  # Import here to avoid circular imports
