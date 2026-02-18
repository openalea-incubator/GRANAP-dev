"""
Plant anatomy base module providing abstract interface.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
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
            
            vor = CellGenerator.voronoi_diagram(self.all_cells)
            
            grouped_cells = CellGenerator.process_voronoi_groups(self.all_cells, vor).cells
            grouped_cells = CellGenerator.smooth_cells(grouped_cells)
            
            # Populate layers with cells
            # Map layer index to layer specific object
            # Note: layers_polygons indices match the order of generation, 
            # but we need to match them to self.layer_manager layers.
            # The indices in generate_layer_polygons are:
            # 0: outside
            # 1..N: actual layers
            # N+1..M: central layers
            
            for cell in grouped_cells:
                # Find the layer name from layers_polygons using id_layer
                # id_layer is 0-indexed index of layers_polygons list
                if 0 <= cell.id_layer < len(layers_polygons):
                    layer_name = layers_polygons[cell.id_layer]["name"]
                    if layer_name != "outside":
                        layer = self.get_layer(layer_name)
                        if layer:
                            layer.cells.append(cell)
            
            # Convert to GeoDataFrame
            cell_dicts = [c.cell_to_dict() for c in grouped_cells]
            for i, c in enumerate(grouped_cells):
                cell_dicts[i]['geometry'] = c.polygon
                
            self._cells_gdf = gpd.GeoDataFrame(cell_dicts)
        
        return self._cells_gdf

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
    def _build_network(self) -> None:
        """
        Populate ``self.graph`` from the cell GeoDataFrame.

        Algorithm
        ---------
        1. Extract polygon vertices (rounded) → Junction nodes.
        2. Extract polygon edges (sorted vertex pairs) → Wall nodes
           at edge midpoints.  Track which cells border each wall.
        3. Assign MECHA-compatible indices:
           ``[0..N_w)`` walls, ``[N_w..N_w+N_j)`` junctions,
           ``[N_w+N_j..N_total)`` cells.
        4. Add edges:
           - **transmembrane** : cell ↔ wall
           - **symplastic**    : cell ↔ cell  (shared wall)
           - **apoplastic**    : wall ↔ junction
        """
        cells_gdf = self.generate_cells()
        n_dec = 6  # rounding precision for vertex deduplication

        # ----- Step 1 & 2: collect vertices and edges per cell -----
        # vertex_key  → junction local id
        vertex_map: Dict[tuple, int] = {}
        next_vertex_id = 0

        # edge_key (sorted pair of vertex keys) → wall local id
        edge_map: Dict[tuple, int] = {}
        next_edge_id = 0

        # wall_id → list of cell_row_index that touch this wall
        wall_to_cell_rows: Dict[int, List[int]] = {}

        # wall_id → (vertex_key_a, vertex_key_b)
        wall_vertex_keys: Dict[int, tuple] = {}

        # wall_id → midpoint (x, y)
        wall_midpoints: Dict[int, tuple] = {}

        for row_idx, row in cells_gdf.iterrows():
            poly = row["geometry"]
            if poly is None or poly.is_empty:
                continue

            # Handle MultiPolygon – use only the largest piece
            if isinstance(poly, MultiPolygon):
                poly = max(poly.geoms, key=lambda g: g.area)

            coords = list(poly.exterior.coords)
            if coords[0] == coords[-1]:
                coords = coords[:-1]
            if len(coords) < 3:
                continue

            n_pts = len(coords)
            # Build rounded vertex keys for this polygon
            vkeys = []
            for x, y in coords:
                vk = (round(x, n_dec), round(y, n_dec))
                if vk not in vertex_map:
                    vertex_map[vk] = next_vertex_id
                    next_vertex_id += 1
                vkeys.append(vk)

            # Build edges (consecutive vertex pairs)
            for i in range(n_pts):
                vk_a = vkeys[i]
                vk_b = vkeys[(i + 1) % n_pts]
                edge_key = tuple(sorted((vk_a, vk_b)))

                if edge_key not in edge_map:
                    edge_map[edge_key] = next_edge_id
                    wall_vertex_keys[next_edge_id] = (vk_a, vk_b)
                    mid_x = (vk_a[0] + vk_b[0]) / 2.0
                    mid_y = (vk_a[1] + vk_b[1]) / 2.0
                    wall_midpoints[next_edge_id] = (mid_x, mid_y)
                    wall_to_cell_rows[next_edge_id] = []
                    next_edge_id += 1

                wall_id = edge_map[edge_key]
                if row_idx not in wall_to_cell_rows[wall_id]:
                    wall_to_cell_rows[wall_id].append(row_idx)

        # ----- Step 3: assign MECHA-compatible node indices -----
        self.n_walls = len(edge_map)
        self.n_junctions = len(vertex_map)
        self.n_cells = len(cells_gdf)

        # Wall node index  = wall_local_id  (already 0-based)
        # Junction node idx = n_walls + junction_local_id
        # Cell node idx     = n_walls + n_junctions + cell_row_index

        junction_key_to_node = {
            vk: self.n_walls + vid for vk, vid in vertex_map.items()
        }

        cell_row_to_node = {
            row_idx: self.n_walls + self.n_junctions + i
            for i, row_idx in enumerate(cells_gdf.index)
        }

        # ----- Add wall nodes -----
        for wall_id, midpoint in wall_midpoints.items():
            # Compute wall length from its two vertices
            vk_a, vk_b = wall_vertex_keys[wall_id]
            length = np.hypot(vk_a[0] - vk_b[0], vk_a[1] - vk_b[1])
            self.graph.add_node(
                wall_id,
                node_type="wall",
                position=midpoint,
                length=length,
            )

        # ----- Add junction nodes -----
        for vk, vid in vertex_map.items():
            node_id = self.n_walls + vid
            self.graph.add_node(
                node_id,
                node_type="junction",
                position=vk,
            )

        # ----- Add cell nodes -----
        for row_idx, row in cells_gdf.iterrows():
            node_id = cell_row_to_node[row_idx]
            centroid = row["geometry"].centroid if row["geometry"] is not None else None
            cx = centroid.x if centroid else row["x"]
            cy = centroid.y if centroid else row["y"]
            self.graph.add_node(
                node_id,
                node_type="cell",
                cell_type=row.get("type", ""),
                position=(cx, cy),
            )

        # ----- Step 4: add edges -----
        # Store wall→cell mapping for fill_matrix filtering
        self._wall_to_cells = {
            wid: [cell_row_to_node[r] for r in rows]
            for wid, rows in wall_to_cell_rows.items()
        }

        for wall_id in range(self.n_walls):
            vk_a, vk_b = wall_vertex_keys[wall_id]
            junc_a = junction_key_to_node[vk_a]
            junc_b = junction_key_to_node[vk_b]
            cell_nodes = self._wall_to_cells[wall_id]
            wall_length = self.graph.nodes[wall_id]["length"]

            # Apoplastic: wall ↔ junction
            self.graph.add_edge(
                wall_id, junc_a,
                path="apoplastic",
                length=wall_length / 2.0,
            )
            self.graph.add_edge(
                wall_id, junc_b,
                path="apoplastic",
                length=wall_length / 2.0,
            )

            # Transmembrane: cell ↔ wall
            for cn in cell_nodes:
                pos_cell = self.graph.nodes[cn]["position"]
                pos_wall = self.graph.nodes[wall_id]["position"]
                dist = np.hypot(
                    pos_wall[0] - pos_cell[0],
                    pos_wall[1] - pos_cell[1],
                )
                self.graph.add_edge(
                    cn, wall_id,
                    path="transmembrane",
                    length=wall_length,
                    dist=dist,
                )

            # Symplastic: cell ↔ cell (only if wall is shared)
            if len(cell_nodes) == 2:
                pos_a = self.graph.nodes[cell_nodes[0]]["position"]
                pos_b = self.graph.nodes[cell_nodes[1]]["position"]
                dist = np.hypot(
                    pos_b[0] - pos_a[0], pos_b[1] - pos_a[1]
                )
                # add_edge is idempotent for the same pair; if two
                # walls are shared between the same two cells the
                # edge is simply updated (latest wall data wins,
                # which is acceptable for the connectivity matrix).
                self.graph.add_edge(
                    cell_nodes[0], cell_nodes[1],
                    path="symplastic",
                    length=wall_length,
                    dist=dist,
                )
    
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
