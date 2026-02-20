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
    def _build_anatnetwork(self) -> None:
        """
        Populate ``self.graph`` from the cell GeoDataFrame.

        Algorithm
        ---------
        1. Extract polygon vertices and edges; track which cells own
           each edge.
        2. Identify **junction vertices** — points where the set of
           adjacent cells changes (triple junctions in a Voronoi).
        3. Walk each cell boundary between consecutive junctions to
           define **walls** (one wall per cell-pair interface).
        4. Assign MECHA-compatible node indices and build the graph.
        """
        cells_gdf = self.generate_cells()
        n_dec = 6  # rounding precision for snapped vertex keys

        # Phase 0 — snap nearby vertices together using a KD-tree
        # This fixes floating-point mismatches between adjacent polygons
        # that would otherwise prevent shared-vertex detection.
        from scipy.spatial import cKDTree

        raw_cell_data: Dict[int, list] = {}   # row_idx → raw coords
        all_raw_verts: list = []              # flat list of (x, y)
        vert_global_idx: Dict[int, List[int]] = {}  # row_idx → [indices]

        for row_idx, row in cells_gdf.iterrows():
            poly = row["geometry"]
            if poly is None or poly.is_empty:
                continue
            if isinstance(poly, MultiPolygon):
                poly = max(poly.geoms, key=lambda g: g.area)
            coords = list(poly.exterior.coords)
            if coords[0] == coords[-1]:
                coords = coords[:-1]
            if len(coords) < 3:
                continue
            indices = []
            for x, y in coords:
                indices.append(len(all_raw_verts))
                all_raw_verts.append((x, y))
            raw_cell_data[row_idx] = coords
            vert_global_idx[row_idx] = indices

        if not all_raw_verts:
            return

        coords_arr = np.array(all_raw_verts)
        kd_tree = cKDTree(coords_arr)

        # Compute snap tolerance: 1 % of 5th-percentile edge length
        edge_lengths = []
        for coords in raw_cell_data.values():
            n = len(coords)
            for k in range(n):
                el = np.hypot(
                    coords[(k + 1) % n][0] - coords[k][0],
                    coords[(k + 1) % n][1] - coords[k][1],
                )
                if el > 0:
                    edge_lengths.append(el)
        snap_tol = (
            np.percentile(edge_lengths, 5) * 0.01
            if edge_lengths
            else 1e-4
        )

        # Cluster nearby vertices → canonical snapped coordinate
        canonical = [None] * len(all_raw_verts)
        visited_snap = [False] * len(all_raw_verts)
        for i in range(len(all_raw_verts)):
            if visited_snap[i]:
                continue
            cluster = kd_tree.query_ball_point(coords_arr[i], snap_tol)
            cx = float(np.mean(coords_arr[cluster, 0]))
            cy = float(np.mean(coords_arr[cluster, 1]))
            snapped = (round(cx, n_dec), round(cy, n_dec))
            for ci in cluster:
                visited_snap[ci] = True
                canonical[ci] = snapped

        # Phase 1 — build cell_vkeys, vertex_to_cells, edge_to_cells
        vertex_to_cells: Dict[tuple, set] = {}
        cell_vkeys: Dict[int, List[tuple]] = {}

        for row_idx, gidxs in vert_global_idx.items():
            vkeys_raw = [canonical[gi] for gi in gidxs]
            # Remove consecutive duplicates introduced by snapping
            vkeys: List[tuple] = [vkeys_raw[0]]
            for vk in vkeys_raw[1:]:
                if vk != vkeys[-1]:
                    vkeys.append(vk)
            if len(vkeys) > 1 and vkeys[-1] == vkeys[0]:
                vkeys = vkeys[:-1]
            if len(vkeys) < 3:
                continue
            cell_vkeys[row_idx] = vkeys
            for vk in vkeys:
                vertex_to_cells.setdefault(vk, set()).add(row_idx)

        # Build edge → set of cells  (an "edge" = one polygon side)
        edge_to_cells: Dict[tuple, set] = {}
        for row_idx, vkeys in cell_vkeys.items():
            n = len(vkeys)
            for i in range(n):
                ek = tuple(sorted((vkeys[i], vkeys[(i + 1) % n])))
                edge_to_cells.setdefault(ek, set()).add(row_idx)

        # Phase 2 — identify junction vertices
        # A vertex is a junction if its incident edges belong to
        # *different* sets of cells (= the boundary topology changes).
        junction_set: set = set()

        for vk in vertex_to_cells:
            # Fast path: vertex shared by ≥3 cells is always a junction
            if len(vertex_to_cells[vk]) >= 3:
                junction_set.add(vk)
                continue
            # Check incident-edge cell-pair signatures
            incident_pairs: set = set()
            for row_idx in vertex_to_cells[vk]:
                vks = cell_vkeys[row_idx]
                n = len(vks)
                for i in range(n):
                    if vks[i] != vk:
                        continue
                    ek_prev = tuple(sorted((vks[(i - 1) % n], vk)))
                    ek_next = tuple(sorted((vk, vks[(i + 1) % n])))
                    if ek_prev in edge_to_cells:
                        incident_pairs.add(frozenset(edge_to_cells[ek_prev]))
                    if ek_next in edge_to_cells:
                        incident_pairs.add(frozenset(edge_to_cells[ek_next]))
            if len(incident_pairs) > 1:
                junction_set.add(vk)

        # Phase 3 — walk cell boundaries to define walls
        # A "wall" = the polyline segment between two consecutive
        # junction vertices along one cell boundary.  Two cells that
        # share the same (juncA, juncB) segment share a wall.
        wall_registry: Dict[tuple, dict] = {}  # wall_key → wall info
        next_wall_id = 0

        for row_idx, vkeys in cell_vkeys.items():
            n = len(vkeys)
            junc_positions = [i for i in range(n) if vkeys[i] in junction_set]

            if len(junc_positions) < 2:
                # Fewer than 2 junctions → treat entire boundary as one wall
                wall_key = tuple(sorted(vkeys))
                if wall_key not in wall_registry:
                    length = sum(
                        np.hypot(vkeys[(k+1) % n][0] - vkeys[k][0],
                                 vkeys[(k+1) % n][1] - vkeys[k][1])
                        for k in range(n)
                    )
                    mid_x = np.mean([v[0] for v in vkeys])
                    mid_y = np.mean([v[1] for v in vkeys])
                    wall_registry[wall_key] = {
                        "id": next_wall_id,
                        "junc_start": vkeys[0],
                        "junc_end": vkeys[0],
                        "midpoint": (mid_x, mid_y),
                        "length": length,
                        "cells": [],
                    }
                    next_wall_id += 1
                if row_idx not in wall_registry[wall_key]["cells"]:
                    wall_registry[wall_key]["cells"].append(row_idx)
                continue

            for jp in range(len(junc_positions)):
                start_idx = junc_positions[jp]
                end_idx = junc_positions[(jp + 1) % len(junc_positions)]

                # Collect vertices along the segment
                segment: List[tuple] = []
                i = start_idx
                while True:
                    segment.append(vkeys[i])
                    if i == end_idx:
                        break
                    i = (i + 1) % n

                if len(segment) < 2:
                    continue

                junc_start = segment[0]
                junc_end = segment[-1]
                wall_key = tuple(sorted((junc_start, junc_end)))

                if wall_key not in wall_registry:
                    length = sum(
                        np.hypot(segment[k+1][0] - segment[k][0],
                                 segment[k+1][1] - segment[k][1])
                        for k in range(len(segment) - 1)
                    )
                    mid_x = np.mean([v[0] for v in segment])
                    mid_y = np.mean([v[1] for v in segment])
                    wall_registry[wall_key] = {
                        "id": next_wall_id,
                        "junc_start": junc_start,
                        "junc_end": junc_end,
                        "midpoint": (mid_x, mid_y),
                        "length": length,
                        "cells": [],
                    }
                    next_wall_id += 1

                if row_idx not in wall_registry[wall_key]["cells"]:
                    wall_registry[wall_key]["cells"].append(row_idx)

        # Phase 4 — assign MECHA-compatible node indices
        self.n_walls = len(wall_registry)

        # Only keep junctions actually referenced by walls
        used_junctions: set = set()
        for wd in wall_registry.values():
            used_junctions.add(wd["junc_start"])
            used_junctions.add(wd["junc_end"])
        junction_list = sorted(used_junctions)
        junction_vk_to_id = {vk: i for i, vk in enumerate(junction_list)}

        self.n_junctions = len(junction_list)
        self.n_cells = len(cells_gdf)

        cell_row_to_node = {
            row_idx: self.n_walls + self.n_junctions + i
            for i, row_idx in enumerate(cells_gdf.index)
        }

        # Phase 5 — add nodes to graph
        # Wall nodes
        for wd in wall_registry.values():
            self.graph.add_node(
                wd["id"],
                indice=wd["id"],
                type="apo",
                position=wd["midpoint"],
                length=wd["length"],
            )

        # Junction nodes
        for vk in junction_list:
            node_id = self.n_walls + junction_vk_to_id[vk]
            self.graph.add_node(
                node_id,
                indice=node_id,
                type="apo",
                position=vk,
                length=0,
            )

        # Cell nodes
        for row_idx, row in cells_gdf.iterrows():
            node_id = cell_row_to_node[row_idx]
            centroid = row["geometry"].centroid if row["geometry"] is not None else None
            area = row["geometry"].area if row["geometry"] is not None else None
            cx = centroid.x if centroid else row["x"]
            cy = centroid.y if centroid else row["y"]
            self.graph.add_node(
                node_id,
                indice=node_id,
                type="cell",
                cgroup=row.get("cgroup", ""),
                cell_type=row.get("type", ""),
                position=(cx, cy),
                area=area,
            )

        # Phase 6 — add edges
        self._wall_to_cells = {
            wd["id"]: [cell_row_to_node[r] for r in wd["cells"]]
            for wd in wall_registry.values()
        }

        for wd in wall_registry.values():
            wall_id = wd["id"]
            cell_nodes = self._wall_to_cells[wall_id]
            wall_length = wd["length"]

            # Transmembrane: cell ↔ wall
            for cn in cell_nodes:
                pos_cell = self.graph.nodes[cn]["position"]
                pos_wall = wd["midpoint"]
                dist_wall_cell = np.hypot(
                    pos_wall[0] - pos_cell[0],
                    pos_wall[1] - pos_cell[1],
                )
                d_vec = np.array([pos_wall[0] - pos_cell[0], pos_wall[1] - pos_cell[1]])
                self.graph.add_edge(
                    cn, wall_id,
                    path="membrane",
                    length=wall_length,
                    dist=dist_wall_cell,
                    d_vec=d_vec,
                )
            
            # each junction connected to the wall node
            for junc in ["junc_start", "junc_end"]:
                junc_id = self.n_walls + junction_vk_to_id[wd[junc]]
                pos_junc = self.graph.nodes[junc_id]["position"]
                dist_junc_wall_node = np.hypot(pos_junc[0] - pos_wall[0], pos_junc[1] - pos_wall[1])
                lateral_distance = dist_wall_cell + dist_junc_wall_node
                d_vec = np.array(pos_junc[0] - pos_wall[0], pos_junc[1] - pos_wall[1])
                
                # Apoplastic: wall ↔ junction
                self.graph.add_edge(
                        junc_id,
                        wall_id,
                        path = 'wall',
                        length = wall_length / 2.0,
                        lateral_distance = lateral_distance,
                        d_vec = d_vec,
                        distnode_wall_cell = dist_wall_cell,
                )
            
            # Symplastic: cell ↔ cell
            if len(cell_nodes) == 2:
                pos_a = self.graph.nodes[cell_nodes[0]]["position"]
                pos_b = self.graph.nodes[cell_nodes[1]]["position"]
                dist = np.hypot(
                    pos_b[0] - pos_a[0], pos_b[1] - pos_a[1]
                )
                d_vec = np.array([pos_b[0] - pos_a[0], pos_b[1] - pos_a[1]])
                self.graph.add_edge(
                    cell_nodes[0], cell_nodes[1],
                    path="plasmodesmata",
                    length=wall_length,
                    dist=dist,
                    d_vec=d_vec,
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
