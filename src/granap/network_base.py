"""
Abstract network base module for hydraulic network construction.

Provides the AbstractNetwork interface that can be used by both
GRANAP (from cell geometry) and future refactoring of MECHA's NetworkBuilder.

Node indexing follows MECHA convention:
    0 .. N_w-1           : Wall nodes (midpoints of shared/boundary edges)
    N_w .. N_w+N_j-1     : Junction nodes (triple-junction vertices)
    N_w+N_j .. N_total-1 : Cell nodes (centroids)
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import networkx as nx
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt


class AbstractNetwork(ABC):
    """
    Abstract base class for hydraulic network graph construction.

    Subclasses must implement ``_build_network`` which populates
    ``self.graph`` with nodes and edges.

    After the graph is built, ``export_to_adjencymatrix`` converts it to
    a sparse adjacency/conductance matrix and ``fill_matrix`` populates
    the matrix entries with hydraulic conductivities.
    """

    def __init__(self):
        # NetworkX graph holding the topology
        self.graph: nx.Graph = nx.Graph()

        # Counts (set by _build_network)
        self.n_walls: int = 0
        self.n_junctions: int = 0
        self.n_cells: int = 0

        # Sparse matrix (set by export_to_adjencymatrix)
        self._matrix: Optional[lil_matrix] = None

        # Mapping:  wall_id -> set of cell indices that border the wall
        self._wall_to_cells: Dict[int, List[int]] = {}

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------
    @abstractmethod
    def _build_anatnetwork(self) -> None:
        """
        Populate ``self.graph`` with wall, junction and cell nodes,
        and connect them with the appropriate edges.

        Must set ``self.n_walls``, ``self.n_junctions``, ``self.n_cells``
        and ``self._wall_to_cells``.
        """
        ...

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def n_total(self) -> int:
        return self.n_walls + self.n_junctions + self.n_cells

    def export_to_adjencymatrix(self) -> lil_matrix:
        """
        Build the network (if needed) and return the sparse adjacency matrix.

        Returns
        -------
        lil_matrix
            Sparse matrix of size ``(n_total, n_total)``.
            Non-zero structure encodes the connectivity; values are
            initially set to 1.0 (call ``fill_matrix`` to assign
            hydraulic conductivities).
        """
        if self._matrix is not None:
            return self._matrix

        # Ensure the graph is built
        if self.graph.number_of_nodes() == 0:
            self._build_anatnetwork()

        n = self.n_total
        mat = lil_matrix((n, n))

        # Populate with 1s for every edge (symmetric)
        for u, v, data in self.graph.edges(data=True):
            mat[u, v] = 1.0
            mat[v, u] = 1.0

        self._matrix = mat
        return self._matrix

    def fill_matrix(
        self,
        K: float,
        label: str,
        cell_type: str,
    ) -> None:
        """
        Fill entries of the adjacency matrix with a hydraulic conductivity.

        Parameters
        ----------
        K : float
            Hydraulic conductivity value to assign.
        label : str
            Path type to target.  One of:
                * ``"apoplastic"``   – wall-node ↔ junction edges
                * ``"transmembrane"`` – cell ↔ wall-node edges
                * ``"symplastic"``   – cell ↔ cell edges
        cell_type : str
            Tissue type filter.  Rules:

            * A single name (e.g. ``"cortex"``) matches edges where
              **both** adjacent cells are of that type.
            * A compound name separated by ``"-"`` (e.g.
              ``"cortex-endodermis"``) matches edges where one adjacent
              cell is of the first type and the other of the second type.

        Raises
        ------
        ValueError
            If the matrix has not been created yet, or the label is
            unknown.
        """
        if self._matrix is None:
            raise ValueError(
                "Matrix not initialised. Call export_to_adjencymatrix() first."
            )

        # Parse cell_type: "cortex" -> ("cortex", "cortex")
        #                   "cortex-endodermis" -> ("cortex", "endodermis")
        parts = cell_type.split("-", maxsplit=1)
        if len(parts) == 1:
            type_a, type_b = parts[0], parts[0]
        else:
            type_a, type_b = parts[0], parts[1]

        valid_labels = {"apoplastic", "transmembrane", "symplastic"}
        if label not in valid_labels:
            raise ValueError(
                f"Unknown label '{label}'. Must be one of {valid_labels}."
            )

        # Determine which walls match the cell_type filter
        matching_walls = self._get_walls_for_types(type_a, type_b)

        for u, v, data in self.graph.edges(data=True):
            path = data.get("path", "")

            if label == "apoplastic" and path == "wall":
                # u or v is a wall node; check if it is in matching_walls
                wall_node = u if u < self.n_walls else (v if v < self.n_walls else None)
                if wall_node is not None and wall_node in matching_walls:
                    self._matrix[u, v] = K
                    self._matrix[v, u] = K

            elif label == "transmembrane" and path == "membrane":
                wall_node = u if u < self.n_walls else (v if v < self.n_walls else None)
                if wall_node is not None and wall_node in matching_walls:
                    self._matrix[u, v] = K
                    self._matrix[v, u] = K

            elif label == "symplastic" and path == "plasmodesmata":
                # Both u and v are cell nodes
                cell_a = u
                cell_b = v
                type_u = self.graph.nodes[cell_a].get("cell_type", "")
                type_v = self.graph.nodes[cell_b].get("cell_type", "")
                if self._types_match(type_u, type_v, type_a, type_b):
                    self._matrix[u, v] = K
                    self._matrix[v, u] = K

    def fill_matrix_from_dict(
        self,
        K_dict: dict, # type: dict[str, float]
        label: str,
    ) -> None:
        """
        Fill entries of the adjacency matrix from a dictionary of hydraulic conductivities.

        Parameters
        ----------
        K_dict : dict[str, float]
            Dictionary matching cell types to hydraulic conductivity values.
        label : str
            Path type to target (e.g. "apoplastic", "transmembrane", "symplastic").
        """
        for cell_type, K in K_dict.items():
            self.fill_matrix(K, label, cell_type)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_walls_for_types(
        self, type_a: str, type_b: str
    ) -> set:
        """Return the set of wall-node indices whose adjacent cells
        match the requested type pair."""
        matching = set()
        for wall_id, cell_indices in self._wall_to_cells.items():
            if len(cell_indices) < 1:
                continue
            cell_types = [
                self.graph.nodes[ci].get("cell_type", "") for ci in cell_indices
            ]
            if len(cell_types) == 1:
                # Boundary wall – only one cell
                if self._types_match(cell_types[0], cell_types[0], type_a, type_b):
                    matching.add(wall_id)
            elif len(cell_types) >= 2:
                if self._types_match(cell_types[0], cell_types[1], type_a, type_b):
                    matching.add(wall_id)
        return matching

    @staticmethod
    def _types_match(
        actual_a: str, actual_b: str, want_a: str, want_b: str
    ) -> bool:
        """Check whether (actual_a, actual_b) matches the requested
        pair in *either* order."""
        return (
            (actual_a == want_a and actual_b == want_b)
            or (actual_a == want_b and actual_b == want_a)
        )

    def plot_network(self, **kwargs):
        position = kwargs.get('position', nx.get_node_attributes(self.graph, 'position'))
        node_types = kwargs.get('node_types', nx.get_node_attributes(self.graph, 'cell_type'))
    
        # Default color map
        default_color_map = {'apo': 'red', 'sym': 'yellow'}
        node_color_map = kwargs.get('node_color_map', default_color_map)
    
        default_edge_color_map = {'wall': 'purple', 'membrane': 'green', 'plasmodesmata': 'gray'}
        edge_color_map = kwargs.get('edge_color_map', default_edge_color_map)
    
        # Determine node colors
        node_colors = []
        for node in self.graph.nodes():
            node_type = node_types.get(node, 'sym')  # Default to 'sym' if type is not found
            node_colors.append(node_color_map.get(node_type, 'blue'))  # Default to 'blue' if color not found
    
            # Determine edge colors
        edge_colors = []
        for u, v, edge_attrs in self.graph.edges(data=True):
            edge_type = edge_attrs.get('path', 'wall')  # Default to 'wall' if path is not found
            edge_colors.append(edge_color_map.get(edge_type, 'purple'))
    
        # Draw the network
        ax = kwargs.get('ax')
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 10)))
    
        nx.draw(
            self.graph,
            position,
            ax=ax,
            node_color=node_colors,
            with_labels=kwargs.get('with_labels', False),
            node_size=kwargs.get('node_size', 10),
            edge_color=edge_colors,
            width=kwargs.get('width', 1),
            alpha=kwargs.get('alpha', 0.7)
        )
    
        # ax.set_title(kwargs.get('title', 'Network Visualization'))
        ax.set_aspect('equal', adjustable='box')
        
        if fig is not None:
            plt.tight_layout()
            plt.show()
