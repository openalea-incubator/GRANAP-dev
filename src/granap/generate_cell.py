"""
Cell generator module for creating cells using Voronoi tessellation.
"""

import numpy as np
import pandas as pd
import shapely as sp
import geopandas as gpd
from scipy.spatial import Voronoi
from typing import List, Dict, Any, Tuple
from shapely.geometry import Polygon, Point, MultiPolygon
from scipy.spatial import cKDTree

from granap.geometry_collection import GeometryProcessor
from granap.cell_class import Cell
from granap.cell_manager import CellManager


class CellGenerator:
    """
    Generates plant cells using Voronoi tessellation.
    
    Handles cell placement on layers, border generation, and
    Voronoi diagram processing.
    """
    
    @staticmethod
    def cells_on_layer(layer_polygon: Polygon, cell_diameter: float, 
                      cell_width: float = 0, shift: float = 0) -> np.ndarray:
        """
        Generate cell center positions along a layer polygon.
        
        Args:
            layer_polygon: Polygon representing the layer boundary
            cell_diameter: Diameter of cells
            cell_width: Optional cell width (0 = use diameter)
            shift: Shift parameter (0-1)
        
        Returns:
            Array of (x, y) cell center coordinates
        """
        x, y = np.array(layer_polygon.exterior.coords.xy)
        perimeter = layer_polygon.length
        
        if cell_width == 0:
            cell_width = cell_diameter
        
        n_cells = int(np.ceil(perimeter / cell_width))
        
        # Calculate shift distance: shift of 1.0 = 1 cell width displacement
        # Randomized between 0 and shift * cell_width
        max_shift = shift * cell_width
        shift_distance = np.random.uniform(0, max_shift) if max_shift > 0 else 0
        
        cells_coords = GeometryProcessor.resample_coords(
            np.column_stack((x, y)), n_cells, shift_distance=shift_distance
        )
        return cells_coords
    
    @staticmethod
    def cell_border(cell_coords: np.ndarray, cell_height: float, 
                   cell_width: float = 0) -> List[np.ndarray]:
        """
        Generate border points for elliptical cells.
        
        Args:
            cell_coords: Array of cell center coordinates
            cell_height: Height of cells
            cell_width: Width of cells (0 = use height)
        
        Returns:
            List of arrays, each containing border points for one cell
        """
        if len(cell_coords) == 0:
            return []
        
        major_axis = cell_height
        minor_axis = cell_width if cell_width != 0 else cell_height
        
        n_points = 15 if cell_height != cell_width else 10
        
        cells_border = []
        for i, cell_coord in enumerate(cell_coords):
            if i == len(cell_coords) - 1:
                next_cell_coord = cell_coords[0]
            else:
                next_cell_coord = cell_coords[i + 1]
            
            prev_cell_coord = cell_coords[i - 1]
            
            axis = np.arctan2(
                next_cell_coord[1] - prev_cell_coord[1],
                next_cell_coord[0] - prev_cell_coord[0]
            )
            
            cells_border.append(
                GeometryProcessor.draw_ellipse(
                    cell_coord, axis, 
                    major_axis / 2, minor_axis / 2, 
                    n_points=n_points
                )
            )
        return cells_border
    
    @staticmethod
    def generate_cells_info(layers_polygons: List[Dict[str, Any]], 
                           center: Point):
        """
        Generate cell information from layer polygons.
        
        Args:
            layers_polygons: List of layer polygon dictionaries
            center: Center point for angle/radius calculations
        
        Returns:
            pd.DataFrame of cells
        """
        all_cells = CellManager()
        id_cell = 1
        id_group = 1
        
        for i_layer, layer in enumerate(layers_polygons):
            cells_coords = CellGenerator.cells_on_layer(
                layer["polygon"], 
                layer["cell_diameter"], 
                layer["cell_width"],
                layer.get("shift", 0)
            )
            
            if layer["cell_width"] != 0 and layer["cell_width"] < layer["cell_diameter"]:
                layer_cell_borders = CellGenerator.cell_border(
                    cells_coords, 
                    layer["cell_width"] * 0.7, 
                    layer["cell_diameter"] * 0.7
                )
            elif layer["cell_width"] != 0 and layer["cell_width"] > layer["cell_diameter"]:
                layer_cell_borders = CellGenerator.cell_border(
                    cells_coords, 
                    layer["cell_width"] * 0.7, 
                    layer["cell_diameter"] * 0.7
                )
            else:
                layer_cell_borders = CellGenerator.cell_border(
                    cells_coords, 
                    layer["cell_diameter"] * 0.7, 
                    layer["cell_width"] * 0.7
                )
            
            for i, cell_coord in enumerate(cells_coords[1:]):
                if layer["name"] == "parenchyma":
                    new_cell = Cell(
                        type=layer["name"],
                        x=cell_coord[0],
                        y=cell_coord[1],
                        diameter=layer["cell_diameter"],
                        id_cell=id_cell,
                        id_layer=i_layer,
                        id_group=id_group,
                        angle=np.arctan2(cell_coord[1] - center.y, 
                                          cell_coord[0] - center.x),
                        radius=np.sqrt((cell_coord[0] - center.x)**2 + 
                                        (cell_coord[1] - center.y)**2),
                        area=np.pi * (layer["cell_diameter"] / 2)**2,
                    )
                    all_cells.add_cell(new_cell)
                    id_cell += 1
                    id_group += 1
                else:
                    cell_border_points = layer_cell_borders[i]
                    for border_point in cell_border_points[1:]:
                        new_cell = Cell(
                            type=layer["name"],
                            x=border_point[0],
                            y=border_point[1],
                            diameter=layer["cell_diameter"],
                            id_cell=id_cell,
                            id_layer=i_layer,
                            id_group=id_group,
                            angle=np.arctan2(cell_coord[1] - center.y, 
                                              cell_coord[0] - center.x),
                            radius=np.sqrt((cell_coord[0] - center.x)**2 + 
                                            (cell_coord[1] - center.y)**2),
                            area=np.pi * (layer["cell_diameter"] / 2)**2,
                        )
                        all_cells.add_cell(new_cell)
                        id_cell += 1
                    id_group += 1
        
        return all_cells

    @staticmethod
    def voronoi_diagram(all_cells: CellManager) -> Voronoi:
        # get all x and y coordinates
        for cell in all_cells.cells:
            cell.jitter()
        cells_df = pd.DataFrame([cell.cell_to_dict() for cell in all_cells.cells])
        vor = Voronoi(cells_df[["x", "y"]])
        return vor
    
    @staticmethod
    def process_voronoi_groups(all_cells: CellManager, 
                               vor: Voronoi) -> List[Cell]:
        """
        Process Voronoi diagram into grouped cell geometries.
        
        Args:
            all_cells: List of Cell objects
            vor: Voronoi diagram
        
        Returns:
            List of updated Cell objects with geometries
        """
        updated_cells = CellManager()
        
        for i, cell in enumerate(all_cells.cells):
            region_idx = vor.point_region[i]
            region_vertices_indices = vor.regions[region_idx]
            
            if -1 in region_vertices_indices or len(region_vertices_indices) == 0:
                cell.polygon = None
            else:
                vertices = vor.vertices[region_vertices_indices]
                poly = sp.Polygon(vertices)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                cell.polygon = poly
            
            if cell.type != "outside" and cell.polygon is not None:
                updated_cells.add_cell(cell)
                
        # Group handling is trickier with objects. 
        # The original code used GeoPandas dissolve to union polygons by group.
       
        # return a list of 'biological' cells (one per group).
        
        cell_dicts = [c.cell_to_dict() for c in updated_cells.cells]
        for i, c in enumerate(updated_cells.cells):
            cell_dicts[i]['geometry'] = c.polygon
            
        gdf = gpd.GeoDataFrame(cell_dicts)
        
        # Dissolve by id_group
        grouped_gdf = gdf.dissolve(by="id_group", as_index=False)
        grouped_gdf["area"] = grouped_gdf.geometry.area
        
        # Now create new Cell objects from the grouped results
        final_cells = CellManager()
        for _, row in grouped_gdf.iterrows():
            # Find a representative original cell to get non-geometric attributes
            # (or use the aggregated ones, but dissolve aggregates strategy might be needed for some?)
            # simple 'first' strategy is default for dissolve.
            
            new_cell = Cell(
                type=row['type'],
                x=row['x'], # Centroid might be better?
                y=row['y'],
                diameter=row['cell_diameter'],
                id_cell=row['id_cell'],
                id_layer=row['id_layer'],
                id_group=row['id_group'],
                angle=row['angle'],
                radius=row['radius'],
                area=row['area'],
                polygon=row['geometry']
            )
            final_cells.add_cell(new_cell)
            
        return final_cells
    
    @staticmethod
    def _build_topology(
        polys: List,
        cell_ids: List[Any],
    ) -> Tuple[Dict[Any, List[tuple]], Dict[tuple, set], Dict[tuple, set], set]:
        """
        Build the shared vertex/edge topology for a collection of polygons.

        Runs KD-tree vertex snapping (Phase 0), then constructs
        ``cell_vkeys``, ``vertex_to_cells``, ``edge_to_cells`` (Phase 1),
        and finally identifies junction vertices (Phase 2).

        This helper is called by both :meth:`simplify_cells` and
        ``Organ._build_anatnetwork`` so the logic lives in one place.

        Args:
            polys:     Sequence of Shapely geometries (``None`` entries are
                       skipped).  Index position must correspond to
                       ``cell_ids``.
            cell_ids:  Opaque identifier for each polygon (list/GeoDataFrame
                       index, integer position, …).

        Returns:
            ``(cell_vkeys, vertex_to_cells, edge_to_cells, junction_set)``

            * ``cell_vkeys``       – ``{cell_id: [snapped (x,y) tuples]}``
            * ``vertex_to_cells``  – ``{(x,y): set(cell_ids)}``
            * ``edge_to_cells``    – ``{edge_key: set(cell_ids)}``
            * ``junction_set``     – set of ``(x,y)`` junction vertices
        """
        n_dec = 6

        # ------------------------------------------------------------------
        # Phase 0 — collect raw vertices and snap nearby ones together
        # ------------------------------------------------------------------
        raw_cell_data: Dict[Any, list] = {}
        all_raw_verts: list = []
        vert_global_idx: Dict[Any, List[int]] = {}

        for cid, poly in zip(cell_ids, polys):
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
            raw_cell_data[cid] = coords
            vert_global_idx[cid] = indices

        if not all_raw_verts:
            return {}, {}, {}, set()

        coords_arr = np.array(all_raw_verts)
        kd_tree = cKDTree(coords_arr)

        # Snap tolerance: 1 % of 5th-percentile edge length
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
        canonical: List = [None] * len(all_raw_verts)
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

        # ------------------------------------------------------------------
        # Phase 1 — build cell_vkeys, vertex_to_cells, edge_to_cells
        # ------------------------------------------------------------------
        vertex_to_cells: Dict[tuple, set] = {}
        cell_vkeys: Dict[Any, List[tuple]] = {}

        for cid, gidxs in vert_global_idx.items():
            vkeys_raw = [canonical[gi] for gi in gidxs]
            vkeys: List[tuple] = [vkeys_raw[0]]
            for vk in vkeys_raw[1:]:
                if vk != vkeys[-1]:
                    vkeys.append(vk)
            if len(vkeys) > 1 and vkeys[-1] == vkeys[0]:
                vkeys = vkeys[:-1]
            if len(vkeys) < 3:
                continue
            cell_vkeys[cid] = vkeys
            for vk in vkeys:
                vertex_to_cells.setdefault(vk, set()).add(cid)

        edge_to_cells: Dict[tuple, set] = {}
        for cid, vkeys in cell_vkeys.items():
            n = len(vkeys)
            for i in range(n):
                ek = tuple(sorted((vkeys[i], vkeys[(i + 1) % n])))
                edge_to_cells.setdefault(ek, set()).add(cid)

        # ------------------------------------------------------------------
        # Phase 2 — identify junction vertices
        # ------------------------------------------------------------------
        junction_set: set = set()

        for vk in vertex_to_cells:
            if len(vertex_to_cells[vk]) >= 3:
                junction_set.add(vk)
                continue
            incident_pairs: set = set()
            for cid in vertex_to_cells[vk]:
                vks = cell_vkeys[cid]
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

        return cell_vkeys, vertex_to_cells, edge_to_cells, junction_set

    @staticmethod
    def simplify_cells(grouped_cells: List[Cell]) -> List[Cell]:
        """
        Simplify cell boundaries by retaining only junction vertices.

        Delegates topology computation to :meth:`_build_topology` (Phases
        0–2: KD-tree snapping, vertex/edge maps, junction detection), then
        rebuilds each polygon keeping only its junction vertices (Phase 3).

        Args:
            grouped_cells: List of Cell objects with polygon geometries.

        Returns:
            The same list with simplified polygon geometries in place.
        """
        polys = [c.polygon for c in grouped_cells]
        cell_ids = list(range(len(grouped_cells)))

        cell_vkeys, _, _, junction_set = CellGenerator._build_topology(
            polys, cell_ids
        )

        if not cell_vkeys:
            return grouped_cells

        # Phase 3 — rebuild each polygon keeping only junction vertices
        for idx, cell in enumerate(grouped_cells):
            if idx not in cell_vkeys:
                continue

            vkeys = cell_vkeys[idx]
            simplified = [vk for vk in vkeys if vk in junction_set]

            if len(simplified) < 3:
                simplified = vkeys

            ring_coords = list(simplified)
            if ring_coords[0] != ring_coords[-1]:
                ring_coords.append(ring_coords[0])

            new_poly = Polygon(ring_coords)
            if not new_poly.is_valid:
                new_poly = new_poly.buffer(0)
            cell.polygon = new_poly

        return grouped_cells
