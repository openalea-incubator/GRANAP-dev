"""
Cell generator module for creating cells using Voronoi tessellation.
"""

import numpy as np
import pandas as pd
import shapely as sp
import geopandas as gpd
from scipy.spatial import Voronoi
from typing import List, Dict, Any, Tuple
from shapely.geometry import Polygon, Point

from granap.geometry_collection import GeometryProcessor
from granap.cell_class import Cell


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
        all_cells: List[Cell] = []
        cells_data_for_voronoi = []
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
                    all_cells.append(new_cell)
                    cells_data_for_voronoi.append({"x": new_cell.x, "y": new_cell.y})
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
                        all_cells.append(new_cell)
                        cells_data_for_voronoi.append({"x": new_cell.x, "y": new_cell.y})
                        id_cell += 1
                    id_group += 1
        
        return all_cells

    @staticmethod
    def voronoi_diagram(all_cells: List[Cell]) -> Voronoi:
        # get all x and y coordinates
        for cell in all_cells:
            cell.jitter()
        cells_df = pd.DataFrame([cell.cell_to_dict() for cell in all_cells])
        vor = Voronoi(cells_df[["x", "y"]])
        return vor
    
    @staticmethod
    def process_voronoi_groups(all_cells: List[Cell], 
                               vor: Voronoi) -> List[Cell]:
        """
        Process Voronoi diagram into grouped cell geometries.
        
        Args:
            all_cells: List of Cell objects
            vor: Voronoi diagram
        
        Returns:
            List of updated Cell objects with geometries
        """
        updated_cells = []
        
        for i, cell in enumerate(all_cells):
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
                updated_cells.append(cell)
                
        # Group handling is trickier with objects. 
        # The original code used GeoPandas dissolve to union polygons by group.
        # We need to replicate that logic or use GeoPandas intermediately.
        
        # Let's use GeoPandas for the geometric operations as before, 
        # but map results back to a simplified list of cells (one per group).
        
        # ... Wait, if we group them, we reduce the number of cells. 
        # The original code dissolved by id_group.
        # This implies that multiple 'sub-cells' (border points) formed one actual biological cell.
        
        # So we should return a list of 'biological' cells (one per group).
        
        cell_dicts = [c.cell_to_dict() for c in updated_cells]
        for i, c in enumerate(updated_cells):
            cell_dicts[i]['geometry'] = c.polygon
            
        gdf = gpd.GeoDataFrame(cell_dicts)
        
        # Dissolve by id_group
        grouped_gdf = gdf.dissolve(by="id_group", as_index=False)
        grouped_gdf["area"] = grouped_gdf.geometry.area
        
        # Now create new Cell objects from the grouped results
        final_cells = []
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
            final_cells.append(new_cell)
            
        return final_cells
    
    @staticmethod
    def smooth_cells(grouped_cells: List[Cell]) -> List[Cell]:
        """
        Smooth cell boundaries by straightening shared edges.
        
        Args:
            grouped_cells: List of Cell objects
        
        Returns:
            List of smoothed Cell objects
        """
        # Convert to GeoDataFrame for processing
        cell_dicts = [c.cell_to_dict() for c in grouped_cells]
        for i, c in enumerate(grouped_cells):
            cell_dicts[i]['geometry'] = c.polygon
        
        grouped_gdf = gpd.GeoDataFrame(cell_dicts)
        from shapely.geometry import MultiPolygon
        
        geoms = grouped_gdf.geometry.tolist()
        
        # Build point map and edge map
        point_map = {}
        next_pt_id = 0
        coords_list = []
        
        def get_pt_id_mem(x, y):
            nonlocal next_pt_id
            k = (round(x, 6), round(y, 6))
            if k not in point_map:
                point_map[k] = next_pt_id
                coords_list.append(k)
                next_pt_id += 1
            return point_map[k]
        
        edge_to_polys = {}
        poly_rings_ids = []
        
        for idx, poly in enumerate(geoms):
            if poly is None or poly.is_empty:
                poly_rings_ids.append([])
                continue
            
            # Handle both Polygon and MultiPolygon
            if isinstance(poly, MultiPolygon):
                # For MultiPolygon, just use the largest polygon
                poly = max(poly.geoms, key=lambda p: p.area)
            
            rings = [poly.exterior]
            rings_pt_ids = []
            
            for ring in rings:
                pts = list(ring.coords)
                if pts[0] == pts[-1]:
                    pts = pts[:-1]
                
                if len(pts) < 3:
                    rings_pt_ids.append([])
                    continue
                
                p_ids = [get_pt_id_mem(x, y) for x, y in pts]
                rings_pt_ids.append(p_ids)
                
                n_pts = len(p_ids)
                for i in range(n_pts):
                    u = p_ids[i]
                    v = p_ids[(i + 1) % n_pts]
                    if u == v:
                        continue
                    
                    edge_key = tuple(sorted((u, v)))
                    if edge_key not in edge_to_polys:
                        edge_to_polys[edge_key] = []
                    edge_to_polys[edge_key].append(idx)
            
            poly_rings_ids.append(rings_pt_ids)
        
        # Reconstruct polygons with straightened boundaries
        new_geoms = []
        
        for idx in range(len(geoms)):
            rings_ids = poly_rings_ids[idx]
            if not rings_ids:
                new_geoms.append(geoms[idx])
                continue
            
            new_rings_coords = []
            
            for ring_pt_ids in rings_ids:
                if not ring_pt_ids:
                    continue
                
                n_pts = len(ring_pt_ids)
                edge_neighbors = []
                
                for i in range(n_pts):
                    u = ring_pt_ids[i]
                    v = ring_pt_ids[(i + 1) % n_pts]
                    edge_key = tuple(sorted((u, v)))
                    
                    neighbors = edge_to_polys.get(edge_key, [])
                    
                    other = None
                    for n_idx in neighbors:
                        if n_idx != idx:
                            other = n_idx
                            break
                    edge_neighbors.append(other)
                
                # Filter vertices
                optimized_ring = []
                for k in range(n_pts):
                    u = ring_pt_ids[k]
                    
                    prev_edge_idx = (k - 1) % n_pts
                    curr_edge_idx = k
                    
                    n_prev = edge_neighbors[prev_edge_idx]
                    n_curr = edge_neighbors[curr_edge_idx]
                    
                    if n_prev != n_curr:
                        optimized_ring.append(u)
                    elif n_prev is None:
                        optimized_ring.append(u)
                
                if len(optimized_ring) < 3:
                    optimized_ring = ring_pt_ids
                
                ring_coords = [coords_list[pid] for pid in optimized_ring]
                new_rings_coords.append(ring_coords)
            
            if not new_rings_coords:
                new_geoms.append(geoms[idx])
            else:
                ext = new_rings_coords[0]
                if ext[0] != ext[-1]:
                    ext.append(ext[0])
                
                new_poly = Polygon(ext)
                new_geoms.append(new_poly)
        
        grouped_gdf.geometry = new_geoms
        
        # Update cells with smoothed geometries
        for i, cell in enumerate(grouped_cells):
            if i < len(new_geoms):
                 cell.polygon = new_geoms[i]
                 
        return grouped_cells
