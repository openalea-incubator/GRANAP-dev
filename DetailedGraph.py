import networkx as nx
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon

def create_detailed_graph(cells_gdf, cell_polys):
    """
    Constructs a detailed graph with Cell, Wall, and Junction nodes using GeoPandas.
    
    Args:
        cells_gdf (gpd.GeoDataFrame): GeoDataFrame containing cell information with polygon geometries
        cell_polys: Unused (kept for compatibility)
                           
    Returns:
        nx.Graph: The detailed graph with Cell, Wall, and Junction nodes
    """
    G = nx.Graph()
    
    # Tracking for unique elements
    junction_map = {}  # Key: (x, y) rounded -> Value: junction_node_id
    wall_map = {}  # Key: frozenset({j1_id, j2_id}) -> Value: wall_node_id
    
    # ID offsets to avoid collisions
    WALL_OFFSET = 100000
    JUNCTION_OFFSET = 500000
    
    next_wall_id = WALL_OFFSET
    next_junction_id = JUNCTION_OFFSET
    
    print(f"Building detailed graph from {len(cells_gdf)} cells...")
    
    # Helper to clean/round coords
    def get_coord_key(x, y):
        return (round(x, 3), round(y, 3))
    
    # Debug counters
    total_edges_checked = 0
    shared_walls_found = 0
    
    # 1. Add Cell Nodes
    for idx, row in cells_gdf.iterrows():
        cell_id = int(row['id_cell'])
        attrs = row.to_dict()
        node_id = cell_id
        
        G.add_node(node_id, **attrs)
        if 'type' not in attrs:
            G.nodes[node_id]['type'] = 'cell_unknown'
    
    # 2. Iterate Polygons to build Walls and Junctions
    for idx, row in cells_gdf.iterrows():
        cell_id = int(row['id_cell'])
        poly = row['geometry']
        
        if not isinstance(poly, (Polygon, MultiPolygon)):
            continue
            
        polys = [poly] if isinstance(poly, Polygon) else poly.geoms
        
        for p in polys:
            if p.is_empty:
                continue
            
            # Exterior coordinates
            coords = list(p.exterior.coords)
            if coords[0] == coords[-1]:
                coords = coords[:-1]
                
            n_pts = len(coords)
            
            # Iterate edges
            for i in range(n_pts):
                p1 = coords[i]
                p2 = coords[(i + 1) % n_pts]
                
                # --- JUNCTIONS ---
                k1 = get_coord_key(*p1)
                k2 = get_coord_key(*p2)
                
                if k1 not in junction_map:
                    jid1 = next_junction_id
                    next_junction_id += 1
                    junction_map[k1] = jid1
                    G.add_node(jid1, type='junction', x=k1[0], y=k1[1], geometry=None)
                else:
                    jid1 = junction_map[k1]
                    
                if k2 not in junction_map:
                    jid2 = next_junction_id
                    next_junction_id += 1
                    junction_map[k2] = jid2
                    G.add_node(jid2, type='junction', x=k2[0], y=k2[1], geometry=None)
                else:
                    jid2 = junction_map[k2]
                
                # --- WALLS ---
                wall_key = frozenset({jid1, jid2})
                total_edges_checked += 1
                
                if wall_key not in wall_map:
                    wid = next_wall_id
                    next_wall_id += 1
                    wall_map[wall_key] = wid
                    
                    # Wall Attributes
                    mid_x = (k1[0] + k2[0]) / 2
                    mid_y = (k1[1] + k2[1]) / 2
                    length = np.hypot(k2[0] - k1[0], k2[1] - k1[1])
                    
                    G.add_node(wid, type='wall', x=mid_x, y=mid_y, length=length, geometry=None)
                    
                    # Edges: Wall <-> Junction
                    G.add_edge(wid, jid1, type='wall_structure') 
                    G.add_edge(wid, jid2, type='wall_structure')
                    
                    # Plasmodesmata preparation
                    G.nodes[wid]['connected_cells'] = []
                else:
                    wid = wall_map[wall_key]
                    shared_walls_found += 1
                
                # --- CONNECT CELL TO WALL ---
                # Edge: Membrane
                cx, cy = row['x'], row['y']
                wx, wy = G.nodes[wid]['x'], G.nodes[wid]['y']
                dist = np.hypot(wx - cx, wy - cy)
                
                G.add_edge(cell_id, wid, type='membrane', dist=dist)
                
                # Track for Plasmodesmata
                G.nodes[wid]['connected_cells'].append(cell_id)


    print(f"DEBUG: Processed {total_edges_checked} polygon edges. Found {shared_walls_found} shared walls (reuse events).")

    # Plasmodesmata concept removed per user request
    # Cells are connected only via membrane->wall->junction topology
    # No direct cell-to-cell edges
    
    print(f"Detailed Graph created:")
    print(f"- Cells: {len(cells_gdf)}")
    print(f"- Junctions: {len(junction_map)}")
    print(f"- Walls: {len(wall_map)}")
    
    return G
