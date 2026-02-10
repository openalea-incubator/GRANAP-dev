import pandas as pd
import numpy as np
from scipy.spatial import Voronoi
from Vertex import vertex
from utils import get_val

def rondy_cortex(params: pd.DataFrame, all_cells: pd.DataFrame, center: float) -> pd.DataFrame:
    """
    Rounds cortex cells and adds intercellular spaces.
    """
    # Params
    icp_ratio = get_val(params, "inter_cellular_space", "ratio")
    icp_size = get_val(params, "inter_cellular_space", "size")
    
    if icp_size is None:
        return all_cells
        
    cor_d = get_val(params, "cortex", "cell_diameter")
    scaling = 0.95
    if icp_size:
        scaling = 1 - (icp_size / cor_d)
        scaling = min(scaling, 0.99)
        
    k_max_xylem = all_cells.loc[all_cells['type'] == 'xylem', 'id_group'].max()
    if pd.isna(k_max_xylem): k_max_xylem = 0
    
    # Filter cortex - R focuses on "cortex" for geometric radius calculation
    cortex_types = ["cortex", "innercortex", "outercortex", "endodermis", "exodermis"]
    all_cortex = all_cells[all_cells['type'].isin(cortex_types)].copy()
    
    if all_cortex.empty:
        return all_cells
        
    # Assign local id_cell for Voronoi mapping
    all_cortex['local_id'] = range(1, len(all_cortex) + 1)
    
    # Voronoi
    coords = all_cortex[['x', 'y']].values
    vor = Voronoi(coords)
    
    # Process regions and vertices
    region_data = []
    for i, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if not region or -1 in region:
            continue
            
        local_id = i + 1
        for v_idx in region:
            v = vor.vertices[v_idx]
            region_data.append({'x': v[0], 'y': v[1], 'id_cell': local_id})
            
    rc2 = pd.DataFrame(region_data)
    
    # Radius bounds for ICP
    inner = all_cortex.loc[all_cortex['type'] == "cortex", 'radius'].min()
    outer = all_cortex.loc[all_cortex['type'] == "cortex", 'radius'].max()
    
    rc2['euc'] = np.sqrt((rc2['x']-center)**2 + (rc2['y']-center)**2)
    
    # Group and sort for geometry
    rc2['mx'] = rc2.groupby('id_cell')['x'].transform('mean')
    rc2['my'] = rc2.groupby('id_cell')['y'].transform('mean')
    rc2['atan'] = np.arctan2(rc2['y'] - rc2['my'], rc2['x'] - rc2['mx'])
    rc2 = rc2.sort_values(['id_cell', 'atan'])
    
    # Merge with original cortex data
    rc1 = rc2.merge(all_cortex[['local_id', 'type', 'radius', 'id_layer', 'order']].rename(columns={'local_id': 'id_cell'}), on='id_cell')
    
    # Intercellular Spaces (ICP)
    rcin = rc2[(rc2['euc'] > inner) & (rc2['euc'] < outer)].drop_duplicates(['x', 'y']).copy()
    all_inter = pd.DataFrame()
    if not rcin.empty:
        rcin['angle'] = np.arctan2(rcin['y'] - center, rcin['x'] - center) % (2*np.pi)
        
        all_inter = pd.DataFrame({
            'angle': rcin['angle'],
            'radius': rcin['euc'],
            'x': rcin['x'], 'y': rcin['y'],
            'id_layer': all_cortex['id_layer'].iloc[0] + 0.5,
            'id_cell': range(1, len(rcin) + 1),
            'type': "inter_cellular_space",
            'order': get_val(params, "cortex", "order") + 0.5,
            'id_group': 0
        })
        
        # ICP proportion logic from R
        if icp_ratio is not None:
            coef_icp = min(1.0, 10.0 * icp_ratio)
            icp_proportion = coef_icp
        else:
            icp_proportion = 0.5
            
        n_keep = int(round(icp_proportion * len(all_inter)))
        if n_keep > 0:
            all_inter = all_inter.sample(n=n_keep, random_state=42)
        else:
            all_inter = pd.DataFrame()

    # Precise Rounding Logic using Vertex
    nodes = vertex(rc1[rc1['type'] == 'cortex'])
    
    # Calculate distance from (mx, my) to wall segment (x1, y1) to (x2, y2)
    dx = nodes['x2'] - nodes['x1']
    dy = nodes['y2'] - nodes['y1']
    # Use standard line distance formula
    nodes['r_dist'] = np.abs(dy * nodes['mx'] - dx * nodes['my'] + nodes['x2']*nodes['y1'] - nodes['y2']*nodes['x1']) / np.sqrt(dx**2 + dy**2)
    
    # Take minimum distance as the rounded radius
    cor = nodes.groupby('id_cell').agg({
        'r_dist': 'min',
        'mx': 'first',
        'my': 'first',
        'id_layer': 'first',
        'type': 'first',
        'order': 'first'
    }).rename(columns={'r_dist': 'precise_radius'}).reset_index()
    
    # Scale boundary layers
    layer_min = all_cortex.loc[all_cortex['type'] == 'cortex', 'id_layer'].min()
    layer_max = all_cortex.loc[all_cortex['type'] == 'cortex', 'id_layer'].max()
    cor.loc[cor['id_layer'].isin([layer_min, layer_max]), 'precise_radius'] *= 0.45
    
    cor['id_group'] = range(1, len(cor) + 1)
    
    # Create rounded frontier points
    frontier_list = []
    circus = np.arange(-0.95, 0.95 + 1e-9, 0.95/4)
    
    for _, cell in cor.iterrows():
        r = cell['precise_radius']
        mx, my = cell['mx'], cell['my']
        
        for val in circus:
            y_offset = np.sqrt(1 - val**2)
            for sign in [1, -1]:
                fx = val * r * scaling + mx
                fy = sign * y_offset * r * scaling + my
                
                # Angle relative to ROOT center
                angle = np.arctan2(fy-center, fx-center) % (2*np.pi)
                
                frontier_list.append({
                    'radius': r,
                    'x': fx,
                    'y': fy,
                    'angle': angle,
                    'id_layer': cell['id_layer'],
                    'id_cell': 1,
                    'type': cell['type'],
                    'order': cell['order'],
                    'id_group': cell['id_group']
                })
                
    cor_frontier = pd.DataFrame(frontier_list)
    
    # Reassemble all_cells
    # Remove old cortex, add new rounded cortex + Intercellular
    non_cortex = all_cells[all_cells['type'] != "cortex"]
    
    # Fix IDs
    cor_frontier['id_group'] += k_max_xylem
    
    combined = pd.concat([non_cortex, cor_frontier, all_inter], ignore_index=True)
    combined['id_cell'] = range(1, len(combined) + 1)
    
    return combined
