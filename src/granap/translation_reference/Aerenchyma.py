import numpy as np
import pandas as pd
from Concavety import concavety
from utils import get_val

def aerenchyma(params, rs1):
    # 1. Setup Parameters
    n_files = int(get_val(params, "aerenchyma", "n_files"))
    proportion = get_val(params, "aerenchyma", "proportion")
    aer_type = get_val(params, "aerenchyma", "type") or get_val(params, "planttype", "param")
    
    if n_files == 0 or proportion == 0:
        return rs1

    # 2. Calculate target area to "fuse"
    # Include all cortex layer types
    cortex_types = ["cortex", "innercortex", "outercortex"]
    cell_areas = rs1.groupby('id_cell')['area'].first()
    cortex_mask = rs1['type'].isin(cortex_types + ["inter_cellular_space"])
    ini_cortex_area = rs1[rs1['type'].isin(cortex_types)].groupby('id_cell')['area'].first().sum()
    
    stk_zone = (ini_cortex_area * proportion) / n_files
    
    # 3. Geometry Setup
    angle_inc = (2 * np.pi) / n_files
    cortex_layers = rs1[rs1['type'].isin(cortex_types)]['id_layer'].unique()
    safe_layer = min(cortex_layers) if len(cortex_layers) > 0 else 0
    
    # Initial angle (with a bit of randomness)
    current_angle = np.random.uniform(0.6, 1.0) * np.pi / n_files
    id_group_max = rs1['id_group'].max()

    # Pre-calculate neighbors for speed (Graph-based approach)
    # This replaces the 'pointy' / 'nei' logic in R
    # Two cells are neighbors if they share an id_point
    point_to_cells = rs1[cortex_mask].groupby('id_point')['id_cell'].apply(set).to_dict()
    
    # Final Result container
    final_rs1 = rs1.copy()

    for j in range(n_files):
        # Filter possible cells for this file
        possi = final_rs1[(final_rs1['id_layer'] != safe_layer) & 
                         (final_rs1['type'].isin(cortex_types + ["inter_cellular_space"]))]
        
        # logic to find seed cell closest to current_angle
        # (This is a simplified version of the R angle_range logic)
        possi_centers = possi.groupby('id_cell')[['angle', 'area']].mean()
        possi_centers['angle_diff'] = np.abs(possi_centers['angle'] - current_angle)
        
        # Start growing the lacuna
        selected_ids = []
        current_area = 0
        
        # Seed cell
        seed_id = possi_centers['angle_diff'].idxmin()
        selected_ids.append(seed_id)
        current_area += cell_areas[seed_id]
        
        # iterative neighbor expansion
        n_try = 0
        while current_area < stk_zone and n_try < 100:
            # Find points belonging to selected cells
            current_points = final_rs1[final_rs1['id_cell'].isin(selected_ids)]['id_point'].unique()
            
            # Find neighbor cells via points
            neighbors = set()
            for pt in current_points:
                neighbors.update(point_to_cells.get(pt, set()))
            
            # Filter neighbors: only those not selected and not in safe layer
            new_neighbors = neighbors - set(selected_ids)
            
            # Additional check: ensure neighbors are in the 'possible' set (not safe layer)
            valid_neighbors = [nid for nid in new_neighbors if nid in possi_centers.index]
            
            if not valid_neighbors: break
            
            # Sort neighbors by angle proximity to keep the lacuna radial
            neigh_df = possi_centers.loc[valid_neighbors]
            next_id = neigh_df['angle_diff'].idxmin()
            
            selected_ids.append(next_id)
            current_area += cell_areas[next_id]
            n_try += 1

        # Create the aerenchyma lacuna (merging selected cells)
        if selected_ids:
            lacuna_nodes = final_rs1[final_rs1['id_cell'].isin(selected_ids)].copy()
            
            # Remove killed cells from main df
            final_rs1 = final_rs1[~final_rs1['id_cell'].isin(selected_ids)]
            
            # Re-order nodes for the new merged lacuna boundary
            lacuna_nodes = concavety(lacuna_nodes) # Defined in previous steps
            lacuna_nodes['type'] = "aerenchyma"
            lacuna_nodes['id_group'] = id_group_max + j + 1
            lacuna_nodes['id_cell'] = final_rs1['id_cell'].max() + 1
            
            final_rs1 = pd.concat([final_rs1, lacuna_nodes], ignore_index=True)

        current_angle += angle_inc
        print(f"Lacuna {j+1}/{n_files} created.")

    return final_rs1
