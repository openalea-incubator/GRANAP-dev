import pandas as pd
import numpy as np

def septa(rs1: pd.DataFrame) -> pd.DataFrame:
    """
    Translates septa.R.
    Simplifies boundaries between aerenchyma lacunae.
    """
    data = rs1.copy()
    data['id_point'] = data['x'].astype(str) + ";" + data['y'].astype(str)
    
    aerenchyma_mask = data['type'] == "aerenchyma"
    aer_indices = data[aerenchyma_mask].index
    
    # Store neighbor info
    # R implementation iterates row-wise which is slow.
    # Goal: identify points in aerenchyma cells that are shared with other cells.
    
    # Map point -> set of (id_cell, type)
    # Using groupby
    point_info = data.groupby('id_point')[['id_cell', 'type']].apply(
        lambda x: list(zip(x['id_cell'], x['type']))
    ).to_dict()
    
    keep_indices = []
    noise_candidates = []
    
    for idx in aer_indices:
        pt = data.at[idx, 'id_point']
        my_cell = data.at[idx, 'id_cell']
        
        neighbors = point_info.get(pt, [])
        # Find neighbors not same cell
        others = [n for n in neighbors if n[0] != my_cell]
        
        # R: ne[1], ne[2]...
        # Store neib1, neib2
        if len(others) >= 1:
            data.at[idx, 'neib1'] = others[0][0]
            data.at[idx, 'neib1_type'] = others[0][1]
        if len(others) >= 2:
            data.at[idx, 'neib2'] = others[1][0]
            data.at[idx, 'neib2_type'] = others[1][1]
            
        # Decision logic from R
        # must <- data[!is.na(neib2)] OR data[neib1_type != "aerenchyma"]
        has_neib2 = len(others) >= 2
        neib1_is_not_aer = len(others) >= 1 and others[0][1] != "aerenchyma"
        
        if has_neib2 or neib1_is_not_aer:
            keep_indices.append(idx)
        else:
            noise_candidates.append(idx)
            
    # Add some noise (5%)
    import random
    n_noise = int(round(len(noise_candidates) * 0.05))
    noise_keep = random.sample(noise_candidates, n_noise)
    keep_indices.extend(noise_keep)
    
    # Construct final
    # Keep non-aerenchyma as is
    non_aer = data[~aerenchyma_mask]
    
    aer_kept = data.loc[keep_indices]
    
    # R sorts by atan? "arrange(id_cell, atan)"
    # Assuming 'atan' column exists (from previous steps like RondyCortex/Vertex)
    final = pd.concat([non_aer, aer_kept], ignore_index=True)
    if 'atan' in final.columns:
        final = final.sort_values(['id_cell', 'atan'])
        
    return final[rs1.columns] # restore columns
