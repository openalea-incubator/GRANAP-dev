import pandas as pd
import numpy as np

def root_hair(rs1: pd.DataFrame, params: pd.DataFrame, center: float) -> pd.DataFrame:
    """
    Translates root_hair.R.
    Adds root hairs to epidermis cells.
    """
    
    def get_val(name, ptype):
        v = params[(params['name'] == name) & (params['type'] == ptype)]['value']
        return v.iloc[0] if not v.empty else 0
        
    n_hair = int(get_val("hair", "n_files"))
    len_hair = get_val("hair", "length")
    
    if n_hair <= 0:
        return rs1
        
    epi_idx = rs1[rs1['type'] == 'epidermis']['id_cell'].unique()
    if len(epi_idx) == 0:
        return rs1
        
    selected_hairs = np.random.choice(epi_idx, min(len(epi_idx), n_hair), replace=False)
    
    new_rows = []
    
    # Iterate over selected cells
    for cell_id in selected_hairs:
        cell_data = rs1[rs1['id_cell'] == cell_id].copy()
        
        # Calculate distance to center
        cell_data['dist'] = np.sqrt((cell_data['x'] - center)**2 + (cell_data['y'] - center)**2)
        
        # Identify "outer" points
        # R logic doesn't explicitly state how outer points are found, but typically points further from center
        # The user provided snippet used mean dist.
        threshold = cell_data['dist'].mean()
        outer_mask = cell_data['dist'] > threshold
        
        # Move outer points
        # This distorts the cell shape to look like a hair
        # R implementation logic? The user snippet was pythonic. Let's trust it.
        
        for idx in cell_data[outer_mask].index:
            cx = cell_data.at[idx, 'x']
            cy = cell_data.at[idx, 'y']
            
            # Vector from center
            dx = cx - center
            dy = cy - center
            mag = np.sqrt(dx**2 + dy**2)
            
            if mag > 0:
                cell_data.at[idx, 'x'] += (dx / mag) * len_hair
                cell_data.at[idx, 'y'] += (dy / mag) * len_hair
                
        new_rows.append(cell_data)
        
    # Reassemble
    rs1 = rs1[~rs1['id_cell'].isin(selected_hairs)]
    if new_rows:
        hair_cells = pd.concat(new_rows)
        rs1 = pd.concat([rs1, hair_cells], ignore_index=True)
        
    return rs1
