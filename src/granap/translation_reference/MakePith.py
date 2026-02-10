import pandas as pd
import numpy as np
from LayerInfo import layer_info
from CreateCells import create_cells
from utils import get_val

def make_pith(all_cells: pd.DataFrame, params: pd.DataFrame, center: float) -> pd.DataFrame:
    """
    Removes central xylem cells and replaces them with pith cells.
    """
    
    # Check parameters
    pith_val = get_val(params, "pith", "layer_diameter")
    if pith_val > 0:
        pith_size = pith_val / 2
        pcell = get_val(params, "pith", "cell_diameter")
    else:
        return all_cells # No pith

    if pith_size > 0:
        # 1. Identify Xylem cells in the center to remove
        xylem = all_cells[all_cells['type'] == 'xylem'].copy()
        
        # Calculate group centroids for xylem
        if 'id_group' in xylem.columns:
             xylem_groups = xylem.groupby('id_group').agg({'x': 'mean', 'y': 'mean'}).reset_index()
             xylem_groups['euc'] = np.sqrt((xylem_groups['x'] - center)**2 + (xylem_groups['y'] - center)**2)
             
             # Inner xylem groups
             inner_groups = xylem_groups[xylem_groups['euc'] < pith_size]['id_group'].unique()
             
             # Remove these from all_cells
             # all_cells <- all_cells[type != xylem OR id_group NOT in inner]
             mask_keep = (all_cells['type'] != 'xylem') | (~all_cells['id_group'].isin(inner_groups))
             all_cells = all_cells[mask_keep]
        
        # 2. Create Pith Layer
        n_pith_lay = int(np.round(1 + (pith_size - pcell/2)/pcell))
        
        pith_layer_df = pd.DataFrame({
            'name': ['stele'],
            'n_layers': n_pith_lay, # need to be verified
            'cell_diameter': pcell,
            'order': [0.5]
        })
        
        pith_layer_df = layer_info(pith_layer_df, params)
        
        # Get randomness
        rand_val = params[(params['name'] == "randomness") & (params['type'] == "value")]
        random_factor = rand_val['value'].iloc[0] if not rand_val.empty else 1.0
        # Scale randomness appropriate for pith cells (smaller cells need smaller absolute jitter?)
        # create_cells uses random_fact directly.
        # usually random_fact is set in CreateAnatomy as: get_val("randomness") / 10 * get_val("stele", "cell_diameter")
        # Let's approximate or just use a standard factor relevant to cell size.
        # If we use pcell (pith cell size):
        r_fact = random_factor / 10 * pcell
        
        # Create cells
        new_cells = create_cells(pith_layer_df, random_fact=r_fact)
        
        # Recenter (R does this to ensure angle 0 is meaningful?)
        # R: new_center <- mean(x[angle==0], y[angle==0])
        # new_cells$x <- new_cells$x - new_center + center
        # This seems to just center the grid if it drifted? Or align 'angle 0' to center?
        # create_cells generates around 'max(radius)'.
        # In create_cells, center is max(all_layers$radius).
        # For pith, center should be roughly 0 relative to itself, but create_cells puts it at 'center'.
        # If I pass pith_layer_df to create_cells, 'center' calculated inside will be max radius of pith.
        # But we want pith at the ACTUAL center of the root.
        
        # Let's see `create_cells` implementation:
        # center = max(all_layers$radius).
        # So it places points around (max_r, max_r) or similar? 
        # No, create_cells uses center = max(all_layers$radius) as origin shift?
        # R code create_cells: center <- max(all_layers$radius). x <- center + radius*cos...
        # So create_cells returns coordinates in a box [0, 2*max_r] roughly.
        
        # We want to shift these NEW cells so their center matches the ROOT center.
        # The new valid center is `center` passed to make_pith.
        # The center of new_cells generated is roughly `new_cells center`.
        
        # Start simple: Calculate centroid of new_cells
        if not new_cells.empty:
            cx = new_cells['x'].mean()
            cy = new_cells['y'].mean()
            
            # Shift to target center
            new_cells['x'] = new_cells['x'] - cx + center
            new_cells['y'] = new_cells['y'] - cy + center
            
            new_cells['id_group'] = 0
            # Update type to "pith" or keep "stele"?
            # R code used name="stele", so type is "stele".
        
        # 3. Finalize
        # Remove old cells that overlap with pith area (safety)
        # R: all_cells[sqrt(...) > pith_size]
        all_cells = all_cells[np.sqrt((all_cells['x'] - center)**2 + (all_cells['y'] - center)**2) > pith_size]
        
        # Merge
        all_cells = pd.concat([new_cells, all_cells], ignore_index=True)
        all_cells['id_cell'] = range(1, len(all_cells) + 1)
        
    return all_cells
