import pandas as pd
import numpy as np

def layer_info(layers: pd.DataFrame, params: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the radial position (radius), perimeter, number of cells, 
    and angular increment for each layer.
    """
    
    all_layers = layers.loc[layers.index.repeat(layers['n_layers'].astype(int))].reset_index(drop=True)
    
    # Initialize columns
    all_layers['radius'] = all_layers['cell_diameter'] / 2
    all_layers['perim'] = all_layers['radius'] * 2 * np.pi
    all_layers['n_cell'] = 1
    all_layers['angle_inc'] = 0.0
    all_layers.loc[0, 'radius'] = 0

    for i in range(1, len(all_layers)):
        prev_r = all_layers.loc[i-1, 'radius'] # Previous layer radius
        prev_d = all_layers.loc[i-1, 'cell_diameter'] # Previous layer cell diameter
        curr_d = all_layers.loc[i, 'cell_diameter'] # Current layer cell diameter

        new_radius = prev_r + (prev_d / 2) + (curr_d / 2) # New radius calculation

        all_layers.loc[i, 'radius'] = new_radius
        # Update perimeter based on new radius
        all_layers.loc[i, 'perim'] = new_radius * 2 * np.pi

        n_cell = round(all_layers.loc[i, 'perim'] / curr_d)
        all_layers.loc[i, 'n_cell'] = n_cell
        
        # Update angular increment based on new number of cells
        if n_cell > 0:
            all_layers.loc[i, 'angle_inc'] = (2 * np.pi) / n_cell
        else:
            all_layers.loc[i, 'angle_inc'] = 0

    return all_layers
