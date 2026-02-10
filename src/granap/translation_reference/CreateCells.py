import pandas as pd
import numpy as np

def create_cells(all_layers: pd.DataFrame, random_fact: float) -> pd.DataFrame:
    """
    Translates create_cells.R.
    Generates initial cell centers based on layer radial configuration.
    """
    center = all_layers['radius'].max()
    all_cells_list = []
    
    k = 1 # Cell ID counter
    
    for _, row in all_layers.iterrows():
        radius = row['radius']
        n_cell = int(row['n_cell'])
        angle_inc = row['angle_inc']
        layer_name = row['name']
        order = row['order']
        layer_idx = row.name + 1 # 1-based index to match R somewhat, or just unique ID
        
        if angle_inc > 0:
            angles = np.linspace(0, 2*np.pi, n_cell, endpoint=False) + angle_inc
        else:
            angles = np.array([0.0])
            n_cell = 1
        
        if n_cell < 1:
            continue

        # Cell IDs
        ids = range(k, k + n_cell)
        k += n_cell
        
        # Calculate coordinates
        # Helper for randomness
        def get_rand(n, mag):
            return np.random.uniform(-mag, mag, n)
            
        if layer_name == "outside":
            x = center + (radius * np.cos(angles))
            y = center + (radius * np.sin(angles))
        elif layer_name == "stele":
            x = center + (radius * np.cos(angles)) + get_rand(n_cell, random_fact)
            y = center + (radius * np.sin(angles)) + get_rand(n_cell, random_fact)
        elif layer_name == "cortex": 
            x = center + (radius * np.cos(angles)) + get_rand(n_cell, random_fact * 3)
            y = center + (radius * np.sin(angles)) + get_rand(n_cell, random_fact * 3)
        else:
            x = center + (radius * np.cos(angles)) + get_rand(n_cell, random_fact)
            y = center + (radius * np.sin(angles)) + get_rand(n_cell, random_fact)
            
        # Create DataFrame for this layer
        layer_df = pd.DataFrame({
            'angle': angles,
            'radius': radius,
            'x': x,
            'y': y,
            'id_layer': layer_idx, 
            'id_cell': ids,
            'type': layer_name,
            'order': order
        })
        all_cells_list.append(layer_df)

    if all_cells_list:
        all_cells = pd.concat(all_cells_list, ignore_index=True)
    else:
        all_cells = pd.DataFrame(columns=['angle', 'radius', 'x', 'y', 'id_layer', 'id_cell', 'type', 'order'])
        
    return all_cells
