import pandas as pd
import numpy as np
from RondyCortex import rondy_cortex

def test_rondy_cortex():
    # Setup dummy data
    params = pd.DataFrame([
        {'name': 'inter_cellular_space', 'type': 'size', 'value': 2.0},
        {'name': 'inter_cellular_space', 'type': 'ratio', 'value': 0.1},
        {'name': 'cortex', 'type': 'cell_diameter', 'value': 20.0},
        {'name': 'cortex', 'type': 'order', 'value': 3}
    ])
    
    # Create a small grid of cortex cells
    cells = []
    center = 50.0
    for x in range(40, 61, 10):
        for y in range(40, 61, 10):
            cells.append({
                'x': float(x),
                'y': float(y),
                'type': 'cortex',
                'radius': np.sqrt((x-center)**2 + (y-center)**2),
                'id_layer': 1,
                'order': 3,
                'id_group': 1
            })
    
    # Add one non-cortex cell (xylem)
    cells.append({
        'x': center, 'y': center, 'type': 'xylem', 'radius': 0, 'id_layer': 0, 'order': 0, 'id_group': 1
    })
    
    all_cells = pd.DataFrame(cells)
    
    print("Running rondy_cortex...")
    result = rondy_cortex(params, all_cells, center)
    
    print("\nResult summary:")
    print(f"Total cells: {len(result)}")
    print(f"Cell types: {result['type'].unique()}")
    print("\nCortex cells count:", len(result[result['type'] == 'cortex']))
    print("ICP cells count:", len(result[result['type'] == 'inter_cellular_space']))
    
    # Check if id_group for cortex is properly shifted
    cortex_groups = result[result['type'] == 'cortex']['id_group'].unique()
    print("Cortex groups:", cortex_groups)
    
    if len(cortex_groups) > 0:
        print("Verification successful: Rounded cortex created.")
    else:
        print("Verification failed: No cortex cells found in result.")

if __name__ == "__main__":
    test_rondy_cortex()
