import numpy as np
import pyvista as pv

def export_to_3d(sim, length=100, num_sections=5):
    """
    Extrudes the 2D anatomy into a 3D PyVista mesh.
    """
    nodes_2d = sim['nodes']
    all_meshes = []
    
    for cell_id in nodes_2d['id_cell'].unique():
        cell_data = nodes_2d[nodes_2d['id_cell'] == cell_id]
        points_2d = cell_data[['x', 'y']].values
        
        # Create a 3D polygon by extruding the 2D points along Z
        # This creates a "prism" for each cell
        poly = pv.PolyData(np.column_stack([points_2d, np.zeros(len(points_2d))]))
        extruded = poly.extrude([0, 0, length], capping=True)
        
        # Add metadata (cell type, etc.)
        extruded.cell_data['type'] = cell_data['type'].iloc[0]
        all_meshes.append(extruded)
        
    return pv.MultiBlock(all_meshes)

# Usage:
# mesh = export_to_3d(sim)
# mesh.plot(scalars='type')
