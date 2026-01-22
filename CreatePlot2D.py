import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PolyCollection, LineCollection
import matplotlib.cm as cm
import numpy as np

def plot_anatomy(sim, col='type', leg=True, apo_bar=0, phi_thck=0, ax=None):
    """
    Python version of plot_anatomy.R using Matplotlib.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    nodes = sim['nodes']
    
    # 1. Logic for Polygon Filling (The default view)
    if col != 'segment':
        # Create a list of Nx2 arrays (coordinates for each cell)
        cells = []
        colors = []
        
        # Mapping types to colors if col == 'type'
        # You can define a standard dictionary for plant tissues
        type_colors = {
            'epidermis': '#FFD700', 'cortex': '#90EE90', 'endodermis': '#FF4500',
            'stele': '#DEB887', 'xylem': '#1E90FF', 'phloem': '#DA70D6',
            'aerenchyma': '#FFFFFF', 'inter_cellular_space': '#000000'
        }

        unique_cells = nodes['id_cell'].unique()
        
        polygons = []
        cell_values = []
        
        for cid in unique_cells:
            cell_data = nodes[nodes['id_cell'] == cid]
            polygons.append(cell_data[['x', 'y']].values)
            
            val = cell_data[col].iloc[0]
            if col == 'type':
                cell_values.append(type_colors.get(val, '#808080'))
            else:
                cell_values.append(val)

        # Use PolyCollection for performance
        if col == 'type':
            pc = PolyCollection(polygons, facecolors=cell_values, edgecolors='white', linewidths=0.5)
        else:
            pc = PolyCollection(polygons, array=np.array(cell_values), cmap='viridis', edgecolors='white', linewidths=0.1)
            if leg: plt.colorbar(pc, ax=ax, label=col)
            
        ax.add_collection(pc)
    
    # 2. Logic for 'segment' view and hydrophobic barriers (Casparian Strips)
    else:
        # Drawing the skeleton (walls)
        segments = nodes[['x1', 'y1', 'x2', 'y2']].values.reshape(-1, 2, 2)
        lc = LineCollection(segments, colors='black', linewidths=0.5)
        ax.add_collection(lc)

        # Casparian strip / Suberin logic
        if apo_bar > 0:
            # Endodermis Suberization (Full wall)
            if apo_bar in [2, 4]:
                endo_walls = nodes[nodes['type'] == 'endodermis'][['x1', 'y1', 'x2', 'y2']].values.reshape(-1, 2, 2)
                ax.add_collection(LineCollection(endo_walls, colors='red', linewidths=2))
            
            # Endodermis Casparian Strips (Points on radial walls)
            if apo_bar in [1, 3]:
                # In Python, we calculate the midpoints
                endo = nodes[nodes['type'] == 'endodermis'].copy()
                endo['x_mid'] = (endo['x1'] + endo['x2']) / 2
                endo['y_mid'] = (endo['y1'] + endo['y2']) / 2
                # Note: The 'd' logic in R filters radial vs tangential walls
                # You'd implement the angle math here
                ax.scatter(endo['x_mid'], endo['y_mid'], color='red', s=10, zorder=3)

    # Aesthetics
    ax.set_aspect('equal')
    ax.axis('off')
    if not leg and col == 'type':
        # Custom legend implementation for categorical types would go here
        pass
    
    plt.tight_layout()
    return ax
