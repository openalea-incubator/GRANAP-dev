import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np

class AnatomyMapper:
    """Class to handle the visualization of a RootAnatomy object."""
    
    def __init__(self, root_instance):
        self.root = root_instance
        # Default color palette for root tissues
        self.colors = {
            'epidermis': '#f1c40f', # Yellow
            'cortex': '#2ecc71',    # Green
            'stele': '#e67e22',     # Orange
            'xylem': '#3498db',     # Blue
            'phloem': '#9b59b6',    # Purple
            'default': '#bdc3c7'    # Grey
        }

    def plot(self, color_by='type', figsize=(10, 10), show_nodes=False):
        fig, ax = plt.subplots(figsize=figsize)
        
        polygons = []
        facecolors = []

        for node_id, data in self.root.graph.nodes(data=True):
            if 'polygon' in data:
                poly = data['polygon']
                
                # --- FIX START ---
                # Check if it's a Shapely polygon and extract coordinates
                if hasattr(poly, 'exterior'):  # It's a Shapely Polygon
                    coords = np.array(poly.exterior.coords)
                elif isinstance(poly, (list, np.ndarray)): # It's already coords
                    coords = np.array(poly)
                else:
                    continue # Skip if format is unknown
                
                polygons.append(coords)
                # --- FIX END ---
                
                cell_type = data.get(color_by, 'default')
                facecolors.append(self.colors.get(cell_type, self.colors['default']))

        if not polygons:
            print("Warning: No polygons found in the graph to plot.")
            return fig, ax

        coll = PolyCollection(polygons, facecolors=facecolors, edgecolors='black', linewidths=0.5)
        ax.add_collection(coll)

        # 3. Optional: Plot cell centers (nodes)
        if show_nodes:
            pos = [d['pos'] for n, d in self.root.graph.nodes(data=True)]
            if pos:
                x, y = zip(*pos)
                ax.scatter(x, y, s=1, color='red', alpha=0.5)

        # 4. Final adjustments
        ax.autoscale_view()
        ax.set_aspect('equal')
        ax.set_title(f"Root Anatomy Visualization ({len(polygons)} cells)")
        plt.axis('off')
        return fig, ax
