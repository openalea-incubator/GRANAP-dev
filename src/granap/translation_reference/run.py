
import pandas as pd
import numpy as nppy
import matplotlib.pyplot as plt
import networkx as nx
from CreateAnatomy import create_anatomy

def test_root_generation():
    print("--- Starting GRANAP Tissue Generation Test ---")

    # 1. Define Parameters (GRANAR Format)
    params_data = [
        {"name": "planttype", "type": "value", "value": 1}, # 1 = Monocot, 2 = Dicot
        {"name": "randomness", "type": "value", "value": 1.0}, # 0 = No randomness, 3 = Maximum randomness
        {"name": "secondarygrowth", "type": "value", "value": 0},
        {"name": "stele", "type": "cell_diameter", "value": 0.0063}, # Cell diameter in millimeters
        {"name": "stele", "type": "layer_diameter", "value": 0.163}, # Layer diameter in millimeters
        {"name": "stele", "type": "order", "value": 1}, # Order of the stele
        {"name": "pericycle", "type": "cell_diameter", "value": 0.009},
        {"name": "pericycle", "type": "n_layers", "value": 1},
        {"name": "pericycle", "type": "order", "value": 2},
        {"name": "endodermis", "type": "cell_diameter", "value": 0.015},
        {"name": "endodermis", "type": "n_layers", "value": 1},
        {"name": "endodermis", "type": "order", "value": 3},
        {"name": "innercortex", "type": "cell_diameter", "value": 0.017},
        {"name": "innercortex", "type": "n_layers", "value": 1},
        {"name": "innercortex", "type": "order", "value": 4},
        {"name": "cortex", "type": "cell_diameter", "value": 0.025},
        {"name": "cortex", "type": "n_layers", "value": 1},
        {"name": "cortex", "type": "order", "value": 5},
        {"name": "outercortex", "type": "cell_diameter", "value": 0.01},
        {"name": "outercortex", "type": "n_layers", "value": 1},
        {"name": "outercortex", "type": "order", "value": 6},
        {"name": "exodermis", "type": "cell_diameter", "value": 0.018},
        {"name": "exodermis", "type": "n_layers", "value": 1},
        {"name": "exodermis", "type": "order", "value": 7},
        {"name": "epidermis", "type": "cell_diameter", "value": 0.006},
        {"name": "epidermis", "type": "n_layers", "value": 1},
        {"name": "epidermis", "type": "order", "value": 8},
        {"name": "xylem", "type": "n_files", "value": 4}, # Number of files
        {"name": "xylem", "type": "max_size", "value": 0.035}, # Maximum size in millimeters
        {"name": "xylem", "type": "order", "value": 1.5}, # Order of the xylem
        {"name": "xylem", "type": "ratio", "value": 3.3}, # Ratio of the xylem
        {"name": "phloem", "type": "n_files", "value": 3}, # Number of files
        {"name": "phloem", "type": "max_size", "value": 0.01}, # Maximum size in millimeters
        {"name": "phloem", "type": "n_layers", "value": 0}, # Ratio of the phloem
        {"name": "aerenchyma", "type": "proportion", "value": 0.5},
        {"name": "aerenchyma", "type": "n_files", "value": 2},
        {"name": "aerenchyma", "type": "type", "value": 1}, # 1 = Strip, 2 = Fused
        {"name": "inter_cellular_space", "type": "ratio", "value": 0},
        {"name": "inter_cellular_space", "type": "size", "value": 0},
        {"name": "hair", "type": "n_files", "value": 5},
        {"name": "hair", "type": "length", "value": 0.07},
        {"name": "pith", "type": "layer_diameter", "value": 0.002},
        {"name": "pith", "type": "cell_diameter", "value": 0.05}
    ]
    params = pd.DataFrame(params_data)

    # 2. Initialize simulation
    print("Step 1: Generating anatomy graph...")
    G = create_anatomy(parameters=params, verbatim=True)
    
    if G is None:
        print("Error: Graph generation failed.")
        return

    # 3. Verification Check
    print(f"\nResults:")
    print(f"- Total Nodes: {G.number_of_nodes()}")
    print(f"- Total Edges: {G.number_of_edges()}")

    # Extract Cell Surface Areas to DataFrame
    print("\nStep 1.5: Cell Surface Areas DataFrame")
    data_list = []
    # Filter only actual cells for this stats dataframe
    for n, d in G.nodes(data=True):
        if d.get('type') not in ['wall', 'junction', 'wall_structure', 'membrane']:
             # Use safe get for coords
            x = d.get('x', d.get('center_x', 0))
            y = d.get('y', d.get('center_y', 0))
            
            data_list.append({
                'id_cell': n,
                'type': d.get('type'),
                'area': d.get('area'),
                'x': x,
                'y': y
            })
    df_areas = pd.DataFrame(data_list)
    print(df_areas.head(10))
    if not df_areas.empty:
        print(f"\nMean Area per Type:\n{df_areas.groupby('type')['area'].mean()}")

    # 4. VISUAL TESTING
    print("\nStep 2: Plotting the graph...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Get positions (Handle both schemes)
    pos = {}
    for n, d in G.nodes(data=True):
        x = d.get('x', d.get('center_x', 0))
        y = d.get('y', d.get('center_y', 0))
        pos[n] = (x, y)
    
    
    # Get colors based on type
    unique_types = set(d.get('type', 'unknown') for n, d in G.nodes(data=True))

    # Fixed Color Map - Comprehensive tissue types
    color_map = {
        # Central tissues
        'stele': '#dfc27d',
        'parenchyma': '#f5deb3',  # Wheat color for parenchyma
        'pericycle': '#f6e8c3',
        
        # Vascular
        'xylem': '#d73027',
        'metaxylem': '#a50026',
        'protoxylem': '#fc8d59',
        'phloem': '#c51b7d',
        'companion_cell': '#f1b6da',
        
        # Endodermis
        'endodermis': '#bf812d',
        
        # Cortex layers (gradient from inner to outer)
        'innercortex': '#66c2a4',    # Teal-green
        'cortex': '#80cdc1',          # Light teal
        'outercortex': '#b2e2e2',    # Pale teal
        
        # Outer layers
        'exodermis': '#8c510a',       # Brown
        'epidermis': '#01665e',       # Dark teal
        
        # Special structures
        'inter_cellular_space': '#e0e0e0',
        'aerenchyma': '#a6cee3',
        'outside': '#ffffff',
        'pith': '#dfc27d',
        
        # Graph structure nodes
        'wall': 'black',
        'junction': 'blue'
    }
    
    # Fallback for any unknown types
    for t in unique_types:
        if t not in color_map:
            # Use hash to get consistent color for unknown types
            import hashlib
            hash_object = hashlib.md5(str(t).encode())
            hex_color = '#' + hash_object.hexdigest()[:6]
            color_map[t] = hex_color
    
    
    node_colors = [color_map.get(d.get('type', 'unknown'), 'gray') for n, d in G.nodes(data=True)]
    
    # Determine node sizes
    node_sizes = []
    for n, d in G.nodes(data=True):
        ctype = d.get('type')
        if ctype == 'junction':
            node_sizes.append(5)
        elif ctype == 'wall':
            node_sizes.append(5)
        else:
            node_sizes.append(20) # Cells big

    # --- PLOT 1: ANATOMY (Polygons) ---
    ax1.set_title("Root Anatomy (Cross Section)")
    ax1.set_aspect('equal')
    
    from matplotlib.patches import Polygon as MplPolygon
    from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon
    
    for n, d in G.nodes(data=True):
        # Only plot cells with geometry
        if 'geometry' in d and d['geometry'] is not None:
            geom = d['geometry']
            color = color_map.get(d['type'], 'gray')
            
            # Helper to add patch
            def add_poly_patch(p_geom, ax):
                x, y = p_geom.exterior.xy
                coords = list(zip(x, y))
                poly_patch = MplPolygon(coords, facecolor=color, edgecolor='black', linewidth=0.5, alpha=0.9)
                ax.add_patch(poly_patch)

            if isinstance(geom, ShapelyPolygon):
                add_poly_patch(geom, ax1)
            elif isinstance(geom, MultiPolygon):
                for poly in geom.geoms:
                    add_poly_patch(poly, ax1)
    
    # Auto-scale anatomy plot
    all_points = []
    for n, d in G.nodes(data=True):
        if 'geometry' in d and d['geometry'] is not None:
             geom = d['geometry']
             if isinstance(geom, ShapelyPolygon):
                  all_points.extend(geom.envelope.exterior.coords)
             elif isinstance(geom, MultiPolygon):
                  all_points.extend(geom.envelope.exterior.coords) # Approximate
                  
    if len(all_points) > 0:
        pts = np.array(all_points)
        ax1.set_xlim(pts[:,0].min(), pts[:,0].max())
        ax1.set_ylim(pts[:,1].min(), pts[:,1].max())

    # Add Legend
    from matplotlib.patches import Patch
    # Filter types for legend (only tissue types)
    legend_types = [t for t in unique_types if t not in ['wall', 'junction']]
    legend_elements = [Patch(facecolor=color_map[t], edgecolor='black', label=t) for t in sorted(legend_types)]
    ax1.legend(handles=legend_elements, loc='upper right', title="Tissue Types", fontsize='small')

    # --- PLOT 2: NETWORK TOPOLOGY (Nodes + Edges) ---
    ax2.set_title("Network Topology (Graph)")
    
    # Filter Edges: Only draw 'wall_structure' edges (Wall <-> Junction)
    wall_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'wall_structure']
    
    # Draw Edges
    nx.draw_networkx_edges(G, pos, ax=ax2, edgelist=wall_edges, edge_color='black', width=0.5, alpha=0.8)
    
    # Draw Nodes: Filter to show only Wall/Junction nodes if we are focusing on walls?
    # Or keep all nodes but only wall edges?
    # User said "just keep the edge between wall".
    # I'll keep nodes for now but maybe make non-wall nodes smaller or invisible?
    # For now, drawing all nodes as configured previously (Junction/Wall small, Cell big).
    nx.draw_networkx_nodes(G, pos, ax=ax2, node_size=node_sizes, node_color=node_colors)
    
    ax2.set_aspect('equal')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_root_generation()
