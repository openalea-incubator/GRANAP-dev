
import sys
import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Add root folder to sys.path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from CreateAnatomy import create_anatomy

def verify_translation():
    print("--- Verifying Translation & NetworkX Integration ---")
    
    # Mock Parameters
    # Create a DataFrame typical of the input XML
    params_data = [
        {"name": "planttype", "type": "value", "value": 1}, # Monocot
        {"name": "randomness", "type": "value", "value": 1.0},
        {"name": "stele", "type": "cell_diameter", "value": 5.0},
        {"name": "stele", "type": "layer_diameter", "value": 50.0},
        {"name": "xylem", "type": "n_cells", "value": 4},
        {"name": "xylem", "type": "cell_diameter", "value": 10.0},
        {"name": "xylem", "type": "order", "value": 4},
        {"name": "xylem", "type": "max_size", "value": 12.0},
        {"name": "xylem", "type": "ratio", "value": 1.0},
        {"name": "endodermis", "type": "cell_diameter", "value": 8.0},
        {"name": "endodermis", "type": "n_layers", "value": 1},
        {"name": "endodermis", "type": "order", "value": 3},
        {"name": "cortex", "type": "cell_diameter", "value": 15.0},
        {"name": "cortex", "type": "n_layers", "value": 2},
        {"name": "cortex", "type": "order", "value": 2},
        {"name": "epidermis", "type": "cell_diameter", "value": 10.0},
        {"name": "epidermis", "type": "n_layers", "value": 1},
        {"name": "epidermis", "type": "order", "value": 1},
        # Pericycle
        {"name": "pericycle", "type": "cell_diameter", "value": 6.0},
        {"name": "pericyle", "type": "n_layers", "value": 1}, # Typo handling potentially needed?
        # Aerenchyma
        {"name": "aerenchyma", "type": "proportion", "value": 0.0}, # Keep simple for first pass
        # Intercellular space
        {"name": "inter_cellular_space", "type": "value", "value": 0},
        {"name": "inter_cellular_space", "type": "size", "value": 0},
        # Secondary growth
        {"name": "secondarygrowth", "type": "value", "value": 0},
        # Pith 
        {"name": "pith", "type": "layer_diameter", "value": 0},
         # Hair
        {"name": "hair", "type": "n_files", "value": 0},
    ]
    
    params = pd.DataFrame(params_data)
    
    # Run simulation
    try:
        print("Running create_anatomy...")
        G = create_anatomy(parameters=params, verbatim=True)
        
        if G is None:
            print("FAILED: create_anatomy returned None.")
            return
            
        print("Successfully generated output object.")
        
        # Check type
        if isinstance(G, nx.Graph):
            print("PASSED: Output is a NetworkX Graph.")
        else:
            print(f"FAILED: Output is {type(G)}, expected networkx.Graph.")
            return
            
        # Check content
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        print(f"Graph Construction Stats:")
        print(f" - Nodes: {n_nodes}")
        print(f" - Edges: {n_edges}")
        
        if n_nodes > 0:
            print("PASSED: Graph has nodes.")
        else:
            print("FAILED: Graph is empty.")
            
        print("\nNode Attributes Check (first node):")
        if n_nodes > 0:
            node_0 = list(G.nodes(data=True))[0]
            print(node_0)
            
        print("\nEdge Attributes Check (first edge):")
        if n_edges > 0:
            edge_0 = list(G.edges(data=True))[0]
            print(edge_0)
            
    except Exception as e:
        print(f"FAILED with Exception: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        import shapely
    except ImportError:
        print("Dependency Error: 'shapely' is not installed. Please run: pip install shapely")
        sys.exit(1)
        
    try:
        import networkx
    except ImportError:
        print("Dependency Error: 'networkx' is not installed. Please run: pip install networkx")
        sys.exit(1)
        
    verify_translation()
