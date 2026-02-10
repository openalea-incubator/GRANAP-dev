import xml.etree.ElementTree as ET
import pandas as pd

def write_anatomy_xml(sim, path=None):
    # 1. Define the group mapping (move this to a config file/class constant later)
    cell_groups = {
        "exodermis": 1, "epidermis": 2, "endodermis": 3, "passage_cell": 3,
        "cortex": 4, "stele": 5, "xylem": 13, "pericycle": 16,
        "companion_cell": 12, "phloem": 11, "inter_cellular_space": 4,
        "aerenchyma": 4, "cambium": 11, "metaxylem": 13
    }

    # Root element
    root = ET.Element("granardata")
    
    # 2. Metadata Section
    metadata = ET.SubElement(root, "metadata")
    params_node = ET.SubElement(metadata, "parameters")
    # Loop through sim['output'] if it exists to add parameter tags
    
    # 3. Cells Section
    cells_root = ET.SubElement(root, "cells", count=str(len(sim['cells'])))
    # Group nodes by id_cell and iterate
    for cid, group in sim['nodes'].groupby('id_cell'):
        cell_type = group['type'].iloc[0]
        group_id = cell_groups.get(cell_type, 0)
        
        cell_el = ET.SubElement(cells_root, "cell", 
                               id=str(cid), # Python is already 0-indexed if you started from 0
                               group=str(group_id), 
                               truncated="false")
        walls_el = ET.SubElement(cell_el, "walls")
        
        # Add walls inside cell
        for wid in group['id_wall'].unique():
            ET.SubElement(walls_el, "wall", id=str(wid))

    # 4. Walls Section
    # Extract unique walls from the 'vertex' data
    walls_data = sim['walls'] # Computed by your Python version of vertex()
    walls_root = ET.SubElement(root, "walls", count=str(len(walls_data)))
    
    for _, w in walls_data.iterrows():
        wall_el = ET.SubElement(walls_root, "wall", 
                               id=str(int(w['id_wall'])), 
                               group="0", edgewall="false")
        points_el = ET.SubElement(wall_el, "points")
        # Point 1
        ET.SubElement(points_el, "point", x=f"{w.x1:.6f}", y=f"{w.y1:.6f}")
        # Point 2
        ET.SubElement(points_el, "point", x=f"{w.x2:.6f}", y=f"{w.y2:.6f}")

    # 5. Definitions (Groups)
    groups_el = ET.SubElement(root, "groups")
    # ... add cellgroups and wallgroups definitions here ...

    # Generate the string
    tree = ET.ElementTree(root)
    if path:
        # Use indent for "pretty printing" if using Python 3.9+
        ET.indent(tree, space="\t", level=0)
        tree.write(path, encoding="utf-8", xml_declaration=True)
        return True
    
    return ET.tostring(root, encoding='unicode')
