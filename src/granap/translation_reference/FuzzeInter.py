import pandas as pd
import numpy as np
from Concavety import concavety

def fuzze_inter(rs1: pd.DataFrame) -> pd.DataFrame:
    """
    Translates fuzze_inter.R.
    Merges connected Intercellular Spaces.
    """
    # Filter spaces
    space = rs1[rs1['type'] == "inter_cellular_space"].copy()
    if space.empty:
        return rs1
        
    # Find points shared by > 1 space cell
    counts = space['id_point'].value_counts()
    double_points = counts[counts > 1].index
    
    if len(double_points) == 0:
        return rs1
        
    to_correct = space[space['id_point'].isin(double_points)]['id_cell'].unique()
    
    if len(to_correct) == 0:
        return rs1
        
    done = set()
    processed_spaces = []
    
    # Map for graph traversal (BFS/DFS to find connected components)
    # Build adjacency list: cell_id -> set of connected cell_ids
    # Connect if share a point in double_points
    
    # point -> list of cells
    point_to_cells = space[space['id_point'].isin(double_points)].groupby('id_point')['id_cell'].apply(set).to_dict()
    
    cell_to_cells = {}
    for pt, cell_ids in point_to_cells.items():
        ids_list = list(cell_ids)
        for i in range(len(ids_list)):
            c1 = ids_list[i]
            if c1 not in cell_to_cells: cell_to_cells[c1] = set()
            for j in range(i+1, len(ids_list)):
                c2 = ids_list[j]
                if c2 not in cell_to_cells: cell_to_cells[c2] = set()
                cell_to_cells[c1].add(c2)
                cell_to_cells[c2].add(c1)
                
    # Iterate through to_correct cells
    all_merged_ids = set()
    
    for start_node in to_correct:
        if start_node in done: continue
        
        # BFS to find component
        component = set([start_node])
        queue = [start_node]
        
        while queue:
            node = queue.pop(0)
            neighbors = cell_to_cells.get(node, set())
            for nei in neighbors:
                if nei not in component:
                    component.add(nei)
                    queue.append(nei)
                    
        component_list = list(component)
        done.update(component_list)
        all_merged_ids.update(component_list)
        
        # Merge this component
        subset = space[space['id_cell'].isin(component_list)]
        if not subset.empty:
             merged = concavety(subset)
             processed_spaces.append(merged)
             
    # Output construction
    # Keep non-merged cells from rs1
    mask_keep = ~rs1['id_cell'].isin(all_merged_ids)
    final_rs1 = rs1[mask_keep]
    
    if processed_spaces:
        merged_df = pd.concat(processed_spaces, ignore_index=True)
        final_rs1 = pd.concat([final_rs1, merged_df], ignore_index=True)
        
    return final_rs1
