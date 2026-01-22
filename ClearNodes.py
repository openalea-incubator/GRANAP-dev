import pandas as pd
from shapely.geometry import Polygon, LineString, Point

def clear_nodes(rs1: pd.DataFrame) -> pd.DataFrame:
    """
    Translates clear_nodes.R.
    Removes cells with 0 area, but preserves cells with minimal geometry.
    """
    rs1 = rs1.dropna(subset=['x', 'y'])
    valid_ids = []
    removed_by_type = {}
    
    for i in rs1['id_cell'].unique():
        subset = rs1[rs1['id_cell'] == i]
        cell_type = subset['type'].iloc[0] if len(subset) > 0 else 'unknown'
        
        # Allow cells with at least 2 vertices (can form a line)
        # Only remove cells with < 2 vertices or zero area
        if len(subset) >= 2:
            try:
                coords = list(zip(subset['x'], subset['y']))
                
                if len(coords) == 2:
                    # Line segment - keep it
                    valid_ids.append(i)
                elif len(coords) >= 3:
                    # Polygon - check area
                    poly = Polygon(coords)
                    area = poly.area
                    # Keep even very small areas (> 1e-10)
                    if area > 1e-10:
                        valid_ids.append(i)
                    else:
                        removed_by_type[cell_type] = removed_by_type.get(cell_type, 0) + 1
            except Exception as e:
                # If polygon creation fails, still try to keep the cell
                valid_ids.append(i)
        else:
            # Only 1 or 0 vertices - truly invalid
            removed_by_type[cell_type] = removed_by_type.get(cell_type, 0) + 1
    
    if removed_by_type:
        print(f"DEBUG clear_nodes: Removed cells by type: {removed_by_type}")
                
    return rs1[rs1['id_cell'].isin(valid_ids)]
