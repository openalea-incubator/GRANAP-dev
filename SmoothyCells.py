import pandas as pd
from Concavety import concavety

def smoothy_cells(data: pd.DataFrame) -> pd.DataFrame:
    """
    Translates smoothy_cells.R.
    Iterates over groups and smooths them using concavety.
    Removes noise points while preserving all tissue types.
    """
    
    smoothed_list = []
    
    if 'id_group' in data.columns:
        groups = data['id_group'].unique()
        
        for grp in groups:
            subset = data[data['id_group'] == grp]
            if not subset.empty:
                # Group 0 contains standard individual cells - do not merge them.
                # Other groups (e.g. Xylem vessels composed of multiple points) should be merged.
                should_merge = (grp != 0)
                processed = concavety(subset, merge=should_merge)
                smoothed_list.append(processed)
    else:
        return data

    if not smoothed_list:
        return pd.DataFrame(columns=data.columns)
        
    data_smooth = pd.concat(smoothed_list, ignore_index=True)
    
    # Filter noise points more carefully to preserve all tissue types
    # Only remove points that appear in < 3 segments AND are not critical tissue types
    # Critical types: all non-cortex, non-outside types
    critical_types = ['xylem', 'metaxylem', 'protoxylem', 'phloem', 'companion_cell', 
                      'stele', 'pericycle', 'endodermis', 'epidermis', 'exodermis',
                      'innercortex', 'outercortex', 'inter_cellular_space', 'pith']
    
    counts = data_smooth['id_point'].value_counts()
    valid_points = counts[counts >= 3].index
    
    # Keep points if: (1) they appear >= 3 times, OR (2) they belong to critical tissue types
    mask = (data_smooth['id_point'].isin(valid_points)) | (data_smooth['type'].isin(critical_types))
    data_smooth = data_smooth[mask]
    
    return data_smooth
