import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def vascular(all_cells: pd.DataFrame, params: pd.DataFrame, layers: pd.DataFrame, center: float) -> pd.DataFrame:
    """
    Translates vascular.R. 
    Handles Monocot vs Dicot vascular patterning and overrides stele cells 
    with Xylem, Phloem, and Companion cells.
    Refactored to remove GeoPandas dependency.
    """
    
    # 1. Parameter Extraction
    def get_param(name, p_type):
        val = params[(params['name'] == name) & (params['type'] == p_type)]['value']
        return val.values[0] if not val.empty else 0
    
    n_xylem_files = get_param("xylem", "n_files")
    proto_meta_ratio = get_param("xylem", "ratio")
    n_proto_xylem = int(np.round(n_xylem_files * proto_meta_ratio))
    plant_type = int(get_param("planttype", "value")) # 1: Monocot, 2: Dicot
    
    # Keep track of max group ID to avoid overlaps
    cortex_cells = all_cells[all_cells['type'] == "cortex"]
    k_max_cortex = cortex_cells['id_group'].max() if not cortex_cells.empty else 0
    
    # Utility for circular frontier points
    def get_circle_points(cx, cy, diameter, n_points=16):
        theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        r = (diameter * 0.8) / 2
        return cx + r * np.cos(theta), cy + r * np.sin(theta)

    # ==========================================
    # DICOT LOGIC (Plant Type 2)
    # ==========================================
    if plant_type == 2:
        max_stele_r = all_cells[all_cells['type'] == "stele"]['radius'].max()
        stele_diam = layers[layers['name'] == "stele"]['cell_diameter'].values[0]
        max_xyl_size = get_param("xylem", "max_size")
        
        # Fit linear model for xylem size gradient
        # R logic: lm(d ~ r)
        X = np.array([0, max_stele_r]).reshape(-1, 1)
        y = np.array([max_xyl_size, stele_diam])
        model = LinearRegression().fit(X, y)
        
        xyl_r, xyl_d = [0.0], [max_xyl_size]
        curr_r = 0.0
        while True:
            # Predict next diameter based on current radius
            next_r = curr_r + xyl_d[-1]
            next_d = model.predict([[next_r]])[0]
            
            if next_r + (next_d/2) > max_stele_r - (stele_diam/2):
                break
            xyl_r.append(next_r)
            xyl_d.append(next_d)
            curr_r = next_r
            
        # Create xylem poles
        angle_seq = np.linspace(0, 2*np.pi, int(n_xylem_files), endpoint=False)
        xyl_elements = []
        
        for group_id, angle in enumerate(angle_seq, 1):
            for r, d in zip(xyl_r, xyl_d):
                px = center + r * np.cos(angle)
                py = center + r * np.sin(angle)
                
                # Generate circular points for Xylem frontier
                cx, cy = get_circle_points(px, py, d)
                for x, y in zip(cx, cy):
                    xyl_elements.append({
                        'angle': angle, 'radius': r, 'x': x, 'y': y,
                        'type': 'xylem', 'id_group': group_id + k_max_cortex, 
                        'id_layer': 20, 'order': 1.5, 'd_bound': d
                    })
        
        # Remove stele cells overlapping with new xylem poles
        new_xyl_df = pd.DataFrame(xyl_elements)
        # Vectorized distance check to remove stele cells
        for _, pole in new_xyl_df.drop_duplicates(['radius', 'angle']).iterrows():
            dist_sq = (all_cells.x - pole.x)**2 + (all_cells.y - pole.y)**2
            all_cells = all_cells[~((all_cells.type == "stele") & (dist_sq < (pole.d_bound/1.5)**2))]
            
        all_cells = pd.concat([all_cells, new_xyl_df], ignore_index=True)
        
        # Phloem logic (Between Xylem poles)
        phl_r = max_stele_r - (stele_diam/2)
        angle_seq_ph = angle_seq + (angle_seq[1] - angle_seq[0])/2 if len(angle_seq)>1 else [np.pi]
        
        for angle in angle_seq_ph:
            px_ph, py_ph = center + phl_r * np.cos(angle), center + phl_r * np.sin(angle)
            # Find nearest stele cells and reassign
            for cell_type, count in [("phloem", 1), ("companion_cell", 2), ("cambium", 6)]:
                for _ in range(count):
                    stele_mask = all_cells['type'] == 'stele'
                    dists = (all_cells.loc[stele_mask, 'x'] - px_ph)**2 + (all_cells.loc[stele_mask, 'y'] - py_ph)**2
                    if not dists.empty:
                        idx = dists.idxmin()
                        all_cells.at[idx, 'type'] = cell_type

    # ==========================================
    # MONOCOT LOGIC (Plant Type 1)
    # ==========================================
    elif plant_type == 1:
        stele_diam = get_param("stele", "cell_diameter")
        max_xyl_size = get_param("xylem", "max_size")
        max_stele_r = all_cells[all_cells['type'] == "stele"]['radius'].max()
        
        r_xyl = 0 if n_xylem_files == 1 else (max_stele_r - stele_diam*1.5 - max_xyl_size/2)
        angle_seq = np.linspace(0, 2*np.pi, int(n_xylem_files), endpoint=False)
        
        xyl_elements = []
        for i, angle in enumerate(angle_seq, 1):
            px = center + r_xyl * np.cos(angle)
            py = center + r_xyl * np.sin(angle)
            cx, cy = get_circle_points(px, py, max_xyl_size)
            for x, y in zip(cx, cy):
                xyl_elements.append({
                    'angle': angle, 'radius': r_xyl, 'x': x, 'y': y,
                    'type': 'xylem', 'id_group': i + k_max_cortex, 
                    'id_layer': 20, 'order': 1.5, 'd_bound': max_xyl_size
                })
        
        # Filter out stele cells inside metaxylem
        new_xyl_df = pd.DataFrame(xyl_elements)
        for _, pole in new_xyl_df.drop_duplicates(['radius', 'angle']).iterrows():
            dist_sq = (all_cells.x - pole.x)**2 + (all_cells.y - pole.y)**2
            all_cells = all_cells[~((all_cells.type == "stele") & (dist_sq < (pole.d_bound/1.5)**2))]

        all_cells = pd.concat([all_cells[all_cells.type != "xylem"], new_xyl_df], ignore_index=True)

        # Protoxylem and Phloem placement (Rim logic)
        for t_name, n_vessels, types_to_assign in [
            ("proto", n_proto_xylem, ["xylem"]),
            ("phlo", n_proto_xylem, ["phloem", "companion_cell", "companion_cell"])
        ]:
            offset = (np.pi/n_vessels) if t_name == "phlo" else 0
            angles = np.linspace(0, 2*np.pi, n_vessels, endpoint=False) + offset
            for ang in angles:
                px = center + (max_stele_r - stele_diam/2) * np.cos(ang)
                py = center + (max_stele_r - stele_diam/2) * np.sin(ang)
                
                for sub_type in types_to_assign:
                    stele_mask = all_cells['type'] == 'stele'
                    dists = (all_cells.loc[stele_mask, 'x'] - px)**2 + (all_cells.loc[stele_mask, 'y'] - py)**2
                    if not dists.empty:
                        all_cells.at[dists.idxmin(), 'type'] = sub_type

    # Final Cleanup
    all_cells['id_cell'] = range(1, len(all_cells) + 1)
    
    return all_cells # Return DataFrame, not GeoDataFrame
