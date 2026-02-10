import pandas as pd
import numpy as np

def pack_xylem(all_cells: pd.DataFrame, params: pd.DataFrame, center: float) -> pd.DataFrame:
    """
    Translates pack_xylem.R.
    Pack xylem vessels and parenchyma in the stele.
    Uses a simplified strictly spiral packing algorithm since packcircles is not available.
    """
    
    # helper
    def get_val(name, ptype):
        v = params[(params['name'] == name) & (params['type'] == ptype)]['value']
        return v.iloc[0] if not v.empty else 0

    cortex_cells = all_cells[all_cells['type'] == "cortex"]
    k_max_cortex = cortex_cells['id_group'].max() if not cortex_cells.empty else 0
    
    nX = int(get_val("xylem", "n_cells"))
    xylem_diam = get_val("xylem", "cell_diameter")
    xylarea = nX * np.pi * (xylem_diam/2)**2
    
    stele_layer_diam = get_val("stele", "layer_diameter")
    stele_cell_diam = get_val("stele", "cell_diameter")
    stele_cell_area = np.pi * (stele_cell_diam/2)**2
    
    max_xyl_diam = get_val("xylem", "max_size")
    max_xyl_area = np.pi * (max_xyl_diam/2)**2
    
    # Calculate number of parenchyma cells
    stele_tot_area = np.pi * (stele_layer_diam/2)**2
    nparenchyma = (stele_tot_area - xylarea) / stele_cell_area
    
    # Adjust count similar to R heuristic rules
    if nparenchyma >= 1000:
        nparenchyma *= 0.86
    elif nparenchyma >= 300:
        nparenchyma *= 0.95
    nparenchyma = int(nparenchyma)
        
    stele_sd = get_val("stele", "SD")
    if stele_sd == 0:
        stele_sd = 0.1 * stele_cell_area * 1e6 # Why 1e6 scaling in R? Maybe units? 
        # R: rnorm(..., mean=stele_cell_area*10^6, sd=...)
        # Then later it seems it uses the area directly?
        # "areas" in R circleProgressiveLayout takes sizecol.
        # If R uses 10^6 scaling for randomness, it implies the output is scaled?
        # But later R divides x/1000? 
        # Let's stick to base units (micrometers usually).
        # We will generate diameters/areas in consistent units.
        stele_sd = 0.1 * stele_cell_area

    # Generate Areas
    # Parenchyma: Normal distribution
    # Xylem: Beta distribution (favors larger?)
    # R: rbeta(nX, 2, 4) * max_Xylem_cell_area
    
    paren_areas = np.random.normal(stele_cell_area, stele_sd, nparenchyma)
    # Clip negative areas
    paren_areas = np.maximum(paren_areas, 0.1 * stele_cell_area)
    
    xylem_areas = np.random.beta(2, 4, nX) * max_xyl_area
    
    # Combine
    areas_df = pd.DataFrame({
        'area': np.concatenate([paren_areas, xylem_areas]),
        'type': ['parenchyma']*nparenchyma + ['xylem']*nX
    })
    
    # R logic checks: "Bind xylem and parenchyma in a way that xylem is in the 10% first cells"
    # This implies sorting/shuffling so Xylem usually appears early (center) or late?
    # R: areas <- rbind(areas%>%filter(type == "xylem"), areas%>%filter(type != "xylem"))
    # Then it shuffles the first 10%.
    # If Xylem is at top, it will be in center.
    
    # Sort so Xylem is first (largest cells usually in center for Monocots/Dicots varies, 
    # but here logic forces Xylem to be packed first = center).
    areas_df = pd.concat([areas_df[areas_df['type'] == 'xylem'], areas_df[areas_df['type'] != 'parenchyma']]) # Wait, filter != parenchyma includes xylem?
    # R: rbind(xylem, others)
    xylem_only = areas_df[areas_df['type'] == 'xylem']
    paren_only = areas_df[areas_df['type'] == 'parenchyma']
    areas_df = pd.concat([xylem_only, paren_only], ignore_index=True)
    
    # Shuffle first 10%
    n_top = int(0.1 * len(areas_df))
    if n_top > 0:
        top_slice = areas_df.iloc[:n_top].sample(frac=1)
        bottom_slice = areas_df.iloc[n_top:]
        areas_df = pd.concat([top_slice, bottom_slice], ignore_index=True)
        
    areas_df['radius'] = np.sqrt(areas_df['area'] / np.pi)
    areas_df['id'] = range(len(areas_df))
    
    # Pack Circles (Spiral Layout)
    # Simple algorithm: place one by one spiraling out.
    # r = c * sqrt(n) approach or just place next circle at min distance?
    # A simple effective qualitative approximation for "Progressive Layout":
    # Place i-th circle at distance proportional to sqrt(i)? 
    # Better: use a greedy placement or just concentric rings.
    
    packed_cells = []
    
    # Center first cell
    packed_cells.append({
        'x': center, 'y': center, 
        'radius': areas_df.iloc[0]['radius'], 
        'type': areas_df.iloc[0]['type'],
        'id': areas_df.iloc[0]['id'],
        'area': areas_df.iloc[0]['area']
    })
    
    # For subsequent cells, place in spiral
    # theta = sqrt(i * pi) * constant?
    # Golden angle spiral: theta = i * 2.4 roughly
    current_r = 0
    golden_angle = np.pi * (3 - np.sqrt(5))
    
    # We need to account for varying sizes.
    # Approximation: Accumulate area to estimate radius from center.
    cum_area = areas_df.iloc[0]['area']
    
    for i in range(1, len(areas_df)):
        row = areas_df.iloc[i]
        r = row['radius']
        
        # Estimate distance from center based on enclosed area
        # Area = pi * R_pos^2 => R_pos = sqrt(Area/pi)
        # We want to place this cell at roughly boundary of current cluster
        dist_from_center = np.sqrt(cum_area / np.pi) + r
        
        theta = i * golden_angle
        
        cx = center + dist_from_center * np.cos(theta)
        cy = center + dist_from_center * np.sin(theta)
        
        packed_cells.append({
            'x': cx, 'y': cy, 'radius': r, 'type': row['type'], 
            'id': row['id'], 'area': row['area']
        })
        
        cum_area += row['area']
        
    packing = pd.DataFrame(packed_cells)
    
    # Scale checking (R did /1000 scaling? Check inputs. Assuming microns, no scale needed unless inputs are huge)
    
    # Xylem frontier logic (from R)
    # Creates detailed border for xylem cells
    xyl = packing[packing['type'] == 'xylem'].copy()
    if not xyl.empty:
        xyl['id_group'] = range(1, len(xyl) + 1)
        xyl['id_layer'] = 1.5
        
        # Create circular points
        scaling = 0.95
        # Generate points
        frontier_list = []
        circus = np.arange(-0.95, 0.95 + 1e-9, 0.95/4) # R seq includes end?
        
        for _, xrow in xyl.iterrows():
            orig_r = xrow['radius']
            mx, my = xrow['x'], xrow['y']
            
            for val in circus:
                y_offset = np.sqrt(1 - val**2)
                # Two points: (val, y_offset) and (val, -y_offset)
                for sign in [1, -1]:
                    x_c = val
                    y_c = sign * y_offset
                    
                    final_x = x_c * orig_r * scaling + mx
                    final_y = y_c * orig_r * scaling + my
                    
                    euc = np.sqrt((mx - center)**2 + (my - center)**2)
                    # angle calculation
                    # R: ifelse(my-center > 0, acos(...), 2pi-acos(...))
                    # This is angle of the GROUP center relative to ROOT center?
                    # R logic computes 'angle' for the frontier point based on the GROUP center?
                    # No, it uses (mx - center) so it's the angle of the xylem CELL center.
                    dx = mx - center
                    # safe acos
                    val_acos = dx/euc if euc > 0 else 0
                    val_acos = np.clip(val_acos, -1, 1)
                    angle = np.arccos(val_acos)
                    if (my - center) < 0:
                        angle = 2 * np.pi - angle
                        
                    frontier_list.append({
                        'radius': orig_r,
                        'x': final_x,
                        'y': final_y,
                        'angle': angle,
                        'id_layer': 1.5,
                        'id_cell': 1, # Placeholder, reassign later
                        'type': 'xylem',
                        'order': get_val("xylem", "order"),
                        'id_group': xrow['id_group']
                    })
                    
        xyl_frontier = pd.DataFrame(frontier_list)
        
        # Parenchyma cells (keep as centers)
        packing_par = packing[packing['type'] == 'parenchyma'].copy()
        
        # R filter loop for parenchyma: euc < stele_diam/2 - cell_diam/2
        # To keep them inside stele
        packing_par['euc'] = np.sqrt((packing_par['x']-center)**2 + (packing_par['y']-center)**2)
        limit = stele_layer_diam/2 - stele_cell_diam/2
        packing_par = packing_par[packing_par['euc'] < limit]
        
        packing_par['angle'] = packing_par.apply(lambda r: np.arctan2(r['y']-center, r['x']-center) % (2*np.pi), axis=1)
        packing_par['id_layer'] = 1.5
        packing_par['id_cell'] = 1
        packing_par['order'] = 1
        packing_par['id_group'] = 0
        packing_par = packing_par.drop(columns=['euc', 'id', 'area'])
        
        # Re-assign IDs
        n_par = len(packing_par)
        packing_par['id_cell'] = range(1, n_par + 1)
        xyl_frontier['id_cell'] = range(n_par + 1, n_par + len(xyl_frontier) + 1)
        
        # Use common columns
        cols = ['radius', 'x', 'y', 'angle', 'id_layer', 'id_cell', 'type', 'order', 'id_group']
        packing = pd.concat([packing_par[cols], xyl_frontier[cols]], ignore_index=True)
        
        # ID Group offset
        packing.loc[packing['type'] == 'xylem', 'id_group'] += k_max_cortex

    return packing
