import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

def concavety(data: pd.DataFrame, merge: bool = True) -> pd.DataFrame:
    """
    Translates concavety.R.
    Merges multiple polygon cells that are adjacent/overlapping if merge=True.
    Otherwise cleans individual polygons.
    """
    
    cell_polys = [] # List of (id_cell, polygon, original_row)
    
    # Iterate over unique id_cell
    for i in data['id_cell'].unique():
        subset = data[data['id_cell'] == i]
        if len(subset) > 2:
            coords = list(zip(subset['x'], subset['y']))
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            
            try:
                poly = Polygon(coords)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                
                if not poly.is_empty:
                    cell_polys.append((i, poly, subset.iloc[0]))
            except Exception:
                pass

    if not cell_polys:
        return pd.DataFrame(columns=data.columns)

    results = []

    if merge:
        # Original logic: Union all
        all_polys = [p for _, p, _ in cell_polys]
        merged_poly = unary_union(all_polys)
        
        # Simplify
        cleaned_poly = merged_poly.simplify(0.001)
        
        if cleaned_poly.is_empty:
            return pd.DataFrame(columns=data.columns)
            
        final_polys = []
        if cleaned_poly.geom_type == 'MultiPolygon':
            # R logic seems to imply resulting in one "cell"? 
            # But if we have disjoint groups?
            # Assuming largest for now as per previous logic
            final_polys = [max(cleaned_poly.geoms, key=lambda p: p.area)]
        elif cleaned_poly.geom_type == 'Polygon':
            final_polys = [cleaned_poly]
            
        # For merged, we use the first cell's info as representative?
        # Or average? Previous code used first row of entire subset.
        rep_row = data.iloc[0]
        
        for poly in final_polys:
            res_df = _poly_to_df(poly, rep_row['id_cell'], rep_row, data)
            results.append(res_df)
            
    else:
        # Keep individual
        for cid, poly, rep_row in cell_polys:
            cleaned_poly = poly.simplify(0.001)
            if cleaned_poly.is_empty: continue
            
            # Handle multipolygon result from simplify?
            if cleaned_poly.geom_type == 'MultiPolygon':
                sub_polys = list(cleaned_poly.geoms)
            elif cleaned_poly.geom_type == 'Polygon':
                sub_polys = [cleaned_poly]
            else:
                continue
                
            for sp in sub_polys:
                # We need to maintain original attributes
                # Pass a subset for 'data' related to this cell only to calculate means correctly
                cell_subset = data[data['id_cell'] == cid]
                res_df = _poly_to_df(sp, cid, rep_row, cell_subset)
                results.append(res_df)

    if not results:
        return pd.DataFrame(columns=data.columns)
        
    final_df = pd.concat(results, ignore_index=True)
    
    # Ensure columns
    for col in data.columns:
        if col not in final_df.columns:
            final_df[col] = np.nan
            
    return final_df[data.columns]

def _poly_to_df(poly, id_cell, rep_row, source_data):
    x, y = poly.exterior.coords.xy
    x = list(x)
    y = list(y)
    
    if len(x) > 1 and x[0] == x[-1] and y[0] == y[-1]:
        x.pop()
        y.pop()
        
    n = len(x)
    
    new_cell = pd.DataFrame({
        'id_cell': [id_cell] * n,
        'x': x,
        'y': y,
        'type': [rep_row['type']] * n,
        'area': [poly.area] * n,
        'dist': [source_data['dist'].mean()] * n,
        'angle': [source_data['angle'].mean()] * n,
        'radius': [source_data['radius'].mean()] * n,
        'id_layer': [source_data['id_layer'].mean()] * n,
        'id_group': [rep_row['id_group']] * n,
        'my': [np.mean(y)] * n,
        'mx': [np.mean(x)] * n,
        'atan': np.linspace(-np.pi, np.pi, n),
        'new': [id_cell] * n,
        'y1': y,
        'y2': np.roll(y, -1),
        'x1': x,
        'x2': np.roll(x, -1),
        'id_point': [f"{cx};{cy}" for cx, cy in zip(x, y)]
    })
    return new_cell
