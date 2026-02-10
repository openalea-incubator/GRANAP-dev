import pandas as pd
import numpy as np

def vertex(rs1: pd.DataFrame) -> pd.DataFrame:
    """
    Translates vertex.R.
    Process Voronoi nodes to create wall segments.
    Input rs1 expected to be nodes/points from Voronoi tesselation or similar structure 
    that contains wall endpoints.
    
    Actually, looking at R code:
    rs1 seems to be a collection of points describing the boundary of cells? 
    R code:
      mutate(id_point= paste0(rs1$x,";",rs1$y))
      group_by(id_cell)
      mutate(xx = c(x[-1],x[1]), yy = c(y[-1],y[1]))
    It connects points in sequence to form a polygon for each cell.
    """
    
    # Ensure sorted order usually (assuming input is already ordered by angle/sequence)
    # The R code assumes the order in rs1 is correct for drawing the polygon.
    
    # Copy to avoid mutating original
    nodes = rs1.copy()
    
    # We need to shift x and y to get the "next" point in the polygon (x2, y2)
    # R: c(x[-1], x[1]) is a shift left (rotate)
    
    nodes['xx'] = nodes.groupby('id_cell')['x'].transform(lambda x: np.roll(x, -1))
    nodes['yy'] = nodes.groupby('id_cell')['y'].transform(lambda x: np.roll(x, -1))
    
    # Geometry calculations
    # R implies we want canonical coordinates (x1, y1) < (x2, y2) or similar to identify unique walls
    # But first, let's just define x1, y1, x2, y2
    
    # R logic:
    # x1 = min(x, xx), x2 = max(x, xx) roughly?
    # R: x1 = ifelse(x > xx, x, xx) -> Wait, if x > xx, x1=x. This means x1 is the LARGER x?
    # Let's follow R exactly.
    # mutate(x1 = ifelse(x > xx, x, xx)) => MAX(x, xx)
    # mutate(x2 = ifelse(x > xx, xx, x)) => MIN(x, xx)
    # mutate(y1 = ifelse(x > xx, y, yy)) => Corresponding y for x1
    # mutate(y2 = ifelse(x > xx, yy, y)) => Corresponding y for x2
    
    # So (x1, y1) is the point with larger X.
    
    # Vectorized implementation
    mask = nodes['x'] > nodes['xx']
    
    nodes['x1'] = np.where(mask, nodes['x'], nodes['xx'])
    nodes['x2'] = np.where(mask, nodes['xx'], nodes['x'])
    nodes['y1'] = np.where(mask, nodes['y'], nodes['yy'])
    nodes['y2'] = np.where(mask, nodes['yy'], nodes['y'])
    
    # Vertical lines special case (x == xx)
    # R: if x == xx, check y.
    # y1 = ifelse(y > yy, yy, y) -> if y > yy, MIN(y, yy). (Wait, R: ifelse(TRUE, yy, y) -> yy is smaller)
    # y2 = ifelse(y > yy, y, yy) -> MAX(y, yy)
    # So for vertical lines, (x1, y1) is the point with SMALLER Y.
    
    v_mask = nodes['x'] == nodes['xx']
    y_gt_yy = nodes['y'] > nodes['yy']
    
    # Apply vertical logic only where v_mask is True
    # If v_mask & y_gt_yy: y1 = yy (smaller), y2 = y (larger)
    # If v_mask & !y_gt_yy: y1 = y (smaller), y2 = yy (larger)
    
    nodes.loc[v_mask, 'y1'] = np.where(y_gt_yy[v_mask], nodes['yy'][v_mask], nodes['y'][v_mask])
    nodes.loc[v_mask, 'y2'] = np.where(y_gt_yy[v_mask], nodes['y'][v_mask], nodes['yy'][v_mask])
    
    # But wait, looking at R again:
    # mutate(y1 = ifelse(x == xx, ifelse(y > yy, yy, y), y1))
    
    # Calculate lengths
    nodes['wall_length'] = np.sqrt((nodes['x2'] - nodes['x1'])**2 + (nodes['y2'] - nodes['y1'])**2)
    
    # Calculate slope and intercept
    # slope = (y2 - y1) / (x2 - x1)
    # division by zero if vertical (x2 == x1), numpy handles as inf
    with np.errstate(divide='ignore', invalid='ignore'):
        nodes['slope'] = (nodes['y2'] - nodes['y1']) / (nodes['x2'] - nodes['x1'])
    
    nodes['intercept'] = nodes['y1'] - nodes['slope'] * nodes['x1']
    
    return nodes
