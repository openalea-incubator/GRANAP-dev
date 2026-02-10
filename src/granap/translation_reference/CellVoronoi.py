import pandas as pd
import geopandas as gpd
import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, Point

def cell_voro(all_cells: pd.DataFrame, vor: Voronoi, center: Point) -> dict:
    """
    Translates cell_voro.R with GeoPandas integration.
    Extracts Voronoi polygons for the given cells and returns GeoDataFrame.
    
    Args:
        all_cells: DataFrame with cell info (must include 'x', 'y', 'id_cell')
        vor: scipy.spatial.Voronoi object computed on all_cells[['x', 'y']]
        center: shapely.geometry.Point, center of cross section (used for dist calculation)
        
    Returns:
        dict with 'all_cells' (GeoDataFrame with polygons) and 'rs2' (GeoDataFrame with vertices)
    """
    
    # 1. Build polygons for each cell and calculate attributes
    all_cells = all_cells.copy()
    all_cells['dist'] = np.sqrt((all_cells['x'] - center.x)**2 + (all_cells['y'] - center.y)**2)
    
    # Create polygon geometries from Voronoi regions
    geometries = []
    areas = []
    
    for i in range(len(all_cells)):
        region_idx = vor.point_region[i]
        region_verts_indices = vor.regions[region_idx]
        
        if -1 in region_verts_indices or len(region_verts_indices) == 0:
            # Infinite region - create None geometry
            geometries.append(None)
            areas.append(np.nan)
            continue
            
        # Get vertices coordinates
        verts = vor.vertices[region_verts_indices]
        
        # Create Shapely polygon
        try:
            poly = Polygon(verts)
            if not poly.is_valid:
                poly = poly.buffer(0)  # Fix invalid polygons
            geometries.append(poly)
            areas.append(poly.area)
        except:
            geometries.append(None)
            areas.append(np.nan)
    
    all_cells['area'] = areas
    
    # Convert to GeoDataFrame
    gdf_cells = gpd.GeoDataFrame(
        all_cells,
        geometry=geometries,
        crs=None  # No CRS for abstract root cross-section
    )
    
    # 2. Build rs2 (vertices dataframe) - points along cell boundaries
    # This maintains compatibility with downstream code that expects vertex points
    rs2_list = []
    
    for (p1_idx, p2_idx), (v1_idx, v2_idx) in zip(vor.ridge_points, vor.ridge_vertices):
        if v1_idx == -1 or v2_idx == -1:
            continue
            
        v1 = vor.vertices[v1_idx]
        v2 = vor.vertices[v2_idx]
        
        # Get id_cell from all_cells
        id1 = all_cells.iloc[p1_idx]['id_cell']
        id2 = all_cells.iloc[p2_idx]['id_cell']
        
        # Add vertices for both cells (R compatibility)
        rs2_list.append({'x': v1[0], 'y': v1[1], 'id_cell': id1})
        rs2_list.append({'x': v2[0], 'y': v2[1], 'id_cell': id1})
        rs2_list.append({'x': v1[0], 'y': v1[1], 'id_cell': id2})
        rs2_list.append({'x': v2[0], 'y': v2[1], 'id_cell': id2})

    rs2_df = pd.DataFrame(rs2_list)
    
    if not rs2_df.empty:
        # Merge with cell info
        merged_rs2 = pd.merge(
            rs2_df, 
            all_cells[['id_cell', 'type', 'area', 'dist', 'angle', 'radius', 'id_layer', 'id_group']], 
            on='id_cell', 
            how='left'
        )
        
        # Filter outside
        merged_rs2 = merged_rs2[merged_rs2['type'] != "outside"]
        
        # Convert to GeoDataFrame with Point geometries
        point_geoms = [Point(row['x'], row['y']) for _, row in merged_rs2.iterrows()]
        gdf_rs2 = gpd.GeoDataFrame(
            merged_rs2,
            geometry=point_geoms,
            crs=None
        )
    else:
        gdf_rs2 = gpd.GeoDataFrame(
            columns=['x', 'y', 'id_cell', 'type', 'area', 'dist', 'angle', 'radius', 'id_layer', 'id_group'],
            geometry=[],
            crs=None
        )

    return {'all_cells': gdf_cells, 'rs2': gdf_rs2}
