import pandas as pd
import numpy as np
import networkx as nx
from shapely.geometry import Polygon
from scipy.spatial import Voronoi
import warnings

# Modules
from ReadXML import ReadXML
from CellLayers import cell_layer
from CreateCells import create_cells
from LayerInfo import layer_info
from MakePith import make_pith
from RondyCortex import rondy_cortex
from Vascular import vascular
from PackXylem import pack_xylem
from CellVoronoi import cell_voro
from SmoothyCells import smoothy_cells
from FuzzeInter import fuzze_inter
from Aerenchyma import aerenchyma
from Septa import septa
from RootHair import root_hair
from ClearNodes import clear_nodes
from Vertex import vertex
from DetailedGraph import create_detailed_graph
from utils import get_val

def create_anatomy(path=None, parameters=None, verbatim=False, maturity_x=False, paraview=False):
    """
    Main function to generate root anatomy.
    Translates create_anatomy.R logic and extends it to return a NetworkX graph.
    Refactored to assume NO GeoPandas available.
    """
    
    # 1. Parameter Loading
    if path is None and parameters is None:
        print("Please specify a parameter set for the simulation")
        return None
        
    if path:
        params = ReadXML(path)
    else:
        params = parameters
        
    if params is None or params.empty:
        return None
        
    if verbatim: print("Params loaded.")
        
    # 2. Setup
    random_fact = get_val(params, "randomness", "value") / 10 * get_val(params, "stele", "cell_diameter") 
    if random_fact == 0: random_fact = get_val(params, "randomness", "value")
    
    proportion_aerenchyma = get_val(params, "aerenchyma", "proportion")
    
    # 3. Cell Layers
    layers = cell_layer(params) # Seems ok
    all_layers = layer_info(layers, params)
    center = all_layers['radius'].max()
    
    # 4. Create Initial Cells
    all_cells = create_cells(all_layers, random_fact)
    if verbatim:
        print("Initial Cell Counts:")
        print(all_cells['type'].value_counts())
    # Initial cell ID group
    all_cells['id_group'] = 0
    
    # 5. Internal Structures
    if get_val(params, "pith", "layer_diameter") > 0:
        all_cells = make_pith(all_cells, params, center)
        
    if get_val(params, "inter_cellular_space", "size") > 0 or get_val(params, "inter_cellular_space", "value") > 0:
        all_cells = rondy_cortex(params, all_cells, center)
        
    if verbatim: print("Add vascular elements")
    sec_growth = get_val(params, "secondarygrowth", "value")
    
    if sec_growth == 0: 
        all_cells = vascular(all_cells, params, layers, center)
    elif sec_growth == 1:
        packing = pack_xylem(all_cells, params, center)
        rm_stele = all_cells[all_cells['type'] != 'stele']
        all_cells = pd.concat([packing, rm_stele], ignore_index=True)
        all_cells['id_cell'] = range(1, len(all_cells) + 1)
        
    if verbatim:
        print(f"Cells after vascular: {len(all_cells)}")
        print(all_cells['type'].value_counts())


    # Preserve parenchyma as distinct tissue type
    # all_cells.loc[all_cells['type'] == "parenchyma", 'type'] = "stele"

    # 6. Voronoi Tesselation
    points = all_cells[['x', 'y']].values
    vor = Voronoi(points)
    
    voro_out = cell_voro(all_cells, vor, center)
    all_cells = voro_out['all_cells']
    rs2 = voro_out['rs2']
    
    if verbatim:
        print(f"rs2 (Voronoi edges) count: {len(rs2)}")
        if len(rs2) > 0:
            print(f"rs2 types: {rs2['type'].value_counts() if 'type' in rs2 else 'No type col'}")
    
    rs1 = rs2.copy()
    if len(rs1) == 0:
        print("WARNING: rs1 is empty after Voronoi.")
        return nx.Graph()

    rs1_grouped = rs1.groupby('id_cell').agg({'x': 'mean', 'y': 'mean'}).rename(columns={'x': 'mx', 'y': 'my'})
    rs1 = rs1.merge(rs1_grouped, on='id_cell')
    rs1['atan'] = np.arctan2(rs1['y'] - rs1['my'], rs1['x'] - rs1['mx'])
    rs1 = rs1.sort_values(by=['id_cell', 'atan'])
    rs1['id_point'] = rs1['x'].astype(str) + ";" + rs1['y'].astype(str)
    
    
    # 7. Smoothing and Merging
    if verbatim: print("Smooth edge of large cells")
    rs1 = smoothy_cells(rs1)
    if verbatim: 
        print(f"After smoothing - unique types: {rs1['type'].unique()}")
    
    if verbatim: print("Merging inter cellular space")
    rs1 = fuzze_inter(rs1)
    if verbatim:
        print(f"After fuzze_inter - unique types: {rs1['type'].unique()}")
    
    # 8. Aerenchyma
    if proportion_aerenchyma > 0:
        rs1 = clear_nodes(rs1)
        if verbatim:
            print(f"After clear_nodes (pre-aerenchyma) - unique types: {rs1['type'].unique()}")
        if verbatim: print("fusing cells for aerenchyma")
        rs1 = aerenchyma(params, rs1)
        if verbatim:
            print(f"After aerenchyma - unique types: {rs1['type'].unique()}")
        if verbatim: print("simplify septa")
        rs1 = septa(rs1)
        if verbatim:
            print(f"After septa - unique types: {rs1['type'].unique()}")
        
    # 9. Root Hair
    if get_val(params, "hair", "n_files") > 0:
        if verbatim: print("Add root hair")
        rs1 = root_hair(rs1, params, center)
        if verbatim:
            print(f"After root_hair - unique types: {rs1['type'].unique()}")
        
    # 10. Final Cleanup
    rs1 = clear_nodes(rs1)
    if verbatim:
        print(f"After final clear_nodes - unique types: {rs1['type'].unique()}")
    
    unique_ids = rs1['id_cell'].unique()
    id_map = {old: new for new, old in enumerate(unique_ids, 1)}
    rs1['id_cell'] = rs1['id_cell'].map(id_map)
    
    ptype = get_val(params, "planttype", "value")
    if ptype == 1:
        x_cells = rs1[rs1['type'] == 'xylem']
        if not x_cells.empty: # Avoid error if empty
             # Recalculate areas to be safe
             tmp_poly_areas = {}
             for cid, grp in rs1.groupby('id_cell'):
                if len(grp) >= 3:
                     try:
                         poly = Polygon(zip(grp['x'], grp['y']))
                         tmp_poly_areas[cid] = poly.area
                     except:
                         tmp_poly_areas[cid] = 0
             
             rs1['area'] = rs1['id_cell'].map(tmp_poly_areas)
             
             x_cells = rs1[rs1['type'] == 'xylem']
             if not x_cells.empty:
                mean_area = x_cells.drop_duplicates('id_cell')['area'].mean()
                large_xylem = x_cells[x_cells['area'] > mean_area]['id_cell'].unique()
                rs1.loc[(rs1['type'] == 'xylem') & (rs1['id_cell'].isin(large_xylem)), 'type'] = 'metaxylem'
            
    if get_val(params, "epidermis", "remove") == 1:
        rs1 = rs1[rs1['type'] != 'epidermis']
        
    # 11. Create NetworkX Graph
    if verbatim: print("Building Detailed NetworkX Graph with Cells, Walls, and Junctions...")
    
    # Pre-calculate polygons and attach to rs1
    # Create a unique dataframe for cells (one row per cell)
    cells_data = []
    
    for cid, grp in rs1.groupby('id_cell'):
        if len(grp) >= 3:
            pts = list(zip(grp['x'], grp['y']))
            poly = Polygon(pts)
            if not poly.is_valid:
                poly = poly.buffer(0)
            
            # Get attributes from first row of group
            first = grp.iloc[0]
            cells_data.append({
                'id_cell': cid,
                'type': first['type'],
                'x': poly.centroid.x,
                'y': poly.centroid.y,
                'area': poly.area,
                'geometry': poly
            })
            
    # Convert to GeoDataFrame for spatial operations
    import geopandas as gpd
    cells_gdf = gpd.GeoDataFrame(cells_data, geometry='geometry', crs=None)
    
    # Build Detailed Graph using GeoDataFrame
    G = create_detailed_graph(cells_gdf, None)

    return G

if __name__ == "__main__":
    pass
