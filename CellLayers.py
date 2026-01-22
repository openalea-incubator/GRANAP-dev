import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict
from LayerInfo import layer_info

def cell_layer(params: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrates the radial 'blueprint' of the root.
    """
    
    # 1. Filter and Pivot (equivalent to tidyr::spread)
    # We want rows: tissue name, columns: cell_diameter, n_layers, order
    layers = params[params['type'].isin(["cell_diameter", "n_layers", "order"])]
    layers = layers.pivot(index='name', columns='type', values='value').reset_index()
    
    # Ensure columns exist even if missing in XML
    for col in ['cell_diameter', 'n_layers', 'order']:
        if col not in layers.columns:
            layers[col] = np.nan

    # 2. Add "outside" layer for Voronoi boundary
    # R logic: uses epidermis cell diameter for the outside boundary padding
    epi_diam = layers.loc[layers['name'] == 'epidermis', 'cell_diameter'].values[0] if 'epidermis' in layers['name'].values else 0.1
    
    outside_layer = pd.DataFrame({
        'name': ['outside'],
        'n_layers': [2.0],
        'cell_diameter': [epi_diam],
        'order': [layers['order'].max() + 1]
    })
    layers = pd.concat([layers, outside_layer], ignore_index=True)

    # 3. Stele special logic
    # Get stele diameter to calculate how many "layers" of cells fit inside the stele radius
    stele_val = params[(params['name'] == 'stele') & (params['type'] == 'layer_diameter')]
    if not stele_val.empty:
        stele_diameter = stele_val['value'].values[0]
        stele_idx = layers.index[layers['name'] == 'stele']
        if len(stele_idx) > 0:
            # Calculate n_layers based on radius (diam/2)
            n_stele_layers = round((stele_diameter / 2) / layers.loc[stele_idx, 'cell_diameter'].values[0])
            layers.loc[stele_idx, 'n_layers'] = n_stele_layers

    layers = layers.dropna(subset=['n_layers']).sort_values('order').reset_index(drop=True)
    
    # 4. Phloem cleanup
    if 'phloem' in layers['name'].values:
        if layers.loc[layers['name'] == 'phloem', 'n_layers'].values[0] == 0:
            layers = layers[layers['name'] != 'phloem']

    return layers

