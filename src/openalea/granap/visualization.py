"""
Visualization module for plant anatomy.

Provides plotting and display functions for anatomical structures.
"""

import matplotlib.pyplot as plt
import geopandas as gpd


def plot_section(section_gdf: gpd.GeoDataFrame) -> None:
    """
    Display the organ cross-section as polygons using GeoPandas and Matplotlib.
    
    Args:
        section_gdf: GeoDataFrame containing cell geometries and types
    """
    if section_gdf.empty:
        print("GeoDataFrame is empty, cannot plot.")
        return

    # GeoPandas handles the figure creation and geometry plotting
    fig, ax = plt.subplots(figsize=(8, 8))

    section_gdf.plot(
        ax=ax,
        column='type',           # Color polygons by the 'type' column
        cmap='viridis',          # Use a nice color map
        edgecolor='black',       # Outline the cells
        linewidth=0.5,           # Line width for the outline
        alpha=0.5,               # Transparency
        legend=True,             # Display the legend
        legend_kwds={'title': 'Cell Type', 'loc': 'best'}
    )
    ax.set_aspect("equal", "box")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Cross Section Preview")
    plt.tight_layout()
    plt.show()


def plot_layers_simple(layers_polygons, organ_name: str = "Organ") -> None:
    """
    Simple plot of layer boundaries.
    
    Args:
        layers_polygons: List of layer polygon dictionaries
        organ_name: Name of the organ for the title
    """
    plt.close('all')
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = plt.cm.viridis(range(len(layers_polygons)))
    
    for polygon_data, color in zip(layers_polygons, colors):
        ax.plot(*polygon_data["polygon"].exterior.xy, 
               color=color, label=polygon_data["name"])
    
    ax.set_aspect('equal')
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(f"{organ_name} - Layer Boundaries")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
