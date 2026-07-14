"""
Visualization module for plant anatomy.

Provides plotting and display functions for anatomical structures.
"""
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Any, Optional

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from shapely.ops import unary_union

from openalea.granap.cell_manager import CellManager

if TYPE_CHECKING:
    from openalea.granap.organ_class import Organ


def plot_section(section_gdf: gpd.GeoDataFrame) -> None:
    """
    Display the organ cross-section as polygons using GeoPandas and Matplotlib.

    Args:
        section_gdf: GeoDataFrame containing cell geometries and types
    """
    if section_gdf.empty:
        print("GeoDataFrame is empty, cannot plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    section_gdf.plot(
        ax=ax,
        column='type',
        cmap='viridis',
        edgecolor='black',
        linewidth=0.5,
        alpha=0.5,
        legend=True,
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


def plot_cells(self, ax=None):
    # plot cells coordinates, colour-coded by type
    if ax is None:
        fig, ax = plt.subplots()

    types = self.get_all_types()
    colors = cm.viridis(np.linspace(0, 1, len(types)))
    type_to_color = dict(zip(types, colors))

    for cell in self.cells:
        if cell.type in type_to_color:
            c = type_to_color[cell.type]
            ax.plot(cell.x, cell.y, marker='o', linestyle='', c=c, label=cell.type)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys())

    ax.set_aspect('equal', adjustable='box')
    plt.show()


# ---------------------------------------------------------------------------
# Tissue-map helpers
# ---------------------------------------------------------------------------

def _dry_run_vascular(organ: "Organ"):
    """
    Run _create_vascular_tissue on a fresh empty state and return the
    collected polygons, leaving the organ's real state untouched.

    Returns
    -------
    layers : list of layer-polygon dicts
    vascular_polys : list of Shapely polygons (xylem mask)
    vascular_tissue_polys : dict {tissue_name: [Polygon, …]}
    """
    layers = organ.generate_layer_polygons()

    saved_all_cells             = organ.all_cells
    saved_vascular_cells        = getattr(organ, 'vascular_cells', CellManager())
    saved_vascular_polys        = list(getattr(organ, 'vascular_polygons', []))
    saved_vascular_tissue_polys = dict(getattr(organ, 'vascular_tissue_polygons', {}))
    saved_rng_state             = organ.rng.bit_generator.state
    try:
        organ.all_cells                = CellManager()
        organ.vascular_cells           = CellManager()
        organ.vascular_polygons        = []
        organ.vascular_tissue_polygons = {}
        polygon_for_vascular           = organ._which_layer_for_vascular(layers)
        organ._create_vascular_tissue(polygon_for_vascular)
        vascular_polys        = list(organ.vascular_polygons)
        vascular_tissue_polys = dict(organ.vascular_tissue_polygons)
    finally:
        organ.all_cells                = saved_all_cells
        organ.vascular_cells           = saved_vascular_cells
        organ.vascular_polygons        = saved_vascular_polys
        organ.vascular_tissue_polygons = saved_vascular_tissue_polys
        organ.rng.bit_generator.state  = saved_rng_state

    return layers, vascular_polys, vascular_tissue_polys


def build_anatomy_tissues(organ: "Organ") -> List[Dict[str, Any]]:
    """
    Return a list of tissue zone descriptors without placing any cell.

    Each entry contains:
      - ``name``    : tissue name (layer name or vascular element type)
      - ``polygon`` : Shapely geometry of the zone
      - ``kind``    : ``"layer"`` or ``"vascular"``

    The organ's state is not modified.
    """
    layers, vascular_polys, _ = _dry_run_vascular(organ)

    tissues: List[Dict[str, Any]] = []
    for i, layer_data in enumerate(layers):
        name = layer_data["name"]
        if name == "outside":
            continue
        poly = layer_data["polygon"]
        ring = poly.difference(layers[i + 1]["polygon"]) if i + 1 < len(layers) else poly
        if not ring.is_empty:
            tissues.append({"name": name, "polygon": ring, "kind": "layer"})

    for vpoly in vascular_polys:
        if not vpoly.is_empty:
            tissues.append({"name": "vascular", "polygon": vpoly, "kind": "vascular"})

    return tissues


def plot_tissues(organ: "Organ",
                 ax=None,
                 show: bool = True,
                 labels: bool = True,
                 show_effective: bool = False,
                 fuse: bool = False) -> Optional[plt.Figure]:
    """
    Plot every tissue zone before placing any cell.

    Layer rings are colour-coded; vascular element polygons are overlaid in
    steel-blue; cambium and phloem zones (if tracked) in their own colours.

    Args:
        organ:          Any Organ subclass instance.
        ax:             Optional matplotlib axes to draw on.
        show:           Call ``plt.show()`` when done.
        labels:         Annotate each zone with its tissue name.
        show_effective: Overlay the effective vascular mask as a red dashed outline.
        fuse:           Merge all sub-rings of the same tissue into one polygon
                        before drawing, so each tissue appears as a single zone.

    Returns:
        Matplotlib figure if a new one was created, else None.
    """
    layers, vascular_polys, vascular_tissue_polys = _dry_run_vascular(organ)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    # Collect all rings per tissue name, in first-occurrence order
    rings_by_name: dict = defaultdict(list)
    oc = layers[0]["polygon"].centroid
    cx, cy = oc.x, oc.y

    for i, layer_data in enumerate(layers):
        name = layer_data["name"]
        if name == "outside":
            continue
        poly = layer_data["polygon"]
        ring = poly.difference(layers[i + 1]["polygon"]) if i + 1 < len(layers) else poly
        if not ring.is_empty:
            rings_by_name[name].append(ring)

    # One colour per unique tissue name
    unique_names = list(rings_by_name.keys())
    name_colors = {
        name: plt.cm.Set3(k / max(len(unique_names) - 1, 1))
        for k, name in enumerate(unique_names)
    }

    # Draw rings (fused or individual)
    for name, ring_list in rings_by_name.items():
        color = name_colors[name]
        polys_to_draw = [unary_union(ring_list)] if fuse else ring_list
        for poly_to_draw in polys_to_draw:
            geoms = list(poly_to_draw.geoms) if hasattr(poly_to_draw, "geoms") else [poly_to_draw]
            for geom in geoms:
                if geom.geom_type != "Polygon":
                    continue
                ax.fill(*geom.exterior.xy, color=color, alpha=0.45)
                ax.plot(*geom.exterior.xy, color="gray", linewidth=0.5)

    # One label per unique tissue, at the ring's natural radius but at an
    # evenly-spaced angle so labels don't pile up on the same side
    if labels:
        items = list(rings_by_name.items())
        n_u = len(items)
        for k, (tissue_name, ring_list) in enumerate(items):
            mid_ring = ring_list[len(ring_list) // 2]
            pt = mid_ring.representative_point()
            r = np.hypot(pt.x - cx, pt.y - cy)
            angle = 2 * np.pi * k / max(n_u, 1)
            lx = cx + r * np.cos(angle)
            ly = cy + r * np.sin(angle)
            ax.text(lx, ly, tissue_name, ha="center", va="center",
                    fontsize=7, color="black",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              alpha=0.65, ec="none"))

    # Xylem vessels (mask polygons)
    for vpoly in vascular_polys:
        if vpoly.is_empty:
            continue
        geoms = list(vpoly.geoms) if hasattr(vpoly, "geoms") else [vpoly]
        for geom in geoms:
            if geom.geom_type != "Polygon":
                continue
            ax.fill(*geom.exterior.xy, color="steelblue", alpha=0.65)
            ax.plot(*geom.exterior.xy, color="navy", linewidth=0.7)

    # Named tissue polygons (cambium → green, phloem → goldenrod, …)
    _tissue_colors = {
        "cambium": "limegreen", "phloem": "goldenrod",
        "xylem": "lightsteelblue",      # bundle xylem zone (vessels overlaid steel-blue)
        "sclerenchyma": "silver",       # bundle sheath / caps (fibres)
        "bundle sheath": "yellowgreen", # parenchyma bundle sheath (no fibres)
        "bundle": "khaki",              # stem vascular-bundle envelope (drawn underneath)
        "medullary cavity": "white",    # hollow pith / protoxylem lacuna = void
    }
    for tissue_name, poly_list in vascular_tissue_polys.items():
        color = _tissue_colors.get(tissue_name, "orange")
        # Cavities (hollow pith / protoxylem lacuna) are voids: draw them nearly
        # opaque so they read as empty rather than tinting the pith beneath.
        is_void = tissue_name == "medullary cavity"
        alpha = 0.95 if is_void else 0.35
        edge = "gray" if is_void else color
        for vpoly in poly_list:
            if vpoly.is_empty:
                continue
            geoms = list(vpoly.geoms) if hasattr(vpoly, "geoms") else [vpoly]
            for geom in geoms:
                if geom.geom_type != "Polygon":
                    continue
                ax.fill(*geom.exterior.xy, color=color, alpha=alpha)
                ax.plot(*geom.exterior.xy, color=edge, linewidth=1.0)

    if show_effective and vascular_polys:
        mask = unary_union(vascular_polys)
        geoms = list(mask.geoms) if hasattr(mask, "geoms") else [mask]
        for geom in geoms:
            ax.plot(*geom.exterior.xy, color="red", linewidth=1.5, linestyle="--")

    # Organ-specific extras (e.g. resin ducts, stomata for needles)
    _extra_style = {
        "resin_duct":  ("burlywood",  "darkorange", 0.75),
        "resin_canal": ("lightyellow", "goldenrod",  0.90),
        "stomata":     ("lightpink",   "crimson",    0.80),
    }
    for tissue_name, poly_list in organ._extra_tissue_polygons(layers).items():
        face, edge, alpha = _extra_style.get(tissue_name, ("orange", "darkorange", 0.7))
        first_geom = None
        for poly in poly_list:
            if poly is None or poly.is_empty:
                continue
            geoms = list(poly.geoms) if hasattr(poly, "geoms") else [poly]
            for geom in geoms:
                if geom.geom_type != "Polygon":
                    continue
                ax.fill(*geom.exterior.xy, color=face, alpha=alpha)
                ax.plot(*geom.exterior.xy, color=edge, linewidth=0.8)
                if first_geom is None:
                    first_geom = geom
        if labels and first_geom is not None:
            pt = first_geom.representative_point()
            ax.text(pt.x, pt.y, tissue_name, ha="center", va="center",
                    fontsize=6, color="black",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.65, ec="none"))

    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(f"{organ.__class__.__name__} — {'fused' if fuse else 'rings'}")

    if fig is not None:
        plt.tight_layout()
        if show:
            plt.show()
        return fig
    return None
