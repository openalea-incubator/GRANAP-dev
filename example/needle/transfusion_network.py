"""Transfusion network — the deliverable figure for
``example/needle/Transfusion_network.md``.

Physiological requirements recap (see the .md for the full context):

1. Transfusion tracheids are dead apoplastic conduits: no symplastic
   (plasmodesmata) connection to anything.
2. Transfusion parenchyma must stay symplastically continuous from the
   endodermis inward to the Strasburger cells / phloem, even though each
   round parenchyma ellipse sits individually embedded in the tracheid
   matrix in the 2-D cross section. ``NeedleAnatomy.network_bridge_specs``
   bridges a parenchyma to a nearby symplastic neighbour across an
   intervening tracheid with a virtual NODE (no polygon), standing in for a
   third, out-of-plane cell.

Visual language matches the authoritative reference diagram
``example/needle/Transfusion_tissue_network.png``:

    orange edge  = membrane (transmembrane) edge
    cyan edge    = symplastic (plasmodesmata) edge
    blue dot     = virtual symplastic node (out-of-plane bridge cell)
    red dot      = existing symplastic node (a real cell)

Renders headlessly (matplotlib Agg) and saves PNGs under
``example/needle/outputs/``, then prints the verification report: tracheid
symplastic-edge count (must be 0), tracheids-with-no-plasmodesmata count
(must equal the tracheid count), bridge/virtual-node counts under both
sharing strategies, and the connected-component count of the symplast
subgraph restricted to {transfusion parenchyma, parenchyma, Strasburger
cell, endodermis, phloem} (must be 1), plus a cgroup histogram.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless: never opens a window, only saves PNGs
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from openalea.granap.needle_class import NeedleAnatomy

from pinus_pinaster import build_pinaster

SEED = 0
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

SYMPLASTIC_CONTINUITY_TYPES = {
    "transfusion parenchyma", "parenchyma", "Strasburger cell", "endodermis", "phloem",
}

# Reference-diagram legend colours.
MEMBRANE_COLOR = "#ff7f0e"   # orange
SYMPLASTIC_COLOR = "#17becf"  # cyan
VIRTUAL_NODE_COLOR = "#1f4fd6"  # blue
REAL_NODE_COLOR = "#d62728"     # red


def _cell_nodes_by_type(graph, cell_type):
    return [nd for nd, d in graph.nodes(data=True) if d.get("cell_type") == cell_type]


def _transfusion_zone_bounds(gdf, pad_factor=0.15):
    """Bounding box (padded) around every transfusion-tissue cell, for zooming."""
    zone = gdf[gdf["type"].isin(["transfusion parenchyma", "transfusion tracheid"])]
    minx, miny, maxx, maxy = zone.geometry.total_bounds
    pad_x = (maxx - minx) * pad_factor
    pad_y = (maxy - miny) * pad_factor
    return minx - pad_x, maxx + pad_x, miny - pad_y, maxy + pad_y


def plot_transfusion_cells(needle, gdf, show=False):
    """Panel 1: cells, zoomed to the transfusion zone, tracheid / parenchyma
    coloured, with every virtual (bridge) node's position overlaid as a blue
    dot -- the bridge is a node now, not a polygon, so there is no "bridge
    cell" region to shade."""
    g = needle.graph
    fig, ax = plt.subplots(figsize=(10, 8))

    real_parenchyma = gdf[(gdf["type"] == "transfusion parenchyma") & ~gdf.get("bridge", False).astype(bool)]
    tracheids = gdf[gdf["type"] == "transfusion tracheid"]

    tracheids.plot(ax=ax, color="#8c8c8c", edgecolor="black", linewidth=0.2, label="transfusion tracheid")
    real_parenchyma.plot(ax=ax, color="#2ca02c", edgecolor="black", linewidth=0.2, label="transfusion parenchyma")

    bridge_nodes = [nd for nd, d in g.nodes(data=True) if d.get("bridge")]
    if bridge_nodes:
        xs = [g.nodes[nd]["position"][0] for nd in bridge_nodes]
        ys = [g.nodes[nd]["position"][1] for nd in bridge_nodes]
        ax.scatter(xs, ys, s=10, color=VIRTUAL_NODE_COLOR, zorder=5, label="virtual symplastic node")

    x0, x1, y0, y1 = _transfusion_zone_bounds(gdf)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Transfusion tissue — cells ({len(bridge_nodes)} virtual nodes)")

    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", color=c, label=lbl)
        for c, lbl in [("#8c8c8c", "transfusion tracheid"),
                       ("#2ca02c", "transfusion parenchyma"),
                       (VIRTUAL_NODE_COLOR, "virtual symplastic node")]
    ]
    ax.legend(handles=handles, loc="upper right", framealpha=0.9)
    plt.tight_layout()

    path = os.path.join(OUT_DIR, "transfusion_network_cells.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    if show:
        plt.show()
    plt.close(fig)


def _relevant_symplast_elements(needle):
    """Collect the nodes/edges the symplast panels care about: every real
    cell of a symplastic-continuity type, plus every virtual (bridge) node
    -- since a bridge always sits between two such real cells. Returns
    (relevant_nodes, plasmodesmata_edges, membrane_edges) where the membrane
    edges are only those touching a virtual node (the ones the reference
    diagram actually draws in orange -- a real cell's membrane edge to its
    own wall is not part of this motif)."""
    g = needle.graph
    bridge_nodes = {nd for nd, d in g.nodes(data=True) if d.get("bridge")}
    real_relevant = {
        nd for nd, d in g.nodes(data=True)
        if d.get("cell_type") in SYMPLASTIC_CONTINUITY_TYPES and not d.get("bridge")
    }
    relevant_nodes = real_relevant | bridge_nodes

    sym_edges = [
        (u, v) for u, v, d in g.edges(data=True)
        if d.get("path") == "plasmodesmata" and u in relevant_nodes and v in relevant_nodes
    ]
    mem_edges = [
        (u, v) for u, v, d in g.edges(data=True)
        if d.get("path") == "membrane" and (u in bridge_nodes or v in bridge_nodes)
    ]
    return relevant_nodes, bridge_nodes, sym_edges, mem_edges


def _draw_symplast_panel(ax, needle, relevant_nodes, bridge_nodes, sym_edges, mem_edges,
                          node_size=14, sym_width=1.0, mem_width=1.0):
    """Shared drawing routine for the broad and zoomed symplast panels --
    orange membrane edges, cyan symplastic edges, blue virtual nodes, red
    real nodes, exactly the reference diagram's legend."""
    g = needle.graph
    position = nx.get_node_attributes(g, "position")

    # Membrane edges reach a wall node (not itself drawn/coloured -- no
    # legend entry for it, matching the given legend exactly).
    nx.draw_networkx_edges(g, position, ax=ax, edgelist=mem_edges,
                            edge_color=MEMBRANE_COLOR, width=mem_width, alpha=0.9)
    nx.draw_networkx_edges(g, position, ax=ax, edgelist=sym_edges,
                            edge_color=SYMPLASTIC_COLOR, width=sym_width, alpha=0.85)

    real_nodes = [nd for nd in relevant_nodes if nd not in bridge_nodes]
    virtual_nodes = [nd for nd in relevant_nodes if nd in bridge_nodes]
    nx.draw_networkx_nodes(g, position, ax=ax, nodelist=real_nodes,
                            node_color=REAL_NODE_COLOR, node_size=node_size)
    nx.draw_networkx_nodes(g, position, ax=ax, nodelist=virtual_nodes,
                            node_color=VIRTUAL_NODE_COLOR, node_size=node_size)


def plot_symplast_subgraph(needle, gdf, show=False):
    """Panel 2: the symplast subgraph over the transfusion zone + endodermis
    + Strasburger cells, in the reference diagram's colour language."""
    relevant_nodes, bridge_nodes, sym_edges, mem_edges = _relevant_symplast_elements(needle)

    fig, ax = plt.subplots(figsize=(10, 8))
    _draw_symplast_panel(ax, needle, relevant_nodes, bridge_nodes, sym_edges, mem_edges)

    x0, x1, y0, y1 = _transfusion_zone_bounds(gdf)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Symplast subgraph — {len(bridge_nodes)} virtual nodes, {len(mem_edges)} membrane bridge edges")

    handles = [
        plt.Line2D([0], [0], color=MEMBRANE_COLOR, label="membrane edge"),
        plt.Line2D([0], [0], color=SYMPLASTIC_COLOR, label="symplastic edge"),
        plt.Line2D([0], [0], marker="o", linestyle="", color=VIRTUAL_NODE_COLOR, label="virtual symplastic node"),
        plt.Line2D([0], [0], marker="o", linestyle="", color=REAL_NODE_COLOR, label="existing symplastic node"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)
    plt.tight_layout()

    path = os.path.join(OUT_DIR, "transfusion_network_symplast.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    if show:
        plt.show()
    plt.close(fig)

    return sym_edges


def plot_symplast_zoom(needle, gdf, n_tracheids=6, show=False):
    """Panel 3: a zoomed view on a handful of tracheids, at roughly the
    reference diagram's scale -- the whole-section symplast panel is
    illegible at this density, so this is the panel that is actually
    comparable to ``Transfusion_tissue_network.png``."""
    g = needle.graph
    relevant_nodes, bridge_nodes, sym_edges, mem_edges = _relevant_symplast_elements(needle)

    if not bridge_nodes:
        print("No virtual nodes to zoom on -- skipping the zoom panel.")
        return

    # Pick a handful of SPATIALLY CLUSTERED virtual nodes (not just the
    # globally highest-degree ones, which can be scattered across the whole
    # cross-section and force a huge, illegible window): anchor on the
    # highest-degree bridge node, then take its ``n_tracheids - 1`` nearest
    # neighbours by position, so the zoom shows real local chaining/sharing.
    position = nx.get_node_attributes(g, "position")
    degree = {nd: 0 for nd in bridge_nodes}
    for u, v in sym_edges + mem_edges:
        if u in degree:
            degree[u] += 1
        if v in degree:
            degree[v] += 1
    anchor = max(bridge_nodes, key=lambda nd: degree[nd])
    ax_, ay_ = position[anchor]
    by_distance = sorted(
        bridge_nodes,
        key=lambda nd: (position[nd][0] - ax_) ** 2 + (position[nd][1] - ay_) ** 2,
    )
    chosen = by_distance[:n_tracheids]

    xs = [position[nd][0] for nd in chosen]
    ys = [position[nd][1] for nd in chosen]
    cx, cy = float(np.mean(xs)), float(np.mean(ys))
    # A window sized off the median real transfusion-cell diameter (a
    # handful of cells across, roughly the reference diagram's cell-to-
    # canvas ratio) -- capped so a couple of outliers among the "nearest"
    # chosen nodes cannot blow the window back out.
    real_dia = gdf.loc[
        gdf["type"].isin(["transfusion tracheid", "transfusion parenchyma"]) & ~gdf.get("bridge", False).astype(bool),
        "geometry",
    ].apply(lambda p: 2 * np.sqrt(p.area / np.pi) if p is not None else np.nan)
    median_dia = float(np.nanmedian(real_dia)) if len(real_dia) else max(np.ptp(xs), np.ptp(ys), 1e-6)
    half_win = 2.2 * median_dia

    fig, ax = plt.subplots(figsize=(9, 8))

    # Filter out virtual (bridge) rows -- geometry is None for those -- BEFORE
    # the ``.cx`` bbox slice, which otherwise has to build a spatial index
    # over a column containing None geometries.
    real_gdf = gdf[gdf["geometry"].notna()]
    zoom_cells = real_gdf.cx[cx - half_win:cx + half_win, cy - half_win:cy + half_win]
    type_colors = {
        "transfusion tracheid": "#d9d9d9",
        "transfusion parenchyma": "#c7e9c0",
        "parenchyma": "#c7e9c0",
        "Strasburger cell": "#ffd8a8",
        "endodermis": "#a6cee3",
        "phloem": "#cab2d6",
    }
    for ctype, color in type_colors.items():
        subset = zoom_cells[zoom_cells["type"] == ctype]
        if not subset.empty:
            subset.plot(ax=ax, color=color, edgecolor="black", linewidth=0.6)
    other = zoom_cells[~zoom_cells["type"].isin(type_colors)]
    if not other.empty:
        other.plot(ax=ax, color="#f0f0f0", edgecolor="black", linewidth=0.4)

    # Restrict the drawn graph elements to the zoom window so the panel
    # stays legible.
    window_nodes = {
        nd for nd in relevant_nodes
        if cx - half_win <= position[nd][0] <= cx + half_win
        and cy - half_win <= position[nd][1] <= cy + half_win
    }
    window_sym = [(u, v) for u, v in sym_edges if u in window_nodes and v in window_nodes]
    window_mem = [(u, v) for u, v in mem_edges if u in window_nodes or v in window_nodes]

    _draw_symplast_panel(ax, needle, window_nodes, bridge_nodes, window_sym, window_mem,
                          node_size=60, sym_width=2.0, mem_width=2.0)

    ax.set_xlim(cx - half_win, cx + half_win)
    ax.set_ylim(cy - half_win, cy + half_win)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Symplast zoom — {len(chosen)} tracheids "
                 f"(compare with Transfusion_tissue_network.png)")

    handles = [
        plt.Line2D([0], [0], color=MEMBRANE_COLOR, label="membrane edge"),
        plt.Line2D([0], [0], color=SYMPLASTIC_COLOR, label="symplastic edge"),
        plt.Line2D([0], [0], marker="o", linestyle="", color=VIRTUAL_NODE_COLOR, label="virtual symplastic node"),
        plt.Line2D([0], [0], marker="o", linestyle="", color=REAL_NODE_COLOR, label="existing symplastic node"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)
    plt.tight_layout()

    path = os.path.join(OUT_DIR, "transfusion_network_symplast_zoom.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_apoplastic_isolation(needle, gdf, show=False):
    """Panel 4: the full network faded with the apoplastic tracheid paths
    highlighted -- tracheids stay connected apoplastically (membrane/wall)
    but are symplastically isolated (no plasmodesmata edge reaches them)."""
    g = needle.graph
    position = nx.get_node_attributes(g, "position")

    tracheid_nodes = _cell_nodes_by_type(g, "transfusion tracheid")
    apo_edges = []
    apo_wall_nodes = set()
    for nd in tracheid_nodes:
        for nb in g.neighbors(nd):
            if g.edges[nd, nb].get("path") == "membrane":
                apo_edges.append((nd, nb))
                apo_wall_nodes.add(nb)
    for wn in list(apo_wall_nodes):
        for nb in g.neighbors(wn):
            if g.edges[wn, nb].get("path") == "wall":
                apo_edges.append((wn, nb))

    fig, ax = plt.subplots(figsize=(12, 10))
    needle.plot_network(ax=ax, show=False, node_size=3, width=0.25, alpha=0.12,
                         title="Transfusion tracheids: apoplastic-connected, symplastically isolated")

    nx.draw_networkx_edges(g, position, ax=ax, edgelist=apo_edges, edge_color="#8c8c8c", width=1.0, alpha=0.8)
    nx.draw_networkx_nodes(g, position, ax=ax, nodelist=tracheid_nodes, node_color="#8c8c8c",
                            node_size=20, label="transfusion tracheid")

    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()

    path = os.path.join(OUT_DIR, "transfusion_network_apoplastic.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    if show:
        plt.show()
    plt.close(fig)


def print_report(needle, gdf, sym_edges):
    g = needle.graph

    tracheid_nodes = _cell_nodes_by_type(g, "transfusion tracheid")
    n_tracheid_with_no_plasmodesmata = sum(
        1 for nd in tracheid_nodes
        if not any(g.edges[nd, nb].get("path") == "plasmodesmata" for nb in g.neighbors(nd))
    )

    bridge_report = getattr(needle, "_bridge_report", {})
    n_virtual_shared = bridge_report.get("n_virtual_shared", 0)
    n_virtual_per_path = bridge_report.get("n_virtual_per_path", 0)
    n_paths = bridge_report.get("n_paths", 0)

    relevant_nodes = [nd for nd, d in g.nodes(data=True) if d.get("cell_type") in SYMPLASTIC_CONTINUITY_TYPES]
    sym_graph = nx.Graph()
    sym_graph.add_nodes_from(relevant_nodes)
    sym_graph.add_edges_from(sym_edges)
    n_components = nx.number_connected_components(sym_graph)

    cgroup_hist = {}
    for _, d in g.nodes(data=True):
        if d.get("type") == "cell":
            cgroup_hist[d.get("cgroup")] = cgroup_hist.get(d.get("cgroup"), 0) + 1

    print()
    print("=== Transfusion network report ===")
    print(f"tracheid cells:                              {len(tracheid_nodes)}")
    print(f"tracheids with zero plasmodesmata edges:      {n_tracheid_with_no_plasmodesmata} "
          f"(must equal tracheid count: {'OK' if n_tracheid_with_no_plasmodesmata == len(tracheid_nodes) else 'MISMATCH'})")
    print(f"bridge paths accepted (source->target):       {n_paths}")
    print(f"virtual nodes -- shared per tracheid:          {n_virtual_shared}")
    print(f"virtual nodes -- one per path (no sharing):    {n_virtual_per_path}")
    print(f"symplast connected components (target types): {n_components} "
          f"(must be 1: {'OK' if n_components == 1 else 'MISMATCH'})")
    print("cgroup histogram (cell nodes):")
    for cg in sorted(cgroup_hist, key=lambda k: (k is None, k)):
        print(f"    cgroup {cg}: {cgroup_hist[cg]}")


def main(show=False):
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=== Pine (Pinus pinaster) needle -- transfusion network ===")
    needle = NeedleAnatomy(build_pinaster(), seed=SEED)
    needle.export_to_adjencymatrix()
    gdf = needle.generate_cells()

    plot_transfusion_cells(needle, gdf, show=show)
    sym_edges = plot_symplast_subgraph(needle, gdf, show=show)
    plot_symplast_zoom(needle, gdf, show=show)
    plot_apoplastic_isolation(needle, gdf, show=show)
    print_report(needle, gdf, sym_edges)


if __name__ == "__main__":
    main(show=False)
