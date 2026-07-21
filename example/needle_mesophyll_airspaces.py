"""Diagnostic: needle mesophyll intercellular air spaces.

Builds a needle cross-section and visualizes the mesophyll intercellular air
spaces, which are inserted as small wall-centred rhombi (a four-vertex diamond,
principal diagonal parallel to the shared mesophyll walls, ~1/3 of the wall
length, 2:1 principal:secondary diagonal ratio).

Two diagnostic figures are produced, using the existing organ visualization
options only:

Figure 1 — cell geometry
    * left panel  : the standard full cross-section (``organ.plot_cells``);
    * right panel : the same section with only the mesophyll cells and the
                    ``air space`` lacunae highlighted, so the rhombic
                    wall-seated air spaces are easy to inspect.

Figure 2 — hydraulic network (``organ.plot_network``)
    Every network node (wall / junction / cell) and connection is drawn. The
    air-spaces contribute their own cell nodes, the shared edges with the
    surrounding mesophyll cells become wall nodes, and the points where each
    rhombus rim crosses an original mesophyll wall become junction nodes — so the
    lacunae are fully wired into the ``AbstractNetwork`` graph.
"""

import sys
import os
import matplotlib.pyplot as plt
import networkx as nx

sys.path.append(os.path.abspath('..'))

from openalea.granap.needle_class import NeedleAnatomy
from openalea.granap.input_data import OrganInputData


SEED = 0


def make_needle() -> NeedleAnatomy:
    """Build a needle with the default mesophyll intercellular air spaces."""
    data = OrganInputData.for_needle()
    data.set_value("resin_duct", "n_files", 2)
    data.set_value("stomata", "n_files", 10)
    return NeedleAnatomy(data, seed=SEED)


def air_space_summary(needle: NeedleAnatomy) -> None:
    """Print a short count/area summary of the generated air spaces."""
    cells = needle.all_cells.cells
    air = [c for c in cells if c.type == "air space" and c.polygon is not None]
    meso = [c for c in cells if c.type == "mesophyll" and c.polygon is not None]
    air_area = sum(c.polygon.area for c in air)
    meso_area = sum(c.polygon.area for c in meso)
    denom = air_area + meso_area
    print("Needle mesophyll air-space diagnostic")
    print(f"  mesophyll cells : {len(meso)}")
    print(f"  air-space cells : {len(air)}")
    if denom > 0:
        print(f"  air fraction    : {air_area / denom:.3f}  (air / (air + mesophyll))")


def plot_mesophyll_airspaces(needle: NeedleAnatomy, ax) -> None:
    """Highlight mesophyll cells (light) and their wall-seated air spaces (red)."""
    gdf = needle.generate_cells()

    meso = gdf[gdf["type"] == "mesophyll"]
    air = gdf[gdf["type"] == "air space"]

    if not meso.empty:
        meso.plot(ax=ax, facecolor="#cfe8c9", edgecolor="black", linewidth=0.5, alpha=0.7)
    if not air.empty:
        air.plot(ax=ax, facecolor="#d62728", edgecolor="black", linewidth=0.6, alpha=0.9)

    ax.set_aspect("equal", "box")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Mesophyll (green) + wall air spaces (red)")


def network_summary(needle: NeedleAnatomy) -> None:
    """Print how the air-space lacunae are wired into the network graph."""
    needle.export_to_adjencymatrix()
    g = needle.graph
    gdf = needle.generate_cells()

    base = needle.n_walls + needle.n_junctions
    type_by_node = {
        base + pos: gdf.loc[ridx, "type"] for pos, ridx in enumerate(gdf.index)
    }
    air_nodes = [n for n, t in type_by_node.items() if t == "air space"]

    # Walls that border both an air space and a mesophyll cell.
    shared_air_meso_walls = 0
    for an in air_nodes:
        for w in g.neighbors(an):
            if w < needle.n_walls:
                if any(
                    type_by_node.get(c) == "mesophyll"
                    for c in g.neighbors(w) if c >= base
                ):
                    shared_air_meso_walls += 1

    print("Network integration of air spaces")
    print(f"  wall nodes            : {needle.n_walls}")
    print(f"  junction nodes        : {needle.n_junctions}")
    print(f"  cell nodes            : {needle.n_cells}")
    print(f"  air-space cell nodes  : {len(air_nodes)}")
    if air_nodes:
        degrees = [g.degree(n) for n in air_nodes]
        print(f"  air-space node degree : min={min(degrees)} "
              f"median={int(sorted(degrees)[len(degrees) // 2])} max={max(degrees)}")
    print(f"  air/mesophyll walls   : {shared_air_meso_walls}")


def _air_space_network_elements(needle: NeedleAnatomy):
    """Return the nodes and edges of the network that belong to the air spaces.

    Collects, for every ``air space`` cell node:
      * the cell node itself,
      * the wall nodes on its boundary and the junction nodes those walls end at,
      * every edge incident to the air-space cell (cell <-> wall) and the
        wall <-> junction edges of those walls.

    Returns ``(air_cell_nodes, wall_nodes, junction_nodes, highlight_edges)``.
    """
    g = needle.graph
    n_walls = needle.n_walls
    base = n_walls + needle.n_junctions

    gdf = needle.generate_cells()
    air_cell_nodes = {
        base + pos
        for pos, ridx in enumerate(gdf.index)
        if gdf.loc[ridx, "type"] == "air space" and (base + pos) in g
    }

    wall_nodes: set = set()
    junction_nodes: set = set()
    highlight_edges: set = set()

    for an in air_cell_nodes:
        for w in g.neighbors(an):
            highlight_edges.add((an, w))
            if w < n_walls:                      # wall node
                wall_nodes.add(w)
                for jn in g.neighbors(w):        # its junction endpoints
                    if n_walls <= jn < base:
                        junction_nodes.add(jn)
                        highlight_edges.add((w, jn))

    return air_cell_nodes, wall_nodes, junction_nodes, highlight_edges


def plot_network_diagnostic(needle: NeedleAnatomy, show=True):
    """Draw the full network, then highlight the air-space nodes and edges."""
    needle.export_to_adjencymatrix()
    g = needle.graph
    position = nx.get_node_attributes(g, "position")

    fig, ax = plt.subplots(figsize=(12, 10))

    # Full network, faded, for context.
    needle.plot_network(ax=ax, show=False, node_size=4, width=0.3, alpha=0.15)

    # Air-space-related nodes and edges.
    air_cells, wall_nodes, junc_nodes, edges = _air_space_network_elements(needle)

    # Highlighted edges (air-space walls and their junction connections).
    nx.draw_networkx_edges(
        g, position, ax=ax, edgelist=list(edges),
        edge_color="#d62728", width=1.4, alpha=0.9,
    )

    # Highlighted nodes: cell (red), wall (orange), junction (blue).
    nx.draw_networkx_nodes(
        g, position, ax=ax, nodelist=list(air_cells),
        node_color="#d62728", node_size=26, label="air-space cell",
    )
    nx.draw_networkx_nodes(
        g, position, ax=ax, nodelist=list(wall_nodes),
        node_color="#ff7f0e", node_size=14, label="air-space wall",
    )
    nx.draw_networkx_nodes(
        g, position, ax=ax, nodelist=list(junc_nodes),
        node_color="#1f77b4", node_size=10, label="air-space junction",
    )

    ax.legend(loc="upper right", scatterpoints=1, framealpha=0.9)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        "Needle hydraulic network — air-space nodes/edges highlighted "
        f"({len(air_cells)} lacunae)"
    )
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def main(show=True):
    needle = make_needle()
    # Force the build so all_cells is populated for the summary/highlight panel.
    needle.generate_cells()
    air_space_summary(needle)
    network_summary(needle)

    # Figure 1 — cell geometry.
    fig_cells, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(18, 9))
    needle.plot_cells(ax=ax_left, show=False, title="Needle cross-section (all tissues)")
    plot_mesophyll_airspaces(needle, ax_right)
    fig_cells.suptitle("Needle mesophyll intercellular air spaces (wall-centred rhombi)")
    fig_cells.tight_layout()

    # Figure 2 — hydraulic network (nodes + connections).
    fig_network = plot_network_diagnostic(needle, show=False)

    if show:
        plt.show()
    return fig_cells, fig_network


if __name__ == "__main__":
    main(show=True)
