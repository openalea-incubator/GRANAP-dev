"""Gallery: full feature set (layers / cells / network / air-link network) for
one monocot leaf and one dicot leaf.

Mirrors ``example/needle/gallery_needle_features.py``, built from the same
``OrganInputData.for_monocot_leaf()`` / ``.for_dicot_leaf()`` presets used by
``example/leaf/gallery_leaf.py``. Both presets already carry air-space tissues
(inter-bundle aerenchyma for the monocot, intercellular spaces in the spongy
mesophyll for the dicot, plus substomatal chambers for both), so the air-link
network highlight has something to show for each.
"""

import sys
import os
import matplotlib.pyplot as plt
import networkx as nx

# Add parent directory to path to allow importing anatomy package
sys.path.append(os.path.abspath('..'))

from openalea.granap.input_data import OrganInputData
from openalea.granap.leaf_class import LeafAnatomy

SEED = 0


def _air_space_network_elements(organ):
    """Return the nodes/edges of the network graph that belong to air-space cells.

    Collects, for every ``air space`` cell node: the cell node itself, the wall
    nodes on its boundary, the junction nodes those walls end at, and every
    edge incident to those (cell <-> wall, wall <-> junction).
    """
    g = organ.graph
    n_walls = organ.n_walls
    base = n_walls + organ.n_junctions

    gdf = organ.generate_cells()
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


def plot_air_link_network(organ, title, show=True):
    """Draw the full network faded, then highlight the air-space nodes/edges."""
    position = nx.get_node_attributes(organ.graph, "position")

    fig, ax = plt.subplots(figsize=(12, 10))

    # Full network, faded, for context.
    organ.plot_network(ax=ax, show=False, node_size=4, width=0.3, alpha=0.15)

    # Air-space-related nodes and edges.
    air_cells, wall_nodes, junc_nodes, edges = _air_space_network_elements(organ)

    nx.draw_networkx_edges(
        organ.graph, position, ax=ax, edgelist=list(edges),
        edge_color="#d62728", width=1.4, alpha=0.9,
    )
    nx.draw_networkx_nodes(
        organ.graph, position, ax=ax, nodelist=list(air_cells),
        node_color="#d62728", node_size=26, label="air-space cell",
    )
    nx.draw_networkx_nodes(
        organ.graph, position, ax=ax, nodelist=list(wall_nodes),
        node_color="#ff7f0e", node_size=14, label="air-space wall",
    )
    nx.draw_networkx_nodes(
        organ.graph, position, ax=ax, nodelist=list(junc_nodes),
        node_color="#1f77b4", node_size=10, label="air-space junction",
    )

    ax.legend(loc="upper right", scatterpoints=1, framealpha=0.9)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{title} ({len(air_cells)} air-space cells)")
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def run_leaf_features(label, data, show, radius):
    leaf = LeafAnatomy(data, seed=SEED)

    leaf.plot_layers(show=show, title=f"{label} Layers")

    leaf.plot_cells(show=show, title=f"{label} Cells")

    _ = leaf.export_to_adjencymatrix(air_link_radius = radius)
    leaf.plot_network(show=show, title=f"{label} Network")

    plot_air_link_network(leaf, f"{label} air-link network", show=show)


def main(show=False):
    run_leaf_features("Monocot Leaf", OrganInputData.for_monocot_leaf(), show, radius=0.1)
    run_leaf_features("Dicot Leaf", OrganInputData.for_dicot_leaf(), show, radius= 0.04)


if __name__ == "__main__":
    main(show=True)
