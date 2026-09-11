import sys
import os
import matplotlib.pyplot as plt
import networkx as nx

# Add parent directory to path to allow importing anatomy package
sys.path.append(os.path.abspath('..'))

from openalea.granap.needle_class import NeedleAnatomy
from openalea.granap.input_data import OrganInputData
from openalea.granap.visualization import plot_layers_simple, plot_section


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


def build_gallery_needle_data() -> OrganInputData:
    """Build the ``OrganInputData`` for the feature-showcase needle.
    """
    # NeedlePlantTypeParams defaults (OrganInputData.for_needle() doesn't
    # override width/thickness), needed below to locate the corners.
    WIDTH, THICKNESS = 1.8, 1.1
    _, _, CORNER_POS, CORNER_NEG = NeedleAnatomy.pole_and_corner_angles(WIDTH, THICKNESS)

    # Configure the input data, then build once.
    data = OrganInputData.for_needle()
    data.set_value("resin_duct", "n_files", 2)
    data.set_value("resin_duct", "sheath_cell_diameter", 0.015)
    data.set_value("stomata", "n_files", 10)
    data.set_value("central_cylinder", "vascular_angle", 20)
    data.set_value("transfusion_tissue", "pack_circles", True)
    data.set_value("transfusion_tissue", "proportion", 0.85)
    data.set_value("transfusion_tissue", "parenchyma_diameter", 0.045)
    data.set_value("transfusion_tissue", "tracheids_diameter", 0.022)
    data.set_value("transfusion_tissue", "transfusion_tracheids_ratio", 1.5)

    # Append with hypodermis corner
    data.params.append({
        "name": "hypodermis_corner", "cell_diameter": 0.0225, "n_layers": 2, "order": 5.1,
        "thickness_profile": NeedleAnatomy.corner_bump_profile(
            [(CORNER_POS, 10.0, 1.0), (CORNER_NEG, 10.0, 1.0)], floor=0.70),
        "zone_angles": {"mode": "wedge", "centers": [CORNER_POS, CORNER_NEG],
                        "half_width": 4.0},
    })
    return data


def main(show=False):
    data = build_gallery_needle_data()
    needle = NeedleAnatomy(data)

    needle.plot_layers(show=show, title=f"Needle Layers")

    needle.plot_cells(show=show, title=f"Needle Cells")

    _ = needle.export_to_adjencymatrix()
    needle.plot_network(show=show, title="Needle Network")

    plot_air_link_network(needle, "Needle air-link network", show=show)

if __name__ == "__main__":
    main(show=True)
