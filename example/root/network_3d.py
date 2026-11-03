"""Build the 3D hydraulic network of a monocot root segment and look at it.

The 3D segment stacks rows of every 2D cell along the root axis (Z), each cell
type at its own height (``axial_heights``); vessels without a height stay one
tube, metaxylem given one becomes vessel elements in series.  The network is the
3D counterpart of the 2D wall / junction / cell graph (doc/NETWORK_3D_PLAN.md):

  walls      side (lateral), end (between rows) and face (segment ends)
  junctions  vertical (junction lines) and rim (end wall meets side walls)
  edges      membrane, wall, plasmodesmata -- each lateral or axial

Writes the segment as an OBJ in ``output/`` and shows a longitudinal slice: the
cell rows near the y = 0 plane, joined by their axial and lateral plasmodesmata.
"""

import os
import sys
from collections import Counter

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(".."))

from openalea.granap.root_class import RootAnatomy
from openalea.granap.input_data import OrganInputData

SEED = 0
OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# Row height per cell type (mm).  Cells sharing the stele layer differ; a type
# left out falls back to its layer's axial_height, else 5 x its cell diameter.
AXIAL_HEIGHTS = {
    "epidermis": 0.10, "exodermis": 0.12, "cortex": 0.15, "endodermis": 0.12,
    "pericycle": 0.10, "stele": 0.08,
    "metaxylem": 0.40,          # vessel elements in series
    "protoxylem": 0.30,
    # "phloem" left out: one tube over the whole segment
}
# Side-wall thickness per type (same unit as the 2D network), and the thinner
# end walls (perforation plates between vessel elements).
CELL_WALL_THICKNESS = {"default": 1.0, "epidermis": 2.0, "exodermis": 2.0, "endodermis": 1.5,
                       "metaxylem": 2.0, "protoxylem": 1.5, "outerwall": 2.0, "air space": 0.001}
END_WALL_THICKNESS = {"default": 1.0, "metaxylem": 0.1, "protoxylem": 0.1, "air space": 0.001}


def build_network():
    root = RootAnatomy(OrganInputData.for_root(), seed=SEED)
    return root.export_3d_network(axial_heights=AXIAL_HEIGHTS, segment_length=0.8, seed=SEED,
                                  cell_wall_thickness=CELL_WALL_THICKNESS,
                                  end_wall_thickness=END_WALL_THICKNESS)


def summary(net) -> None:
    g = net.graph
    print(f"segment {net.result.z_max - net.result.z_min:.2f} mm: "
          f"{net.n_walls} walls, {net.n_junctions} junctions, {net.n_cells} cells, "
          f"{g.number_of_edges()} edges")
    print("  nodes:", dict(Counter(k for _, k in g.nodes(data="kind") if k)))
    print("  edges:", dict(Counter((d["path"], d["orientation"]) for *_, d in g.edges(data=True))))
    print("  heights from:", net.result.height_sources)


def fill_conductivities(net) -> None:
    """Different symplastic conductivity across side walls and end walls."""
    net.export_to_adjencymatrix()
    net.fill_matrix_from_dict({"cortex": 1e-12, "stele": 1e-12}, "symplastic", orientation="lateral")
    net.fill_matrix_from_dict({"cortex": 5e-12, "stele": 5e-12}, "symplastic", orientation="axial")


def plot_slice(net, half_width: float = 0.02):
    """Cell nodes within ``half_width`` of the y = 0 plane, x against z."""
    g = net.graph
    near = {n for n, d in g.nodes(data=True)
            if d["type"] == "cell" and abs(d["position"][1]) < half_width}
    types = sorted({g.nodes[n]["cell_type"] for n in near})
    colours = dict(zip(types, plt.cm.tab10.colors))

    fig, ax = plt.subplots(figsize=(10, 6))
    for u, v, d in g.edges(data=True):
        if d["path"] == "plasmodesmata" and u in near and v in near:
            (x0, _, z0), (x1, _, z1) = g.nodes[u]["position"], g.nodes[v]["position"]
            ax.plot([x0, x1], [z0, z1], lw=0.6,
                    color="0.2" if d["orientation"] == "axial" else "0.75", zorder=1)
    for t in types:
        pts = [g.nodes[n]["position"] for n in near if g.nodes[n]["cell_type"] == t]
        ax.scatter([p[0] for p in pts], [p[2] for p in pts], s=10, color=colours[t],
                   label=t, zorder=2)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z, along the root (mm)")
    ax.set_title("Longitudinal slice: cell rows and plasmodesmata (dark = axial)")
    ax.legend(markerscale=2, fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig


def main():
    net = build_network()
    summary(net)
    fill_conductivities(net)
    os.makedirs(OUTPUT, exist_ok=True)
    path = os.path.join(OUTPUT, "root_network_3d.obj")
    net.result.export_obj(path)
    print("OBJ written to", path)
    plot_slice(net)
    plt.show()


if __name__ == "__main__":
    main()
