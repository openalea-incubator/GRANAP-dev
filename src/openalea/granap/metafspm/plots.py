"""
Drawing GRANAP's sections from a DataStructure (its Compartments' tissues and positions, their templates' polygons).

    draw_section(ax, data_structure, segment)                 one section's cells, coloured by tissue (or by a node variable)
    draw_sections(data_structure, titles=None)                every section of the DataStructure, one panel each
    draw_plant(ax, data_structure, offsets)                   every section at its place, with the axial junctions
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as PolygonPatch

from openalea.granap.metafspm.library import template_by_index

_PALETTE = [matplotlib.colormaps[name](i) for name, n in (("tab20", 20), ("tab20b", 20), ("Set3", 12)) for i in range(n)]
TISSUE_COLOURS = {}                                    # one colour per tissue, kept across the figures


def tissue_colour(tissue):
    return TISSUE_COLOURS.setdefault(tissue, _PALETTE[len(TISSUE_COLOURS) % len(_PALETTE)])


def draw_section(ax, data_structure, segment, values=None, title=None, cmap="viridis", norm=None, offset=(0., 0.)):
    """
    The segment's cells (its template's polygons, mm, shifted by *offset*), coloured by tissue, or by *values* (one per
    DataStructure node, e.g. data_structure.get("psi")). Returns the PatchCollection.
    """
    mtg = data_structure.mtg
    template = template_by_index(data_structure, data_structure.get("anatomy_template")[data_structure.index_of([int(segment)], "SubOrgan")[0]])
    owners = np.asarray(data_structure.entity_ids("SubOrgan"))[data_structure.owner("SubOrgan")]
    nodes = np.flatnonzero((owners == int(segment)) & (np.asarray(data_structure.get("compartment_type")) == 0))
    vids = data_structure.entity_ids("node")[nodes]
    row, tissue = mtg.view("template_row"), np.asarray(data_structure.get("tissue"), dtype=object)
    patches, keep = [], []
    for i, v in zip(nodes, vids):
        polygon = template.polygons[int(row[int(v)])]
        if polygon is not None and polygon.geom_type == "Polygon":
            patches.append(PolygonPatch(np.asarray(polygon.exterior.coords) + np.asarray(offset)))
            keep.append(i)
    if values is None:
        collection = PatchCollection(patches, facecolor=[tissue_colour(tissue[i]) for i in keep], edgecolor="0.3",
                                     linewidth=0.1)
    else:
        collection = PatchCollection(patches, cmap=cmap, norm=norm, edgecolor="none")
        collection.set_array(np.asarray(values, dtype=float)[keep])
    ax.add_collection(collection)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title, fontsize=9)
    return collection


def tissue_legend(fig, ncol=7):
    fig.legend(handles=[matplotlib.patches.Patch(color=c, label=t) for t, c in TISSUE_COLOURS.items()],
               loc="lower center", ncol=ncol, fontsize=7)


def draw_sections(data_structure, titles=None, columns=4, panel=(3.6, 3.6)):
    """Every section of *data_structure* (its SubOrgans with an anatomy), coloured by tissue, with the tissues' legend; *panel*:
    each panel's (width, height), inches."""
    segments = [int(s) for s in np.asarray(data_structure.entity_ids("SubOrgan"))[np.asarray(data_structure.get("anatomy_template")) >= 0]]
    columns = min(columns, len(segments))
    rows = -(-len(segments) // columns)
    width, height = panel
    legend = 1.0                                                         # inches below the panels, for the legend
    fig, axes = plt.subplots(rows, columns, figsize=(width * columns, height * rows + legend), squeeze=False)
    for ax, segment, title in zip(axes.flat, segments, titles or [f"segment {s}" for s in segments]):
        draw_section(ax, data_structure, segment, title=title)
    for ax in axes.flat[len(segments):]:
        ax.axis("off")
    fig.tight_layout(rect=(0, legend / (height * rows + legend), 1, 1))
    tissue_legend(fig)
    return fig


def draw_plant(ax, data_structure, offsets, junctions=True, linewidth=0.3, alpha=0.5):
    """
    Every section of *data_structure* at its drawing offset (*offsets*: {segment: (dx, dy)} mm, the segments to draw), coloured
    by tissue, and the axial junctions between them as lines (xylem black, phloem magenta).
    """
    for segment, offset in offsets.items():
        draw_section(ax, data_structure, segment, offset=offset)
    if junctions:
        view, owner = data_structure.to_graph_view(), data_structure.owner("SubOrgan")
        segments = np.asarray(data_structure.entity_ids("SubOrgan"))
        shift = np.array([offsets.get(int(s), (np.nan, np.nan)) for s in segments], dtype=float)
        x, y, tissue = data_structure.get("section_x"), data_structure.get("section_y"), np.asarray(data_structure.get("tissue"), dtype=object)
        e = np.flatnonzero(data_structure.get("network_path") == 5)
        a, b = view.tail[e], view.head[e]
        for xylem, colour in ((True, "black"), (False, "magenta")):
            chosen = np.isin(tissue[a], ("metaxylem", "protoxylem", "xylem")) == xylem
            ends = np.stack([np.c_[x[a] + shift[owner[a], 0], y[a] + shift[owner[a], 1]],
                             np.c_[x[b] + shift[owner[b], 0], y[b] + shift[owner[b], 1]]], axis=1)[chosen]
            ax.add_collection(matplotlib.collections.LineCollection(ends[np.isfinite(ends).all(axis=(1, 2))],
                                                                    colors=colour, linewidths=linewidth, alpha=alpha,
                                                                    zorder=3))
    ax.autoscale_view()
    ax.set_aspect("equal")
