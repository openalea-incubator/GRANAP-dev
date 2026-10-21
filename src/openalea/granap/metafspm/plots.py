"""
Drawing GRANAP's sections from a DataStructure (its Compartments' tissues and positions, their templates' polygons).

    draw_section(ax, ds, segment)                 one section's cells, coloured by tissue (or by a node variable)
    draw_sections(ds, titles=None)                every section of the DataStructure, one panel each
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


def draw_section(ax, ds, segment, values=None, title=None, cmap="viridis", norm=None, offset=(0., 0.)):
    """
    The segment's cells (its template's polygons, mm, shifted by *offset*), coloured by tissue, or by *values* (one per
    DataStructure node, e.g. ds.get("psi")). Returns the PatchCollection.
    """
    g = ds.mtg
    template = template_by_index(g.property("anatomy_template")[int(segment)])
    owners = np.asarray(ds.entity_ids("SubOrgan"))[ds.owner("SubOrgan")]
    nodes = np.flatnonzero((owners == int(segment)) & (np.asarray(ds.get("kind")) == 0))
    vids = ds.entity_ids("node")[nodes]
    row, tissue = g.property("template_row"), np.asarray(ds.get("tissue"), dtype=object)
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


def draw_sections(ds, titles=None, columns=4, panel=(3.6, 3.6)):
    """Every section of *ds* (its SubOrgans with an anatomy), coloured by tissue, with the tissues' legend; *panel*:
    each panel's (width, height), inches."""
    g = ds.mtg
    template_of = g.property("anatomy_template")
    segments = [int(s) for s in ds.entity_ids("SubOrgan") if template_of.get(int(s), -1.) >= 0]
    columns = min(columns, len(segments))
    rows = -(-len(segments) // columns)
    width, height = panel
    legend = 1.0                                                         # inches below the panels, for the legend
    fig, axes = plt.subplots(rows, columns, figsize=(width * columns, height * rows + legend), squeeze=False)
    for ax, segment, title in zip(axes.flat, segments, titles or [f"segment {s}" for s in segments]):
        draw_section(ax, ds, segment, title=title)
    for ax in axes.flat[len(segments):]:
        ax.axis("off")
    fig.tight_layout(rect=(0, legend / (height * rows + legend), 1, 1))
    tissue_legend(fig)
    return fig
