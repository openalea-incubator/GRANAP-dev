"""
A root system populated with GRANAP anatomies through the MPG (devplan_metafspm.md, P3): a main root of four
segments and a lateral of two on its second segment, the lateral's class parametrized with a thinner cortex.

Each segment's cross-section is drawn from the MPG: its Cells (their template's polygons, coloured by tissue), placed
along its axis (the main root on the top row, the lateral below its bearer), and the vessel junctions between
successive sections as lines between their conducting cells (xylem in blue, phloem in red). The title gives the
whole system's graph: Compartments, Connections and junctions.

    python root_system.py            (from GRANAP/example/mpg; figure in ./outputs/root_system.png)
"""
import os
import sys
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Polygon as PolygonPatch

from openalea.metafspm.data_structure.data_api import MPGDataStructure
from openalea.metafspm.scene.population import build_population

from openalea.granap.mpg.component import GranapAnatomy, junction_rules
from openalea.granap.mpg.library import AnatomyLibrary

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "test", "mpg_capabilities"))
from roots import SmallRootSystem, planting                                          # noqa: E402

LATERAL = (0.6, 0., 0.)
SPACING = 1.4                                    # mm between the sections' centres


class Anatomy(GranapAnatomy):
    library = AnatomyLibrary(class_parameters={LATERAL: {"cortex": {"n_layers": 2}}})


def layout(g, segments):
    """The drawing offset (mm) of each segment's section: the main root on a row, a lateral below its bearer."""
    offsets, column = {}, {}
    main = [s for s in segments if g.parent(s) not in segments]
    main_axis = max(main, key=lambda s: g.property("radius")[s])
    chain = [main_axis]
    while True:
        children = [c for c in g.children(chain[-1]) if c in segments and g.property("edge_type").get(c) == '<']
        if not children:
            break
        chain.append(children[0])
    for i, s in enumerate(chain):
        offsets[s], column[s] = np.array([i * SPACING, 0.]), i
    for s in segments:
        if s in offsets:
            continue
        depth, v = 1, s
        while g.parent(v) not in offsets:
            v, depth = g.parent(v), depth + 1
        bearer = g.parent(v)
        offsets[s] = offsets[bearer] + np.array([0., -depth * SPACING])
    return offsets


def main():
    start = time.perf_counter()
    g, _ = build_population(planting(), initiators=(SmallRootSystem, Anatomy))
    g.convert_properties_to_arraydict()
    ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan, nodes="Compartment", wiring=junction_rules(g))
    built = time.perf_counter() - start
    segments = sorted(int(v) for v in g.property("anatomy_class"))
    offsets = layout(g, segments)
    owners = g.compartments_by_owner(g.scales.SubOrgan)
    template, row = g.property("template"), g.property("template_row")
    tissue, family, position = g.property("tissue"), g.property("family"), g.property("node__position")
    templates = {t.index: t for t in Anatomy.library.templates.values()}
    types = sorted({tissue[v] for s in segments for v in owners[s] if v in tissue})
    palette = matplotlib.colormaps["tab20"]
    colours = {t: palette(i % 20) for i, t in enumerate(types)}

    fig, ax = plt.subplots(figsize=(15, 8.5))
    patches, faces = [], []
    for s in segments:
        for v in owners[s]:
            if v not in row or v not in tissue:
                continue                                       # walls and junctions: no polygon of their own
            polygon = templates[int(template[v])].polygons[int(row[v])]
            if polygon is None or polygon.is_empty:
                continue
            for part in (polygon.geoms if polygon.geom_type == "MultiPolygon" else [polygon]):
                patches.append(PolygonPatch(np.asarray(part.exterior.coords) + offsets[s], closed=True))
                faces.append(colours[tissue[v]])
        x, y = offsets[s]
        ax.annotate(f"segment {s}\nclass {g.property('anatomy_class')[s][0]:g} mm", (x, y + 0.45), ha="center",
                    fontsize=8)
    ax.add_collection(PatchCollection(patches, facecolor=faces, edgecolor="0.3", linewidth=0.15, alpha=0.75))
    a, b = g.property("n_id_a"), g.property("n_id_b")
    section_of = {v: s for s in segments for v in owners[s]}
    lines = {0: [], 1: []}
    for e in g.junction_vids():
        u, v = int(a[e]), int(b[e])
        lines[int(family[u])].append([np.asarray(position[u]) + offsets[section_of[u]],
                                 np.asarray(position[v]) + offsets[section_of[v]]])
    for fam, colour in ((0, "tab:blue"), (1, "tab:red")):
        if lines[fam]:
            ax.add_collection(LineCollection(lines[fam], colors=colour, linewidths=0.4, alpha=0.6))
    ax.autoscale_view()
    ax.margins(0.05, 0.12)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("along the root (sections spaced for drawing, mm)")
    handles = [matplotlib.patches.Patch(color=colours[t], label=t, alpha=0.75) for t in types]
    handles += [matplotlib.lines.Line2D([], [], color="tab:blue", label="xylem junctions"),
                matplotlib.lines.Line2D([], [], color="tab:red", label="phloem junctions")]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    ax.set_title(f"{len(segments)} segments, {len(Anatomy.library.templates)} anatomy classes: {ds.n_nodes()} "
                 f"Compartments, {ds.n_edges()} Connections ({len(g.junction_vids())} vessel junctions), built in "
                 f"{built:.1f} s")
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "root_system.png")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    print(path)


if __name__ == "__main__":
    main()
