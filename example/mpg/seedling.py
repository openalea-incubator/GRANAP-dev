"""
A seedling populated with GRANAP anatomies of every kind through the MPG (devplan_metafspm.md, P3): a stem of three
elements (monocot stem preset), two leaves of two elements on its second and third elements (monocot leaf preset),
and roots from its base (a main root of three segments with a lateral, the root preset, the lateral's class with a
thinner cortex). GranapAnatomy gives every segment its kind's anatomy, from one library per kind.

Each segment's cross-section is drawn at its real size from the MPG (its Cells: their template's polygons, coloured
by tissue): the stem going up, the leaves to its right, the roots going down, and the vessel junctions between
successive sections (stem to stem, stem to leaf, stem to root, root to root) as lines between their conducting cells
(xylem in blue, phloem in red).

    python seedling.py            (from GRANAP/example/mpg; figure in ./outputs/seedling.png)
"""
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Polygon as PolygonPatch

from openalea.metafspm.coupling.component import StructuralComponent
from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.metafspm.data_structure.data_api import MPGDataStructure
from openalea.metafspm.scene.population import build_population

from openalea.granap.leaf_class import LeafAnatomy
from openalea.granap.stem_class import StemAnatomy
from openalea.granap.metafspm.component import GranapAnatomy, OrganAnatomy, junction_rules
from openalea.granap.metafspm.library import AnatomyLibrary, find_template, preset_parameters

SEGMENT = 5e-3                                               # m
STEM_RADIUS, ROOT_RADIUS, LATERAL_RADIUS = 2.5e-3, 0.55e-3, 0.3e-3
LATERAL = (0.6, 0., 0.)


def _chain(g, complex_, n, label, radius, bearer=None):
    """n SubOrgan elements of *label* in *complex_* (an Organ), the first linked to *bearer* ('+'), then '<'."""
    s = g.scales
    props = PropsConfig(scale=s.SubOrgan, label=label, edge_type='+' if bearer is not None else '/')
    first = g.add_component_with_topo(complex_, bearer, **props) if bearer is not None else \
        g.add_component(complex_, **props)
    elements = [first]
    for _ in range(n - 1):
        elements.append(g.add_child(elements[-1], **PropsConfig(scale=s.SubOrgan, edge_type='<', label=label)))
    for rank, v in enumerate(elements):
        g.property("radius")[v] = radius
        g.property("length")[v] = SEGMENT
        g.property("distance_from_tip")[v] = (n - 1 - rank) * SEGMENT
    return elements


def _organ(g, plant, axis_label, organ_label, shoot):
    s, L = g.scales, g.labels
    kind = "Shoot" if shoot else "Root"
    axis = g.add_component(plant, **PropsConfig(scale=s.Axis, edge_type='/', label=getattr(L.Axis, axis_label)))
    gu = g.add_component(axis, **PropsConfig(scale=s.GrowthUnit, edge_type='/', label=getattr(L.GrowthUnit, kind)))
    phytomer = g.add_component(gu, **PropsConfig(scale=s.Phytomer, edge_type='/', label=getattr(L.Phytomer, kind)))
    return g.add_component(phytomer, **PropsConfig(scale=s.Organ, edge_type='/', label=getattr(L.Organ, organ_label)))


class SeedlingArchitecture(StructuralComponent):
    """The architecture initiator: the stem, two leaves, the roots from the stem's base."""

    @classmethod
    def initiate_plant(cls, g, plant, parameters):
        L = g.labels.SubOrgan
        stem = _chain(g, _organ(g, plant, "Shoot", "StemInternode", True), 3, L.StemElement, STEM_RADIUS)
        for bearer in stem[1:]:
            _chain(g, _organ(g, plant, "Shoot", "Leaf", True), 2, L.LeafElement, STEM_RADIUS, bearer=bearer)
        main = _chain(g, _organ(g, plant, "Root", "RootInternode", False), 3, L.RootSegment, ROOT_RADIUS,
                      bearer=stem[0])
        _chain(g, _organ(g, plant, "Root", "RootInternode", False), 2, L.RootSegment, LATERAL_RADIUS, bearer=main[1])


class SeedlingAnatomy(GranapAnatomy):
    library = AnatomyLibrary(class_parameters={LATERAL: {"cortex": {"n_layers": 2}}})
    shoot = {"StemElement": OrganAnatomy(AnatomyLibrary(parameters=preset_parameters("for_monocot_stem"),
                                                        organ_class=StemAnatomy)),
             "LeafElement": OrganAnatomy(AnatomyLibrary(parameters=preset_parameters("for_monocot_leaf"),
                                                        organ_class=LeafAnatomy), size=None)}


def _extent(template, axis=None):
    """The section's size (mm): its larger side, or its width (axis 0) or height (axis 1)."""
    bounds = np.array([p.bounds for p in template.polygons if p is not None and not p.is_empty])
    sides = (bounds[:, 2].max() - bounds[:, 0].min(), bounds[:, 3].max() - bounds[:, 1].min())
    return max(sides) if axis is None else sides[axis]


def layout(g, segments, size):
    """Offsets: the stem going up from the origin, each leaf to the right of its bearer, the roots going down."""
    labels, L = g.property("label"), g.labels.SubOrgan
    parent = {s: g.parent(s) for s in segments}
    offsets = {}
    stem = sorted(s for s in segments if labels[s] == L.StemElement)
    y = 0.
    for s in stem:
        offsets[s] = np.array([0., y])
        y += size[s] / 2. + max(size[s] for s in stem) * 0.15 + size[s] / 2.
    pending = [s for s in segments if s not in offsets]
    while pending:
        for s in list(pending):
            p = parent[s]
            if p not in offsets:
                continue
            gap = 0.6 * (size[s] + size[p])
            if labels[s] == L.LeafElement:
                offsets[s] = offsets[p] + np.array([gap, 0.])
            elif labels[s] == L.RootSegment and labels[p] == L.StemElement:
                offsets[s] = offsets[p] + np.array([0., -gap])
            elif g.property("edge_type").get(s) == '+':
                offsets[s] = offsets[p] + np.array([gap, 0.])            # a lateral root, to the side
            else:
                offsets[s] = offsets[p] + np.array([0., -gap])
            pending.remove(s)
    return offsets


def main():
    start = time.perf_counter()
    table = pd.DataFrame([dict(plant="p0", model=None, x=0., y=0., z=0., rotation=0., scenario={"parameters": {}})])
    g, _ = build_population(table, initiators=(SeedlingArchitecture, SeedlingAnatomy))
    g.convert_properties_to_arraydict()
    ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan, nodes="Compartment", wiring=junction_rules(g))
    built = time.perf_counter() - start
    libraries = [organ.library for organ in SeedlingAnatomy.organs().values()]
    segments = sorted(int(v) for v in g.property("anatomy_class"))
    owners = g.compartments_by_owner(g.scales.SubOrgan)
    template, row = g.property("template"), g.property("template_row")
    tissue, family = g.property("tissue"), g.property("family")
    px, py = g.property("section_x"), g.property("section_y")
    template_of = {s: find_template(libraries, int(template[owners[s][0]])) for s in segments}
    size = {s: _extent(template_of[s]) for s in segments}
    offsets = layout(g, segments, size)
    types = sorted({tissue[v] for s in segments for v in owners[s] if v in tissue})
    palette = matplotlib.colormaps["tab20"]
    colours = {t: palette(i % 20) for i, t in enumerate(types)}

    fig, ax = plt.subplots(figsize=(15, 9.5))
    patches, faces = [], []
    for s in segments:
        polygons = template_of[s].polygons
        for v in owners[s]:
            if v not in row or v not in tissue:
                continue
            polygon = polygons[int(row[v])]
            if polygon is None or polygon.is_empty:
                continue
            for part in (polygon.geoms if polygon.geom_type == "MultiPolygon" else [polygon]):
                patches.append(PolygonPatch(np.asarray(part.exterior.coords) + offsets[s], closed=True))
                faces.append(colours[tissue[v]])
    ax.add_collection(PatchCollection(patches, facecolor=faces, edgecolor="0.3", linewidth=0.1, alpha=0.75))
    a, b = g.property("n_id_a"), g.property("n_id_b")
    section_of = {v: s for s in segments for v in owners[s]}
    lines = {0: [], 1: []}
    for e in g.junction_vids():
        u, v = int(a[e]), int(b[e])
        lines[int(family[u])].append([np.array([px[u], py[u]]) + offsets[section_of[u]],
                                      np.array([px[v], py[v]]) + offsets[section_of[v]]])
    for fam, colour in ((0, "tab:blue"), (1, "tab:red")):
        if lines[fam]:
            ax.add_collection(LineCollection(lines[fam], colors=colour, linewidths=0.12, alpha=0.25, zorder=0))
    names = {g.labels.SubOrgan.StemElement: "stem", g.labels.SubOrgan.LeafElement: "leaf",
             g.labels.SubOrgan.RootSegment: "root"}
    for s in segments:
        kind = names[g.property("label")[s]]
        height = _extent(template_of[s], axis=1)
        ax.annotate(f"{kind} {s}", offsets[s] + np.array([0., height / 2. + 0.06]), ha="center", va="bottom",
                    fontsize=8)
    ax.autoscale_view()
    ax.margins(0.04, 0.06)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("sections at their real size, spaced for drawing (mm)")
    handles = [matplotlib.patches.Patch(color=colours[t], label=t, alpha=0.75) for t in types]
    handles += [matplotlib.lines.Line2D([], [], color="tab:blue", label="xylem junctions"),
                matplotlib.lines.Line2D([], [], color="tab:red", label="phloem junctions")]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    n_templates = sum(len(library.templates) for library in libraries)
    ax.set_title(f"Seedling: {len(segments)} sections ({n_templates} anatomy templates: stems, leaves, roots), "
                 f"{ds.n_nodes()} Compartments, {ds.n_edges()} Connections, {len(g.junction_vids())} vessel "
                 f"junctions, built in {built:.0f} s", fontsize=10)
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "seedling.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(path)


if __name__ == "__main__":
    main()
