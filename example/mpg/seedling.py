"""
A seedling populated with GRANAP anatomies of every kind: a stem of three elements (monocot stem preset), two leaves
of two elements on its second and third elements (monocot leaf preset), and roots from its base (a main root of three
segments with a lateral, the root preset, the lateral's class with a thinner cortex). The plant's scenario turns on
the stems' and leaves' anatomies (stem_anatomy, leaf_anatomy).

Each segment's cross-section is drawn at its real size from the plant graph, the stem going up, the leaves to its
right, the roots going down, with the axial junctions between linked sections (stem to stem, stem to leaf, stem to
root, root to root) as lines between their conducting cells (xylem black, phloem magenta).

    python seedling.py            (from GRANAP/example/mpg; figure in ./outputs/seedling.png)
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from openalea.metafspm import CompositeModel, Planting, planting, Scene, Translator

from openalea.granap.metafspm.architectures import SegmentArchitecture, segment_chain
from openalea.granap.metafspm.component import GranapAnatomy, OrganAnatomy, junction_rules
from openalea.granap.metafspm.library import AnatomyLibrary, template_by_index
from openalea.granap.metafspm.plots import draw_plant, tissue_legend

SEGMENT = 5e-3                                               # m
STEM_RADIUS, ROOT_RADIUS, LATERAL_RADIUS = 2.5e-3, 0.55e-3, 0.3e-3
LEAF_RADIUS = 2e-3                                           # half the monocot leaf preset's 4 mm width


def _elements(n, radius):
    """n elements of *radius*, base to tip: [(radius, length, distance from the tip)]."""
    return [(radius, SEGMENT, (n - 1 - rank) * SEGMENT) for rank in range(n)]


class SeedlingArchitecture(SegmentArchitecture):
    """The architecture: the stem, two leaves, the roots from the stem's base (its elements' radius, length and
    distance from the tip, SegmentArchitecture's variables)."""

    @classmethod
    def initiate_plant(cls, mtg, plant, parameters):
        L = mtg.labels.SubOrgan
        stem = segment_chain(mtg, mtg.add_organ(plant, "StemInternode", complex_label="Shoot"), L.StemElement,
                             _elements(3, STEM_RADIUS))
        for bearer in stem[1:]:
            segment_chain(mtg, mtg.add_organ(plant, "Leaf", complex_label="Shoot"), L.LeafElement,
                          _elements(2, LEAF_RADIUS), bearer=bearer)
        main = segment_chain(mtg, mtg.add_organ(plant, "RootInternode", complex_label="Root"), L.RootSegment,
                             _elements(3, ROOT_RADIUS), bearer=stem[0])
        segment_chain(mtg, mtg.add_organ(plant, "RootInternode", complex_label="Root"), L.RootSegment,
                      _elements(2, LATERAL_RADIUS), bearer=main[1])


class SeedlingAnatomy(GranapAnatomy):
    """GRANAP's anatomies, the thinner roots (the lateral, diameter under 0.9 mm) with a thinner cortex (the advanced
    hook: a class's own parameters). Stems and leaves: the scenario's stem_anatomy and leaf_anatomy."""
    root = OrganAnatomy(AnatomyLibrary(class_parameters=lambda key: {"cortex": {"n_layers": 2}} if key[0] < 0.9
                                       else {}))


class Seedling(CompositeModel):
    """The seedling's architecture and anatomies. Its translator links the stems' class input, stem_radius, to the
    architecture's radius (applied at initiation too, before the anatomies are built); the other inputs are shared
    by name."""
    initiators = (SeedlingArchitecture, SeedlingAnatomy)
    from_scale, nodes, wiring = "SubOrgan", "Compartment", junction_rules
    translator = Translator().link("SeedlingAnatomy", "stem_radius", "SeedlingArchitecture", {"radius": 1.})

    parts = {"architecture": SeedlingArchitecture, "anatomy": SeedlingAnatomy}


def layout(data_structure, segments):
    """Offsets (mm): the stem going up, each leaf to the right of its linked parent, the roots going down, a
    lateral root to the side of its bearer; spaced by the sections' sizes."""
    mtg = data_structure.structure
    L = mtg.labels.SubOrgan
    label = mtg.view("label")
    template_of = dict(zip(data_structure.entity_ids("SubOrgan").tolist(), data_structure.get("anatomy_template")))

    def size(s):
        bounds = np.array([p.bounds for p in template_by_index(data_structure, template_of[s]).polygons if p is not None])
        return max(bounds[:, 2].max() - bounds[:, 0].min(), bounds[:, 3].max() - bounds[:, 1].min())

    offsets = {}
    for s in segments:                                             # linked parents come first (smaller vids)
        parent = mtg.linked_parent(s, mtg.scales.SubOrgan)
        if parent is None:
            offsets[s] = np.zeros(2)
            continue
        gap = 0.6 * (size(s) + size(parent))
        if label[s] == L.StemElement:
            step = (0., gap)
        elif label[s] == L.LeafElement or (label[parent] == L.RootSegment and mtg.view("edge_type").get(s) == "+"):
            step = (gap, 0.)                                       # a leaf, or a lateral root: to the side
        else:
            step = (0., -gap)                                      # down the roots
        offsets[s] = offsets[parent] + np.array(step)
    return offsets


def main():
    scene = Scene(planting([Planting()], Seedling, {"stem_anatomy": {}, "leaf_anatomy": {}}))
    data_structure = scene.populations[0].data_structure
    mtg = data_structure.mtg
    segments = [int(s) for s in data_structure.entity_ids("SubOrgan")]
    offsets = layout(data_structure, segments)

    fig, ax = plt.subplots(figsize=(15, 10))
    draw_plant(ax, data_structure, offsets, linewidth=0.15, alpha=0.25)
    names = {mtg.labels.SubOrgan.StemElement: "stem", mtg.labels.SubOrgan.LeafElement: "leaf",
             mtg.labels.SubOrgan.RootSegment: "root"}
    for s, (x, y) in offsets.items():
        ax.annotate(f"{names[mtg.view('label')[s]]} {s}", (x, y), ha="center", va="center", fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.6, lw=0))
    ax.margins(0.04, 0.06)
    ax.set_xticks([])
    ax.set_yticks([])
    n_templates = len(set(data_structure.get("anatomy_template").tolist()))
    ax.set_title(f"Seedling: {len(segments)} sections ({n_templates} anatomy templates: stems, leaves, roots), "
                 f"{data_structure.n_nodes()} Compartments, {data_structure.n_edges()} Connections, "
                 f"{int((data_structure.get('network_path') == 5).sum())} axial junctions", fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    tissue_legend(fig)
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "seedling.png")
    fig.savefig(path, dpi=150)
    print(path)


if __name__ == "__main__":
    main()
