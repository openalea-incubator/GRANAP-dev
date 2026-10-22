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
import pandas as pd

from openalea.metafspm.coupling.component import StructuralComponent
from openalea.metafspm.coupling.composite_wrapper import CompositeModel
from openalea.metafspm.coupling.translator import Translator
from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.metafspm.scene.scene import Scene

from openalea.granap.metafspm.component import GranapAnatomy, OrganAnatomy, junction_rules
from openalea.granap.metafspm.library import AnatomyLibrary, template_by_index
from openalea.granap.metafspm.plots import draw_plant, tissue_legend

SEGMENT = 5e-3                                               # m
STEM_RADIUS, ROOT_RADIUS, LATERAL_RADIUS = 2.5e-3, 0.55e-3, 0.3e-3


def _organ(g, plant, kind, organ_label):
    """An organ of a new axis of *plant* (kind: "Shoot" or "Root")."""
    s, L = g.scales, g.labels
    complex_ = plant
    for scale, labels, name in ((s.Axis, L.Axis, kind), (s.GrowthUnit, L.GrowthUnit, kind),
                                (s.Phytomer, L.Phytomer, kind), (s.Organ, L.Organ, organ_label)):
        complex_ = g.add_component(complex_, **PropsConfig(scale=scale, edge_type='/', label=getattr(labels, name)))
    return complex_


def _chain(g, organ, n, label, radius, bearer=None):
    """n SubOrgan elements in *organ*, the first branched ('+') on *bearer* when given, then each after the previous."""
    s = g.scales
    props = PropsConfig(scale=s.SubOrgan, label=label, edge_type='+' if bearer is not None else '/')
    elements = [g.add_component_with_topo(organ, bearer, **props) if bearer is not None else
                g.add_component(organ, **props)]
    for _ in range(n - 1):
        elements.append(g.add_child(elements[-1], **PropsConfig(scale=s.SubOrgan, edge_type='<', label=label)))
    for rank, v in enumerate(elements):
        g.property("radius")[v], g.property("length")[v] = radius, SEGMENT
        g.property("distance_from_tip")[v] = (n - 1 - rank) * SEGMENT
    return elements


class SeedlingArchitecture(StructuralComponent):
    """The architecture: the stem, two leaves, the roots from the stem's base."""

    @classmethod
    def initiate_plant(cls, g, plant, parameters):
        L = g.labels.SubOrgan
        stem = _chain(g, _organ(g, plant, "Shoot", "StemInternode"), 3, L.StemElement, STEM_RADIUS)
        for bearer in stem[1:]:
            _chain(g, _organ(g, plant, "Shoot", "Leaf"), 2, L.LeafElement, STEM_RADIUS, bearer=bearer)
        main = _chain(g, _organ(g, plant, "Root", "RootInternode"), 3, L.RootSegment, ROOT_RADIUS, bearer=stem[0])
        _chain(g, _organ(g, plant, "Root", "RootInternode"), 2, L.RootSegment, LATERAL_RADIUS, bearer=main[1])


class SeedlingAnatomy(GranapAnatomy):
    """GRANAP's anatomies, the thinner roots (the lateral, diameter under 0.9 mm) with a thinner cortex (the advanced
    hook: a class's own parameters). Stems and leaves: the scenario's stem_anatomy and leaf_anatomy."""
    root = OrganAnatomy(AnatomyLibrary(class_parameters=lambda key: {"cortex": {"n_layers": 2}} if key[0] < 0.9
                                       else {}))


class Seedling(CompositeModel):
    initiators = (SeedlingArchitecture, SeedlingAnatomy)
    from_scale, nodes, wiring = "SubOrgan", "Compartment", junction_rules

    def __init__(self, data_structure, time_step):
        self.anatomy = SeedlingAnatomy(data_structure=data_structure, time_step=time_step)
        self.declare_data_and_couple_components(translator_path=Translator(), components=(self.anatomy,))

    def run(self):
        self.anatomy()


def layout(g, segments):
    """Offsets (mm): the stem going up, each leaf to the right of its linked parent, the roots going down, a
    lateral root to the side of its bearer; spaced by the sections' sizes."""
    L = g.labels.SubOrgan
    label = g.property("label")
    template_of = g.property("anatomy_template")

    def size(s):
        bounds = np.array([p.bounds for p in template_by_index(template_of[s]).polygons if p is not None])
        return max(bounds[:, 2].max() - bounds[:, 0].min(), bounds[:, 3].max() - bounds[:, 1].min())

    offsets = {}
    for s in segments:                                             # linked parents come first (smaller vids)
        parent = g.linked_parent(s, g.scales.SubOrgan)
        if parent is None:
            offsets[s] = np.zeros(2)
            continue
        gap = 0.6 * (size(s) + size(parent))
        if label[s] == L.StemElement:
            step = (0., gap)
        elif label[s] == L.LeafElement or (label[parent] == L.RootSegment and g.property("edge_type").get(s) == "+"):
            step = (gap, 0.)                                       # a leaf, or a lateral root: to the side
        else:
            step = (0., -gap)                                      # down the roots
        offsets[s] = offsets[parent] + np.array(step)
    return offsets


def main():
    scene = Scene(pd.DataFrame([dict(plant="seedling", model=Seedling, x=0., y=0., z=0., rotation=0.,
                                     scenario={"parameters": {"stem_anatomy": {}, "leaf_anatomy": {}}})]))
    ds = scene.populations[0].data_structure
    g = ds.mtg
    segments = [int(s) for s in ds.entity_ids("SubOrgan")]
    offsets = layout(g, segments)

    fig, ax = plt.subplots(figsize=(15, 10))
    draw_plant(ax, ds, offsets, linewidth=0.15, alpha=0.25)
    names = {g.labels.SubOrgan.StemElement: "stem", g.labels.SubOrgan.LeafElement: "leaf",
             g.labels.SubOrgan.RootSegment: "root"}
    for s, (x, y) in offsets.items():
        ax.annotate(f"{names[g.property('label')[s]]} {s}", (x, y), ha="center", va="center", fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.6, lw=0))
    ax.margins(0.04, 0.06)
    ax.set_xticks([])
    ax.set_yticks([])
    n_templates = len(set(g.property("anatomy_template").values()))
    ax.set_title(f"Seedling: {len(segments)} sections ({n_templates} anatomy templates: stems, leaves, roots), "
                 f"{ds.n_nodes()} Compartments, {ds.n_edges()} Connections, "
                 f"{int((ds.get('network_path') == 5).sum())} axial junctions", fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    tissue_legend(fig)
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "seedling.png")
    fig.savefig(path, dpi=150)
    print(path)


if __name__ == "__main__":
    main()
