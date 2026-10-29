"""
A root system populated with GRANAP's anatomies: a main root of four segments and a lateral of two on its second
segment, the lateral's (thinner) class parametrized with a thinner cortex.

Each segment's cross-section is drawn from the plant graph (its cells, coloured by tissue), the main root on the top
row and the lateral below its bearer, with the axial junctions between successive sections as lines between their
conducting cells (xylem black, phloem magenta). Each segment is joined to its linked parent (the multiscale
traversal of the plant graph): along the main axis, its predecessor; the lateral's first segment, its bearer.

    python root_system.py            (from GRANAP/example/mpg; figure in ./outputs/root_system.png)
"""
import os

import matplotlib.pyplot as plt
import pandas as pd

from openalea.metafspm.coupling.composite_wrapper import CompositeModel
from openalea.metafspm.coupling.translator import Translator
from openalea.metafspm.scene.scene import Scene

from openalea.granap.metafspm.architectures import RootAxis
from openalea.granap.metafspm.component import GranapAnatomy, OrganAnatomy, junction_rules
from openalea.granap.metafspm.library import AnatomyLibrary
from openalea.granap.metafspm.plots import draw_plant, tissue_legend

SEGMENT, MAIN, LATERAL = 5e-3, 0.55e-3, 0.3e-3   # m
SPACING = 1.4                                    # mm between the sections' centres, for drawing


class RootSystemAnatomy(GranapAnatomy):
    """GRANAP's roots, the thinner classes (diameter under 0.9 mm: the lateral) with a thinner cortex (the advanced
    hook: a class's own parameters)."""
    root = OrganAnatomy(AnatomyLibrary(class_parameters=lambda key: {"cortex": {"n_layers": 2}} if key[0] < 0.9
                                       else {}))


class RootSystem(CompositeModel):
    initiators = (RootAxis, RootSystemAnatomy)
    from_scale, nodes, wiring = "SubOrgan", "Compartment", junction_rules

    def __init__(self, data_structure, time_step):
        self.anatomy = RootSystemAnatomy(data_structure=data_structure, time_step=time_step)
        self.declare_data_and_couple_components(translator_path=Translator(), components=(self.anatomy,))

    def run(self):
        self.anatomy()


def layout(g, segments):
    """Each segment's drawing offset: along its axis from its linked parent, a lateral's first one below its bearer."""
    offsets = {}
    for segment in segments:                                       # parents come first (smaller vids)
        parent = g.linked_parent(segment, g.scales.SubOrgan)
        if parent is None:
            offsets[segment] = (0., 0.)
        else:
            x, y = offsets[parent]
            branched = g.complex_at_scale(segment, g.scales.Axis) != g.complex_at_scale(parent, g.scales.Axis)
            offsets[segment] = (x, y - SPACING) if branched else (x + SPACING, y)
    return offsets


def main():
    scenario = {"root_segments": [[MAIN, SEGMENT, (3 - i) * SEGMENT] for i in range(4)],
                "root_laterals": [[1, [[LATERAL, SEGMENT, (1 - i) * SEGMENT] for i in range(2)]]]}
    scene = Scene(pd.DataFrame([dict(plant="root system", model=RootSystem, x=0., y=0., z=0., rotation=0.,
                                     scenario={"parameters": scenario})]))
    ds = scene.populations[0].data_structure
    g = ds.mtg
    segments = [int(s) for s in ds.entity_ids("SubOrgan")]
    offsets = layout(g, segments)

    fig, ax = plt.subplots(figsize=(15, 7.5))
    draw_plant(ax, ds, offsets)
    for s, (x, y) in offsets.items():
        ax.annotate(f"segment {s}\nclass {g.property('anatomy_class')[s][0]:g} mm", (x, y + 0.5), ha="center",
                    fontsize=8)
    ax.margins(0.05, 0.12)
    ax.set_xticks([])
    ax.set_yticks([])
    n_templates = len(set(g.property("anatomy_template").values()))
    ax.set_title(f"{len(segments)} segments, {n_templates} anatomy templates: {ds.n_nodes()} Compartments, "
                 f"{ds.n_edges()} Connections ({int((ds.get('network_path') == 5).sum())} axial junctions)")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    tissue_legend(fig)
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "root_system.png")
    fig.savefig(path, dpi=160)
    print(path)


if __name__ == "__main__":
    main()
