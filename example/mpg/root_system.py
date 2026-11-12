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
import numpy as np

from openalea.metafspm import CompositeModel, Planting, planting, Scene

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

    parts = {"architecture": RootAxis, "anatomy": RootSystemAnatomy}


def layout(mtg, segments):
    """Each segment's drawing offset: along its axis from its linked parent, a lateral's first one below its bearer."""
    offsets = {}
    for segment in segments:                                       # parents come first (smaller vids)
        parent = mtg.linked_parent(segment, mtg.scales.SubOrgan)
        if parent is None:
            offsets[segment] = (0., 0.)
        else:
            x, y = offsets[parent]
            branched = mtg.complex_at_scale(segment, mtg.scales.Axis) != mtg.complex_at_scale(parent, mtg.scales.Axis)
            offsets[segment] = (x, y - SPACING) if branched else (x + SPACING, y)
    return offsets


def main():
    scenario = {"root_segments": [[MAIN, SEGMENT, (3 - i) * SEGMENT] for i in range(4)],
                "root_laterals": [[1, [[LATERAL, SEGMENT, (1 - i) * SEGMENT] for i in range(2)]]]}
    scene = Scene(planting([Planting()], RootSystem, scenario))
    data_structure = scene.populations[0].data_structure
    mtg = data_structure.mtg
    segments = [int(s) for s in data_structure.entity_ids("SubOrgan")]
    offsets = layout(mtg, segments)

    fig, ax = plt.subplots(figsize=(15, 7.5))
    draw_plant(ax, data_structure, offsets)
    diameter = dict(zip(segments, 2e3 * np.asarray(data_structure.get("radius"))))          # mm: the class's binned key
    for s, (x, y) in offsets.items():
        ax.annotate(f"segment {s}\ndiameter {diameter[s]:g} mm", (x, y + 0.5), ha="center", fontsize=8)
    ax.margins(0.05, 0.12)
    ax.set_xticks([])
    ax.set_yticks([])
    n_templates = len(set(data_structure.get("anatomy_template").tolist()))
    ax.set_title(f"{len(segments)} segments, {n_templates} anatomy templates: {data_structure.n_nodes()} Compartments, "
                 f"{data_structure.n_edges()} Connections ({int((data_structure.get('network_path') == 5).sum())} axial junctions)")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    tissue_legend(fig)
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "root_system.png")
    fig.savefig(path, dpi=160)
    print(path)


if __name__ == "__main__":
    main()
