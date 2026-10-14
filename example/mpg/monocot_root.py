"""
The simplest use of GRANAP's anatomy component in metafspm: a monocot root of three segments gets its cross-section
anatomies, grows one segment, and the component gives the new segment its anatomy. Each segment's section is then drawn
from the MPG, coloured by tissue.

    python monocot_root.py            (from GRANAP/example/mpg; figure in ./outputs/monocot_root.png)
"""
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as PolygonPatch

from openalea.metafspm.coupling.choregrapher import Choregrapher
from openalea.metafspm.coupling.component import StructuralComponent
from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.metafspm.data_structure.data_api import MPGDataStructure
from openalea.metafspm.scene.population import build_population

from openalea.granap.metafspm.component import GranapAnatomy, junction_rules

SEGMENT, RADIUS = 5e-3, 0.55e-3                   # m


# 1. The architecture: one root axis of three segments (what a root architecture model would build)
class Root(StructuralComponent):
    @classmethod
    def initiate_plant(cls, g, plant, parameters):
        s, L = g.scales, g.labels
        complex_ = plant
        for scale, label in ((s.Axis, L.Axis.Root), (s.GrowthUnit, L.GrowthUnit.Root), (s.Phytomer, L.Phytomer.Root),
                             (s.Organ, L.Organ.RootInternode)):
            complex_ = g.add_component(complex_, **PropsConfig(scale=scale, edge_type='/', label=label))
        segment = g.add_component(complex_, **PropsConfig(scale=s.SubOrgan, edge_type='/',
                                                          label=L.SubOrgan.RootSegment))
        for rank in range(3):
            if rank:
                segment = g.add_child(segment, **PropsConfig(scale=s.SubOrgan, edge_type='<',
                                                             label=L.SubOrgan.RootSegment))
            set_segment(g, segment, distance_from_tip=(2 - rank) * SEGMENT)


def set_segment(g, segment, distance_from_tip):
    g.property("radius")[segment] = RADIUS
    g.property("length")[segment] = SEGMENT
    g.property("distance_from_tip")[segment] = distance_from_tip


# 2. The anatomy component: GRANAP's default library is the monocot root preset
class Anatomy(GranapAnatomy):
    pass


def main():
    # 3. Instantiate: the plant, its anatomies, the DataStructure over their Compartments, the component
    Choregrapher().add_simulation_time_step(3600.)
    plants = pd.DataFrame([dict(plant="p0", model=None, x=0., y=0., z=0., rotation=0., scenario={"parameters": {}})])
    g, _ = build_population(plants, initiators=(Root, Anatomy))
    g.convert_properties_to_arraydict()
    ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan, nodes="Compartment", wiring=junction_rules(g))
    anatomy = Anatomy(data_structure=ds)

    # 4. Run: the root grows one segment at its tip; the component gives it its anatomy
    segments = sorted(int(v) for v in g.property("anatomy_class"))
    tip = g.add_child(segments[-1], **PropsConfig(scale=g.scales.SubOrgan, edge_type='<',
                                                  label=g.labels.SubOrgan.RootSegment))
    set_segment(g, tip, distance_from_tip=0.)
    anatomy()
    print("new segment:", anatomy.last_update["emerged"], f"- {ds.n_nodes()} Compartments, {ds.n_edges()} Connections")

    # 5. Plot each segment's section from the MPG: its Cells, drawn with their template's polygons
    segments = sorted(int(v) for v in g.property("anatomy_class"))
    owners = g.compartments_by_owner(g.scales.SubOrgan)
    tissue, template, row = g.property("tissue"), g.property("template"), g.property("template_row")
    templates = {t.index: t for t in Anatomy.library.templates.values()}
    types = sorted({tissue[v] for s in segments for v in owners[s] if tissue.get(v) is not None})
    colours = {t: matplotlib.colormaps["tab20"](i) for i, t in enumerate(types)}
    fig, axes = plt.subplots(1, len(segments), figsize=(4 * len(segments), 4.6))
    for ax, s in zip(axes, segments):
        patches, faces = [], []
        for v in owners[s]:
            if v in row and tissue.get(v) is not None:                 # protoplasts (walls have no polygon)
                polygon = templates[int(template[v])].polygons[int(row[v])]
                if polygon is not None and polygon.geom_type == "Polygon":
                    patches.append(PolygonPatch(np.asarray(polygon.exterior.coords)))
                    faces.append(colours[tissue[v]])
        ax.add_collection(PatchCollection(patches, facecolor=faces, edgecolor="0.3", linewidth=0.15))
        ax.autoscale_view()
        ax.set_aspect("equal")
        ax.set_title(f"segment {s}" + (" (new)" if s == tip else ""))
        ax.set_xlabel("mm")
    fig.legend(handles=[matplotlib.patches.Patch(color=colours[t], label=t) for t in types], loc="center right",
               fontsize=8)
    fig.tight_layout(rect=(0, 0, 0.88, 1))
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(folder, exist_ok=True)
    fig.savefig(os.path.join(folder, "monocot_root.png"), dpi=150)
    print(os.path.join(folder, "monocot_root.png"))


if __name__ == "__main__":
    main()
