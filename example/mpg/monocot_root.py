"""
The simplest use of GRANAP's anatomy component in metafspm: a monocot root of three segments gets its cross-section
anatomies, grows one segment, and the component gives the new segment its anatomy. Each segment's section is then
drawn from the plant graph, coloured by tissue.

    python monocot_root.py            (from GRANAP/example/mpg; figure in ./outputs/monocot_root.png)
"""
import os

import pandas as pd

from openalea.metafspm.coupling.composite_wrapper import CompositeModel
from openalea.metafspm.coupling.translator import Translator
from openalea.metafspm.scene.scene import Scene

from openalea.granap.metafspm.architectures import RootAxis
from openalea.granap.metafspm.component import GranapAnatomy, junction_rules
from openalea.granap.metafspm.plots import draw_sections

SEGMENT, RADIUS = 5e-3, 0.55e-3                   # m


class RootAnatomies(CompositeModel):
    """A root's architecture (a stand-in) and GRANAP's anatomies of its segments."""
    initiators = (RootAxis, GranapAnatomy)
    from_scale, nodes, wiring = "SubOrgan", "Compartment", junction_rules

    def __init__(self, data_structure, time_step):
        self.anatomy = GranapAnatomy(data_structure=data_structure, time_step=time_step)
        self.declare_data_and_couple_components(translator_path=Translator(), components=(self.anatomy,))

    def run(self):
        self.anatomy()


def main():
    # one plant, its scenario: three segments, base to tip (radius, length, distance from the tip), the root preset
    scenario = {"root_segments": [[RADIUS, SEGMENT, (2 - i) * SEGMENT] for i in range(3)]}
    scene = Scene(pd.DataFrame([dict(plant="root", model=RootAnatomies, x=0., y=0., z=0., rotation=0.,
                                     scenario={"parameters": scenario})]))
    ds = scene.populations[0].data_structure
    g = ds.mtg

    # the root grows one segment at its tip (an architecture model's role), the component gives it its anatomy
    tip = int(ds.entity_ids("SubOrgan")[-1])
    new = g.add_child(tip, edge_type="<", label=g.labels.SubOrgan.RootSegment)
    g.property("radius")[new], g.property("length")[new], g.property("distance_from_tip")[new] = RADIUS, SEGMENT, 0.
    scene.run()
    print("new segment:", scene.populations[0].instance.anatomy.last_update["emerged"],
          f"- {ds.n_nodes()} Compartments, {ds.n_edges()} Connections")

    segments = [int(s) for s in ds.entity_ids("SubOrgan")]
    fig = draw_sections(ds, titles=[f"segment {s}" + (" (new)" if s == new else "") for s in segments])
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(folder, exist_ok=True)
    fig.savefig(os.path.join(folder, "monocot_root.png"), dpi=150)
    print(os.path.join(folder, "monocot_root.png"))


if __name__ == "__main__":
    main()
