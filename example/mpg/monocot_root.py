"""
The simplest use of GRANAP's anatomy component in metafspm: a monocot root of three segments gets its cross-section
anatomies, grows one segment at its tip (the stand-in architecture's growth), and the component gives the new segment
its anatomy. Each segment's section is then
drawn from the plant graph, coloured by tissue.

    python monocot_root.py            (from GRANAP/example/mpg; figure in ./outputs/monocot_root.png)
"""
import os

from openalea.metafspm.coupling.composite_wrapper import CompositeModel
from openalea.metafspm.scene.population import Planting, planting
from openalea.metafspm.scene.scene import Scene

from openalea.granap.metafspm.architectures import RootAxis
from openalea.granap.metafspm.component import GranapAnatomy, junction_rules
from openalea.granap.metafspm.plots import draw_sections

SEGMENT, RADIUS = 5e-3, 0.55e-3                   # m


class RootAnatomies(CompositeModel):
    """A root's architecture (a stand-in) and GRANAP's anatomies of its segments."""
    initiators = (RootAxis, GranapAnatomy)
    from_scale, nodes, wiring = "SubOrgan", "Compartment", junction_rules

    # the root grows (root_growth), then the new segment gets its anatomy; the segments' radius, length...: shared
    # by name
    parts = {"architecture": RootAxis, "anatomy": GranapAnatomy}


def grown():
    """The scene after one step: the root built, then grown by one segment at its tip."""
    # one plant, its scenario: three segments, base to tip (radius, length, distance from the tip), the root preset;
    # at each step, a new segment at the main root's tip (axis 0)
    scenario = {"root_segments": [[RADIUS, SEGMENT, (2 - i) * SEGMENT] for i in range(3)],
                "root_growth": [[0, RADIUS, SEGMENT]]}
    scene = Scene(planting([Planting()], RootAnatomies, scenario))
    scene.run()
    return scene


def main():
    scene = grown()
    ds = scene.populations[0].data_structure
    new = scene.populations[0].instance.anatomy.last_update["emerged"]
    print("new segment:", new, f"- {ds.n_nodes()} Compartments, {ds.n_edges()} Connections")

    segments = [int(s) for s in ds.entity_ids("SubOrgan")]
    fig = draw_sections(ds, titles=[f"segment {s}" + (" (new)" if s in new else "") for s in segments])
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(folder, exist_ok=True)
    fig.savefig(os.path.join(folder, "monocot_root.png"), dpi=150)
    print(os.path.join(folder, "monocot_root.png"))


if __name__ == "__main__":
    main()
