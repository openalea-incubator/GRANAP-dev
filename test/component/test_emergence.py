"""A plant emerging later in an anatomy-mode scene: the sections' 3D quantities are computed for every plant, emerged
or not (the step's arrays are not restricted to the active entities, its edge means cover every edge)."""
import numpy as np

from openalea.metafspm.coupling.composite_wrapper import CompositeModel
from openalea.metafspm.scene.population import Planting, planting
from openalea.metafspm.scene.scene import Scene

from openalea.granap.metafspm.component import GranapAnatomy, junction_rules

from roots import SmallRootSystem


class Roots(CompositeModel):
    initiators = (SmallRootSystem, GranapAnatomy)
    nodes = "Compartment"
    wiring = staticmethod(junction_rules)

    def __init__(self, data_structure, time_step):
        self.anatomy = GranapAnatomy(data_structure=data_structure, time_step=time_step)
        self.couple(self.anatomy)

    def run(self):
        self.anatomy()


def test_a_plant_not_yet_emerged_has_its_sections_geometry():
    table = planting([Planting(), Planting(x=0.5)], Roots, emergence_times=[0., 7200.], xrange=1., yrange=1.)
    scene = Scene(table, time_step=3600.)
    for _ in range(3):
        scene.run()
        ds = scene.populations[0].data_structure
        assert np.isfinite(ds.get("wall_surface")).all() and (np.asarray(ds.get("cell_volume")) >= 0).all()
