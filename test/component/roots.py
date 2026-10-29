"""
A small root system for the capability tests: a main root of four segments (radius 0.55 mm) and one lateral of two
(radius 0.3 mm) on its second segment, lengths in m as in metafspm's plant models; distances from the tips set.
"""
import numpy as np
import pandas as pd

from openalea.metafspm.coupling.component import StructuralComponent
from openalea.metafspm.data_structure.configs import PropsConfig

SEGMENT = 5e-3                      # m
MAIN_RADIUS, LATERAL_RADIUS = 0.55e-3, 0.3e-3


def _axis(g, plant, n, radius, bearer=None):
    s, L = g.scales, g.labels
    axis = g.add_component(plant, **PropsConfig(scale=s.Axis, edge_type='/', label=L.Axis.Root))
    gu = g.add_component(axis, **PropsConfig(scale=s.GrowthUnit, edge_type='/', label=L.GrowthUnit.Root))
    phytomer = g.add_component(gu, **PropsConfig(scale=s.Phytomer, edge_type='/', label=L.Phytomer.Root))
    organ = g.add_component(phytomer, **PropsConfig(scale=s.Organ, edge_type='/', label=L.Organ.RootInternode))
    props = PropsConfig(scale=s.SubOrgan, label=L.SubOrgan.RootSegment, edge_type='+' if bearer else '/')
    first = (g.add_component_with_topo(organ, bearer, **props) if bearer is not None
             else g.add_component(organ, **props))
    segments = [first]
    for _ in range(n - 1):
        segments.append(g.add_child(segments[-1], **PropsConfig(scale=s.SubOrgan, edge_type='<',
                                                                label=L.SubOrgan.RootSegment)))
    for rank, v in enumerate(segments):
        g.property("radius")[v] = radius
        g.property("length")[v] = SEGMENT
        g.property("distance_from_tip")[v] = (n - 1 - rank) * SEGMENT
    return segments


class SmallRootSystem(StructuralComponent):
    """The architecture initiator: a main root and a lateral."""

    @classmethod
    def initiate_plant(cls, g, plant, parameters):
        main = _axis(g, plant, 4, MAIN_RADIUS)
        _axis(g, plant, 2, LATERAL_RADIUS, bearer=main[1])


def planting():
    table = pd.DataFrame([dict(plant="p0", model=None, x=0., y=0., z=0., rotation=0., scenario={"parameters": {}})])
    return table
