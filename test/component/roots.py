"""
A small root system for the capability tests: a main root of four segments (radius 0.55 mm) and one lateral of two
(radius 0.3 mm) on its second segment, lengths in m as in metafspm's plant models; distances from the tips set.
"""
import numpy as np

from openalea.metafspm.coupling.component import StructuralComponent
from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.metafspm.scene import population
from openalea.metafspm.scene.population import Planting

SEGMENT = 5e-3                      # m
MAIN_RADIUS, LATERAL_RADIUS = 0.55e-3, 0.3e-3


def _axis(mtg, plant, n, radius, bearer=None):
    s, L = mtg.scales, mtg.labels
    axis = mtg.add_component(plant, **PropsConfig(scale=s.Axis, edge_type='/', label=L.Axis.Root))
    gu = mtg.add_component(axis, **PropsConfig(scale=s.GrowthUnit, edge_type='/', label=L.GrowthUnit.Root))
    phytomer = mtg.add_component(gu, **PropsConfig(scale=s.Phytomer, edge_type='/', label=L.Phytomer.Root))
    organ = mtg.add_component(phytomer, **PropsConfig(scale=s.Organ, edge_type='/', label=L.Organ.RootInternode))
    props = PropsConfig(scale=s.SubOrgan, label=L.SubOrgan.RootSegment, edge_type='+' if bearer else '/')
    first = (mtg.add_component_with_topo(organ, bearer, **props) if bearer is not None
             else mtg.add_component(organ, **props))
    segments = [first]
    for _ in range(n - 1):
        segments.append(mtg.add_child(segments[-1], **PropsConfig(scale=s.SubOrgan, edge_type='<',
                                                                label=L.SubOrgan.RootSegment)))
    for rank, v in enumerate(segments):
        mtg.set_values("radius", [v], [radius])
        mtg.set_values("length", [v], [SEGMENT])
        mtg.set_values("distance_from_tip", [v], [(n - 1 - rank) * SEGMENT])
    return segments


class SmallRootSystem(StructuralComponent):
    """The architecture initiator: a main root and a lateral."""

    @classmethod
    def initiate_plant(cls, mtg, plant, parameters):
        main = _axis(mtg, plant, 4, MAIN_RADIUS)
        _axis(mtg, plant, 2, LATERAL_RADIUS, bearer=main[1])


def planting():
    return population.planting([Planting()], None)


def initiated(table, *initiators, time_step: float = 3600., reduction=None, translator=None):
    """The plants of *table* built by *initiators* (a root system or a shoot, then GRANAP's anatomies), their
    Compartments wired at the segments' junctions, in one MPGDataStructure: metafspm's population_data_structure with
    a model of no component (the tests build theirs on population.data_structure)."""
    from openalea.granap.metafspm.component import junction_rules
    from openalea.metafspm.coupling.composite_wrapper import CompositeModel

    model = type("Initiated", (CompositeModel,), dict(initiators=tuple(initiators), nodes="Compartment",
                                                      wiring=staticmethod(junction_rules), reduction=reduction,
                                                      translator=translator))
    return population.population_data_structure(table, model, time_step=time_step)
