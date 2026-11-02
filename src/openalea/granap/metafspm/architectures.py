"""
Stand-ins for an architecture model, to give GRANAP's anatomies (and the models coupled to them) a plant to work on
in examples and tutorials. Each is a metafspm StructuralComponent, an initiator, reading its geometry from the plant's
scenario (the keys below; lengths in m), and declaring the segments' variables it gives (radius, length,
distance_from_tip: SegmentArchitecture), which GranapAnatomy reads as inputs:

    RootAxis      one root axis of segments, base to tip, and its laterals
                    root_segments: [[radius, length, distance_from_tip], ...]   default one 200 µm section
                    root_laterals: [[bearer segment's index, [[radius, length, distance_from_tip], ...]], ...]
                    root_growth:   [[axis, radius, length], ...]: at each step, a new segment at the tip of each
                                   axis listed (0 the main root, 1, 2, ... the laterals in root_laterals' order)
    Needle        one leaf organ (a needle) of leaf elements, base to tip
                    leaf_elements: n (1), leaf_element_length (2e-4), leaf_radius (0.6e-3)

    class RootAnatomies(CompositeModel):
        initiators = (RootAxis, GranapAnatomy)
        from_scale, nodes, wiring = "SubOrgan", "Compartment", junction_rules

        def __init__(self, data_structure, time_step):
            self.architecture = RootAxis(data_structure=data_structure, time_step=time_step)   # first: it owns the
            self.anatomy = GranapAnatomy(data_structure=data_structure, time_step=time_step)   # segments' variables
            self.couple(self.architecture, self.anatomy)

        def run(self):
            self.architecture()          # the roots grow (root_growth)
            self.anatomy()               # the new segments get their anatomies

    table = planting([Planting()], RootAnatomies, {"root_segments": [[0.55e-3, 2e-4, 0.]]})
"""
from dataclasses import dataclass

from openalea.metafspm.coupling.component import StructuralComponent, parameter, state_variable
from openalea.metafspm.data_structure.configs import PropsConfig, ScalesConfig
from openalea.metafspm.solve.decorator import segmentation

SCALES = ScalesConfig()
ROOT_SEGMENTS = [[0.55e-3, 2e-4, 0.]]                  # GRANAP's monocot root preset, one 200 µm section


def new_organ(g, plant, axis, unit, phytomer, organ):
    """An organ under a new axis, growth unit and phytomer of *plant* (labels by name)."""
    s, L = g.scales, g.labels
    complex_ = plant
    for scale, labels, name in ((s.Axis, L.Axis, axis), (s.GrowthUnit, L.GrowthUnit, unit),
                                (s.Phytomer, L.Phytomer, phytomer), (s.Organ, L.Organ, organ)):
        complex_ = g.add_component(complex_, **PropsConfig(scale=scale, edge_type='/', label=getattr(labels, name)))
    return complex_


def segment_chain(g, organ, label, segments, bearer=None):
    """SubOrgan segments [(radius, length, distance_from_tip)] under *organ*, each the successor of the previous; the
    first branched ('+') on the segment *bearer* when given (a lateral), else the organ's first. Their variables
    (SegmentArchitecture's) are written on them."""
    s = g.scales
    previous, out = None, []
    for radius, length, distance in segments:
        props = PropsConfig(scale=s.SubOrgan, edge_type='<' if previous else '+' if bearer else '/', label=label)
        previous = (g.add_child(previous, **props) if previous else
                    g.add_component_with_topo(organ, bearer, **props) if bearer else g.add_component(organ, **props))
        g.property("radius")[previous], g.property("length")[previous] = float(radius), float(length)
        g.property("distance_from_tip")[previous] = float(distance)
        out.append(previous)
    return out


@dataclass
class SegmentArchitecture(StructuralComponent):
    """
    The variables an architecture gives its segments (SubOrgans), declared by the architecture, their owner: their
    radius, length and distance from their axis' tip. GranapAnatomy reads them as inputs of the same names: the
    DataStructure is the coupling, once a plant model couples both (built first, the architecture holds them at the
    SubOrgan scale). Subclasses build the plants (initiate_plant).
    """
    radius: float = state_variable(unit="m", description="the segment's radius", default=0., scale=SCALES.SubOrgan,
                                   kind="descriptor", references="stand-in architecture")
    length: float = state_variable(unit="m", description="the segment's length", default=0., scale=SCALES.SubOrgan,
                                   kind="descriptor", references="stand-in architecture")
    distance_from_tip: float = state_variable(unit="m", description="the segment's distance from its axis' tip",
                                              default=0., scale=SCALES.SubOrgan, kind="descriptor",
                                              references="stand-in architecture")


@dataclass
class RootAxis(SegmentArchitecture):
    """One root axis of segments, base to tip, and its laterals, each an axis of its own branched on a segment of the
    main one (scenario: root_segments, root_laterals); the axes root_growth lists grow a segment at each step."""
    root_growth: object = parameter(unit="", description="at each step, a new segment at the tip of each axis listed: "
                                    "[[axis, radius (m), length (m)], ...], axis 0 the main root, 1, 2, ... the "
                                    "laterals in root_laterals' order (None: no growth)", default=None,
                                    dtype="object", scale=SCALES.Plant, references="stand-in architecture")

    scenario_keys = ("root_segments", "root_laterals")

    @classmethod
    def initiate_plant(cls, g, plant, parameters):
        p = parameters or {}
        label = g.labels.SubOrgan.RootSegment
        main = segment_chain(g, new_organ(g, plant, "Root", "Root", "Root", "RootInternode"), label,
                             p.get("root_segments", ROOT_SEGMENTS))
        for bearer, segments in p.get("root_laterals", []):
            segment_chain(g, new_organ(g, plant, "Root", "Root", "Root", "RootInternode"), label, segments,
                          bearer=main[bearer])

    @segmentation
    def _tip_growth(self):
        """MPG-style: each axis that its plant's root_growth lists gets a new segment at its tip, of the radius and
        length given, at distance 0 from the tip; the axis' other segments get that much farther from it."""
        g = self.mtg
        plants = self.data_structure.entity_ids("Plant").tolist()
        growth = dict(zip(plants, self.parameter_values("root_growth", "Plant")))
        radius, length, distance = (g.property(name) for name in ("radius", "length", "distance_from_tip"))
        label = g.labels.SubOrgan.RootSegment
        for plant in self.active_ids("Plant").tolist():
            axes = g.components_at_scale(plant, scale=g.scales.Axis)
            for axis, new_radius, new_length in growth.get(plant) or []:
                segments = [int(v) for v in g.components_at_scale(axes[int(axis)], scale=g.scales.SubOrgan)]
                new = g.add_child(segments[-1], **PropsConfig(scale=g.scales.SubOrgan, edge_type='<', label=label))
                for segment in segments:
                    distance[segment] = float(distance.get(segment, 0.)) + float(new_length)
                radius[new], length[new], distance[new] = float(new_radius), float(new_length), 0.


@dataclass
class Needle(SegmentArchitecture):
    """One needle: a leaf organ of leaf elements, base to tip (scenario: leaf_elements, leaf_element_length,
    leaf_radius)."""
    scenario_keys = ("leaf_elements", "leaf_element_length", "leaf_radius")

    @classmethod
    def initiate_plant(cls, g, plant, parameters):
        p = parameters or {}
        n, length, radius = int(p.get("leaf_elements", 1)), p.get("leaf_element_length", 2e-4), p.get("leaf_radius",
                                                                                                        0.6e-3)
        organ = new_organ(g, plant, "Shoot", "Shoot", "Shoot", "Leaf")
        segment_chain(g, organ, g.labels.SubOrgan.LeafElement,
                      [(radius, length, (n - 1 - i) * length) for i in range(n)])
