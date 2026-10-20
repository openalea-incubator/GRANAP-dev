"""
Stand-ins for an architecture model, to give GRANAP's anatomies (and the models coupled to them) a plant to work on
in examples and tutorials. Each is a metafspm StructuralComponent, an initiator, reading its geometry from the plant's
scenario (the keys below; lengths in m):

    RootAxis      one root axis of segments, base to tip
                    root_segments: [[radius, length, distance_from_tip], ...]   default one 200 µm section
    Needle        one leaf organ (a needle) of leaf elements, base to tip
                    leaf_elements: n (1), leaf_element_length (2e-4), leaf_radius (0.6e-3)

    table = pd.DataFrame([dict(plant="p0", model=MyModel, x=0., y=0., z=0., rotation=0.,
                               scenario={"parameters": {"root_segments": [[0.55e-3, 2e-4, 0.]]}})])
"""
from openalea.metafspm.coupling.component import StructuralComponent
from openalea.metafspm.data_structure.configs import PropsConfig

ROOT_SEGMENTS = [[0.55e-3, 2e-4, 0.]]                  # GRANAP's monocot root preset, one 200 µm section


def _organ(g, plant, axis, unit, phytomer, organ):
    """An organ under a new axis, growth unit and phytomer of *plant* (labels by name)."""
    s, L = g.scales, g.labels
    complex_ = plant
    for scale, labels, name in ((s.Axis, L.Axis, axis), (s.GrowthUnit, L.GrowthUnit, unit),
                                (s.Phytomer, L.Phytomer, phytomer), (s.Organ, L.Organ, organ)):
        complex_ = g.add_component(complex_, **PropsConfig(scale=scale, edge_type='/', label=getattr(labels, name)))
    return complex_


def _chain(g, organ, label, segments):
    """SubOrgan segments [(radius, length, distance_from_tip)] under *organ*, each the successor of the previous."""
    s = g.scales
    previous, out = None, []
    for radius, length, distance in segments:
        props = PropsConfig(scale=s.SubOrgan, edge_type='<' if previous else '/', label=label)
        previous = g.add_child(previous, **props) if previous else g.add_component(organ, **props)
        g.property("radius")[previous], g.property("length")[previous] = float(radius), float(length)
        g.property("distance_from_tip")[previous] = float(distance)
        out.append(previous)
    return out


class RootAxis(StructuralComponent):
    """One root axis of segments, base to tip (scenario: root_segments)."""
    scenario_keys = ("root_segments",)

    @classmethod
    def initiate_plant(cls, g, plant, parameters):
        segments = (parameters or {}).get("root_segments", ROOT_SEGMENTS)
        organ = _organ(g, plant, "Root", "Root", "Root", "RootInternode")
        _chain(g, organ, g.labels.SubOrgan.RootSegment, segments)


class Needle(StructuralComponent):
    """One needle: a leaf organ of leaf elements, base to tip (scenario: leaf_elements, leaf_element_length,
    leaf_radius)."""
    scenario_keys = ("leaf_elements", "leaf_element_length", "leaf_radius")

    @classmethod
    def initiate_plant(cls, g, plant, parameters):
        p = parameters or {}
        n, length, radius = int(p.get("leaf_elements", 1)), p.get("leaf_element_length", 2e-4), p.get("leaf_radius",
                                                                                                        0.6e-3)
        organ = _organ(g, plant, "Shoot", "Shoot", "Shoot", "Leaf")
        _chain(g, organ, g.labels.SubOrgan.LeafElement, [(radius, length, (n - 1 - i) * length) for i in range(n)])
