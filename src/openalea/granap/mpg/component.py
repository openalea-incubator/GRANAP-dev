"""
GRANAP as a metafspm structural component (devplan_metafspm.md, P3).

    class Seedling(CompositeModel):
        initiators = (RootArchitecture, GranapAnatomy)          # the anatomies after the architecture
        from_scale, nodes = "SubOrgan", "Compartment"
        wiring = junction_rules                                 # vessels joined between successive sections

GranapAnatomy.initiate_plant gives each root segment of a plant the anatomy of its class (binned diameter, distance
from the tip, distance from the root-shoot junction), from the library: one template generated per class, a copy per
segment (its own vertices and state, the template's geometry). junction_rules pairs the conducting cells of
successive sections (xylem with xylem, phloem with phloem): by track_id where both have one, else the nearest in
(angle, relative radius) in both directions, so that a fusion or a split gives one-to-many junctions (Q-P3b).
"""
import numpy as np

from openalea.metafspm.coupling.component import StructuralComponent

from openalea.granap.mpg.library import AnatomyLibrary, ClassBins


def _segments(g, plant):
    """The root segments of *plant* (SubOrgan vertices labelled RootSegment)."""
    label = g.labels.SubOrgan.RootSegment
    labels = g.property("label")
    return [int(v) for v in g.components_at_scale(plant, scale=g.scales.SubOrgan) if labels.get(v) == label]


def distance_from_junction(g, segment, length="length") -> float:
    """The length of the segments from *segment* up to the root-shoot junction (its own length excluded)."""
    lengths, total = g.property(length), 0.
    parent = g.parent(segment)
    while parent is not None and g.scale(parent) == g.scales.SubOrgan and \
            g.property("label").get(parent) == g.labels.SubOrgan.RootSegment:
        total += float(lengths.get(parent, 0.))
        parent = g.parent(parent)
    return total


class GranapAnatomy(StructuralComponent):
    """
    The cross-section anatomies of root segments. Class attributes (set on a subclass, or before initiation):
      library      the AnatomyLibrary (its parameters(key) function: default_root_parameters)
      bins         the ClassBins (mm)
      mm_per_unit  the MPG's length unit in mm (1000: lengths in m)
      radius, length, distance_from_tip   the names of the segments' properties
    """
    library = AnatomyLibrary()
    bins = ClassBins()
    mm_per_unit = 1000.
    radius, length, distance_from_tip = "radius", "length", "distance_from_tip"

    @classmethod
    def segment_class(cls, g, segment) -> tuple:
        to_mm = cls.mm_per_unit
        diameter = 2. * float(g.property(cls.radius)[segment]) * to_mm
        tip = float(g.property(cls.distance_from_tip).get(segment, 0.)) * to_mm
        junction = distance_from_junction(g, segment, cls.length) * to_mm
        return cls.bins.key(diameter, tip, junction)

    @classmethod
    def initiate_plant(cls, g, plant_vid: int, parameters: dict) -> None:
        """Each root segment of the plant gets its class's anatomy, instantiated from the library."""
        anatomy_class = g.properties().setdefault("anatomy_class", {})
        for segment in _segments(g, plant_vid):
            key = cls.segment_class(g, segment)
            cls.library.template(key).instantiate(g, segment)
            anatomy_class[segment] = key


def _vessels(g, compartments, family):
    """The conducting protoplasts of *family* among *compartments*: (vids, track ids, normalised positions)."""
    fam, track = g.property("family"), g.property("track_id")
    angle, radius = g.property("angle"), g.property("relative_radius")
    vids = [v for v in compartments if fam.get(v, -1) == family]
    xy = np.array([[radius[v] * np.cos(angle[v]), radius[v] * np.sin(angle[v])] for v in vids]).reshape(-1, 2)
    return vids, [track.get(v) for v in vids], xy


def vessel_pairs(g, parent_compartments, child_compartments, family) -> list:
    """The junctions between two successive sections for one family of conducting cells."""
    parents, parent_tracks, p_xy = _vessels(g, parent_compartments, family)
    children, child_tracks, c_xy = _vessels(g, child_compartments, family)
    if not parents or not children:
        return []
    pairs = set()
    by_track = {t: i for i, t in enumerate(parent_tracks) if t is not None}
    tracked_children = set()
    for j, t in enumerate(child_tracks):
        if t is not None and t in by_track:
            pairs.add((parents[by_track[t]], children[j]))
            tracked_children.add(j)
    tracked_parents = {by_track[t] for t in child_tracks if t is not None and t in by_track}
    distance = np.linalg.norm(p_xy[:, None, :] - c_xy[None, :, :], axis=2)
    for j in range(len(children)):
        if j not in tracked_children:
            pairs.add((parents[int(np.argmin(distance[:, j]))], children[j]))
    for i in range(len(parents)):
        if i not in tracked_parents:
            pairs.add((parents[i], children[int(np.argmin(distance[i]))]))
    return sorted(pairs)


def junction_rules(g) -> list:
    """The wiring rules of anatomy mode: axial xylem junctions (apoplastic) and phloem junctions (symplastic)."""
    return [
        {"rule": lambda g, p, c, pc, cc: vessel_pairs(g, pc, cc, 0), "edge_label": g.labels.Connection.Apoplastic},
        {"rule": lambda g, p, c, pc, cc: vessel_pairs(g, pc, cc, 1), "edge_label": g.labels.Connection.Symplastic},
    ]
