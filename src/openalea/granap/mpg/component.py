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
from dataclasses import dataclass

import numpy as np

from openalea.metafspm.coupling.component import StructuralComponent
from openalea.metafspm.solve.decorator import postsegmentation

from openalea.granap.mpg.library import AnatomyLibrary, ClassBins


@dataclass
class OrganAnatomy:
    """The anatomy of one kind of segment: its library and the property giving its class's size (None: one class)."""
    library: AnatomyLibrary
    size: str = "radius"


def _segments(g, plant, labels_handled=("RootSegment",)):
    """The segments of *plant* (SubOrgan vertices) whose label is handled (default: root segments)."""
    codes = {getattr(g.labels.SubOrgan, name) for name in labels_handled}
    labels = g.property("label")
    return [int(v) for v in g.components_at_scale(plant, scale=g.scales.SubOrgan) if labels.get(v) in codes]


def distance_from_junction(g, segment, length="length") -> float:
    """The length of the segments from *segment* up to the root-shoot junction (its own length excluded)."""
    lengths, total = g.property(length), 0.
    parent = g.parent(segment)
    while parent is not None and g.scale(parent) == g.scales.SubOrgan and \
            g.property("label").get(parent) == g.labels.SubOrgan.RootSegment:
        total += float(lengths.get(parent, 0.))
        parent = g.parent(parent)
    return total


def remove_anatomy(g, segment) -> None:
    """Remove *segment*'s anatomy (its Layers with their Cells and protoplasts, its anchored walls, junctions and
    anatomy Connections), keeping the segment."""
    s = g.scales
    anchored = [c for c in list(g._children.get(segment, [])) if g.scale(c) in (s.Compartment, s.Connection)]
    g.remove_connections([c for c in anchored if g.scale(c) == s.Connection])
    g.remove_connections([c for c in anchored if g.scale(c) == s.Compartment])
    g.remove_connections([c for c in list(g._components.get(segment, [])) if g.scale(c) == s.Layer])


def granap_inheritance(ds, new_vids, is_known, fresh=()) -> np.ndarray:
    """
    The source of each new Compartment (Q-G3): the Compartment of the same template node in the nearest upstream
    section that has it; else the known protoplast of the same tissue nearest in (angle, relative radius) in the
    nearest upstream section having that tissue; else none (-1, the default). Sections in *fresh* (replaced
    anatomies, Q-G2) inherit nothing.
    """
    g = ds.mtg
    from_scale = ds._from_scale
    new_vids = np.asarray(new_vids, dtype=np.int64)
    sources = np.full(new_vids.size, -1, dtype=np.int64)
    sections = ds._sections_of(new_vids)
    owners = g.compartments_by_owner(from_scale)
    template, node = g.property("template"), g.property("template_node")
    tissue, angle, radius = g.property("tissue"), g.property("angle"), g.property("relative_radius")
    upstream_cache, node_maps = {}, {}

    def upstream(section):
        if section not in upstream_cache:
            chain, vertex = [], g.linked_parent(int(section), from_scale)
            while vertex is not None:
                chain.append(int(vertex))
                vertex = g.linked_parent(vertex, from_scale)
            upstream_cache[section] = chain
        return upstream_cache[section]

    def node_map(section):
        if section not in node_maps:
            comps = np.asarray(owners.get(section, []), dtype=np.int64)
            known = comps[np.asarray(is_known(comps), dtype=bool)] if comps.size else comps
            node_maps[section] = {(template.get(int(v)), node.get(int(v))): int(v) for v in known}
        return node_maps[section]

    for i, (vid, section) in enumerate(zip(new_vids.tolist(), sections.tolist())):
        if section < 0 or section in fresh:
            continue
        key = (template.get(vid), node.get(vid))
        for ancestor in upstream(section):
            hit = node_map(ancestor).get(key)
            if hit is not None:
                sources[i] = hit
                break
        if sources[i] >= 0 or tissue.get(vid) is None:
            continue
        for ancestor in upstream(section):                       # same tissue, nearest relative position
            candidates = [v for v in node_map(ancestor).values() if tissue.get(v) == tissue.get(vid) and v in angle]
            if candidates and vid in angle:
                here = np.array([radius[vid] * np.cos(angle[vid]), radius[vid] * np.sin(angle[vid])])
                there = np.array([[radius[v] * np.cos(angle[v]), radius[v] * np.sin(angle[v])] for v in candidates])
                sources[i] = candidates[int(np.argmin(np.linalg.norm(there - here, axis=1)))]
                break
    return sources


class GranapAnatomy(StructuralComponent):
    """
    The cross-section anatomies of root segments. Class attributes (set on a subclass, or before initiation):
      library      the AnatomyLibrary (its parameters(key) function: default_root_parameters)
      bins         the ClassBins (mm)
      mm_per_unit  the MPG's length unit in mm (1000: lengths in m)
      radius, length, distance_from_tip   the names of the segments' properties
    """
    library = AnatomyLibrary()
    shoot = {}                       # {SubOrgan label name: OrganAnatomy}, e.g. stems and leaves (roots: library)
    bins = ClassBins()
    mm_per_unit = 1000.
    radius, length, distance_from_tip = "radius", "length", "distance_from_tip"

    @classmethod
    def organs(cls) -> dict:
        """{SubOrgan label name: OrganAnatomy}: the root segments with cls.library, and cls.shoot."""
        return {"RootSegment": OrganAnatomy(cls.library, cls.radius), **cls.shoot}

    @classmethod
    def organ_of(cls, g, segment) -> OrganAnatomy:
        code = g.property("label").get(segment)
        return next(organ for name, organ in cls.organs().items() if getattr(g.labels.SubOrgan, name) == code)

    @classmethod
    def segment_class(cls, g, segment) -> tuple:
        to_mm, organ = cls.mm_per_unit, cls.organ_of(g, segment)
        if organ.size is None:
            return cls.bins.key(0.)
        diameter = 2. * float(g.property(organ.size)[segment]) * to_mm
        if g.property("label").get(segment) != g.labels.SubOrgan.RootSegment:
            return cls.bins.key(diameter)
        tip = float(g.property(cls.distance_from_tip).get(segment, 0.)) * to_mm
        junction = distance_from_junction(g, segment, cls.length) * to_mm
        return cls.bins.key(diameter, tip, junction)

    @classmethod
    def initiate_plant(cls, g, plant_vid: int, parameters: dict) -> None:
        """Each root segment of the plant gets its class's anatomy, instantiated from the library."""
        cls.update_anatomies(g, _segments(g, plant_vid, tuple(cls.organs())))

    @classmethod
    def update_anatomies(cls, g, segments) -> dict:
        """
        Give each segment the anatomy of its current class (Q12): a segment without one (just emerged) gets its
        class's copy; a segment whose class changed has its anatomy replaced. Returns {"emerged", "replaced"}.
        """
        anatomy_class = g.properties().setdefault("anatomy_class", {})
        done = {"emerged": [], "replaced": []}
        for segment in segments:
            key = cls.segment_class(g, segment)
            current = anatomy_class.get(segment)
            if current == key:
                continue
            if current is not None:
                remove_anatomy(g, segment)
                done["replaced"].append(segment)
            else:
                done["emerged"].append(segment)
            cls.organ_of(g, segment).library.template(key).instantiate(g, segment)
            anatomy_class[segment] = key
        return done

    def __post_init__(self):
        super().__post_init__()
        self._replaced = set()
        self.data_structure.inheritance = lambda ds, new, known: granap_inheritance(ds, new, known, self._replaced)

    @postsegmentation
    def _anatomies(self):
        """MPG-style: every root segment of the MPG gets its class's anatomy (emerged, or replaced on a change)."""
        g = self.mtg
        segments = [s for plant in g.components_at_scale(g.root, scale=g.scales.Plant)
                    for s in _segments(g, plant, tuple(self.organs()))]
        alive = set(segments)
        anatomy_class = g.properties().get("anatomy_class", {})
        for gone in [s for s in list(anatomy_class) if s not in alive]:      # pruned segments: their record only
            del anatomy_class[gone]
        done = self.update_anatomies(g, segments)
        self._replaced = set(done["replaced"])
        self.last_update = done


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
