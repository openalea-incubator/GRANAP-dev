"""
GRANAP as a metafspm structural component.

    class Seedling(CompositeModel):
        initiators = (RootArchitecture, GranapAnatomy)          # the anatomies after the architecture
        from_scale, nodes = "SubOrgan", "Compartment"
        wiring = junction_rules                                 # vessels joined between successive sections

GranapAnatomy.initiate_plant gives each root segment of a plant the anatomy of its class (binned diameter, distance
from the tip, distance from the root-shoot junction), from the library: one template generated per class, a copy per
segment (its own vertices and state, the template's geometry). junction_rules pairs the conducting cells of
successive sections (xylem with xylem, phloem with phloem): by track_id where both have one, else the nearest in
(angle, relative radius) in both directions, so that a fusion or a split gives one-to-many junctions.

A SubOrgan without anatomy (an Adel HiddenElement, the part of an organ hidden in the sheaths' tube; a kind the
scenario does not give, e.g. a sheath without sheath_anatomy) is a pass-through: a section whose linked parent has no
anatomy is joined to its nearest ancestor with one (anatomy_ancestor), the junctions' length spanning the elements
between, so that the water goes from a stem element through the hidden elements to the next element.
"""
from dataclasses import dataclass, fields
from functools import partial

import numpy as np

from openalea.metafspm.coupling.component import (StructuralComponent, input_variable, parameter, planting_variable,
                                                  state_variable)
from openalea.metafspm.data_structure.configs import ScalesConfig
from openalea.metafspm.solve.decorator import postsegmentation, state

from openalea.granap.metafspm import columns
from openalea.granap.metafspm.build import JUNCTION, LEAF_SURFACE, SOIL_NODE, WALL, _props as _build_props
from openalea.granap.metafspm.library import (AnatomyLibrary, ClassInput, class_key, class_targets, merged,
                                              preset_parameters, template_by_index)

SCALES = ScalesConfig()
MM_PER_M = 1000.                 # GRANAP's sections are in mm; the MPG's lengths in m


# The scenario keys of the anatomies, by SubOrgan label: each a dict {"preset": ..., "seed": ..., section: {field: value}}
# on top of its preset (GRANAP's OrganInputData), read from each plant's scenario when the plant is built
ANATOMY_KEYS = {"RootSegment": "root_anatomy", "StemElement": "stem_anatomy", "LeafElement": "leaf_anatomy",
                "Sheath": "sheath_anatomy", "Seed": "seed_anatomy"}
DEFAULT_PRESETS = {"root_anatomy": "for_root", "stem_anatomy": "for_monocot_stem", "leaf_anatomy": "for_monocot_leaf",
                   "sheath_anatomy": "for_monocot_leaf", "seed_anatomy": "for_monocot_stem"}
# The class's size input. A seed is, for now, a conducting node of the stem's anatomy (a grass's seed: the bearer of
# the seminal roots and of the subcrown internode): its size input is the stems', stem_radius, its section's radius
SIZES = {"root_anatomy": "radius", "stem_anatomy": "stem_radius", "leaf_anatomy": None,
         "sheath_anatomy": None, "seed_anatomy": "stem_radius"}
# Kinds named by their organ: {Organ label name: the SubOrgan label of its elements} (a sheath's elements are stem
# elements, Adel's, with an anatomy of their own)
ORGAN_KINDS = {"Sheath": "StemElement"}
# A kind's default overrides, under the plant's: a sheath is, for now, a monocot leaf bent round the stem (its
# mid-line sagging 2 mm below its 4 mm chord: a U)
DEFAULT_OVERRIDES = {"sheath_anatomy": {"planttype": {"fold_sag": 2.0}}}
RESERVED = ("preset", "seed")
_VESSEL_PAIRS = {}
_SECTION_FIELDS = {}                               # {scenario key: its default preset's numeric fields}
_UNITS = {"mm", "µm", "um", "mm2", "mm3", "°", "deg", "degrees", "rad", "%", "MPa", "s", "mm-1", "1/mm"}


def section_fields(key: str) -> dict:
    """The numeric fields of an anatomy kind's default preset (*key*: root_anatomy, stem_anatomy...), and of GRANAP's
    optional sections (the cells' wall thickness by tissue, cell_wall_thickness), for its sectioned parameter:
    {"section.field": {default, unit, description, min_value, max_value}}, the defaults the preset's, the bounds its
    fields' (ge, le), the unit the first parenthesised unit of their description."""
    if key not in _SECTION_FIELDS:
        import re
        from openalea.granap.input_data import OPTIONAL_SECTIONS, OrganInputData
        fields_of = {}
        preset = getattr(OrganInputData, DEFAULT_PRESETS[key])()
        optional = [section() for name, section in OPTIONAL_SECTIONS.items() if preset.get(name) is None]
        for section in list(preset.params) + optional:
            for name, info in type(section).model_fields.items():
                value = getattr(section, name)
                if name == "name" or isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                bounds = {type(m).__name__: m for m in info.metadata}
                low = getattr(bounds.get("Ge"), "ge", getattr(bounds.get("Gt"), "gt", None))
                high = getattr(bounds.get("Le"), "le", getattr(bounds.get("Lt"), "lt", None))
                description = info.description or info.title or ""
                unit = next((u for u in re.findall(r"\(([^()]{1,8})\)", description) if u.strip() in _UNITS), "")
                fields_of[f"{section.name}.{name}"] = dict(default=value, unit=unit.strip(),
                                                           description=description, min_value=low, max_value=high)
        _SECTION_FIELDS[key] = fields_of
    return _SECTION_FIELDS[key]                                 # {(parent template, child template, family): template-node pairs}
_LIBRARIES = {}                                   # {(preset, seed): AnatomyLibrary}, shared by the components


def organ_class(preset: str):
    """GRANAP's organ class of an OrganInputData preset: the needle's, a stem's, a leaf's, else a root's."""
    from openalea.granap.leaf_class import LeafAnatomy
    from openalea.granap.needle_class import NeedleAnatomy
    from openalea.granap.root_class import RootAnatomy
    from openalea.granap.stem_class import StemAnatomy
    if "needle" in preset:
        return NeedleAnatomy
    return StemAnatomy if "stem" in preset else LeafAnatomy if "leaf" in preset else RootAnatomy


def preset_library(preset: str, seed: int = 0) -> AnatomyLibrary:
    """The library of an OrganInputData preset (its organ class, generated with *seed*), one per (preset, seed)."""
    key = (preset, int(seed))
    if key not in _LIBRARIES:
        _LIBRARIES[key] = AnatomyLibrary(parameters=preset_parameters(preset), organ_class=organ_class(preset),
                                         seed=int(seed))
    return _LIBRARIES[key]


@dataclass
class OrganAnatomy:
    """
    The anatomy of one kind of segment: its library and its class inputs (default: the diameter from *size*, binned by
    0.05 mm; the distance from the tip and from the root-shoot junction, not binned). *size*: the SubOrgan property
    giving the section's size, its radius in m (None: one class).
    """
    library: AnatomyLibrary
    size: str = "radius"
    class_inputs: tuple = None

    def __post_init__(self):
        if self.class_inputs is None:
            self.class_inputs = default_class_inputs(self.size)


def default_class_inputs(size="radius") -> tuple:
    """The default class inputs: diameter (mm, from the radius in m, bins of 0.05 mm), distance from the tip and from
    the root-shoot junction (mm, not binned: give them a width to bin them)."""
    return (ClassInput(size or "radius", 0.05 if size else 0., 2. * MM_PER_M),
            ClassInput("distance_from_tip", 0., MM_PER_M),
            ClassInput("distance_from_junction", 0., MM_PER_M, compute=lambda g, segment: distance_from_junction(g, segment)))


def _segments(g, plant, labels_handled=("RootSegment",), kind=None):
    """The segments of *plant* (SubOrgan vertices) whose kind is handled (default: root segments); *kind*: a function
    (g, segment) -> its kind's name (GranapAnatomy._label_name), by default its SubOrgan label's."""
    codes = {getattr(g.labels.SubOrgan, name, None) for name in labels_handled}
    codes |= {getattr(g.labels.SubOrgan, ORGAN_KINDS[name]) for name in labels_handled if name in ORGAN_KINDS}
    labels = g.property("label")
    candidates = [int(v) for v in g.components_at_scale(plant, scale=g.scales.SubOrgan) if labels.get(v) in codes]
    if kind is None:
        return candidates
    return [v for v in candidates if kind(g, v) in labels_handled]


def _basal_with_anatomy(g, elements, anatomy_class):
    """The basal element of an organ's *elements* among those with an anatomy (*anatomy_class*'s): none of its
    ancestors within the organ has one (an element without one, e.g. not grown yet, holds no Compartment)."""
    inside = set(elements)
    with_anatomy = [e for e in elements if e in anatomy_class] or elements
    owned = set(with_anatomy)
    for element in with_anatomy:
        parent = g.parent(element)
        while parent in inside and parent not in owned:
            parent = g.parent(parent)
        if parent not in inside:
            return element
    return with_anatomy[0]


def distance_from_junction(g, segment, length="length") -> float:
    """The length of the segments from *segment* up to the root-shoot junction (its own length excluded)."""
    lengths, total = g.property(length), 0.
    parent = g.parent(segment)
    while parent is not None and g.scale(parent) == g.scales.SubOrgan and \
            g.property("label").get(parent) == g.labels.SubOrgan.RootSegment:
        total += float(lengths.get(parent, 0.))
        parent = g.parent(parent)
    return total


def _rectangle_sides(rectangle):
    """The two side lengths of a (rotated) rectangle polygon (mm)."""
    xy = np.asarray(rectangle.exterior.coords)[:3]
    return float(np.hypot(*(xy[1] - xy[0]))), float(np.hypot(*(xy[2] - xy[1])))


def remove_anatomy(g, segment) -> None:
    """Remove *segment*'s anatomy (its Layers with their Cells and protoplasts, its anchored walls, junctions and
    anatomy Connections), keeping the segment."""
    s = g.scales
    g.remove_instance(segment)                       # an instanced anatomy (none for a copy)
    anchored = g.anchored(segment)
    g.remove_connections([c for c in anchored if g.scale(c) == s.Connection])
    g.remove_connections([c for c in anchored if g.scale(c) == s.Compartment])
    g.remove_connections([c for c in g.components(segment) if g.scale(c) == s.Layer])


def _protoplasts(g, segment) -> list:
    """The Compartments decomposing *segment* (its Cells' protoplasts), in g.components_at_scale's order: its Layers',
    then their Cells', then each Cell's protoplasts, in creation order (for an instance, from its template)."""
    block = g.instances.block(segment) if g.has_instances() else None
    if block is None:
        return g.components_at_scale(segment, scale=g.scales.Compartment)
    arrays = g.topology_arrays()
    vids = np.arange(block[0], block[0] + block[1], dtype=np.int64)
    scale, complex_ = arrays["scale"], arrays["complex"]
    protoplasts = vids[(scale[vids] == g.scales.Compartment) & (complex_[vids] >= block[0])]
    cells = complex_[protoplasts]
    order = np.lexsort((protoplasts, cells, complex_[cells]))
    return protoplasts[order].tolist()


def _instance(g, comps):
    """(template, template node of each of *comps*) when *comps* are exactly one copy of a template, else None."""
    vids = np.asarray(comps, dtype=np.int64)
    if vids.size == 0:
        return None
    order = np.argsort(vids, kind="stable")
    found, templates = g.values_at("template", vids[order])
    found_nodes, nodes = g.values_at("template_node", vids[order])
    if found.size != vids.size or found_nodes.size != vids.size or not (templates == templates[0]).all():
        return None
    out = np.empty(vids.size, dtype=np.int64)
    out[order] = np.asarray(nodes, dtype=np.int64)
    if not np.array_equal(np.sort(out), np.arange(vids.size)):
        return None
    return int(templates[0]), out


def _positions(g, vids):
    """(angle and radius present, x, y) of *vids*, computed value by value as the original rule does."""
    angle, radius = g.view("angle"), g.view("relative_radius")
    present = np.array([v in angle for v in vids], dtype=bool)
    xy = np.zeros((len(vids), 2))
    for i, v in enumerate(vids):
        if present[i]:
            xy[i] = (radius[v] * np.cos(angle[v]), radius[v] * np.sin(angle[v]))
    return present, xy


def granap_inheritance(ds, new_vids, is_known, fresh=()) -> np.ndarray:
    """
    The source of each new Compartment (Q-G3): the Compartment of the same template node in the nearest upstream
    section that has it; else the known protoplast of the same tissue nearest in (angle, relative radius) in the
    nearest upstream section having that tissue; else none (-1, the default). Sections in *fresh* (replaced
    anatomies, Q-G2) inherit nothing.

    Sections are template copies: the rule is applied per section and per upstream section, on their template nodes
    (a section that is not a whole copy, or an upstream section partly known, falls back to the rule vertex by
    vertex, _inheritance_by_vertex). The same sources.
    """
    g = ds.mtg
    from_scale = ds.from_scale
    new_vids = np.asarray(new_vids, dtype=np.int64)
    sources = np.full(new_vids.size, -1, dtype=np.int64)
    sections = ds.sections_of(new_vids)
    owners = g.compartments_by_owner(from_scale)
    tissue = g.view("tissue")
    upstream_cache, known_sections = {}, {}

    def upstream(section):
        if section not in upstream_cache:
            chain, vertex = [], g.linked_parent(int(section), from_scale)
            while vertex is not None:
                chain.append(int(vertex))
                vertex = g.linked_parent(vertex, from_scale)
            upstream_cache[section] = chain
        return upstream_cache[section]

    def known_section(section):
        """("all", copy) when every Compartment is known and they form one copy; "none"; or "partial"."""
        if section not in known_sections:
            comps = owners.get(section, [])
            known = np.asarray(is_known(np.asarray(comps, dtype=np.int64)), dtype=bool) if comps else \
                np.zeros(0, dtype=bool)
            if not known.any():
                known_sections[section] = ("none", None)
            elif known.all() and _instance(g, comps) is not None:
                known_sections[section] = ("all", (comps, _instance(g, comps)))
            else:
                known_sections[section] = ("partial", None)
        return known_sections[section]

    slow = []
    for section in np.unique(sections).tolist():
        rows = np.flatnonzero(sections == section)
        if section < 0 or section in fresh:
            continue
        here = _instance(g, new_vids[rows].tolist())
        chain = upstream(section)
        states = [known_section(a) for a in chain]
        if here is None or any(state == "partial" for state, _ in states):
            slow.append(rows)
            continue
        template, nodes = here
        ancestors = [copy for state, copy in states if state == "all"]
        same = next((copy for copy in ancestors if copy[1][0] == template), None)
        if same is not None:                                     # the same template node upstream
            comps, (_, their_nodes) = same
            vid_of_node = np.empty(their_nodes.size, dtype=np.int64)
            vid_of_node[their_nodes] = comps
            sources[rows] = vid_of_node[nodes]
            continue
        vids = new_vids[rows].tolist()
        own_tissue = [tissue.get(v) for v in vids]
        present, xy = _positions(g, vids)
        pending = np.array([t is not None for t in own_tissue], dtype=bool)
        for comps, _ in ancestors:                               # same tissue, nearest relative position
            if not pending.any():
                break
            their_present, their_xy = _positions(g, comps)
            their_tissue = [tissue.get(v) for v in comps]
            by_tissue = {}
            for j, t in enumerate(their_tissue):
                if their_present[j]:
                    by_tissue.setdefault(t, []).append(j)
            for i in np.flatnonzero(pending).tolist():
                candidates = by_tissue.get(own_tissue[i])
                if candidates and present[i]:
                    distance = np.linalg.norm(their_xy[candidates] - xy[i], axis=1)
                    sources[rows[i]] = comps[candidates[int(np.argmin(distance))]]
                    pending[i] = False
    if slow:
        rows = np.concatenate(slow)
        sources[rows] = _inheritance_by_vertex(ds, new_vids[rows], is_known, fresh)
    return sources


class GranapInheritance:
    """The DataStructure's inheritance rule (ds.inheritance) for GRANAP's anatomies: granap_inheritance, the sections
    of *fresh* (the anatomies the last update replaced) inheriting nothing. An object, not a closure: the Scene's
    checkpoints keep it."""

    def __init__(self, fresh=()):
        self.fresh = set(fresh)

    def __call__(self, ds, new_vids, is_known) -> np.ndarray:
        return granap_inheritance(ds, new_vids, is_known, self.fresh)


def _inheritance_by_vertex(ds, new_vids, is_known, fresh=()) -> np.ndarray:
    """granap_inheritance, Compartment by Compartment (for sections that are not whole template copies)."""
    g = ds.mtg
    from_scale = ds.from_scale
    new_vids = np.asarray(new_vids, dtype=np.int64)
    sources = np.full(new_vids.size, -1, dtype=np.int64)
    sections = ds.sections_of(new_vids)
    owners = g.compartments_by_owner(from_scale)
    template, node = g.view("template"), g.view("template_node")
    tissue, angle, radius = g.view("tissue"), g.view("angle"), g.view("relative_radius")
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


@dataclass
class GranapAnatomy(StructuralComponent):
    """
    The cross-section anatomies of a population's segments (roots, and the stems and leaves of *shoot*): each segment
    gets the anatomy of its class, one template generated per class and plant parametrization, a copy per segment.
    One instance serves the whole population.

    Parametrized by each plant's scenario, one dict per kind of segment, on top of GRANAP's preset:

        scenario = {"root_anatomy": {"cortex": {"n_layers": 3}},
                    "leaf_anatomy": {"preset": "for_needle", "seed": 42}}

      root_anatomy   root segments (default preset for_root; always generated)
      stem_anatomy   stem elements (default for_monocot_stem; generated when the key is given)
      leaf_anatomy   leaf elements (default for_monocot_leaf; generated when the key is given)
      sheath_anatomy the stem elements of a Sheath organ (default for_monocot_leaf bent round the stem; generated
                     when the key is given, else a sheath has no anatomy)
      seed_anatomy   the seed (SubOrgan label Seed: the seminal roots' bearer; default for_monocot_stem, the seed a
                     conducting node of the stem's anatomy for now, classed by stem_radius as the stems; generated
                     when the key is given)
      Each: "preset" (an OrganInputData preset: its organ class follows, e.g. for_needle → NeedleAnatomy), "seed",
      and GRANAP's sections and fields ({section: {field: value}}), among them the cells' wall thickness by tissue
      (the optional section cell_wall_thickness, µm: "root_anatomy.cell_wall_thickness.endodermis": 3.). Within a
      plant, segments of a kind are classed by their diameter (stems and roots, bins of 0.05 mm) and get one
      template per class.

    Advanced (class attributes): root, shoot ({SubOrgan label name: OrganAnatomy}) replace a kind's library and
    class inputs (e.g. other anatomies by distance from the tip, ClassInput targets, class_parameters); the plant's
    dict (but its preset) still applies on top. library: the roots' default library. mm_per_unit: the MPG's length
    unit in mm (1000: lengths in m).

    Declared below: the inputs read from the architecture model (shared by name with an architecture declaring them,
    as the stand-ins of architectures.py do, or linked by the plant model's translator, e.g. stem_radius), the
    planting's positions, and the state variables a coupled model (e.g. MECHA's) reads: the network of every section
    (Compartments and Connections, quantities in mm, mm2, mm3), its codes, and the 3D quantities recomputed as the
    segments elongate.
    """
    # ── Inputs: the segments' architecture ──
    radius: float = input_variable(unit="m", description="the segment's radius (the diameter class input)",
                                   references="GRANAP", by="Architecture", scale=SCALES.SubOrgan,
                                   kind="intensive")
    stem_radius: float = input_variable(unit="m", description="a stem element's (or the seed's) radius (the stems' "
                                        "and the seed's diameter class input; e.g. linked from an architecture's "
                                        "organ diameter)",
                                        references="GRANAP", by="Architecture", scale=SCALES.SubOrgan,
                                        kind="descriptor")
    distance_from_tip: float = input_variable(unit="m", description="the segment's distance from its axis' tip",
                                              references="GRANAP", by="Architecture", scale=SCALES.SubOrgan,
                                              kind="intensive")
    length: float = input_variable(unit="m", description="the segment's length, broadcast to its Compartments",
                                   references="GRANAP", by="Architecture", scale=SCALES.SubOrgan, location="node",
                                   kind="intensive")
    # ── Inputs: the segments' ends, where display_triangles places the sections (optional: not registered when no
    # architecture gives them, the sections then along a straight axis; the triangles light models get stay the
    # architecture's) ──
    x1: float = input_variable(unit="m", description="the segment's start, x (display only)",
                               references="GRANAP", by="Architecture", scale=SCALES.SubOrgan, kind="descriptor",
                               optional=True)
    y1: float = input_variable(unit="m", description="the segment's start, y (display only)",
                               references="GRANAP", by="Architecture", scale=SCALES.SubOrgan, kind="descriptor",
                               optional=True)
    z1: float = input_variable(unit="m", description="the segment's start, z (display only)",
                               references="GRANAP", by="Architecture", scale=SCALES.SubOrgan, kind="descriptor",
                               optional=True)
    x2: float = input_variable(unit="m", description="the segment's end, x (display only)",
                               references="GRANAP", by="Architecture", scale=SCALES.SubOrgan, kind="descriptor",
                               optional=True)
    y2: float = input_variable(unit="m", description="the segment's end, y (display only)",
                               references="GRANAP", by="Architecture", scale=SCALES.SubOrgan, kind="descriptor",
                               optional=True)
    z2: float = input_variable(unit="m", description="the segment's end, z (display only)",
                               references="GRANAP", by="Architecture", scale=SCALES.SubOrgan, kind="descriptor",
                               optional=True)
    # ── Inputs: the plants' positions (the planting's), where display_triangles draws the sections ──
    x: float = planting_variable("x")
    y: float = planting_variable("y")
    z: float = planting_variable("z")

    # ── Parameters: each plant's anatomies (its scenario's) ──
    root_anatomy: object = parameter(unit="", description="the plant's root anatomy: {'preset', 'seed', section: "
                                     "{field: value}} on top of its preset (None: the library's)", default=None,
                                     dtype="object", scale=SCALES.Plant, references="GRANAP",
                                     sections=lambda: section_fields("root_anatomy"))
    stem_anatomy: object = parameter(unit="", description="the plant's stem anatomy, as root_anatomy (None: no stem "
                                     "anatomy unless shoot gives one)", default=None, dtype="object",
                                     scale=SCALES.Plant, references="GRANAP",
                                     sections=lambda: section_fields("stem_anatomy"))
    leaf_anatomy: object = parameter(unit="", description="the plant's leaf anatomy, as root_anatomy (None: none)",
                                     default=None, dtype="object", scale=SCALES.Plant, references="GRANAP",
                                     sections=lambda: section_fields("leaf_anatomy"))
    sheath_anatomy: object = parameter(unit="", description="the plant's sheath anatomy, as root_anatomy (None: "
                                       "none; given: by default a monocot leaf bent round the stem)", default=None,
                                       dtype="object", scale=SCALES.Plant, references="GRANAP",
                                       sections=lambda: section_fields("sheath_anatomy"))
    seed_anatomy: object = parameter(unit="", description="the plant's seed anatomy, as root_anatomy (None: none; "
                                     "given: by default the monocot stem's, the seed a conducting node, classed by "
                                     "its stem_radius)", default=None, dtype="object", scale=SCALES.Plant,
                                     references="GRANAP", sections=lambda: section_fields("seed_anatomy"))
    soil_node: bool = parameter(unit="", description="True: each root segment's anatomy has one soil node (kind 4), "
                                "joined to its border walls and junctions (network_path 7): the section's one soil "
                                "boundary", default=False, scale=SCALES.Plant, references="GRANAP")
    instanced_anatomy: bool = parameter(unit="", description="True: each segment's anatomy an instance of its template "
                                        "(g.add_instance), no vertex of its own; False: a copy (the reference)",
                                        default=False, scale=SCALES.Plant, references="GRANAP")

    # ── State variables: the anatomies ──
    anatomy_template: float = state_variable(unit="", description="the segment's anatomy template (-1: none)",
                                             references="GRANAP", default=-1., scale=SCALES.SubOrgan,
                                             kind="descriptor")
    tissue: object = state_variable(unit="", description="the protoplast's tissue (None: a wall or a junction)",
                                    references="GRANAP", default=None, scale=SCALES.Compartment, location="node",
                                    dtype=object, kind="descriptor")
    kind: float = state_variable(unit="", description="0 protoplast, 1 wall, 2 wall junction, 3 a leaf organ's surface",
                                 references="GRANAP", default=0., scale=SCALES.Compartment, location="node",
                                 kind="descriptor")
    pore_width: float = state_variable(unit="mm",
                                       description="a stomatal pore's aperture width in the section (NaN: not a pore)",
                                       references="GRANAP", default=np.nan, scale=SCALES.Compartment,
                                       location="node", kind="NonInertialIntensive")
    pore_depth: float = state_variable(unit="mm",
                                       description="a stomatal pore's depth through the epidermis (NaN: not a pore)",
                                       references="GRANAP", default=np.nan, scale=SCALES.Compartment,
                                       location="node", kind="NonInertialIntensive")
    cgroup: float = state_variable(unit="", description="the protoplast's MECHA cell group (NaN: apoplast)",
                                   references="GRANAP", default=np.nan, scale=SCALES.Compartment, location="node",
                                   kind="descriptor")
    section_x: float = state_variable(unit="mm", description="position in the section", references="GRANAP",
                                      default=np.nan, scale=SCALES.Compartment, location="node",
                                      kind="NonInertialIntensive")
    section_y: float = state_variable(unit="mm", description="position in the section", references="GRANAP",
                                      default=np.nan, scale=SCALES.Compartment, location="node",
                                      kind="NonInertialIntensive")
    wall_length: float = state_variable(unit="mm", description="the wall's length in the section (0: not a wall)",
                                        references="GRANAP", default=0., scale=SCALES.Compartment, location="node",
                                        kind="NonInertialExtensive")
    wall_thickness: float = state_variable(unit="µm", description="the wall's thickness: its cells' wall layers (by "
                                                                  "tissue: the anatomy's cell_wall_thickness section; "
                                                                  "NaN: not a wall)", references="GRANAP",
                                           default=np.nan,
                                           scale=SCALES.Compartment, location="node",
                                           kind="NonInertialIntensive")
    cell_area: float = state_variable(unit="mm2", description="the protoplast's area in the section (0: apoplast)",
                                      references="GRANAP", default=0., scale=SCALES.Compartment, location="node",
                                      kind="NonInertialExtensive")
    wall_surface: float = state_variable(unit="mm2", description="the wall's surface: its length times the segment's",
                                         references="GRANAP", default=0., scale=SCALES.Compartment, location="node",
                                         kind="NonInertialExtensive")
    cell_volume: float = state_variable(unit="mm3",
                                        description="the protoplast's volume: its area times the segment's length",
                                        references="GRANAP", default=0., scale=SCALES.Compartment, location="node",
                                        kind="NonInertialExtensive")
    network_path: float = state_variable(unit="", description="edge type: membrane 0, wall_air 1, wall 2, "
                                                              "plasmodesmata 3, air_link 4, axial junction 5, "
                                                              "stomatal 6", references="GRANAP", default=np.nan,
                                         scale=SCALES.Connection, location="edge", kind="descriptor")
    edge_length: float = state_variable(unit="mm", description="the edge's length (the wall's for membranes; between "
                                                               "the sections for axial junctions)", references="GRANAP",
                                        default=np.nan, scale=SCALES.Connection, location="edge",
                                        kind="NonInertialIntensive")
    edge_dx: float = state_variable(unit="mm", description="the edge's vector in the section, x", references="GRANAP",
                                    default=np.nan, scale=SCALES.Connection, location="edge",
                                    kind="NonInertialIntensive")
    edge_dy: float = state_variable(unit="mm", description="the edge's vector in the section, y", references="GRANAP",
                                    default=np.nan, scale=SCALES.Connection, location="edge",
                                    kind="NonInertialIntensive")
    edge_dist: float = state_variable(unit="mm", description="the distance between the edge's nodes (NaN: wall edges)",
                                      references="GRANAP", default=np.nan, scale=SCALES.Connection, location="edge",
                                      kind="NonInertialIntensive")
    edge_wall_thickness: float = state_variable(unit="µm",
                                                description="the wall's thickness on membrane and wall edges",
                                                references="GRANAP", default=np.nan, scale=SCALES.Connection,
                                                location="edge", kind="NonInertialIntensive")
    lateral_distance: float = state_variable(unit="mm", description="wall edges: the lateral distance",
                                             references="GRANAP", default=np.nan, scale=SCALES.Connection,
                                             location="edge", kind="NonInertialIntensive")
    distnode_wall_cell: float = state_variable(unit="mm", description="wall edges: the mean wall-cell distance",
                                               references="GRANAP", default=np.nan, scale=SCALES.Connection,
                                               location="edge", kind="NonInertialIntensive")
    membrane_surface: float = state_variable(unit="mm2", description="membrane edges: the membrane's surface (the "
                                                                     "wall's length times the segment's); 0 otherwise",
                                             references="GRANAP", default=0., scale=SCALES.Connection,
                                             location="edge", kind="NonInertialExtensive")

    library = AnatomyLibrary()
    root = None                      # the roots' OrganAnatomy (None: library, its classes by diameter)
    shoot = {}                       # {SubOrgan label name: OrganAnatomy}, e.g. stems and leaves (roots: library)
    leaf_labels = ("LeafElement", "Sheath")   # the kinds of leaves (SubOrgan labels, Sheath): their organs get a
                                              # surface node
    mm_per_unit = MM_PER_M

    @classmethod
    def organs(cls) -> dict:
        """{SubOrgan label name: OrganAnatomy}: the root segments (cls.root, or cls.library by diameter), and cls.shoot."""
        return {"RootSegment": cls.root or OrganAnatomy(cls.library, "radius"), **cls.shoot}

    @classmethod
    def _label_name(cls, g, segment) -> str:
        """The segment's kind: its SubOrgan label's name, or its organ's for the kinds named by their organ (Sheath)."""
        labels = g.property("label")
        code = labels.get(segment)
        name = next((name for name in ANATOMY_KEYS if getattr(g.labels.SubOrgan, name, None) == code), None)
        organ = None
        for organ_name, element in ORGAN_KINDS.items():
            if name == element:
                organ = int(g.complex_at_scale(segment, g.scales.Organ)) if organ is None else organ
                if labels.get(organ) == getattr(g.labels.Organ, organ_name):
                    return organ_name
        return name

    @classmethod
    def anatomy_settings(cls, g, segment) -> dict:
        """The segment's plant's scenario dict for the segment's kind (root_anatomy...), {} if not given."""
        plant = g.complex_at_scale(segment, g.scales.Plant)
        key = ANATOMY_KEYS.get(cls._label_name(g, segment))
        return dict(cls._plant_value(g, plant, key) or {}) if key is not None else {}

    @classmethod
    def _plant_value(cls, g, plant, name):
        """The plant's value of the declared Plant parameter *name* (its MPG property), else its declared default."""
        value = g.properties().get(name, {}).get(plant)
        return getattr(cls, name) if value is None else value

    @classmethod
    def handled_labels(cls, g, plant) -> tuple:
        """The SubOrgan labels the plant's anatomies are generated for: roots, the explicit shoot, the kinds its
        scenario gives."""
        return tuple(name for name, key in ANATOMY_KEYS.items()
                     if name == "RootSegment" or name in cls.shoot or cls._plant_value(g, plant, key) is not None)

    @classmethod
    def organ_of(cls, g, segment) -> OrganAnatomy:
        """The segment's library and class inputs: the explicit ones (root, shoot), else its scenario's preset."""
        name = cls._label_name(g, segment)
        explicit = cls.root if name == "RootSegment" else cls.shoot.get(name)
        settings = cls.anatomy_settings(g, segment)
        if explicit is not None:
            if "preset" in settings:
                raise ValueError(f"{cls.__name__}: {ANATOMY_KEYS[name]} gives a preset, but {name} has an explicit "
                                 "OrganAnatomy (root / shoot): give its sections and fields only")
            return explicit
        key = ANATOMY_KEYS[name]
        if name == "RootSegment" and not set(RESERVED) & set(settings):
            return OrganAnatomy(cls.library, "radius")
        return OrganAnatomy(preset_library(settings.get("preset", DEFAULT_PRESETS[key]), settings.get("seed", 0)),
                            SIZES[key])

    @classmethod
    def has_soil_node(cls, g, segment) -> bool:
        """Whether the segment is a root segment whose plant has soil nodes (its scenario's soil_node)."""
        if g.property("label").get(segment) != g.labels.SubOrgan.RootSegment:
            return False
        plant = g.complex_at_scale(segment, g.scales.Plant)
        return bool(cls._plant_value(g, plant, "soil_node"))

    @classmethod
    def add_soil_node(cls, g, segment, written) -> tuple:
        """
        The section's soil node (kind 4, tissue "soil", anchored to the segment as its walls) and its Connections
        (network_path 7, from the soil node) to every border candidate of the section *written*: the walls with a
        single membrane (the outer walls, MECHA's border walls among them) and the junctions on them. Returns (the
        soil node, its Connections); *written*'s apoplast and connections extended with them.
        """
        s, L = g.scales, g.labels
        nodes = np.asarray(sorted(written["apoplast"]), dtype=np.int64)
        edges = np.asarray(sorted(written["connections"]), dtype=np.int64)
        kind = _values_at(g, "kind", nodes, np.nan)
        path = _values_at(g, "network_path", edges, np.nan)
        a = _values_at(g, "n_id_a", edges, -1.).astype(np.int64)
        b = _values_at(g, "n_id_b", edges, -1.).astype(np.int64)
        walls, junctions = nodes[kind == WALL], nodes[kind == JUNCTION]
        membrane = path == columns.PATH_CODES["membrane"]
        wall_end = np.where(np.isin(a, walls), a, b)
        count = dict(zip(*np.unique(wall_end[membrane & (np.isin(a, walls) | np.isin(b, walls))], return_counts=True)))
        outer = np.array([w for w in walls.tolist() if count.get(w, 0) == 1], dtype=np.int64)
        on_wall = path == columns.PATH_CODES["wall"]
        touching = (np.isin(a, outer) & np.isin(b, junctions)) | (np.isin(b, outer) & np.isin(a, junctions))
        on_outer = np.unique(np.where(np.isin(a, junctions), a, b)[on_wall & touching])
        candidates = np.union1d(outer, on_outer).tolist()
        soil = g.add_components_bulk(s.anchors[s.Compartment], 1, topo_parents=[segment], **_build_props(
            s.Compartment, label=L.Compartment.Apoplastic, kind=float(SOIL_NODE)))[0]
        g.set_values("tissue", [soil], ["soil"])
        # its place in the section's template, after the template's nodes: the section stays one copy of it (the
        # vessel pairs cached per template, the inheritance by template node)
        g.set_values("template", [soil], [float(written["template"])])
        g.set_values("template_node", [soil], [float(len(written["protoplasts"]) + len(written["apoplast"]))])
        for name in columns.NODE_COLUMNS + ("pore_width", "pore_depth"):
            g.set_values(name, [soil], [0. if name in ("wall_length", "cell_area") else np.nan])
        for name, value in cls._declared_initial("node").items():
            if soil not in g.property(name):
                g.set_values(name, [soil], [value])
        n = len(candidates)
        links = g.add_components_bulk(s.anchors[s.Connection], n, topo_parents=[segment] * n, **_build_props(
            s.Connection, label=L.Connection.Apoplastic, n_id_a=[int(soil)] * n, n_id_b=candidates))
        values = {name: [np.nan] * n for name in columns.EDGE_COLUMNS}
        values.update(network_path=[float(columns.SOIL_PATH)] * n, membrane_surface=[0.] * n)
        for name, column in values.items():
            g.set_values(name, links, column)
        for name, value in cls._declared_initial("edge").items():
            if name not in values:
                g.set_values(name, links, [value] * n)
        written["apoplast"] = list(written["apoplast"]) + [int(soil)]
        written["connections"] = list(written["connections"]) + list(links)
        return soil, links

    @staticmethod
    def conduits(template) -> np.ndarray:
        """The conduits of a SectionTemplate (the protoplasts of families 0 and 1, which the junctions join): a
        reduction's ports (solve.reduction: the nodes a section keeps for its neighbours)."""
        out = np.zeros(template.size, dtype=bool)
        rows, values = template.properties.get("family", (np.zeros(0, dtype=np.int64), np.zeros(0)))
        out[rows] = np.isin(np.asarray(values, dtype=float), (0., 1.))
        return out

    @classmethod
    def is_instanced(cls, g, segment) -> bool:
        """Whether the segment's plant has instanced anatomies (its scenario's instanced_anatomy)."""
        plant = g.complex_at_scale(segment, g.scales.Plant)
        return bool(cls._plant_value(g, plant, "instanced_anatomy"))

    @classmethod
    def segment_class(cls, g, segment) -> tuple:
        """The segment's anatomy class: its organ's class inputs, binned."""
        organ = cls.organ_of(g, segment)
        cls.check_class_inputs(organ)
        return class_key(organ.class_inputs, g, segment)

    @classmethod
    def check_class_inputs(cls, organ) -> None:
        """A class input reads a variable the component declares (an input the plant model's translator links, e.g.
        stem_radius from an architecture's organ diameter), never another model's property by name."""
        declared = {f.name for f in fields(cls) if f.metadata.get("variable_type") in ("input", "parameter")}
        for c in organ.class_inputs:
            if c.compute is None and c.width > 0 and c.variable not in declared:
                raise ValueError(f"{cls.__name__}: the class input '{c.variable}' is not a declared input: declare it "
                                 f"(input_variable(..., by=\"Architecture\", scale=SubOrgan)) and link it in the "
                                 "plant model's translator")

    @classmethod
    def initiate_plant(cls, g, plant_vid: int, parameters: dict) -> None:
        """Each segment of the plant gets its class's anatomy, parametrized by the plant's scenario (root_anatomy,
        stem_anatomy, leaf_anatomy), kept on the plant for the anatomies generated later."""
        parameters = parameters or {}
        for key in tuple(ANATOMY_KEYS.values()) + ("soil_node", "instanced_anatomy"):   # its declared Plant parameters
            if key in parameters:
                value = parameters[key]
                g.property(key)[plant_vid] = dict(value) if isinstance(value, dict) else value
        segments = _segments(g, plant_vid, cls.handled_labels(g, plant_vid), cls._label_name)
        cls.update_anatomies(g, segments)
        cls.update_leaf_surfaces(g, segments)

    @classmethod
    def update_anatomies(cls, g, segments, lengths=None) -> dict:
        """
        Give each segment the anatomy of its current class (Q12): a segment without one (just emerged) gets its
        class's copy; a segment whose class changed has its anatomy replaced. Returns {"emerged", "replaced"}.
        A segment without length (an Adel internode or sheath before it grows: 0, or none given) gets none until it
        has one: it is a pass-through meanwhile, as a hidden element, the junctions spanning it (a section of zero
        height would have no radial conductance, its nodes cut off from the network). *lengths*: {segment: length},
        by default the MPG's property.
        """
        anatomy_class = g.properties().setdefault("anatomy_class", {})
        template_of = g.properties().setdefault("anatomy_template", {})
        lengths = g.property("length") if lengths is None else lengths
        done = {"emerged": [], "replaced": []}
        for segment in segments:
            current = anatomy_class.get(segment)
            if current is None and not float(lengths.get(segment, 0.) or 0.) > 0.:
                continue
            key = cls.segment_class(g, segment)
            if current == key:
                continue
            if current is not None:
                remove_anatomy(g, segment)
                done["replaced"].append(segment)
            else:
                done["emerged"].append(segment)
            organ = cls.organ_of(g, segment)
            own = {k: v for k, v in cls.anatomy_settings(g, segment).items() if k not in RESERVED}
            plant = merged(DEFAULT_OVERRIDES.get(ANATOMY_KEYS.get(cls._label_name(g, segment)), {}), own) or None
            template = organ.library.template(key, plant=plant, targets=class_targets(organ.class_inputs, key))
            instanced, soil = cls.is_instanced(g, segment), cls.has_soil_node(g, segment)
            reduction = g.reduction                             # the plant model's (solve.reduction)
            if reduction is not None and reduction.reduces(cls._label_name(g, segment)):
                instanced = True
                full = template.section_template(cls.add_soil_node if soil else None, "soil" if soil else "")
                written = template.instance_of(g, segment, reduction.template_for(full))
            elif instanced:
                written = template.instance(g, segment, cls.add_soil_node if soil else None, "soil" if soil else "")
            else:
                written = template.instantiate(g, segment)
                if soil:
                    cls.add_soil_node(g, segment, written)
            _write_3d(g, written, float(g.property("length").get(segment, 0.)) * cls.mm_per_unit)
            if not instanced:                           # the pores' (leaf surfaces); an instance has its template's
                nodes = list(written["protoplasts"]) + list(written["apoplast"])
                g.set_values("pore_width", nodes, [np.nan] * len(nodes))
                g.set_values("pore_depth", nodes, [np.nan] * len(nodes))
            anatomy_class[segment] = key
            template_of[segment] = float(template.index)
        return done

    @classmethod
    def update_leaf_surfaces(cls, g, segments) -> None:
        """
        The leaf organs of *segments* get (again) their surface: one Compartment of kind 3 (tissue "leaf surface"),
        owned by the organ's basal leaf element with an anatomy (one without, e.g. not grown yet, owns none), and one
        Connection (network_path 6) from every stomatal pore of each of the organ's sections to it, anchored to the
        pore's section. The pores get their
        aperture's width and depth (pore_width, pore_depth: the short and long sides of their polygon's minimum
        rotated rectangle). The organ's previous surface and stomatal Connections are removed first, so that none
        dangles when a section was replaced.
        """
        import shapely
        s, L = g.scales, g.labels
        labels = g.property("label")
        organs = sorted({int(g.complex_at_scale(v, s.Organ)) for v in segments if cls._label_name(g, v) in cls.leaf_labels})
        if not organs:
            return
        surfaces = g.properties().setdefault("leaf_surface", {})
        anatomy_class = g.properties().get("anatomy_class", {})
        path, tissue = g.view("network_path"), g.view("tissue")
        template_row = g.view("template_row")
        for organ in organs:
            elements = [int(v) for v in g.components_at_scale(organ, scale=s.SubOrgan)
                        if cls._label_name(g, v) in cls.leaf_labels]
            stomatal = [c for e in elements for c in g.anchored(e)
                        if g.scale(c) == s.Connection and path.get(c) == columns.STOMATAL_PATH]
            g.remove_connections(stomatal)
            old = surfaces.pop(organ, None)
            if old is not None and g.has_vertex(old):
                g.remove_connections([old])
            pores = {e: [int(c) for c in _protoplasts(g, e) if tissue.get(c) == "pore"] for e in elements}
            if not any(pores.values()):
                continue
            basal = _basal_with_anatomy(g, elements, anatomy_class)
            surface = g.add_components_bulk(s.anchors[s.Compartment], 1, topo_parents=[basal], **_build_props(
                s.Compartment, label=L.Compartment.Apoplastic, kind=float(LEAF_SURFACE)))[0]
            g.set_values("tissue", [surface], ["leaf surface"])
            for name in columns.NODE_COLUMNS + ("pore_width", "pore_depth"):
                g.set_values(name, [surface], [0. if name in ("wall_length", "cell_area") else np.nan])
            for name, value in cls._declared_initial("node").items():           # the other declared variables
                if surface not in g.property(name):
                    g.set_values(name, [surface], [value])
            surfaces[organ] = int(surface)
            template_of = g.property("anatomy_template")
            for element, vids in pores.items():
                if not vids:
                    continue
                template = template_by_index(template_of[element])
                rectangles = [shapely.minimum_rotated_rectangle(template.polygons[int(template_row[v])]) for v in vids]
                sides = np.array([sorted(_rectangle_sides(r)) for r in rectangles])
                g.set_values("pore_width", vids, sides[:, 0].tolist())
                g.set_values("pore_depth", vids, sides[:, 1].tolist())
                n = len(vids)
                edges = g.add_components_bulk(s.anchors[s.Connection], n, topo_parents=[element] * n, **_build_props(
                    s.Connection, label=L.Connection.Apoplastic, n_id_a=vids, n_id_b=[int(surface)] * n))
                values = {name: [np.nan] * n for name in columns.EDGE_COLUMNS}
                values.update(network_path=[float(columns.STOMATAL_PATH)] * n, edge_length=sides[:, 1].tolist(),
                              membrane_surface=[0.] * n)
                for name, column in values.items():
                    g.set_values(name, edges, column)
                for name, value in cls._declared_initial("edge").items():
                    if name not in values:
                        g.set_values(name, edges, [value] * n)

    @classmethod
    def _declared_initial(cls, location) -> dict:
        """{name: initial value} of the declared state variables at *location* ("node" or "edge")."""
        from dataclasses import fields
        return {f.name: f.metadata.get("initialize", f.default) for f in fields(cls)
                if f.metadata.get("variable_type") == "state_variable" and f.metadata.get("location") == location}

    def __post_init__(self):
        g = self.data_structure.mtg
        segments = set(self.data_structure.entity_ids("SubOrgan").tolist())
        template_of = g.properties().setdefault("anatomy_template", {})
        for segment in segments - set(template_of):            # segments without an anatomy (labels not handled)
            template_of[segment] = -1.
        super().__post_init__()
        self.data_structure.inheritance = GranapInheritance()

    @postsegmentation
    def _anatomies(self):
        """MPG-style: every segment of the emerged plants gets its class's anatomy (emerged, or replaced on a change);
        the records of the segments pruned in any plant go."""
        g = self.mtg
        by_plant = {int(plant): _segments(g, plant, self.handled_labels(g, plant), self._label_name)
                    for plant in g.components_at_scale(g.root, scale=g.scales.Plant)}
        emerged = set(self.active_ids("Plant").tolist())
        segments = [s for plant, these in by_plant.items() if plant in emerged for s in these]
        alive = {s for these in by_plant.values() for s in these}
        anatomy_class = g.properties().get("anatomy_class", {})
        for gone in [s for s in list(anatomy_class) if s not in alive]:      # pruned segments: their record only
            del anatomy_class[gone]
        done = self.update_anatomies(g, segments, _suborgan_lengths(self.data_structure))
        self.update_leaf_surfaces(g, done["emerged"] + done["replaced"])
        self._wire_pass_through(done["emerged"] + done["replaced"])
        template_of = g.properties()["anatomy_template"]
        for segment in g.components_at_scale(g.root, scale=g.scales.SubOrgan):
            template_of.setdefault(int(segment), -1.)
        self.data_structure.inheritance.fresh = set(done["replaced"])
        self.last_update = done

    def _wire_pass_through(self, sections) -> None:
        """
        The pass-through junctions of the *sections* given an anatomy (emerged, replaced): those from each to its
        nearest descendants with an anatomy across SubOrgans without one. The DataStructure rewires a section's
        children only, not these descendants (their linked parent, without anatomy, unchanged): their upstream
        junctions (to the ancestor they had, an emerged section having been a pass-through) are removed and wired
        again here, with the DataStructure's wiring rules.
        """
        rules = getattr(self.data_structure, "_wiring", None)
        if not sections or not rules:
            return
        g = self.mtg
        owners = g.compartments_by_owner(g.scales.SubOrgan)
        descendants = anatomy_descendants(g, sections, owners)
        if descendants:
            heads = {int(c) for v in descendants for c in owners.get(v, [])}
            n_id_b = g.property("n_id_b")
            g.remove_connections([int(e) for e in g.junction_vids() if int(n_id_b[e]) in heads])
            g.wire_junctions(g.scales.SubOrgan, rules, children=descendants)

    def _pass_through_spans(self, network_path) -> tuple:
        """(the junctions across SubOrgans without anatomy, as edge indices; for each, those SubOrgans), the
        pass-through junctions whose length (_wall_surface) spans them; None when there is none. Cached until the
        topology changes."""
        ds = self.data_structure
        key = (ds.topology_version, ds.n_edges())
        cached = self.__dict__.get("_pass_through")
        if cached is not None and cached[0] == key:
            return cached[1]
        g = ds.mtg
        owners = g.compartments_by_owner(g.scales.SubOrgan)
        spans = None
        if not all(owners.get(int(v)) for v in ds.entity_ids("SubOrgan")):
            edges = np.asarray(ds.edges(), dtype=np.int64).reshape(-1, 2)
            rows = np.flatnonzero(np.asarray(network_path) == columns.JUNCTION_PATH)
            parents, children = ds.sections_of(edges[rows, 0]), ds.sections_of(edges[rows, 1])
            found_rows, hidden = [], []
            for child in np.unique(children).tolist():
                ancestor, between = anatomy_ancestor(g, child, owners)
                if between:
                    here = rows[(children == child) & (parents == ancestor)]
                    found_rows.extend(here.tolist())
                    hidden.extend([between] * here.size)
            if found_rows:
                spans = (np.asarray(found_rows, dtype=np.int64), hidden)
        self.__dict__["_pass_through"] = (key, spans)
        return spans

    def display_triangles(self, where=None):
        """
        The sections drawn for display (encoding3d frames, viewers), not for light models (triangles() stays the
        architecture's): each segment's section, flat, across the middle of the segment, perpendicular to it; each
        protoplast its cell's polygon, triangulated (coupling.geometry.Triangles, each triangle its node index). A
        segment is placed by its ends (x1 ... z2) when the architecture gives them, else along a straight axis from its
        plant's position (roots down, the shoot up), at its distance from the axis' base. *where*: a node mask.
        Computed when asked (a frame written): each template's polygons are triangulated once.
        """
        from openalea.metafspm.coupling.geometry import Triangles
        ds = self.data_structure
        g = ds.structure
        properties = g.properties()                   # read only: g.property() would create a missing property
        template_of, template_row = properties.get("anatomy_template", {}), g.view("template_row")
        arrays = g.topology_arrays()
        scale, complex_ = arrays["scale"], arrays["complex"]
        node_ids = np.asarray(ds.entity_ids("node"), dtype=np.int64)
        kept = np.ones(node_ids.size, dtype=bool) if where is None else np.asarray(where, dtype=bool)
        wanted = set(node_ids[kept].tolist())
        plant_position = dict(zip(ds.entity_ids("Plant").tolist(),
                                  np.column_stack([np.asarray(ds.get(name), dtype=float) for name in ("x", "y", "z")])))
        frame_values = {name: _per_segment(self, name) for name in SEGMENT_ENDS + ("length",)}
        vertices, owners = [], []
        for segment, compartments in g.compartments_by_owner(g.scales.SubOrgan).items():
            index = template_of.get(segment, -1.)
            if index is None or index < 0:
                continue
            local, rows = _template_triangles(template_by_index(index))
            row_of = {int(template_row[c]): int(c) for c in compartments
                      if int(c) in wanted and c in template_row and scale[int(c)] == g.scales.Compartment
                      and complex_[int(c)] >= 0 and scale[complex_[int(c)]] == g.scales.Cell}
            drawn = np.array([row in row_of for row in rows], dtype=bool)
            if not drawn.any():
                continue
            centre, axis = _segment_frame(g, segment, plant_position, frame_values)
            u, v = _perpendiculars(axis)
            xy = local[drawn] / self.mm_per_unit                                   # (T, 3, 2), in m
            vertices.append(centre + xy[..., :1] * u + xy[..., 1:] * v)
            owners.append(ds.index_of([row_of[row] for row in rows[drawn]], "node"))
        if not vertices:
            return Triangles(np.zeros((0, 3, 3)), np.zeros(0, dtype=np.int64))
        return Triangles(np.concatenate(vertices), np.concatenate(owners))

    @state(include_inactive=True)        # the sections' geometry, emerged or not
    def _wall_surface(self, wall_length, cell_area, length, edge_length, network_path) \
            -> dict:
        """
        The 3D quantities, as the segments elongate: the walls' surfaces, the protoplasts' volumes, the membranes'
        surfaces (2D values times the segment's length), and the axial junctions' lengths (between the two sections'
        centres, the mean of the segments' lengths, plus, across a pass-through, the lengths of the SubOrgans without
        anatomy between).
        """
        length_mm = np.asarray(length, dtype=float) * self.mm_per_unit
        edge_mm = self.data_structure.map_to(length_mm, "node", "edge", "mean")      # the mean of the edge's two nodes
        path = np.asarray(network_path)
        membranes = (path == columns.PATH_CODES["membrane"]) | (path == columns.PATH_CODES["wall_air"])
        junction_mm = edge_mm
        spans = self._pass_through_spans(path)
        if spans is not None:                   # a pass-through junction: plus the SubOrgans without anatomy between
            rows, hidden = spans
            lengths = _suborgan_lengths(self.data_structure)
            junction_mm = np.array(edge_mm, dtype=float)
            junction_mm[rows] += np.array([sum(lengths.get(v, 0.) for v in between) for between in hidden]) * \
                self.mm_per_unit
        return {"wall_surface": wall_length * length_mm, "cell_volume": cell_area * length_mm,
                "membrane_surface": np.where(membranes, edge_length * edge_mm, 0.),
                "edge_length": np.where(path == columns.JUNCTION_PATH, junction_mm, edge_length)}


_TEMPLATE_TRIANGLES = {}                  # {template index: (triangles (T, 3, 2) in mm, the cell row of each)}


def _template_triangles(template) -> tuple:
    """A template's cells' polygons as triangles in the section's plane (mm), and each triangle's cell row; once per
    template, its polygons being shared by every segment of its class."""
    if template.index not in _TEMPLATE_TRIANGLES:
        import shapely
        triangles, rows = [], []
        for row, polygon in enumerate(template.polygons):
            if polygon is None or polygon.is_empty:
                continue
            for piece in shapely.constrained_delaunay_triangles(polygon).geoms:
                triangles.append(np.asarray(piece.exterior.coords)[:3, :2])
                rows.append(row)
        _TEMPLATE_TRIANGLES[template.index] = (np.asarray(triangles, dtype=float).reshape(-1, 3, 2),
                                               np.asarray(rows, dtype=np.int64))
    return _TEMPLATE_TRIANGLES[template.index]


SEGMENT_ENDS = ("x1", "y1", "z1", "x2", "y2", "z2")


def _suborgan_lengths(ds) -> dict:
    """{SubOrgan: length} of every SubOrgan, those without anatomy (no Compartment, so no node value) included: the
    DataStructure's when the architecture holds length at the SubOrgans, else the MPG's property."""
    if ds.has("length") and ds.location("length") == "SubOrgan":
        return dict(zip(ds.entity_ids("SubOrgan").tolist(), np.asarray(ds.get("length"), dtype=float).tolist()))
    return {int(v): float(value) for v, value in ds.mtg.property("length").items()}


def _per_segment(component, name):
    """{segment: value} of the component's declared SubOrgan-scale input *name*, read at its own place
    (component.read: mapped when another component holds it elsewhere, e.g. broadcast to the nodes), or None when
    nothing gives it (an optional input: its default, NaN)."""
    ds = component.data_structure
    values = np.asarray(component.read(name), dtype=float)
    location = component.variable_specs[name].location
    if location != "SubOrgan":                        # declared at the nodes: the segment's (broadcast) value
        values = ds.map_to(values, location, "SubOrgan", "mean")
    if np.isnan(values).all():
        return None
    return dict(zip(ds.entity_ids("SubOrgan").tolist(), np.asarray(values, dtype=float).tolist()))


def _segment_frame(g, segment, plant_position, frame_values) -> tuple:
    """(middle, unit axis) of *segment*: from its ends (x1 ... z2) when the architecture gives them, else along a
    straight axis from its plant's position (*plant_position*: {plant: (x, y, z)}, the planting's), roots down and the
    shoot up, at its distance from the axis' base. *frame_values*: {name: {segment: value} or None}, the segments'
    ends and lengths (_per_segment)."""
    properties = g.properties()                       # read only: g.property() would create a missing property
    ends = [(frame_values[name] or {}).get(segment) for name in SEGMENT_ENDS]
    if all(value is not None and not np.isnan(value) for value in ends):
        start, end = np.asarray(ends[:3], dtype=float), np.asarray(ends[3:], dtype=float)
        length = np.linalg.norm(end - start)
        if length > 0:
            return (start + end) / 2., (end - start) / length
    lengths = frame_values["length"] or {}
    below, parent = 0., g.parent(segment)
    while parent is not None and g.scale(parent) == g.scales.SubOrgan:
        below += float(lengths.get(parent, 0.))
        parent = g.parent(parent)
    plant = g.complex_at_scale(segment, g.scales.Plant)
    base = np.asarray(plant_position[int(plant)], dtype=float)
    direction = -1. if properties.get("label", {}).get(segment) == g.labels.SubOrgan.RootSegment else 1.
    middle = below + float(lengths.get(segment, 0.)) / 2.
    return base + np.array([0., 0., direction * middle]), np.array([0., 0., direction])


def _perpendiculars(axis) -> tuple:
    """Two unit vectors spanning the plane perpendicular to *axis* (the section's x and y)."""
    helper = np.array([1., 0., 0.]) if abs(axis[0]) < 0.9 else np.array([0., 1., 0.])
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    return u, np.cross(axis, u)


def _vessels(g, compartments, family):
    """The conducting protoplasts of *family* among *compartments*: (vids, track ids, normalised positions)."""
    fam = g.values_of("family", compartments)
    vids = [v for v in compartments if fam.get(v, -1) == family]
    track, angle, radius = (g.values_of(name, vids) for name in ("track_id", "angle", "relative_radius"))
    xy = np.array([[radius[v] * np.cos(angle[v]), radius[v] * np.sin(angle[v])] for v in vids]).reshape(-1, 2)
    return vids, [track.get(v) for v in vids], xy


def vessel_pairs(g, parent_compartments, child_compartments, family) -> list:
    """
    The junctions between two successive sections for one family of conducting cells. When both sections are
    template copies, the pairs depend only on the two templates: computed once per (parent template, child template,
    family) on their template nodes, then given each pair's vids.
    """
    parent, child = _instance(g, parent_compartments), _instance(g, child_compartments)
    if parent is None or child is None:
        return _vessel_pairs(g, parent_compartments, child_compartments, family)
    key = (parent[0], child[0], family)                   # template ids: unique across the libraries
    local = _VESSEL_PAIRS.get(key)
    if local is None:
        node_a, node_b = dict(zip(parent_compartments, parent[1].tolist())), \
            dict(zip(child_compartments, child[1].tolist()))
        local = _VESSEL_PAIRS[key] = [(node_a[a], node_b[b])
                              for a, b in _vessel_pairs(g, parent_compartments, child_compartments, family)]
    vid_a, vid_b = np.empty(parent[1].size, dtype=np.int64), np.empty(child[1].size, dtype=np.int64)
    vid_a[parent[1]], vid_b[child[1]] = parent_compartments, child_compartments
    return sorted((int(vid_a[a]), int(vid_b[b])) for a, b in local)


def _vessel_pairs(g, parent_compartments, child_compartments, family) -> list:
    """vessel_pairs for any two sections: by track id, then the nearest in both directions."""
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


def _write_3d(g, written, length_mm) -> None:
    """A new copy's 3D quantities: its walls' surfaces, protoplasts' volumes and membranes' surfaces, for a segment of
    *length_mm* (then recomputed at each step as it elongates, GranapAnatomy._wall_surface)."""
    nodes = np.asarray(list(written["protoplasts"]) + list(written["apoplast"]), dtype=np.int64)
    g.set_values("wall_surface", nodes.tolist(), (_values_at(g, "wall_length", nodes, 0.) * length_mm).tolist())
    g.set_values("cell_volume", nodes.tolist(), (_values_at(g, "cell_area", nodes, 0.) * length_mm).tolist())
    edges = np.asarray(list(written["connections"]), dtype=np.int64)
    path = _values_at(g, "network_path", edges, np.nan)
    membranes = np.isin(path, (columns.PATH_CODES["membrane"], columns.PATH_CODES["wall_air"]))
    edge_length = _values_at(g, "edge_length", edges, np.nan)
    if np.isnan(edge_length[membranes]).any():
        raise KeyError("a membrane Connection without edge_length")
    g.set_values("membrane_surface", edges.tolist(), np.where(membranes, edge_length * length_mm, 0.).tolist())


def _values_at(g, name, vids, default) -> np.ndarray:
    """Property *name* at *vids* (ascending), *default* where a vertex has none: in bulk (MPG._property_at, the
    instanced vertices' values included)."""
    out = np.full(vids.size, float(default))
    found, values = g.values_at(name, vids)
    if found.size:
        out[np.searchsorted(vids, found)] = np.asarray(values, dtype=np.float64)
    return out


def anatomy_ancestor(g, section, owners=None) -> tuple:
    """
    (the nearest ancestor of *section* with an anatomy, [the SubOrgans without one in between, nearest first]): its
    linked parent when that has an anatomy (Compartments), else the first with one up the linked parents, through
    the SubOrgans without (a pass-through: an Adel HiddenElement, a kind the scenario does not give). (None, [...]):
    no ancestor has one. *owners*: g.compartments_by_owner of the sections' scale (default: read).
    """
    scale = g.scale(section)
    owners = g.compartments_by_owner(scale) if owners is None else owners
    between, vertex = [], g.linked_parent(section, scale)
    while vertex is not None and not owners.get(vertex):
        between.append(int(vertex))
        vertex = g.linked_parent(vertex, scale)
    return (None if vertex is None else int(vertex)), between


def anatomy_descendants(g, sections, owners=None) -> list:
    """The sections with an anatomy whose nearest ancestor with one (anatomy_ancestor) is in *sections* across at
    least one SubOrgan without anatomy: the other ends of *sections*' pass-through junctions."""
    sections = {int(v) for v in sections}
    if not sections:
        return []
    scale = g.scale(next(iter(sections)))
    owners = g.compartments_by_owner(scale) if owners is None else owners
    every = [int(v) for v in g.components_at_scale(g.root, scale=scale)]
    if all(owners.get(v) for v in every):                  # every section has an anatomy: no pass-through
        return []
    out = []
    for vertex in sorted(every):
        if owners.get(vertex):
            parent = g.linked_parent(vertex, scale)
            if parent is not None and not owners.get(parent):
                ancestor, _ = anatomy_ancestor(g, vertex, owners)
                if ancestor in sections:
                    out.append(int(vertex))
    return out


def _junction_properties(mm_per_unit, g, parent, child, pairs) -> dict:
    """The properties of the axial junctions *pairs* between the sections of *parent* and *child* (junction_geometry);
    across SubOrgans without anatomy (a pass-through), between *child* and its nearest ancestor with one, their
    length spanning the SubOrgans in between."""
    n = len(pairs)
    lengths = g.property("length")
    owners = g.compartments_by_owner(g.scale(child))
    if owners.get(parent):
        between = 0.5 * (float(lengths.get(parent, 0.)) + float(lengths.get(child, 0.))) * mm_per_unit
    else:
        ancestor, hidden = anatomy_ancestor(g, child, owners)
        between = (0.5 * (float(lengths.get(ancestor, 0.)) + float(lengths.get(child, 0.))) +
                   sum(float(lengths.get(v, 0.)) for v in hidden)) * mm_per_unit
    x, y = g.view("section_x"), g.view("section_y")
    out = {name: [np.nan] * n for name in columns.EDGE_COLUMNS}
    out.update(network_path=[float(columns.JUNCTION_PATH)] * n, edge_length=[between] * n,
               edge_dx=[x[b] - x[a] for a, b in pairs], edge_dy=[y[b] - y[a] for a, b in pairs],
               membrane_surface=[0.] * n)
    return out


def junction_geometry(mm_per_unit=MM_PER_M):
    """
    The geometry of the axial junctions a wiring rule creates (Q-C1a): their length between the two sections' centres
    (the mean of the segments' lengths, plus the lengths of the SubOrgans without anatomy between: a pass-through),
    the in-plane offset between the two vessels, the path code of an axial
    junction (5); the other network columns NaN, the membrane surface 0. A function (g, parent, child, pairs) ->
    {property: values}, picklable (the Scene's checkpoints keep the wiring).
    """
    return partial(_junction_properties, mm_per_unit)


def _family_pairs(family, g, parent, child, parent_compartments, child_compartments) -> list:
    """A wiring rule: the junctions of one family of conducting cells between two sections (vessel_pairs). When the
    child's linked parent has no anatomy (a pass-through), between the child and its nearest ancestor with one
    (anatomy_ancestor); a child without anatomy has none."""
    if not parent_compartments and child_compartments:
        owners = g.compartments_by_owner(g.scale(child))
        ancestor, _ = anatomy_ancestor(g, child, owners)
        if ancestor is None:
            return []
        parent_compartments = owners[ancestor]
    return vessel_pairs(g, parent_compartments, child_compartments, family)


def junction_rules(g, mm_per_unit=MM_PER_M) -> list:
    """
    The wiring rules of anatomy mode: axial xylem junctions (apoplastic) and phloem junctions (symplastic), with their
    geometry (junction_geometry); functions, not closures, so that the Scene's checkpoints keep them. A section whose
    linked parent has no anatomy is joined to its nearest ancestor with one (a pass-through, anatomy_ancestor).
    """
    geometry = junction_geometry(mm_per_unit)
    return [
        {"rule": partial(_family_pairs, 0), "edge_label": g.labels.Connection.Apoplastic, "properties": geometry},
        {"rule": partial(_family_pairs, 1), "edge_label": g.labels.Connection.Symplastic, "properties": geometry},
    ]
