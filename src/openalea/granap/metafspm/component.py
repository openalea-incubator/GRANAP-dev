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
scenario does not give, e.g. a sheath without sheath_anatomy; a section without length) is a pass-through: a section
whose linked parent has no anatomy is joined to its nearest ancestor with one (anatomy_ancestor), the junctions'
length spanning the elements between, so that the water goes from a stem element through the hidden elements to the
next element.
"""
import dataclasses
from dataclasses import dataclass, fields
from functools import partial

import numpy as np

from openalea.metafspm.coupling.component import (StructuralComponent, input_variable, parameter, planting_variable,
                                                  state_variable)
from openalea.metafspm.data_structure.configs import ScalesConfig
from openalea.metafspm.solve.decorator import postsegmentation, state

from openalea.granap.metafspm import columns, kernels
from openalea.granap.metafspm.build import JUNCTION, LEAF_SURFACE, SOIL_NODE, WALL, _props as _build_props
from openalea.granap.metafspm.library import (AnatomyLibrary, ClassInput, class_key, class_keys, class_targets,
                                              merged, preset_parameters, template_by_index)

SCALES = ScalesConfig()
MM_PER_M = 1000.                 # GRANAP's sections are in mm; the MPG's lengths in m


# The scenario keys of the anatomies, by SubOrgan label: each a dict {"preset": ..., "seed": ..., section: {field: value}}
# on top of its preset (GRANAP's OrganInputData), read from each plant's scenario when the plant is built
ANATOMY_KEYS = {"RootSegment": "root_anatomy", "RootApex": "root_apex_anatomy", "StemElement": "stem_anatomy",
                "LeafElement": "leaf_anatomy", "Sheath": "sheath_anatomy", "Seed": "seed_anatomy"}
DEFAULT_PRESETS = {"root_anatomy": "for_root", "root_apex_anatomy": "for_root", "stem_anatomy": "for_monocot_stem",
                   "leaf_anatomy": "for_monocot_leaf", "sheath_anatomy": "for_monocot_leaf",
                   "seed_anatomy": "for_monocot_stem"}
# The class's size input. A seed is, for now, a conducting node of the stem's anatomy (a grass's seed: the bearer of
# the seminal roots and of the subcrown internode): its size input is the stems', stem_radius, its section's radius
SIZES = {"root_anatomy": "radius", "root_apex_anatomy": "radius", "stem_anatomy": "stem_radius",
         "leaf_anatomy": None, "sheath_anatomy": None, "seed_anatomy": "stem_radius"}
# The kinds taking another kind's anatomy when their own key is not given: {kind: the kind whose anatomy it takes}. A
# root apex (an architecture's axis tip, ArchiSimple's RootApex: the part of the root beyond its last segment) is,
# by default, a root segment of its class: the root's anatomy, its scenario dict and its explicit OrganAnatomy
# (root). Its tip's tissues are the root's: the meristem is not represented; its maturity is its barriers' (MECHA's
# Development by its distance from the tip: at barrier 0, MECHA's xylem immature, without xylem junction)
KIND_DEFAULTS = {"RootApex": "RootSegment"}
# The root kinds (SubOrgan labels): the sections with a soil node, drawn downwards
ROOT_KINDS = ("RootSegment", "RootApex")
# Kinds named by their organ: {Organ label name: the SubOrgan label of its elements} (a sheath's elements are stem
# elements, Adel's, with an anatomy of their own)
ORGAN_KINDS = {"Sheath": "StemElement"}
# A kind's default overrides, under the plant's: a sheath is, for now, a monocot leaf bent round the stem (its
# mid-line sagging 2 mm below its 4 mm chord: a U)
DEFAULT_OVERRIDES = {"sheath_anatomy": {"planttype": {"fold_sag": 2.0}}}
RESERVED = ("preset", "seed")
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
    return _SECTION_FIELDS[key]


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
    """The library of an OrganInputData preset (its organ class, generated with *seed*); its templates are each
    population's (AnatomyLibrary.template)."""
    return AnatomyLibrary(parameters=preset_parameters(preset), organ_class=organ_class(preset), seed=int(seed))


def population_preset_library(mtg, preset: str, seed: int = 0) -> AnatomyLibrary:
    """preset_library's, one per (preset, seed) in the MPG *mtg*'s population (its registry,
    mtg.objects("anatomy_libraries"): rebuilt when missing, not checkpointed)."""
    return mtg.objects("anatomy_libraries", saved=False).get((preset, int(seed)), lambda: preset_library(preset, seed))


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
            ClassInput("distance_from_junction", 0., MM_PER_M, compute=lambda mtg, segment: distance_from_junction(mtg, segment)))


def _segments(mtg, plant, labels_handled=("RootSegment",), kind=None):
    """The segments of *plant* (SubOrgan vertices) whose kind is handled (default: root segments); *kind*: a function
    (mtg, segment) -> its kind's name (GranapAnatomy._label_name), by default its SubOrgan label's."""
    codes = {getattr(mtg.labels.SubOrgan, name, None) for name in labels_handled}
    codes |= {getattr(mtg.labels.SubOrgan, ORGAN_KINDS[name]) for name in labels_handled if name in ORGAN_KINDS}
    components = np.asarray(mtg.components_at_scale(plant, scale=mtg.scales.SubOrgan), dtype=np.int64)
    labels = _aligned(mtg, "label", components)                   # one read of the labels
    candidates = [int(v) for v, code in zip(components.tolist(), labels.tolist()) if code in codes]
    if kind is None:
        return candidates
    if getattr(kind, "__func__", None) is GranapAnatomy._label_name.__func__:     # the candidates' kinds in one read
        return [v for v, name in zip(candidates, kind.__self__._label_names(mtg, candidates)) if name in labels_handled]
    return [v for v in candidates if kind(mtg, v) in labels_handled]


def _aligned(mtg, name, vids, default=None) -> np.ndarray:
    """Property *name* at each of *vids*, in their order (an object array, *default* where there is none): one bulk
    read (mtg.values_at)."""
    vids = np.asarray(vids, dtype=np.int64)
    unique, inverse = np.unique(vids, return_inverse=True)
    found, values = mtg.values_at(name, unique)
    out = np.empty(unique.size, dtype=object)
    out[:] = [default] * unique.size
    out[np.searchsorted(unique, found)] = values.tolist()
    return out[inverse]


def _label_codes(mtg, names) -> set:
    """The SubOrgan label codes of the label *names* the MPG has."""
    return {getattr(mtg.labels.SubOrgan, name) for name in names if hasattr(mtg.labels.SubOrgan, name)}


def _on_top(base: dict, own: dict) -> dict:
    """A scenario dict *own* on top of *base*: its sections' fields over theirs, its preset and seed over theirs."""
    out = {key: dict(value) if isinstance(value, dict) else value for key, value in base.items()}
    for key, value in own.items():
        out[key] = {**out[key], **value} if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


# The SubOrgan labels GRANAP gives an anatomy (a reduction's where= names them; a sheath's elements are StemElements)
REDUCIBLE_LABELS = ("RootSegment", "RootApex", "StemElement", "LeafElement", "Seed")


def check_reduced_labels(mtg, reduction) -> None:
    """
    Warns, once per filter and population (the MPG *mtg*'s registry, mtg.objects), when a plant model's reduction (solve.reduction: its where=, a filter of the SubOrgans)
    names a SubOrgan label that GRANAP gives no anatomy (REDUCIBLE_LABELS: RootSegment, RootApex, StemElement,
    LeafElement, Seed; a sheath's elements are StemElements of a Sheath organ, "Organ.label"): nothing is reduced
    for it.
    """
    import warnings
    labels = tuple(reduction.labels())
    checked = mtg.objects("granap_checked_reductions", saved=False)
    if not labels or labels in checked:
        return
    checked.add(labels, True)
    unknown = [label for label in labels if label not in REDUCIBLE_LABELS]
    if unknown:
        warnings.warn(f"the reduction's labels {unknown} are no SubOrgan label with a GRANAP anatomy "
                      f"({', '.join(REDUCIBLE_LABELS)}; a sheath's elements: StemElement): nothing is reduced for "
                      "them", stacklevel=3)


def _basal_with_anatomy(mtg, elements, anatomy_class):
    """The basal element of an organ's *elements* among those with an anatomy (*anatomy_class*'s): none of its
    ancestors within the organ has one (an element without one, e.g. not grown yet, holds no Compartment)."""
    inside = set(elements)
    with_anatomy = [e for e in elements if e in anatomy_class] or elements
    owned = set(with_anatomy)
    for element in with_anatomy:
        parent = mtg.parent(element)
        while parent in inside and parent not in owned:
            parent = mtg.parent(parent)
        if parent not in inside:
            return element
    return with_anatomy[0]


def distance_from_junction(mtg, segment, length="length") -> float:
    """The length of the segments from *segment* up to the root-shoot junction (its own length excluded)."""
    lengths, labels, total = mtg.view(length), mtg.view("label"), 0.          # read vertex by vertex: the columns
    parent = mtg.parent(segment)
    while parent is not None and mtg.scale(parent) == mtg.scales.SubOrgan and \
            labels.get(parent) == mtg.labels.SubOrgan.RootSegment:
        total += float(lengths.get(parent, 0.))
        parent = mtg.parent(parent)
    return total


def _rectangle_sides(rectangle):
    """The two side lengths of a (rotated) rectangle polygon (mm)."""
    xy = np.asarray(rectangle.exterior.coords)[:3]
    return float(np.hypot(*(xy[1] - xy[0]))), float(np.hypot(*(xy[2] - xy[1])))


def remove_anatomy(mtg, segment) -> None:
    """Remove *segment*'s anatomy (its Layers with their Cells and protoplasts, its anchored walls, junctions and
    anatomy Connections), keeping the segment."""
    s = mtg.scales
    mtg.remove_instance(segment)                       # an instanced anatomy (none for a copy)
    anchored = mtg.anchored(segment)
    mtg.remove_connections([c for c in anchored if mtg.scale(c) == s.Connection])
    mtg.remove_connections([c for c in anchored if mtg.scale(c) == s.Compartment])
    mtg.remove_connections([c for c in mtg.components(segment) if mtg.scale(c) == s.Layer])


def _protoplasts(mtg, segment) -> list:
    """The Compartments decomposing *segment* (its Cells' protoplasts), in mtg.components_at_scale's order: its Layers',
    then their Cells', then each Cell's protoplasts, in creation order (for an instance, from its template)."""
    block = mtg.instances.block(segment) if mtg.has_instances() else None
    if block is None:
        return mtg.components_at_scale(segment, scale=mtg.scales.Compartment)
    arrays = mtg.topology_arrays()
    vids = np.arange(block[0], block[0] + block[1], dtype=np.int64)
    scale, complex_ = arrays["scale"], arrays["complex"]
    protoplasts = vids[(scale[vids] == mtg.scales.Compartment) & (complex_[vids] >= block[0])]
    cells = complex_[protoplasts]
    order = np.lexsort((protoplasts, cells, complex_[cells]))
    return protoplasts[order].tolist()


def _instance(mtg, comps):
    """(template, template node of each of *comps*) when *comps* are exactly one copy of a template, else None."""
    vids = np.asarray(comps, dtype=np.int64)
    if vids.size == 0:
        return None
    order = np.argsort(vids, kind="stable")
    found, templates = mtg.values_at("template", vids[order])
    found_nodes, nodes = mtg.values_at("template_node", vids[order])
    if found.size != vids.size or found_nodes.size != vids.size or not (templates == templates[0]).all():
        return None
    out = np.empty(vids.size, dtype=np.int64)
    out[order] = np.asarray(nodes, dtype=np.int64)
    if not np.array_equal(np.sort(out), np.arange(vids.size)):
        return None
    return int(templates[0]), out


TABLE_COLUMNS = ("template", "template_node")     # _compartment_table's: set when a Compartment is created


def _compartment_table(mtg) -> dict:
    """{"vids": every Compartment owned by a SubOrgan (ascending), name: (present, values: NaN where none)} of its
    template and template node (written when it is created, never after), for the wiring of the junctions
    (vessel_pairs): a structure-derived value of the MPG (MPG._derived), patched from its delta, the new
    Compartments read in bulk."""
    def read(vids):
        out = [vids]
        for name in TABLE_COLUMNS:
            out.extend(_present_values(mtg, name, vids))
        return tuple(out)

    def build(arrays):
        return read(np.asarray(mtg.owned_compartments(mtg.scales.SubOrgan), dtype=np.int64))

    def patch(value, arrays, removed, added):
        owned = np.asarray(mtg.owned_compartments(mtg.scales.SubOrgan), dtype=np.int64)
        new = read(owned[owned >= added[0]] if added.size else owned[:0])
        kept = ~np.isin(value[0], removed, assume_unique=True)
        return tuple(np.concatenate((old[kept], fresh)) for old, fresh in zip(value, new))

    value = mtg._derived("granap_compartments", build, patch)
    table = {"vids": value[0]}
    for k, name in enumerate(TABLE_COLUMNS):
        table[name] = (value[1 + 2 * k], value[2 + 2 * k])
    return table


def _table_values(mtg, name, vids) -> tuple:
    """(which of *vids* have a value of *name*, the values: NaN where none) from _compartment_table;
    the vertices it does not hold (not owned by a SubOrgan) read from the MPG."""
    table = _compartment_table(mtg)
    vids = np.asarray(vids, dtype=np.int64)
    keys = table["vids"]
    position = np.searchsorted(keys, vids).clip(0, max(keys.size - 1, 0))
    held = (keys[position] == vids) if keys.size else np.zeros(vids.size, dtype=bool)
    present, values = np.zeros(vids.size, dtype=bool), np.full(vids.size, np.nan)
    if keys.size:
        present[held], values[held] = table[name][0][position[held]], table[name][1][position[held]]
    if not held.all():
        rest = np.unique(vids[~held])
        their_present, their_values = _present_values(mtg, name, rest)
        at = np.searchsorted(rest, vids[~held])
        present[~held], values[~held] = their_present[at], their_values[at]
    return present, values


def _present_values(mtg, name, vids) -> tuple:
    """(which of *vids* (ascending) have property *name*, the values: NaN where none), in one bulk read."""
    present, values = np.zeros(vids.size, dtype=bool), np.full(vids.size, np.nan)
    found, found_values = mtg.values_at(name, vids)
    if found.size:
        at = np.searchsorted(vids, found)
        present[at], values[at] = True, np.asarray(found_values, dtype=np.float64)
    return present, values


def _positions(mtg, vids):
    """(angle and radius present, x, y) of *vids*, computed value by value as the original rule does."""
    angle, radius = mtg.view("angle"), mtg.view("relative_radius")
    present = np.array([v in angle for v in vids], dtype=bool)
    xy = np.zeros((len(vids), 2))
    for i, v in enumerate(vids):
        if present[i]:
            xy[i] = (radius[v] * np.cos(angle[v]), radius[v] * np.sin(angle[v]))
    return present, xy


def granap_inheritance(data_structure, new_vids, is_known, fresh=()) -> np.ndarray:
    """
    The source of each new Compartment (Q-G3): the Compartment of the same template node in the nearest upstream
    section that has it; else the known protoplast of the same tissue nearest in (angle, relative radius) in the
    nearest upstream section having that tissue; else none (-1, the default). Sections in *fresh* (replaced
    anatomies, Q-G2) inherit nothing.

    Sections are template copies: the rule is applied per section and per upstream section, on their template nodes
    (a section that is not a whole copy, or an upstream section partly known, falls back to the rule vertex by
    vertex, _inheritance_by_vertex). The same sources.
    """
    mtg = data_structure.mtg
    from_scale = data_structure.from_scale
    new_vids = np.asarray(new_vids, dtype=np.int64)
    sources = np.full(new_vids.size, -1, dtype=np.int64)
    sections = data_structure.sections_of(new_vids)
    owners = mtg.compartments_csr(from_scale)
    tissue = mtg.view("tissue")
    linked = mtg.linked_parents(from_scale)
    upstream_cache, known_sections = {}, {}

    def upstream(section):
        if section not in upstream_cache:
            chain, vertex = [], linked.get(int(section))
            while vertex is not None:
                chain.append(int(vertex))
                vertex = linked.get(vertex)
            upstream_cache[section] = chain
        return upstream_cache[section]

    def known_section(section):
        """("all", copy) when every Compartment is known and they form one copy; "none"; or "partial"."""
        if section not in known_sections:
            comps = owners.compartments_of(section).tolist()
            known = np.asarray(is_known(np.asarray(comps, dtype=np.int64)), dtype=bool) if comps else \
                np.zeros(0, dtype=bool)
            if not known.any():
                known_sections[section] = ("none", None)
            elif known.all() and _instance(mtg, comps) is not None:
                known_sections[section] = ("all", (comps, _instance(mtg, comps)))
            else:
                known_sections[section] = ("partial", None)
        return known_sections[section]

    slow = []
    for section in np.unique(sections).tolist():
        rows = np.flatnonzero(sections == section)
        if section < 0 or section in fresh:
            continue
        here = _instance(mtg, new_vids[rows].tolist())
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
        present, xy = _positions(mtg, vids)
        pending = np.array([t is not None for t in own_tissue], dtype=bool)
        for comps, _ in ancestors:                               # same tissue, nearest relative position
            if not pending.any():
                break
            their_present, their_xy = _positions(mtg, comps)
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
        sources[rows] = _inheritance_by_vertex(data_structure, new_vids[rows], is_known, fresh)
    return sources


class GranapInheritance:
    """The DataStructure's inheritance rule (data_structure.inheritance) for GRANAP's anatomies: granap_inheritance, the sections
    of *fresh* (the anatomies the last update replaced) inheriting nothing. An object, not a closure: the Scene's
    checkpoints keep it."""

    def __init__(self, fresh=()):
        self.fresh = set(fresh)

    def __call__(self, data_structure, new_vids, is_known) -> np.ndarray:
        return granap_inheritance(data_structure, new_vids, is_known, self.fresh)


def _inheritance_by_vertex(data_structure, new_vids, is_known, fresh=()) -> np.ndarray:
    """granap_inheritance, Compartment by Compartment (for sections that are not whole template copies)."""
    mtg = data_structure.mtg
    from_scale = data_structure.from_scale
    new_vids = np.asarray(new_vids, dtype=np.int64)
    sources = np.full(new_vids.size, -1, dtype=np.int64)
    sections = data_structure.sections_of(new_vids)
    owners = mtg.compartments_csr(from_scale)
    template, node = mtg.view("template"), mtg.view("template_node")
    tissue, angle, radius = mtg.view("tissue"), mtg.view("angle"), mtg.view("relative_radius")
    linked = mtg.linked_parents(from_scale)
    upstream_cache, node_maps = {}, {}

    def upstream(section):
        if section not in upstream_cache:
            chain, vertex = [], linked.get(int(section))
            while vertex is not None:
                chain.append(int(vertex))
                vertex = linked.get(vertex)
            upstream_cache[section] = chain
        return upstream_cache[section]

    def node_map(section):
        if section not in node_maps:
            comps = owners.compartments_of(section)
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
      root_apex_anatomy
                     root apices (SubOrgan label RootApex: an axis' tip, ArchiSimple's), on top of root_anatomy
                     (default: the root segments' anatomy, of their class; always generated, as the segments', once
                     the apex has a length); its meristem is not represented, its maturity is its barriers' (MECHA's
                     Development, by its distance from the tip)
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

    The named entity sets it publishes (entity_sets: assemblies select by these names, not by the codes), on the
    plant's DataStructure (define_mask(entities="GranapAnatomy.soil_nodes")) and on the section templates (a
    reduction's ports and keep): soil_nodes (each root section's soil node, its one soil boundary), conduits (the
    protoplasts of families 0 and 1, which the junctions join: a reduction's ports), leaf_surfaces (a leaf organ's
    surface node).
    """
    entity_sets = {"soil_nodes": {"compartment_type": float(SOIL_NODE)}, "conduits": {"family": [0., 1.]},
                   "leaf_surfaces": {"compartment_type": float(LEAF_SURFACE)}}

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
    # where the architecture draws a segment when not at its topological place (metafspm's display ends: the
    # sections are drawn there; NaN, or not given: at x1 ... z2)
    display_x1: float = input_variable(unit="m", description="where the segment's start is drawn, x (display only)",
                                       references="metafspm", by="Architecture", scale=SCALES.SubOrgan, default=np.nan,
                                       kind="descriptor", optional=True)
    display_y1: float = input_variable(unit="m", description="where the segment's start is drawn, y (display only)",
                                       references="metafspm", by="Architecture", scale=SCALES.SubOrgan, default=np.nan,
                                       kind="descriptor", optional=True)
    display_z1: float = input_variable(unit="m", description="where the segment's start is drawn, z (display only)",
                                       references="metafspm", by="Architecture", scale=SCALES.SubOrgan, default=np.nan,
                                       kind="descriptor", optional=True)
    display_x2: float = input_variable(unit="m", description="where the segment's end is drawn, x (display only)",
                                       references="metafspm", by="Architecture", scale=SCALES.SubOrgan, default=np.nan,
                                       kind="descriptor", optional=True)
    display_y2: float = input_variable(unit="m", description="where the segment's end is drawn, y (display only)",
                                       references="metafspm", by="Architecture", scale=SCALES.SubOrgan, default=np.nan,
                                       kind="descriptor", optional=True)
    display_z2: float = input_variable(unit="m", description="where the segment's end is drawn, z (display only)",
                                       references="metafspm", by="Architecture", scale=SCALES.SubOrgan, default=np.nan,
                                       kind="descriptor", optional=True)
    # ── Inputs: the plants' positions (the planting's), where display_triangles draws the sections ──
    x: float = planting_variable("x")
    y: float = planting_variable("y")
    z: float = planting_variable("z")

    # ── Parameters: each plant's anatomies (its scenario's) ──
    root_anatomy: object = parameter(unit="", description="the plant's root anatomy: {'preset', 'seed', section: "
                                     "{field: value}} on top of its preset (None: the library's)", default=None,
                                     dtype="object", scale=SCALES.Plant, references="GRANAP",
                                     sections=lambda: section_fields("root_anatomy"))
    root_apex_anatomy: object = parameter(unit="", description="the plant's root apex anatomy (an axis' tip, a "
                                          "RootApex), as root_anatomy and on top of it (None: the root segments' "
                                          "anatomy, root_anatomy's)", default=None, dtype="object",
                                          scale=SCALES.Plant, references="GRANAP",
                                          sections=lambda: section_fields("root_apex_anatomy"))
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
    soil_node: bool = parameter(unit="", description="True: each root section's anatomy (a root segment's, a root "
                                "apex's) has one soil node (kind 4), "
                                "joined to its border walls and junctions (network_path 7): the section's one soil "
                                "boundary", default=False, scale=SCALES.Plant, references="GRANAP")
    instanced_anatomy: bool = parameter(unit="", description="True: each segment's anatomy an instance of its template "
                                        "(mtg.add_instance), no vertex of its own; False: a copy (the reference)",
                                        default=False, scale=SCALES.Plant, references="GRANAP")

    # ── State variables: the anatomies ──
    anatomy_template: float = state_variable(unit="", description="the segment's anatomy template (-1: none)",
                                             references="GRANAP", default=-1., scale=SCALES.SubOrgan,
                                             kind="descriptor")
    tissue: object = state_variable(unit="", description="the protoplast's tissue (None: a wall or a junction)",
                                    references="GRANAP", default=None, scale=SCALES.Compartment, location="node",
                                    dtype=object, kind="descriptor")
    compartment_type: float = state_variable(unit="", description="0 protoplast, 1 wall, 2 wall junction, 3 a leaf organ's surface",
                                 references="GRANAP", default=0., scale=SCALES.Compartment, location="node",
                                 kind="descriptor")
    pore_width: float = state_variable(unit="mm",
                                       description="a stomatal pore's aperture width in the section (NaN: not a pore)",
                                       references="GRANAP", default=np.nan, scale=SCALES.Compartment,
                                       location="node", kind="descriptor")
    pore_depth: float = state_variable(unit="mm",
                                       description="a stomatal pore's depth through the epidermis (NaN: not a pore)",
                                       references="GRANAP", default=np.nan, scale=SCALES.Compartment,
                                       location="node", kind="descriptor")
    cgroup: float = state_variable(unit="", description="the protoplast's MECHA cell group (NaN: apoplast)",
                                   references="GRANAP", default=np.nan, scale=SCALES.Compartment, location="node",
                                   kind="descriptor")
    section_x: float = state_variable(unit="mm", description="position in the section", references="GRANAP",
                                      default=np.nan, scale=SCALES.Compartment, location="node",
                                      kind="descriptor")
    section_y: float = state_variable(unit="mm", description="position in the section", references="GRANAP",
                                      default=np.nan, scale=SCALES.Compartment, location="node",
                                      kind="descriptor")
    wall_length: float = state_variable(unit="mm", description="the wall's length in the section (0: not a wall)",
                                        references="GRANAP", default=0., scale=SCALES.Compartment, location="node",
                                        kind="non_inertial_extensive")
    wall_thickness: float = state_variable(unit="µm", description="the wall's thickness: its cells' wall layers (by "
                                                                  "tissue: the anatomy's cell_wall_thickness section; "
                                                                  "NaN: not a wall)", references="GRANAP",
                                           default=np.nan,
                                           scale=SCALES.Compartment, location="node",
                                           kind="descriptor")
    cell_area: float = state_variable(unit="mm2", description="the protoplast's area in the section (0: apoplast)",
                                      references="GRANAP", default=0., scale=SCALES.Compartment, location="node",
                                      kind="non_inertial_extensive")
    wall_surface: float = state_variable(unit="mm2", description="the wall's surface: its length times the segment's",
                                         references="GRANAP", default=0., scale=SCALES.Compartment, location="node",
                                         kind="non_inertial_extensive")
    cell_volume: float = state_variable(unit="mm3",
                                        description="the protoplast's volume: its area times the segment's length",
                                        references="GRANAP", default=0., scale=SCALES.Compartment, location="node",
                                        kind="non_inertial_extensive")
    network_path: float = state_variable(unit="", description="edge type: membrane 0, wall_air 1, wall 2, "
                                                              "plasmodesmata 3, air_link 4, axial junction 5, "
                                                              "stomatal 6", references="GRANAP", default=np.nan,
                                         scale=SCALES.Connection, location="edge", kind="descriptor")
    edge_length: float = state_variable(unit="mm", description="the edge's length (the wall's for membranes; between "
                                                               "the sections for axial junctions)", references="GRANAP",
                                        default=np.nan, scale=SCALES.Connection, location="edge",
                                        kind="non_inertial_intensive")
    edge_dx: float = state_variable(unit="mm", description="the edge's vector in the section, x", references="GRANAP",
                                    default=np.nan, scale=SCALES.Connection, location="edge",
                                    kind="non_inertial_intensive")
    edge_dy: float = state_variable(unit="mm", description="the edge's vector in the section, y", references="GRANAP",
                                    default=np.nan, scale=SCALES.Connection, location="edge",
                                    kind="non_inertial_intensive")
    edge_dist: float = state_variable(unit="mm", description="the distance between the edge's nodes (NaN: wall edges)",
                                      references="GRANAP", default=np.nan, scale=SCALES.Connection, location="edge",
                                      kind="non_inertial_intensive")
    edge_wall_thickness: float = state_variable(unit="µm",
                                                description="the wall's thickness on membrane and wall edges",
                                                references="GRANAP", default=np.nan, scale=SCALES.Connection,
                                                location="edge", kind="non_inertial_intensive")
    lateral_distance: float = state_variable(unit="mm", description="wall edges: the lateral distance",
                                             references="GRANAP", default=np.nan, scale=SCALES.Connection,
                                             location="edge", kind="non_inertial_intensive")
    distnode_wall_cell: float = state_variable(unit="mm", description="wall edges: the mean wall-cell distance",
                                               references="GRANAP", default=np.nan, scale=SCALES.Connection,
                                               location="edge", kind="non_inertial_intensive")
    membrane_surface: float = state_variable(unit="mm2", description="membrane edges: the membrane's surface (the "
                                                                     "wall's length times the segment's); 0 otherwise",
                                             references="GRANAP", default=0., scale=SCALES.Connection,
                                             location="edge", kind="non_inertial_extensive")

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
    def _label_name(cls, mtg, segment) -> str:
        """The segment's kind: its SubOrgan label's name, or its organ's for the kinds named by their organ (Sheath)."""
        labels = mtg.view("label")                               # one segment: read vertex by vertex (the column)
        code = labels.get(segment)
        name = next((name for name in ANATOMY_KEYS if getattr(mtg.labels.SubOrgan, name, None) == code), None)
        organ = None
        for organ_name, element in ORGAN_KINDS.items():
            if name == element:
                organ = int(mtg.complex_at_scale(segment, mtg.scales.Organ)) if organ is None else organ
                if labels.get(organ) == getattr(mtg.labels.Organ, organ_name):
                    return organ_name
        return name

    @classmethod
    def _label_names(cls, mtg, segments) -> list:
        """_label_name of each of *segments*, from one read of their labels and one of their organs'."""
        segments = np.asarray(segments, dtype=np.int64)
        if segments.size == 0:
            return []
        name_of = {}
        for name in reversed(ANATOMY_KEYS):                  # the first name of a code, as next() over ANATOMY_KEYS
            name_of[getattr(mtg.labels.SubOrgan, name, None)] = name
        names = [name_of.get(code) for code in _aligned(mtg, "label", segments).tolist()]
        organ_kinds = [k for k, name in enumerate(names) if name in ORGAN_KINDS.values()]
        if organ_kinds:
            organs = np.asarray(mtg.complex_at_scale_array(segments[organ_kinds], mtg.scales.Organ), dtype=np.int64)
            organ_labels = _aligned(mtg, "label", organs).tolist()
            for k, organ_label in zip(organ_kinds, organ_labels):
                for organ_name, element in ORGAN_KINDS.items():
                    if names[k] == element and organ_label == getattr(mtg.labels.Organ, organ_name):
                        names[k] = organ_name
                        break
        return names

    @classmethod
    def anatomy_settings(cls, mtg, segment) -> dict:
        """The segment's plant's scenario dict for the segment's kind (root_anatomy...), {} if not given; a kind of
        KIND_DEFAULTS (a root apex) on top of its default kind's (root_anatomy)."""
        return cls._settings(mtg, mtg.complex_at_scale(segment, mtg.scales.Plant), cls._label_name(mtg, segment))

    @classmethod
    def _settings(cls, mtg, plant, name) -> dict:
        """anatomy_settings of a segment of kind *name* in *plant*."""
        settings = {}
        for kind in (KIND_DEFAULTS.get(name), name):
            key = ANATOMY_KEYS.get(kind)
            if key is not None:
                settings = _on_top(settings, cls._plant_value(mtg, plant, key) or {})
        return settings

    @classmethod
    def _plant_value(cls, mtg, plant, name):
        """The plant's value of the declared Plant parameter *name* (its MPG property), else its declared default."""
        value = mtg.properties().get(name, {}).get(plant)
        return getattr(cls, name) if value is None else value

    @classmethod
    def handled_labels(cls, mtg, plant) -> tuple:
        """The SubOrgan labels the plant's anatomies are generated for: roots, the explicit shoot, the kinds its
        scenario gives."""
        handled = {name for name, key in ANATOMY_KEYS.items()
                   if name == "RootSegment" or name in cls.shoot or cls._plant_value(mtg, plant, key) is not None}
        handled |= {name for name, default in KIND_DEFAULTS.items() if default in handled}     # a root apex
        return tuple(name for name in ANATOMY_KEYS if name in handled)

    @classmethod
    def organ_of(cls, mtg, segment) -> OrganAnatomy:
        """The segment's library and class inputs: the explicit ones (root, shoot), else its scenario's preset. A kind
        of KIND_DEFAULTS (a root apex) without an explicit one of its own takes its default kind's (the root's)."""
        return cls._organ(mtg, mtg.complex_at_scale(segment, mtg.scales.Plant), cls._label_name(mtg, segment))

    @classmethod
    def _organ(cls, mtg, plant, name) -> OrganAnatomy:
        """organ_of a segment of kind *name* in *plant*."""
        base = name if name in cls.shoot else KIND_DEFAULTS.get(name, name)
        explicit = cls.shoot.get(name) if name in cls.shoot else cls.root if base == "RootSegment" else \
            cls.shoot.get(base)
        settings = cls._settings(mtg, plant, name)
        if explicit is not None:
            if "preset" in settings:
                raise ValueError(f"{cls.__name__}: {ANATOMY_KEYS[name]} gives a preset, but {name} has an explicit "
                                 "OrganAnatomy (root / shoot): give its sections and fields only")
            return explicit
        key = ANATOMY_KEYS[base]
        if base == "RootSegment" and not set(RESERVED) & set(settings):
            return OrganAnatomy(cls.library, "radius")
        return OrganAnatomy(population_preset_library(mtg, settings.get("preset", DEFAULT_PRESETS[key]),
                                                      settings.get("seed", 0)),
                            SIZES[key])

    @classmethod
    def has_soil_node(cls, mtg, segment) -> bool:
        """Whether the segment is a root section (a root segment or apex, ROOT_KINDS) whose plant has soil nodes (its
        scenario's soil_node)."""
        if mtg.view("label").get(segment) not in _label_codes(mtg, ROOT_KINDS):
            return False
        plant = mtg.complex_at_scale(segment, mtg.scales.Plant)
        return bool(cls._plant_value(mtg, plant, "soil_node"))

    @classmethod
    def add_soil_node(cls, mtg, segment, written) -> tuple:
        """
        The section's soil node (kind 4, tissue "soil", anchored to the segment as its walls) and its Connections
        (network_path 7, from the soil node) to every border candidate of the section *written*: the walls with a
        single membrane (the outer walls, MECHA's border walls among them) and the junctions on them. Returns (the
        soil node, its Connections); *written*'s apoplast and connections extended with them.
        """
        s, L = mtg.scales, mtg.labels
        nodes = np.asarray(sorted(written["apoplast"]), dtype=np.int64)
        edges = np.asarray(sorted(written["connections"]), dtype=np.int64)
        kind = _values_at(mtg, "compartment_type", nodes, np.nan)
        path = _values_at(mtg, "network_path", edges, np.nan)
        a = _values_at(mtg, "n_id_a", edges, -1.).astype(np.int64)
        b = _values_at(mtg, "n_id_b", edges, -1.).astype(np.int64)
        walls, junctions = nodes[kind == WALL], nodes[kind == JUNCTION]
        membrane = path == columns.PATH_CODES["membrane"]
        wall_end = np.where(np.isin(a, walls), a, b)
        count = dict(zip(*np.unique(wall_end[membrane & (np.isin(a, walls) | np.isin(b, walls))], return_counts=True)))
        outer = np.array([w for w in walls.tolist() if count.get(w, 0) == 1], dtype=np.int64)
        on_wall = path == columns.PATH_CODES["wall"]
        touching = (np.isin(a, outer) & np.isin(b, junctions)) | (np.isin(b, outer) & np.isin(a, junctions))
        on_outer = np.unique(np.where(np.isin(a, junctions), a, b)[on_wall & touching])
        candidates = np.union1d(outer, on_outer).tolist()
        soil = mtg.add_components_bulk(s.anchors[s.Compartment], 1, topo_parents=[segment], **_build_props(
            s.Compartment, label=L.Compartment.Apoplastic, compartment_type=float(SOIL_NODE)))[0]
        mtg.set_values("tissue", [soil], ["soil"])
        # its place in the section's template, after the template's nodes: the section stays one copy of it (the
        # vessel pairs cached per template, the inheritance by template node)
        mtg.set_values("template", [soil], [float(written["template"])])
        mtg.set_values("template_node", [soil], [float(len(written["protoplasts"]) + len(written["apoplast"]))])
        for name in columns.NODE_COLUMNS + ("pore_width", "pore_depth"):
            mtg.set_values(name, [soil], [0. if name in ("wall_length", "cell_area") else np.nan])
        for name, value in cls._declared_initial("node").items():
            if soil not in mtg.view(name):
                mtg.set_values(name, [soil], [value])
        n = len(candidates)
        links = mtg.add_components_bulk(s.anchors[s.Connection], n, topo_parents=[segment] * n, **_build_props(
            s.Connection, label=L.Connection.Apoplastic, n_id_a=[int(soil)] * n, n_id_b=candidates))
        values = {name: [np.nan] * n for name in columns.EDGE_COLUMNS}
        values.update(network_path=[float(columns.SOIL_PATH)] * n, membrane_surface=[0.] * n)
        for name, column in values.items():
            mtg.set_values(name, links, column)
        for name, value in cls._declared_initial("edge").items():
            if name not in values:
                mtg.set_values(name, links, [value] * n)
        written["apoplast"] = list(written["apoplast"]) + [int(soil)]
        written["connections"] = list(written["connections"]) + list(links)
        return soil, links

    @classmethod
    def is_instanced(cls, mtg, segment) -> bool:
        """Whether the segment's plant has instanced anatomies (its scenario's instanced_anatomy)."""
        plant = mtg.complex_at_scale(segment, mtg.scales.Plant)
        return bool(cls._plant_value(mtg, plant, "instanced_anatomy"))

    @classmethod
    def segment_class(cls, mtg, segment) -> tuple:
        """The segment's anatomy class: its organ's class inputs, binned."""
        organ = cls.organ_of(mtg, segment)
        cls.check_class_inputs(organ)
        return class_key(organ.class_inputs, mtg, segment)

    @classmethod
    def segment_classes(cls, mtg, segments, names=None) -> list:
        """segment_class of each of *segments* (*names*: their kinds, _label_names' by default): each kind's organ
        found once per plant, each class input read in one bulk read."""
        segments = np.asarray(segments, dtype=np.int64)
        names = cls._label_names(mtg, segments) if names is None else list(names)
        plants = np.asarray(mtg.complex_at_scale_array(segments, mtg.scales.Plant), dtype=np.int64).tolist() \
            if segments.size else []
        groups = {}
        for row, group in enumerate(zip(plants, names)):
            groups.setdefault(group, []).append(row)
        keys = [None] * segments.size
        for (plant, name), rows in groups.items():
            organ = cls._organ(mtg, plant, name)
            cls.check_class_inputs(organ)
            for row, key in zip(rows, class_keys(organ.class_inputs, mtg, segments[rows])):
                keys[row] = key
        return keys

    @classmethod
    def check_class_inputs(cls, organ) -> None:
        """A class input reads a variable the component declares (an input the plant model's translator links, e.g.
        stem_radius from an architecture's organ diameter), never another model's property by name."""
        declared = cls.__dict__.get("_declared_inputs")
        if declared is None:                                    # read once per class (fields() at every segment cost
            declared = frozenset(f.name for f in fields(cls)    # a second a step at day 14)
                                 if f.metadata.get("role") in ("input_variable", "parameter"))
            setattr(cls, "_declared_inputs", declared)
        for c in organ.class_inputs:
            if c.compute is None and c.width > 0 and c.variable not in declared:
                raise ValueError(f"{cls.__name__}: the class input '{c.variable}' is not a declared input: declare it "
                                 f"(input_variable(..., by=\"Architecture\", scale=SubOrgan)) and link it in the "
                                 "plant model's translator")

    @classmethod
    def initiate_plant(cls, mtg, plant_vid: int, parameters: dict) -> None:
        """Each segment of the plant gets its class's anatomy, parametrized by the plant's scenario (root_anatomy,
        stem_anatomy, leaf_anatomy), kept on the plant for the anatomies generated later."""
        parameters = parameters or {}
        for key in tuple(ANATOMY_KEYS.values()) + ("soil_node", "instanced_anatomy"):   # its declared Plant parameters
            if key in parameters:
                value = parameters[key]
                mtg.set_values(key, [plant_vid], [dict(value) if isinstance(value, dict) else value])
        segments = _segments(mtg, plant_vid, cls.handled_labels(mtg, plant_vid), cls._label_name)
        cls.update_anatomies(mtg, segments)
        cls.update_leaf_surfaces(mtg, segments)

    @classmethod
    def update_anatomies(cls, mtg, segments, lengths=None) -> dict:
        """
        Give each segment the anatomy of its current class (Q12): a segment without one (just emerged) gets its
        class's copy; a segment whose class changed has its anatomy replaced. Returns {"emerged", "replaced"}.
        A segment without length (an Adel internode or sheath before it grows: 0, or none given) gets none until it
        has one: it is a pass-through meanwhile, as a hidden element, the junctions spanning it (a section of zero
        height would have no radial conductance, its nodes cut off from the network). *lengths*: {segment: length},
        by default the MPG's property. A segment whose kind changed (anatomy_kind: a root apex the architecture
        relabels a segment as it cuts it off its tip, ArchiSimple's) keeps its anatomy when its new kind gives it the
        same (the same template, reduced or not as before), else has it replaced.
        """
        anatomy_class = mtg.properties().setdefault("anatomy_class", {})
        template_of = mtg.properties().setdefault("anatomy_template", {})
        kind_of = mtg.properties().setdefault("anatomy_kind", {})
        lengths = mtg.view("length") if lengths is None else lengths
        reduction = mtg.reduction                                 # the plant model's (solve.reduction)
        if reduction is not None:
            check_reduced_labels(mtg, reduction)
        done = {"emerged": [], "replaced": []}
        segments = [int(segment) for segment in segments]
        names = cls._label_names(mtg, segments)                  # the segments' kinds, classes and records in bulk
        currents, kinds = _aligned(mtg, "anatomy_class", segments), _aligned(mtg, "anatomy_kind", segments)
        rows = [i for i, (segment, current) in enumerate(zip(segments, currents.tolist()))
                if current is not None or float(lengths.get(segment, 0.) or 0.) > 0.]
        keys = cls.segment_classes(mtg, [segments[i] for i in rows], [names[i] for i in rows])
        copies = []                                            # the new anatomies, their 3D quantities written at once
        for i, key in zip(rows, keys):
            segment, name, current, recorded = segments[i], names[i], currents[i], kinds[i]
            previous = name if recorded is None else recorded
            if current == key and (previous == name or cls._same_anatomy(mtg, segment, key, previous, name)):
                if recorded != name:
                    kind_of[segment] = name
                continue
            if current is not None:
                remove_anatomy(mtg, segment)
                done["replaced"].append(segment)
            else:
                done["emerged"].append(segment)
            template = cls._template(mtg, segment, key)
            instanced, soil = cls.is_instanced(mtg, segment), cls.has_soil_node(mtg, segment)
            if reduction is not None and reduction.reduces(mtg, segment):
                instanced = True
                full = template.section_template(cls.add_soil_node if soil else None, "soil" if soil else "")
                written = template.instance_of(mtg, segment, reduction.template_for(full))
            elif instanced:
                written = template.instance(mtg, segment, cls.add_soil_node if soil else None, "soil" if soil else "")
            else:
                written = template.instantiate(mtg, segment)
                if soil:
                    cls.add_soil_node(mtg, segment, written)
            copies.append((segment, written))
            if not instanced:                           # the pores' (leaf surfaces); an instance has its template's
                nodes = list(written["protoplasts"]) + list(written["apoplast"])
                mtg.set_values("pore_width", nodes, [np.nan] * len(nodes))
                mtg.set_values("pore_depth", nodes, [np.nan] * len(nodes))
            anatomy_class[segment] = key
            template_of[segment] = float(template.index)
            kind_of[segment] = name
        if copies:
            length_mm = _aligned(mtg, "length", [segment for segment, _ in copies], 0.).astype(float) * cls.mm_per_unit
            _write_3d(mtg, [written for _, written in copies], length_mm)
        return done

    @classmethod
    def _template(cls, mtg, segment, key):
        """The template of the segment's class *key*: its organ's library's, with its plant's overrides (its kind's
        defaults, DEFAULT_OVERRIDES, under its scenario dict) and its class inputs' targets."""
        organ = cls.organ_of(mtg, segment)
        own = {k: v for k, v in cls.anatomy_settings(mtg, segment).items() if k not in RESERVED}
        plant = merged(DEFAULT_OVERRIDES.get(ANATOMY_KEYS.get(cls._label_name(mtg, segment)), {}), own) or None
        return organ.library.template(mtg, key, plant=plant, targets=class_targets(organ.class_inputs, key))

    @classmethod
    def _same_anatomy(cls, mtg, segment, key, previous, name) -> bool:
        """Whether the segment, of class *key*, its kind changed from *previous* to *name*, would get the anatomy it
        has: the same template, reduced (the plant model's reduction, its where= on the segment as it is now) as its
        instance is or not reduced as it is."""
        if float(mtg.view("anatomy_template").get(segment, -1.)) != float(cls._template(mtg, segment, key).index):
            return False
        reduction = mtg.reduction
        if reduction is None:
            return True
        current = mtg.instances.template_of(segment)
        return reduction.reduces(mtg, segment) == (current is not None and reduction.is_reduced(current))

    @classmethod
    def update_leaf_surfaces(cls, mtg, segments) -> None:
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
        s, L = mtg.scales, mtg.labels
        leaves = [v for v, name in zip(segments, cls._label_names(mtg, segments)) if name in cls.leaf_labels]
        organs = sorted({int(organ) for organ in mtg.complex_at_scale_array(np.asarray(leaves, dtype=np.int64), s.Organ)})
        if not organs:
            return
        surfaces = mtg.properties().setdefault("leaf_surface", {})
        anatomy_class = mtg.properties().get("anatomy_class", {})
        path, tissue = mtg.view("network_path"), mtg.view("tissue")
        template_row = mtg.view("template_row")
        for organ in organs:
            components = [int(v) for v in mtg.components_at_scale(organ, scale=s.SubOrgan)]
            elements = [v for v, name in zip(components, cls._label_names(mtg, components)) if name in cls.leaf_labels]
            stomatal = [c for e in elements for c in mtg.anchored(e)
                        if mtg.scale(c) == s.Connection and path.get(c) == columns.STOMATAL_PATH]
            mtg.remove_connections(stomatal)
            old = surfaces.pop(organ, None)
            if old is not None and mtg.has_vertex(old):
                mtg.remove_connections([old])
            pores = {e: [int(c) for c in _protoplasts(mtg, e) if tissue.get(c) == "pore"] for e in elements}
            if not any(pores.values()):
                continue
            basal = _basal_with_anatomy(mtg, elements, anatomy_class)
            surface = mtg.add_components_bulk(s.anchors[s.Compartment], 1, topo_parents=[basal], **_build_props(
                s.Compartment, label=L.Compartment.Apoplastic, compartment_type=float(LEAF_SURFACE)))[0]
            mtg.set_values("tissue", [surface], ["leaf surface"])
            for name in columns.NODE_COLUMNS + ("pore_width", "pore_depth"):
                mtg.set_values(name, [surface], [0. if name in ("wall_length", "cell_area") else np.nan])
            for name, value in cls._declared_initial("node").items():           # the other declared variables
                if surface not in mtg.view(name):
                    mtg.set_values(name, [surface], [value])
            surfaces[organ] = int(surface)
            template_of = mtg.view("anatomy_template")
            for element, vids in pores.items():
                if not vids:
                    continue
                template = template_by_index(mtg, template_of[element])
                rectangles = [shapely.minimum_rotated_rectangle(template.polygons[int(template_row[v])]) for v in vids]
                sides = np.array([sorted(_rectangle_sides(r)) for r in rectangles])
                mtg.set_values("pore_width", vids, sides[:, 0].tolist())
                mtg.set_values("pore_depth", vids, sides[:, 1].tolist())
                n = len(vids)
                edges = mtg.add_components_bulk(s.anchors[s.Connection], n, topo_parents=[element] * n, **_build_props(
                    s.Connection, label=L.Connection.Apoplastic, n_id_a=vids, n_id_b=[int(surface)] * n))
                values = {name: [np.nan] * n for name in columns.EDGE_COLUMNS}
                values.update(network_path=[float(columns.STOMATAL_PATH)] * n, edge_length=sides[:, 1].tolist(),
                              membrane_surface=[0.] * n)
                for name, column in values.items():
                    mtg.set_values(name, edges, column)
                for name, value in cls._declared_initial("edge").items():
                    if name not in values:
                        mtg.set_values(name, edges, [value] * n)

    @classmethod
    def _declared_initial(cls, location) -> dict:
        """{name: initial value} of the declared state variables at *location* ("node" or "edge")."""
        from dataclasses import fields
        return {f.name: f.metadata.get("initialize", f.default) for f in fields(cls)
                if f.metadata.get("role") == "state_variable" and f.metadata.get("location") == location}

    def __post_init__(self):
        mtg = self.data_structure.mtg
        segments = set(self.data_structure.entity_ids("SubOrgan").tolist())
        template_of = mtg.properties().setdefault("anatomy_template", {})
        for segment in segments - set(template_of):            # segments without an anatomy (labels not handled)
            template_of[segment] = -1.
        super().__post_init__()
        self.data_structure.inheritance = GranapInheritance()

    @postsegmentation
    def _anatomies(self):
        """MPG-style: every segment of the emerged plants gets its class's anatomy (emerged, or replaced on a change);
        the records of the segments pruned in any plant go."""
        mtg = self.mtg
        by_plant = {int(plant): _segments(mtg, plant, self.handled_labels(mtg, plant), self._label_name)
                    for plant in mtg.components_at_scale(mtg.root, scale=mtg.scales.Plant)}
        emerged = set(self.active_ids("Plant").tolist())
        segments = [s for plant, these in by_plant.items() if plant in emerged for s in these]
        alive = np.asarray(sorted(s for these in by_plant.values() for s in these), dtype=np.int64)
        for record in ("anatomy_class", "anatomy_kind"):
            records = mtg.properties().get(record)
            if records:                                                       # pruned segments: their record only
                keys = records.keys_array()
                records.delete_many(keys[~np.isin(keys, alive)])
        done = self.update_anatomies(mtg, segments, _suborgan_lengths(self.data_structure))
        self.update_leaf_surfaces(mtg, done["emerged"] + done["replaced"])
        self._wire_pass_through(done["emerged"] + done["replaced"])
        every = mtg._vertices_at_scale(mtg.scales.SubOrgan)
        missing = every[~np.isin(every, mtg.properties()["anatomy_template"].keys_array())]
        mtg.set_values("anatomy_template", missing, np.full(missing.size, -1.))   # the segments without an anatomy
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
        mtg = self.mtg
        owners = mtg.compartments_csr(mtg.scales.SubOrgan)
        descendants = anatomy_descendants(mtg, sections, owners)
        if descendants:
            heads = set(np.concatenate([owners.compartments_of(v) for v in descendants]).tolist())
            n_id_b = mtg.view("n_id_b")
            mtg.remove_connections([int(e) for e in mtg.junction_vids() if int(n_id_b[e]) in heads])
            mtg.wire_junctions(mtg.scales.SubOrgan, rules, children=descendants)

    def _pass_through_spans(self, network_path) -> tuple:
        """(the junctions across SubOrgans without anatomy, as edge indices; each one's span; the spans' offsets into
        the SubOrgans without anatomy they cross, those SubOrgans, nearest first), the pass-through junctions whose
        length (_wall_surface) spans them; None when there is none. Cached until the topology changes."""
        data_structure = self.data_structure
        key = (data_structure.topology_version, data_structure.n_edges())
        cached = self.__dict__.get("_pass_through_cache")
        if cached is not None and cached[0] == key:
            return cached[1]
        mtg = data_structure.mtg
        owners = mtg.compartments_csr(mtg.scales.SubOrgan)
        spans = None
        if not owners.has_all(data_structure.entity_ids("SubOrgan")):
            edges = data_structure.edge_endpoints()
            rows = np.flatnonzero(np.asarray(network_path) == columns.JUNCTION_PATH)
            parents, children = data_structure.sections_of(edges[rows, 0]), data_structure.sections_of(edges[rows, 1])
            found_rows, span_of, offsets, hidden = [], [], [0], []
            for child in np.unique(children).tolist():
                ancestor, between = anatomy_ancestor(mtg, child, owners)
                if between:
                    here = rows[(children == child) & (parents == ancestor)]
                    found_rows.extend(here.tolist())
                    span_of.extend([len(offsets) - 1] * here.size)
                    hidden.extend(between)
                    offsets.append(len(hidden))
            if found_rows:
                spans = (np.asarray(found_rows, dtype=np.int64), np.asarray(span_of, dtype=np.int64),
                         np.asarray(offsets, dtype=np.int64), np.asarray(hidden, dtype=np.int64))
        self.__dict__["_pass_through_cache"] = (key, spans)
        return spans

    def display_triangles(self, where=None):
        """
        The sections drawn for display (encoding3d frames, viewers), not for light models (triangles() stays the
        architecture's): each segment's section, flat, across the middle of the segment, perpendicular to it; each
        protoplast its cell's polygon, triangulated (coupling.geometry.Triangles, each triangle its node index). A
        segment is placed by its ends (x1 ... z2) when the architecture gives them, else along a straight axis from its
        plant's position (roots down, the shoot up), at its distance from the axis' base. *where*: a node mask.
        Computed when asked (a frame written): each template's polygons are triangulated once.

        A reduced section (the plant model's reduction: its kept nodes only in the DataStructure) is drawn whole: its
        interior's protoplasts too, owned by no node (owner -1) but by their display ids (owner_ids:
        solve.reduction.display_ids), which metafspm's encoding3d frames write as display rows with the reduced
        system's unknown recovered (MECHA's psi); with *where*, when one of its kept nodes is wanted.
        """
        from openalea.metafspm.coupling.geometry import Triangles
        data_structure = self.data_structure
        mtg = data_structure.structure
        properties = mtg.properties()                   # the columns, read as they are
        template_of, template_row = properties.get("anatomy_template", {}), mtg.view("template_row")
        arrays = mtg.topology_arrays()
        scale, complex_ = arrays["scale"], arrays["complex"]
        node_ids = np.asarray(data_structure.entity_ids("node"), dtype=np.int64)
        kept = np.ones(node_ids.size, dtype=bool) if where is None else np.asarray(where, dtype=bool)
        wanted = set(node_ids[kept].tolist())
        wanted_at = np.zeros(scale.size, dtype=bool)          # by vid: a segment with none of them is not drawn
        wanted_at[node_ids[kept]] = True
        plant_position = dict(zip(data_structure.entity_ids("Plant").tolist(),
                                  np.column_stack([np.asarray(data_structure.get(name), dtype=float) for name in ("x", "y", "z")])))
        frame_values = {name: _per_segment(self, name) for name in SEGMENT_ENDS + DISPLAY_ENDS + ("length",)}
        reduction = getattr(mtg, "reduction", None)              # the plant model's: its reduced sections drawn whole
        reduced = ({int(owner): key for owner, key in reduction.reduced_groups(data_structure).values()}
                   if reduction is not None and mtg._instanced() else {})
        vertices, owners, owner_ids = [], [], []
        for segment, compartments in mtg.compartments_csr(mtg.scales.SubOrgan).items():
            index = template_of.get(segment, -1.)
            if index is None or index < 0 or (where is not None and not wanted_at[compartments].any()):
                continue                    # (no compartment wanted: nothing of its own drawn, nor its interior)
            local, rows = _template_triangles(template_by_index(mtg, index))
            row_of = {int(template_row[c]): int(c) for c in compartments.tolist()
                      if int(c) in wanted and c in template_row and scale[int(c)] == mtg.scales.Compartment
                      and complex_[int(c)] >= 0 and scale[complex_[int(c)]] == mtg.scales.Cell}
            drawn = np.array([row in row_of for row in rows], dtype=bool)
            interior = {}
            if segment in reduced and (where is None or row_of):      # its interior's protoplasts, by display id
                from openalea.metafspm.solve.reduction import display_ids
                cell_rows, full_rows = _interior_protoplasts(mtg, reduction, reduced[segment])
                interior = dict(zip(cell_rows.tolist(), display_ids(segment, full_rows).tolist()))
            shown = np.array([row in interior and row not in row_of for row in rows], dtype=bool)
            if not drawn.any() and not shown.any():
                continue
            centre, axis = _segment_frame(mtg, segment, plant_position, frame_values)
            u, v = _perpendiculars(axis)
            for mask, own in ((drawn, True), (shown, False)):
                if not mask.any():
                    continue
                xy = local[mask] / self.mm_per_unit                                # (T, 3, 2), in m
                vertices.append(centre + xy[..., :1] * u + xy[..., 1:] * v)
                if own:
                    vids = [row_of[row] for row in rows[mask]]
                    owners.append(data_structure.index_of(vids, "node"))
                    owner_ids.append(np.asarray(vids, dtype=np.int64))
                else:
                    owners.append(np.full(int(mask.sum()), -1, dtype=np.int64))
                    owner_ids.append(np.array([interior[row] for row in rows[mask]], dtype=np.int64))
        if not vertices:
            return Triangles(np.zeros((0, 3, 3)), np.zeros(0, dtype=np.int64))
        owners = np.concatenate(owners)
        return Triangles(np.concatenate(vertices), owners,
                         np.concatenate(owner_ids) if (owners < 0).any() else None)

    def display_instances(self, owners):
        """
        The reduced sections among the segments *owners* (the plant model's reduction: their kept nodes only in the
        DataStructure) drawn as instances of their full templates, for metafspm's encoding3d frames
        (display="instances", the default): each full template's protoplasts once, its cells' polygons triangulated
        flat in the section's own plane (x, y in m, z 0), each triangle owned by its protoplast's row in the full
        template; each segment's transform placing it as display_triangles draws it (across the middle of the
        segment, perpendicular to it: its columns the section's two axes, the segment's axis and its middle). The
        frame writes the templates and the kinds' recovery operators once (shared parts) and the reduced system's
        unknown (MECHA's psi) on every protoplast as a derived field of the kept nodes' values, rebuilt by the reader
        for the sections selected.
        """
        from openalea.metafspm.coupling.geometry import TemplateShapes
        data_structure = self.data_structure
        mtg = data_structure.structure
        reduction = getattr(mtg, "reduction", None)
        reduced = ({int(owner): key for owner, key in reduction.reduced_groups(data_structure).values()}
                   if reduction is not None and mtg._instanced() else {})
        template_of = mtg.properties().get("anatomy_template", {})
        plant_position = dict(zip(data_structure.entity_ids("Plant").tolist(),
                                  np.column_stack([np.asarray(data_structure.get(name), dtype=float) for name in ("x", "y", "z")])))
        frame_values = {name: _per_segment(self, name) for name in SEGMENT_ENDS + DISPLAY_ENDS + ("length",)}
        drawn, shape_of, transforms, shapes = [], [], [], {}
        for segment in np.asarray(owners, dtype=np.int64).tolist():
            index = template_of.get(segment, -1.)
            if segment not in reduced or index is None or index < 0:
                continue
            full = reduction.full_template(reduced[segment])
            if full.key not in shapes:
                shapes[full.key] = _template_shape(mtg, template_by_index(mtg, index), full, self.mm_per_unit)
            centre, axis = _segment_frame(mtg, segment, plant_position, frame_values)
            u, v = _perpendiculars(axis)
            transform = np.eye(4)
            transform[:3, 0], transform[:3, 1], transform[:3, 2], transform[:3, 3] = u, v, axis, centre
            drawn.append(segment)
            shape_of.append(full.key)
            transforms.append(transform)
        return TemplateShapes(np.asarray(drawn, dtype=np.int64), shapes, shape_of,
                              np.asarray(transforms).reshape(-1, 4, 4))

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
            rows, span_of, offsets, hidden = spans
            vids, lengths = _suborgan_length_arrays(self.data_structure)
            position = np.searchsorted(vids, hidden).clip(0, max(vids.size - 1, 0))
            found = (vids[position] == hidden) if vids.size else np.zeros(hidden.size, dtype=bool)
            spanned = kernels.python_sums(np.where(found, lengths[position] if vids.size else 0., 0.), offsets)
            junction_mm = np.array(edge_mm, dtype=float)
            junction_mm[rows] += spanned[span_of] * self.mm_per_unit
        return {"wall_surface": wall_length * length_mm, "cell_volume": cell_area * length_mm,
                "membrane_surface": np.where(membranes, edge_length * edge_mm, 0.),
                "edge_length": np.where(path == columns.JUNCTION_PATH, junction_mm, edge_length)}


def _interior_protoplasts(mtg, reduction, key) -> tuple:
    """(their cell rows in GRANAP's template, their rows in the full SectionTemplate) of the protoplasts a reduced
    template of key *key* eliminates (the full template's Compartments in a Cell that the reduced one does not hold):
    the interior cells a reduced section's drawing adds; once per reduced template of the population (the MPG *mtg*'s
    reduction, its registry mtg.objects: not checkpointed)."""
    cache = mtg.objects("granap_interior_protoplasts", saved=False)
    if key not in cache:
        from openalea.metafspm.data_structure.configs import ScalesConfig
        full = reduction.full_template(key)
        held = np.zeros(full.size, dtype=bool)
        held[reduction.full_rows(key)] = True
        where, values = full.properties.get("template_row", (np.zeros(0, dtype=np.int64), np.zeros(0)))
        where = np.asarray(where, dtype=np.int64)
        complex_ = full.complex[where]
        in_cell = (complex_ >= 0) & (full.scale[np.maximum(complex_, 0)] == ScalesConfig.Cell)
        keep = (full.scale[where] == ScalesConfig.Compartment) & in_cell & ~held[where]
        cache.add(key, (np.asarray(values, dtype=float)[keep].astype(np.int64), where[keep]))
    return cache[key]


def _template_shape(mtg, template, full, mm_per_unit: float):
    """The shape of a section of GRANAP's *template* whose full SectionTemplate is *full*, in the section's own plane
    (x, y in m, z 0: Triangles), each triangle owned by its protoplast's row in *full* (a Compartment in a Cell, by
    its template_row); once per full template of the population (its registry mtg.objects: not checkpointed)."""
    from openalea.metafspm.coupling.geometry import Triangles
    cache = mtg.objects("granap_display_shapes", saved=False)
    if full.key not in cache:
        local, rows = _template_triangles(template)
        where, values = full.properties.get("template_row", (np.zeros(0, dtype=np.int64), np.zeros(0)))
        where = np.asarray(where, dtype=np.int64)
        complex_ = full.complex[where]
        in_cell = (complex_ >= 0) & (full.scale[np.maximum(complex_, 0)] == ScalesConfig.Cell)
        protoplast = (full.scale[where] == ScalesConfig.Compartment) & in_cell
        row_of = dict(zip(np.asarray(values, dtype=float)[protoplast].astype(np.int64).tolist(),
                          where[protoplast].tolist()))
        shown = np.array([int(row) in row_of for row in rows.tolist()], dtype=bool)
        xy = local[shown] / mm_per_unit
        vertices = np.concatenate([xy, np.zeros(xy.shape[:2] + (1,))], axis=2)
        cache.add(full.key, Triangles(vertices, np.array([row_of[int(row)] for row in rows[shown].tolist()],
                                                          dtype=np.int64)))
    return cache[full.key]


def _template_triangles(template) -> tuple:
    """A template's cells' polygons as triangles in the section's plane (mm), and each triangle's cell row; once per
    template, its polygons being shared by every segment of its class (kept on the template)."""
    if "_cache_triangles" not in template.__dict__:
        import shapely
        triangles, rows = [], []
        for row, polygon in enumerate(template.polygons):
            if polygon is None or polygon.is_empty:
                continue
            for piece in shapely.constrained_delaunay_triangles(polygon).geoms:
                triangles.append(np.asarray(piece.exterior.coords)[:3, :2])
                rows.append(row)
        template.__dict__["_cache_triangles"] = (np.asarray(triangles, dtype=float).reshape(-1, 3, 2),
                                                 np.asarray(rows, dtype=np.int64))
    return template.__dict__["_cache_triangles"]


SEGMENT_ENDS = ("x1", "y1", "z1", "x2", "y2", "z2")
DISPLAY_ENDS = tuple(f"display_{name}" for name in SEGMENT_ENDS)        # metafspm's: where a segment is drawn


def _suborgan_length_arrays(data_structure) -> tuple:
    """(SubOrgans, ascending; their lengths) of every SubOrgan, those without anatomy (no Compartment, so no node
    value) included: the DataStructure's when the architecture holds length at the SubOrgans, else the MPG's
    property."""
    if data_structure.has("length") and data_structure.location("length") == "SubOrgan":
        vids, lengths = np.asarray(data_structure.entity_ids("SubOrgan"), dtype=np.int64), np.asarray(data_structure.get("length"), dtype=float)
    else:
        vids, lengths = data_structure.mtg.properties()["length"].items_arrays()
        vids, lengths = np.asarray(vids, dtype=np.int64), np.asarray(lengths, dtype=float)
    order = np.argsort(vids, kind="stable")
    return vids[order], lengths[order]


def _suborgan_lengths(data_structure) -> dict:
    """{SubOrgan: length} of every SubOrgan (_suborgan_length_arrays')."""
    vids, lengths = _suborgan_length_arrays(data_structure)
    return dict(zip(vids.tolist(), lengths.tolist()))


def _per_segment(component, name):
    """{segment: value} of the component's declared SubOrgan-scale input *name*, read at its own place
    (component.read: mapped when another component holds it elsewhere, e.g. broadcast to the nodes), or None when
    nothing gives it (an optional input: its default, NaN)."""
    data_structure = component.data_structure
    values = np.asarray(component.read(name), dtype=float)
    location = component.variable_specs[name].location
    if location != "SubOrgan":                        # declared at the nodes: the segment's (broadcast) value
        values = data_structure.map_to(values, location, "SubOrgan", "mean")
    if np.isnan(values).all():
        return None
    return dict(zip(data_structure.entity_ids("SubOrgan").tolist(), np.asarray(values, dtype=float).tolist()))


def _segment_frame(mtg, segment, plant_position, frame_values) -> tuple:
    """(middle, unit axis) of *segment*: from where it is drawn (its display ends, else its ends x1 ... z2) when the
    architecture gives them, else along a
    straight axis from its plant's position (*plant_position*: {plant: (x, y, z)}, the planting's), roots down and the
    shoot up, at its distance from the axis' base. *frame_values*: {name: {segment: value} or None}, the segments'
    ends and lengths (_per_segment)."""
    properties = mtg.properties()                       # the columns, read as they are
    ends = [(frame_values.get(name) or {}).get(segment) for name in DISPLAY_ENDS]      # where it is drawn, else
    if not all(value is not None and not np.isnan(value) for value in ends):          # at its topological place
        ends = [(frame_values[name] or {}).get(segment) for name in SEGMENT_ENDS]
    if all(value is not None and not np.isnan(value) for value in ends):
        start, end = np.asarray(ends[:3], dtype=float), np.asarray(ends[3:], dtype=float)
        length = np.linalg.norm(end - start)
        if length > 0:
            return (start + end) / 2., (end - start) / length
    lengths = frame_values["length"] or {}
    below, parent = 0., mtg.parent(segment)
    while parent is not None and mtg.scale(parent) == mtg.scales.SubOrgan:
        below += float(lengths.get(parent, 0.))
        parent = mtg.parent(parent)
    plant = mtg.complex_at_scale(segment, mtg.scales.Plant)
    base = np.asarray(plant_position[int(plant)], dtype=float)
    direction = -1. if properties.get("label", {}).get(segment) in _label_codes(mtg, ROOT_KINDS) else 1.
    middle = below + float(lengths.get(segment, 0.)) / 2.
    return base + np.array([0., 0., direction * middle]), np.array([0., 0., direction])


def _perpendiculars(axis) -> tuple:
    """Two unit vectors spanning the plane perpendicular to *axis* (the section's x and y)."""
    helper = np.array([1., 0., 0.]) if abs(axis[0]) < 0.9 else np.array([0., 1., 0.])
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    return u, np.cross(axis, u)


def _vessels(mtg, compartments, family):
    """The conducting protoplasts of *family* among *compartments*: (vids, track ids, normalised positions)."""
    fam = mtg.values_of("family", compartments)
    vids = [v for v in compartments if fam.get(v, -1) == family]
    track, angle, radius = (mtg.values_of(name, vids) for name in ("track_id", "angle", "relative_radius"))
    xy = np.array([[radius[v] * np.cos(angle[v]), radius[v] * np.sin(angle[v])] for v in vids]).reshape(-1, 2)
    return vids, [track.get(v) for v in vids], xy


def vessel_pairs(mtg, parent_compartments, child_compartments, family) -> list:
    """
    The junctions [(parent-side vid, child-side vid), ...] between two successive sections for one family of
    conducting cells: vessel_junctions for one pair, sorted.
    """
    _, tails, heads = vessel_junctions(mtg, [np.asarray(parent_compartments, dtype=np.int64)],
                                       [np.asarray(child_compartments, dtype=np.int64)], family)
    return list(zip(tails.tolist(), heads.tolist()))


def vessel_junctions(mtg, parent_compartments, child_compartments, family) -> tuple:
    """
    The junctions of one family of conducting cells between the sections of many pairs, at once: *parent_compartments*
    and *child_compartments* hold each pair's two sides (arrays of Compartment vids). (pair, tails, heads): each
    junction's pair index and its parent-side and child-side vids, by pair, then sorted.

    When both sides of a pair are template copies (whole, or a reduced copy's kept nodes), its junctions depend only
    on the two templates and their nodes: computed once per (parent template and nodes, child template and nodes,
    family) on their template nodes (the population's cache), then given each pair's vids through one lookup of
    (side, template node). Every side's templates and nodes are read in one bulk read of the Compartments' table
    (_compartment_table). The other pairs (a side not of one template with each node once, or with an untemplated
    conducting cell of the family) take the direct rule (_vessel_pairs).
    """
    n = len(parent_compartments)
    sides = [np.asarray(side, dtype=np.int64) for side in list(parent_compartments) + list(child_compartments)]
    counts = np.array([side.size for side in sides], dtype=np.int64)
    vids = np.concatenate(sides) if sides else np.zeros(0, dtype=np.int64)
    side_of = np.repeat(np.arange(2 * n, dtype=np.int64), counts)
    has, templates = _table_values(mtg, "template", vids)
    has_node, nodes = _table_values(mtg, "template_node", vids)
    nodes = np.where(has & has_node, nodes, -1.).astype(np.int64)
    # a side is templated when its vertices of a template are all one template's, each node once
    templated_count = np.bincount(side_of[has], minlength=2 * n)
    bad = np.zeros(2 * n, dtype=bool)
    bad[side_of[has & ~has_node]] = True
    first = np.full(2 * n, np.nan)
    with_template, at = np.unique(side_of[has], return_index=True)
    first[with_template] = templates[has][at]                        # each side's first template
    bad[side_of[has][templates[has] != first[side_of[has]]]] = True
    order = np.lexsort((nodes, ~has, side_of))                        # by side, the templated first, by node
    repeated = has[order][1:] & has[order][:-1] & (side_of[order][1:] == side_of[order][:-1]) \
        & (nodes[order][1:] == nodes[order][:-1])
    bad[side_of[order][1:][repeated]] = True
    rest = np.unique(vids[~has])                     # an untemplated vertex (a soil node): a conducting cell?
    if rest.size:
        found, values = mtg.values_at("family", rest)
        conducting = found[np.asarray(values, dtype=np.float64) == family]
        bad[side_of[~has & np.isin(vids, conducting)]] = True
    templated = (templated_count > 0) & ~bad
    starts = np.concatenate(([0], np.cumsum(counts)))
    by_node = order[has[order]]                                       # each side's templated entries, by node
    node_starts = np.concatenate(([0], np.cumsum(templated_count)))
    stride = int(nodes.max(initial=0)) + 1
    lookup_keys = side_of[by_node] * stride + nodes[by_node]          # ascending: (side, template node)
    lookup_vids = vids[by_node]
    cache = mtg.objects("granap_vessel_pairs", saved=False)  # the population's (not checkpointed); template indices:
    pairs_out, tails_out, heads_out = [], [], []           # of their content
    for i in range(n):
        a, b = i, n + i
        if not counts[a] or not counts[b]:                 # a section without anatomy: no junction
            continue
        if not (templated[a] and templated[b]):
            direct = _vessel_pairs(mtg, sides[a].tolist(), sides[b].tolist(), family)
            if direct:
                pairs_out.append(np.full(len(direct), i, dtype=np.int64))
                tails_out.append(np.array([t for t, _ in direct], dtype=np.int64))
                heads_out.append(np.array([h for _, h in direct], dtype=np.int64))
            continue
        key = (int(first[a]), nodes[by_node[node_starts[a]:node_starts[a + 1]]].tobytes(),
               int(first[b]), nodes[by_node[node_starts[b]:node_starts[b + 1]]].tobytes(), family)
        local = cache.get(key)
        if local is None:                                  # the vessels are templated: the same pairs without the rest
            ends = [(vids[starts[side]:starts[side + 1]][has[starts[side]:starts[side + 1]]],
                     nodes[starts[side]:starts[side + 1]][has[starts[side]:starts[side + 1]]]) for side in (a, b)]
            node_a, node_b = (dict(zip(side_vids.tolist(), side_nodes.tolist())) for side_vids, side_nodes in ends)
            local = np.array([(node_a[p], node_b[c]) for p, c in
                              _vessel_pairs(mtg, ends[0][0].tolist(), ends[1][0].tolist(), family)],
                             dtype=np.int64).reshape(-1, 2)
            cache.add(key, local)
        if not local.size:
            continue
        pairs_out.append(np.full(local.shape[0], i, dtype=np.int64))
        tails_out.append(lookup_vids[np.searchsorted(lookup_keys, a * stride + local[:, 0])])
        heads_out.append(lookup_vids[np.searchsorted(lookup_keys, b * stride + local[:, 1])])
    if not pairs_out:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty
    pair, tails, heads = np.concatenate(pairs_out), np.concatenate(tails_out), np.concatenate(heads_out)
    order = np.lexsort((heads, tails, pair))
    return pair[order], tails[order], heads[order]


def _vessel_pairs(mtg, parent_compartments, child_compartments, family) -> list:
    """vessel_pairs for any two sections: by track id, then the nearest in both directions."""
    parents, parent_tracks, p_xy = _vessels(mtg, parent_compartments, family)
    children, child_tracks, c_xy = _vessels(mtg, child_compartments, family)
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


def _write_3d(mtg, copies, length_mm) -> None:
    """The new copies' 3D quantities, all at once: their walls' surfaces, protoplasts' volumes and membranes'
    surfaces, for segments of *length_mm* (one per copy: a written dict, its vids by kind; then recomputed at each
    step as they elongate, GranapAnatomy._wall_surface)."""
    def gathered(kinds):
        vids = [np.asarray(list(v for kind in kinds for v in written[kind]), dtype=np.int64) for written in copies]
        lengths = np.repeat(np.asarray(length_mm, dtype=float), [v.size for v in vids])
        vids = np.concatenate(vids) if vids else np.zeros(0, dtype=np.int64)
        order = np.argsort(vids, kind="stable")                  # ascending: one bulk read per property
        return vids[order], lengths[order]

    nodes, node_mm = gathered(("protoplasts", "apoplast"))
    mtg.set_values("wall_surface", nodes, _values_at(mtg, "wall_length", nodes, 0.) * node_mm)
    mtg.set_values("cell_volume", nodes, _values_at(mtg, "cell_area", nodes, 0.) * node_mm)
    edges, edge_mm = gathered(("connections",))
    path = _values_at(mtg, "network_path", edges, np.nan)
    membranes = np.isin(path, (columns.PATH_CODES["membrane"], columns.PATH_CODES["wall_air"]))
    edge_length = _values_at(mtg, "edge_length", edges, np.nan)
    if np.isnan(edge_length[membranes]).any():
        raise KeyError("a membrane Connection without edge_length")
    mtg.set_values("membrane_surface", edges, np.where(membranes, edge_length * edge_mm, 0.))


def _values_at(mtg, name, vids, default) -> np.ndarray:
    """Property *name* at *vids* (ascending), *default* where a vertex has none: in bulk (MPG._property_at, the
    instanced vertices' values included)."""
    out = np.full(vids.size, float(default))
    found, values = mtg.values_at(name, vids)
    if found.size:
        out[np.searchsorted(vids, found)] = np.asarray(values, dtype=np.float64)
    return out


def anatomy_ancestor(mtg, section, owners=None) -> tuple:
    """
    (the nearest ancestor of *section* with an anatomy, [the SubOrgans without one in between, nearest first]): its
    linked parent when that has an anatomy (Compartments), else the first with one up the linked parents, through
    the SubOrgans without (a pass-through: an Adel HiddenElement, a kind the scenario does not give). (None, [...]):
    no ancestor has one. *owners*: mtg.compartments_csr of the sections' scale (default: read).
    """
    scale = mtg.scale(section)
    owners = mtg.compartments_csr(scale) if owners is None else owners
    parents = mtg.linked_parents(scale)
    between, vertex = [], parents.get(section)
    while vertex is not None and not owners.has(vertex):
        between.append(int(vertex))
        vertex = parents.get(vertex)
    return (None if vertex is None else int(vertex)), between


def anatomy_descendants(mtg, sections, owners=None) -> list:
    """The sections with an anatomy whose nearest ancestor with one (anatomy_ancestor) is in *sections* across at
    least one SubOrgan without anatomy: the other ends of *sections*' pass-through junctions."""
    sections = {int(v) for v in sections}
    if not sections:
        return []
    scale = mtg.scale(next(iter(sections)))
    owners = mtg.compartments_csr(scale) if owners is None else owners
    parents = mtg.linked_parents(scale)
    if owners.has_all(list(parents)):                      # every section has an anatomy: no pass-through
        return []
    out = []
    for vertex in sorted(parents):
        if owners.has(vertex):
            parent = parents[vertex]
            if parent is not None and not owners.has(parent):
                ancestor, _ = anatomy_ancestor(mtg, vertex, owners)
                if ancestor in sections:
                    out.append(int(vertex))
    return out


def _junction_properties(mm_per_unit, mtg, section_pairs, pair, tails, heads) -> dict:
    """The properties of the axial junctions (*pair*, *tails*, *heads*) of *section_pairs* (junction_geometry), for
    every pair at once; across SubOrgans without anatomy (a pass-through), between the child and its nearest ancestor
    with one, their length spanning the SubOrgans in between."""
    n = pair.size
    vertices = np.unique(np.concatenate((section_pairs.parents, section_pairs.children)))
    lengths = _values_at(mtg, "length", vertices, 0.)
    parent_length = lengths[np.searchsorted(vertices, section_pairs.parents)]
    child_length = lengths[np.searchsorted(vertices, section_pairs.children)]
    between = 0.5 * (parent_length + child_length) * mm_per_unit
    passing = np.unique(pair[section_pairs.parent_at[pair] < 0])     # the parent without anatomy: a pass-through
    if passing.size:
        view = mtg.view("length")
        owners = mtg.compartments_csr(mtg.scale(int(section_pairs.children[passing[0]])))
        for index in passing.tolist():
            child = int(section_pairs.children[index])
            ancestor, hidden = anatomy_ancestor(mtg, child, owners)
            between[index] = (0.5 * (float(view.get(ancestor, 0.)) + float(view.get(child, 0.))) +
                              sum(float(view.get(v, 0.)) for v in hidden)) * mm_per_unit
    x, y = (_pair_values(mtg, name, tails, heads) for name in ("section_x", "section_y"))
    out = {name: np.full(n, np.nan) for name in columns.EDGE_COLUMNS}
    out.update(network_path=np.full(n, float(columns.JUNCTION_PATH)), edge_length=between[pair],
               edge_dx=x[1] - x[0], edge_dy=y[1] - y[0], membrane_surface=np.zeros(n))
    return out


def _pair_values(mtg, name, tails, heads) -> tuple:
    """Property *name* at the junctions' *tails* and *heads* (in bulk: MPG.values_at, the instanced vertices' values
    included); a KeyError when a vertex has none."""
    vids = np.unique(np.concatenate([tails, heads]))
    if vids.size == 0:
        return np.zeros(0), np.zeros(0)
    found, values = mtg.values_at(name, vids)
    if found.size != vids.size:
        raise KeyError(f"{name}: no value on vertex {int(np.setdiff1d(vids, found)[0])}")
    values = np.asarray(values, dtype=np.float64)
    return values[np.searchsorted(vids, tails)], values[np.searchsorted(vids, heads)]


def junction_geometry(mm_per_unit=MM_PER_M):
    """
    The geometry of the axial junctions a wiring rule creates (Q-C1a): their length between the two sections' centres
    (the mean of the segments' lengths, plus the lengths of the SubOrgans without anatomy between: a pass-through),
    the in-plane offset between the two vessels, the path code of an axial
    junction (5); the other network columns NaN, the membrane surface 0. A function (mtg, section_pairs, pair, tails,
    heads) -> {property: one value per junction}, for every pair at once (metafspm's wiring rules), picklable (the
    Scene's checkpoints keep the wiring).
    """
    return partial(_junction_properties, mm_per_unit)


def family_junctions(family, mtg, section_pairs) -> tuple:
    """A wiring rule over every pair of sections at once (metafspm's SectionPairs): the junctions (pair, tails,
    heads) of one family of conducting cells (vessel_junctions). When a child's linked parent has no anatomy (a
    pass-through), between the child and its nearest ancestor with one (anatomy_ancestor); a child without anatomy has
    none."""
    parents = [section_pairs.parent_compartments(i) for i in range(len(section_pairs))]
    children = [section_pairs.child_compartments(i) for i in range(len(section_pairs))]
    owners = None
    for index in np.flatnonzero((section_pairs.parent_at < 0) & (section_pairs.child_at >= 0)).tolist():
        child = int(section_pairs.children[index])
        if owners is None:
            owners = mtg.compartments_csr(mtg.scale(child))
        ancestor, _ = anatomy_ancestor(mtg, child, owners)
        if ancestor is not None:
            parents[index] = np.asarray(owners.compartments_of(ancestor), dtype=np.int64)
    return vessel_junctions(mtg, parents, children, family)


def junction_rules(mtg, mm_per_unit=MM_PER_M) -> list:
    """
    The wiring rules of anatomy mode: axial xylem junctions (apoplastic) and phloem junctions (symplastic), with their
    geometry (junction_geometry), each evaluated over every pair of sections at once; functions, not closures, so that
    the Scene's checkpoints keep them. A section whose linked parent has no anatomy is joined to its nearest ancestor
    with one (a pass-through, anatomy_ancestor).
    """
    geometry = junction_geometry(mm_per_unit)
    return [
        {"pairs": partial(family_junctions, 0), "edge_label": mtg.labels.Connection.Apoplastic, "properties": geometry},
        {"pairs": partial(family_junctions, 1), "edge_label": mtg.labels.Connection.Symplastic, "properties": geometry},
    ]
