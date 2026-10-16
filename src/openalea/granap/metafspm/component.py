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

from openalea.metafspm.coupling.component import StructuralComponent, input_variable, state_variable
from openalea.metafspm.data_structure.configs import ScalesConfig
from openalea.metafspm.solve.decorator import postsegmentation, state

from openalea.granap.metafspm import columns
from openalea.granap.metafspm.build import LEAF_SURFACE, _props as _build_props
from openalea.granap.metafspm.library import AnatomyLibrary, ClassInput, class_key, class_targets

SCALES = ScalesConfig()
MM_PER_M = 1000.                 # GRANAP's sections are in mm; the MPG's lengths in m


def _doc(unit, description, **values):
    return dict(unit=unit, unit_comment="", description=description, min_value=-1e12, max_value=1e12,
                value_comment="", references="GRANAP", DOI=[], **values)


@dataclass
class OrganAnatomy:
    """
    The anatomy of one kind of segment: its library and its class inputs (default: the diameter from *size*, binned by
    0.05 mm; the distance from the tip and from the root-shoot junction, not binned). *size*: the SubOrgan property
    giving the section's size (None: one class).
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


def _rectangle_sides(rectangle):
    """The two side lengths of a (rotated) rectangle polygon (mm)."""
    xy = np.asarray(rectangle.exterior.coords)[:3]
    return float(np.hypot(*(xy[1] - xy[0]))), float(np.hypot(*(xy[2] - xy[1])))


def remove_anatomy(g, segment) -> None:
    """Remove *segment*'s anatomy (its Layers with their Cells and protoplasts, its anchored walls, junctions and
    anatomy Connections), keeping the segment."""
    s = g.scales
    anchored = [c for c in list(g._children.get(segment, [])) if g.scale(c) in (s.Compartment, s.Connection)]
    g.remove_connections([c for c in anchored if g.scale(c) == s.Connection])
    g.remove_connections([c for c in anchored if g.scale(c) == s.Compartment])
    g.remove_connections([c for c in list(g._components.get(segment, [])) if g.scale(c) == s.Layer])


def _instance(g, comps):
    """(template, template node of each of *comps*) when *comps* are exactly one copy of a template, else None."""
    vids = np.asarray(comps, dtype=np.int64)
    if vids.size == 0:
        return None
    order = np.argsort(vids, kind="stable")
    found, templates = g._property_at("template", vids[order])
    found_nodes, nodes = g._property_at("template_node", vids[order])
    if found.size != vids.size or found_nodes.size != vids.size or not (templates == templates[0]).all():
        return None
    out = np.empty(vids.size, dtype=np.int64)
    out[order] = np.asarray(nodes, dtype=np.int64)
    if not np.array_equal(np.sort(out), np.arange(vids.size)):
        return None
    return int(templates[0]), out


def _positions(g, vids):
    """(angle and radius present, x, y) of *vids*, computed value by value as the original rule does."""
    angle, radius = g.property("angle"), g.property("relative_radius")
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
    from_scale = ds._from_scale
    new_vids = np.asarray(new_vids, dtype=np.int64)
    sources = np.full(new_vids.size, -1, dtype=np.int64)
    sections = ds._sections_of(new_vids)
    owners = g.compartments_by_owner(from_scale)
    tissue = g.property("tissue")
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


def _inheritance_by_vertex(ds, new_vids, is_known, fresh=()) -> np.ndarray:
    """granap_inheritance, Compartment by Compartment (for sections that are not whole template copies)."""
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


@dataclass
class GranapAnatomy(StructuralComponent):
    """
    The cross-section anatomies of a population's segments (roots, and the stems and leaves of *shoot*): each segment
    gets the anatomy of its class, one template generated per class and plant parametrization, a copy per segment.
    One instance serves the whole population.

    Instructions for the generation:
      library        the roots' AnatomyLibrary (its parameters(key, overrides=...): GRANAP's root preset by default)
      shoot          {SubOrgan label name: OrganAnatomy}, e.g. stems and leaves
      a plant's scenario parameters may give granap_overrides: {section: {field: value}} applied to GRANAP's
      parameters for that plant (e.g. a genotype's cortex), before the class's own and its class inputs' targets
      each organ's class_inputs (OrganAnatomy) bin SubOrgan inputs of the architecture model into the class, and may
      set GRANAP fields from them (ClassInput.target)
      mm_per_unit    the MPG's length unit in mm (1000: lengths in m)

    Declared below: the inputs read from the architecture model (role "Architecture" in the composite's translator),
    and the state variables a coupled model (e.g. MECHA's) reads: the network of every section (Compartments and
    Connections, quantities in mm, mm2, mm3), its codes, and the 3D quantities recomputed as the segments elongate.
    """
    # ── Inputs: the segments' architecture ──
    radius: float = input_variable(**_doc("m", "the segment's radius (the diameter class input)"),
                                   by="Architecture", scale=SCALES.SubOrgan, state_variable_type="intensive")
    distance_from_tip: float = input_variable(**_doc("m", "the segment's distance from its axis' tip"),
                                              by="Architecture", scale=SCALES.SubOrgan, state_variable_type="intensive")
    length: float = input_variable(**_doc("m", "the segment's length, broadcast to its Compartments"),
                                   by="Architecture", scale=SCALES.SubOrgan, location="node",
                                   state_variable_type="intensive")

    # ── State variables: the anatomies ──
    anatomy_template: float = state_variable(**_doc("", "the segment's anatomy template (-1: none)"), initialize=-1.,
                                             scale=SCALES.SubOrgan, state_variable_type="descriptor")
    tissue: object = state_variable(**_doc("", "the protoplast's tissue (None: a wall or a junction)"), initialize=None,
                                    scale=SCALES.Compartment, location="node", dtype=object,
                                    state_variable_type="descriptor")
    kind: float = state_variable(**_doc("", "0 protoplast, 1 wall, 2 wall junction, 3 a leaf organ's surface"),
                                 initialize=0.,
                                 scale=SCALES.Compartment, location="node", state_variable_type="descriptor")
    pore_width: float = state_variable(**_doc("mm", "a stomatal pore's aperture width in the section (NaN: not a pore)"),
                                       initialize=np.nan, scale=SCALES.Compartment, location="node",
                                       state_variable_type="NonInertialIntensive")
    pore_depth: float = state_variable(**_doc("mm", "a stomatal pore's depth through the epidermis (NaN: not a pore)"),
                                       initialize=np.nan, scale=SCALES.Compartment, location="node",
                                       state_variable_type="NonInertialIntensive")
    cgroup: float = state_variable(**_doc("", "the protoplast's MECHA cell group (NaN: apoplast)"), initialize=np.nan,
                                   scale=SCALES.Compartment, location="node", state_variable_type="descriptor")
    section_x: float = state_variable(**_doc("mm", "position in the section"), initialize=np.nan,
                                      scale=SCALES.Compartment, location="node", state_variable_type="NonInertialIntensive")
    section_y: float = state_variable(**_doc("mm", "position in the section"), initialize=np.nan,
                                      scale=SCALES.Compartment, location="node", state_variable_type="NonInertialIntensive")
    wall_length: float = state_variable(**_doc("mm", "the wall's length in the section (0: not a wall)"), initialize=0.,
                                        scale=SCALES.Compartment, location="node",
                                        state_variable_type="NonInertialExtensive")
    wall_thickness: float = state_variable(**_doc("µm", "the wall's thickness: its cells' wall layers (the "
                                                        "AnatomyWriter's per-tissue values, µm; NaN: not a wall)"),
                                           initialize=np.nan,
                                           scale=SCALES.Compartment, location="node",
                                           state_variable_type="NonInertialIntensive")
    cell_area: float = state_variable(**_doc("mm2", "the protoplast's area in the section (0: apoplast)"),
                                      initialize=0., scale=SCALES.Compartment, location="node",
                                      state_variable_type="NonInertialExtensive")
    wall_surface: float = state_variable(**_doc("mm2", "the wall's surface: its length times the segment's"),
                                         initialize=0., scale=SCALES.Compartment, location="node",
                                         state_variable_type="NonInertialExtensive")
    cell_volume: float = state_variable(**_doc("mm3", "the protoplast's volume: its area times the segment's length"),
                                        initialize=0., scale=SCALES.Compartment, location="node",
                                        state_variable_type="NonInertialExtensive")
    network_path: float = state_variable(**_doc("", "edge type: membrane 0, wall_air 1, wall 2, plasmodesmata 3, "
                                                    "air_link 4, axial junction 5, stomatal 6"),
                                         initialize=np.nan,
                                         scale=SCALES.Connection, location="edge", state_variable_type="descriptor")
    edge_length: float = state_variable(**_doc("mm", "the edge's length (the wall's for membranes; between the sections "
                                                     "for axial junctions)"), initialize=np.nan,
                                        scale=SCALES.Connection, location="edge",
                                        state_variable_type="NonInertialIntensive")
    edge_dx: float = state_variable(**_doc("mm", "the edge's vector in the section, x"), initialize=np.nan,
                                    scale=SCALES.Connection, location="edge", state_variable_type="NonInertialIntensive")
    edge_dy: float = state_variable(**_doc("mm", "the edge's vector in the section, y"), initialize=np.nan,
                                    scale=SCALES.Connection, location="edge", state_variable_type="NonInertialIntensive")
    edge_dist: float = state_variable(**_doc("mm", "the distance between the edge's nodes (NaN: wall edges)"),
                                      initialize=np.nan, scale=SCALES.Connection, location="edge",
                                      state_variable_type="NonInertialIntensive")
    edge_wall_thickness: float = state_variable(**_doc("µm", "the wall's thickness on membrane and wall edges"),
                                                initialize=np.nan, scale=SCALES.Connection, location="edge",
                                                state_variable_type="NonInertialIntensive")
    lateral_distance: float = state_variable(**_doc("mm", "wall edges: the lateral distance"), initialize=np.nan,
                                             scale=SCALES.Connection, location="edge",
                                             state_variable_type="NonInertialIntensive")
    distnode_wall_cell: float = state_variable(**_doc("mm", "wall edges: the mean wall-cell distance"),
                                               initialize=np.nan, scale=SCALES.Connection, location="edge",
                                               state_variable_type="NonInertialIntensive")
    membrane_surface: float = state_variable(**_doc("mm2", "membrane edges: the membrane's surface (the wall's length "
                                                           "times the segment's); 0 otherwise"), initialize=0.,
                                             scale=SCALES.Connection, location="edge",
                                             state_variable_type="NonInertialExtensive")

    library = AnatomyLibrary()
    shoot = {}                       # {SubOrgan label name: OrganAnatomy}, e.g. stems and leaves (roots: library)
    leaf_labels = ("LeafElement",)   # the SubOrgan labels of leaves: their organs get a surface node (plan §60)
    mm_per_unit = MM_PER_M

    @classmethod
    def organs(cls) -> dict:
        """{SubOrgan label name: OrganAnatomy}: the root segments with cls.library, and cls.shoot."""
        return {"RootSegment": OrganAnatomy(cls.library, "radius"), **cls.shoot}

    @classmethod
    def organ_of(cls, g, segment) -> OrganAnatomy:
        code = g.property("label").get(segment)
        return next(organ for name, organ in cls.organs().items() if getattr(g.labels.SubOrgan, name) == code)

    @classmethod
    def segment_class(cls, g, segment) -> tuple:
        """The segment's anatomy class: its organ's class inputs, binned."""
        return class_key(cls.organ_of(g, segment).class_inputs, g, segment)

    @classmethod
    def initiate_plant(cls, g, plant_vid: int, parameters: dict) -> None:
        """Each segment of the plant gets its class's anatomy; the plant's granap_overrides (scenario) are kept on it."""
        overrides = (parameters or {}).get("granap_overrides")
        if overrides:
            g.properties().setdefault("granap_overrides", {})[plant_vid] = overrides
        segments = _segments(g, plant_vid, tuple(cls.organs()))
        cls.update_anatomies(g, segments)
        cls.update_leaf_surfaces(g, segments)

    @classmethod
    def update_anatomies(cls, g, segments) -> dict:
        """
        Give each segment the anatomy of its current class (Q12): a segment without one (just emerged) gets its
        class's copy; a segment whose class changed has its anatomy replaced. Returns {"emerged", "replaced"}.
        """
        anatomy_class = g.properties().setdefault("anatomy_class", {})
        template_of = g.properties().setdefault("anatomy_template", {})
        plants = g.properties().get("granap_overrides", {})
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
            organ = cls.organ_of(g, segment)
            plant = plants.get(g.complex_at_scale(segment, g.scales.Plant)) if plants else None
            template = organ.library.template(key, plant=plant, targets=class_targets(organ.class_inputs, key))
            written = template.instantiate(g, segment)
            _write_3d(g, written, float(g.property("length").get(segment, 0.)) * cls.mm_per_unit)
            anatomy_class[segment] = key
            template_of[segment] = float(template.index)
        return done

    @classmethod
    def update_leaf_surfaces(cls, g, segments) -> None:
        """
        The leaf organs of *segments* get (again) their surface (plan §60): one Compartment of kind 3 (tissue
        "leaf surface"), owned by the organ's basal leaf element, and one Connection (network_path 6) from every
        stomatal pore of each of the organ's sections to it, anchored to the pore's section. The pores get their
        aperture's width and depth (pore_width, pore_depth: the short and long sides of their polygon's minimum
        rotated rectangle). The organ's previous surface and stomatal Connections are removed first, so that none
        dangles when a section was replaced.
        """
        import shapely
        s, L = g.scales, g.labels
        labels = g.property("label")
        codes = {getattr(L.SubOrgan, name) for name in cls.leaf_labels if name in cls.organs()}
        organs = sorted({int(g.complex_at_scale(v, s.Organ)) for v in segments if labels.get(v) in codes})
        if not organs:
            return
        surfaces = g.properties().setdefault("leaf_surface", {})
        path, tissue = g.property("network_path"), g.property("tissue")
        template_row = g.property("template_row")
        for organ in organs:
            elements = [int(v) for v in g.components_at_scale(organ, scale=s.SubOrgan) if labels.get(v) in codes]
            stomatal = [c for e in elements for c in list(g._children.get(e, []))
                        if g.scale(c) == s.Connection and path.get(c) == columns.STOMATAL_PATH]
            g.remove_connections(stomatal)
            old = surfaces.pop(organ, None)
            if old is not None and g.has_vertex(old):
                g.remove_connections([old])
            pores = {e: [int(c) for c in g.components_at_scale(e, scale=s.Compartment) if tissue.get(c) == "pore"]
                     for e in elements}
            if not any(pores.values()):
                continue
            inside = set(elements)
            basal = next((e for e in elements if g.parent(e) not in inside), elements[0])
            surface = g.add_components_bulk(s.anchors[s.Compartment], 1, topo_parents=[basal], **_build_props(
                s.Compartment, label=L.Compartment.Apoplastic, kind=float(LEAF_SURFACE)))[0]
            g.set_values("tissue", [surface], ["leaf surface"])
            for name in columns.NODE_COLUMNS:
                g.set_values(name, [surface], [0. if name in ("wall_length", "cell_area") else np.nan])
            surfaces[organ] = int(surface)
            template_of = g.property("anatomy_template")
            for element, vids in pores.items():
                if not vids:
                    continue
                template = cls.organ_of(g, element).library.by_index(int(template_of[element]))
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

    def __post_init__(self):
        g = self.data_structure.mtg
        segments = set(self.data_structure.entity_ids("SubOrgan").tolist())
        template_of = g.properties().setdefault("anatomy_template", {})
        for segment in segments - set(template_of):            # segments without an anatomy (labels not handled)
            template_of[segment] = -1.
        super().__post_init__()
        self._replaced = set()
        self.data_structure.inheritance = lambda ds, new, known: granap_inheritance(ds, new, known, self._replaced)

    @postsegmentation
    def _anatomies(self):
        """MPG-style: every segment of the MPG gets its class's anatomy (emerged, or replaced on a change)."""
        g = self.mtg
        handled = tuple(self.organs())
        segments = [s for plant in g.components_at_scale(g.root, scale=g.scales.Plant)
                    for s in _segments(g, plant, handled)]
        alive = set(segments)
        anatomy_class = g.properties().get("anatomy_class", {})
        for gone in [s for s in list(anatomy_class) if s not in alive]:      # pruned segments: their record only
            del anatomy_class[gone]
        done = self.update_anatomies(g, segments)
        self.update_leaf_surfaces(g, done["emerged"] + done["replaced"])
        template_of = g.properties()["anatomy_template"]
        for segment in g.components_at_scale(g.root, scale=g.scales.SubOrgan):
            template_of.setdefault(int(segment), -1.)
        self._replaced = set(done["replaced"])
        self.last_update = done

    @state
    def _wall_surface(self, wall_length, cell_area, length, edge_length, network_path) \
            -> tuple[np.ndarray, str, np.ndarray, str, np.ndarray, str, np.ndarray]:
        """
        The 3D quantities, as the segments elongate: the walls' surfaces, the protoplasts' volumes, the membranes'
        surfaces (2D values times the segment's length), and the axial junctions' lengths (between the two sections'
        centres, the mean of the segments' lengths).
        """
        length_mm = np.asarray(length, dtype=float) * self.mm_per_unit
        ends = abs(self.data_structure.incidence_matrix()).T                # (edges, nodes): the edge's two nodes
        edge_mm = 0.5 * np.asarray(ends @ length_mm).reshape(-1)
        path = np.asarray(network_path)
        membranes = (path == columns.PATH_CODES["membrane"]) | (path == columns.PATH_CODES["wall_air"])
        return (wall_length * length_mm,
                "cell_volume", cell_area * length_mm,
                "membrane_surface", np.where(membranes, edge_length * edge_mm, 0.),
                "edge_length", np.where(path == columns.JUNCTION_PATH, edge_mm, edge_length))


def _vessels(g, compartments, family):
    """The conducting protoplasts of *family* among *compartments*: (vids, track ids, normalised positions)."""
    fam, track = g.property("family"), g.property("track_id")
    angle, radius = g.property("angle"), g.property("relative_radius")
    vids = [v for v in compartments if fam.get(v, -1) == family]
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
    cache = g.__dict__.setdefault("_granap_vessel_pairs", {})
    key = (parent[0], child[0], family)
    local = cache.get(key)
    if local is None:
        node_a, node_b = dict(zip(parent_compartments, parent[1].tolist())), \
            dict(zip(child_compartments, child[1].tolist()))
        local = cache[key] = [(node_a[a], node_b[b])
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
    nodes = list(written["protoplasts"]) + list(written["apoplast"])
    wall_length, area = g.property("wall_length"), g.property("cell_area")
    g.set_values("wall_surface", nodes, [wall_length.get(v, 0.) * length_mm for v in nodes])
    g.set_values("cell_volume", nodes, [area.get(v, 0.) * length_mm for v in nodes])
    edges = list(written["connections"])
    path, edge_length = g.property("network_path"), g.property("edge_length")
    membranes = (columns.PATH_CODES["membrane"], columns.PATH_CODES["wall_air"])
    g.set_values("membrane_surface", edges, [edge_length[e] * length_mm if path.get(e) in membranes else 0.
                                             for e in edges])


def junction_geometry(mm_per_unit=MM_PER_M):
    """
    The geometry of the axial junctions a wiring rule creates (Q-C1a): their length between the two sections' centres
    (the mean of the segments' lengths), the in-plane offset between the two vessels, the path code of an axial
    junction (5); the other network columns NaN, the membrane surface 0.
    """
    def properties(g, parent, child, pairs):
        n = len(pairs)
        lengths = g.property("length")
        between = 0.5 * (float(lengths.get(parent, 0.)) + float(lengths.get(child, 0.))) * mm_per_unit
        x, y = g.property("section_x"), g.property("section_y")
        out = {name: [np.nan] * n for name in columns.EDGE_COLUMNS}
        out.update(network_path=[float(columns.JUNCTION_PATH)] * n, edge_length=[between] * n,
                   edge_dx=[x[b] - x[a] for a, b in pairs], edge_dy=[y[b] - y[a] for a, b in pairs],
                   membrane_surface=[0.] * n)
        return out
    return properties


def junction_rules(g, mm_per_unit=MM_PER_M) -> list:
    """
    The wiring rules of anatomy mode: axial xylem junctions (apoplastic) and phloem junctions (symplastic), with their
    geometry (junction_geometry).
    """
    geometry = junction_geometry(mm_per_unit)
    return [
        {"rule": lambda g, p, c, pc, cc: vessel_pairs(g, pc, cc, 0), "edge_label": g.labels.Connection.Apoplastic,
         "properties": geometry},
        {"rule": lambda g, p, c, pc, cc: vessel_pairs(g, pc, cc, 1), "edge_label": g.labels.Connection.Symplastic,
         "properties": geometry},
    ]
