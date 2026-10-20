"""
The anatomy library (devplan_metafspm.md, P3): one template per anatomy class, generated once, instantiated on every
segment of the class.

    key = class_key(class_inputs, g, segment)                   a class: its class inputs' binned values
    library = AnatomyLibrary(class_parameters={key: {"cortex": {"n_layers": 3}}})   # a class's own parameters
    template = library.template(key, plant=overrides)           generated at the first request (per plant overrides)
    written = template.instantiate(g, segment)                   the segment's own Layers, Cells, protoplasts,
                                                                 walls, junctions and Connections, in bulk

A template keeps the section as tables (one row per vertex, its properties); a segment's copy holds its own vertices
and properties (its state), but not the geometry: its Cells point to the template's polygons and rings
(template, template_row; Q-P3d), stored once per class.
"""
import copy
import itertools
from dataclasses import dataclass, field

import numpy as np

from openalea.granap.input_data import OrganInputData
from openalea.granap.metafspm.build import JUNCTION, PROTOPLAST, WALL, _props
from openalea.granap.metafspm.vessels import vessel_ids

GEOMETRY = ("cell__polygon", "boundary")          # kept once, in the template
_TEMPLATE_IDS = itertools.count()                 # template ids, unique across libraries (roots, stems, leaves)
LINKS = ("n_id_a", "n_id_b")                       # rewritten per instance
SKIP = GEOMETRY + LINKS + ("cell_uid", "scale", "edge_type", "isanchor")


@dataclass(frozen=True)
class ClassInput:
    """
    One dimension of an anatomy class: a SubOrgan input variable (declared by the anatomy component, given by the
    architecture model), binned. A segment's class is the tuple of its class inputs' binned values.

      variable   the SubOrgan property (e.g. "radius")
      width      the bin width, in the unit of value * factor (0: one bin, the variable is not read)
      factor     the variable's value times factor gives the binned quantity (e.g. radius in m -> diameter in mm: 2000)
      target     (GRANAP section, field) set to the binned value in the class's parameters (e.g. ("cortex", "n_layers")),
                 or None: the class is keyed only (the class's own parametrization says what differs)
      compute    a function (g, segment) -> value computing the variable from the segment's inputs instead of reading
                 it (e.g. the distance from the root-shoot junction, from the segments' lengths)
    """
    variable: str
    width: float = 0.
    factor: float = 1.
    target: tuple = None
    compute: object = None

    def binned(self, value) -> float:
        return 0. if self.width <= 0 else round(round(float(value) * self.factor / self.width) * self.width, 9)

    def value(self, g, segment) -> float:
        if self.width <= 0:
            return 0.
        if self.compute is not None:
            return self.binned(self.compute(g, segment))
        values = g.property(self.variable)
        if segment not in values:
            raise KeyError(f"class input '{self.variable}': no value on segment {segment} (an input the architecture "
                           f"model gives)")
        return self.binned(values[segment])


def class_key(class_inputs, g, segment) -> tuple:
    """A segment's anatomy class: its class inputs' binned values."""
    return tuple(c.value(g, segment) for c in class_inputs)


def class_targets(class_inputs, key) -> dict:
    """The class inputs' parameter overrides of the class *key* ({section: {field: binned value}})."""
    out = {}
    for c, value in zip(class_inputs, key):
        if c.target is not None and c.width > 0:
            section, field_name = c.target
            out.setdefault(section, {})[field_name] = value
    return out


def merged(*overrides) -> dict:
    """{section: {field: value}} overrides applied in order (the later win)."""
    out = {}
    for layer in overrides:
        for section, values in (layer or {}).items():
            out.setdefault(section, {}).update(values)
    return out


def _hashable(value):
    """*value* as a hashable key: dicts as sorted (key, value) tuples, lists and sets as tuples, recursively (a
    field's value may be a dict, e.g. a size distribution)."""
    if isinstance(value, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple, set)):
        return tuple(_hashable(v) for v in (sorted(value) if isinstance(value, set) else value))
    return value


def _frozen(overrides) -> tuple:
    return tuple(sorted((section, _hashable(values)) for section, values in (overrides or {}).items()))


def preset_parameters(preset: str):
    """A parameters(key, overrides=...) function on an OrganInputData preset (e.g. "for_monocot_stem"): the preset
    unchanged, then the class's overrides (as default_root_parameters for roots)."""
    def parameters(key, base: OrganInputData = None, overrides: dict = None) -> OrganInputData:
        return default_root_parameters(key, base=getattr(OrganInputData, preset)() if base is None else base,
                                       overrides=overrides)
    parameters.__name__ = f"{preset}_parameters"
    return parameters


def default_root_parameters(key, base: OrganInputData = None, overrides: dict = None) -> OrganInputData:
    """
    A root's parameters for the class *key*: *base* (default OrganInputData.for_root()) unchanged, then *overrides*
    ({param: {field: value}}, e.g. {"stele": {"thickness": 0.2}, "cortex": {"n_layers": 3}}) set on it. No rule
    derives parameters from the class (F-P3a, your answer b): a class whose anatomy differs (a thinner lateral, a
    maturing segment) says so in its own parametrization (AnatomyLibrary's class_parameters).
    """
    data = OrganInputData.for_root() if base is None else copy.deepcopy(base)
    for name, values in (overrides or {}).items():
        for field_name, value in values.items():
            data.set_value(name, field_name, value)
    return data


def _vertex_props(g, vids, skip=SKIP) -> dict:
    """{property: (local rows, values)} of the vertices *vids*, the entries present only."""
    position = {int(v): i for i, v in enumerate(vids)}
    out = {}
    for name, values in g.properties().items():
        if name in skip:
            continue
        rows, kept = [], []
        for v, value in values.items():
            i = position.get(int(v))
            if i is not None:
                rows.append(i)
                kept.append(value)
        if rows:
            out[name] = (np.asarray(rows, dtype=np.int64), kept)
    return out


@dataclass
class Template:
    """A generated section as tables, ready to be instantiated on segments."""
    key: tuple
    index: int
    layers: dict                       # properties of the Layers
    cell_layer: np.ndarray             # the Layer (local index) of each Cell, in cell order
    cells: dict                        # properties of the Cells
    protoplasts: dict
    apoplast: dict
    connections: dict
    links: np.ndarray                  # (n_connections, 2) local node indices: protoplasts first, then apoplast
    polygons: list                     # the cells' polygons, shared by every instance
    rings: dict                        # {cell row: snapped ring}, shared
    organ: object = field(repr=False, default=None)

    @classmethod
    def from_organ(cls, organ, key, index) -> "Template":
        """The section of a generated MPG-backed organ (its stored network included), with vessel ids."""
        organ.generate_section()
        g, w = organ.mpg, organ._mpg_written
        cells, protoplasts, apoplast, connections = w["cells"], w["protoplasts"], w["apoplast"], w["connections"]
        layer_index = {int(v): i for i, v in enumerate(w["layers"])}
        ids = vessel_ids(organ)
        cell_props = _vertex_props(g, cells)
        proto_props = _vertex_props(g, protoplasts)
        rows = np.arange(len(cells))
        for name in ("vessel_id", "pole_id", "family", "angle", "relative_radius"):
            for props in (cell_props, proto_props):
                props[name] = (rows, ids[name].tolist())
        tracked = [(r, c.track_id) for r, c in enumerate(organ.all_cells.cells) if c.track_id is not None]
        if tracked:                                              # the series' tracked vessels, for the junctions
            proto_props["track_id"] = (np.array([r for r, _ in tracked], dtype=np.int64), [t for _, t in tracked])
        local = {int(v): i for i, v in enumerate(list(protoplasts) + list(apoplast))}
        a, b = g.property("n_id_a"), g.property("n_id_b")
        links = np.array([[local[int(a[e])], local[int(b[e])]] for e in connections], dtype=np.int64)
        boundary = g.property("boundary")
        return cls(key=key, index=index, layers=_vertex_props(g, w["layers"]),
                   cell_layer=np.array([layer_index[int(g.complex(c))] for c in cells], dtype=np.int64),
                   cells=cell_props, protoplasts=proto_props, apoplast=_vertex_props(g, apoplast),
                   connections=_vertex_props(g, connections), links=links,
                   polygons=[g.property("cell__polygon").get(int(c)) for c in cells],
                   rings={r: boundary[int(p)] for r, p in enumerate(protoplasts) if int(p) in boundary},
                   organ=organ)

    def instantiate(self, g, segment) -> dict:
        """The segment's own copy of the section, created in bulk; returns its vids by kind."""
        s = g.scales
        n_cells = len(self.cell_layer)
        n_layers = len(next(iter(self.layers.values()))[0]) if self.layers else 0
        layers = g.add_components_bulk(segment, n_layers, **_props(s.Layer))
        cells = g.add_components_bulk(np.asarray(layers, dtype=np.int64)[self.cell_layer].tolist(), n_cells,
                                      **_props(s.Cell, template=self.index, template_row=list(range(n_cells))))
        protoplasts = g.add_components_bulk(cells, n_cells, **_props(
            s.Compartment, template=self.index, template_row=list(range(n_cells))))
        n_apo = len(self.apoplast["kind"][0]) if "kind" in self.apoplast else 0
        apoplast = g.add_components_bulk(s.anchors[s.Compartment], n_apo, topo_parents=[segment] * n_apo,
                                         **_props(s.Compartment))
        nodes = np.asarray(list(protoplasts) + list(apoplast), dtype=np.int64)
        n_edges = len(self.links)
        connections = g.add_components_bulk(
            s.anchors[s.Connection], n_edges, topo_parents=[segment] * n_edges,
            **_props(s.Connection, n_id_a=nodes[self.links[:, 0]].tolist(), n_id_b=nodes[self.links[:, 1]].tolist()))
        g.set_values("template_node", nodes, np.arange(nodes.size))          # each Compartment's place in the template
        g.set_values("template", nodes, [self.index] * nodes.size)
        for vids, table in ((layers, self.layers), (cells, self.cells), (protoplasts, self.protoplasts),
                            (apoplast, self.apoplast), (connections, self.connections)):
            vids = np.asarray(vids, dtype=np.int64)
            for name, (rows, values) in table.items():
                g.set_values(name, vids[rows], values)
        return {"layers": layers, "cells": cells, "protoplasts": protoplasts, "apoplast": apoplast,
                "connections": connections, "template": self.index}


class AnatomyLibrary:
    """
    Templates by class, generated at their first request. A class's parameters: *parameters(key, overrides=...)*
    (default: the root preset), with, in order, the plant's overrides (its scenario's granap_overrides), the class's
    own parametrization from *class_parameters* ({key: overrides} or a callable key -> overrides), and the class
    inputs' targets; overrides are {section: {field: value}}. Templates are keyed by (class, plant's overrides): plants
    parametrized alike share them.
    """

    def __init__(self, parameters=default_root_parameters, class_parameters=None, seed: int = 0, organ_class=None):
        from openalea.granap.root_class import RootAnatomy
        self.parameters, self.class_parameters, self.seed = parameters, class_parameters or {}, seed
        self.organ_class = organ_class or RootAnatomy
        self.templates = {}

    def overrides(self, key) -> dict:
        source = self.class_parameters
        return (source(key) if callable(source) else source.get(key)) or {}

    def template(self, key, plant: dict = None, targets: dict = None) -> Template:
        """The template of class *key* for a plant with overrides *plant*, the class inputs' *targets* applied."""
        entry = key if not plant else (key, _frozen(plant))
        if entry not in self.templates:
            overrides = merged(plant, self.overrides(key), targets)
            organ = self.organ_class(self.parameters(key, overrides=overrides), seed=self.seed)
            self.templates[entry] = Template.from_organ(organ, key, next(_TEMPLATE_IDS))
        return self.templates[entry]

    def by_index(self, index) -> Template:
        return next(t for t in self.templates.values() if t.index == index)


def find_template(libraries, index) -> Template:
    """The template of id *index* among several libraries."""
    for library in libraries:
        for template in library.templates.values():
            if template.index == index:
                return template
    raise KeyError(f"no template {index}")
