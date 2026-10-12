"""
The anatomy library (devplan_metafspm.md, P3): one template per anatomy class, generated once, instantiated on every
segment of the class.

    key = ClassBins().key(diameter_mm, tip_mm, junction_mm)     a class: binned diameter and distances (mm)
    library = AnatomyLibrary(class_parameters={key: {"cortex": {"n_layers": 3}}})   # a class's own parameters
    template = library.template(key)                             generated at the first request (MPG-backed organ)
    written = template.instantiate(g, segment)                   the segment's own Layers, Cells, protoplasts,
                                                                 walls, junctions and Connections, in bulk

A template keeps the section as tables (one row per vertex, its properties); a segment's copy holds its own vertices
and properties (its state), but not the geometry: its Cells point to the template's polygons and rings
(template, template_row; Q-P3d), stored once per class.
"""
import copy
from dataclasses import dataclass, field

import numpy as np

from openalea.granap.input_data import OrganInputData
from openalea.granap.mpg.build import JUNCTION, PROTOPLAST, WALL, _props
from openalea.granap.mpg.vessels import vessel_ids

GEOMETRY = ("cell__polygon", "boundary")          # kept once, in the template
LINKS = ("n_id_a", "n_id_b")                       # rewritten per instance
SKIP = GEOMETRY + LINKS + ("cell_uid", "scale", "edge_type", "isanchor")


@dataclass(frozen=True)
class ClassBins:
    """Bin widths (mm) of an anatomy class; a width of 0 ignores that dimension (one bin)."""
    diameter: float = 0.05
    tip: float = 0.
    junction: float = 0.

    def key(self, diameter, tip=0., junction=0.) -> tuple:
        def binned(value, width):
            return 0. if width <= 0 else round(round(float(value) / width) * width, 9)
        return binned(diameter, self.diameter), binned(tip, self.tip), binned(junction, self.junction)


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
        organ.generate_cells()
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
        properties = g.properties()
        for vids, table in ((layers, self.layers), (cells, self.cells), (protoplasts, self.protoplasts),
                            (apoplast, self.apoplast), (connections, self.connections)):
            vids = np.asarray(vids, dtype=np.int64)
            for name, (rows, values) in table.items():
                properties.setdefault(name, {}).update(dict(zip(vids[rows].tolist(), values)))
        return {"layers": layers, "cells": cells, "protoplasts": protoplasts, "apoplast": apoplast,
                "connections": connections, "template": self.index}


class AnatomyLibrary:
    """
    Templates by class, generated at their first request. A class's parameters: *parameters(key, overrides=...)*
    (default: the root preset), with the class's own parametrization from *class_parameters*: {key: overrides} or
    a callable key -> overrides (None: the base parameters), where overrides is {param: {field: value}}.
    """

    def __init__(self, parameters=default_root_parameters, class_parameters=None, seed: int = 0, organ_class=None):
        from openalea.granap import mpg
        self.parameters, self.class_parameters, self.seed = parameters, class_parameters or {}, seed
        self.organ_class = organ_class or mpg.RootAnatomy
        self.templates = {}

    def overrides(self, key) -> dict:
        source = self.class_parameters
        return (source(key) if callable(source) else source.get(key)) or {}

    def template(self, key) -> Template:
        if key not in self.templates:
            organ = self.organ_class(self.parameters(key, overrides=self.overrides(key)), seed=self.seed)
            self.templates[key] = Template.from_organ(organ, key, len(self.templates))
        return self.templates[key]

    def by_index(self, index) -> Template:
        return next(t for t in self.templates.values() if t.index == index)
