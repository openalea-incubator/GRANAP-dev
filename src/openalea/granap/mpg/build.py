"""
A GRANAP cross-section written into a metafspm MPG (devplan_metafspm.md, P1).

    g, segment = single_plant_mpg()                 one vertex per scale down to one SubOrgan
    write_cells(g, segment, organ)                  Layers (one per connected region of one tissue type), Cells (every
                                                    attribute of each GRANAP Cell) and one protoplast per Cell
    write_network(g, segment, organ)                the walls and wall junctions (Compartments anchored to the segment,
                                                    shared by the cells), the virtual bridge cells, and every network
                                                    edge (Connections), with all the exporter's attributes

The section is stored at SubOrgan -> Layer -> Cell -> Compartment, the protoplasts under their Cell, the walls and
junctions under the segment's Compartment anchor; every MECHA node is a Compartment and every MECHA edge a
Connection (metafspm devplan_anatomy_scales.md). The MECHA node index of each Compartment is kept (mecha_index), so
that the network rebuilt from the MPG (views.py) is the exporter's, node for node.
"""
import numpy as np
from shapely import STRtree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.metafspm.data_structure.mpg import MPG

PROTOPLAST, WALL, JUNCTION = 0, 1, 2          # Compartment kinds
PATH_LABELS = {"membrane": "Transmembrane", "wall_air": "Transmembrane", "plasmodesmata": "Symplastic",
               "wall": "Apoplastic", "air_link": "Apoplastic"}


def single_plant_mpg():
    """An MPG with one vertex per scale from the Plant to one SubOrgan (a root segment); returns (g, segment)."""
    g = MPG()
    s, L = g.scales, g.labels
    chain = [(s.Plant, None), (s.Axis, L.Axis.Root), (s.GrowthUnit, L.GrowthUnit.Root),
             (s.Phytomer, L.Phytomer.Root), (s.Organ, L.Organ.RootInternode), (s.SubOrgan, L.SubOrgan.RootSegment)]
    vertex = g.root
    for scale, label in chain:
        props = dict(scale=scale, edge_type='/')
        if label is not None:
            props["label"] = label
        vertex = g.add_component(vertex, **PropsConfig(**props))
    return g, vertex


def _props(scale, **values):
    out = dict(PropsConfig(scale=scale, edge_type='/'))
    out.update(values)
    return out


def _write_present(g, prefix, vids, records):
    """For each vertex of *vids* and its record (a dict), the record's entries as properties prefix + name, only
    those the record has (an absent attribute stays absent, not None)."""
    properties = g.properties()
    for v, record in zip(vids, records):
        for name, value in record.items():
            properties.setdefault(prefix + name, {})[v] = value


def tissue_regions(cells) -> np.ndarray:
    """Per cell, its region: the connected component of the cells of its type that touch (Layer scale, Q4 b)."""
    n = len(cells)
    polygons = [c.polygon for c in cells]
    types = np.array([c.type for c in cells], dtype=object)
    real = np.array([p is not None and not p.is_empty for p in polygons])
    rows, cols = np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    if real.any():
        index = np.flatnonzero(real)
        tree = STRtree([polygons[i] for i in index])
        a, b = tree.query([polygons[i] for i in index], predicate="intersects")
        a, b = index[a], index[b]
        same = (types[a] == types[b]) & (a != b)
        rows, cols = a[same], b[same]
    adjacency = coo_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n))
    _, labels = connected_components(adjacency, directed=False)
    return labels


def write_cells(g, segment, organ) -> dict:
    """
    The Layers, Cells and protoplasts of *organ*'s cells (organ.all_cells, after generate_cells) under *segment*.
    Each Cell vertex holds every attribute of its GRANAP Cell (cell__<name>, the polygon included) and its row in
    the cell list (cell_row). Returns {"layers", "cells", "protoplasts"}: vids, cells and protoplasts in row order.
    """
    s = g.scales
    cells = list(organ.all_cells.cells)
    regions = tissue_regions(cells)
    order = np.argsort(regions, kind="stable")                    # the Cells of each Layer, consecutive
    region_ids, first = np.unique(regions[order], return_index=True)
    layer_types = [cells[order[i]].type for i in first]
    layers = g.add_components_bulk(segment, len(region_ids), **_props(s.Layer, tissue=layer_types))
    cell_vids = np.empty(len(cells), dtype=np.int64)
    bounds = list(first) + [len(cells)]
    for layer, start, stop in zip(layers, bounds[:-1], bounds[1:]):
        rows = order[start:stop]
        vids = g.add_components_bulk(layer, rows.size, **_props(s.Cell, cell_row=rows.tolist(),
                                                                 tissue=[cells[r].type for r in rows]))
        _write_present(g, "cell__", vids, [vars(cells[r]) for r in rows])
        cell_vids[rows] = vids
    protoplasts = g.add_components_bulk(cell_vids.tolist(), len(cells), **_props(
        s.Compartment, kind=PROTOPLAST, label=g.labels.Compartment.Symplastic,
        tissue=[c.type for c in cells], cell_row=list(range(len(cells)))))
    return {"layers": layers, "cells": cell_vids.tolist(), "protoplasts": protoplasts}


def network_cell_rows(gdf) -> list:
    """
    The rows of the cells behind the exporter's cell nodes, in node order: the cells with a valid geometry in gdf
    order, then the bridge cells (whose row labels are their positions in organ.all_cells).
    """
    bridge = gdf["bridge"].astype(bool) if "bridge" in gdf.columns else np.zeros(len(gdf), dtype=bool)
    valid = gdf["geometry"].notna() & gdf["geometry"].apply(lambda p: p is not None and not p.is_empty)
    return [int(r) for r in gdf.index[valid & ~bridge]] + [int(r) for r in gdf.index[bridge]]


def write_network(g, segment, organ, written: dict) -> dict:
    """
    The network of *organ* (organ.graph, built by the exporter) under *segment*: its wall and junction nodes as
    Compartments anchored to the segment, its virtual bridge cells (appended to organ.all_cells by the exporter) as
    Cells with their protoplasts in a Layer of their own, and every edge as a Connection with its attributes. Node
    attributes go to the Compartments (node__<name>), and each Compartment keeps its MECHA index (mecha_index).
    """
    s, L = g.scales, g.labels
    graph = organ.graph
    cells = list(organ.all_cells.cells)
    known = len(written["cells"])
    if len(cells) > known:                                       # bridge cells, added by the export
        layer = g.add_components_bulk(segment, 1, **_props(s.Layer, tissue="bridge"))[0]
        rows = list(range(known, len(cells)))
        vids = g.add_components_bulk(layer, len(rows), **_props(s.Cell, cell_row=rows,
                                                                tissue=[cells[r].type for r in rows]))
        _write_present(g, "cell__", vids, [vars(cells[r]) for r in rows])
        protoplasts = g.add_components_bulk(vids, len(rows), **_props(
            s.Compartment, kind=PROTOPLAST, label=L.Compartment.Symplastic,
            tissue=[cells[r].type for r in rows], cell_row=rows))
        written["layers"].append(layer)
        written["cells"] += vids
        written["protoplasts"] += protoplasts
    n_walls, n_junctions = organ.n_walls, organ.n_junctions
    gdf = organ._cells_gdf
    exported_rows = network_cell_rows(gdf)
    vid_of_node = {}
    for i, row in enumerate(exported_rows):
        vid_of_node[n_walls + n_junctions + i] = written["protoplasts"][row]
    nodes = sorted(graph.nodes)
    apoplast = [n for n in nodes if n < n_walls + n_junctions]
    anchor = s.anchors[s.Compartment]
    apo_vids = g.add_components_bulk(anchor, len(apoplast), topo_parents=[segment] * len(apoplast), **_props(
        s.Compartment, label=L.Compartment.Apoplastic,
        kind=[WALL if n < n_walls else JUNCTION for n in apoplast]))
    vid_of_node.update(zip(apoplast, apo_vids))
    _write_present(g, "node__", [vid_of_node[n] for n in nodes], [graph.nodes[n] for n in nodes])
    mecha = g.properties().setdefault("mecha_index", {})
    for n in nodes:
        mecha[vid_of_node[n]] = int(n)
    edges = list(graph.edges(data=True))
    connections = g.add_components_bulk(
        s.anchors[s.Connection], len(edges), topo_parents=[segment] * len(edges),
        **_props(s.Connection, n_id_a=[vid_of_node[u] for u, _, _ in edges],
                 n_id_b=[vid_of_node[v] for _, v, _ in edges],
                 edge_u=[int(u) for u, _, _ in edges], edge_v=[int(v) for _, v, _ in edges],
                 label=[getattr(L.Connection, PATH_LABELS.get(data.get("path"), "Apoplastic")) for _, _, data in edges]))
    _write_present(g, "edge__", connections, [data for _, _, data in edges])
    written.update(apoplast=apo_vids, connections=connections, n_walls=n_walls, n_junctions=n_junctions,
                   wall_to_cells={int(w): list(c) for w, c in organ._wall_to_cells.items()},
                   gdf_columns=list(gdf.columns))
    return written
