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


UID = "cell_uid"                              # the stable id a cell carries through the stages (Q-P2c)


def _record(cell) -> dict:
    """A GRANAP Cell's attributes as written to its Cell vertex (cell__<name>), its pipeline uid left out."""
    return {name: value for name, value in vars(cell).items() if name != UID}


def write_cells(g, segment, organ, written: dict = None, stage: str = "final") -> dict:
    """
    Write the current cells of *organ* (organ.all_cells) into the section under *segment*, at a stage of the
    pipeline (P2: grouping, intercellular, aerenchyma, fused_gaps, final). Cells are identified across stages by
    their uid (given here to the cells that have none: those just grouped or created by the stage):
      * a surviving cell keeps its Cell vertex and protoplast, its attributes rewritten (cell__<name>, polygon
        included) and its row in the cell list updated (cell_row);
      * a cell that is gone (fused, nested, carved away) is removed with its protoplast;
      * a new cell (an air space) gets a Cell vertex and a protoplast;
      * the Layers are the connected regions of one tissue type of the current cells: each stage builds them anew,
        moves the Cells into them and removes the emptied ones.
    *written* (the record of the previous stages, None at the first) is updated and returned: Layers, Cells and
    protoplasts in row order, the uid -> vid maps, and the stages written.
    """
    s, L = g.scales, g.labels
    if written is None:
        written = {"layers": [], "cells": [], "protoplasts": [], "cell_of_uid": {}, "protoplast_of_uid": {},
                   "next_uid": 0, "stages": []}
    cells = list(organ.all_cells.cells)
    for cell in cells:
        if not hasattr(cell, UID):
            setattr(cell, UID, written["next_uid"])
            written["next_uid"] += 1
    uids = [getattr(c, UID) for c in cells]
    if len(set(uids)) != len(uids):                    # a cell duplicated by a stage: the copies are new cells
        seen = set()
        for cell in cells:
            if cell.cell_uid in seen:
                cell.cell_uid = written["next_uid"]
                written["next_uid"] += 1
            seen.add(cell.cell_uid)
        uids = [c.cell_uid for c in cells]
    present = set(uids)
    gone = [u for u in written["cell_of_uid"] if u not in present]
    g.remove_connections([written["protoplast_of_uid"].pop(u) for u in gone])        # vertices and properties
    g.remove_connections([written["cell_of_uid"].pop(u) for u in gone])
    old_layers = list(written["layers"])
    regions = tissue_regions(cells)
    region_ids, first = np.unique(regions, return_index=True)
    layers = g.add_components_bulk(segment, len(region_ids), **_props(s.Layer, tissue=[cells[i].type for i in first]))
    layer_of_row = np.asarray(layers, dtype=np.int64)[np.searchsorted(region_ids, regions)]
    for layer in layers:                                           # the surviving Cells, moved to their region
        rows = np.flatnonzero(layer_of_row == layer)
        kept = [written["cell_of_uid"][uids[r]] for r in rows if uids[r] in written["cell_of_uid"]]
        if kept:
            g.move_components(kept, int(layer))
    g.remove_connections(old_layers)                               # emptied: their Cells were all moved
    new_rows = [r for r, uid in enumerate(uids) if uid not in written["cell_of_uid"]]
    for layer in layers:
        rows = [r for r in new_rows if layer_of_row[r] == layer]
        if not rows:
            continue
        vids = g.add_components_bulk(int(layer), len(rows), **_props(s.Cell))
        protoplasts = g.add_components_bulk(vids, len(rows), **_props(
            s.Compartment, kind=PROTOPLAST, label=L.Compartment.Symplastic))
        for r, v, p in zip(rows, vids, protoplasts):
            written["cell_of_uid"][uids[r]], written["protoplast_of_uid"][uids[r]] = v, p
    cell_vids = [written["cell_of_uid"][u] for u in uids]
    protoplasts = [written["protoplast_of_uid"][u] for u in uids]
    properties = g.properties()
    for name in [n for n in properties if n.startswith("cell__")]:
        values = properties[name]
        for v in cell_vids:
            values.pop(v, None)                                    # an attribute a stage removed stays removed
    _write_present(g, "cell__", cell_vids, [_record(c) for c in cells])
    for prop, values in (("cell_row", range(len(cells))), ("tissue", [c.type for c in cells]), (UID, uids)):
        target = properties.setdefault(prop, {})
        for v, p, value in zip(cell_vids, protoplasts, values):
            target[v] = value
            if prop != UID:
                target[p] = value
    written.update(layers=list(layers), cells=cell_vids, protoplasts=protoplasts)
    written["stages"].append(stage)
    return written


def network_cell_rows(gdf) -> list:
    """
    The rows of the cells behind the exporter's cell nodes, in node order: the cells with a valid geometry in gdf
    order, then the bridge cells (whose row labels are their positions in organ.all_cells).
    """
    bridge = gdf["bridge"].astype(bool) if "bridge" in gdf.columns else np.zeros(len(gdf), dtype=bool)
    valid = gdf["geometry"].notna() & gdf["geometry"].apply(lambda p: p is not None and not p.is_empty)
    return [int(r) for r in gdf.index[valid & ~bridge]] + [int(r) for r in gdf.index[bridge]]


def clear_network(g, written: dict) -> None:
    """Remove the stored network of the section: its Connections, walls, junctions and bridge cells, and the network
    attributes of the protoplasts."""
    if "connections" not in written:
        return
    g.remove_connections(written.pop("connections"))
    g.remove_connections(written.pop("apoplast"))
    bridges = written.pop("bridge_cells", [])
    if bridges:
        g.remove_connections(written.pop("bridge_protoplasts"))
        g.remove_connections(bridges)
        g.remove_connections([written.pop("bridge_layer")])
        written["layers"] = [v for v in written["layers"] if g.has_vertex(v)]
        keep = len(written["cells"]) - len(bridges)
        written["cells"], written["protoplasts"] = written["cells"][:keep], written["protoplasts"][:keep]
    for name in [n for n in g.properties() if n.startswith("node__") or n == "mecha_index"]:
        values = g.property(name)
        for v in written["protoplasts"]:
            values.pop(v, None)


def write_network(g, segment, organ, written: dict, network=None) -> dict:
    """
    The network built by the exporter into *network* (default: the organ itself, organ.graph) under *segment*,
    replacing the one stored before: its wall and junction nodes as Compartments anchored to the segment, its
    virtual bridge cells (appended to organ.all_cells by the exporter) as Cells with their protoplasts in a Layer of
    their own, and every edge as a Connection with its attributes. Node attributes go to the Compartments
    (node__<name>), and each Compartment keeps its MECHA index (mecha_index).
    """
    s, L = g.scales, g.labels
    network = organ if network is None else network
    clear_network(g, written)
    graph = network.graph
    cells = list(organ.all_cells.cells)
    known = len(written["cells"])
    if len(cells) > known:                                       # bridge cells, added by the export
        layer = g.add_components_bulk(segment, 1, **_props(s.Layer, tissue="bridge"))[0]
        rows = list(range(known, len(cells)))
        vids = g.add_components_bulk(layer, len(rows), **_props(s.Cell, cell_row=rows,
                                                                tissue=[cells[r].type for r in rows]))
        _write_present(g, "cell__", vids, [_record(cells[r]) for r in rows])
        protoplasts = g.add_components_bulk(vids, len(rows), **_props(
            s.Compartment, kind=PROTOPLAST, label=L.Compartment.Symplastic,
            tissue=[cells[r].type for r in rows], cell_row=rows))
        written["layers"].append(layer)
        written["cells"] += vids
        written["protoplasts"] += protoplasts
        written.update(bridge_layer=layer, bridge_cells=vids, bridge_protoplasts=protoplasts)
    n_walls, n_junctions = network.n_walls, network.n_junctions
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
                   wall_to_cells={int(w): list(c) for w, c in network._wall_to_cells.items()},
                   gdf_columns=list(gdf.columns))
    return written
