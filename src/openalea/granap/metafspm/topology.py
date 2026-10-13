"""
The section's topology, built once and stored in the MPG (devplan_metafspm.md, P2.2-P2.3).

GRANAP derives walls and junctions from the cell polygons with CellGenerator._build_topology: vertex snapping
(phase 0), then the rings, the vertex and edge maps (phase 1) and the junctions (phase 2). Only phase 0 needs the
polygons; everything else follows from the snapped rings. The rings are stored once per section (each protoplast's
boundary), and the topology is rebuilt from them, identical to GRANAP's down to the iteration order of its dicts and
sets, so that the walls the exporter and the XML writer number from it are the same.

    store_topology(g, segment, organ, written)       snap once, store the rings
    topology_from_mpg(g, segment, written)            GRANAP's 5-tuple, from the MPG (Organ.section_topology returns it
                                                      to the exporter and the XML writer)
"""
from openalea.granap.generate_cell import CellGenerator

def _valid_rows(cells):
    """The rows the exporter and the XML writer tessellate (a valid geometry), their polygons, and the protected
    rows among them (protect_topology), as they read them from the cell GeoDataFrame."""
    rows = [r for r, c in enumerate(cells) if c.polygon is not None and not c.polygon.is_empty]
    return rows, [cells[r].polygon for r in rows], set(r for r in rows if cells[r].protect_topology)


def store_topology(g, segment, organ, written: dict) -> tuple:
    """Snap the final cells' vertices once (GRANAP's phase 0), store each ring on its protoplast; returns the topology."""
    ids, polys, protect = _valid_rows(organ.all_cells.cells)
    topology = CellGenerator._build_topology(polys, ids, protect_ids=protect)
    cell_vkeys = topology[0]
    boundary = g.properties().setdefault("boundary", {})
    for row, vkeys in cell_vkeys.items():
        boundary[written["protoplasts"][int(row)]] = list(vkeys)
    written["topology"] = {"rows": [int(r) for r in cell_vkeys], "ids": [int(i) for i in ids],
                           "protect": None if protect is None else sorted(int(i) for i in protect)}
    return topology


def topology_from_mpg(g, segment, written: dict) -> tuple:
    """GRANAP's (cell_vkeys, vertex_to_cells, edge_to_cells, junction_set, protected_shape_set), from the MPG."""
    record = written["topology"]
    boundary = g.property("boundary")
    cell_vkeys = {row: list(boundary[written["protoplasts"][row]]) for row in record["rows"]}
    vertex_to_cells = {}
    for cid, vkeys in cell_vkeys.items():                 # phase 1, as GRANAP builds it
        for vk in vkeys:
            vertex_to_cells.setdefault(vk, set()).add(cid)
    edge_to_cells = {}
    for cid, vkeys in cell_vkeys.items():
        n = len(vkeys)
        for i in range(n):
            ek = tuple(sorted((vkeys[i], vkeys[(i + 1) % n])))
            edge_to_cells.setdefault(ek, set()).add(cid)
    junction_set = set()                                  # phase 2
    for vk in vertex_to_cells:
        if len(vertex_to_cells[vk]) >= 3:
            junction_set.add(vk)
            continue
        incident_pairs = set()
        for cid in vertex_to_cells[vk]:
            vks = cell_vkeys[cid]
            n = len(vks)
            for i in range(n):
                if vks[i] != vk:
                    continue
                ek_prev = tuple(sorted((vks[(i - 1) % n], vk)))
                ek_next = tuple(sorted((vk, vks[(i + 1) % n])))
                if ek_prev in edge_to_cells:
                    incident_pairs.add(frozenset(edge_to_cells[ek_prev]))
                if ek_next in edge_to_cells:
                    incident_pairs.add(frozenset(edge_to_cells[ek_next]))
        if len(incident_pairs) > 1:
            junction_set.add(vk)
    protected_shape_set = set()
    if record["protect"]:
        for cid in record["protect"]:
            for vk in cell_vkeys.get(cid, ()):
                protected_shape_set.add(vk)
    return cell_vkeys, vertex_to_cells, edge_to_cells, junction_set, protected_shape_set
