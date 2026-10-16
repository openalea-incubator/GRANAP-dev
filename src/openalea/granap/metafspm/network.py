"""
The section's hydraulic network built in the MPG (devplan_metafspm.md, §28, N1).

GRANAP's NetworkExporter (anatomy_writer.py; converted copy in mpg/pipeline/network.py) walks the cell rings of a
GeoDataFrame into a networkx graph, which P2 then copied into the MPG. Here the same network is built from the MPG
itself, the section's Cells and the rings stored on their protoplasts (mpg/topology.py), with array operations, and
written straight into Compartments (walls, junctions) and Connections:

    build_network(organ)        the stored network of organ's section (the exporter's default settings, no bridges)

The exporter's rules and its rounding are kept: walls numbered in first-encounter order along the rings, sums taken in
the exporter's order, numpy's mean reproduced (sequential below 8 values, numpy's own above), junctions sorted as
tuples, the air links searched in the iteration order of GRANAP's sets and of networkx's neighbours. The graph rebuilt
from the MPG (views.network_from_mpg) is the exporter's, node for node and edge for edge; the Connections come in the
exporter's order of creation.
"""
import numpy as np
import shapely

from openalea.granap.anatomy_writer import CGROUP_MAP, DEFAULT_CELL_WALL_THICKNESS
from openalea.granap.metafspm import columns
from openalea.granap.metafspm.build import JUNCTION, PATH_LABELS, WALL, _props, clear_network
from openalea.granap.metafspm.topology import topology_from_mpg

_PAIRWISE_BLOCK = 128          # numpy's pairwise summation: one block up to 128 values


def _starts(lens):
    """The start of each run of *lens* consecutive values (empty for none)."""
    lens = np.asarray(lens, dtype=np.int64)
    return np.cumsum(lens) - lens


def _sequential_sums(values, lens):
    """Per run of *lens* consecutive values, their sum in order from 0 (Python's sum)."""
    lens = np.asarray(lens, dtype=np.int64)
    starts = _starts(lens)
    out = np.zeros(lens.size)
    for j in range(int(lens.max()) if lens.size else 0):
        m = lens > j
        out[m] = out[m] + values[starts[m] + j]
    return out


def _pairwise_sums(columns):
    """numpy's pairwise sum of n <= 128 values (one block: sequential below 8, eight accumulators above), done
    elementwise over the rows of *columns* (one array per position)."""
    n = len(columns)
    if n < 8:
        res = np.full(columns[0].shape, -0.0)
        for c in columns:
            res = res + c
        return res
    r = [columns[j].copy() for j in range(8)]
    i = 8
    while i < n - n % 8:
        for j in range(8):
            r[j] = r[j] + columns[i + j]
        i += 8
    res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
    for k in range(i, n):
        res = res + columns[k]
    return res


def _means(values, lens):
    """numpy's mean of each run of *lens* consecutive values, bit for bit: its pairwise sum emulated up to 128
    values, numpy's own mean above."""
    lens = np.asarray(lens, dtype=np.int64)
    out = np.empty(lens.size)
    starts = _starts(lens)
    for n in np.unique(lens).tolist():
        runs = np.flatnonzero(lens == n)
        if n <= _PAIRWISE_BLOCK:
            out[runs] = _pairwise_sums([values[starts[runs] + j] for j in range(n)]) / n
        else:
            for i in runs:
                out[i] = np.mean(values[starts[i]:starts[i] + n])
    return out


def _frame_column(values):
    """A column's values as pandas gives them from records: numbers mixed with None become floats (None: NaN)."""
    if any(v is None for v in values) and any(v is not None for v in values):
        if all(v is None or isinstance(v, (int, float, np.number)) and not isinstance(v, bool) for v in values):
            return [np.nan if v is None else float(v) for v in values]
    return list(values)


def _section_cells(g, written):
    """The section's cells as the exporter reads its frame: per row (cell order, bridges left out), polygon, type,
    protect_topology and cgroup; and the rows with a valid geometry."""
    n_rows = len(written["cells"]) - len(written.get("bridge_cells", []))
    vids = written["cells"][:n_rows]
    props = g.properties()

    def column(name, default=None):
        values = props.get("cell__" + name, {})
        return [values.get(v, default) for v in vids]

    polygons = np.array(column("polygon"), dtype=object)
    present = np.array([p is not None for p in polygons], dtype=bool)
    valid = present.copy()
    valid[present] = ~shapely.is_empty(polygons[present])
    return dict(polygons=polygons, types=column("type"), protect=column("protect_topology", False),
                cgroup=_frame_column(column("cgroup")), valid=np.flatnonzero(valid))


def _walls(cell_vkeys, junction_set, protected_shape_set):
    """
    The exporter's phase 3: every ring cut at its junctions, a wall per (junction pair, protected shape) in
    first-encounter order. Returns the vertices (id -> key), and per wall: its vertex sequences (for its length and
    midpoint), junctions, signature, and the cells along it in encounter order.
    """
    order = list(cell_vkeys)
    rings = [cell_vkeys[r] for r in order]
    n_per = np.array([len(r) for r in rings], dtype=np.int64)
    index = {}
    flat = np.fromiter((index.setdefault(vk, len(index)) for ring in rings for vk in ring), dtype=np.int64,
                       count=int(n_per.sum()))
    verts = list(index)
    xy = np.array(verts, dtype=float).reshape(-1, 2)
    is_j = np.fromiter((vk in junction_set for vk in verts), dtype=bool, count=len(verts))
    is_p = np.fromiter((vk in protected_shape_set for vk in verts), dtype=bool, count=len(verts))
    offsets = np.concatenate(([0], np.cumsum(n_per)))
    jflat = np.flatnonzero(is_j[flat])
    jcell = np.searchsorted(offsets, jflat, side="right") - 1
    n_junc = np.bincount(jcell, minlength=len(rings))
    cells_idx, first_pos = np.unique(jcell, return_index=True)
    j0 = np.zeros(len(rings), dtype=np.int64)
    j0[cells_idx] = jflat[first_pos] - offsets[cells_idx]

    # the rings with two junctions or more, each rotated to start at its first junction and closed
    normal = np.flatnonzero(n_junc >= 2)
    closed = n_per[normal] + 1
    cell_rep = np.repeat(normal, closed)
    local = np.arange(int(closed.sum())) - np.repeat(_starts(closed), closed)
    cv = flat[offsets[cell_rep] + (j0[cell_rep] + local) % n_per[cell_rep]]
    jp = np.flatnonzero(is_j[cv])
    last = np.r_[cell_rep[jp][1:] != cell_rep[jp][:-1], True] if jp.size else np.zeros(0, dtype=bool)
    P, E = jp[~last], jp[np.flatnonzero(~last) + 1]
    seg_cell = cell_rep[P]
    seg_k = np.arange(P.size) - np.repeat(_starts(n_junc[normal]), n_junc[normal])
    js, je = cv[P], cv[E]
    cp = np.concatenate(([0], np.cumsum(is_p[cv])))
    signatures = {(): 0}
    sig = np.zeros(P.size, dtype=np.int64)
    for s in np.flatnonzero(cp[E] - cp[P + 1] > 0):
        key = tuple(verts[v] for v in cv[P[s] + 1:E[s]] if is_p[v])
        sig[s] = signatures.setdefault(key, len(signatures))
    V, S = len(verts), len(signatures)
    if V * V * S >= 2 ** 62:
        raise OverflowError("section too large for the wall keys")
    code = (np.minimum(js, je) * V + np.maximum(js, je)) * S + sig

    # the rings with fewer than two junctions: one wall, the whole boundary
    special = np.flatnonzero(n_junc < 2)
    ring_keys = {}
    special_code = np.array([-1 - ring_keys.setdefault(tuple(sorted(rings[c])), len(ring_keys)) for c in special],
                            dtype=np.int64)
    n_k = int(n_junc.max()) + 1 if len(rings) else 1
    all_cell = np.concatenate((seg_cell, special))
    rank = np.argsort(np.concatenate((seg_cell * n_k + seg_k, special * n_k)), kind="stable")
    all_code = np.concatenate((code, special_code))[rank]
    all_cell = all_cell[rank]
    source = np.concatenate((np.arange(P.size), -1 - np.arange(special.size)))[rank]   # segment, or -1 - special

    uniq, first, inverse = np.unique(all_code, return_index=True, return_inverse=True)
    wall_rank = np.empty(uniq.size, dtype=np.int64)
    wall_rank[np.argsort(first, kind="stable")] = np.arange(uniq.size)
    seg_wall = wall_rank[inverse.ravel()]
    first_of_wall = first[np.argsort(wall_rank)]                 # the segment that created each wall
    n_walls = uniq.size

    sequences, mean_lens, j_start, j_end, wall_sig = [], np.empty(n_walls, dtype=np.int64), \
        np.empty(n_walls, dtype=np.int64), np.empty(n_walls, dtype=np.int64), np.zeros(n_walls, dtype=np.int64)
    for w, f in enumerate(first_of_wall):
        s = source[f]
        if s >= 0:
            seq = cv[P[s]:E[s] + 1]
            mean_lens[w], j_start[w], j_end[w], wall_sig[w] = seq.size, js[s], je[s], sig[s]
        else:
            c = special[-1 - s]
            ring = flat[offsets[c]:offsets[c + 1]]
            seq = np.append(ring, ring[0])
            mean_lens[w], j_start[w], j_end[w] = ring.size, ring[0], ring[0]
            wall_sig[w] = -1                                     # no shape signature key at all
        sequences.append(seq)
    seq_lens = np.array([s.size for s in sequences], dtype=np.int64)
    concat = np.concatenate(sequences) if sequences else np.zeros(0, dtype=np.int64)
    steps = np.hypot(xy[concat[1:], 0] - xy[concat[:-1], 0], xy[concat[1:], 1] - xy[concat[:-1], 1])
    starts = _starts(seq_lens)
    step_index = np.concatenate([np.arange(a, a + n - 1) for a, n in zip(starts, seq_lens)]) if n_walls else \
        np.zeros(0, dtype=np.int64)
    lengths = _sequential_sums(steps[step_index], seq_lens - 1)
    mean_index = np.concatenate([np.arange(a, a + n) for a, n in zip(starts, mean_lens)]) if n_walls else \
        np.zeros(0, dtype=np.int64)
    mid_x = _means(xy[concat[mean_index], 0], mean_lens)
    mid_y = _means(xy[concat[mean_index], 1], mean_lens)

    row_of = np.asarray(order, dtype=np.int64)
    pair = seg_wall * len(rings) + all_cell
    _, pair_first = np.unique(pair, return_index=True)
    pair_first.sort()
    by_wall = pair_first[np.argsort(seg_wall[pair_first], kind="stable")]
    wall_cells = [[] for _ in range(n_walls)]
    for w, c in zip(seg_wall[by_wall].tolist(), row_of[all_cell[by_wall]].tolist()):
        wall_cells[w].append(c)
    return dict(verts=verts, xy=xy, n_walls=n_walls, j_start=j_start, j_end=j_end, signature=wall_sig,
                lengths=lengths, mid_x=mid_x, mid_y=mid_y, cells=wall_cells)


def _thickness(cell_wall_thickness):
    def get(c_type):
        if isinstance(cell_wall_thickness, dict):
            return float(cell_wall_thickness.get(c_type, cell_wall_thickness.get("default", 1)))
        return float(cell_wall_thickness)
    return get


def build_network(organ, cell_wall_thickness=DEFAULT_CELL_WALL_THICKNESS) -> dict:
    """
    The network of organ's section with the exporter's default settings and no bridges, built from the MPG and
    written into it (replacing the stored one): walls and junctions as Compartments anchored to the segment, every edge
    a Connection, the nodes' and edges' quantities as numeric columns (columns.py) and the MECHA indices
    (mecha_index). Returns the updated record of the section.
    """
    g, segment, written = organ.mpg, organ.segment, organ._mpg_written
    s, L = g.scales, g.labels
    clear_network(g, written)
    cells = _section_cells(g, written)
    valid, types = cells["valid"], cells["types"]
    cell_vkeys, _, _, junction_set, protected_shape_set = topology_from_mpg(g, segment, written)
    written.update(apoplast=[], connections=[], n_walls=0, n_junctions=0, n_cells=0, wall_to_cells={})
    if not cell_vkeys:
        return written
    valid_list = valid.tolist()
    protected_air_cells = set(r for r in valid_list if cells["protect"][r] and types[r] == "air space")
    # the stomatal pores are air too (plan §60): by their type, without protect_topology (which would split walls)
    pore_cells = [r for r in valid_list if types[r] in getattr(organ, "PORE_TYPES", ())]
    apoplastic_types = set(getattr(organ, "APOPLASTIC_ONLY_TYPES", ()))
    apoplastic_cells = set(r for r in valid_list if types[r] in apoplastic_types)
    get_thickness = _thickness(cell_wall_thickness)

    walls = _walls(cell_vkeys, junction_set, protected_shape_set)
    n_walls, xy, wall_cells = walls["n_walls"], walls["xy"], walls["cells"]
    lengths, mid_x, mid_y = walls["lengths"], walls["mid_x"], walls["mid_y"]
    counts = np.array([len(wc) for wc in wall_cells], dtype=np.int64)
    starts = _starts(counts)
    mem_w = np.repeat(np.arange(n_walls), counts)                  # (wall, cell) in the walls' encounter order
    mem_r = np.fromiter((r for wc in wall_cells for r in wc), dtype=np.int64, count=int(counts.sum()))
    type_thickness = {t: get_thickness(t) for t in set(types[r] for r in valid_list)}
    n_rows = len(types)
    row_thickness = np.zeros(n_rows)
    row_thickness[valid] = [type_thickness[types[r]] for r in valid_list]
    thickness = _sequential_sums(row_thickness[mem_r], counts)     # 0.0 + each cell's, then the outer wall's
    thickness = np.where(counts == 1, thickness + get_thickness("outerwall"), thickness)

    # phase 4: node indices; the junctions sorted as (x, y) tuples
    used = np.unique(np.concatenate((walls["j_start"], walls["j_end"])))
    used = used[np.lexsort((xy[used, 1], xy[used, 0]))]
    n_junctions = used.size
    junction_node = np.full(len(walls["verts"]), -1, dtype=np.int64)
    junction_node[used] = n_walls + np.arange(n_junctions)
    n_cells = valid.size
    node_of_row = np.full(n_rows, -1, dtype=np.int64)
    node_of_row[valid] = n_walls + n_junctions + np.arange(n_cells)
    polygons = cells["polygons"][valid]
    centroids = shapely.centroid(polygons)
    cx, cy = np.full(n_rows, np.nan), np.full(n_rows, np.nan)
    cx[valid], cy[valid] = shapely.get_x(centroids), shapely.get_y(centroids)
    areas = shapely.area(polygons)

    # phase 6: membrane (or wall_air), wall and plasmodesmata edges, in the exporter's order (wall by wall)
    is_air = np.zeros(n_rows, dtype=bool)
    is_air[list(protected_air_cells) + pore_cells] = True
    is_apoplastic = np.zeros(n_rows, dtype=bool)
    is_apoplastic[list(apoplastic_cells)] = True
    has_sig = walls["signature"] > 0
    slots = int(counts.max()) + 3
    blocks = []                                     # (order, u, v, {attribute: values})
    dx, dy = mid_x[mem_w] - cx[mem_r], mid_y[mem_w] - cy[mem_r]
    dist = np.hypot(dx, dy)
    blocks.append((mem_w * slots + (np.arange(mem_w.size) - np.repeat(starts, counts)), node_of_row[mem_r], mem_w,
                   dict(path=np.where(is_air[mem_r], "wall_air", "membrane").tolist(),      # every air wall (§60)
                        length=lengths[mem_w], dist=dist, d_vec=list(np.column_stack((dx, dy))),
                        wall_thickness=thickness[mem_w].tolist())))
    mean_dist = _means(dist, counts)
    distinct = walls["j_start"] != walls["j_end"]
    jw = np.concatenate((np.arange(n_walls), np.flatnonzero(distinct)))
    jv = np.concatenate((walls["j_start"], walls["j_end"][distinct]))
    jdx, jdy = xy[jv, 0] - mid_x[jw], xy[jv, 1] - mid_y[jw]
    blocks.append((jw * slots + np.where(np.arange(jw.size) < n_walls, slots - 3, slots - 2), junction_node[jv], jw,
                   dict(path=["wall"] * jw.size, length=lengths[jw] / 2.0,
                        lateral_distance=mean_dist[jw] + np.hypot(jdx, jdy), d_vec=list(np.column_stack((jdx, jdy))),
                        distnode_wall_cell=mean_dist[jw].tolist(), wall_thickness=thickness[jw].tolist())))
    two = np.flatnonzero(counts == 2)
    a, b = mem_r[starts[two]], mem_r[starts[two] + 1]
    keep = (is_air[a] == is_air[b]) & ~is_apoplastic[a] & ~is_apoplastic[b]
    two, a, b = two[keep], a[keep], b[keep]
    na, nb = node_of_row[a], node_of_row[b]
    pair = np.minimum(na, nb) * (n_walls + n_junctions + n_cells) + np.maximum(na, nb)
    _, first = np.unique(pair, return_index=True)                  # networkx: the edge where first created,
    last = pair.size - 1 - np.unique(pair[::-1], return_index=True)[1]          # its attributes the last ones
    by_creation = np.argsort(first)
    first, last = first[by_creation], last[by_creation]
    pdx, pdy = cx[b[last]] - cx[a[last]], cy[b[last]] - cy[a[last]]
    pdist = np.hypot(pdx, pdy)
    blocks.append((two[first] * slots + slots - 1, na[first], nb[first],
                   dict(path=["plasmodesmata"] * first.size, length=lengths[two[last]], dist=pdist,
                        d_vec=list(np.column_stack((pdx, pdy))))))

    cell_node = dict(zip(valid_list, node_of_row[valid].tolist()))
    air_links = []
    if protected_air_cells:
        position = {cell_node[r]: (float(cx[r]), float(cy[r])) for r in protected_air_cells}
        linked = set(map(frozenset, zip(na[first].tolist(), nb[first].tolist())))
        air_links = _air_links(walls, wall_cells, junction_node, n_walls, used, protected_air_cells, cell_node,
                               position, linked, lengths, cells["polygons"])
    air_links += _pore_links(pore_cells, protected_air_cells, cell_node, cx, cy, cells["polygons"])
    for i, (u_, v_, attributes) in enumerate(air_links):
        blocks.append((np.array([n_walls * slots + i]), np.array([u_]), np.array([v_]),
                       {k: [x] for k, x in attributes.items()}))

    # into the MPG
    anchor = s.anchors[s.Compartment]
    n_apo = n_walls + n_junctions
    apo_vids = g.add_components_bulk(anchor, n_apo, topo_parents=[segment] * n_apo, **_props(
        s.Compartment, label=L.Compartment.Apoplastic, kind=[WALL] * n_walls + [JUNCTION] * n_junctions))
    protoplasts = [written["protoplasts"][r] for r in valid_list]
    vid_of_node = np.asarray(list(apo_vids) + protoplasts, dtype=np.int64)
    properties = g.properties()

    def put(name, vids, values):
        properties.setdefault(name, {}).update(zip(vids, values))

    nan_w, nan_j, nan_c = np.full(n_walls, np.nan), np.full(n_junctions, np.nan), np.full(n_cells, np.nan)
    cgroup = [cells["cgroup"][r] or CGROUP_MAP.get(types[r], 4) for r in valid_list]
    nodes = vid_of_node.tolist()
    columns.write(g, {                                       # C1: one numeric column per quantity
        "section_x": np.concatenate((mid_x, xy[used, 0], cx[valid])),
        "section_y": np.concatenate((mid_y, xy[used, 1], cy[valid])),
        "wall_length": np.concatenate((lengths, np.zeros(n_junctions), np.zeros(n_cells))),     # extensive: 0
        "wall_thickness": np.concatenate((thickness, nan_j, nan_c)),
        "cell_area": np.concatenate((np.zeros(n_walls), np.zeros(n_junctions), areas)),        # extensive: 0
        "cgroup": np.concatenate((nan_w, nan_j, np.asarray(cgroup, dtype=float))),
        "protect_topology": np.concatenate((nan_w, nan_j, [float(bool(cells["protect"][r])) for r in valid_list])),
        "bridge": np.concatenate((nan_w, nan_j, np.zeros(n_cells))),
    }, nodes)
    put("mecha_index", vid_of_node.tolist(), range(n_apo + n_cells))

    order = np.concatenate([blk[0] for blk in blocks])
    rank = np.empty(order.size, dtype=np.int64)
    rank[np.argsort(order, kind="stable")] = np.arange(order.size)
    u, v = np.empty(order.size, dtype=np.int64), np.empty(order.size, dtype=np.int64)
    paths = np.empty(order.size, dtype=object)
    offset = 0
    for blk_order, blk_u, blk_v, attributes in blocks:
        at = rank[offset:offset + blk_order.size]
        u[at], v[at], paths[at] = blk_u, blk_v, attributes["path"]
        offset += blk_order.size
    labels = {path: getattr(L.Connection, PATH_LABELS[path]) for path in set(paths.tolist())}
    connections = np.asarray(g.add_components_bulk(
        s.anchors[s.Connection], order.size, topo_parents=[segment] * order.size,
        **_props(s.Connection, n_id_a=vid_of_node[u].tolist(), n_id_b=vid_of_node[v].tolist(), edge_u=u.tolist(),
                 edge_v=v.tolist(), label=[labels[p] for p in paths.tolist()])), dtype=np.int64)
    offset = 0
    for blk_order, _, _, attributes in blocks:
        vids = connections[rank[offset:offset + blk_order.size]].tolist()
        offset += blk_order.size
        k = len(vids)
        d_vec = np.asarray(attributes["d_vec"], dtype=float).reshape(k, 2)
        block = {name: np.full(k, np.nan) for name in columns.EDGE_COLUMNS}
        block["network_path"] = np.array([columns.PATH_CODES[p] for p in attributes["path"]], dtype=float)
        block["edge_dx"], block["edge_dy"] = d_vec[:, 0], d_vec[:, 1]
        for name, values in attributes.items():
            if name in columns.VIA:
                put("edge__" + name, vids, values)
            elif name not in ("path", "d_vec"):
                block[columns._EDGE_COLUMN[name]] = np.asarray(values, dtype=float)
        columns.write(g, block, vids)
    nodes = node_of_row[mem_r].tolist()
    wall_to_cells = {w: nodes[i:i + c] for w, (i, c) in enumerate(zip(starts.tolist(), counts.tolist()))}
    written.update(apoplast=list(apo_vids), connections=connections.tolist(), n_walls=n_walls,
                   n_junctions=n_junctions, n_cells=n_cells, wall_to_cells=wall_to_cells)
    return written


def _pore_links(pore_cells, air_cells, cell_node, cx, cy, polygons):
    """
    One air link from each stomatal pore to its sub-stomatal chamber, the protected air space nearest to it (the guard
    cells meet between them, so they share no wall; place_stomata creates them together, a few µm apart).
    """
    edges = []
    rows = sorted(air_cells)
    if not rows:
        return edges
    for r in pore_cells:
        distances = shapely.distance(polygons[r], polygons[rows])
        chamber = rows[int(np.argmin(distances))]
        dx, dy = float(cx[chamber] - cx[r]), float(cy[chamber] - cy[r])
        dist = float(np.hypot(dx, dy))
        edges.append((cell_node[r], cell_node[chamber], dict(path="air_link", length=dist, dist=dist,
                                                           d_vec=np.array([dx, dy]))))
    return edges


def _air_links(walls, wall_cells, junction_node, n_walls, used, protected_air_cells, cell_node, position, linked,
               lengths, polygons):
    """
    The exporter's phase 8: air spaces linked through the junctions their protected shapes induced (an air space,
    a new junction, an old junction, a new junction, an air space), then those within the median wall length of each
    other. Searched in the order GRANAP's sets and networkx's neighbours give.
    """
    air_nodes = {cell_node[r] for r in protected_air_cells}       # GRANAP's set, built in the same order
    signature = walls["signature"]
    j_start, j_end = junction_node[walls["j_start"]], junction_node[walls["j_end"]]
    new_junction_nodes = set()
    for w in np.flatnonzero(signature > 0).tolist():
        new_junction_nodes.add(int(j_start[w]))
        new_junction_nodes.add(int(j_end[w]))
    junction_nodes = set(range(n_walls, n_walls + used.size))
    old_junction_nodes = junction_nodes - new_junction_nodes

    def wall_neighbours(w):
        """networkx's neighbours of a wall node: its cells (membrane edges), then its junctions."""
        out = list(dict.fromkeys(cell_node[r] for r in wall_cells[w]))
        for j in (int(j_start[w]), int(j_end[w])):
            if j not in out:
                out.append(j)
        return out

    walls_of_junction = {}
    for w in range(n_walls):                                 # wall edges were created wall by wall
        for j in dict.fromkeys((int(j_start[w]), int(j_end[w]))):
            walls_of_junction.setdefault(j, []).append(w)
    walls_of_cell = {}
    for w in range(n_walls):
        for r in wall_cells[w]:
            walls_of_cell.setdefault(cell_node[r], []).append(w)
    air_to_new_junctions = {}
    for air in air_nodes:
        attached = set()
        for w in walls_of_cell.get(air, []):
            for j in wall_neighbours(w):
                if j in new_junction_nodes:
                    attached.add(j)
        air_to_new_junctions[air] = attached

    linked = set(linked)
    edges = []
    for air_a in air_nodes:
        for air_b in air_nodes:
            if air_a >= air_b or frozenset((air_a, air_b)) in linked:
                continue
            juncs_a, juncs_b = air_to_new_junctions.get(air_a, set()), air_to_new_junctions.get(air_b, set())
            if not juncs_a or not juncs_b:
                continue
            found = _motif(juncs_a, juncs_b, walls_of_junction, wall_neighbours, old_junction_nodes)
            if found is None:
                continue
            j1, old_j, j2 = found
            pa, pb = position[air_a], position[air_b]
            dist = np.hypot(pb[0] - pa[0], pb[1] - pa[1])
            edges.append((air_a, air_b, dict(path="air_link", length=dist, dist=dist,
                                             d_vec=np.array([pb[0] - pa[0], pb[1] - pa[1]]),
                                             via_new_junction_a=j1, via_old_junction=old_j, via_new_junction_b=j2)))
            linked.add(frozenset((air_a, air_b)))

    positive = [x for x in lengths.tolist() if x > 0]
    radius = float(np.median(np.asarray(lengths)[np.asarray(lengths) > 0])) if positive else 0.0
    rows = list(protected_air_cells)
    for i in range(len(rows)):
        air_a, geom_a = cell_node[rows[i]], polygons[rows[i]]
        for j in range(i + 1, len(rows)):
            air_b = cell_node[rows[j]]
            if frozenset((air_a, air_b)) in linked:
                continue
            if geom_a.distance(polygons[rows[j]]) > radius:
                continue
            pa, pb = position[air_a], position[air_b]
            dist = np.hypot(pb[0] - pa[0], pb[1] - pa[1])
            edges.append((air_a, air_b, dict(path="air_link", length=dist, dist=dist,
                                             d_vec=np.array([pb[0] - pa[0], pb[1] - pa[1]]))))
            linked.add(frozenset((air_a, air_b)))
    return edges


def _motif(juncs_a, juncs_b, walls_of_junction, wall_neighbours, old_junction_nodes):
    """The first (new junction, old junction, new junction) path between two air spaces, in the exporter's order."""
    for j1 in juncs_a:
        for wall_1 in walls_of_junction.get(j1, []):
            for old_j in wall_neighbours(wall_1):
                if old_j not in old_junction_nodes:
                    continue
                for wall_2 in walls_of_junction.get(old_j, []):
                    if wall_2 == wall_1:
                        continue
                    for j2 in wall_neighbours(wall_2):
                        if j2 != j1 and j2 in juncs_b:
                            return j1, old_j, j2
    return None
