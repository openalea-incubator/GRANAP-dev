"""
The section's topology, converted (devplan_metafspm.md, §37, B5): GRANAP's CellGenerator._build_topology
(generate_cell.py:479), the vertex snapping (phase 0), the vertex and edge maps (phase 1) and the junctions (phase 2),
with the same results, down to the insertion order of every dict and set it returns:

  * the rings and their edge lengths read in bulk (shapely.get_coordinates, the same coordinates);
  * each cluster's mean summed as numpy does (in order below 8 values, numpy's own mean above), in plain floats
    (an explicit loop: Python's sum() of floats is compensated since 3.12, not in order);
  * an edge key ordered by one comparison instead of sorted(), as sorted() orders two items;
  * phase 2 reads each vertex's places in its rings from a map built in phase 1, instead of scanning every ring of
    the cells around it for the vertex (quadratic in the ring size).

Every other line is GRANAP's.
"""
from typing import Any, Dict, List, Optional

import numpy as np
import shapely
from scipy.spatial import cKDTree
from shapely.geometry import MultiPolygon

_EXACT_MEAN = 8                  # numpy's mean is a sequential sum below 8 values


def _rings(polys, cell_ids):
    """GRANAP's phase-0 input: per kept cell, its exterior ring's coordinates, closing vertex dropped."""
    kept_ids, shapes = [], []
    for cid, poly in zip(cell_ids, polys):
        if poly is None or poly.is_empty:
            continue
        if isinstance(poly, MultiPolygon):
            poly = max(poly.geoms, key=lambda g: g.area)
        kept_ids.append(cid)
        shapes.append(poly)
    if not shapes:
        return [], []
    coords, index = shapely.get_coordinates(shapely.get_exterior_ring(np.asarray(shapes, dtype=object)),
                                            return_index=True)
    bounds = np.searchsorted(index, np.arange(len(shapes) + 1))
    rings, ids = [], []
    for k, cid in enumerate(kept_ids):
        ring = coords[bounds[k]:bounds[k + 1]]
        if len(ring) and ring[0, 0] == ring[-1, 0] and ring[0, 1] == ring[-1, 1]:
            ring = ring[:-1]
        if len(ring) < 3:
            continue
        rings.append(ring)
        ids.append(cid)
    return ids, rings


def build_topology(polys: List, cell_ids: List[Any], protect_ids: Optional[set] = None):
    """GRANAP's _build_topology: (cell_vkeys, vertex_to_cells, edge_to_cells, junction_set, protected_shape_set)."""
    n_dec = 6

    # Phase 0 — collect raw vertices and snap nearby ones together
    ids, rings = _rings(polys, cell_ids)
    if not rings:
        return {}, {}, {}, set(), set()
    sizes = np.array([len(r) for r in rings], dtype=np.int64)
    coords_arr = np.concatenate(rings)
    starts = np.cumsum(sizes) - sizes
    kd_tree = cKDTree(coords_arr)

    # Snap tolerance: 1 % of 5th-percentile edge length (the edges in GRANAP's order)
    following = np.arange(coords_arr.shape[0]) + 1
    following[starts + sizes - 1] = starts
    diffs = coords_arr - coords_arr[following]
    all_edges = np.hypot(diffs[:, 0], diffs[:, 1])
    all_edges = all_edges[all_edges > 0]
    snap_tol = np.percentile(all_edges, 5) * 0.01 if all_edges.size else 1e-4

    neighbors = kd_tree.query_ball_point(coords_arr, snap_tol, workers=-1)
    xs, ys = coords_arr[:, 0].tolist(), coords_arr[:, 1].tolist()
    n_raw = len(xs)
    canonical: List = [None] * n_raw
    visited_snap = [False] * n_raw
    for i in range(n_raw):
        if visited_snap[i]:
            continue
        cluster = neighbors[i]
        n = len(cluster)
        if n < _EXACT_MEAN:                      # not sum(): Python's is compensated for floats (3.12+)
            cx = cy = 0.
            for c in cluster:
                cx += xs[c]
                cy += ys[c]
            cx, cy = cx / n, cy / n
        else:
            cx, cy = float(np.mean(coords_arr[cluster, 0])), float(np.mean(coords_arr[cluster, 1]))
        snapped = (round(cx, n_dec), round(cy, n_dec))
        for ci in cluster:
            visited_snap[ci] = True
            canonical[ci] = snapped

    # Phase 1 — build cell_vkeys, vertex_to_cells, edge_to_cells
    vertex_to_cells: Dict[tuple, set] = {}
    cell_vkeys: Dict[Any, List[tuple]] = {}
    places: Dict[tuple, list] = {}                     # vertex -> [(cell, position in its ring)]
    for cid, start, size in zip(ids, starts.tolist(), sizes.tolist()):
        vkeys_raw = canonical[start:start + size]
        vkeys: List[tuple] = [vkeys_raw[0]]
        for vk in vkeys_raw[1:]:
            if vk != vkeys[-1]:
                vkeys.append(vk)
        if len(vkeys) > 1 and vkeys[-1] == vkeys[0]:
            vkeys = vkeys[:-1]
        if len(vkeys) < 3:
            continue
        cell_vkeys[cid] = vkeys
        for i, vk in enumerate(vkeys):
            vertex_to_cells.setdefault(vk, set()).add(cid)
            places.setdefault(vk, []).append((cid, i))

    def edge_key(a, b):
        return (b, a) if b < a else (a, b)              # tuple(sorted((a, b)))

    edge_to_cells: Dict[tuple, set] = {}
    for cid, vkeys in cell_vkeys.items():
        n = len(vkeys)
        for i in range(n):
            edge_to_cells.setdefault(edge_key(vkeys[i], vkeys[(i + 1) % n]), set()).add(cid)

    # Phase 2 — identify junction vertices
    junction_set: set = set()
    for vk, owners in vertex_to_cells.items():
        if len(owners) >= 3:
            junction_set.add(vk)
            continue
        incident_pairs: set = set()
        for cid, i in places[vk]:
            vks = cell_vkeys[cid]
            n = len(vks)
            ek_prev = edge_key(vks[(i - 1) % n], vk)
            ek_next = edge_key(vk, vks[(i + 1) % n])
            if ek_prev in edge_to_cells:
                incident_pairs.add(frozenset(edge_to_cells[ek_prev]))
            if ek_next in edge_to_cells:
                incident_pairs.add(frozenset(edge_to_cells[ek_next]))
        if len(incident_pairs) > 1:
            junction_set.add(vk)

    # keep track of the vertices that are protected
    protected_shape_set: set = set()
    if protect_ids:
        for cid in protect_ids:
            for vk in cell_vkeys.get(cid, ()):  # skip ids dropped in Phase 1
                protected_shape_set.add(vk)

    return cell_vkeys, vertex_to_cells, edge_to_cells, junction_set, protected_shape_set
