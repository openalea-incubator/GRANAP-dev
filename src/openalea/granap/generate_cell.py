"""
Cell generator module for creating cells using Voronoi tessellation.
"""

import numpy as np
import shapely as sp
from scipy.spatial import Voronoi
from typing import List, Dict, Any, Tuple
from shapely.geometry import Polygon, Point, MultiPolygon, MultiPoint
from shapely.ops import unary_union
from shapely.strtree import STRtree
from scipy.spatial import cKDTree

from openalea.granap.geometry_collection import GeometryProcessor
from openalea.granap.cell_class import Cell
from openalea.granap.cell_manager import CellManager


class CellGenerator:
    """
    Generates plant cells using Voronoi tessellation.
    
    Handles cell placement on layers, border generation, and
    Voronoi diagram processing.
    """
    
    @staticmethod
    def cells_on_layer(layer_polygon: Polygon, cell_diameter: float,
                      cell_width: float = 0, shift: float = 0, rng=None) -> np.ndarray:
        """
        Generate cell center positions along a layer polygon.
        
        Args:
            layer_polygon: Polygon representing the layer boundary
            cell_diameter: Diameter of cells
            cell_width: Optional cell width (0 = use diameter)
            shift: Shift parameter (0-1)
        
        Returns:
            Array of (x, y) cell center coordinates
        """
        x, y = np.array(layer_polygon.exterior.coords.xy)
        perimeter = layer_polygon.length
        
        if cell_width == 0:
            cell_width = cell_diameter
        
        n_cells = int(np.ceil(perimeter / cell_width))
        
        # Calculate shift distance: shift of 1.0 = 1 cell width displacement
        # Randomized between 0 and shift * cell_width
        max_shift = shift * cell_width
        _rng = rng if rng is not None else np.random
        shift_distance = _rng.uniform(0, max_shift) if max_shift > 0 else 0
        
        cells_coords = GeometryProcessor.resample_coords(
            np.column_stack((x, y)), n_cells, shift_distance=shift_distance
        )
        return cells_coords
    
    @staticmethod
    def cell_border(cell_coords: np.ndarray, cell_height: float, 
                   cell_width: float = 0) -> List[np.ndarray]:
        """
        Generate border points for elliptical cells.
        
        Args:
            cell_coords: Array of cell center coordinates
            cell_height: Height of cells
            cell_width: Width of cells (0 = use height)
        
        Returns:
            List of arrays, each containing border points for one cell
        """
        if len(cell_coords) == 0:
            return []
        
        major_axis = cell_height
        minor_axis = cell_width if cell_width != 0 else cell_height

        n_points = 15 if cell_height != cell_width else 10

        # Vectorised orientation: next/prev neighbours via array roll
        next_coords = np.roll(cell_coords, -1, axis=0)
        prev_coords = np.roll(cell_coords,  1, axis=0)
        axes = np.arctan2(
            next_coords[:, 1] - prev_coords[:, 1],
            next_coords[:, 0] - prev_coords[:, 0],
        )

        cells_border = [
            GeometryProcessor.draw_ellipse(
                cell_coord, axis,
                major_axis / 2, minor_axis / 2,
                n_points=n_points,
            )
            for cell_coord, axis in zip(cell_coords, axes)
        ]
        return cells_border
    
    @staticmethod
    def generate_cells_info(layers_polygons: List[Dict[str, Any]],
                           center: Point, rng=None):
        """
        Generate cell information from layer polygons.
        
        Args:
            layers_polygons: List of layer polygon dictionaries
            center: Center point for angle/radius calculations
        
        Returns:
            pd.DataFrame of cells
        """
        all_cells = CellManager()
        id_cell = 1
        id_group = 1
        
        _rng = rng if rng is not None else np.random
        for i_layer, layer in enumerate(layers_polygons):
            cells_coords = CellGenerator.cells_on_layer(
                layer["polygon"],
                layer["cell_diameter"],
                layer["cell_width"],
                layer.get("shift", 0),
                rng=rng,
            )

            if layer.get("transfusion_type"):
                # Per-seed type assignment: each seed gets its own type and radius.
                # Orientation is computed from the global seed sequence so border
                # ellipses align with the ring, same as the standard path.
                # Border points are clipped to the layer polygon so that no
                # transfusion Voronoi seed escapes into the endodermis ring,
                # which would otherwise distort the endodermis shape.
                next_c = np.roll(cells_coords, -1, axis=0)
                prev_c = np.roll(cells_coords,  1, axis=0)
                seed_axes = np.arctan2(
                    next_c[:, 1] - prev_c[:, 1],
                    next_c[:, 0] - prev_c[:, 0],
                )
                tt_d = layer["tt_diameter"]
                tp_d = layer["tp_diameter"]
                p_tt = layer["p_tt"]
                layer_poly = layer["polygon"]
                sp.prepare(layer_poly)  # speeds the repeated covers() queries below

                for i, cell_coord in enumerate(cells_coords[1:]):
                    seed_type = (
                        "transfusion tracheid"
                        if _rng.random() < p_tt
                        else "transfusion parenchyma"
                    )
                    d = tt_d if seed_type == "transfusion tracheid" else tp_d
                    axis = seed_axes[i + 1]
                    border_pts = GeometryProcessor.draw_ellipse(
                        cell_coord, axis, d * 0.7 / 2, d * 0.7 / 2, n_points=10
                    )
                    cell_angle = np.arctan2(
                        cell_coord[1] - center.y, cell_coord[0] - center.x
                    )
                    cell_radius = np.sqrt(
                        (cell_coord[0] - center.x) ** 2
                        + (cell_coord[1] - center.y) ** 2
                    )
                    # Vectorised covers(): one C call for the whole ring of border
                    # points instead of constructing a shapely Point per point.
                    ring = border_pts[1:]
                    covered = sp.covers(layer_poly, sp.points(ring[:, 0], ring[:, 1]))
                    for border_point, is_covered in zip(ring, covered):
                        if not is_covered:
                            continue
                        new_cell = Cell(
                            type=seed_type,
                            x=border_point[0],
                            y=border_point[1],
                            diameter=d,
                            id_cell=id_cell,
                            id_layer=i_layer,
                            id_group=id_group,
                            angle=cell_angle,
                            radius=cell_radius,
                            area=np.pi * (d / 2) ** 2,
                        )
                        all_cells.add_cell(new_cell)
                        id_cell += 1
                    id_group += 1
                continue  # skip standard border generation for this layer

            if layer["cell_width"] != 0:
                layer_cell_borders = CellGenerator.cell_border(
                    cells_coords,
                    layer["cell_width"] * 0.7,
                    layer["cell_diameter"] * 0.7,
                )
            else:
                layer_cell_borders = CellGenerator.cell_border(
                    cells_coords,
                    layer["cell_diameter"] * 0.7,
                    layer["cell_width"] * 0.7,
                )

            # The next-inner layer polygon is loop-invariant — fetch and prepare
            # it once (prepared geometry speeds the per-group contains() queries).
            next_inner = (
                layers_polygons[i_layer + 1]["polygon"]
                if i_layer + 1 < len(layers_polygons)
                else None
            )
            if next_inner is not None:
                sp.prepare(next_inner)

            for i, cell_coord in enumerate(cells_coords[1:]):
                if layer["name"] == "parenchyma":
                    new_cell = Cell(
                        type=layer["name"],
                        x=cell_coord[0],
                        y=cell_coord[1],
                        diameter=layer["cell_diameter"],
                        id_cell=id_cell,
                        id_layer=i_layer,
                        id_group=id_group,
                        angle=np.arctan2(cell_coord[1] - center.y,
                                          cell_coord[0] - center.x),
                        radius=np.sqrt((cell_coord[0] - center.x)**2 +
                                        (cell_coord[1] - center.y)**2),
                        area=np.pi * (layer["cell_diameter"] / 2)**2,
                    )
                    all_cells.add_cell(new_cell)
                    id_cell += 1
                    id_group += 1
                else:
                    cell_border_points = layer_cell_borders[i]
                    ring = cell_border_points[1:]
                    # Discard border points that bleed into the next inner layer —
                    # they would create seeds inside the wrong tissue zone.  One
                    # vectorised contains_xy() call replaces a shapely Point per point.
                    if next_inner is not None and len(ring):
                        bleeds = sp.contains_xy(next_inner, ring[:, 0], ring[:, 1])
                    else:
                        bleeds = np.zeros(len(ring), dtype=bool)
                    for border_point, bleed in zip(ring, bleeds):
                        if bleed:
                            continue
                        new_cell = Cell(
                            type=layer["name"],
                            x=border_point[0],
                            y=border_point[1],
                            diameter=layer["cell_diameter"],
                            id_cell=id_cell,
                            id_layer=i_layer,
                            id_group=id_group,
                            angle=np.arctan2(cell_coord[1] - center.y,
                                              cell_coord[0] - center.x),
                            radius=np.sqrt((cell_coord[0] - center.x)**2 +
                                            (cell_coord[1] - center.y)**2),
                            area=np.pi * (layer["cell_diameter"] / 2)**2,
                        )
                        all_cells.add_cell(new_cell)
                        id_cell += 1
                    id_group += 1

        all_cells = CellGenerator.resolve_cell_border_overlaps(all_cells)
        return all_cells

    @staticmethod
    def resolve_cell_border_overlaps(all_cells: CellManager) -> CellManager:
        """
        Remove cell_border points from lower-priority id_groups that overlap
        with higher-priority ones.

        Priority order:
          1. Higher id_layer wins (inner layers have precedence over outer).
          2. Within the same id_layer, higher id_group wins.

        A convex-hull footprint is built from each group's cell positions.
        Cells from a lower-priority group whose position falls inside a
        higher-priority footprint are removed.
        """

        if not all_cells.cells:
            return all_cells

        # --- build group metadata ----------------------------------------
        groups: dict = {}  # id_group -> {id_layer, indices, poly}
        for idx, cell in enumerate(all_cells.cells):
            g = cell.id_group
            if g not in groups:
                groups[g] = {
                    "id_layer": cell.id_layer,
                    "cell_diameter": cell.diameter,
                    "id_group": g,
                    "indices": [],
                }
            groups[g]["indices"].append(idx)

        # build a convex-hull footprint for each group
        for meta in groups.values():
            pts = [
                (all_cells.cells[i].x, all_cells.cells[i].y)
                for i in meta["indices"]
            ]
            if len(pts) >= 3:
                meta["poly"] = MultiPoint(pts).convex_hull.buffer(meta["cell_diameter"] * 0.2)
            elif pts:
                r = all_cells.cells[meta["indices"][0]].diameter / 2
                meta["poly"] = Point(pts[0]).buffer(r)
            else:
                meta["poly"] = None

        # --- sort groups from highest to lowest priority -----------------
        sorted_groups = sorted(
            groups.values(),
            key=lambda m: (m["id_layer"], m["id_group"]),
            reverse=True,
        )

        # --- remove overlapping lower-priority cells --------------------
        # Build STRtree over all group footprints so we only run the
        # expensive contains() check for geometrically nearby pairs.
        valid_groups = [m for m in sorted_groups if m["poly"] is not None]
        if not valid_groups:
            return all_cells

        polys = [m["poly"] for m in valid_groups]
        tree = STRtree(polys)

        ids_to_remove: set = set()
        cells = all_cells.cells
        for i, high in enumerate(valid_groups):
            high_poly = polys[i]
            sp.prepare(high_poly)  # repeated contains() queries below
            # query returns indices whose bounding boxes overlap high_poly
            for j in tree.query(high_poly):
                if j <= i:          # only lower-priority groups
                    continue
                if not high_poly.intersects(polys[j]):
                    continue
                idxs = [idx for idx in valid_groups[j]["indices"]
                        if idx not in ids_to_remove]
                if not idxs:
                    continue
                # One vectorised contains_xy() for the whole low group instead of
                # constructing a shapely Point per candidate cell.
                xs = np.fromiter((cells[idx].x for idx in idxs), float, len(idxs))
                ys = np.fromiter((cells[idx].y for idx in idxs), float, len(idxs))
                inside = sp.contains_xy(high_poly, xs, ys)
                ids_to_remove.update(idx for idx, ins in zip(idxs, inside) if ins)

        if ids_to_remove:
            all_cells.cells = [
                c for i, c in enumerate(all_cells.cells)
                if i not in ids_to_remove
            ]

        return all_cells

    @staticmethod
    def voronoi_diagram(all_cells: CellManager, rng=None) -> Voronoi:
        cells = all_cells.cells
        n = len(cells)
        if n == 0:
            return Voronoi(np.empty((0, 2)))
        _rng = rng if rng is not None else np.random
        shift = 0.0001
        xs    = np.fromiter((c.x for c in cells),        float, n)
        ys    = np.fromiter((c.y for c in cells),        float, n)
        diams = np.fromiter((c.diameter for c in cells), float, n)
        draws = _rng.uniform(-shift, shift, size=(n, 2))
        xs = xs + draws[:, 0] * diams
        ys = ys + draws[:, 1] * diams
        angles = np.arctan2(ys, xs)
        radii  = np.sqrt(xs ** 2 + ys ** 2)
        for c, x, y, a, r in zip(cells, xs, ys, angles, radii):
            c.x, c.y, c.angle, c.radius = x, y, a, r
        return Voronoi(np.column_stack((xs, ys)))
    
    @staticmethod
    def process_voronoi_groups(all_cells: CellManager, 
                               vor: Voronoi) -> List[Cell]:
        """
        Process Voronoi diagram into grouped cell geometries.
        
        Args:
            all_cells: List of Cell objects
            vor: Voronoi diagram
        
        Returns:
            List of updated Cell objects with geometries
        """
        updated_cells = CellManager()
        
        for i, cell in enumerate(all_cells.cells):
            region_idx = vor.point_region[i]
            region_vertices_indices = vor.regions[region_idx]
            
            if -1 in region_vertices_indices or len(region_vertices_indices) == 0:
                cell.polygon = None
            else:
                vertices = vor.vertices[region_vertices_indices]
                # Voronoi regions are convex by construction, hence always valid,
                # so the per-polygon is_valid check + buffer(0) repair (~one call
                # per seed) was a no-op in the normal case — dropped (perf
                # proposal F1). The -1 / empty-region guard above already handles
                # unbounded / degenerate regions.
                cell.polygon = sp.Polygon(vertices)
            
            if cell.type != "outside" and cell.polygon is not None:
                updated_cells.add_cell(cell)
                
        # Union the per-seed Voronoi polygons into one 'biological' cell per
        # id_group. This replaces a GeoPandas ``dissolve(by="id_group")`` — at
        # ~500k seeds the GeoDataFrame build + pandas groupby + iterrows was
        # ~13s of pure overhead (see doc/performance_proposals.md (1)). Plain
        # Python grouping + shapely ``unary_union`` (what dissolve calls
        # internally) is byte-identical: dissolve's default aggregation is
        # 'first' per group, so the representative cell is the first-seen one;
        # groups are emitted in sorted id_group order to match dissolve.
        groups: Dict[Any, List] = {}
        rep: Dict[Any, Cell] = {}
        for c in updated_cells.cells:
            groups.setdefault(c.id_group, []).append(c.polygon)
            rep.setdefault(c.id_group, c)  # first-seen == dissolve's 'first'

        final_cells = CellManager()
        for gid in sorted(groups):
            poly = unary_union(groups[gid])
            r = rep[gid]
            final_cells.add_cell(Cell(
                type=r.type,
                x=r.x,
                y=r.y,
                diameter=r.diameter,
                id_cell=r.id_cell,
                id_layer=r.id_layer,
                id_group=gid,
                track_id=r.track_id,      # carry the tracked-vessel id through grouping
                angle=r.angle,
                radius=r.radius,
                area=poly.area,
                polygon=poly,
            ))

        return final_cells
    
    @staticmethod
    def _build_topology(
        polys: List,
        cell_ids: List[Any],
    ) -> Tuple[Dict[Any, List[tuple]], Dict[tuple, set], Dict[tuple, set], set]:
        """
        Build the shared vertex/edge topology for a collection of polygons.

        Runs KD-tree vertex snapping (Phase 0), then constructs
        ``cell_vkeys``, ``vertex_to_cells``, ``edge_to_cells`` (Phase 1),
        and finally identifies junction vertices (Phase 2).

        This helper is called by both :meth:`simplify_cells` and
        ``Organ._build_anatnetwork`` so the logic lives in one place.

        Args:
            polys:     Sequence of Shapely geometries (``None`` entries are
                       skipped).  Index position must correspond to
                       ``cell_ids``.
            cell_ids:  Opaque identifier for each polygon (list/GeoDataFrame
                       index, integer position, ...).

        Returns:
            ``(cell_vkeys, vertex_to_cells, edge_to_cells, junction_set)``

            * ``cell_vkeys``       - ``{cell_id: [snapped (x,y) tuples]}``
            * ``vertex_to_cells``  - ``{(x,y): set(cell_ids)}``
            * ``edge_to_cells``    - ``{edge_key: set(cell_ids)}``
            * ``junction_set``     - set of ``(x,y)`` junction vertices
        """
        n_dec = 6

        # ------------------------------------------------------------------
        # Phase 0 — collect raw vertices and snap nearby ones together
        # ------------------------------------------------------------------
        raw_cell_data: Dict[Any, list] = {}
        all_raw_verts: list = []
        vert_global_idx: Dict[Any, List[int]] = {}

        for cid, poly in zip(cell_ids, polys):
            if poly is None or poly.is_empty:
                continue
            if isinstance(poly, MultiPolygon):
                poly = max(poly.geoms, key=lambda g: g.area)
            coords = list(poly.exterior.coords)
            if coords[0] == coords[-1]:
                coords = coords[:-1]
            if len(coords) < 3:
                continue
            indices = []
            for x, y in coords:
                indices.append(len(all_raw_verts))
                all_raw_verts.append((x, y))
            raw_cell_data[cid] = coords
            vert_global_idx[cid] = indices

        if not all_raw_verts:
            return {}, {}, {}, set()

        coords_arr = np.array(all_raw_verts)
        kd_tree = cKDTree(coords_arr)

        # Snap tolerance: 1 % of 5th-percentile edge length.  Edge lengths are
        # computed vectorised per cell (one np.hypot over the whole ring) rather
        # than scalar-by-scalar in a Python loop; the resulting set is identical.
        edge_chunks = []
        for coords in raw_cell_data.values():
            pts = np.asarray(coords, dtype=float)
            diffs = pts - np.roll(pts, -1, axis=0)
            el = np.hypot(diffs[:, 0], diffs[:, 1])
            edge_chunks.append(el[el > 0])
        all_edges = (
            np.concatenate(edge_chunks) if edge_chunks else np.empty(0)
        )
        snap_tol = (
            np.percentile(all_edges, 5) * 0.01
            if all_edges.size
            else 1e-4
        )

        # Cluster nearby vertices -> canonical snapped coordinate.  All ball
        # queries are issued in one parallel C batch; the greedy single-pass
        # assignment below is byte-identical to querying point-by-point (each
        # seed's cluster is still exactly the points within snap_tol of it).
        neighbors = kd_tree.query_ball_point(coords_arr, snap_tol, workers=-1)
        canonical: List = [None] * len(all_raw_verts)
        visited_snap = [False] * len(all_raw_verts)
        for i in range(len(all_raw_verts)):
            if visited_snap[i]:
                continue
            cluster = neighbors[i]
            cx = float(np.mean(coords_arr[cluster, 0]))
            cy = float(np.mean(coords_arr[cluster, 1]))
            snapped = (round(cx, n_dec), round(cy, n_dec))
            for ci in cluster:
                visited_snap[ci] = True
                canonical[ci] = snapped

        # ------------------------------------------------------------------
        # Phase 1 — build cell_vkeys, vertex_to_cells, edge_to_cells
        # ------------------------------------------------------------------
        vertex_to_cells: Dict[tuple, set] = {}
        cell_vkeys: Dict[Any, List[tuple]] = {}

        for cid, gidxs in vert_global_idx.items():
            vkeys_raw = [canonical[gi] for gi in gidxs]
            vkeys: List[tuple] = [vkeys_raw[0]]
            for vk in vkeys_raw[1:]:
                if vk != vkeys[-1]:
                    vkeys.append(vk)
            if len(vkeys) > 1 and vkeys[-1] == vkeys[0]:
                vkeys = vkeys[:-1]
            if len(vkeys) < 3:
                continue
            cell_vkeys[cid] = vkeys
            for vk in vkeys:
                vertex_to_cells.setdefault(vk, set()).add(cid)

        edge_to_cells: Dict[tuple, set] = {}
        for cid, vkeys in cell_vkeys.items():
            n = len(vkeys)
            for i in range(n):
                ek = tuple(sorted((vkeys[i], vkeys[(i + 1) % n])))
                edge_to_cells.setdefault(ek, set()).add(cid)

        # ------------------------------------------------------------------
        # Phase 2 — identify junction vertices
        # ------------------------------------------------------------------
        junction_set: set = set()

        for vk in vertex_to_cells:
            if len(vertex_to_cells[vk]) >= 3:
                junction_set.add(vk)
                continue
            incident_pairs: set = set()
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

        return cell_vkeys, vertex_to_cells, edge_to_cells, junction_set

    @staticmethod
    def remove_nested_cells(grouped_cells: List[Cell], min_overlap: float = 0.1) -> List[Cell]:
        """Drop cells that sit (almost) entirely inside another cell, keeping the
        bigger one.

        Voronoi grouping can leave one group's cell overlapping and buried inside a
        larger neighbour — e.g. a companion seeded right against a sieve, whose few
        border points give it a Voronoi footprint the bigger cell's union swallows.
        It then renders as one cell drawn inside another.

        A cell ``i`` is treated as nested when a larger cell ``j`` contains ``i``'s
        interior point *and* covers at least ``min_overlap`` of ``i``'s area — so
        ordinary edge-sharing neighbours (whose overlap is ~0) are never touched.
        The nested cell is removed and merged into the smallest such enclosing cell
        (a no-op when it is already fully covered), keeping the larger cell intact.
        """
        valid = [c for c in grouped_cells
                 if c.polygon is not None and not c.polygon.is_empty]
        if len(valid) < 2:
            return grouped_cells

        polys = [c.polygon for c in valid]
        areas = [p.area for p in polys]
        tree  = STRtree(polys)

        # An interior point of a Voronoi-partitioned cell only falls inside another
        # cell when the two genuinely overlap (nesting), so querying by that point
        # keeps the (rare) candidate set tiny; the area check then confirms it.
        parent = [None] * len(valid)
        for i, poly in enumerate(polys):
            pt = poly.representative_point()
            best, best_area = None, None
            for pos in tree.query(pt):
                j = int(pos)
                if j == i or areas[j] <= areas[i]:
                    continue
                if polys[j].contains(pt) and \
                        poly.intersection(polys[j]).area >= min_overlap * areas[i]:
                    if best is None or areas[j] < best_area:
                        best, best_area = j, areas[j]
            parent[i] = best

        if not any(p is not None for p in parent):
            return grouped_cells

        def _ancestor(i):
            while parent[i] is not None:
                i = parent[i]
            return i

        merge_into: Dict[int, list] = {}
        removed = set()
        for i in range(len(valid)):
            if parent[i] is not None:
                removed.add(i)
                merge_into.setdefault(_ancestor(i), []).append(polys[i])

        result = [c for c in grouped_cells if c.polygon is None or c.polygon.is_empty]
        for i, c in enumerate(valid):
            if i in removed:
                continue
            if i in merge_into:
                c.polygon = unary_union([c.polygon, *merge_into[i]])
                c.area = c.polygon.area
            result.append(c)
        return result

    @staticmethod
    def simplify_cells(grouped_cells: List[Cell]) -> List[Cell]:
        """
        Simplify cell boundaries by retaining only junction vertices.

        Delegates topology computation to :meth:`_build_topology` (Phases
        0-2: KD-tree snapping, vertex/edge maps, junction detection), then
        rebuilds each polygon keeping only its junction vertices (Phase 3).

        Args:
            grouped_cells: List of Cell objects with polygon geometries.

        Returns:
            The same list with simplified polygon geometries in place.
        """
        polys = [c.polygon for c in grouped_cells]
        cell_ids = list(range(len(grouped_cells)))

        cell_vkeys, _, _, junction_set = CellGenerator._build_topology(
            polys, cell_ids
        )

        if not cell_vkeys:
            return grouped_cells

        # Phase 3 — rebuild each polygon keeping only junction vertices
        for idx, cell in enumerate(grouped_cells):
            if idx not in cell_vkeys:
                continue

            vkeys = cell_vkeys[idx]
            simplified = [vk for vk in vkeys if vk in junction_set]

            if len(simplified) < 3:
                simplified = vkeys

            ring_coords = list(simplified)
            if ring_coords[0] != ring_coords[-1]:
                ring_coords.append(ring_coords[0])

            new_poly = Polygon(ring_coords)
            if not new_poly.is_valid:
                new_poly = new_poly.buffer(0)
            cell.polygon = new_poly

        return grouped_cells

    @staticmethod
    def create_stomata(cells, stomata_setting, debug= False):
        """
        Create stomata on a cell.

        Args:
            cells: triplet of Cell object.
            stomata_setting: Dictionary with stomata settings.
            debug: Whether to plot the stomata.
        """

        width = stomata_setting["width"]
        depth = stomata_setting["depth"]
        sub_chamber = stomata_setting["sub_chamber"]

        # get unique id_group of the cells
        id_groups = [cell.id_group for cell in cells]
        id_groups = np.unique(id_groups)
        cell = cells[0] # template cell

        triplet = CellManager()
        triplet.cells = cells
        cell_prev_cx, cell_prev_cy = triplet.get_centroid_of_group(id_groups[0])
        cx, cy = triplet.get_centroid_of_group(id_groups[1])
        cell_next_cx, cell_next_cy = triplet.get_centroid_of_group(id_groups[2])

        # use axis of the cell triplet as the orientation
        dx = cell_next_cx - cell_prev_cx
        dy = cell_next_cy - cell_prev_cy
        tangent_angle = np.arctan2(dy, dx)
        angle = tangent_angle + np.pi/2 # perpendicular (inward) orientation

    
        def local_to_global_poly(local_pts):
            global_pts = []
            tangential_angle = angle + np.pi/2
            inward_angle = angle + np.pi

            for lx, ly in local_pts:
                # - 0.4*cell.height*np.cos(tangential_angle) 
                # - 0.4*cell.height*np.cos(inward_angle)
                gx = cx + lx * np.cos(tangential_angle) - 0.4*cell.height*np.cos(tangential_angle) + ly * np.cos(inward_angle) - 0.4*cell.height*np.cos(inward_angle)
                gy = cy + lx * np.sin(tangential_angle) - 0.4*cell.height*np.sin(tangential_angle) + ly * np.sin(inward_angle) - 0.4*cell.height*np.sin(inward_angle)
                global_pts.append((gx, gy))
            return Polygon(global_pts)
    
        def create_local_ellipse(cx_l, cy_l, rx, ry):
            pts = []
            for t in np.linspace(0, 2*np.pi, 30):
                pts.append((cx_l + rx * np.cos(t), cy_l + ry * np.sin(t)))
            return local_to_global_poly(pts)
    
        def create_local_rectangle(cx_l, cy_l, w, h):
            pts = [
                (cx_l - w/2, cy_l - h/2),
                (cx_l + w/2, cy_l - h/2),
                (cx_l + w/2, cy_l + h/2),
                (cx_l - w/2, cy_l + h/2)
            ]
            return local_to_global_poly(pts)
    
        # Create guard cells
        gc_rx = cell.width / 2
        gc_ry = cell.width / 2
        gc1_x = -width / 2
        gc2_x = width / 2
        gc_y = depth
    
        guard_cell_1_ellipse = create_local_ellipse(gc1_x, gc_y, gc_rx, gc_ry/2)
        guard_cell_2_ellipse = create_local_ellipse(gc2_x, gc_y, gc_rx, gc_ry/2)
    
        rect_w = cell.width * 0.6
        rect_h = depth
        rect_y = depth / 2
    
        guard_cell_1_rect = create_local_rectangle(gc1_x - 0.2 * cell.width, rect_y, rect_w, rect_h)
        guard_cell_2_rect = create_local_rectangle(gc2_x + 0.2 * cell.width, rect_y, rect_w, rect_h)
    
        guard_cell_1_poly = unary_union([guard_cell_1_ellipse, guard_cell_1_rect])
        guard_cell_2_poly = unary_union([guard_cell_2_ellipse, guard_cell_2_rect])
    
        guard_cell_1_poly = GeometryProcessor.buffer_polygon(guard_cell_1_poly, 0, 0.5)
        guard_cell_2_poly = GeometryProcessor.buffer_polygon(guard_cell_2_poly, 0, 0.5)
    
        # Create sub-stomatal chamber
        chamber_rx = width
        chamber_ry = sub_chamber
        chamber_y = gc_y
        sub_stomatal_chamber = create_local_ellipse(0, chamber_y, chamber_rx * 0.75, chamber_ry)
    
        # Create pore
        pore_w = width
        if pore_w < 0:
            pore_w = 0.005  # fallback
        pore_h = chamber_y
        pore_poly = create_local_rectangle(0, pore_h / 2, pore_w, pore_h)
    
        # Combine geometries
        spacing_poly = pore_poly.difference(unary_union([guard_cell_1_poly, guard_cell_2_poly]))
        sub_stomatal_chamber = sub_stomatal_chamber.difference(unary_union([spacing_poly, guard_cell_1_poly, guard_cell_2_poly]))
    
        if hasattr(sub_stomatal_chamber, 'geoms'):
            sub_stomatal_chamber = sub_stomatal_chamber.geoms[0]
    
        carve_poly = unary_union([guard_cell_1_poly, guard_cell_2_poly, sub_stomatal_chamber, spacing_poly])
    
        if debug:
            print(carve_poly.area)

        return carve_poly, guard_cell_1_poly, guard_cell_2_poly, sub_stomatal_chamber, spacing_poly

    
    
