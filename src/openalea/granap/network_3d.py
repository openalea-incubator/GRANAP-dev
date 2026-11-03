"""
3D hydraulic network of an extruded segment (see doc/NETWORK_3D_PLAN.md).

The 3D segment (:func:`generate_cell_3d.generate_cells_3d`) stacks rows of each
2D cell along Z, every row reusing the SAME 2D section.  So every 3D wall is a
2D wall times a height interval, and the 3D network is *lifted* from the 2D
walls and junctions (:func:`anatomy_writer.build_wall_registry`) by interval
arithmetic on the rows -- exact, no 3D geometry matching.

Node indexing follows the MECHA convention of the 2D network
(:mod:`network_base`): walls, then junctions, then cells.

Walls (``kind``):
  - ``side``  -- a 2D wall over the height where the rows of its two cells (or
    its one cell, on the organ surface) overlap; ``orientation="lateral"``,
    area = 2D length x height.
  - ``end``   -- between two consecutive rows of one 2D cell (or two vessel
    elements); ``orientation="axial"``, area = the cell's area.
  - ``face``  -- the bottom of the first row / the top of the last row, where
    the segment meets the next one; ``boundary=True``.

Junctions (``kind``):
  - ``vertical`` -- a 2D junction over the height between two consecutive row
    boundaries of any cell around it (where side walls meet);
  - ``rim``      -- a 2D wall at a row boundary of one of its cells (where an end
    wall or a face meets the side walls).

Edges use the 2D ``path`` names -- ``membrane`` (cell - wall), ``wall`` (wall -
junction), ``plasmodesmata`` (cell - cell) -- plus ``orientation`` and
``area``, so ``fill_matrix(K, label, "cortex-endodermis")`` keeps its meaning.

Wall thickness, as in 2D: a side wall is the sum of ``cell_wall_thickness`` of
the cells on each side, plus ``"outerwall"`` on the surface; an end wall the
sum of ``end_wall_thickness`` of the rows below and above; a face the
``end_wall_thickness`` of its one row (the other half belongs to the next
segment).  Types missing from a dict take its ``"default"`` (logged).
"""

import logging
from bisect import bisect_right
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from openalea.granap.network_base import AbstractNetwork

log = logging.getLogger(__name__)

Thickness = Union[float, Dict[str, float]]


def _merge_breaks(values, tol: float) -> List[float]:
    """Sorted ``values`` with any value within ``tol`` of the previous kept one dropped."""
    out: List[float] = []
    for v in sorted(values):
        if not out or v - out[-1] > tol:
            out.append(v)
    return out


class Network3D(AbstractNetwork):
    """The hydraulic network of a :class:`Cells3DResult`.

    Args:
        result: a 3D segment from ``organ.generate_cells_3d(...)`` (needs its
            ``section`` and the per-cell ``cell_2d``/``row``/``z0``/``z1``).
        cell_wall_thickness: per-type thickness of the side walls (the 2D
            ``DEFAULT_CELL_WALL_THICKNESS`` by default), or one value for all.
        end_wall_thickness: per-type thickness of the end walls and faces;
            ``None`` = the same as ``cell_wall_thickness``.
    """

    # Read by consumers (MECHA) to switch to their 3D path; a 2D network has none.
    dimension = 3

    def __init__(self, result, cell_wall_thickness: Optional[Thickness] = None,
                 end_wall_thickness: Optional[Thickness] = None):
        super().__init__()
        self._cells_table = None
        from openalea.granap.anatomy_writer import DEFAULT_CELL_WALL_THICKNESS
        if result.section is None:
            raise ValueError("Network3D needs a Cells3DResult built with its 2D section "
                             "(organ.generate_cells_3d).")
        self.result = result
        self.cell_wall_thickness = (DEFAULT_CELL_WALL_THICKNESS if cell_wall_thickness is None
                                    else cell_wall_thickness)
        self.end_wall_thickness = (self.cell_wall_thickness if end_wall_thickness is None
                                   else end_wall_thickness)
        self._build_anatnetwork()

    # ------------------------------------------------------------------
    def _build_anatnetwork(self, air_link_radius: Optional[float] = None) -> None:
        """Populate ``self.graph`` (``air_link_radius`` is unused: no air links in 3D)."""
        from openalea.granap.anatomy_writer import build_wall_registry

        res = self.result
        cells3d = res.cells
        reg = build_wall_registry(res.section, self.cell_wall_thickness)
        if reg is None or not cells3d:
            return
        gdf = reg.cells_gdf
        tol = 1e-9 * max(res.z_max - res.z_min, 1e-12)
        missing = set()

        def thick(table: Thickness, c_type: str) -> float:
            if isinstance(table, dict):
                if c_type not in table:
                    missing.add(c_type)
                return float(table.get(c_type, table.get("default", 1)))
            return float(table)

        # ---- the rows of each 2D cell, bottom to top -------------------
        rows: Dict[int, List[int]] = {}
        for k, c in enumerate(cells3d):
            rows.setdefault(c["cell_2d"], []).append(k)
        for c2d in rows:
            rows[c2d].sort(key=lambda k: cells3d[k]["z0"])
        row_starts = {c2d: [cells3d[k]["z0"] for k in ks] for c2d, ks in rows.items()}
        breaks = {c2d: _merge_breaks([cells3d[k]["z0"] for k in ks] + [cells3d[ks[-1]]["z1"]], tol)
                  for c2d, ks in rows.items()}

        def row_at(c2d: int, z: float) -> int:
            """The 3D cell of 2D cell ``c2d`` whose row holds height ``z``."""
            ks = rows[c2d]
            return ks[max(0, bisect_right(row_starts[c2d], z) - 1)]

        walls2d = [w for w in reg.walls.values() if all(c in rows for c in w["cells"])]
        walls_of_cell: Dict[int, List[int]] = {}
        walls_at_junction: Dict[tuple, List[int]] = {}
        for wi, w in enumerate(walls2d):
            for c in w["cells"]:
                walls_of_cell.setdefault(c, []).append(wi)
            for j in {w["junc_start"], w["junc_end"]}:
                walls_at_junction.setdefault(j, []).append(wi)

        centroid = {c2d: gdf.loc[c2d, "geometry"].centroid for c2d in rows}
        area2d = {c2d: cells3d[ks[0]]["volume"] / (cells3d[ks[0]]["z1"] - cells3d[ks[0]]["z0"])
                  for c2d, ks in rows.items()}
        perimeter2d = {c2d: gdf.loc[c2d, "geometry"].length for c2d in rows}

        # ---- walls ------------------------------------------------------
        wall_nodes: List[dict] = []      # attributes, in id order
        wall_cells: List[List[int]] = []  # 3D cell indices on each side
        wall_breaks: List[List[float]] = []  # per 2D wall: its breakpoints

        # side pieces
        side_of: Dict[int, List[Tuple[float, float, int]]] = {}  # 2D wall -> [(z0, z1, wall id)]
        for wi, w in enumerate(walls2d):
            zs = _merge_breaks([z for c in w["cells"] for z in breaks[c]], tol)
            wall_breaks.append(zs)
            mx, my = w["midpoint"]
            for z0, z1 in zip(zs, zs[1:]):
                zm = 0.5 * (z0 + z1)
                ks = [row_at(c, zm) for c in w["cells"]]
                t = sum(thick(self.cell_wall_thickness, cells3d[k]["type"]) for k in ks)
                if len(ks) == 1:
                    t += thick(self.cell_wall_thickness, "outerwall")
                wid = len(wall_nodes)
                wall_nodes.append(dict(kind="side", orientation="lateral",
                                       position=(mx, my, zm), length=w["length"],
                                       height=z1 - z0, area=w["length"] * (z1 - z0),
                                       wall_thickness=t, z0=z0, z1=z1, wall_2d=w["id"]))
                wall_cells.append(ks)
                side_of.setdefault(wi, []).append((z0, z1, wid))

        # end walls and faces: (2D cell, z) -> wall id
        axial_at: Dict[Tuple[int, float], int] = {}
        for c2d, zs in breaks.items():
            cx, cy = centroid[c2d].x, centroid[c2d].y
            for i, z in enumerate(zs):
                below = row_at(c2d, z - tol) if i > 0 else None
                above = row_at(c2d, z + tol) if i < len(zs) - 1 else None
                ks = [k for k in (below, above) if k is not None]
                t = sum(thick(self.end_wall_thickness, cells3d[k]["type"]) for k in ks)
                face = len(ks) == 1
                wid = len(wall_nodes)
                wall_nodes.append(dict(kind="face" if face else "end", orientation="axial",
                                       position=(cx, cy, z), length=perimeter2d[c2d],
                                       area=area2d[c2d], wall_thickness=t, z=z,
                                       cell_2d=c2d, boundary=face))
                wall_cells.append(ks)
                axial_at[(c2d, z)] = wid

        # ---- junctions --------------------------------------------------
        junc_nodes: List[dict] = []
        vertical_of: Dict[tuple, List[Tuple[float, float, int]]] = {}  # 2D junction -> pieces
        for j, wis in walls_at_junction.items():
            cells_here = {c for wi in wis for c in walls2d[wi]["cells"]}
            zs = _merge_breaks([z for c in cells_here for z in breaks[c]], tol)
            for z0, z1 in zip(zs, zs[1:]):
                jid = len(junc_nodes)
                junc_nodes.append(dict(kind="vertical", position=(j[0], j[1], 0.5 * (z0 + z1)),
                                       length=z1 - z0, z0=z0, z1=z1,
                                       junction_2d=reg.junction_ids[j]))
                vertical_of.setdefault(j, []).append((z0, z1, jid))
        rim_at: Dict[Tuple[int, float], int] = {}  # (2D wall index, z) -> junction id
        for wi, zs in enumerate(wall_breaks):
            w = walls2d[wi]
            mx, my = w["midpoint"]
            for z in zs:
                rim_at[(wi, z)] = len(junc_nodes)
                junc_nodes.append(dict(kind="rim", position=(mx, my, z), length=w["length"],
                                       z=z, wall_2d=w["id"]))

        # ---- node ids ---------------------------------------------------
        n_w, n_j = len(wall_nodes), len(junc_nodes)
        self.n_walls, self.n_junctions, self.n_cells = n_w, n_j, len(cells3d)
        jnode = lambda jid: n_w + jid
        cnode = lambda k: n_w + n_j + k

        g = self.graph
        for wid, a in enumerate(wall_nodes):
            g.add_node(wid, indice=wid, type="apo", **a)
        for jid, a in enumerate(junc_nodes):
            g.add_node(jnode(jid), indice=jnode(jid), type="apo", **a)
        for k, c in enumerate(cells3d):
            c2d = c["cell_2d"]
            row = gdf.loc[c2d]
            g.add_node(cnode(k), indice=cnode(k), type="cell",
                       cgroup=row.get("cgroup", ""), cell_type=c["type"],
                       position=(centroid[c2d].x, centroid[c2d].y, 0.5 * (c["z0"] + c["z1"])),
                       area=area2d[c2d], volume=c["volume"], height=c["z1"] - c["z0"],
                       z0=c["z0"], z1=c["z1"], cell_2d=c2d, row=c["row"])

        self._wall_to_cells = {wid: [cnode(k) for k in ks] for wid, ks in enumerate(wall_cells)}

        def vec(a, b):
            d = np.asarray(g.nodes[b]["position"], float) - np.asarray(g.nodes[a]["position"], float)
            return d, float(np.linalg.norm(d))

        def link_wall(wid: int, jid: int, contact: float) -> None:
            d, dist = vec(wid, jnode(jid))
            g.add_edge(jnode(jid), wid, path="wall", length=contact, dist=dist, d_vec=d,
                       wall_thickness=wall_nodes[wid]["wall_thickness"],
                       orientation=wall_nodes[wid]["orientation"])

        # ---- edges: apoplast (wall - junction) --------------------------
        for wi, pieces in side_of.items():
            w = walls2d[wi]
            for z0, z1, wid in pieces:
                for j in {w["junc_start"], w["junc_end"]}:
                    for jz0, jz1, jid in vertical_of[j]:
                        if jz0 >= z0 - tol and jz1 <= z1 + tol:
                            link_wall(wid, jid, jz1 - jz0)
                for z in (z0, z1):
                    link_wall(wid, rim_at[(wi, z)], w["length"])
        def rim_near(wi: int, z: float) -> Optional[int]:
            """The rim of 2D wall ``wi`` at height ``z`` -- the wall's breakpoint
            within ``tol`` of it (merging may have kept a neighbour's value)."""
            zs = wall_breaks[wi]
            i = bisect_right(zs, z + tol) - 1
            return rim_at[(wi, zs[i])] if i >= 0 and abs(zs[i] - z) <= tol else None

        for (c2d, z), wid in axial_at.items():
            for wi in walls_of_cell.get(c2d, []):
                rim = rim_near(wi, z)
                if rim is not None:
                    link_wall(wid, rim, walls2d[wi]["length"])

        # ---- edges: membrane (cell - wall) and plasmodesmata (cell - cell)
        for wid, ks in enumerate(wall_cells):
            a = wall_nodes[wid]
            for k in ks:
                d, dist = vec(cnode(k), wid)
                g.add_edge(cnode(k), wid, path="membrane", length=a["length"], area=a["area"],
                           dist=dist, d_vec=d, wall_thickness=a["wall_thickness"],
                           orientation=a["orientation"])
            if len(ks) == 2:
                u, v = cnode(ks[0]), cnode(ks[1])
                d, dist = vec(u, v)
                if g.has_edge(u, v):        # two 2D walls between the same pair of rows
                    e = g.edges[u, v]
                    e["area"] += a["area"]
                    e["length"] += a["length"]
                else:
                    g.add_edge(u, v, path="plasmodesmata", length=a["length"], area=a["area"],
                               dist=dist, d_vec=d, orientation=a["orientation"])

        if missing:
            log.info("Network3D: no wall thickness for %s; using the 'default' entry.",
                     ", ".join(sorted(missing)))

    # ------------------------------------------------------------------
    @property
    def _cells_gdf(self):
        """One row per 3D cell, in cell-node order (row k = node
        ``n_walls + n_junctions + k``): its 2D cell's polygon and centroid
        (``geometry``, ``x``, ``y``, ``area``), its own ``type``, and
        ``id_cell`` = k, ``cell_2d``, ``row``, ``z0``, ``z1``.  The same table
        the 2D organ exposes, so consumers that read it work on both."""
        if self._cells_table is None:
            import geopandas as gpd
            section = self.result.section
            records = []
            for k, c in enumerate(self.result.cells):
                geom = section.loc[c["cell_2d"], "geometry"]
                centre = geom.centroid
                records.append(dict(id_cell=k, type=c["type"], x=centre.x, y=centre.y,
                                    area=geom.area, cell_2d=c["cell_2d"], row=c["row"],
                                    z0=c["z0"], z1=c["z1"], geometry=geom))
            self._cells_table = gpd.GeoDataFrame(records, geometry="geometry")
        return self._cells_table

    def nodes_of(self, kind: str) -> List[int]:
        """Node ids of one ``kind``: side / end / face (walls), vertical / rim
        (junctions)."""
        return [n for n, k in self.graph.nodes(data="kind") if k == kind]
