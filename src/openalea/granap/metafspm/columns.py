"""
The section's hydraulic network as numeric variables (devplan_metafspm.md, §42, C1): one column per quantity, on every
Compartment and every anatomy Connection: where a quantity does not apply, 0 for the extensive ones (wall_length,
cell_area: their sums over a segment stay defined), NaN for the others. These are the MPG properties a
DataStructure reads as declared variables (numeric, present on every node or edge); GRANAP's networkx graph is
rebuilt from them (views.network_from_mpg) with exactly the exporter's keys and values.

    Compartments    section_x, section_y            the position in the section (mm)
                    wall_length                     walls (0 at junctions and protoplasts)
                    wall_thickness                  walls (µm, as the exporter's: the cells' wall layers)
                    cell_area, cgroup,              protoplasts (cell_area 0 elsewhere)
                    protect_topology, bridge
    Connections     network_path                    PATH_CODES: membrane 0, wall_air 1, wall 2, plasmodesmata 3,
                                                    air_link 4
                    edge_length, edge_dx, edge_dy   every edge (d_vec = (edge_dx, edge_dy))
                    edge_dist                       all but wall edges
                    edge_wall_thickness             membrane, wall_air and wall edges
                    lateral_distance,               wall edges
                    distnode_wall_cell

The air links' via_new_junction_a, via_old_junction and via_new_junction_b (node indices) stay edge__via_* properties,
on the air links only. The units are GRANAP's (mm, mm2).
"""
import math

import numpy as np

PATH_CODES = {"membrane": 0, "wall_air": 1, "wall": 2, "plasmodesmata": 3, "air_link": 4}
JUNCTION_PATH = 5                # the axial junctions between successive sections (created by the wiring)
STOMATAL_PATH = 6                # a stomatal pore to its leaf organ's surface node (plan §60)
PATHS = {code: path for path, code in PATH_CODES.items()}
NAN = float("nan")

NODE_COLUMNS = ("section_x", "section_y", "wall_length", "wall_thickness", "cell_area", "cgroup", "protect_topology",
                "bridge")
EDGE_COLUMNS = ("network_path", "edge_length", "edge_dx", "edge_dy", "edge_dist", "edge_wall_thickness",
                "lateral_distance", "distnode_wall_cell")
VIA = ("via_new_junction_a", "via_old_junction", "via_new_junction_b")
_EDGE_KEYS = {          # the exporter's attributes of an edge, by path (besides path and d_vec)
    "membrane": ("length", "dist", "wall_thickness"),
    "wall_air": ("length", "dist", "wall_thickness"),
    "wall": ("length", "lateral_distance", "distnode_wall_cell", "wall_thickness"),
    "plasmodesmata": ("length", "dist"),
    "air_link": ("length", "dist"),
}
_EDGE_COLUMN = {"length": "edge_length", "dist": "edge_dist", "wall_thickness": "edge_wall_thickness",
                "lateral_distance": "lateral_distance", "distnode_wall_cell": "distnode_wall_cell"}


def node_columns(attrs: dict) -> dict:
    """The columns of one exporter node (its attribute dict)."""
    x, y = attrs["position"]
    out = dict.fromkeys(NODE_COLUMNS, NAN)
    out.update(section_x=x, section_y=y, wall_length=0., cell_area=0.)       # extensive quantities: 0 where absent
    if attrs.get("type") == "apo":
        out["wall_length"] = attrs.get("length", NAN)
        out["wall_thickness"] = attrs.get("wall_thickness", NAN)
    else:
        cgroup = attrs.get("cgroup")
        out.update(cell_area=NAN if attrs.get("area") is None else attrs["area"],
                   cgroup=NAN if cgroup is None else cgroup,
                   protect_topology=float(attrs.get("protect_topology", False)), bridge=float(attrs.get("bridge", False)))
    return out


def edge_columns(attrs: dict) -> dict:
    """The columns of one exporter edge (its attribute dict)."""
    out = dict.fromkeys(EDGE_COLUMNS, NAN)
    out["network_path"] = PATH_CODES[attrs["path"]]
    out["edge_dx"], out["edge_dy"] = attrs["d_vec"]
    for key in _EDGE_KEYS[attrs["path"]]:
        out[_EDGE_COLUMN[key]] = attrs[key]
    return out


def write(g, name_values: dict, vids) -> None:
    """Each column of *name_values* ({name: values aligned with vids}) into the MPG."""
    for name, values in name_values.items():
        g.set_values(name, vids, values)


def _number(value):
    """A stored number as the exporter had it: an integral cgroup as an int."""
    return int(value) if float(value).is_integer() else value


def node_attributes(g, vids, kinds, mecha, tissue) -> list:
    """The exporter's attribute dict of each Compartment of *vids* (kinds: WALL 1, JUNCTION 2, protoplasts 0)."""
    props = g.properties()
    column = {name: props.get(name, {}) for name in NODE_COLUMNS}
    out = []
    for v, kind in zip(vids, kinds):
        position = (column["section_x"][v], column["section_y"][v])
        if kind == 1:
            out.append(dict(indice=int(mecha[v]), type="apo", position=position, length=column["wall_length"][v],
                            wall_thickness=column["wall_thickness"][v]))
        elif kind == 2:
            out.append(dict(indice=int(mecha[v]), type="apo", position=position, length=0))
        else:
            cgroup = column["cgroup"][v]
            out.append(dict(indice=int(mecha[v]), type="cell",
                            cgroup=cgroup if math.isnan(cgroup) else _number(cgroup), cell_type=tissue[v],
                            protect_topology=bool(column["protect_topology"][v]), bridge=bool(column["bridge"][v]),
                            position=position, area=column["cell_area"][v]))
    return out


def edge_attributes(g, connections) -> list:
    """The exporter's attribute dict of each Connection of *connections*."""
    props = g.properties()
    column = {name: props.get(name, {}) for name in EDGE_COLUMNS}
    via = {name: props.get("edge__" + name, {}) for name in VIA}
    out = []
    for e in connections:
        path = PATHS[int(column["network_path"][e])]
        attrs = {"path": path}
        for key in _EDGE_KEYS[path]:
            attrs[key] = column[_EDGE_COLUMN[key]][e]
        attrs["d_vec"] = np.array([column["edge_dx"][e], column["edge_dy"][e]])
        if path == "air_link":
            for name in VIA:
                if e in via[name]:
                    attrs[name] = via[name][e]
        out.append(attrs)
    return out
