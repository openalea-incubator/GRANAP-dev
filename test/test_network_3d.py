"""Tests for the 3D hydraulic network (doc/NETWORK_3D_PLAN.md).

P1: ``build_wall_registry`` -- the 2D walls and junctions, shared by the 2D
network export and the 3D network.
P2: ``Network3D`` -- the 3D network lifted from them.
"""

from collections import Counter

import networkx as nx
import pytest

from openalea.granap.input_data import OrganInputData
from openalea.granap.root_class import RootAnatomy
from openalea.granap.anatomy_writer import build_wall_registry, NetworkExporter
from openalea.granap.generate_cell_3d import Cells3DResult
from openalea.granap.network_base import AbstractNetwork
from openalea.granap.network_3d import Network3D


def _small_root():
    data = OrganInputData.for_root()
    for layer in ("epidermis", "exodermis", "cortex", "endodermis", "pericycle"):
        data.set_values(layer, cell_diameter=0.06, cell_width=0.06)
    data.set_values("stele", cell_diameter=0.03, cell_diameter_center=0.03)
    return RootAnatomy(data, seed=0)


@pytest.fixture(scope="module")
def section():
    return _small_root().generate_cells()


def test_registry_walls_and_junctions_are_consistent(section):
    reg = build_wall_registry(section)
    walls = list(reg.walls.values())

    assert [w["id"] for w in walls] == list(range(len(walls)))
    assert reg.junction_ids == {vk: i for i, vk in enumerate(reg.junctions)}
    for w in walls:
        assert 1 <= len(w["cells"]) <= 2
        assert set(w["cells"]) <= set(reg.cells_gdf.index)
        assert w["junc_start"] in reg.junction_ids and w["junc_end"] in reg.junction_ids
        assert w["length"] > 0


def test_registry_walls_cover_each_cell_boundary(section):
    """The walls of a cell add up to its perimeter (up to vertex snapping)."""
    reg = build_wall_registry(section)
    perimeter = {i: 0.0 for i in reg.cells_gdf.index}
    for w in reg.walls.values():
        for c in w["cells"]:
            perimeter[c] += w["length"]
    for i, p in perimeter.items():
        assert p == pytest.approx(reg.cells_gdf.loc[i, "geometry"].length, rel=1e-3)


def test_registry_wall_thickness(section):
    """Shared wall: t(a) + t(b); outer wall: t(a) + t('outerwall');
    a type missing from the dict takes 'default'."""
    t = {"cortex": 1.0, "epidermis": 3.0, "outerwall": 10.0, "default": 0.5}
    reg = build_wall_registry(section, cell_wall_thickness=t)
    types = reg.cells_gdf["type"]
    for w in reg.walls.values():
        expected = sum(t.get(types[c], t["default"]) for c in w["cells"])
        if len(w["cells"]) == 1:
            expected += t["outerwall"]
        assert w["wall_thickness"] == pytest.approx(expected)


def test_registry_of_the_3d_section_covers_every_3d_cell():
    """The 3D build's own section gives the walls of the 3D cells."""
    result = _small_root().generate_cells_3d(n_axial_repeats=2.0, seed=0)
    reg = build_wall_registry(result.section)
    assert {c["cell_2d"] for c in result.cells} <= set(reg.cells_gdf.index)


# ---------------------------------------------------------------------------
# P2: Network3D
# ---------------------------------------------------------------------------

HEIGHTS = {"stele": 0.05, "cortex": 0.12, "metaxylem": 0.3}


@pytest.fixture(scope="module")
def net():
    return _small_root().export_3d_network(axial_heights=HEIGHTS, segment_length=0.8, seed=0)


def _nodes(net, kind):
    return [(n, net.graph.nodes[n]) for n in net.nodes_of(kind)]


def _cell_nodes(net):
    return [(n, d) for n, d in net.graph.nodes(data=True) if d["type"] == "cell"]


def test_node_order_follows_mecha(net):
    kinds = [net.graph.nodes[n].get("kind") for n in range(net.n_total)]
    walls, juncs, cells = (kinds[:net.n_walls], kinds[net.n_walls:net.n_walls + net.n_junctions],
                           kinds[net.n_walls + net.n_junctions:])
    assert set(walls) == {"side", "end", "face"}
    assert set(juncs) == {"vertical", "rim"}
    assert set(cells) == {None} and len(cells) == len(net.result.cells)


def test_graph_is_connected(net):
    assert nx.is_connected(net.graph)


def test_side_wall_areas_add_up(net):
    """Per 2D wall: its side pieces cover the whole segment.
    Per cell row: its side walls cover its perimeter x height."""
    span = net.result.z_max - net.result.z_min
    per_wall = Counter()
    length = {}
    for _, d in _nodes(net, "side"):
        per_wall[d["wall_2d"]] += d["height"]
        length[d["wall_2d"]] = d["length"]
    assert all(h == pytest.approx(span) for h in per_wall.values())

    section = net.result.section
    for n, d in _cell_nodes(net):
        side = sum(net.graph.nodes[w]["area"] for w in net.graph[n]
                   if net.graph.nodes[w].get("kind") == "side")
        perimeter = section.loc[d["cell_2d"], "geometry"].length
        assert side == pytest.approx(perimeter * d["height"], rel=1e-3)


def test_axial_walls_join_consecutive_rows(net):
    """End walls sit between rows k and k+1 of one file and carry an axial
    plasmodesmata; faces close the first and last rows."""
    g = net.graph
    for w, d in _nodes(net, "end"):
        a, b = sorted(net._wall_to_cells[w], key=lambda n: g.nodes[n]["z0"])
        assert g.nodes[a]["cell_2d"] == g.nodes[b]["cell_2d"] == d["cell_2d"]
        assert g.nodes[b]["row"] == g.nodes[a]["row"] + 1
        assert g.nodes[a]["z1"] == pytest.approx(d["z"]) == pytest.approx(g.nodes[b]["z0"])
        assert g.edges[a, b]["path"] == "plasmodesmata"
        assert g.edges[a, b]["orientation"] == "axial"
        assert d["area"] == pytest.approx(g.nodes[a]["area"])
    faces = Counter(d["cell_2d"] for _, d in _nodes(net, "face"))
    assert set(faces.values()) == {2}
    assert len(faces) == len({c["cell_2d"] for c in net.result.cells})


def test_side_wall_cells_overlap_the_piece(net):
    g = net.graph
    for w, d in _nodes(net, "side"):
        for c in net._wall_to_cells[w]:
            assert g.nodes[c]["z0"] <= d["z0"] + 1e-9 and g.nodes[c]["z1"] >= d["z1"] - 1e-9


def test_vessels(net):
    """metaxylem has a height -> elements in series joined by end walls;
    phloem has none -> one tube, only its two faces."""
    g = net.graph
    files = {}
    for n, d in _cell_nodes(net):
        files.setdefault((d["cell_type"], d["cell_2d"]), []).append(n)
    meta = [ns for (t, _), ns in files.items() if t == "metaxylem"]
    phloem = [ns for (t, _), ns in files.items() if t == "phloem"]
    assert meta and all(len(ns) >= 2 for ns in meta)
    assert phloem and all(len(ns) == 1 for ns in phloem)
    for ns in phloem:
        axial = [w for w in g[ns[0]] if g.nodes[w].get("kind") in ("end", "face")]
        assert sorted(g.nodes[w]["kind"] for w in axial) == ["face", "face"]


def test_wall_thickness():
    t = {"default": 1.0, "cortex": 2.0, "epidermis": 3.0, "outerwall": 10.0}
    end = {"default": 0.5, "cortex": 0.25, "metaxylem": 0.1}
    net = _small_root().export_3d_network(axial_heights=HEIGHTS, segment_length=0.8, seed=0,
                                          cell_wall_thickness=t, end_wall_thickness=end)
    g = net.graph
    ctype = lambda n: g.nodes[n]["cell_type"]
    for w, d in g.nodes(data=True):
        if d.get("kind") not in ("side", "end", "face"):
            continue
        cells = net._wall_to_cells[w]
        if d["kind"] == "side":
            expected = sum(t.get(ctype(c), t["default"]) for c in cells)
            if len(cells) == 1:
                expected += t["outerwall"]
        else:
            expected = sum(end.get(ctype(c), end["default"]) for c in cells)
        assert d["wall_thickness"] == pytest.approx(expected)


class _Net2D(AbstractNetwork):
    def _build_anatnetwork(self, air_link_radius=None):
        pass


def test_one_row_per_cell_reduces_to_the_2d_network(net):
    """With every cell one full-span row, the 3D network is the 2D one: one
    side piece per 2D wall, one vertical piece per 2D junction, the same
    wall-junction links and the same lateral plasmodesmata."""
    res = net.result
    one_row = [dict(c, row=0, z0=res.z_min, z1=res.z_max) for c in res.cells if c["row"] == 0]
    net3 = Network3D(Cells3DResult(cells=one_row, z_min=res.z_min, z_max=res.z_max,
                                   section=res.section))

    class _Section:
        def generate_cells(self):
            return res.section
    net2 = _Net2D()
    NetworkExporter(_Section()).export(net2)

    g3, g2 = net3.graph, net2.graph
    assert len(net3.nodes_of("side")) == net2.n_walls
    assert len(net3.nodes_of("vertical")) == net2.n_junctions
    assert net3.n_cells == net2.n_cells

    def vertical_link(u, v):
        """(wall_2d, junction_2d) of a wall - vertical junction edge, else None."""
        for w, j in ((u, v), (v, u)):
            if g3.nodes[j].get("kind") == "vertical" and g3.nodes[w].get("kind") == "side":
                return g3.nodes[w]["wall_2d"], g3.nodes[j]["junction_2d"]
        return None
    links3 = {vertical_link(u, v) for u, v, d in g3.edges(data=True) if d["path"] == "wall"} - {None}
    links2 = {(min(u, v), max(u, v) - net2.n_walls)
              for u, v, d in g2.edges(data=True) if d["path"] == "wall"}
    assert links3 == links2

    pd3 = {tuple(sorted((g3.nodes[a]["cell_2d"], g3.nodes[b]["cell_2d"])))
           for a, b, d in g3.edges(data=True) if d["path"] == "plasmodesmata"}
    first = net2.n_walls + net2.n_junctions
    pd2 = {tuple(sorted((a - first, b - first)))
           for a, b, d in g2.edges(data=True) if d["path"] == "plasmodesmata"}
    assert pd3 == pd2


def test_adjacency_matrix_and_fill(net):
    m = net.export_to_adjencymatrix()
    assert m.shape == (net.n_total, net.n_total)
    net.fill_matrix(5.0, "transmembrane", "cortex")
    net.fill_matrix(7.0, "symplastic", "cortex-endodermis")
    assert (m != m.T).nnz == 0
    values = {v for row in m.data for v in row}
    assert {5.0, 7.0} <= values


def test_dicot_root_builds():
    net = RootAnatomy(OrganInputData.for_dicot_root(), seed=0).export_3d_network(
        segment_length=0.3, seed=0)
    assert net.n_cells == len(net.result.cells)
    assert nx.is_connected(net.graph)
