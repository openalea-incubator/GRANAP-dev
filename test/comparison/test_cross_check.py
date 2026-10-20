"""
GRANAP against its legacy source (openalea.granap.legacy), entity by entity (devplan_metafspm.md, Q7), at seed=0 on the pinned geometry
stack: the same cells (every attribute, the polygons equal), the same cell GeoDataFrame, the same network (nodes,
edges and every attribute, counts, wall map), and the MPG holding them at the Compartment and Connection scales.
"""
import os

import numpy as np
import pytest

from openalea.granap.input_data import OrganInputData
from openalea.granap.needle_class import NeedleAnatomy as Needle
from openalea.granap.root_class import RootAnatomy as Root
from openalea.granap.stem_class import StemAnatomy as Stem
from openalea.granap.legacy.input_data import OrganInputData as LegacyInputData
from openalea.granap.legacy.needle_class import NeedleAnatomy
from openalea.granap.legacy.root_class import RootAnatomy
from openalea.granap.legacy.stem_class import StemAnatomy
from openalea.granap.metafspm.build import JUNCTION, PROTOPLAST, WALL

SEED = 0


def legacy(data):
    """The same input for the legacy source (its own OrganInputData class)."""
    return LegacyInputData.from_dict_list(data.to_dict_list())


def _dicot_secondary():
    data = OrganInputData.for_dicot_root()
    data.set_value("secondary_growth", "value", True)
    return data


def _monocot_arch():
    data = OrganInputData.for_root()
    data.set_value("xylem", "xylem_shape", "arch")
    return data


def _needle_features():
    from needle_configs import build_gallery_needle_data
    return build_gallery_needle_data()


def _monocot_distributions():
    """The tissue branch's size distributions (plan §62): lognormal vessels, gamma sieve elements."""
    data = OrganInputData.for_root()
    data.set_value("xylem", "vessel_diameter_distribution", {"family": "lognormal"})
    data.set_value("phloem", "sieve_diameter_distribution", {"family": "gamma"})
    return data


def _monocot_no_metaxylem():
    """The tissue branch's zero-metaxylem apex: protoxylem and phloem poles only."""
    data = OrganInputData.for_root()
    data.set_value("xylem", "n_vascular_bundles", 0)
    return data


def _dicot_stem_outline():
    """The tissue branch's eustele ring following the pith/cortex outline."""
    data = OrganInputData.for_dicot_stem()
    data.set_value("vascular_bundle", "ring_shape", "outline")
    return data


ORGANS = {          # name: (legacy class, GRANAP's class, input data, network compared)
    "monocot_default": (RootAnatomy, Root, OrganInputData.for_root, True),
    "monocot_arch": (RootAnatomy, Root, _monocot_arch, False),
    "dicot_primary": (RootAnatomy, Root, OrganInputData.for_dicot_root, False),
    "dicot_secondary": (RootAnatomy, Root, _dicot_secondary, False),
    "needle_default": (NeedleAnatomy, Needle, OrganInputData.for_needle, True),
    "needle_features": (NeedleAnatomy, Needle, _needle_features, False),
    "dicot_stem": (StemAnatomy, Stem, OrganInputData.for_dicot_stem, False),
    "monocot_stem": (StemAnatomy, Stem, OrganInputData.for_monocot_stem, False),
    "monocot_distributions": (RootAnatomy, Root, _monocot_distributions, True),
    "monocot_no_metaxylem": (RootAnatomy, Root, _monocot_no_metaxylem, False),
    "dicot_stem_outline": (StemAnatomy, Stem, _dicot_stem_outline, False),
}
QUICK = {"monocot_default"}


def _same(a, b):
    if a is None or b is None:
        return a is b
    if hasattr(a, "equals") and hasattr(a, "geom_type"):
        return a.equals_exact(b, 0.) or a.equals(b)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b))
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return a == b


def _check_cells(old, new):
    assert len(old.all_cells.cells) == len(new.all_cells.cells)
    for i, (a, b) in enumerate(zip(old.all_cells.cells, new.all_cells.cells)):
        assert set(vars(a)) == set(vars(b)), f"cell {i}: attributes {set(vars(a)) ^ set(vars(b))}"
        for name in vars(a):
            assert _same(getattr(a, name), getattr(b, name)), f"cell {i}.{name}: {getattr(a, name)!r}"
    og, ng = old._cells_gdf, new._cells_gdf
    assert list(og.columns) == list(ng.columns) and list(og.index) == list(ng.index)
    assert og.drop(columns="geometry").equals(ng.drop(columns="geometry"))
    assert all(_same(a, b) for a, b in zip(og.geometry, ng.geometry))


def _apply_plan60(old):
    """
    The legacy's network with plan §60's rules applied (the reviewed differences of the main package): the stomatal
    pores are air (their membranes wall_air, their plasmodesmata gone) and every membrane of a protected air space is
    wall_air; each pore is air-linked to its nearest protected air space (its chamber). In place; returns old.
    """
    import numpy as np
    graph, nwj = old.graph, old.n_walls + old.n_junctions
    cells = old._cells_gdf
    node = lambda row: nwj + int(row)                                                     # noqa: E731
    pore_types = set(getattr(old, "PORE_TYPES", ("pore",)))
    chambers = sorted(r for r in cells.index if cells.loc[r, "type"] == "air space" and
                      bool(cells.loc[r, "protect_topology"]) and graph.has_node(node(r)))
    pores = sorted(r for r in cells.index if cells.loc[r, "type"] in pore_types and graph.has_node(node(r)))
    air = {node(r) for r in chambers} | {node(r) for r in pores}
    for u, v, data in list(graph.edges(data=True)):
        if data.get("path") == "membrane" and (u in air or v in air):
            data["path"] = "wall_air"
        elif data.get("path") == "plasmodesmata" and ((u in air) != (v in air)):
            graph.remove_edge(u, v)
    for r in pores:
        if not chambers:
            break
        geometry = cells.loc[r, "geometry"]
        chamber = chambers[int(np.argmin([geometry.distance(cells.loc[c, "geometry"]) for c in chambers]))]
        p, c = graph.nodes[node(r)]["position"], graph.nodes[node(chamber)]["position"]
        dist = np.hypot(c[0] - p[0], c[1] - p[1])
        graph.add_edge(node(r), node(chamber), path="air_link", length=dist, dist=dist,
                       d_vec=np.array([c[0] - p[0], c[1] - p[1]]))
    old._matrix = None                                        # its adjacency matrix, rebuilt from the graph
    return old


def _check_network(old, new):
    assert (old.n_walls, old.n_junctions, old.n_cells) == (new.n_walls, new.n_junctions, new.n_cells)
    assert set(old.graph.nodes) == set(new.graph.nodes)
    for n, data in old.graph.nodes(data=True):
        other = new.graph.nodes[n]
        assert set(data) == set(other) and all(_same(data[k], other[k]) for k in data), f"node {n}"
    assert old.graph.number_of_edges() == new.graph.number_of_edges()
    for u, v, data in old.graph.edges(data=True):
        other = new.graph.edges[u, v]
        assert set(data) == set(other) and all(_same(data[k], other[k]) for k in data), f"edge {u}-{v}"
    assert old._wall_to_cells == new._wall_to_cells
    assert (old.export_to_adjencymatrix() != new.export_to_adjencymatrix()).nnz == 0


def _check_mpg(new):
    """Every MECHA node is a Compartment and every MECHA edge a Connection of the section."""
    g = new.mpg
    kinds = g.property("kind")
    compartments = [v for v, k in kinds.items()]
    assert sum(kinds[v] == PROTOPLAST for v in compartments) == len(new.all_cells.cells)
    if new.graph.number_of_nodes():
        assert sum(kinds[v] == WALL for v in compartments) == new.n_walls
        assert sum(kinds[v] == JUNCTION for v in compartments) == new.n_junctions
        assert len(g.property("edge_u")) == new.graph.number_of_edges()


@pytest.mark.parametrize("name", [pytest.param(n, marks=() if n in QUICK else pytest.mark.slow) for n in ORGANS])
def test_the_mpg_backed_organ_gives_the_same_section(name, tmp_path):
    original, backed, data, network = ORGANS[name]
    old, new = original(legacy(data()), seed=SEED), backed(data(), seed=SEED)
    old.generate_cells()
    new.generate_cells()
    _check_cells(old, new)
    if network:
        old.write_to_xml(str(tmp_path / "old.xml"))
        new.write_to_xml(str(tmp_path / "new.xml"))                # walls numbered from the stored topology
        assert (tmp_path / "old.xml").read_bytes() == (tmp_path / "new.xml").read_bytes()
        assert getattr(new, "mpg_topology_reads", 0) >= 1
        old.export_to_adjencymatrix()
        new.export_to_adjencymatrix()
        _check_network(_apply_plan60(old), new)            # the legacy's, with plan §60's reviewed differences
        _check_cells(old, new)                  # the bridge cells too, after the export
    _check_mpg(new)
