"""
GRANAP's legacy source (openalea.granap.legacy) and its MPG-backed organs, entity by entity (devplan_metafspm.md, Q7), at seed=0 on the pinned geometry
stack: the same cells (every attribute, the polygons equal), the same cell GeoDataFrame, the same network (nodes,
edges and every attribute, counts, wall map), and the MPG holding them at the Compartment and Connection scales.
"""
import os

import numpy as np
import pytest

from openalea.granap import mpg
from openalea.granap.input_data import OrganInputData
from openalea.granap.legacy.input_data import OrganInputData as LegacyInputData
from openalea.granap.legacy.needle_class import NeedleAnatomy
from openalea.granap.legacy.root_class import RootAnatomy
from openalea.granap.legacy.stem_class import StemAnatomy
from openalea.granap.mpg.build import JUNCTION, PROTOPLAST, WALL

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


ORGANS = {          # name: (original class, MPG-backed class, input data, network compared)
    "monocot_default": (RootAnatomy, mpg.RootAnatomy, OrganInputData.for_root, True),
    "monocot_arch": (RootAnatomy, mpg.RootAnatomy, _monocot_arch, False),
    "dicot_primary": (RootAnatomy, mpg.RootAnatomy, OrganInputData.for_dicot_root, False),
    "dicot_secondary": (RootAnatomy, mpg.RootAnatomy, _dicot_secondary, False),
    "needle_default": (NeedleAnatomy, mpg.NeedleAnatomy, OrganInputData.for_needle, True),
    "needle_features": (NeedleAnatomy, mpg.NeedleAnatomy, _needle_features, False),
    "dicot_stem": (StemAnatomy, mpg.StemAnatomy, OrganInputData.for_dicot_stem, False),
    "monocot_stem": (StemAnatomy, mpg.StemAnatomy, OrganInputData.for_monocot_stem, False),
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
        _check_network(old, new)
        _check_cells(old, new)                  # the bridge cells too, after the export
    _check_mpg(new)
