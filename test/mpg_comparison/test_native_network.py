"""
N1 (devplan_metafspm.md, §28): the stored network built in the MPG from its cells and rings is the exporter's, node
for node, edge for edge, every attribute, on every organ of the cross-check and on both leaves (protected air spaces,
air links); and the adjacency matrix read from the Connections is the one built from the graph.
"""
import pytest

from openalea.granap import mpg
from openalea.granap.input_data import OrganInputData
from openalea.granap.network_base import AbstractNetwork

from test_cross_check import ORGANS, SEED, _check_network

CASES = {name: (backed, data) for name, (_, backed, data, _) in ORGANS.items()}
CASES.update(monocot_leaf=(mpg.LeafAnatomy, OrganInputData.for_monocot_leaf),
             dicot_leaf=(mpg.LeafAnatomy, OrganInputData.for_dicot_leaf))
QUICK = {"monocot_default", "dicot_leaf"}


def _stored(backed, data, native):
    organ = backed(data(), seed=SEED)
    organ.native_network = native
    organ.generate_cells()
    organ._build_anatnetwork(bridges=False)                    # the stored network, as stored
    return organ


@pytest.mark.parametrize("name", [pytest.param(n, marks=() if n in QUICK else pytest.mark.slow) for n in CASES])
def test_the_network_built_in_the_mpg_is_the_exporters(name):
    backed, data = CASES[name]
    exported, native = _stored(backed, data, False), _stored(backed, data, True)
    _check_network(exported, native)
    graph_matrix = AbstractNetwork.export_to_adjencymatrix.__get__(exported)()       # from the graph (GRANAP's)
    assert (graph_matrix != native.export_to_adjencymatrix()).nnz == 0
