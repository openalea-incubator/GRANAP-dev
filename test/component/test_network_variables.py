"""
C1: the section's hydraulic network as numeric variables. A component declares them at
the Compartment and Connection scales; the DataStructure reads them from the MPG, on every node and edge, and they hold
the exporter's values (those of GRANAP's graph, by MECHA index).
"""
from dataclasses import dataclass

import numpy as np
import pytest

from openalea.metafspm.coupling.component import FunctionalComponent, state_variable
from openalea.metafspm.data_structure.configs import ScalesConfig
from openalea.metafspm.data_structure.data_api import MPGDataStructure

from openalea.granap.input_data import OrganInputData
from openalea.granap.root_class import RootAnatomy
from openalea.granap.metafspm.columns import PATH_CODES

DOC = dict(unit="mm", unit_comment="", description="", min_value=-1e9, max_value=1e9, value_comment="",
           references="", DOI=[])
S = ScalesConfig()


@dataclass
class Network(FunctionalComponent):
    """Declares GRANAP's network columns, as a hydraulic model would."""
    section_x: float = state_variable(**DOC, default=0., scale=S.Compartment, location="node")
    wall_length: float = state_variable(**DOC, default=0., scale=S.Compartment, location="node")
    cell_area: float = state_variable(**DOC, default=0., scale=S.Compartment, location="node")
    edge_length: float = state_variable(**DOC, default=0., scale=S.Connection, location="edge")
    edge_dx: float = state_variable(**DOC, default=0., scale=S.Connection, location="edge")
    network_path: float = state_variable(**DOC, default=0., scale=S.Connection, location="edge")


@pytest.fixture(scope="module")
def section():
    organ = RootAnatomy(OrganInputData.for_root(), seed=0)
    organ.generate_cells()
    mtg = organ.mpg
    organ.export_to_adjencymatrix()
    graph = organ.graph                                       # GRANAP's view, built before the conversion
    mtg.convert_properties_to_arraydict()
    data_structure = MPGDataStructure(mtg, from_scale=mtg.scales.SubOrgan, nodes="Compartment")
    data_structure.simulation_time_step = 3600.
    Network(data_structure=data_structure)
    return organ, mtg, data_structure, graph


def test_the_network_columns_are_read_as_variables_on_every_node_and_edge(section):
    organ, mtg, data_structure, graph = section
    for name in ("section_x", "wall_length", "cell_area"):
        assert data_structure.has(name) and data_structure.get(name).shape == (data_structure.n_nodes(),)
    for name in ("edge_length", "edge_dx", "network_path"):
        assert data_structure.has(name) and data_structure.get(name).shape == (data_structure.n_edges(),)
        assert not np.isnan(data_structure.get(name)).any()


def test_the_variables_hold_the_exporters_values(section):
    organ, mtg, data_structure, graph = section
    mecha = mtg.property("mecha_index")
    nodes = data_structure.entity_ids("node")
    x, wall_length, area = data_structure.get("section_x"), data_structure.get("wall_length"), data_structure.get("cell_area")
    for i, v in enumerate(nodes.tolist()):
        attrs = graph.nodes[int(mecha[v])]
        assert x[i] == attrs["position"][0]
        if attrs["type"] == "cell":
            assert area[i] == attrs["area"] and wall_length[i] == 0.          # extensive: 0 where absent
        else:
            assert wall_length[i] == attrs["length"] and area[i] == 0.
    u, w = mtg.property("edge_u"), mtg.property("edge_v")
    edges = data_structure.entity_ids("edge")
    length, dx, path = data_structure.get("edge_length"), data_structure.get("edge_dx"), data_structure.get("network_path")
    for i, e in enumerate(edges.tolist()):
        attrs = graph.edges[int(u[e]), int(w[e])]
        assert length[i] == attrs["length"] and path[i] == PATH_CODES[attrs["path"]]
        assert abs(dx[i]) == abs(attrs["d_vec"][0])           # the sign follows the Connection's orientation
