"""
C1 (devplan_metafspm.md, §42): the section's hydraulic network as numeric variables. A component declares them at
the Compartment and Connection scales; the DataStructure reads them from the MPG, on every node and edge, and they hold
the exporter's values (those of GRANAP's graph, by MECHA index).
"""
from dataclasses import dataclass

import numpy as np
import pytest

from openalea.metafspm.coupling.choregrapher import Choregrapher
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
    section_x: float = state_variable(**DOC, initialize=0., scale=S.Compartment, location="node")
    wall_length: float = state_variable(**DOC, initialize=0., scale=S.Compartment, location="node")
    cell_area: float = state_variable(**DOC, initialize=0., scale=S.Compartment, location="node")
    edge_length: float = state_variable(**DOC, initialize=0., scale=S.Connection, location="edge")
    edge_dx: float = state_variable(**DOC, initialize=0., scale=S.Connection, location="edge")
    network_path: float = state_variable(**DOC, initialize=0., scale=S.Connection, location="edge")


@pytest.fixture(scope="module")
def section():
    organ = RootAnatomy(OrganInputData.for_root(), seed=0)
    organ.generate_cells()
    g = organ.mpg
    organ.export_to_adjencymatrix()
    graph = organ.graph                                       # GRANAP's view, built before the conversion
    g.convert_properties_to_arraydict()
    Choregrapher().add_simulation_time_step(3600.)
    ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan, nodes="Compartment")
    Network(data_structure=ds)
    return organ, g, ds, graph


def test_the_network_columns_are_read_as_variables_on_every_node_and_edge(section):
    organ, g, ds, graph = section
    for name in ("section_x", "wall_length", "cell_area"):
        assert ds.has(name) and ds.get(name).shape == (ds.n_nodes(),)
    for name in ("edge_length", "edge_dx", "network_path"):
        assert ds.has(name) and ds.get(name).shape == (ds.n_edges(),)
        assert not np.isnan(ds.get(name)).any()


def test_the_variables_hold_the_exporters_values(section):
    organ, g, ds, graph = section
    mecha = g.property("mecha_index")
    nodes = ds.entity_ids("node")
    x, wall_length, area = ds.get("section_x"), ds.get("wall_length"), ds.get("cell_area")
    for i, v in enumerate(nodes.tolist()):
        attrs = graph.nodes[int(mecha[v])]
        assert x[i] == attrs["position"][0]
        if attrs["type"] == "cell":
            assert area[i] == attrs["area"] and np.isnan(wall_length[i])
        else:
            assert wall_length[i] == attrs["length"] and np.isnan(area[i])
    u, w = g.property("edge_u"), g.property("edge_v")
    edges = ds.entity_ids("edge")
    length, dx, path = ds.get("edge_length"), ds.get("edge_dx"), ds.get("network_path")
    for i, e in enumerate(edges.tolist()):
        attrs = graph.edges[int(u[e]), int(w[e])]
        assert length[i] == attrs["length"] and path[i] == PATH_CODES[attrs["path"]]
        assert abs(dx[i]) == abs(attrs["d_vec"][0])           # the sign follows the Connection's orientation
