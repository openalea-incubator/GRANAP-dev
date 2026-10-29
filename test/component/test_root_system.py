"""
A root system populated with GRANAP anatomies (P3): one template per class, a copy per segment, vessel ids, junctions
between successive sections, and the hydraulic graph of the whole system solved by a metafspm graph system.
"""
from dataclasses import dataclass

import numpy as np
import pytest

from openalea.metafspm.coupling.component import FunctionalComponent, state_variable
from openalea.metafspm.data_structure.data_api import MPGDataStructure
from openalea.metafspm.scene.population import build_population
from openalea.metafspm.solve.decorator import boundary_set, edge_law, graph_system, node_balance

from openalea.granap.metafspm.component import GranapAnatomy, junction_rules
from openalea.granap.metafspm.library import AnatomyLibrary
from openalea.granap.metafspm.vessels import CONDUCTING

from roots import SmallRootSystem, planting

DOC = dict(unit="", unit_comment="", description="", min_value=-1e9, max_value=1e9, value_comment="",
           references="", DOI=[])


LATERAL = (0.6, 0., 0.)                            # the lateral's class: 0.6 mm, distances not binned


class Anatomy(GranapAnatomy):
    # a library of its own; the lateral class parametrized with a thinner cortex (its own differentiation)
    library = AnatomyLibrary(class_parameters={LATERAL: {"cortex": {"n_layers": 2}}})


@pytest.fixture(scope="module")
def plant():
    g, plants = build_population(planting(), initiators=(SmallRootSystem, Anatomy))
    g.convert_properties_to_arraydict()
    ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan, nodes="Compartment", wiring=junction_rules(g))
    return g, plants[0], ds


def _segments(g):
    return sorted(int(v) for v in g.property("anatomy_class"))


def test_each_class_has_one_template_and_each_segment_its_own_copy(plant):
    g, _, ds = plant
    segments = _segments(g)
    assert len(segments) == 6 and len(Anatomy.library.templates) == 2
    owners = g.compartments_by_owner(g.scales.SubOrgan)
    by_class = {}
    for segment in segments:
        by_class.setdefault(g.property("anatomy_class")[segment], []).append(segment)
    for key, members in by_class.items():
        template = Anatomy.library.template(key)
        expected = len(template.cell_layer) + len(template.apoplast["kind"][0])
        assert all(len(owners[s]) == expected for s in members)
    all_compartments = [v for s in segments for v in owners[s]]
    assert len(all_compartments) == len(set(all_compartments)) == ds.n_nodes()     # disjoint copies
    template_of = g.property("template")
    cells = ds.entity_ids("Cell")
    assert len(cells) == sum(len(Anatomy.library.template(g.property("anatomy_class")[s]).cell_layer)
                             for s in segments)
    assert "cell__polygon" not in g.properties() or not set(cells) & set(g.property("cell__polygon"))
    assert {template_of[int(c)] for c in cells} == {t.index for t in Anatomy.library.templates.values()}


def test_vessels_are_numbered_within_each_section(plant):
    g, _, ds = plant
    owners = g.compartments_by_owner(g.scales.SubOrgan)
    v_id, fam, tis = g.property("vessel_id"), g.property("family"), g.property("tissue")
    for segment, compartments in owners.items():
        protoplasts = [v for v in compartments if v in v_id]
        conducting = [v for v in protoplasts if tis[v] in CONDUCTING]
        ids = [v_id[v] for v in conducting]
        assert len(ids) == len(set(ids)) > 0 and min(ids) == 0
        assert all(v_id[v] == -1 for v in protoplasts if tis[v] not in CONDUCTING)
        assert {fam[v] for v in conducting} == {0, 1}


def test_successive_sections_are_joined_vessel_to_vessel(plant):
    g, _, ds = plant
    junctions = g.junction_vids()
    a, b, label, fam = g.property("n_id_a"), g.property("n_id_b"), g.property("label"), g.property("family")
    owners = g.compartments_by_owner(g.scales.SubOrgan)
    section_of = {v: s for s, comps in owners.items() for v in comps}
    links = {}
    for e in junctions:
        assert fam[int(a[e])] == fam[int(b[e])]                                     # xylem to xylem, phloem to phloem
        expected = g.labels.Connection.Apoplastic if fam[int(a[e])] == 0 else g.labels.Connection.Symplastic
        assert label[e] == expected
        links.setdefault((section_of[int(a[e])], section_of[int(b[e])]), []).append(e)
    assert len(links) == 5                                                           # 3 along the main root, 1 to
    for (parent, child), edges in links.items():                                    # the lateral, 1 along it
        children = {int(b[e]) for e in edges}
        conducting = [v for v in owners[child] if fam.get(v, -1) >= 0]
        assert children == set(conducting)                                           # every vessel of the child
        parents = {int(a[e]) for e in edges}
        assert parents == {v for v in owners[parent] if fam.get(v, -1) >= 0}         # and of the parent


def test_the_hydraulic_graph_holds_every_compartment_and_connection(plant):
    g, _, ds = plant
    view = ds.to_graph_view()
    n_connections = sum(1 for v, s in g.property("scale").items() if s == g.scales.Connection
                        and not g.property("isanchor").get(v, False))
    assert view.incidence.shape == (ds.n_nodes(), ds.n_edges()) and ds.n_edges() == n_connections


@dataclass
class Hydraulics(FunctionalComponent):
    """A steady potential over the root system: the collar's xylem held at -1, the epidermis exchanging with -0.2."""
    psi: float = state_variable(**DOC, initialize=-0.5, location="node")
    flux: float = state_variable(**DOC, initialize=0., location="edge")

    @graph_system(node_unknowns=["psi"], edge_unknowns=["flux"], solver="newton")
    class _flow:
        soil = boundary_set(filters={"is_epidermis": ">0"}, kind="robin", value="soil_psi", weight=0.5)
        collar = boundary_set(filters={"is_collar_xylem": ">0"}, kind="dirichlet", value="collar_psi")

        @node_balance(field="psi")
        def _balance(self, flux):
            return np.asarray(self._graph_view.incidence @ flux).reshape(-1)

        @edge_law(field="flux", explicit=True)
        def _flow(self, psi):
            return np.asarray(self._graph_view.edge_difference @ psi).reshape(-1)


def test_a_graph_system_solves_on_the_root_systems_anatomies(plant):
    from openalea.metafspm.coupling.choregrapher import Choregrapher
    g, _, ds = plant
    Choregrapher().add_simulation_time_step(3600.)
    tissue = g.property("tissue")
    nodes = ds.entity_ids("node")
    first = _segments(g)[0]
    section = ds.entity_ids("SubOrgan")[ds.owner("SubOrgan")]
    epidermis = np.array([tissue.get(int(v)) == "epidermis" for v in nodes], dtype=float)
    xylem = np.array([tissue.get(int(v)) in ("metaxylem", "protoxylem", "xylem") for v in nodes])
    collar = (xylem & (section == first)).astype(float)
    for name, values in (("is_epidermis", epidermis), ("is_collar_xylem", collar),
                         ("soil_psi", np.full(nodes.size, -0.2)), ("collar_psi", np.full(nodes.size, -1.))):
        ds.register(name, values, location="node")
    Hydraulics(data_structure=ds)()
    B = ds.incidence_matrix()
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve
    A = (B @ B.T + sp.diags(0.5 * epidermis)).tolil()
    rhs = 0.5 * epidermis * -0.2
    for i in np.flatnonzero(collar):
        A.rows[i], A.data[i] = [i], [1.]
        rhs[i] = -1.
    expected = spsolve(A.tocsr(), rhs)
    np.testing.assert_allclose(ds.get("psi"), expected, rtol=1e-8, atol=1e-10)
    assert -1. <= ds.get("psi").min() and ds.get("psi").max() <= -0.2


def test_a_class_differs_only_by_its_own_parametrization(plant):
    g, _, ds = plant
    main = Anatomy.library.template((1.1, 0., 0.))
    lateral = Anatomy.library.template(LATERAL)
    assert len(lateral.cell_layer) < len(main.cell_layer)          # two cortex layers instead of five
    unchanged = AnatomyLibrary().template((0.6, 0., 0.))            # without a parametrization: the preset
    assert len(unchanged.cell_layer) == len(main.cell_layer)


def test_create_from_input_takes_a_seed():
    from openalea.granap.input_data import OrganInputData
    from openalea.granap.organ_class import Organ
    create_from_input = Organ.create_from_input
    a, b = (create_from_input(OrganInputData.for_root(), seed=3) for _ in range(2))
    assert a.rng.integers(1 << 30) == b.rng.integers(1 << 30)                      # reproducible (Q8)
