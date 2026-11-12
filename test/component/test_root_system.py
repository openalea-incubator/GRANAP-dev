"""
A root system populated with GRANAP anatomies (P3): one template per class, a copy per segment, vessel ids, junctions
between successive sections, and the hydraulic graph of the whole system solved by a metafspm graph system.
"""
from dataclasses import dataclass

import numpy as np
import pytest

from openalea.metafspm.coupling.component import FunctionalComponent, state_variable
from openalea.metafspm.data_structure.mpg import MPG
from openalea.metafspm.solve.decorator import boundary_set, edge_law, graph_system, node_balance

from openalea.granap.metafspm.component import GranapAnatomy
from openalea.granap.metafspm.library import AnatomyLibrary, template_registry
from openalea.granap.metafspm.vessels import CONDUCTING

from roots import SmallRootSystem, initiated, planting

DOC = dict(unit="", unit_comment="", description="", min_value=-1e9, max_value=1e9, value_comment="",
           references="", DOI=[])


LATERAL = (0.6, 0., 0.)                            # the lateral's class: 0.6 mm, distances not binned


class Anatomy(GranapAnatomy):
    # a library of its own; the lateral class parametrized with a thinner cortex (its own differentiation)
    library = AnatomyLibrary(class_parameters={LATERAL: {"cortex": {"n_layers": 2}}})


@pytest.fixture(scope="module")
def plant():
    population = initiated(planting(), SmallRootSystem, Anatomy)
    mtg, plants = population.data_structure.structure, population.plants
    data_structure = population.data_structure
    return mtg, plants[0], data_structure


def _segments(mtg):
    return sorted(int(v) for v in mtg.property("anatomy_class"))


def test_each_class_has_one_template_and_each_segment_its_own_copy(plant):
    mtg, _, data_structure = plant
    segments = _segments(mtg)
    assert len(segments) == 6 and len(template_registry(mtg)) == 2
    owners = mtg.compartments_by_owner(mtg.scales.SubOrgan)
    by_class = {}
    for segment in segments:
        by_class.setdefault(mtg.property("anatomy_class")[segment], []).append(segment)
    for key, members in by_class.items():
        template = Anatomy.library.template(mtg, key)
        expected = len(template.cell_layer) + len(template.apoplast["compartment_type"][0])
        assert all(len(owners[s]) == expected for s in members)
    all_compartments = [v for s in segments for v in owners[s]]
    assert len(all_compartments) == len(set(all_compartments)) == data_structure.n_nodes()     # disjoint copies
    template_of = mtg.property("template")
    cells = data_structure.entity_ids("Cell")
    assert len(cells) == sum(len(Anatomy.library.template(mtg, mtg.property("anatomy_class")[s]).cell_layer)
                             for s in segments)
    assert "cell__polygon" not in mtg.properties() or not set(cells) & set(mtg.property("cell__polygon"))
    assert {template_of[int(c)] for c in cells} == {t.index for t in template_registry(mtg).values()}


def test_vessels_are_numbered_within_each_section(plant):
    mtg, _, data_structure = plant
    owners = mtg.compartments_by_owner(mtg.scales.SubOrgan)
    v_id, fam, tis = mtg.property("vessel_id"), mtg.property("family"), mtg.property("tissue")
    for segment, compartments in owners.items():
        protoplasts = [v for v in compartments if v in v_id]
        conducting = [v for v in protoplasts if tis[v] in CONDUCTING]
        ids = [v_id[v] for v in conducting]
        assert len(ids) == len(set(ids)) > 0 and min(ids) == 0
        assert all(v_id[v] == -1 for v in protoplasts if tis[v] not in CONDUCTING)
        assert {fam[v] for v in conducting} == {0, 1}


def test_successive_sections_are_joined_vessel_to_vessel(plant):
    mtg, _, data_structure = plant
    junctions = mtg.junction_vids()
    a, b, label, fam = mtg.property("n_id_a"), mtg.property("n_id_b"), mtg.property("label"), mtg.property("family")
    owners = mtg.compartments_by_owner(mtg.scales.SubOrgan)
    section_of = {v: s for s, comps in owners.items() for v in comps}
    links = {}
    for e in junctions:
        assert fam[int(a[e])] == fam[int(b[e])]                                     # xylem to xylem, phloem to phloem
        expected = mtg.labels.Connection.Apoplastic if fam[int(a[e])] == 0 else mtg.labels.Connection.Symplastic
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
    mtg, _, data_structure = plant
    view = data_structure.to_graph_view()
    n_connections = sum(1 for v, s in mtg.view("scale").items() if s == mtg.scales.Connection
                        and not mtg.view("isanchor").get(v, False))
    assert view.incidence.shape == (data_structure.n_nodes(), data_structure.n_edges()) and data_structure.n_edges() == n_connections


@dataclass
class Hydraulics(FunctionalComponent):
    """A steady potential over the root system: the collar's xylem held at -1, the epidermis exchanging with -0.2."""
    psi: float = state_variable(**DOC, default=-0.5, location="node")
    flux: float = state_variable(**DOC, default=0., location="edge")

    @graph_system(node_unknowns=["psi"], edge_unknowns=["flux"], solver="newton")
    class _flow:
        soil = boundary_set(where={"is_epidermis": ">0"}, condition="robin", value="soil_psi", weight=0.5)
        collar = boundary_set(where={"is_collar_xylem": ">0"}, condition="dirichlet", value="collar_psi")

        @node_balance(field="psi")
        def _balance(self, flux):
            return np.asarray(self.graph_view.incidence @ flux).reshape(-1)

        @edge_law(field="flux", explicit=True)
        def _flow(self, psi):
            return np.asarray(self.graph_view.edge_difference @ psi).reshape(-1)


def test_a_graph_system_solves_on_the_root_systems_anatomies(plant):
    mtg, _, data_structure = plant
    data_structure.simulation_time_step = 3600.
    tissue = mtg.property("tissue")
    nodes = data_structure.entity_ids("node")
    first = _segments(mtg)[0]
    section = data_structure.entity_ids("SubOrgan")[data_structure.owner("SubOrgan")]
    epidermis = np.array([tissue.get(int(v)) == "epidermis" for v in nodes], dtype=float)
    xylem = np.array([tissue.get(int(v)) in ("metaxylem", "protoxylem", "xylem") for v in nodes])
    collar = (xylem & (section == first)).astype(float)
    for name, values in (("is_epidermis", epidermis), ("is_collar_xylem", collar),
                         ("soil_psi", np.full(nodes.size, -0.2)), ("collar_psi", np.full(nodes.size, -1.))):
        data_structure.register(name, values, location="node")
    Hydraulics(data_structure=data_structure)()
    B = data_structure.to_graph_view().incidence
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve
    A = (B @ B.T + sp.diags(0.5 * epidermis)).tolil()
    rhs = 0.5 * epidermis * -0.2
    for i in np.flatnonzero(collar):
        A.rows[i], A.data[i] = [i], [1.]
        rhs[i] = -1.
    expected = spsolve(A.tocsr(), rhs)
    np.testing.assert_allclose(data_structure.get("psi"), expected, rtol=1e-8, atol=1e-10)
    assert -1. <= data_structure.get("psi").min() and data_structure.get("psi").max() <= -0.2


def test_a_class_differs_only_by_its_own_parametrization(plant):
    mtg, _, data_structure = plant
    main = Anatomy.library.template(mtg, (1.1, 0., 0.))
    lateral = Anatomy.library.template(mtg, LATERAL)
    assert len(lateral.cell_layer) < len(main.cell_layer)          # two cortex layers instead of five
    unchanged = AnatomyLibrary().template(MPG(), (0.6, 0., 0.))     # without a parametrization: the preset
    assert len(unchanged.cell_layer) == len(main.cell_layer)


def test_create_from_input_takes_a_seed():
    from openalea.granap.input_data import OrganInputData
    from openalea.granap.organ_class import Organ
    create_from_input = Organ.create_from_input
    a, b = (create_from_input(OrganInputData.for_root(), seed=3) for _ in range(2))
    assert a.rng.integers(1 << 30) == b.rng.integers(1 << 30)                      # reproducible (Q8)
