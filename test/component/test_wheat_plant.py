"""
A wheat plant as metafspm's use case 1 builds it (devplan_usecase.md, section 2.1): one graph from the seed (a SubOrgan
labelled Seed, with the stem's anatomy) through the subcrown internode to the crown, then Adel's phytomer (each organ
starting with a HiddenElement, without anatomy), a seminal root on the seed and a nodal root on a crown element. The
conduits are joined across the hidden elements (a pass-through), from the root tips to the leaf.
"""
import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from openalea.metafspm.coupling.choregrapher import Choregrapher
from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.metafspm.scene.population import Planting, planting

from openalea.granap.metafspm.architectures import SegmentArchitecture, new_organ, segment_chain
from openalea.granap.metafspm.component import GranapAnatomy, anatomy_ancestor, vessel_pairs

from roots import initiated

STEM_RADIUS, ROOT_RADIUS = 1e-3, 0.4e-3                # m
SEED_LENGTH, ELEMENT, HIDDEN = 2e-3, 5e-3, 1e-2        # m
WHOLE_SHOOT = {"seed_anatomy": {}, "stem_anatomy": {}, "sheath_anatomy": {}, "leaf_anatomy": {}}
NO_SHEATH = {"seed_anatomy": {}, "stem_anatomy": {}, "leaf_anatomy": {}}    # a sheath's elements without anatomy


def _following(g, organ, label, lengths, after, edge="<"):
    """SubOrgans of *label* under *organ*, each the successor of the previous, the first after *after* by *edge*."""
    out, previous = [], after
    for rank, length in enumerate(lengths):
        props = PropsConfig(scale=g.scales.SubOrgan, edge_type=edge if rank == 0 else "<", label=label)
        previous = g.add_component_with_topo(organ, previous, **props) if rank == 0 else g.add_child(previous, **props)
        g.property("length")[previous] = float(length)
        g.property("stem_radius")[previous] = STEM_RADIUS
        g.property("distance_from_tip")[previous] = 0.
        out.append(previous)
    return out


class WheatSeedling(SegmentArchitecture):
    """The seed, two subcrown internode elements up to the crown, the first phytomer as Adel builds it (its internode:
    a HiddenElement, then a StemElement; its sheath: the same, branched on the internode's; its blade: a
    HiddenElement, then two LeafElements, on the sheath's), a seminal root of two segments on the seed and a nodal
    root of one on the crown's StemElement. The plant's elements by role in the property "role". Its segments'
    variables (SegmentArchitecture's: their length among them) held at the SubOrgans, the hidden elements' too."""

    @classmethod
    def initiate_plant(cls, g, plant, parameters):
        L = g.labels.SubOrgan
        seed_organ = new_organ(g, plant, "Shoot", "Shoot", "Shoot", "StemInternode")
        seed = g.add_component(seed_organ, **PropsConfig(scale=g.scales.SubOrgan, edge_type="/", label=L.Seed))
        g.property("length")[seed], g.property("stem_radius")[seed] = SEED_LENGTH, STEM_RADIUS
        g.property("distance_from_tip")[seed] = 0.
        phytomer = g.complex(seed_organ)
        organ = lambda label: g.add_component(phytomer, **PropsConfig(scale=g.scales.Organ, edge_type="/",
                                                                      label=getattr(g.labels.Organ, label)))
        subcrown = _following(g, organ("StemInternode"), L.StemElement, [ELEMENT, ELEMENT], seed)
        internode = organ("StemInternode")
        (internode_hidden,) = _following(g, internode, L.HiddenElement, [HIDDEN], subcrown[-1])
        (crown,) = _following(g, internode, L.StemElement, [ELEMENT], internode_hidden)
        sheath = organ("Sheath")
        (sheath_hidden,) = _following(g, sheath, L.HiddenElement, [HIDDEN], crown, edge="+")
        (sheath_element,) = _following(g, sheath, L.StemElement, [ELEMENT], sheath_hidden)
        blade = organ("Blade")
        (blade_hidden,) = _following(g, blade, L.HiddenElement, [HIDDEN], sheath_element)
        leaf = _following(g, blade, L.LeafElement, [ELEMENT, ELEMENT], blade_hidden)
        seminal = segment_chain(g, new_organ(g, plant, "Root", "Root", "Root", "RootInternode"), L.RootSegment,
                                [(ROOT_RADIUS, ELEMENT, ELEMENT), (ROOT_RADIUS, ELEMENT, 0.)], bearer=seed)
        nodal = segment_chain(g, new_organ(g, plant, "Root", "Root", "Root", "RootInternode"), L.RootSegment,
                              [(ROOT_RADIUS, ELEMENT, 0.)], bearer=crown)
        roles = dict(seed=[seed], subcrown=subcrown, hidden=[internode_hidden, sheath_hidden, blade_hidden],
                     crown=[crown], sheath=[sheath_element], leaf=leaf, seminal=seminal, nodal=nodal)
        for role, vids in roles.items():
            for v in vids:
                g.property("role")[v] = role


@pytest.fixture(scope="module")
def plants():
    Choregrapher().set_simulation_time_step(3600.)
    table = planting([Planting(x=0.), Planting(x=0.1)], None, plant_scenarios=[WHOLE_SHOOT, NO_SHEATH])
    population = initiated(table, WheatSeedling, GranapAnatomy)
    return population.data_structure


def _roles(g, plant):
    role = g.property("role")
    out = {}
    for v in g.components_at_scale(plant, scale=g.scales.SubOrgan):
        out.setdefault(role[int(v)], []).append(int(v))
    return {name: sorted(vids) for name, vids in out.items()}


def _family(g, vids):
    family = g.values_of("family", vids)
    return np.array([family.get(v, -1) for v in vids])


def _expected_junctions(g, sections):
    """{(ancestor section, section, family): the vessel pairs}: each section with an anatomy joined to its nearest
    ancestor with one."""
    owners = g.compartments_by_owner(g.scales.SubOrgan)
    out = {}
    for section in sections:
        ancestor, _ = anatomy_ancestor(g, section, owners)
        if ancestor is not None and owners.get(section):
            for family in (0, 1):
                out[(ancestor, section, family)] = vessel_pairs(g, owners[ancestor], owners[section], family)
    return out


def _actual_junctions(g, ds, sections):
    a, b = g.property("n_id_a"), g.property("n_id_b")
    junctions = [int(e) for e in g.junction_vids()]
    tails, heads = [int(a[e]) for e in junctions], [int(b[e]) for e in junctions]
    parent_sections, child_sections = ds.sections_of(tails), ds.sections_of(heads)
    families = _family(g, tails)
    out = {}
    for tail, head, parent, child, family in zip(tails, heads, parent_sections.tolist(), child_sections.tolist(),
                                                 families.tolist()):
        if child in sections:
            out.setdefault((parent, child, family), []).append((tail, head))
    return {key: sorted(pairs) for key, pairs in out.items()}


@pytest.mark.parametrize("plant_index", [0, 1], ids=["whole shoot", "sheath without anatomy"])
def test_the_seed_gets_the_stems_anatomy_and_the_hidden_elements_none(plants, plant_index):
    ds = plants
    g = ds.structure
    plant = int(ds.entity_ids("Plant")[plant_index])
    roles = _roles(g, plant)
    template = g.property("anatomy_template")
    assert template[roles["seed"][0]] == template[roles["subcrown"][0]] >= 0       # the stem preset, one class
    owners = g.compartments_by_owner(g.scales.SubOrgan)
    assert all(not owners.get(v) and template.get(v, -1.) == -1. for v in roles["hidden"])
    assert bool(owners.get(roles["sheath"][0])) == (plant_index == 0)


@pytest.mark.parametrize("plant_index", [0, 1], ids=["whole shoot", "sheath without anatomy"])
def test_the_conduits_are_joined_from_the_root_tips_to_the_leaf_across_the_hidden_elements(plants, plant_index):
    ds = plants
    g = ds.structure
    plant = int(ds.entity_ids("Plant")[plant_index])
    roles = _roles(g, plant)
    owners = g.compartments_by_owner(g.scales.SubOrgan)
    sections = sorted(v for vids in roles.values() for v in vids if owners.get(v))
    expected, actual = _expected_junctions(g, sections), _actual_junctions(g, ds, set(sections))
    assert actual == {key: pairs for key, pairs in expected.items() if pairs}       # each pair once, nothing else
    # the pass-through junctions: across the internode's hidden element, the sheath's (and its element without
    # anatomy), the blade's
    crossings = {(roles["subcrown"][-1], roles["crown"][0])}
    crossings |= {(roles["crown"][0], roles["sheath"][0]), (roles["sheath"][0], roles["leaf"][0])} \
        if plant_index == 0 else {(roles["crown"][0], roles["leaf"][0])}
    for parent, child in crossings:
        for family in (0, 1):
            assert len(actual[(parent, child, family)]) >= max(len(_vessels(g, owners[parent], family)),
                                                              len(_vessels(g, owners[child], family))) > 0
    # every conduit is joined to its neighbours' (upstream and downstream), and the conduits form one network from
    # the root tips to the leaf's tip, by family
    conduits = {v for s in sections for v in owners[s] if _family(g, [v])[0] in (0, 1)}
    joined_up, joined_down = set(), set()
    for (parent, child, family), pairs in actual.items():
        joined_down |= {tail for tail, _ in pairs}
        joined_up |= {head for _, head in pairs}
    downstream = {ancestor for ancestor, _, _ in expected}
    for s in sections:
        mine = {v for v in owners[s] if v in conduits}
        if s != roles["seed"][0]:
            assert mine <= joined_up, f"section {s}: a conduit without upstream junction"
        if s in downstream:
            assert mine <= joined_down, f"section {s}: a conduit without downstream junction"
    for family in (0, 1):
        nodes = sorted(v for v in conduits if _family(g, [v])[0] == family)
        index = {v: i for i, v in enumerate(nodes)}
        edges = np.array([(index[a], index[b]) for (_, _, f), pairs in actual.items() if f == family
                          for a, b in pairs]).reshape(-1, 2)
        graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(nodes), len(nodes)))
        _, labels = connected_components(graph, directed=False)
        component = lambda section: {int(labels[index[v]]) for v in owners[section] if v in index}
        leaf_tip = component(roles["leaf"][-1])
        for tip in (roles["seminal"][-1], roles["nodal"][-1]):
            assert component(tip) & leaf_tip, f"family {family}: the root tip {tip} is not joined to the leaf"


def _vessels(g, compartments, family):
    return [v for v, f in zip(compartments, _family(g, compartments)) if f == family]


def test_a_pass_through_junction_spans_the_hidden_elements(plants):
    ds = plants
    g = ds.structure
    plant = int(ds.entity_ids("Plant")[1])
    roles = _roles(g, plant)
    a, b, length = g.property("n_id_a"), g.property("n_id_b"), g.property("edge_length")
    owners = g.compartments_by_owner(g.scales.SubOrgan)
    crown, leaf = set(owners[roles["crown"][0]]), set(owners[roles["leaf"][0]])
    spanning = [e for e in g.junction_vids() if int(a[e]) in crown and int(b[e]) in leaf]
    # half the crown element, the sheath's hidden element and element (no anatomy), the blade's hidden element, half
    # the leaf element (mm)
    assert spanning and np.allclose([length[e] for e in spanning],
                                    (0.5 * (ELEMENT + ELEMENT) + HIDDEN + ELEMENT + HIDDEN) * 1e3)


def test_a_replaced_ancestor_is_joined_again_across_the_hidden_elements():
    """The DataStructure rewires a replaced section's children only: the component joins the replaced section to
    its descendants across the hidden elements; nothing dangles, nothing is doubled; the junctions' length is kept as
    the elements grow."""
    Choregrapher().set_simulation_time_step(3600.)
    table = planting([Planting()], None, plant_scenarios=[NO_SHEATH])
    ds = initiated(table, WheatSeedling, GranapAnatomy).data_structure
    g = ds.structure
    WheatSeedling(data_structure=ds)                  # first: it holds the segments' lengths, the hidden elements'
    anatomy = GranapAnatomy(data_structure=ds)
    plant = int(ds.entity_ids("Plant")[0])
    roles = _roles(g, plant)
    top = roles["subcrown"][-1]
    stem_radius = ds.get("stem_radius").copy()
    stem_radius[ds.entity_ids("SubOrgan") == top] = 1.5 * STEM_RADIUS            # another class: replaced
    ds.set("stem_radius", stem_radius)
    anatomy()
    assert anatomy.last_update == {"emerged": [], "replaced": [top]}
    owners = g.compartments_by_owner(g.scales.SubOrgan)
    sections = sorted(v for vids in roles.values() for v in vids if owners.get(v))
    expected, actual = _expected_junctions(g, sections), _actual_junctions(g, ds, set(sections))
    assert actual == {key: pairs for key, pairs in expected.items() if pairs}
    assert all(actual[(top, roles["crown"][0], family)] for family in (0, 1))
    a, b = g.property("n_id_a"), g.property("n_id_b")
    assert all(g.has_vertex(int(a[e])) and g.has_vertex(int(b[e])) for e in g.junction_vids())
    edge_length = ds.get("edge_length")
    edges = np.asarray(ds.edges(), dtype=np.int64).reshape(-1, 2)
    across = np.isin(edges[:, 0], owners[top]) & np.isin(edges[:, 1], owners[roles["crown"][0]])
    assert across.any() and np.allclose(edge_length[across], (0.5 * (ELEMENT + ELEMENT) + HIDDEN) * 1e3)
    anatomy()                                                                    # a step more: kept
    assert np.allclose(ds.get("edge_length")[across], (0.5 * (ELEMENT + ELEMENT) + HIDDEN) * 1e3)
