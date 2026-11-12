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

from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.metafspm.scene.population import Planting, planting

from openalea.granap.metafspm.architectures import SegmentArchitecture, segment_chain
from openalea.granap.metafspm.component import GranapAnatomy, anatomy_ancestor, vessel_pairs

from roots import initiated

pytestmark = pytest.mark.slow    # minutes long: run before a release or a merge to main (pytest -m slow)

STEM_RADIUS, ROOT_RADIUS = 1e-3, 0.4e-3                # m
SEED_LENGTH, ELEMENT, HIDDEN = 2e-3, 5e-3, 1e-2        # m
WHOLE_SHOOT = {"seed_anatomy": {}, "stem_anatomy": {}, "sheath_anatomy": {}, "leaf_anatomy": {}}
NO_SHEATH = {"seed_anatomy": {}, "stem_anatomy": {}, "leaf_anatomy": {}}    # a sheath's elements without anatomy


def _following(mtg, organ, label, lengths, after, edge="<"):
    """SubOrgans of *label* under *organ*, each the successor of the previous, the first after *after* by *edge*."""
    out, previous = [], after
    for rank, length in enumerate(lengths):
        props = PropsConfig(scale=mtg.scales.SubOrgan, edge_type=edge if rank == 0 else "<", label=label)
        previous = mtg.add_component_with_topo(organ, previous, **props) if rank == 0 else mtg.add_child(previous, **props)
        mtg.set_values("length", [previous], [float(length)])
        mtg.set_values("stem_radius", [previous], [STEM_RADIUS])
        mtg.set_values("distance_from_tip", [previous], [0.])
        out.append(previous)
    return out


class WheatSeedling(SegmentArchitecture):
    """The seed, two subcrown internode elements up to the crown, the first phytomer as Adel builds it (its internode:
    a HiddenElement, then a StemElement; its sheath: the same, branched on the internode's; its blade: a
    HiddenElement, then two LeafElements, on the sheath's), a seminal root of two segments on the seed and a nodal
    root of one on the crown's StemElement. The plant's elements by role in the property "role". Its segments'
    variables (SegmentArchitecture's: their length among them) held at the SubOrgans, the hidden elements' too."""
    sheath_length = ELEMENT                        # the sheath's StemElement's
    leaf_lengths = (ELEMENT, ELEMENT)              # the blade's LeafElements'

    @classmethod
    def initiate_plant(cls, mtg, plant, parameters):
        L = mtg.labels.SubOrgan
        seed_organ = mtg.add_organ(plant, "StemInternode", complex_label="Shoot")
        seed = mtg.add_component(seed_organ, **PropsConfig(scale=mtg.scales.SubOrgan, edge_type="/", label=L.Seed))
        for name, value in (("length", SEED_LENGTH), ("stem_radius", STEM_RADIUS)):
            mtg.set_values(name, [seed], [value])
        mtg.set_values("distance_from_tip", [seed], [0.])
        phytomer = mtg.complex(seed_organ)
        organ = lambda label: mtg.add_component(phytomer, **PropsConfig(scale=mtg.scales.Organ, edge_type="/",
                                                                      label=getattr(mtg.labels.Organ, label)))
        subcrown = _following(mtg, organ("StemInternode"), L.StemElement, [ELEMENT, ELEMENT], seed)
        internode = organ("StemInternode")
        (internode_hidden,) = _following(mtg, internode, L.HiddenElement, [HIDDEN], subcrown[-1])
        (crown,) = _following(mtg, internode, L.StemElement, [ELEMENT], internode_hidden)
        sheath = organ("Sheath")
        (sheath_hidden,) = _following(mtg, sheath, L.HiddenElement, [HIDDEN], crown, edge="+")
        (sheath_element,) = _following(mtg, sheath, L.StemElement, [cls.sheath_length], sheath_hidden)
        blade = organ("Blade")
        (blade_hidden,) = _following(mtg, blade, L.HiddenElement, [HIDDEN], sheath_element)
        leaf = _following(mtg, blade, L.LeafElement, list(cls.leaf_lengths), blade_hidden)
        seminal = segment_chain(mtg, mtg.add_organ(plant, "RootInternode", complex_label="Root"), L.RootSegment,
                                [(ROOT_RADIUS, ELEMENT, ELEMENT), (ROOT_RADIUS, ELEMENT, 0.)], bearer=seed)
        nodal = segment_chain(mtg, mtg.add_organ(plant, "RootInternode", complex_label="Root"), L.RootSegment,
                              [(ROOT_RADIUS, ELEMENT, 0.)], bearer=crown)
        roles = dict(seed=[seed], subcrown=subcrown, hidden=[internode_hidden, sheath_hidden, blade_hidden],
                     crown=[crown], sheath=[sheath_element], leaf=leaf, seminal=seminal, nodal=nodal)
        for role, vids in roles.items():
            for v in vids:
                mtg.set_values("role", [v], [role])


@pytest.fixture(scope="module")
def plants():
    table = planting([Planting(x=0.), Planting(x=0.1)], None, plant_scenarios=[WHOLE_SHOOT, NO_SHEATH])
    population = initiated(table, WheatSeedling, GranapAnatomy)
    return population.data_structure


def _roles(mtg, plant):
    role = mtg.property("role")
    out = {}
    for v in mtg.components_at_scale(plant, scale=mtg.scales.SubOrgan):
        out.setdefault(role[int(v)], []).append(int(v))
    return {name: sorted(vids) for name, vids in out.items()}


def _family(mtg, vids):
    family = mtg.values_of("family", vids)
    return np.array([family.get(v, -1) for v in vids])


def _expected_junctions(mtg, sections):
    """{(ancestor section, section, family): the vessel pairs}: each section with an anatomy joined to its nearest
    ancestor with one."""
    owners = mtg.compartments_by_owner(mtg.scales.SubOrgan)
    out = {}
    for section in sections:
        ancestor, _ = anatomy_ancestor(mtg, section, mtg.compartments_csr(mtg.scales.SubOrgan))
        if ancestor is not None and owners.get(section):
            for family in (0, 1):
                out[(ancestor, section, family)] = vessel_pairs(mtg, owners[ancestor], owners[section], family)
    return out


def _actual_junctions(mtg, data_structure, sections):
    a, b = mtg.property("n_id_a"), mtg.property("n_id_b")
    junctions = [int(e) for e in mtg.junction_vids()]
    tails, heads = [int(a[e]) for e in junctions], [int(b[e]) for e in junctions]
    parent_sections, child_sections = data_structure.sections_of(tails), data_structure.sections_of(heads)
    families = _family(mtg, tails)
    out = {}
    for tail, head, parent, child, family in zip(tails, heads, parent_sections.tolist(), child_sections.tolist(),
                                                 families.tolist()):
        if child in sections:
            out.setdefault((parent, child, family), []).append((tail, head))
    return {key: sorted(pairs) for key, pairs in out.items()}


@pytest.mark.parametrize("plant_index", [0, 1], ids=["whole shoot", "sheath without anatomy"])
def test_the_seed_gets_the_stems_anatomy_and_the_hidden_elements_none(plants, plant_index):
    data_structure = plants
    mtg = data_structure.structure
    plant = int(data_structure.entity_ids("Plant")[plant_index])
    roles = _roles(mtg, plant)
    template = mtg.property("anatomy_template")
    assert template[roles["seed"][0]] == template[roles["subcrown"][0]] >= 0       # the stem preset, one class
    owners = mtg.compartments_by_owner(mtg.scales.SubOrgan)
    assert all(not owners.get(v) and template.get(v, -1.) == -1. for v in roles["hidden"])
    assert bool(owners.get(roles["sheath"][0])) == (plant_index == 0)


@pytest.mark.parametrize("plant_index", [0, 1], ids=["whole shoot", "sheath without anatomy"])
def test_the_conduits_are_joined_from_the_root_tips_to_the_leaf_across_the_hidden_elements(plants, plant_index):
    data_structure = plants
    mtg = data_structure.structure
    plant = int(data_structure.entity_ids("Plant")[plant_index])
    roles = _roles(mtg, plant)
    owners = mtg.compartments_by_owner(mtg.scales.SubOrgan)
    sections = sorted(v for vids in roles.values() for v in vids if owners.get(v))
    expected, actual = _expected_junctions(mtg, sections), _actual_junctions(mtg, data_structure, set(sections))
    assert actual == {key: pairs for key, pairs in expected.items() if pairs}       # each pair once, nothing else
    # the pass-through junctions: across the internode's hidden element, the sheath's (and its element without
    # anatomy), the blade's
    crossings = {(roles["subcrown"][-1], roles["crown"][0])}
    crossings |= {(roles["crown"][0], roles["sheath"][0]), (roles["sheath"][0], roles["leaf"][0])} \
        if plant_index == 0 else {(roles["crown"][0], roles["leaf"][0])}
    for parent, child in crossings:
        for family in (0, 1):
            assert len(actual[(parent, child, family)]) >= max(len(_vessels(mtg, owners[parent], family)),
                                                              len(_vessels(mtg, owners[child], family))) > 0
    # every conduit is joined to its neighbours' (upstream and downstream), and the conduits form one network from
    # the root tips to the leaf's tip, by family
    conduits = {v for s in sections for v in owners[s] if _family(mtg, [v])[0] in (0, 1)}
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
        nodes = sorted(v for v in conduits if _family(mtg, [v])[0] == family)
        index = {v: i for i, v in enumerate(nodes)}
        edges = np.array([(index[a], index[b]) for (_, _, f), pairs in actual.items() if f == family
                          for a, b in pairs]).reshape(-1, 2)
        graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(nodes), len(nodes)))
        _, labels = connected_components(graph, directed=False)
        component = lambda section: {int(labels[index[v]]) for v in owners[section] if v in index}
        leaf_tip = component(roles["leaf"][-1])
        for tip in (roles["seminal"][-1], roles["nodal"][-1]):
            assert component(tip) & leaf_tip, f"family {family}: the root tip {tip} is not joined to the leaf"


def _vessels(mtg, compartments, family):
    return [v for v, f in zip(compartments, _family(mtg, compartments)) if f == family]


def test_a_pass_through_junction_spans_the_hidden_elements(plants):
    data_structure = plants
    mtg = data_structure.structure
    plant = int(data_structure.entity_ids("Plant")[1])
    roles = _roles(mtg, plant)
    a, b, length = mtg.property("n_id_a"), mtg.property("n_id_b"), mtg.property("edge_length")
    owners = mtg.compartments_by_owner(mtg.scales.SubOrgan)
    crown, leaf = set(owners[roles["crown"][0]]), set(owners[roles["leaf"][0]])
    spanning = [e for e in mtg.junction_vids() if int(a[e]) in crown and int(b[e]) in leaf]
    # half the crown element, the sheath's hidden element and element (no anatomy), the blade's hidden element, half
    # the leaf element (mm)
    assert spanning and np.allclose([length[e] for e in spanning],
                                    (0.5 * (ELEMENT + ELEMENT) + HIDDEN + ELEMENT + HIDDEN) * 1e3)


def test_a_replaced_ancestor_is_joined_again_across_the_hidden_elements():
    """The DataStructure rewires a replaced section's children only: the component joins the replaced section to
    its descendants across the hidden elements; nothing dangles, nothing is doubled; the junctions' length is kept as
    the elements grow."""
    table = planting([Planting()], None, plant_scenarios=[NO_SHEATH])
    data_structure = initiated(table, WheatSeedling, GranapAnatomy).data_structure
    mtg = data_structure.structure
    WheatSeedling(data_structure=data_structure)                  # first: it holds the segments' lengths, the hidden elements'
    anatomy = GranapAnatomy(data_structure=data_structure)
    plant = int(data_structure.entity_ids("Plant")[0])
    roles = _roles(mtg, plant)
    top = roles["subcrown"][-1]
    stem_radius = data_structure.get("stem_radius").copy()
    stem_radius[data_structure.entity_ids("SubOrgan") == top] = 1.5 * STEM_RADIUS            # another class: replaced
    data_structure.set("stem_radius", stem_radius)
    anatomy()
    assert anatomy.last_update == {"emerged": [], "replaced": [top]}
    owners = mtg.compartments_by_owner(mtg.scales.SubOrgan)
    sections = sorted(v for vids in roles.values() for v in vids if owners.get(v))
    expected, actual = _expected_junctions(mtg, sections), _actual_junctions(mtg, data_structure, set(sections))
    assert actual == {key: pairs for key, pairs in expected.items() if pairs}
    assert all(actual[(top, roles["crown"][0], family)] for family in (0, 1))
    a, b = mtg.property("n_id_a"), mtg.property("n_id_b")
    assert all(mtg.has_vertex(int(a[e])) and mtg.has_vertex(int(b[e])) for e in mtg.junction_vids())
    edge_length = data_structure.get("edge_length")
    edges = np.asarray(data_structure.edges(), dtype=np.int64).reshape(-1, 2)
    across = np.isin(edges[:, 0], owners[top]) & np.isin(edges[:, 1], owners[roles["crown"][0]])
    assert across.any() and np.allclose(edge_length[across], (0.5 * (ELEMENT + ELEMENT) + HIDDEN) * 1e3)
    anatomy()                                                                    # a step more: kept
    assert np.allclose(data_structure.get("edge_length")[across], (0.5 * (ELEMENT + ELEMENT) + HIDDEN) * 1e3)


class UngrownSheathSeedling(WheatSeedling):
    """The seedling, its sheath's StemElement not grown yet (length 0, as Adel's before it elongates)."""
    sheath_length = 0.


def test_a_section_without_length_gets_its_anatomy_when_it_grows():
    """A section of zero length has no anatomy (its section would have no radial conductance): a pass-through, the
    junctions spanning it, as a hidden element. Once it has a length, it gets its anatomy, joined to its nearest
    ancestor and descendant with one; the junctions across it are gone; the conduits stay one network."""
    table = planting([Planting()], None, plant_scenarios=[WHOLE_SHOOT])
    data_structure = initiated(table, UngrownSheathSeedling, GranapAnatomy).data_structure
    mtg = data_structure.structure
    plant = int(data_structure.entity_ids("Plant")[0])
    roles = _roles(mtg, plant)
    sheath, crown, leaf = roles["sheath"][0], roles["crown"][0], roles["leaf"][0]
    owners = mtg.compartments_by_owner(mtg.scales.SubOrgan)
    assert not owners.get(sheath) and mtg.property("anatomy_template").get(sheath, -1.) == -1.
    sections = sorted(v for vids in roles.values() for v in vids if owners.get(v))
    actual = _actual_junctions(mtg, data_structure, set(sections))
    assert all(actual[(crown, leaf, family)] for family in (0, 1))                  # across the ungrown sheath
    UngrownSheathSeedling(data_structure=data_structure)
    anatomy = GranapAnatomy(data_structure=data_structure)
    lengths = data_structure.get("length").copy()
    lengths[data_structure.entity_ids("SubOrgan") == sheath] = ELEMENT                           # it grows
    data_structure.set("length", lengths)
    anatomy()
    assert anatomy.last_update == {"emerged": [sheath], "replaced": []}
    owners = mtg.compartments_by_owner(mtg.scales.SubOrgan)
    sections = sorted(v for vids in roles.values() for v in vids if owners.get(v))
    assert sheath in sections
    expected, actual = _expected_junctions(mtg, sections), _actual_junctions(mtg, data_structure, set(sections))
    assert actual == {key: pairs for key, pairs in expected.items() if pairs}       # each pair once, nothing else
    assert all(actual[(crown, sheath, family)] and actual[(sheath, leaf, family)] for family in (0, 1))
    assert not any(key[:2] == (crown, leaf) for key in actual)
    a, b = mtg.property("n_id_a"), mtg.property("n_id_b")
    assert all(mtg.has_vertex(int(a[e])) and mtg.has_vertex(int(b[e])) for e in mtg.junction_vids())


class UngrownBladeBaseSeedling(WheatSeedling):
    """The seedling, its blade's basal LeafElement not grown yet (length 0)."""
    leaf_lengths = (0., ELEMENT)


def test_the_leaf_surface_is_on_the_basal_element_with_an_anatomy():
    """A leaf organ's surface is owned by its basal element with an anatomy: not by an element without length (it has
    no anatomy, MECHA no height for it); once that element grows, the surface moves to it, nothing dangling."""
    table = planting([Planting()], None, plant_scenarios=[WHOLE_SHOOT])
    data_structure = initiated(table, UngrownBladeBaseSeedling, GranapAnatomy).data_structure
    mtg = data_structure.structure
    plant = int(data_structure.entity_ids("Plant")[0])
    first, second = _roles(mtg, plant)["leaf"]
    organ = int(mtg.complex_at_scale(first, mtg.scales.Organ))

    def surface_owner():
        surface = mtg.property("leaf_surface")[organ]
        owners = mtg.compartments_by_owner(mtg.scales.SubOrgan)
        return [v for v in (first, second) if surface in owners.get(v, [])]

    assert not mtg.compartments_by_owner(mtg.scales.SubOrgan).get(first) and surface_owner() == [second]
    UngrownBladeBaseSeedling(data_structure=data_structure)
    anatomy = GranapAnatomy(data_structure=data_structure)
    lengths = data_structure.get("length").copy()
    lengths[data_structure.entity_ids("SubOrgan") == first] = ELEMENT                            # it grows
    data_structure.set("length", lengths)
    anatomy()
    assert anatomy.last_update["emerged"] == [first] and surface_owner() == [first]
    assert len(mtg.property("leaf_surface")) == 2                                      # the blade's and the sheath's
    a, b = mtg.property("n_id_a"), mtg.property("n_id_b")
    stomatal = [c for c, path in mtg.property("network_path").items() if path == 6 and mtg.has_vertex(int(c))]
    assert all(mtg.has_vertex(int(a[c])) and mtg.has_vertex(int(b[c])) for c in stomatal)
