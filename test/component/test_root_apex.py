"""
Root apices (SubOrgan label RootApex, ArchiSimple's axis tips: the part of the root beyond its last segment): a kind of
section of their own, root_apex_anatomy, by default the root segments' anatomy of their class. An apex with a length
gets it, its soil node with the segments', joined to its parent segment as a segment is; an apex without length
(a primordium) has none until it grows. An apex the architecture relabels a segment (ArchiSimple cutting it off the
tip) keeps its anatomy when the segments' kind gives the same, else has it replaced. A reduction selects its
sections by a filter of the SubOrgans (where=): a sheath's elements are StemElements, reduced with the stems, or
apart from them by their organ's label ("Organ.label"); GRANAP warns of a label it gives no anatomy.
"""
import warnings

import numpy as np
import pytest

from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.metafspm.scene.population import Planting, planting
from openalea.metafspm.solve.reduction import Reduction

from openalea.granap.metafspm.architectures import SegmentArchitecture, segment_chain
from openalea.granap.metafspm.component import GranapAnatomy, anatomy_ancestor, vessel_pairs

from roots import initiated
from test_wheat_plant import WHOLE_SHOOT, WheatSeedling

RADIUS, SEGMENT, APEX = 0.4e-3, 4.5e-3, 2e-3              # m


class RootWithApices(SegmentArchitecture):
    """A main root of two segments ending in an apex APEX long, and a lateral on its first segment: an apex without
    length (a primordium). The roles in the property "role"."""
    apex_length = APEX

    @classmethod
    def initiate_plant(cls, g, plant, parameters):
        L = g.labels.SubOrgan
        main = segment_chain(g, g.add_organ(plant, "RootInternode", kind="Root"), L.RootSegment,
                             [(RADIUS, SEGMENT, SEGMENT + APEX), (RADIUS, SEGMENT, APEX)])
        apex = g.add_child(main[-1], **PropsConfig(scale=g.scales.SubOrgan, edge_type="<", label=L.RootApex))
        lateral_organ = g.add_organ(plant, "RootInternode", kind="Root")
        primordium = g.add_component_with_topo(lateral_organ, main[0], **PropsConfig(
            scale=g.scales.SubOrgan, edge_type="+", label=L.RootApex))
        for vertex, length in ((apex, cls.apex_length), (primordium, 0.)):
            for name, value in (("radius", RADIUS), ("length", length)):
                g.set_values(name, [vertex], [value])
            g.set_values("distance_from_tip", [vertex], [0.])
        for role, vertices in dict(main=main, apex=[apex], primordium=[primordium]).items():
            for vertex in vertices:
                g.set_values("role", [vertex], [role])


def _roles(g):
    out = {}
    for vertex, role in g.property("role").items():
        out.setdefault(role, []).append(int(vertex))
    return {role: sorted(vertices) for role, vertices in out.items()}


def _built(scenario=None, architecture=RootWithApices, reduction=None):
    table = planting([Planting()], None, plant_scenarios=[dict(scenario or {})])
    ds = initiated(table, architecture, GranapAnatomy, reduction=reduction).data_structure
    return ds, ds.structure


def _junctions(g, ds, parent, child):
    """The junctions from *parent*'s section to *child*'s, by family."""
    a, b = g.property("n_id_a"), g.property("n_id_b")
    junctions = [int(e) for e in g.junction_vids()]
    tails, heads = [int(a[e]) for e in junctions], [int(b[e]) for e in junctions]
    parents, children = ds.sections_of(tails).tolist(), ds.sections_of(heads).tolist()
    family = g.values_of("family", tails)
    out = {0: [], 1: []}
    for tail, head, p, c in zip(tails, heads, parents, children):
        if (p, c) == (parent, child):
            out[int(family[tail])].append((tail, head))
    return {f: sorted(pairs) for f, pairs in out.items()}


def test_an_apex_with_a_length_gets_the_root_segments_anatomy_and_is_joined_to_its_segment():
    ds, g = _built({"soil_node": True})
    roles = _roles(g)
    (apex,), (primordium,), main = roles["apex"], roles["primordium"], roles["main"]
    owners = g.compartments_by_owner(g.scales.SubOrgan)
    template = g.property("anatomy_template")
    assert owners.get(apex) and template[apex] == template[main[-1]] >= 0      # the segments' class: one template
    assert g.property("anatomy_kind")[apex] == "RootApex"
    assert not owners.get(primordium) and template.get(primordium, -1.) == -1.  # no length: none yet
    kind = g.values_of("kind", owners[apex])
    assert sum(1 for v in owners[apex] if kind.get(v) == 4.) == 1               # its soil node
    junctions = _junctions(g, ds, main[-1], apex)
    for family in (0, 1):
        assert junctions[family] == vessel_pairs(g, owners[main[-1]], owners[apex], family) != []
    assert anatomy_ancestor(g, apex, g.compartments_csr(g.scales.SubOrgan)) == (main[-1], [])


def test_an_apex_without_length_gets_its_anatomy_when_it_grows():
    ds, g = _built()
    roles = _roles(g)
    (primordium,), main = roles["primordium"], roles["main"]
    RootWithApices(data_structure=ds)
    anatomy = GranapAnatomy(data_structure=ds)
    lengths = ds.get("length").copy()
    lengths[ds.entity_ids("SubOrgan") == primordium] = APEX
    ds.set("length", lengths)
    anatomy()
    assert anatomy.last_update == {"emerged": [primordium], "replaced": []}
    owners = g.compartments_by_owner(g.scales.SubOrgan)
    junctions = _junctions(g, ds, main[0], primordium)
    assert all(junctions[family] == vessel_pairs(g, owners[main[0]], owners[primordium], family) != []
               for family in (0, 1))


def test_root_apex_anatomy_is_on_top_of_root_anatomy():
    """The apex's scenario dict on top of the root's (its sections' fields over theirs); given, its own template."""
    root = {"cortex": {"n_layers": 3}}
    ds, g = _built({"root_anatomy": root, "root_apex_anatomy": {"cortex": {"n_layers": 2}, "seed": 1}})
    roles = _roles(g)
    (apex,), main = roles["apex"], roles["main"]
    assert GranapAnatomy.anatomy_settings(g, main[0]) == root
    assert GranapAnatomy.anatomy_settings(g, apex) == {"cortex": {"n_layers": 2}, "seed": 1}
    template = g.property("anatomy_template")
    assert template[apex] >= 0 and template[apex] != template[main[-1]]
    same, g_same = _built({"root_anatomy": root})
    assert GranapAnatomy.anatomy_settings(g_same, _roles(g_same)["apex"][0]) == root


@pytest.mark.parametrize("apex_anatomy, replaced", [(None, False), ({"cortex": {"n_layers": 2}}, True)],
                         ids=["the segments' anatomy: kept", "an apex anatomy of its own: replaced"])
def test_an_apex_relabelled_a_segment_keeps_or_replaces_its_anatomy(apex_anatomy, replaced):
    scenario = {"soil_node": True} if apex_anatomy is None else {"soil_node": True, "root_apex_anatomy": apex_anatomy}
    ds, g = _built(scenario)
    (apex,), main = _roles(g)["apex"], _roles(g)["main"]
    RootWithApices(data_structure=ds)
    anatomy = GranapAnatomy(data_structure=ds)
    before = g.property("anatomy_template")[apex]
    g.set_values("label", [apex], [g.labels.SubOrgan.RootSegment])            # ArchiSimple cuts it off the tip
    anatomy()
    assert anatomy.last_update == {"emerged": [], "replaced": [apex] if replaced else []}
    assert g.property("anatomy_kind")[apex] == "RootSegment"
    template = g.property("anatomy_template")
    assert (template[apex] == template[main[-1]]) and ((template[apex] != before) == replaced)
    owners = g.compartments_by_owner(g.scales.SubOrgan)
    junctions = _junctions(g, ds, main[-1], apex)
    assert all(junctions[family] == vessel_pairs(g, owners[main[-1]], owners[apex], family) != []
               for family in (0, 1))


def _reduction(where):
    return Reduction(where=where, ports="GranapAnatomy.conduits")


def _reduced_sections(g):
    instances = g.instances if g.has_instances() else None
    return set() if instances is None else {int(owner) for owner in instances.owner.tolist()}


def test_the_stem_elements_are_reduced_with_their_sheaths():
    """where={"label": "StemElement"}: every stem element, a sheath's too (no organ label trick); no warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ds, g = _built(WHOLE_SHOOT, WheatSeedling, _reduction({"label": "StemElement"}))
    assert not [w for w in caught if "reduction's labels" in str(w.message)]
    roles = _roles(g)
    reduced = _reduced_sections(g)
    assert set(roles["sheath"]) <= reduced and set(roles["crown"]) <= reduced
    assert not set(roles["leaf"]) & reduced


def test_the_sheaths_apart_from_the_stems_by_their_organ():
    """where= filters across scales: the stem elements of the internodes (Organ.label), not the sheaths'."""
    ds, g = _built(WHOLE_SHOOT, WheatSeedling, _reduction({"label": "StemElement", "Organ.label": "StemInternode"}))
    roles = _roles(g)
    reduced = _reduced_sections(g)
    assert set(roles["crown"]) <= reduced and not set(roles["sheath"]) & reduced


def test_a_label_without_anatomy_warns():
    with pytest.warns(UserWarning, match="no SubOrgan label with a GRANAP anatomy"):
        _built({}, RootWithApices, _reduction({"label": ("RootSegment", "HiddenElement")}))


def test_the_junctions_of_reduced_sections_are_their_templates_pairs_computed_once():
    """Reduced sections (their kept nodes, a part of their template's) are joined as their whole vessels would be:
    vessel_pairs gives the pairs of the direct rule (_vessel_pairs), computed once per pair of templates and nodes
    (the population's cache) and given each section's vids."""
    from openalea.granap.metafspm.component import _table_values, _vessel_pairs
    ds, g = _built(reduction=_reduction({"label": "RootSegment"}))
    main = _roles(g)["main"]
    assert set(main) <= _reduced_sections(g)
    owners = g.compartments_by_owner(g.scales.SubOrgan)
    parent, child = owners[main[0]], owners[main[1]]
    for section in (parent, child):                              # one template's nodes (a soil node may have none)
        has, templates = _table_values(g, "template", np.asarray(section, dtype=np.int64))
        assert has.sum() >= len(section) - 1 and np.unique(templates[has]).size == 1
    cache = g.objects("granap_vessel_pairs", saved=False)
    for family in (0, 1):
        direct = _vessel_pairs(g, parent, child, family)
        assert direct and vessel_pairs(g, parent, child, family) == direct
        assert _junctions(g, ds, main[0], main[1])[family] == direct
    entries = len(cache)
    assert entries > 0                                           # cached, not computed by the direct rule each time
    assert vessel_pairs(g, parent, child, 0) == _vessel_pairs(g, parent, child, 0) and len(cache) == entries
