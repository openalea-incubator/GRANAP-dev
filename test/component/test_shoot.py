"""
Stems and leaves too: one library per kind of segment (GranapAnatomy.shoot), each segment its kind's anatomy, and
vessel junctions across the stem-leaf and stem-root links.
"""
from openalea.metafspm.coupling.component import StructuralComponent
from openalea.metafspm.scene.population import Planting, planting

from openalea.granap.stem_class import StemAnatomy
from openalea.granap.leaf_class import LeafAnatomy
from openalea.granap.metafspm.architectures import segment_chain
from openalea.granap.metafspm.component import GranapAnatomy, OrganAnatomy
from openalea.granap.metafspm.library import AnatomyLibrary, preset_parameters, template_by_index

from roots import initiated

SEGMENT = 5e-3                      # m


def _elements(n, radius):
    """n elements of *radius*, base to tip: [(radius, length, distance from the tip)]."""
    return [(radius, SEGMENT, (n - 1 - rank) * SEGMENT) for rank in range(n)]


class SmallShoot(StructuralComponent):
    @classmethod
    def initiate_plant(cls, mtg, plant, parameters):
        L = mtg.labels.SubOrgan
        stem = segment_chain(mtg, mtg.add_organ(plant, "StemInternode", complex_label="Shoot"), L.StemElement,
                             _elements(2, 2.5e-3))
        segment_chain(mtg, mtg.add_organ(plant, "Leaf", complex_label="Shoot"), L.LeafElement, _elements(1, 2.5e-3),
                      bearer=stem[1])
        segment_chain(mtg, mtg.add_organ(plant, "RootInternode", complex_label="Root"), L.RootSegment,
                      _elements(1, 0.55e-3), bearer=stem[0])


class ShootAnatomy(GranapAnatomy):
    library = AnatomyLibrary()
    shoot = {"StemElement": OrganAnatomy(AnatomyLibrary(parameters=preset_parameters("for_monocot_stem"),
                                                        organ_class=StemAnatomy)),
             "LeafElement": OrganAnatomy(AnatomyLibrary(parameters=preset_parameters("for_monocot_leaf"),
                                                        organ_class=LeafAnatomy), size=None)}


def test_stems_leaves_and_roots_get_their_kinds_anatomy_and_are_joined():
    table = planting([Planting()], None)
    population = initiated(table, SmallShoot, ShootAnatomy)
    mtg = population.data_structure.structure
    data_structure = population.data_structure
    L, labels = mtg.labels.SubOrgan, mtg.property("label")
    segments = sorted(int(v) for v in mtg.property("anatomy_class"))
    kinds = {s: labels[s] for s in segments}
    assert sorted(kinds.values()) == sorted([L.StemElement] * 2 + [L.LeafElement, L.RootSegment])
    template = mtg.property("template")
    owners = mtg.compartments_by_owner(mtg.scales.SubOrgan)
    by_kind = {}
    for s in segments:
        by_kind.setdefault(kinds[s], set()).add(int(template[owners[s][0]]))
    libraries = {L.StemElement: ShootAnatomy.shoot["StemElement"].library,
                 L.LeafElement: ShootAnatomy.shoot["LeafElement"].library, L.RootSegment: ShootAnatomy.library}
    for kind, ids in by_kind.items():                                   # each kind from its own library
        assert all(isinstance(template_by_index(mtg, index).organ, libraries[kind].organ_class) for index in ids)
    a, b = mtg.property("n_id_a"), mtg.property("n_id_b")
    section_of = {v: s for s in segments for v in owners[s]}
    linked = {(kinds[section_of[int(a[e])]], kinds[section_of[int(b[e])]]) for e in mtg.junction_vids()}
    assert {(L.StemElement, L.StemElement), (L.StemElement, L.LeafElement), (L.StemElement, L.RootSegment)} <= linked
    assert data_structure.n_nodes() == sum(len(owners[s]) for s in segments)


class TwoElementLeaf(StructuralComponent):
    @classmethod
    def initiate_plant(cls, mtg, plant, parameters):
        L = mtg.labels.SubOrgan
        stem = segment_chain(mtg, mtg.add_organ(plant, "StemInternode", complex_label="Shoot"), L.StemElement,
                             _elements(1, 2.5e-3))
        segment_chain(mtg, mtg.add_organ(plant, "Leaf", complex_label="Shoot"), L.LeafElement, _elements(2, 2.5e-3),
                      bearer=stem[0])


def test_leaf_surface_and_stomatal_connections():
    """
    Each leaf organ gets one surface Compartment (kind 3), owned by its basal element, joined to every
    stomatal pore of its sections (network_path 6); the pores are air (wall_air walls, no plasmodesmata, one air link
    to their chamber) with their aperture's width and depth; rebuilding the surface leaves nothing dangling.
    """
    table = planting([Planting()], None)
    population = initiated(table, TwoElementLeaf, ShootAnatomy)
    mtg = population.data_structure.structure
    data_structure = population.data_structure
    L, s = mtg.labels.SubOrgan, mtg.scales
    labels = mtg.property("label")
    leaves = sorted(int(v) for v in mtg.property("anatomy_class") if labels[int(v)] == L.LeafElement)
    organs = {int(mtg.complex_at_scale(v, s.Organ)) for v in leaves}
    assert len(leaves) == 2 and len(organs) == 1
    surfaces = mtg.property("leaf_surface")
    surface = surfaces[organs.pop()]
    kind, tissue, path = mtg.property("compartment_type"), mtg.property("tissue"), mtg.property("network_path")
    assert kind[surface] == 3 and tissue[surface] == "leaf surface"
    basal = next(v for v in leaves if mtg.parent(v) not in set(leaves))
    assert mtg.compartments_by_owner(s.SubOrgan)[basal].count(surface) == 1
    a, b = mtg.property("n_id_a"), mtg.property("n_id_b")
    stomatal = [c for c in path if path[c] == 6]
    pores = sorted(int(v) for v in tissue if tissue[v] == "pore" and mtg.scale(int(v)) == s.Compartment)
    assert len(pores) > 0 and sorted(int(a[c]) for c in stomatal) == pores and {int(b[c]) for c in stomatal} == {surface}
    width, depth = mtg.property("pore_width"), mtg.property("pore_depth")
    assert all(0 < width[p] <= depth[p] for p in pores)
    # the pores are air nodes: no plasmodesmata, wall_air walls, one air link (to an air space)
    edges_of = {}
    for c in path:
        if mtg.scale(c) == s.Connection and c in a:
            for end, other in ((a[c], b[c]), (b[c], a[c])):
                edges_of.setdefault(int(end), []).append((int(path[c]), int(other)))
    for p in pores:
        kinds = [k for k, _ in edges_of[p]]
        assert 3 not in kinds and 0 not in kinds and kinds.count(4) == 1 and 1 in kinds
        assert tissue[next(o for k, o in edges_of[p] if k == 4)] == "air space"
    # rebuilt: the same surface structure, nothing left of the old one
    n_edges = data_structure.n_edges()
    ShootAnatomy.update_leaf_surfaces(mtg, leaves)
    path = mtg.property("network_path")
    assert sum(1 for c in path if path[c] == 6) == len(pores)
    assert not mtg.has_vertex(surface) and len(mtg.property("leaf_surface")) == 1
