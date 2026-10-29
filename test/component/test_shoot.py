"""
Stems and leaves too: one library per kind of segment (GranapAnatomy.shoot), each segment its kind's anatomy, and
vessel junctions across the stem-leaf and stem-root links.
"""
import os
import sys

import pandas as pd

from openalea.metafspm.data_structure.data_api import MPGDataStructure
from openalea.metafspm.scene.population import build_population

from openalea.granap.stem_class import StemAnatomy
from openalea.granap.leaf_class import LeafAnatomy
from openalea.granap.metafspm.component import GranapAnatomy, OrganAnatomy, junction_rules
from openalea.granap.metafspm.library import AnatomyLibrary, preset_parameters

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "example", "mpg"))
from seedling import _chain, _organ                                                    # noqa: E402

from openalea.metafspm.coupling.component import StructuralComponent                  # noqa: E402


class SmallShoot(StructuralComponent):
    @classmethod
    def initiate_plant(cls, g, plant, parameters):
        L = g.labels.SubOrgan
        stem = _chain(g, _organ(g, plant, "Shoot", "StemInternode"), 2, L.StemElement, 2.5e-3)
        _chain(g, _organ(g, plant, "Shoot", "Leaf"), 1, L.LeafElement, 2.5e-3, bearer=stem[1])
        _chain(g, _organ(g, plant, "Root", "RootInternode"), 1, L.RootSegment, 0.55e-3, bearer=stem[0])


class ShootAnatomy(GranapAnatomy):
    library = AnatomyLibrary()
    shoot = {"StemElement": OrganAnatomy(AnatomyLibrary(parameters=preset_parameters("for_monocot_stem"),
                                                        organ_class=StemAnatomy)),
             "LeafElement": OrganAnatomy(AnatomyLibrary(parameters=preset_parameters("for_monocot_leaf"),
                                                        organ_class=LeafAnatomy), size=None)}


def test_stems_leaves_and_roots_get_their_kinds_anatomy_and_are_joined():
    table = pd.DataFrame([dict(plant="p0", model=None, x=0., y=0., z=0., rotation=0., scenario={"parameters": {}})])
    g, _ = build_population(table, initiators=(SmallShoot, ShootAnatomy))
    g.convert_properties_to_arraydict()
    ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan, nodes="Compartment", wiring=junction_rules(g))
    L, labels = g.labels.SubOrgan, g.property("label")
    segments = sorted(int(v) for v in g.property("anatomy_class"))
    kinds = {s: labels[s] for s in segments}
    assert sorted(kinds.values()) == sorted([L.StemElement] * 2 + [L.LeafElement, L.RootSegment])
    template = g.property("template")
    owners = g.compartments_by_owner(g.scales.SubOrgan)
    by_kind = {}
    for s in segments:
        by_kind.setdefault(kinds[s], set()).add(int(template[owners[s][0]]))
    libraries = {L.StemElement: ShootAnatomy.shoot["StemElement"].library,
                 L.LeafElement: ShootAnatomy.shoot["LeafElement"].library, L.RootSegment: ShootAnatomy.library}
    for kind, ids in by_kind.items():                                   # each kind from its own library
        assert ids == {t.index for t in libraries[kind].templates.values()}
    a, b = g.property("n_id_a"), g.property("n_id_b")
    section_of = {v: s for s in segments for v in owners[s]}
    linked = {(kinds[section_of[int(a[e])]], kinds[section_of[int(b[e])]]) for e in g.junction_vids()}
    assert {(L.StemElement, L.StemElement), (L.StemElement, L.LeafElement), (L.StemElement, L.RootSegment)} <= linked
    assert ds.n_nodes() == sum(len(owners[s]) for s in segments)


class TwoElementLeaf(StructuralComponent):
    @classmethod
    def initiate_plant(cls, g, plant, parameters):
        L = g.labels.SubOrgan
        stem = _chain(g, _organ(g, plant, "Shoot", "StemInternode"), 1, L.StemElement, 2.5e-3)
        _chain(g, _organ(g, plant, "Shoot", "Leaf"), 2, L.LeafElement, 2.5e-3, bearer=stem[0])


def test_leaf_surface_and_stomatal_connections():
    """
    Each leaf organ gets one surface Compartment (kind 3), owned by its basal element, joined to every
    stomatal pore of its sections (network_path 6); the pores are air (wall_air walls, no plasmodesmata, one air link
    to their chamber) with their aperture's width and depth; rebuilding the surface leaves nothing dangling.
    """
    table = pd.DataFrame([dict(plant="p0", model=None, x=0., y=0., z=0., rotation=0., scenario={"parameters": {}})])
    g, _ = build_population(table, initiators=(TwoElementLeaf, ShootAnatomy))
    g.convert_properties_to_arraydict()
    ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan, nodes="Compartment", wiring=junction_rules(g))
    L, s = g.labels.SubOrgan, g.scales
    labels = g.property("label")
    leaves = sorted(int(v) for v in g.property("anatomy_class") if labels[int(v)] == L.LeafElement)
    organs = {int(g.complex_at_scale(v, s.Organ)) for v in leaves}
    assert len(leaves) == 2 and len(organs) == 1
    surfaces = g.property("leaf_surface")
    surface = surfaces[organs.pop()]
    kind, tissue, path = g.property("kind"), g.property("tissue"), g.property("network_path")
    assert kind[surface] == 3 and tissue[surface] == "leaf surface"
    basal = next(v for v in leaves if g.parent(v) not in set(leaves))
    assert g.compartments_by_owner(s.SubOrgan)[basal].count(surface) == 1
    a, b = g.property("n_id_a"), g.property("n_id_b")
    stomatal = [c for c in path if path[c] == 6]
    pores = sorted(int(v) for v in tissue if tissue[v] == "pore" and g.scale(int(v)) == s.Compartment)
    assert len(pores) > 0 and sorted(int(a[c]) for c in stomatal) == pores and {int(b[c]) for c in stomatal} == {surface}
    width, depth = g.property("pore_width"), g.property("pore_depth")
    assert all(0 < width[p] <= depth[p] for p in pores)
    # the pores are air nodes: no plasmodesmata, wall_air walls, one air link (to an air space)
    edges_of = {}
    for c in path:
        if g.scale(c) == s.Connection and c in a:
            for end, other in ((a[c], b[c]), (b[c], a[c])):
                edges_of.setdefault(int(end), []).append((int(path[c]), int(other)))
    for p in pores:
        kinds = [k for k, _ in edges_of[p]]
        assert 3 not in kinds and 0 not in kinds and kinds.count(4) == 1 and 1 in kinds
        assert tissue[next(o for k, o in edges_of[p] if k == 4)] == "air space"
    # rebuilt: the same surface structure, nothing left of the old one
    n_edges = ds.n_edges()
    ShootAnatomy.update_leaf_surfaces(g, leaves)
    path = g.property("network_path")
    assert sum(1 for c in path if path[c] == 6) == len(pores)
    assert not g.has_vertex(surface) and len(g.property("leaf_surface")) == 1
