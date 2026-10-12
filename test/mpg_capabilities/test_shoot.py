"""
Stems and leaves too: one library per kind of segment (GranapAnatomy.shoot), each segment its kind's anatomy, and
vessel junctions across the stem-leaf and stem-root links.
"""
import os
import sys

import pandas as pd

from openalea.metafspm.data_structure.data_api import MPGDataStructure
from openalea.metafspm.scene.population import build_population

from openalea.granap import mpg
from openalea.granap.mpg.component import GranapAnatomy, OrganAnatomy, junction_rules
from openalea.granap.mpg.library import AnatomyLibrary, preset_parameters

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "example", "mpg"))
from seedling import _chain, _organ                                                    # noqa: E402

from openalea.metafspm.coupling.component import StructuralComponent                  # noqa: E402


class SmallShoot(StructuralComponent):
    @classmethod
    def initiate_plant(cls, g, plant, parameters):
        L = g.labels.SubOrgan
        stem = _chain(g, _organ(g, plant, "Shoot", "StemInternode", True), 2, L.StemElement, 2.5e-3)
        _chain(g, _organ(g, plant, "Shoot", "Leaf", True), 1, L.LeafElement, 2.5e-3, bearer=stem[1])
        _chain(g, _organ(g, plant, "Root", "RootInternode", False), 1, L.RootSegment, 0.55e-3, bearer=stem[0])


class ShootAnatomy(GranapAnatomy):
    library = AnatomyLibrary()
    shoot = {"StemElement": OrganAnatomy(AnatomyLibrary(parameters=preset_parameters("for_monocot_stem"),
                                                        organ_class=mpg.StemAnatomy)),
             "LeafElement": OrganAnatomy(AnatomyLibrary(parameters=preset_parameters("for_monocot_leaf"),
                                                        organ_class=mpg.LeafAnatomy), size=None)}


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
