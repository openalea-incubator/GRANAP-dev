"""
Q12: the anatomies follow the root system. A new segment gets its class's anatomy and inherits from the same cells
upstream; a segment whose class changes has its anatomy replaced, its new cells starting from the defaults; a pruned
segment takes its anatomy with it. Junctions are rewired only where the anatomies or links changed.
"""
import numpy as np
import pytest

from openalea.metafspm.data_structure.configs import PropsConfig

from openalea.granap.metafspm.component import GranapAnatomy
from openalea.granap.metafspm.library import AnatomyLibrary

from roots import MAIN_RADIUS, SEGMENT, SmallRootSystem, initiated, planting

LATERAL = (0.6, 0., 0.)


class Anatomy(GranapAnatomy):
    library = AnatomyLibrary(class_parameters={LATERAL: {"cortex": {"n_layers": 2}}})


@pytest.fixture
def plant():
    population = initiated(planting(), SmallRootSystem, Anatomy)
    mtg, plants = population.data_structure.structure, population.plants
    data_structure = population.data_structure
    data_structure.register("solute", np.arange(data_structure.n_nodes(), dtype=float), location="node", on_grow="inherit", default=-1.)
    return mtg, data_structure, Anatomy(data_structure=data_structure)


def _root_segments(mtg):
    return sorted(int(v) for v in mtg.property("anatomy_class"))


def _tips(mtg):
    segments = _root_segments(mtg)
    return [s for s in segments if not [c for c in mtg.children(s) if c in segments]]


def _index(data_structure, vids):
    return data_structure.index_of(np.asarray(vids, dtype=np.int64))


def test_a_new_segment_gets_its_anatomy_and_inherits_from_the_same_cells_upstream(plant):
    mtg, data_structure, anatomy = plant
    main_tip = max(_tips(mtg), key=lambda s: mtg.property("radius")[s])
    before = mtg.compartments_by_owner(mtg.scales.SubOrgan)[main_tip]
    junctions = set(mtg.junction_vids())
    new = mtg.add_child(main_tip, **PropsConfig(scale=mtg.scales.SubOrgan, edge_type='<',
                                              label=mtg.labels.SubOrgan.RootSegment))
    for name, value in (("radius", MAIN_RADIUS), ("length", SEGMENT), ("distance_from_tip", 0.)):
        mtg.set_values(name, [new], [value])
    anatomy()
    assert anatomy.last_update == {"emerged": [new], "replaced": []}
    compartments = mtg.compartments_by_owner(mtg.scales.SubOrgan)[new]
    assert len(compartments) == len(before)                                       # the same class's copy
    assert data_structure.rewired == [new]                                                    # only its link
    node, solute = mtg.property("template_node"), data_structure.get("solute")
    upstream = {node[v]: v for v in before}
    for v in compartments[:200]:
        assert solute[_index(data_structure, [v])[0]] == solute[_index(data_structure, [upstream[node[v]]])[0]]   # the same cell upstream
    assert len(set(mtg.junction_vids()) - junctions) > 0


def test_a_segment_changing_class_is_replaced_and_starts_from_the_defaults(plant):
    mtg, data_structure, anatomy = plant
    lateral = min(_root_segments(mtg), key=lambda s: (mtg.property("radius")[s], s))     # the lateral's first segment
    old = list(mtg.compartments_by_owner(mtg.scales.SubOrgan)[lateral])
    radius = data_structure.get("radius").copy()                     # it thickens into the main class, as the architecture
    radius[data_structure.entity_ids("SubOrgan") == lateral] = MAIN_RADIUS    # model would write it: through the DataStructure
    data_structure.set("radius", radius)
    anatomy()
    assert anatomy.last_update == {"emerged": [], "replaced": [lateral]}
    new = mtg.compartments_by_owner(mtg.scales.SubOrgan)[lateral]
    assert not set(old) & set(new) and not any(mtg.has_vertex(v) for v in old)       # the old anatomy is gone
    assert len(new) == len(Anatomy.library.template(mtg, (1.1, 0., 0.)).cell_layer) + \
        len(Anatomy.library.template(mtg, (1.1, 0., 0.)).apoplast["compartment_type"][0])
    np.testing.assert_array_equal(data_structure.get("solute")[_index(data_structure, new)], -1.)        # defaults, nothing carried (Q-G2)
    assert lateral in data_structure.rewired


def test_a_pruned_segment_takes_its_anatomy_with_it(plant):
    mtg, data_structure, anatomy = plant
    tip = min(_tips(mtg), key=lambda s: mtg.property("radius")[s])                    # the lateral's tip
    gone = list(mtg.compartments_by_owner(mtg.scales.SubOrgan)[tip])
    n_nodes = data_structure.n_nodes()
    mtg.remove_vertex(tip)
    data_structure.update_topology()               # as the architecture's own (MPG-style) pruning step would, by the framework
    anatomy()
    assert not any(mtg.has_vertex(v) for v in gone) and tip not in mtg.property("anatomy_class")
    assert data_structure.n_nodes() == n_nodes - len(gone)
