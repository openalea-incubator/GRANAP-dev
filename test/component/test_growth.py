"""
Q12: the anatomies follow the root system. A new segment gets its class's anatomy and inherits from the same cells
upstream; a segment whose class changes has its anatomy replaced, its new cells starting from the defaults; a pruned
segment takes its anatomy with it. Junctions are rewired only where the anatomies or links changed.
"""
import numpy as np
import pytest

from openalea.metafspm.coupling.choregrapher import Choregrapher
from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.metafspm.data_structure.data_api import MPGDataStructure
from openalea.metafspm.scene.population import build_population

from openalea.granap.metafspm.component import GranapAnatomy, junction_rules
from openalea.granap.metafspm.library import AnatomyLibrary

from roots import MAIN_RADIUS, SEGMENT, SmallRootSystem, planting

LATERAL = (0.6, 0., 0.)


class Anatomy(GranapAnatomy):
    library = AnatomyLibrary(class_parameters={LATERAL: {"cortex": {"n_layers": 2}}})


@pytest.fixture
def plant():
    Choregrapher().add_simulation_time_step(3600.)
    g, plants = build_population(planting(), initiators=(SmallRootSystem, Anatomy))
    g.convert_properties_to_arraydict()
    ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan, nodes="Compartment", wiring=junction_rules(g))
    ds.register("solute", np.arange(ds.n_nodes(), dtype=float), location="node", on_grow="inherit", default=-1.)
    return g, ds, Anatomy(data_structure=ds)


def _root_segments(g):
    return sorted(int(v) for v in g.property("anatomy_class"))


def _tips(g):
    segments = _root_segments(g)
    return [s for s in segments if not [c for c in g.children(s) if c in segments]]


def _index(ds, vids):
    return ds.index_of(np.asarray(vids, dtype=np.int64))


def test_a_new_segment_gets_its_anatomy_and_inherits_from_the_same_cells_upstream(plant):
    g, ds, anatomy = plant
    main_tip = max(_tips(g), key=lambda s: g.property("radius")[s])
    before = g.compartments_by_owner(g.scales.SubOrgan)[main_tip]
    junctions = set(g.junction_vids())
    new = g.add_child(main_tip, **PropsConfig(scale=g.scales.SubOrgan, edge_type='<',
                                              label=g.labels.SubOrgan.RootSegment))
    for name, value in (("radius", MAIN_RADIUS), ("length", SEGMENT), ("distance_from_tip", 0.)):
        g.property(name)[new] = value
    anatomy()
    assert anatomy.last_update == {"emerged": [new], "replaced": []}
    compartments = g.compartments_by_owner(g.scales.SubOrgan)[new]
    assert len(compartments) == len(before)                                       # the same class's copy
    assert ds.rewired == [new]                                                    # only its link
    node, solute = g.property("template_node"), ds.get("solute")
    upstream = {node[v]: v for v in before}
    for v in compartments[:200]:
        assert solute[_index(ds, [v])[0]] == solute[_index(ds, [upstream[node[v]]])[0]]   # the same cell upstream
    assert len(set(g.junction_vids()) - junctions) > 0


def test_a_segment_changing_class_is_replaced_and_starts_from_the_defaults(plant):
    g, ds, anatomy = plant
    lateral = min(_root_segments(g), key=lambda s: (g.property("radius")[s], s))     # the lateral's first segment
    old = list(g.compartments_by_owner(g.scales.SubOrgan)[lateral])
    radius = ds.get("radius").copy()                     # it thickens into the main class, as the architecture
    radius[ds.entity_ids("SubOrgan") == lateral] = MAIN_RADIUS    # model would write it: through the DataStructure
    ds.set("radius", radius)
    anatomy()
    assert anatomy.last_update == {"emerged": [], "replaced": [lateral]}
    new = g.compartments_by_owner(g.scales.SubOrgan)[lateral]
    assert not set(old) & set(new) and not any(v in g._scale for v in old)       # the old anatomy is gone
    assert len(new) == len(Anatomy.library.template((1.1, 0., 0.)).cell_layer) + \
        len(Anatomy.library.template((1.1, 0., 0.)).apoplast["kind"][0])
    np.testing.assert_array_equal(ds.get("solute")[_index(ds, new)], -1.)        # defaults, nothing carried (Q-G2)
    assert lateral in ds.rewired


def test_a_pruned_segment_takes_its_anatomy_with_it(plant):
    g, ds, anatomy = plant
    tip = min(_tips(g), key=lambda s: g.property("radius")[s])                    # the lateral's tip
    gone = list(g.compartments_by_owner(g.scales.SubOrgan)[tip])
    n_nodes = ds.n_nodes()
    g.remove_vertex(tip)
    ds.update_topology()               # as the architecture's own (MPG-style) pruning step would, by the framework
    anatomy()
    assert not any(v in g._scale for v in gone) and tip not in g.property("anatomy_class")
    assert ds.n_nodes() == n_nodes - len(gone)
