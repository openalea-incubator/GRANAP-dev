"""
The anatomy component's parametrization API: the roots' OrganAnatomy (root), and plant overrides whose values are
themselves dicts (a size distribution), keyed as templates.
"""
from openalea.granap.metafspm.component import GranapAnatomy, OrganAnatomy
from openalea.granap.metafspm.library import AnatomyLibrary, ClassInput, _frozen


def test_root_gives_the_roots_their_library_and_class_inputs():
    inputs = (ClassInput("radius", 0.1, 2000.),)
    Root = type("Root", (GranapAnatomy,), {"root": OrganAnatomy(AnatomyLibrary(), class_inputs=inputs)})
    assert Root.organs()["RootSegment"] is Root.root and Root.organs()["RootSegment"].class_inputs == inputs
    default = GranapAnatomy.organs()["RootSegment"]                     # without: the library, by diameter
    assert default.library is GranapAnatomy.library and default.class_inputs[0].variable == "radius"


def test_nested_override_values_are_hashable_and_order_free():
    a = {"xylem": {"vessel_diameter_distribution": {"family": "lognormal", "values": []}, "n_protoxylem": 8}}
    b = {"xylem": {"n_protoxylem": 8, "vessel_diameter_distribution": {"values": [], "family": "lognormal"}}}
    assert hash(_frozen(a)) == hash(_frozen(b)) and _frozen(a) == _frozen(b)
    flat = {"stele": {"thickness": 0.45}}
    assert _frozen(flat) == (("stele", (("thickness", 0.45),)),)       # flat overrides: the keys as before


def test_a_lateral_is_joined_to_the_segment_it_branches_on():
    """RootAxis's laterals: the first segment's linked parent is its bearer, on another axis, and the DataStructure
    joins their sections' xylem and phloem."""
    import numpy as np
    from openalea.metafspm.scene.population import Planting, planting
    from openalea.granap.metafspm.architectures import RootAxis
    from roots import initiated
    scenario = {"root_segments": [[0.55e-3, 2e-4, 0.01], [0.55e-3, 2e-4, 0.]],
                "root_laterals": [[0, [[0.55e-3, 2e-4, 0.]]]]}
    table = planting([Planting()], None, scenario)
    population = initiated(table, RootAxis, GranapAnatomy)
    g = population.data_structure.structure
    ds = population.data_structure
    GranapAnatomy(data_structure=ds)
    main, _, lateral = [int(s) for s in ds.entity_ids("SubOrgan")]
    assert g.linked_parent(lateral, g.scales.SubOrgan) == main
    assert g.complex_at_scale(lateral, g.scales.Axis) != g.complex_at_scale(main, g.scales.Axis)
    view, owner = ds.to_graph_view(), ds.owner("SubOrgan")
    index = {int(s): i for i, s in enumerate(ds.entity_ids("SubOrgan"))}
    junctions = (ds.get("network_path") == 5) & (owner[view.tail] == index[main]) & (owner[view.head] == index[lateral])
    assert junctions.sum() > 0 and set(np.unique(ds.get("tissue")[view.head[junctions]])) >= {"metaxylem", "phloem"}


def test_root_growth_grows_the_listed_axes_at_each_step():
    """RootAxis's root_growth: at each step, one new segment (one NewNodes batch) at the tip of each axis listed, an
    axis listed twice two in a row, the axis' other segments farther from its tip; the plants without it do not grow;
    the new segments get their anatomies in the same step."""
    import numpy as np
    from openalea.metafspm.scene.population import Planting, planting
    from openalea.metafspm.scene.scene import Scene
    from openalea.granap.simulation import _plant_model
    grown = {"root_segments": [[0.3e-3, 2e-4, 2e-4], [0.3e-3, 2e-4, 0.]],
             "root_laterals": [[0, [[0.2e-3, 1e-4, 0.]]]],
             "root_growth": [[1, 0.2e-3, 3e-4], [0, 0.3e-3, 2e-4], [1, 0.25e-3, 1e-4]]}
    scene = Scene(planting([Planting(), Planting(x=0.1)], _plant_model(),
                           plant_scenarios=[grown, {"root_segments": [[0.3e-3, 2e-4, 0.]]}]))
    ds = scene.populations[0].data_structure
    g = ds.structure
    for _ in range(2):
        scene.run()
    first, second = (g.components_at_scale(int(plant), scale=g.scales.Axis) for plant in ds.entity_ids("Plant"))
    main, lateral = ([int(s) for s in g.components_at_scale(axis, scale=g.scales.SubOrgan)] for axis in first)
    assert len(main) == 4 and len(lateral) == 5 and len(g.components_at_scale(second[0], scale=g.scales.SubOrgan)) == 1
    radius, distance = (g.property(name) for name in ("radius", "distance_from_tip"))
    np.testing.assert_allclose([radius[s] for s in lateral], [0.2e-3, 0.2e-3, 0.25e-3, 0.2e-3, 0.25e-3])
    np.testing.assert_allclose([distance[s] for s in lateral], [8e-4, 5e-4, 4e-4, 1e-4, 0.], atol=1e-15)
    np.testing.assert_allclose([distance[s] for s in main], [6e-4, 4e-4, 2e-4, 0.], atol=1e-15)
    assert all(g.parent(new) == old for old, new in zip(lateral, lateral[1:]))
    assert set(lateral[1:]) <= set(int(s) for s in g.property("anatomy_class"))      # their anatomies
