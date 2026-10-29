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
    import pandas as pd
    from openalea.metafspm.data_structure.data_api import MPGDataStructure
    from openalea.metafspm.scene.population import build_population
    from openalea.granap.metafspm.architectures import RootAxis
    from openalea.granap.metafspm.component import junction_rules
    scenario = {"root_segments": [[0.55e-3, 2e-4, 0.01], [0.55e-3, 2e-4, 0.]],
                "root_laterals": [[0, [[0.55e-3, 2e-4, 0.]]]]}
    table = pd.DataFrame([dict(plant="p", model=None, x=0., y=0., z=0., rotation=0., scenario={"parameters": scenario})])
    g, _ = build_population(table, initiators=(RootAxis, GranapAnatomy))
    g.convert_properties_to_arraydict()
    ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan, nodes="Compartment", wiring=junction_rules(g))
    GranapAnatomy(data_structure=ds)
    main, _, lateral = [int(s) for s in ds.entity_ids("SubOrgan")]
    assert g.linked_parent(lateral, g.scales.SubOrgan) == main
    assert g.complex_at_scale(lateral, g.scales.Axis) != g.complex_at_scale(main, g.scales.Axis)
    view, owner = ds.to_graph_view(), ds.owner("SubOrgan")
    index = {int(s): i for i, s in enumerate(ds.entity_ids("SubOrgan"))}
    junctions = (ds.get("network_path") == 5) & (owner[view.tail] == index[main]) & (owner[view.head] == index[lateral])
    assert junctions.sum() > 0 and set(np.unique(ds.get("tissue")[view.head[junctions]])) >= {"metaxylem", "phloem"}
