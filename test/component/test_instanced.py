"""
Instanced anatomies (a plant's scenario key instanced_anatomy): each segment's anatomy an instance of its class's
template (metafspm's data_structure.instances), no vertex of its own, against the same plant with copies: the same
nodes, edges, junctions, leaf surfaces, variables, growth, class changes and pruning.
"""
import numpy as np
import pytest

from openalea.metafspm.coupling.choregrapher import Choregrapher
from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.metafspm.scene.population import Planting, planting

from roots import MAIN_RADIUS, SEGMENT, SmallRootSystem, initiated
from test_growth import Anatomy
from test_shoot import ShootAnatomy, SmallShoot, TwoElementLeaf


def _both(architecture, anatomy, **scenario):
    Choregrapher().set_simulation_time_step(3600.)
    out = []
    for instanced in (False, True):
        population = initiated(planting([Planting()], None, dict(scenario, instanced_anatomy=instanced)), architecture,
                               anatomy)
        out.append(population.data_structure)
    return out


def _endpoints(g, edges):
    a, b = g.view("n_id_a"), g.view("n_id_b")
    return [(int(a[e]), int(b[e])) for e in edges]


def _same(copied, instanced):
    g, h = copied.structure, instanced.structure
    assert copied.entity_ids("node").tolist() == instanced.entity_ids("node").tolist()
    assert copied.entity_ids("edge").tolist() == instanced.entity_ids("edge").tolist()
    assert copied.edges() == instanced.edges()
    assert sorted(_endpoints(g, g.junction_vids())) == sorted(_endpoints(h, h.junction_vids()))
    for scale in ("SubOrgan", "Cell"):
        np.testing.assert_array_equal(copied.owner(scale), instanced.owner(scale))
    left, right = copied._var_stores(), instanced._var_stores()
    assert {k: sorted(v) for k, v in left.items()} == {k: sorted(v) for k, v in right.items()}
    for location, store in left.items():
        for name, values in store.items():
            if name == "instanced_anatomy":                          # the plants' switch itself
                continue
            other = right[location][name]
            if values.dtype == object:
                assert values.tolist() == other.tolist(), name
            else:
                np.testing.assert_array_equal(values, other, err_msg=name)
    assert len(h) < len(g)
    assert h.has_instances() and not g.has_instances()


def test_a_shoot_with_instanced_anatomies_is_its_copy():
    copied, instanced = _both(SmallShoot, ShootAnatomy)
    _same(copied, instanced)


def test_leaf_surfaces_and_their_stomata_on_instanced_anatomies():
    copied, instanced = _both(TwoElementLeaf, ShootAnatomy)
    _same(copied, instanced)
    g, h = copied.structure, instanced.structure
    stomatal = [[c for c in x._vertices_at_scale(x.scales.Connection) if x.view("network_path").get(int(c)) == 6]
                for x in (g, h)]
    assert len(stomatal[0]) > 0 and _endpoints(g, stomatal[0]) == _endpoints(h, stomatal[1])
    pores = [v for v, _ in _endpoints(g, stomatal[0])]
    assert g.values_of("pore_width", pores) == h.values_of("pore_width", pores)


def test_growth_class_changes_and_pruning_on_instanced_anatomies():
    states = []
    for ds in _both(SmallRootSystem, Anatomy):
        g = ds.structure
        ds.register("solute", np.arange(ds.n_nodes(), dtype=float), location="node", on_grow="inherit", default=-1.)
        anatomy = Anatomy(data_structure=ds)
        segments = sorted(int(v) for v in g.property("anatomy_class"))
        tips = [s for s in segments if not [c for c in g.children(s) if c in segments]]
        main_tip = max(tips, key=lambda s: g.property("radius")[s])
        new = g.add_child(main_tip, **PropsConfig(scale=g.scales.SubOrgan, edge_type='<',
                                                  label=g.labels.SubOrgan.RootSegment))
        for name, value in (("radius", MAIN_RADIUS), ("length", SEGMENT), ("distance_from_tip", 0.)):
            g.property(name)[new] = value
        anatomy()                                                    # growth: inherits from the same cells upstream
        grown = ds.get("solute").copy()
        lateral = min(segments, key=lambda s: (g.property("radius")[s], s))
        radius = ds.get("radius").copy()
        radius[ds.entity_ids("SubOrgan") == lateral] = MAIN_RADIUS
        ds.set("radius", radius)
        anatomy()                                                    # a class change: replaced, defaults
        changed = ds.get("solute").copy()
        tip = min((s for s in tips if s != lateral), key=lambda s: g.property("radius")[s])
        g.remove_vertex(tip)
        ds.update_topology()
        anatomy()                                                    # pruning
        states.append((ds, grown, changed, ds.get("solute").copy()))
    (copied, *values), (instanced, *other) = states
    _same(copied, instanced)
    for left, right in zip(values, other):
        np.testing.assert_array_equal(left, right)


def test_a_root_section_has_one_soil_node_joined_to_its_outer_walls():
    """soil_node: each root section's one soil boundary, a soil Compartment (kind 4) joined (network_path 7) to the
    walls with a single membrane and the junctions on them; the same copied and instanced."""
    copied, instanced = _both(SmallRootSystem, Anatomy, soil_node=True)
    for ds in (copied, instanced):
        Anatomy(data_structure=ds)                                     # its variables, read from the MPG
    _same(copied, instanced)
    ds = copied
    kind, path, tissue = ds.get("kind"), ds.get("network_path"), ds.get("tissue")
    owner, segments = ds.owner("SubOrgan"), ds.entity_ids("SubOrgan")
    soil = np.flatnonzero(kind == 4)
    assert sorted(owner[soil].tolist()) == list(range(segments.size)) and set(tissue[soil]) == {"soil"}
    view = ds.to_graph_view()
    links = np.flatnonzero(path == 7)
    assert set(view.tail[links].tolist()) == set(soil.tolist())
    ends = view.head[links]
    assert (owner[ends] == owner[view.tail[links]]).all()                  # within its own section
    membranes = np.flatnonzero(path == 0)
    counts = np.bincount(np.r_[view.tail[membranes], view.head[membranes]], minlength=ds.n_nodes())
    walls = ends[kind[ends] == 1]
    assert walls.size > 0 and (counts[walls] == 1).all()                     # outer walls: one membrane each
    assert ((kind[ends] == 1) | (kind[ends] == 2)).all()


def test_a_state_changed_in_the_data_structure_survives_the_next_anatomy_step():
    """An instanced vertex's value is held by the DataStructure (not written back to the MPG): the anatomy step's
    read-back keeps it, as a copy's (written back, read back), also when a new segment grows."""
    results = []
    for ds in _both(SmallRootSystem, Anatomy):
        g = ds.structure
        anatomy = Anatomy(data_structure=ds)
        x = ds.get("section_x").copy()
        x[7] = 123.
        ds.set("section_x", x)
        anatomy()
        assert ds.get("section_x")[7] == 123.
        segments = sorted(int(v) for v in g.property("anatomy_class"))
        tip = max(segments)
        new = g.add_child(tip, **PropsConfig(scale=g.scales.SubOrgan, edge_type='<',
                                             label=g.labels.SubOrgan.RootSegment))
        for name, value in (("radius", g.property("radius")[tip]), ("length", SEGMENT), ("distance_from_tip", 0.)):
            g.property(name)[new] = value
        anatomy()
        results.append(ds)
    copied, instanced = results
    assert copied.get("section_x")[7] == instanced.get("section_x")[7] == 123.
    _same(copied, instanced)


def _soil_nodes(template):
    rows, values = template.properties["kind"]
    out = np.zeros(template.size, dtype=bool)
    out[rows] = np.asarray(values, dtype=float) == 4.
    return out


def test_a_reduced_root_section_holds_its_conduits_and_soil_node():
    """A plant model's Reduction (solve.reduction) on the root segments: each an instance of its reduced template,
    its conduits (GranapAnatomy.conduits, families 0 and 1) and its soil node only, the conduits joined to the
    neighbouring sections' by the junctions as in the full anatomy."""
    from openalea.metafspm.solve.reduction import Reduction
    from openalea.granap.metafspm.component import GranapAnatomy
    Choregrapher().set_simulation_time_step(3600.)
    full = initiated(planting([Planting()], None, {"instanced_anatomy": True, "soil_node": True}), SmallRootSystem,
                     Anatomy).data_structure
    reduction = Reduction(labels=("RootSegment",), ports=GranapAnatomy.conduits, keep=_soil_nodes)
    reduced = initiated(planting([Planting()], None, {"soil_node": True}), SmallRootSystem, Anatomy,
                        reduction=reduction).data_structure
    for ds in (full, reduced):
        Anatomy(data_structure=ds)

    def family(ds):
        values = ds.structure.values_of("family", ds.entity_ids("node"))
        return np.array([values.get(int(v), -1.) for v in ds.entity_ids("node")])

    kind = reduced.get("kind")
    assert ((kind == 4) | np.isin(family(reduced), (0., 1.))).all()
    full_kept = (full.get("kind") == 4) | np.isin(family(full), (0., 1.))
    assert reduced.n_nodes() == int(full_kept.sum())

    def junctions(ds):
        """Each junction's two ends as (section, template node): the vids differ (a reduced block is smaller)."""
        view, nodes, owner = ds.to_graph_view(), ds.entity_ids("node"), ds.owner("SubOrgan")
        node = ds.structure.values_of("template_node", nodes)
        place = [(int(owner[i]), int(node[int(v)])) for i, v in enumerate(nodes.tolist())]
        edges = np.flatnonzero(np.asarray(ds.get("network_path")) == 5)
        return sorted((place[a], place[b]) for a, b in zip(view.tail[edges].tolist(), view.head[edges].tolist()))

    assert junctions(full) == junctions(reduced) and len(junctions(full)) > 0


def test_a_reduced_section_is_drawn_whole_and_its_frame_holds_its_interior():
    """A reduced section's drawing (display_triangles) is its full anatomy's: its kept protoplasts by node, its
    interior's by display id (owner -1, owner_ids), the same triangles as the unreduced plant's; metafspm's frame
    writes the interior as display rows (its Cells, Layers and Compartments, linked to the section), the mesh's
    triangles on them; from_frame gives the reduced plant back."""
    from openalea.metafspm.io.encoding3d import DISPLAY_ONLY, from_frame, to_frame
    from openalea.metafspm.solve.reduction import Reduction, is_display_id
    from openalea.granap.metafspm.component import GranapAnatomy
    Choregrapher().set_simulation_time_step(3600.)
    full = initiated(planting([Planting()], None, {"instanced_anatomy": True, "soil_node": True}), SmallRootSystem,
                     Anatomy).data_structure
    reduction = Reduction(labels=("RootSegment",), ports=GranapAnatomy.conduits, keep=_soil_nodes)
    reduced = initiated(planting([Planting()], None, {"soil_node": True}), SmallRootSystem, Anatomy,
                        reduction=reduction).data_structure
    drawn_full = Anatomy(data_structure=full).display_triangles()
    anatomy = Anatomy(data_structure=reduced)
    drawn = anatomy.display_triangles()
    assert len(drawn) == len(drawn_full) and (drawn.owner < 0).any()
    np.testing.assert_allclose(np.sort(drawn.vertices.reshape(-1)), np.sort(drawn_full.vertices.reshape(-1)))
    assert is_display_id(drawn.owner_ids[drawn.owner < 0]).all()
    np.testing.assert_array_equal(drawn.owner_ids[drawn.owner >= 0],
                                  reduced.entity_ids("node")[drawn.owner[drawn.owner >= 0]])
    frame = to_frame({"plant": reduced}, components={"plant": [anatomy]})
    plant = frame.data_structures[0]
    compartments = plant.entity_set("Compartment").columns
    shown = np.asarray(compartments[DISPLAY_ONLY], dtype=bool)
    assert shown.sum() + reduced.n_nodes() == full.n_nodes()
    mesh = next(mesh for mesh in plant.meshes if mesh.name == type(anatomy).__name__)
    corner = mesh.indices[::3]                                          # each triangle's first vertex (shared ones)
    np.testing.assert_array_equal(compartments["id"][mesh.feature_rows["Compartment"][corner]], drawn.owner_ids)
    sections = plant.entity_set("SubOrgan").columns["id"][mesh.feature_rows["SubOrgan"][corner]]
    owner_of_node = reduced.entity_ids("SubOrgan")[reduced.owner("SubOrgan")]
    expected = np.where(drawn.owner >= 0, owner_of_node[np.maximum(drawn.owner, 0)],
                        (drawn.owner_ids - 2 ** 52) // 2 ** 24)
    np.testing.assert_array_equal(sections, expected)
    assert (mesh.feature_rows["Cell"][corner] >= 0).all()               # every protoplast in its Cell
    assert from_frame(frame)["plant"].n_nodes() == reduced.n_nodes()
