"""GranapAnatomy.display_triangles(): each protoplast its cell's polygon, the section drawn flat across its segment,
for display only (the DataStructure's triangles, those light models get, stay the architecture's)."""
import numpy as np
import pytest

from openalea.granap.metafspm.component import GranapAnatomy
from openalea.granap.simulation import simulate


@pytest.fixture(scope="module")
def section():
    result = simulate()
    anatomy = next(c for c in result.scene.components if isinstance(c, GranapAnatomy))
    return result.data_structure, anatomy


def _areas(vertices):
    a, b, c = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def test_each_protoplast_is_its_cells_polygon(section):
    ds, anatomy = section
    triangles = anatomy.display_triangles()
    kind = np.asarray(ds.get("kind"))
    assert len(triangles) and (kind[triangles.owner] == 0).all()                    # protoplasts only
    drawn = np.unique(triangles.owner)
    assert drawn.size == int((np.asarray(ds.get("cell_area")) > 0).sum())
    area_mm2 = np.bincount(triangles.owner, weights=_areas(triangles.vertices), minlength=ds.n_nodes()) * 1e6
    np.testing.assert_allclose(area_mm2[drawn], np.asarray(ds.get("cell_area"))[drawn], rtol=1e-6)


def test_the_section_lies_across_its_segment(section):
    ds, anatomy = section
    flat = anatomy.display_triangles().vertices
    assert np.ptp(flat[..., 2]) == 0.                       # no segment ends: a vertical axis, the section horizontal
    ends = dict(x1=0.01, y1=0.02, z1=-0.03, x2=0.01 + 2e-4, y2=0.02, z2=-0.03 - 2e-4)
    assert not any(ds.has(name) for name in ends)           # optional inputs nothing gives: not registered
    for name, value in ends.items():                        # given, as an architecture declaring them would
        ds.register(name, np.full(ds.entity_ids("SubOrgan").size, np.nan), location="SubOrgan")
        ds.get(name)[0] = value
    try:
        placed = anatomy.display_triangles().vertices.reshape(-1, 3)
        axis = np.array([1., 0., -1.]) / np.sqrt(2.)
        middle = np.array([0.01 + 1e-4, 0.02, -0.03 - 1e-4])
        np.testing.assert_allclose((placed - middle) @ axis, 0., atol=1e-15)      # across the segment's middle
    finally:
        for name in ends:
            ds.unregister(name)


def test_drawing_leaves_the_mpg_as_it_was(section):
    ds, anatomy = section
    before = {name: len(prop) for name, prop in ds.structure.properties().items()}
    anatomy.display_triangles()
    assert {name: len(prop) for name, prop in ds.structure.properties().items()} == before


def test_a_mask_and_the_light_geometry(section):
    ds, anatomy = section
    where = np.zeros(ds.n_nodes(), dtype=bool)
    where[np.flatnonzero(np.asarray(ds.get("kind")) == 0)[:10]] = True
    assert set(anatomy.display_triangles(where).owner.tolist()) <= set(np.flatnonzero(where).tolist())
    with pytest.raises(NotImplementedError):                 # not a geometry provider: the stand-in draws nothing
        ds.triangles()


def test_written_in_a_frame(section):
    pytest.importorskip("openalea.encoding3d")
    from openalea.metafspm.io.encoding3d import to_frame
    ds, anatomy = section
    meshes = {mesh.name: mesh for mesh in to_frame({"root": ds}, components={"root": [anatomy]}).data_structures[0].meshes}
    assert set(meshes) == {"GranapAnatomy"}
    mesh = meshes["GranapAnatomy"]                       # its triangles by indices, their vertices shared
    assert mesh.indices.size == 3 * len(anatomy.display_triangles()) and len(mesh.positions) < mesh.indices.size / 1.5


def test_a_scene_resumed_from_its_checkpoint_draws_its_sections(tmp_path):
    """A Scene's checkpoint keeps the anatomy templates (the MPG's registry, g.objects("anatomy_templates"), saved
    with it): a scene built again has its own templates (no module-level registry: the saved scene's are not
    shared), of the same indices (numbers of their content); loaded, the segments' anatomy_template resolve to the
    checkpoint's templates (their tables, not their organs) and the first frame draws the sections as the saved scene
    did."""
    from openalea.granap.metafspm.library import template_by_index
    saved = simulate().scene
    anatomy = next(c for c in saved.components if isinstance(c, GranapAnatomy))
    saved_ds = saved.populations[0].data_structure
    index = int(saved_ds.get("anatomy_template")[0])
    drawn = anatomy.display_triangles()
    saved.checkpoint(str(tmp_path / "checkpoint"))
    resumed = simulate().scene
    ds = resumed.populations[0].data_structure
    assert int(ds.get("anatomy_template")[0]) == index                             # an index of its content
    assert template_by_index(ds, index) is not template_by_index(saved_ds, index)  # its own
    resumed.load_checkpoint(str(tmp_path / "checkpoint"))
    assert int(ds.get("anatomy_template")[0]) == index
    redrawn = next(c for c in resumed.components if isinstance(c, GranapAnatomy)).display_triangles()
    np.testing.assert_array_equal(redrawn.owner, drawn.owner)
    np.testing.assert_allclose(redrawn.vertices, drawn.vertices)
    assert template_by_index(ds, index).organ is None                   # restored: its tables, not its organ


def test_the_templates_are_saved_once_by_the_checkpoints(tmp_path):
    """The anatomy templates are shared parts of the checkpoints (metafspm.scene.shared): a second checkpoint into
    the same shared folder writes none again and refers to the same parts."""
    import os
    from openalea.metafspm.scene.shared import references
    scene = simulate().scene
    shared = tmp_path / "shared"
    scene.checkpoint(str(tmp_path / "first"), shared=str(shared))
    parts = sorted(os.listdir(shared))
    times = {name: os.path.getmtime(shared / name) for name in parts}
    scene.checkpoint(str(tmp_path / "second"), shared=str(shared))
    assert sorted(os.listdir(shared)) == parts and any(name.startswith("anatomy_templates") for name in parts)
    assert all(os.path.getmtime(shared / name) == times[name] for name in parts)
    assert references(str(tmp_path / "first"))[1] == references(str(tmp_path / "second"))[1]
    assert os.path.getsize(tmp_path / "second" / "states.pkl") < 10_000            # the templates referred to only
