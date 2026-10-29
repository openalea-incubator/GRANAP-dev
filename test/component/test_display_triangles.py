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
    g = ds.structure
    segment = int(ds.entity_ids("SubOrgan")[0])
    flat = anatomy.display_triangles().vertices
    assert np.ptp(flat[..., 2]) == 0.                       # no segment ends: a vertical axis, the section horizontal
    ends = dict(x1=0.01, y1=0.02, z1=-0.03, x2=0.01 + 2e-4, y2=0.02, z2=-0.03 - 2e-4)
    for name, value in ends.items():
        if name not in g.properties():
            g.add_property(name)
        g.property(name)[segment] = value
    try:
        placed = anatomy.display_triangles().vertices.reshape(-1, 3)
        axis = np.array([1., 0., -1.]) / np.sqrt(2.)
        middle = np.array([0.01 + 1e-4, 0.02, -0.03 - 1e-4])
        np.testing.assert_allclose((placed - middle) @ axis, 0., atol=1e-15)      # across the segment's middle
    finally:
        for name in ends:
            g.property(name).pop(segment, None)


def test_a_mask_and_the_light_geometry(section):
    ds, anatomy = section
    where = np.zeros(ds.n_nodes(), dtype=bool)
    where[np.flatnonzero(np.asarray(ds.get("kind")) == 0)[:10]] = True
    assert set(anatomy.display_triangles(where).owner.tolist()) <= set(np.flatnonzero(where).tolist())
    with pytest.raises(NotImplementedError):                 # not a geometry provider: the stand-in draws nothing
        ds.triangles()


def test_written_in_a_frame(section):
    pytest.importorskip("openalea.fspm_tf")
    from openalea.metafspm.io.fspm_tf import to_frame
    ds, anatomy = section
    meshes = {mesh.name: mesh for mesh in to_frame({"root": ds}, components={"root": [anatomy]}).data_structures[0].meshes}
    assert set(meshes) == {"GranapAnatomy"}
    assert len(meshes["GranapAnatomy"].positions) == 3 * len(anatomy.display_triangles())
