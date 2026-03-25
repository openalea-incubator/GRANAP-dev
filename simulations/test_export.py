from granap.root_class import RootAnatomy
from granap.needle_class import NeedleAnatomy
from granap.anatomy_writer import AnatomyWriter
from bvpy.utils.visu_pyvista import visualize

def test_exports():
    print("Generating needle anatomy...")
    needle = NeedleAnatomy()
    needle.update_params("resin_duct", "n_files", 2)
    needle.update_params("stomata", "n_files", 2)
    needle.generate_cells()
    print("Writing XML...")
    needle.write_to_xml("test_needle.xml")
    print("Writing OBJ...")
    needle.write_to_obj("test_needle.obj")
    print("Writing GEO...")
    needle.write_to_geo("test_needle.geo", cell_wall_thickness = 1)
    print("Export test successful!")
    domain = needle.to_domain(cell_wall_thickness=1, celldomain = True, cell_size=50, dim=2, clear=True)
    domain.discretize()
    print("n cells:", domain.mesh.num_cells())
    print("sub_domain_names:", domain.sub_domain_names)
    assert domain.mesh is not None
    assert domain.mesh.num_cells() > 0
    print("OrganDomain test passed!")


if __name__ == "__main__":
    test_exports()
