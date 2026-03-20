from granap.root_class import RootAnatomy
from granap.needle_class import NeedleAnatomy
from granap.anatomy_writer import AnatomyWriter

def test_exports():
    print("Generating needle anatomy...")
    needle = NeedleAnatomy()
    needle.update_params("resin_duct", "n_files", 0)
    needle.update_params("stomata", "n_files", 0)
    needle.generate_cells()
    print("Writing XML...")
    needle.write_to_xml("test_needle.xml")
    print("Writing OBJ...")
    needle.write_to_obj("test_needle.obj")
    print("Writing GEO...")
    needle.write_to_geo("test_needle.geo", cell_wall_thickness = 1)
    print("Export test successful!")

if __name__ == "__main__":
    test_exports()
