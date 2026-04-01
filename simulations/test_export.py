from granap.root_class import RootAnatomy
from granap.needle_class import NeedleAnatomy
from granap.anatomy_writer import AnatomyWriter


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
    print("Writing SVG...")
    needle.write_to_svg("test_needle.svg")
    print("Writing GEO...")
    needle.write_to_geo("test_needle.geo")
    print("Export needle test successful!")

    print("Generating root anatomy...")
    root = RootAnatomy()
    root.generate_cells()
    print("Writing XML...")
    root.write_to_xml("test_root.xml")
    print("Writing OBJ...")
    root.write_to_obj("test_root.obj")
    print("Writing GEO...")
    root.write_to_geo("test_root.geo")
    print("Writing SVG...")
    root.write_to_svg("test_root.svg")
    
    print("Export root test successful!")

if __name__ == "__main__":
    test_exports()
