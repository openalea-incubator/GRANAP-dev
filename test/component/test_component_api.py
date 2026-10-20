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
