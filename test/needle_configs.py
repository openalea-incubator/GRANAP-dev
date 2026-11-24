"""Needle config builders duplicated from ``example/needle/*.py``.

These are copies of ``build_pinaster`` (``example/needle/pinus_pinaster.py``),
``build_gallery_needle_data`` (``example/needle/gallery_needle_features.py``),
and ``build_nigra`` (``example/needle/pinus_nigra.py``), stripped of their
plotting/``main`` code. The test suite needs these builders, but ``test/`` is
the only directory conda-build's isolated test phase materializes (see
``conda/meta.yaml``'s ``test.source_files``) -- ``example/`` is not copied
there, so the tests cannot import the originals directly. Keep this file in
sync by hand if the originals change.
"""

from openalea.granap.needle_class import NeedleAnatomy
from openalea.granap.input_data import OrganInputData


def build_pinaster():
    WIDTH, THICKNESS = 2.232, 1.29
    _, ABAXIAL_POLE, CORNER_POS, CORNER_NEG = NeedleAnatomy.pole_and_corner_angles(WIDTH, THICKNESS)

    MESOPHYLL_FLOOR = 0.9
    ABAXIAL_MESOPHYLL_PROFILE = sorted([
        (ABAXIAL_POLE, 1.0),
        (CORNER_NEG, MESOPHYLL_FLOOR),
        (CORNER_POS, MESOPHYLL_FLOOR),
    ])

    HYPODERMIS_FLOOR = 0.85
    NODULE_PROFILE = NeedleAnatomy.corner_bump_profile(
        [(CORNER_POS, 10.0, 1.0), (CORNER_NEG, 10.0, 1.0)],
        floor=HYPODERMIS_FLOOR,
    )
    CORNER_ZONE_HALF_WIDTH = 4.0

    return [
        {"name": "planttype", "value": 3, "organ": "needle",
         "width": 2.232, "thickness": 1.29},
        {"name": "central_cylinder",
         "vascular_width": 0.33, "vascular_height": 0.25, "vascular_angle": 30,
         "cell_diameter": 0.02},
        {"name": "endodermis", "cell_diameter": 0.035, "cell_width": 0.045, "order": 3},
        {"name": "mesophyll", "cell_diameter": 0.1, "cell_width": 0.06, "order": 4},
        {"name": "mesophyll_abaxial", "cell_diameter": 0.08, "cell_width": 0.065, "order": 4.1,
         "thickness_profile": ABAXIAL_MESOPHYLL_PROFILE,
         "zone_angles": {"mode": "half", "pole": ABAXIAL_POLE}},
        {"name": "palisade", "cell_diameter": 0.09, "cell_width": 0.04, "order": 4.5},
        {"name": "hypodermis", "cell_diameter": 0.02, "cell_width":0.0225, "n_layers": 2, "order": 5},
        {"name": "hypodermis_corner", "cell_diameter": 0.0225, "n_layers": 3, "order": 5.1,
         "thickness_profile": NODULE_PROFILE,
         "zone_angles": {"mode": "wedge", "centers": [CORNER_POS, CORNER_NEG],
                         "half_width": CORNER_ZONE_HALF_WIDTH}},
        {"name": "epidermis", "cell_diameter": 0.015, "cell_width":0.02, "order": 6},
        {"name": "transfusion_tissue", "n_layers": 2, "pack_circles": True,
         "diameter_max": 0.05, "proportion": 0.85,
         "parenchyma_diameter": 0.045, "tracheids_diameter": 0.022,
         "transfusion_tracheids_ratio": 1.0},
        {"name": "xylem", "cell_diameter": 0.013, "n_files": 3, "n_clusters": 4, "n_per_cluster": 3},
        {"name": "phloem", "cell_diameter": 0.008, "n_files": 3},
        {"name": "cambium", "cell_diameter": 0.01},
        {"name": "Strasburger cells", "cell_diameter": 0.02},
        {"name": "resin_duct", "n_files": 2, "lumen_diameter": 0.037,
         "cell_diameter": 0.006, "cell_width": 0.013,
         "sheath_cell_diameter": 0.018, "sheath_cell_width": 0.023},
        {"name": "stomata", "n_adaxial": 8, "n_abaxial": 11, "edge_margin": 0.1,
         "width": 0.021, "depth": 0.08, "sub_chamber": 0.06, "chamber_clearance": 2.0,
         "guard_cell_diameter": 0.02, "guard_cell_aspect": 0.7,
         "sunken": True},
        {"name": "inter_cellular_spaces", "tissue": ["mesophyll", "palisade"],
         "smoothness": [0.3, 0.0], "slit_width": [0, 0.002], "slit_every": [0, 2]},
    ]


def build_gallery_needle_data() -> OrganInputData:
    """Build the ``OrganInputData`` for the feature-showcase needle."""
    WIDTH, THICKNESS = 1.8, 1.1
    _, _, CORNER_POS, CORNER_NEG = NeedleAnatomy.pole_and_corner_angles(WIDTH, THICKNESS)

    data = OrganInputData.for_needle()
    data.set_value("resin_duct", "n_files", 2)
    data.set_value("resin_duct", "sheath_cell_diameter", 0.015)
    data.set_value("stomata", "n_files", 10)
    data.set_value("central_cylinder", "vascular_angle", 20)
    data.set_value("transfusion_tissue", "pack_circles", True)
    data.set_value("transfusion_tissue", "proportion", 0.85)
    data.set_value("transfusion_tissue", "parenchyma_diameter", 0.045)
    data.set_value("transfusion_tissue", "tracheids_diameter", 0.022)
    data.set_value("transfusion_tissue", "transfusion_tracheids_ratio", 1.5)

    data.params.append({
        "name": "hypodermis_corner", "cell_diameter": 0.0225, "n_layers": 2, "order": 5.1,
        "thickness_profile": NeedleAnatomy.corner_bump_profile(
            [(CORNER_POS, 10.0, 1.0), (CORNER_NEG, 10.0, 1.0)], floor=0.70),
        "zone_angles": {"mode": "wedge", "centers": [CORNER_POS, CORNER_NEG],
                        "half_width": 4.0},
    })
    return data


def build_nigra():
    WIDTH, THICKNESS = 1.396, 0.955
    _, ABAXIAL_POLE, CORNER_POS, CORNER_NEG = NeedleAnatomy.pole_and_corner_angles(WIDTH, THICKNESS)

    DUCTS_LARGE = [(-0.499, 0.245), (0.468, 0.238), (0.030, 0.756)]
    DUCTS_SMALL = [(0.431, 0.474)]

    MESOPHYLL_FLOOR = 0.9
    ABAXIAL_MESOPHYLL_PROFILE = sorted([
        (ABAXIAL_POLE, 1.0),
        (CORNER_NEG, MESOPHYLL_FLOOR),
        (CORNER_POS, MESOPHYLL_FLOOR),
    ])
    CAP_PROFILE = NeedleAnatomy.corner_bump_profile(
        [(ABAXIAL_POLE, 50.0, 1.0)],
        floor=MESOPHYLL_FLOOR,
    )
    CAP_ZONE_HALF_WIDTH = 45.0

    return [
        {"name": "planttype", "value": 3, "organ": "needle",
         "width": WIDTH, "thickness": THICKNESS},
        {"name": "central_cylinder", "shape": "ellipse",
         "layer_length": 0.737, "layer_thickness": 0.448,
         "vascular_width": 0.22, "vascular_height": 0.19, "vascular_angle": 25,
         "cell_diameter": 0.02},
        {"name": "endodermis", "cell_diameter": 0.03, "cell_width": 0.055, "order": 3,
         "n_points": 30},
        {"name": "mesophyll", "cell_diameter": 0.052, "cell_width": 0.044, "order": 4},
        {"name": "mesophyll_abaxial", "cell_diameter": 0.052, "cell_width": 0.044, "order": 4.1,
         "thickness_profile": ABAXIAL_MESOPHYLL_PROFILE,
         "zone_angles": {"mode": "half", "pole": ABAXIAL_POLE}},
        {"name": "mesophyll_abaxial_cap", "cell_diameter": 0.052, "cell_width": 0.044, "order": 4.2,
         "thickness_profile": CAP_PROFILE,
         "zone_angles": {"mode": "wedge", "centers": [ABAXIAL_POLE],
                         "half_width": CAP_ZONE_HALF_WIDTH}},
        {"name": "palisade", "cell_diameter": 0.055, "cell_width": 0.038, "order": 4.5},
        {"name": "hypodermis", "cell_diameter": 0.025, "cell_width": 0.03,
         "n_layers": 2, "order": 5},
        {"name": "epidermis", "cell_diameter": 0.02, "cell_width": 0.015, "order": 6},
        {"name": "transfusion_tissue", "n_layers": 2, "pack_circles": True,
         "diameter_max": 0.055, "proportion": 0.85,
         "parenchyma_diameter": 0.0505, "tracheids_diameter": 0.015,
         "transfusion_tracheids_ratio": 1.9},
        {"name": "xylem", "cell_diameter": 0.008, "n_files": 10, "n_clusters": 4, "n_per_cluster": 3},
        {"name": "phloem", "cell_diameter": 0.005, "n_files": 26},
        {"name": "cambium", "cell_diameter": 0.006},
        {"name": "Strasburger cells", "cell_diameter": 0.02},
        {"name": "resin_duct", "positions": DUCTS_LARGE,
         "lumen_diameter": 0.037,
         "cell_diameter": 0.006, "cell_width": 0.013,
         "sheath_cell_diameter": 0.018, "sheath_cell_width": 0.023},
        {"name": "resin_duct", "positions": DUCTS_SMALL,
         "lumen_diameter": 0.022,
         "cell_diameter": 0.004, "cell_width": 0.008,
         "sheath_cell_diameter": 0.011, "sheath_cell_width": 0.014},
        {"name": "stomata", "n_adaxial": 4, "n_abaxial": 9, "edge_margin": 0.1,
         "width": 0.018, "depth": 0.06, "sub_chamber": 0.045,
         "chamber_clearance": 2.0,
         "guard_cell_diameter": 0.018, "guard_cell_aspect": 0.7,
         "sunken": True},
        {"name": "inter_cellular_spaces", "tissue": ["mesophyll", "palisade"],
         "smoothness": [0.3, 0.0], "slit_width": [0, 0.002], "slit_every": [0, 2]},
    ]
