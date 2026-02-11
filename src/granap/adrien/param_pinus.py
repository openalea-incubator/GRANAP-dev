



params_pinaster = [
    # P. pinaster
    {"name": "planttype", "value": 3, "organ": "needle", "genus": "Pinus", "species": "pinaster"}, # 1 = Monocot, 2 = Dicot, 3 = Gymnosperm
    {"name": "macro", "width": 1.45, "thickness": 0.96, "vascular_shape": "half_ellipse"},
    {"name": "randomness", "value": 1.0, "smoothness": 0.3}, # 0 = No randomness, 3 = Maximum randomness; smoothness is the smoothing factor (0 = no smoothing, 1 = maximum smoothing)
    {"name": "secondarygrowth", "value": 0},
    {"name": "central_cylinder", "cell_diameter": 0.02, "layer_thickness": 0.43, "layer_length": 1.05, "transfusion_layers": 2, "transfusion_tracheids_ratio": 0.5}, # Cell diameter in millimeters
    {"name": "transfusion_tracheids", "cell_diameter": 0.05, "cell_width": 0.03},
    {"name": "transfusion_parenchyma", "cell_diameter": 0.05, "cell_width": 0.04},
    {"name": "endodermis", "cell_diameter": 0.02, "cell_width": 0.05, "n_layers": 1, "order": 3},
    {"name": "mesophyll", "cell_diameter": 0.08, "cell_width": 0.045, "n_layers": 3, "order": 4},
    {"name": "hypodermis", "cell_diameter": 0.0225, "n_layers": 2, "order": 5},
    {"name": "epidermis", "cell_diameter": 0.02, "n_layers": 1, "order": 6},
    {"name": "xylem", "n_files": 10, "cell_diameter": 0.007, "n_clusters": 4, "n_per_cluster": 3}, # Number of files
    {"name": "phloem", "n_files": 8, "cell_diameter": 0.003}, 
    {"name": "cambium", "cell_diameter": 0.003}, 
    {"name": "resin_ducts", "diameter": 0.5, "n_files": 17},
    {"name": "inter_cellular_space", "ratio": 0.5},
    {"name": "stomata", "n_files": 22, "width": 0.07},
    {"name": "Strasburger cells", "layer_diameter": 0.002, "cell_diameter": 0.05}
]

params_nigra = [
    # P. nigra
    {"name": "planttype", "value": 3, "organ": "needle", "genus": "Pinus", "species": "nigra"}, # 1 = Monocot, 2 = Dicot, 3 = Gymnosperm
    {"name": "macro", "width": 1.45, "thickness": 0.96, "vascular_shape": "ellipse"},
    {"name": "randomness", "value": 1.0, "smoothness": 0.3}, # 0 = No randomness, 3 = Maximum randomness; smoothness is the smoothing factor (0 = no smoothing, 1 = maximum smoothing)
    {"name": "secondarygrowth", "value": 0},
    {"name": "central_cylinder", "cell_diameter": 0.02, "layer_thickness": 0.46, "layer_length": 0.9, "transfusion_layers": 2, "transfusion_tracheids_ratio": 0.5}, # Cell diameter in millimeters
    {"name": "transfusion_tracheids", "cell_diameter": 0.05, "cell_width": 0.03},
    {"name": "transfusion_parenchyma", "cell_diameter": 0.05, "cell_width": 0.04},
    {"name": "endodermis", "cell_diameter": 0.02, "cell_width": 0.05, "n_layers": 1, "order": 3},
    {"name": "mesophyll", "cell_diameter": 0.08, "cell_width": 0.045, "n_layers": 3, "order": 4},
    {"name": "hypodermis", "cell_diameter": 0.0225, "n_layers": 2, "order": 5},
    {"name": "epidermis", "cell_diameter": 0.02, "n_layers": 1, "order": 6},
    {"name": "xylem", "n_files": 10, "cell_diameter": 0.007, "n_clusters": 4, "n_per_cluster": 3}, # Number of files
    {"name": "phloem", "n_files": 8, "cell_diameter": 0.003}, 
    {"name": "cambium", "cell_diameter": 0.003}, 
    {"name": "resin_ducts", "diameter": 0.5, "n_files": 17},
    {"name": "inter_cellular_space", "ratio": 0.5},
    {"name": "stomata", "n_files": 22, "width": 0.07},
    {"name": "Strasburger cells", "layer_diameter": 0.002, "cell_diameter": 0.05}
]

params_resinosa = [
    # P. resinosa
    {"name": "planttype", "value": 3, "organ": "needle", "genus": "Pinus", "species": "resinosa"}, # 1 = Monocot, 2 = Dicot, 3 = Gymnosperm
    {"name": "randomness", "value": 1.0, "smoothness": 0.3}, # 0 = No randomness, 3 = Maximum randomness; smoothness is the smoothing factor (0 = no smoothing, 1 = maximum smoothing)
    {"name": "secondarygrowth", "value": 0},
    {"name": "central_cylinder", "cell_diameter": 0.02, "layer_thickness": 0.43, "layer_length": 1.05, "transfusion_layers": 2, "transfusion_tracheids_ratio": 0.5}, # Cell diameter in millimeters
    {"name": "transfusion_tracheids", "cell_diameter": 0.05, "cell_width": 0.03},
    {"name": "transfusion_parenchyma", "cell_diameter": 0.05, "cell_width": 0.04},
    {"name": "endodermis", "cell_diameter": 0.02, "cell_width": 0.05, "n_layers": 1, "order": 3},
    {"name": "mesophyll", "cell_diameter": 0.08, "cell_width": 0.045, "n_layers": 3, "order": 4},
    {"name": "hypodermis", "cell_diameter": 0.0225, "n_layers": 2, "order": 5},
    {"name": "epidermis", "cell_diameter": 0.02, "n_layers": 1, "order": 6},
    {"name": "xylem", "n_files": 10, "cell_diameter": 0.007, "n_clusters": 4, "n_per_cluster": 3}, # Number of files
    {"name": "phloem", "n_files": 8, "cell_diameter": 0.003}, 
    {"name": "cambium", "cell_diameter": 0.003}, 
    {"name": "resin_ducts", "diameter": 0.5, "n_files": 17},
    {"name": "inter_cellular_space", "ratio": 0.5},
    {"name": "stomata", "n_files": 22, "width": 0.07},
    {"name": "Strasburger cells", "layer_diameter": 0.002, "cell_diameter": 0.05}
]

params_sylvestris = [
    # P. sylvestris
    {"name": "planttype", "value": 3, "organ": "needle", "genus": "Pinus", "species": "sylvestris"}, # 1 = Monocot, 2 = Dicot, 3 = Gymnosperm
    {"name": "randomness", "value": 1.0, "smoothness": 0.3}, # 0 = No randomness, 3 = Maximum randomness; smoothness is the smoothing factor (0 = no smoothing, 1 = maximum smoothing)
    {"name": "secondarygrowth", "value": 0},
    {"name": "central_cylinder", "cell_diameter": 0.02, "layer_thickness": 0.43, "layer_length": 1.05, "transfusion_layers": 2, "transfusion_tracheids_ratio": 0.5}, # Cell diameter in millimeters
    {"name": "transfusion_tracheids", "cell_diameter": 0.05, "cell_width": 0.03},
    {"name": "transfusion_parenchyma", "cell_diameter": 0.05, "cell_width": 0.04},
    {"name": "endodermis", "cell_diameter": 0.02, "cell_width": 0.05, "n_layers": 1, "order": 3},
    {"name": "mesophyll", "cell_diameter": 0.08, "cell_width": 0.045, "n_layers": 3, "order": 4},
    {"name": "hypodermis", "cell_diameter": 0.0225, "n_layers": 2, "order": 5},
    {"name": "epidermis", "cell_diameter": 0.02, "n_layers": 1, "order": 6},
    {"name": "xylem", "n_files": 10, "cell_diameter": 0.007, "n_clusters": 4, "n_per_cluster": 3}, # Number of files
    {"name": "phloem", "n_files": 8, "cell_diameter": 0.003}, 
    {"name": "cambium", "cell_diameter": 0.003}, 
    {"name": "resin_ducts", "diameter": 0.5, "n_files": 17},
    {"name": "inter_cellular_space", "ratio": 0.5},
    {"name": "stomata", "n_files": 22, "width": 0.07},
    {"name": "Strasburger cells", "layer_diameter": 0.002, "cell_diameter": 0.05}
]

param_data = [params_pinaster, params_nigra, params_resinosa, params_sylvestris]