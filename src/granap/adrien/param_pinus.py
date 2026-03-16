
            
params_pinaster = [          
            # P. pinaster
            {"name": "planttype", "value": 3, "organ": "needle", "width": 1.8, "thickness": 1.1}, # global parameters
            {"name": "randomness", "value": 1.0, "smoothness": 0.3}, # 0 = No randomness, 3 = Maximum randomness; smoothness is the smoothing factor (0 = no smoothing, 1 = maximum smoothing)
            {"name": "central_cylinder", "shape": "half_ellipse", "cell_diameter": 0.02, "layer_thickness": 0.43, "layer_length": 1.05, "vascular_width": 0.15, "vascular_height": 0.2}, # Cell diameter in millimeters
            {"name": "transfusion_tissue", "tracheids_diameter": 0.05, "parenchyma_diameter": 0.03, "transfusion_tracheids_ratio": 0.5, "n_layers":2},
            {"name": "endodermis", "cell_diameter": 0.02, "cell_width": 0.05, "n_layers": 1, "order": 3, "shift": 5},
            {"name": "mesophyll", "cell_diameter": 0.08, "cell_width": 0.045, "n_layers": 3, "order": 4, "shift":10},
            {"name": "hypodermis", "cell_diameter": 0.0225, "n_layers": 2, "order": 5},
            {"name": "epidermis", "cell_diameter": 0.02, "n_layers": 1, "order": 6},
            {"name": "xylem", "n_files": 10, "cell_diameter": 0.007, "n_clusters": 4, "n_per_cluster": 3}, # Number of files
            {"name": "phloem", "n_files": 8, "cell_diameter": 0.003}, 
            {"name": "cambium", "cell_diameter": 0.002}, 
            {"name": "resin_duct", "diameter": 0.1, "n_files": 3, "cell_diameter": 0.02},
            {"name": "inter_cellular_spaces", "mesophyll": 0.01},
            {"name": "stomata", "n_files": 4, "width": 0.025, "depth": 0.06, "sub_chamber": 0.04},
            {"name": "Strasburger cells", "layer_diameter": 0.002, "cell_diameter": 0.05}
        ]

params_nigra = [
    # P. nigra
    {"name": "planttype", "value": 3, "organ": "needle", "genus": "Pinus", "species": "nigra"}, # 1 = Monocot, 2 = Dicot, 3 = Gymnosperm
    {"name": "randomness", "value": 1.0, "smoothness": 0.3}, # 0 = No randomness, 3 = Maximum randomness; smoothness is the smoothing factor (0 = no smoothing, 1 = maximum smoothing)
    {"name": "central_cylinder", "cell_diameter": 0.02, "layer_thickness": 0.46, "layer_length": 0.9, "transfusion_layers": 2, "transfusion_tracheids_ratio": 0.5}, # Cell diameter in millimeters
    {"name": "transfusion_tissue", "tracheids_diameter": 0.05, "parenchyma_diameter": 0.03, "transfusion_tracheids_ratio": 0.5, "n_layers":2},
    {"name": "endodermis", "cell_diameter": 0.02, "cell_width": 0.05, "n_layers": 1, "order": 3},
    {"name": "mesophyll", "cell_diameter": 0.08, "cell_width": 0.045, "n_layers": 3, "order": 4},
    {"name": "hypodermis", "cell_diameter": 0.0225, "n_layers": 2, "order": 5},
    {"name": "epidermis", "cell_diameter": 0.02, "n_layers": 1, "order": 6},
    {"name": "xylem", "n_files": 10, "cell_diameter": 0.007, "n_clusters": 4, "n_per_cluster": 3}, # Number of files
    {"name": "phloem", "n_files": 8, "cell_diameter": 0.003}, 
    {"name": "cambium", "cell_diameter": 0.003}, 
    {"name": "resin_ducts", "diameter": 0.1, "n_files": 10, "cell_diameter": 0.02},
    {"name": "inter_cellular_space", "mesophyll": 0.01},
    {"name": "stomata", "n_files": 12, "width": 0.025, "depth": 0.06, "sub_chamber": 0.04},
    {"name": "Strasburger cells", "layer_diameter": 0.002, "cell_diameter": 0.05}
]


params_sylvestris = [
    # P. sylvestris
    {"name": "planttype", "value": 3, "organ": "needle", "genus": "Pinus", "species": "sylvestris"}, # 1 = Monocot, 2 = Dicot, 3 = Gymnosperm
    {"name": "randomness", "value": 1.0, "smoothness": 0.3}, # 0 = No randomness, 3 = Maximum randomness; smoothness is the smoothing factor (0 = no smoothing, 1 = maximum smoothing)
    {"name": "central_cylinder", "cell_diameter": 0.02, "layer_thickness": 0.43, "layer_length": 1.05, "transfusion_layers": 2, "transfusion_tracheids_ratio": 0.5}, # Cell diameter in millimeters
    {"name": "transfusion_tissue", "tracheids_diameter": 0.05, "parenchyma_diameter": 0.03, "transfusion_tracheids_ratio": 0.5, "n_layers":2},
    {"name": "endodermis", "cell_diameter": 0.02, "cell_width": 0.05, "n_layers": 1, "order": 3},
    {"name": "mesophyll", "cell_diameter": 0.08, "cell_width": 0.045, "n_layers": 3, "order": 4},
    {"name": "hypodermis", "cell_diameter": 0.0225, "n_layers": 2, "order": 5},
    {"name": "epidermis", "cell_diameter": 0.02, "n_layers": 1, "order": 6},
    {"name": "xylem", "n_files": 10, "cell_diameter": 0.007, "n_clusters": 4, "n_per_cluster": 3}, # Number of files
    {"name": "phloem", "n_files": 8, "cell_diameter": 0.003}, 
    {"name": "cambium", "cell_diameter": 0.003}, 
    {"name": "resin_ducts", "diameter": 0.1, "n_files": 10, "cell_diameter": 0.02},
    {"name": "inter_cellular_space", "mesophyll": 0.01},
    {"name": "stomata", "n_files": 12, "width": 0.025, "depth": 0.06, "sub_chamber": 0.04},
    {"name": "Strasburger cells", "layer_diameter": 0.002, "cell_diameter": 0.05}
]

param_data = [params_pinaster, params_nigra, params_sylvestris]