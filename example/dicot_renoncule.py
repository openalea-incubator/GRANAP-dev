"""Build a dicot root from the ``for_dicot_root`` preset and plot it."""

import os
import sys

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(".."))

from openalea.granap.root_class import RootAnatomy
from openalea.granap.input_data import OrganInputData

SEED = 0


def main():
    data = OrganInputData.for_dicot_root()
    root = RootAnatomy(data, seed=SEED)
    # ── Primary growth tuning surface (override-friendly) ──────────────────
    # Stele size + radial parenchyma gradient.
    root.update_params("stele", "thickness",            0.31)
    root.update_params("stele", "cell_diameter",        0.01)
    root.update_params("stele", "cell_diameter_center", 0.030)
    # Star xylem (n arms, radial extent, vessel size gradient centre->tip).
    root.update_params("xylem", "n_vascular_peak",   6)
    root.update_params("xylem", "inner_radius",      0.05)
    root.update_params("xylem", "outer_radius",      0.155)
    root.update_params("xylem", "arc_top",           0.01)
    root.update_params("xylem", "arc_bottom",        0.015)
    root.update_params("xylem", "vessel_diameter",     0.03)
    root.update_params("xylem", "vessel_diameter_min", 0.01)
    root.update_params("xylem", "pith_radius",       0.0)
    root.update_params("xylem", "gradient_inflection", 0.05/0.14)
    root.update_params("xylem", "gradient_steepness", 3)
    
    # Primary phloem (between the star arms, near the cambium).
    root.update_params("phloem", "sieve_diameter",    0.007)
    root.update_params("phloem", "relative_distance", 0.1)
    root.update_params("phloem", "cluster_height",    0.05)
    root.update_params("phloem", "cluster_width", 0.03)
    
    # Primary cambium ring (valleys first, maturing outward).
    root.update_params("cambium", "inner_distance",   0.051)
    root.update_params("cambium", "outer_distance",   0.14)
    root.update_params("cambium", "visible_distance", 0.1)
    root.update_params("cambium", "arc_top",           0.026)
    root.update_params("cambium", "arc_bottom",        0.036)
    root.update_params("cambium", "cell_diameter", 0.005)
    root.update_params("cambium", "cell_width", 0.01)

    # Pericycle
    root.update_params("endodermis", "cell_diameter", 0.02)
    root.update_params("endodermis", "cell_width", 0.028)
    # Endoderme
    root.update_params("pericycle", "cell_diameter", 0.04)
    root.update_params("pericycle", "cell_width", 0.025)

    # Inner cortex
    data.params.append({
        'name': 'inner_cortex',
        'cell_diameter': 0.065,
        'cell_width': 0.05,
        'n_layers': 1,
        'shift': 0.5,
        'order': 3.5,
    })

    # Cortex
    root.update_params("cortex", "cell_diameter", 0.17)
    root.update_params("cortex", "cell_width", 0.1)
    root.update_params("cortex", "n_layers", 13)

    # Outer Cortex
    data.params.append({
        'name': 'outer_cortex',
        'cell_diameter': 0.066,
        'cell_width': 0.11,
        'n_layers': 5,
        'shift': 0.5,
        'order': 4.5,
    })

    # Exodermis
    root.update_params("exodermis", "cell_diameter", 0.023)
    root.update_params("exodermis", "cell_width", 0.05)

    # Epidermis
    root.update_params("epidermis", "cell_diameter", 0.018)
    root.update_params("epidermis", "cell_width", 0.047)

    root.update_params("inter_cellular_spaces", "smoothness", 0.05)
    root.update_params("inter_cellular_spaces", "tissue", ["inner_cortex", "cortex", "outer_cortex"])
    root.generate_cells()

    # Merge the inner/outer cortex tags into a single "cortex" tag.
    root.retag_cells("inner_cortex", "cortex")
    root.retag_cells("outer_cortex", "cortex")

    fig, ax = plt.subplots(figsize=(9, 9))
    root.plot_cells(show=False, ax=ax, title="for_dicot_root")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()


