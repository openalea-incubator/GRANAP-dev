"""Build a dicot root from scratch (no ``for_*`` preset), tune to nettle cytomine data and plot it.

Configuration model: an organ is configured through its ``OrganInputData`` and
then **built once** — construction snapshots the params and parses the vascular
geometry. So all tuning is applied to ``data`` *before* the organ is built, using
``data.set_value(...)`` for existing entries and ``data.params.append(...)`` for
new tissue layers.

Here the layer stack is assembled explicitly: start from an empty
``OrganInputData(params=[])`` and ``params.append(...)`` each dicot layer, rather
than calling ``OrganInputData.for_dicot_root()``.
"""

import os
import sys

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(".."))

from openalea.granap.root_class import RootAnatomy
from openalea.granap.input_data import (
    OrganInputData,
    PlantTypeParams,
    SteleDicotParams,
    DicotXylemParams,
    DicotCambiumParams,
    DicotSecondaryGrowthParams,
    DicotSecondaryXylemParams,
    SecondaryCambiumParams,
    DicotSecondaryPhloemParams
)

SEED = 0


def main(show=True):
    data = OrganInputData(params = [])
    data.params.append(PlantTypeParams(value = 2))  # dicot
    data.params.append(SteleDicotParams())
    data.params.append(DicotXylemParams())
    data.params.append(DicotCambiumParams())
    data.params.append(DicotSecondaryGrowthParams())
    data.params.append(DicotSecondaryXylemParams())
    data.params.append(SecondaryCambiumParams())
    data.params.append(DicotSecondaryPhloemParams())

    # Focus-ellipse outline from a measured contour profile :
    shape_params = {"shape": "focus_ellipse", "profile": [
        (0.0, 2.5),
        (1.28, 2),
        (1.9, 0.0),
    ]}
    data.params.append({"name": "base_shape", **shape_params})

    data.set_value("secondary_growth", "value", True)
    # -- Stele size + radial parenchyma gradient ----------------------------
    data.set_value("stele", "cell_diameter",        0.012)
    data.set_value("stele", "cell_diameter_center", 0.01)

    # -- Star xylem (n arms, radial extent, vessel size gradient) -----------
    data.set_value("xylem", "n_vascular_peak",     2)
    data.set_value("xylem", "radius_valley_side",        0.018)
    data.set_value("xylem", "radius_peak_side",        0.08)
    data.set_value("xylem", "arc_peak_side",             0.013)
    data.set_value("xylem", "arc_valley_side",          0.014)
    data.set_value("xylem", "vessel_diameter",     0.035)
    data.set_value("xylem", "vessel_diameter_min", 0.012)
    data.set_value("xylem", "pith_radius",         0.0)
    data.set_value("xylem", "gradient_inflection", 0.05)
    data.set_value("xylem", "gradient_steepness",  3)
    data.set_value("xylem", "enforce_gradient_min", 1)
    data.set_value("xylem", "allow_ellipse", False)
    # -- Secondary xylem  ---------------------------------------------------
    data.set_value("secondary_xylem", "prop_stele", 0.65)
    data.set_value("secondary_xylem", "cell_diameter", 0.012)
    data.set_value("secondary_xylem", "cell_width", 0.010)
    data.set_value("secondary_xylem", "vessel_diameter", 0.1)
    data.set_value("secondary_xylem", "vessel_diameter_sd", 0.015)
    data.set_value("secondary_xylem", "vessel_diameter_min", 0.05)
    data.set_value("secondary_xylem", "gradient_function", "gaussian")
    data.set_value("secondary_xylem", "allow_ellipse", True)
    data.set_value("secondary_xylem", "prop_vessel_ring", 0.307)
    data.set_value("secondary_xylem", "must_be_adjacent", False)
    data.set_value("secondary_xylem", "parenchyma_diameter", 0.022)
    data.set_value("secondary_xylem", "parenchyma_diameter_sd", 0.003)
    data.set_value("secondary_xylem", "parenchyma_width_sd", 0.005)
    data.set_value("secondary_xylem", "parenchyma_width", 0.015)     

    # -- Primary cambium ring (valleys first, maturing outward) -------------
    data.set_value("cambium", "radius_valley_side",   0.019)
    data.set_value("cambium", "radius_peak_side",   0.082)
    data.set_value("cambium", "visible_distance", 0.0)
    data.set_value("cambium", "arc_peak_side",          0.014)
    data.set_value("cambium", "arc_valley_side",       0.015)
    # -- Secondary cambium ring  --------------------------------------------
    # Smooth best-fit superellipse (focus_ellipse) fitted to the measured
    # cambium contour: 
    n_layer = 4
    data.set_value("secondary_cambium", "shape", "focus_ellipse")
    data.set_value("secondary_cambium", "profile", [
        (0.00,  2.40),
        (0.70,  2.10),
        (1.06,  1.55),
        (1.415, 0.00),
    ])
    data.set_value("secondary_cambium", "cell_diameter",    0.009)
    data.set_value("secondary_cambium", "cell_width",       0.029)
    data.set_value("secondary_cambium", "n_layers", n_layer)
    # Secondary Phloem (between the star arms, near the cambium) -----------
    data.set_value("secondary_phloem", "height",    0.12)
    data.set_value("secondary_phloem", "top_width", 1.5)
    data.set_value("secondary_phloem", "alive_distance", 0.09)
    data.set_value("secondary_phloem", "sieve_diameter", 0.022)
    data.set_value("secondary_phloem", "sieve_diameter_sd", 0.001)
    data.set_value("secondary_phloem", "sieve_diameter_min", 0.016)
    data.set_value("secondary_phloem", "prop_sieve", 0.5)
    data.set_value("secondary_phloem", "companion_diameter", 0.01)
    data.set_value("secondary_phloem", "companion_width", 0.002)
    data.set_value("secondary_phloem", "parenchyma_diameter", 0.012)
    data.set_value("secondary_phloem", "parenchyma_width", 0.016)


    # -- Phellogen layer ----------------------------------------------------
    data.params.append({
        "name": "phellem",
        "cell_diameter": 0.013,
        "cell_width": 0.04,
        "n_layers": 1,
        "shift": 0.5,
        "order": 4.5,
    })
    data.params.append({
        "name": "phellogen",
        "cell_diameter": 0.012,
        "cell_width": 0.04,
        "n_layers": 2,
        "shift": 0.5,
        "order": 4,
    })
    data.params.append({
        "name": "phelloderm",
        "cell_diameter": 0.014,
        "cell_width": 0.05,
        "n_layers": 10,
        "shift": 0.5,
        "order": 3.5,
    })


    # Build once, after all configuration is in place.
    root = RootAnatomy(data, seed=SEED)
    root.generate_cells()


    fig, ax = plt.subplots(figsize=(9, 9))
    root.plot_cells(show=False, ax=ax, title="Nettle root cross section")
    plt.tight_layout()
    if show:
        plt.show()


if __name__ == "__main__":
    main()
