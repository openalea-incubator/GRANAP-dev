"""Build a dicot root from the ``for_dicot_root`` preset, update to renoncule cytomine data and plot it

Configuration model: an organ is configured through its ``OrganInputData`` and
then **built once** — construction snapshots the params and parses the vascular
geometry. So all tuning is applied to ``data`` *before* the organ is built, using
``data.set_value(...)`` for existing entries and ``data.params.append(...)`` for
new tissue layers.
"""

import os
import sys

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(".."))

from openalea.granap.root_class import RootAnatomy
from openalea.granap.input_data import OrganInputData

SEED = 0


def main(show=True):
    data = OrganInputData.for_dicot_root()

    # ── Stele size + radial parenchyma gradient ────────────────────────────
    data.set_value("stele", "thickness",            0.31)
    data.set_value("stele", "cell_diameter",        0.01)
    data.set_value("stele", "cell_diameter_center", 0.030)

    # ── Star xylem (n arms, radial extent, vessel size gradient) ───────────
    data.set_value("xylem", "n_vascular_peak",     6)
    data.set_value("xylem", "radius_valley_side",        0.05)
    data.set_value("xylem", "radius_peak_side",        0.155)
    data.set_value("xylem", "arc_peak_side",             0.01)
    data.set_value("xylem", "arc_valley_side",          0.015)
    data.set_value("xylem", "vessel_diameter",     0.03)
    data.set_value("xylem", "vessel_diameter_min", 0.01)
    data.set_value("xylem", "pith_radius",         0.0)
    data.set_value("xylem", "gradient_inflection", 0.05 / 0.14)
    data.set_value("xylem", "gradient_steepness",  3)

    # ── Primary phloem (between the star arms, near the cambium) ───────────
    data.set_value("phloem", "sieve_diameter",    0.007)
    data.set_value("phloem", "relative_distance", 0.1)
    data.set_value("phloem", "cluster_height",    0.05)
    data.set_value("phloem", "cluster_width",     0.03)

    # ── Primary cambium ring (valleys first, maturing outward) ─────────────
    data.set_value("cambium", "radius_valley_side",   0.051)
    data.set_value("cambium", "radius_peak_side",   0.14)
    data.set_value("cambium", "visible_distance", 0.1)
    data.set_value("cambium", "arc_peak_side",          0.026)
    data.set_value("cambium", "arc_valley_side",       0.036)
    data.set_value("cambium", "cell_diameter",    0.005)
    data.set_value("cambium", "cell_width",       0.01)

    # ── Endodermis / pericycle ─────────────────────────────────────────────
    data.set_value("endodermis", "cell_diameter", 0.02)
    data.set_value("endodermis", "cell_width",    0.028)
    data.set_value("pericycle", "cell_diameter",  0.04)
    data.set_value("pericycle", "cell_width",     0.025)

    # ── Cortex layers (inner / main / outer) ───────────────────────────────
    data.params.append({
        "name": "inner_cortex",
        "cell_diameter": 0.065,
        "cell_width": 0.05,
        "n_layers": 1,
        "shift": 0.5,
        "order": 3.5,
    })
    data.set_value("cortex", "cell_diameter", 0.17)
    data.set_value("cortex", "cell_width",    0.1)
    data.set_value("cortex", "n_layers",      13)
    data.params.append({
        "name": "outer_cortex",
        "cell_diameter": 0.066,
        "cell_width": 0.11,
        "n_layers": 5,
        "shift": 0.5,
        "order": 4.5,
    })

    # ── Exodermis / epidermis ──────────────────────────────────────────────
    data.set_value("exodermis", "cell_diameter", 0.023)
    data.set_value("exodermis", "cell_width",    0.05)
    data.set_value("epidermis", "cell_diameter", 0.018)
    data.set_value("epidermis", "cell_width",    0.047)

    # ── Intercellular spaces across the cortex tissues ─────────────────────
    data.set_value("inter_cellular_spaces", "smoothness", 0.05)
    data.set_value("inter_cellular_spaces", "tissue",
                   ["inner_cortex", "cortex", "outer_cortex"])

    # Build once, after all configuration is in place.
    root = RootAnatomy(data, seed=SEED)
    root.generate_cells()

    # Merge the inner/outer cortex tags into a single "cortex" tag.
    root.retag_cells("inner_cortex", "cortex")
    root.retag_cells("outer_cortex", "cortex")

    fig, ax = plt.subplots(figsize=(9, 9))
    root.plot_cells(show=False, ax=ax, title="Renoncule root cross section")
    plt.tight_layout()
    if show:
        plt.show()


if __name__ == "__main__":
    main()
