"""Build a dicot root from the ``for_dicot_root`` preset, update to Oak data from Berkshire Community College Bioscience Image Library and plot it

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
    data = OrganInputData.for_woody_root()
    data.remove_param("phloem")

    data.set_value("stele", "thickness", 5 + 0.4 + 0.01)
    data.set_values("secondary_cambium",
                    radius_valley_side = 2.5, 
                    radius_peak_side = 2.5, 
                    n_layers = 4)
    data.set_values("secondary_xylem", 
                    vessel_diameter = 0.06,
                    prop_stele = 0.9,
                    n_ring = 4)
    data.set_values("medullar_rays",
                    n_medullar = 16,
                    allow_non_vascular = False)

    # Build once, after all configuration is in place.
    root = RootAnatomy(data, seed=SEED)
    root.generate_cells()

    fig, ax = plt.subplots(figsize=(9, 9))
    root.plot_cells(show=False, ax=ax, title="Oak root cross section")
    plt.tight_layout()
    if show:
        plt.show()


if __name__ == "__main__":
    main()
