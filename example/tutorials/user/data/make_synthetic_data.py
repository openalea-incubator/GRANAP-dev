"""
The tutorials' example data: SYNTHETIC, not measurements. They stand for what the analysis of a root section's image
would give, so that the user tutorial 05 can show how to build a section from measured traits, compare it with
observations and fit a parameter.

- measured_traits.csv: the traits "measured on the image" (mm): the stele's diameter, the cortex's layers and cells'
  diameter, the metaxylem vessels' number and diameter: the values of a section made by GRANAP (seed 3) with 2 %
  measurement noise on the sizes;
- observed_tissues.csv: that section's radius (mm), its cortex, stele and epidermis areas (mm2), with 3 % noise, and
  its air share. Its aerenchyma proportion, 0.2, is not among the traits: the tutorial fits it.

    python make_synthetic_data.py          (rewrites the two files in this folder)
"""
from pathlib import Path

import numpy as np
import pandas as pd

from openalea.granap.simulation import simulate

HERE = Path(__file__).parent
TRUE = {"stele": {"thickness": 0.32},
        "cortex": {"n_layers": 6, "cell_diameter": 0.045, "cell_width": 0.045},
        "xylem": {"n_vascular_bundles": 6, "vessel_diameter": 0.07},
        "aerenchyma": {"aerenchyma_proportion": 0.2}}


def main():
    noise = np.random.default_rng(0)
    measured = lambda value: round(float(value) * noise.normal(1., 0.02), 4)      # noqa: E731
    traits = pd.DataFrame({"trait": ["stele_diameter", "cortex_layers", "cortex_cell_diameter", "metaxylem_vessels",
                                     "metaxylem_diameter"],
                           "value": [measured(TRUE["stele"]["thickness"]), TRUE["cortex"]["n_layers"],
                                     measured(TRUE["cortex"]["cell_diameter"]), TRUE["xylem"]["n_vascular_bundles"],
                                     measured(TRUE["xylem"]["vessel_diameter"])]})
    traits.to_csv(HERE / "measured_traits.csv", index=False)

    section = simulate(TRUE, seed=3)
    areas = section.summary.set_index("tissue")["area"]
    observed = {"radius": round(section.totals["radius"] * noise.normal(1., 0.03), 4),
                **{f"{tissue}_area": round(areas[tissue] * noise.normal(1., 0.03), 5)
                   for tissue in ("cortex", "stele", "epidermis")},
                "air_share": round(section.totals["air_share"], 4)}
    pd.DataFrame({"quantity": list(observed), "value": list(observed.values())}).to_csv(
        HERE / "observed_tissues.csv", index=False)


if __name__ == "__main__":
    main()
