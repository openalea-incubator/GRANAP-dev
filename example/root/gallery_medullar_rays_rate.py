"""Demo: radius-dependent medullar-ray initiation (``n_medullar_rate``).

``n_medullar`` rays start at the primary cambium; ``n_medullar_rate`` (rays per mm
of radius) adds more rays that *appear further out*, so ray density grows toward
the periphery and the inner wood stays sparse — as in real secondary xylem.
"""

import sys
import os
import time
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('..'))

from openalea.granap.root_class import RootAnatomy
from openalea.granap.input_data import OrganInputData, DicotMedularRaysParams

SEED = 0


def make_root(n_medullar: int, rate: float, start_radius_sd: float = 0.0) -> RootAnatomy:
    data = OrganInputData.for_dicot_root()
    data.set_value("secondary_growth", "value", True)
    data.set_value("stele", "thickness", 1.2)
    data.params.append(DicotMedularRaysParams(
        n_medullar=n_medullar,
        n_medullar_rate=rate,
        start_radius=0.0,
        start_radius_sd=start_radius_sd,
        allow_non_vascular=True,
    ))
    root = RootAnatomy(data, seed=SEED)
    root.generate_cells()
    return root


scenarios = [
    {"label": "rate = 0 (fixed 6 rays)",      "n_medullar": 6, "rate": 0.0,   "sd": 0.0},
    {"label": "rate = 50 (~10 / 0.2 mm)",     "n_medullar": 6, "rate": 50.0,  "sd": 0.08},
    {"label": "rate = 150 (dense periphery)", "n_medullar": 6, "rate": 150.0, "sd": 0.12},
]


def main(show=True):
    fig, axs = plt.subplots(1, 3, figsize=(21, 7))
    for ax, s in zip(axs, scenarios):
        print(f"\n=== {s['label']} ===")
        t0 = time.time()
        root = make_root(s["n_medullar"], s["rate"], s["sd"])
        n_ray = sum(1 for c in root.vascular_cells.cells if c.type == "medullar_ray")
        print(f"  Time: {time.time() - t0:.2f}s   medullar-ray cells: {n_ray}")
        root.plot_cells(show=False, ax=ax, title=s["label"])
        leg = ax.get_legend()
        if leg:
            leg.remove()
        ax.set_xlim(-0.55, 0.55)
        ax.set_ylim(-0.55, 0.55)

    plt.suptitle("Medullar rays — radius-dependent initiation (n_medullar_rate)", fontsize=14)
    plt.tight_layout()
    if show:
        plt.show()


if __name__ == "__main__":
    main()
