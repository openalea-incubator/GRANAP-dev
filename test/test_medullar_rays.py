"""Visualisation test for medullar ray placement (n_medullar × allow_non_vascular)."""

import sys
import os
import time
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('..'))

from openalea.granap.root_class import RootAnatomy
from openalea.granap.input_data import OrganInputData

SEED = 0

def make_root(n_medullar: int, allow_non_vascular: bool) -> RootAnatomy:
    data = OrganInputData.for_dicot_root()
    data.set_value("secondary_growth",   "value",              True)
    data.set_value("stele",              "thickness",          1.0)
    data.set_value("medullar_rays",      "n_medullar",         n_medullar)
    data.set_value("medullar_rays",      "allow_non_vascular", allow_non_vascular)
    return RootAnatomy(data, seed=SEED)


scenarios = [
    {"n_medullar": 12, "allow_non_vascular": False},
    {"n_medullar": 12, "allow_non_vascular": True},
    {"n_medullar": 3, "allow_non_vascular": False},
    {"n_medullar": 3, "allow_non_vascular": True},
]


def test_medullar_rays(show=False):
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    for ax, s in zip(axs, scenarios):
        label = f"n_medullar={s['n_medullar']}\nallow_non_vascular={s['allow_non_vascular']}"
        print(f"\n=== {label.replace(chr(10), ' | ')} ===")
        t0 = time.time()
        root = make_root(s["n_medullar"], s["allow_non_vascular"])
        root.generate_cells()
        elapsed = time.time() - t0
        print(f"  Time: {elapsed:.2f}s")

        types = {}
        for c in root.all_cells.cells:
            types[c.type] = types.get(c.type, 0) + 1
        for c in root.vascular_cells.cells:
            types[c.type] = types.get(c.type, 0) + 1
        for t, n in sorted(types.items()):
            print(f"    {t:25s}: {n}")

        mr_groups = len(set(
            c.id_group for c in root.vascular_cells.cells if c.type == "medullar_ray"
        ))
        print(f"  medullar_ray cell groups: {mr_groups}")

        root.plot_cells(show=False, ax=ax, title=label)
        legend = ax.get_legend()
        if legend:
            legend.remove()

    plt.suptitle("Medullar rays — n_medullar × allow_non_vascular", fontsize=13)
    plt.tight_layout()
    if show:
        plt.show()


if __name__ == "__main__":
    test_medullar_rays(show=True)
