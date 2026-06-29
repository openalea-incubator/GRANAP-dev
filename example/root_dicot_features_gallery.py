"""Gallery: dicot root — star-shaped xylem with Apollonian packing."""

import sys
import os
import time
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('..'))

from openalea.granap.root_class import RootAnatomy
from openalea.granap.input_data import OrganInputData


SEED = 0


def make_dicot_root(cambium_kwargs: dict = None, **xylem_kwargs) -> RootAnatomy:
    data = OrganInputData.for_dicot_root()
    for field, value in xylem_kwargs.items():
        data.set_value("xylem", field, value)
    if cambium_kwargs:
        for field, value in cambium_kwargs.items():
            data.set_value("cambium", field, value)
    return RootAnatomy(data, seed=SEED)


def cell_type_counts(root: RootAnatomy) -> dict:
    counts = {}
    for c in root.all_cells.cells:
        counts[c.type] = counts.get(c.type, 0) + 1
    return counts


BASE_KWARGS = {"arc_top": 0.04, "arc_bottom": 0.06}

BASE_CAMBIUM = {
    "cell_diameter":    0.006,
    "cell_width":       0.01,
    "inner_distance":   0.11,
    "visible_distance": 0.27,
    "arc_top":          0.05,
    "arc_bottom":       0.07,
}

scenarios = [
    {"label": "Diarch (2 peaks)\n",  "kwargs": {**BASE_KWARGS, "n_vascular_peak": 2}, "cambium": BASE_CAMBIUM},
    {"label": "Tétrarch (4 peaks)\n", "kwargs": {**BASE_KWARGS, "n_vascular_peak": 4}, "cambium": BASE_CAMBIUM},
    {"label": "Heptarch (7 peaks)",   "kwargs": {**BASE_KWARGS, "n_vascular_peak": 7}, "cambium": BASE_CAMBIUM},
    {"label": "Narrow peaks",
     "kwargs": {**BASE_KWARGS, "arc_bottom": 0.05, "arc_top": 0.05,
                "vessel_diameter": 0.07, "vessel_diameter_min": 0.04, "inner_radius": 0.04},
     "cambium": BASE_CAMBIUM},
    {"label": "Wide star\n(inner=0.15, outer=0.20)",
     "kwargs": {**BASE_KWARGS, "inner_radius": 0.15, "outer_radius": 0.20},
     "cambium": {**BASE_CAMBIUM, "inner_distance": 0.19, "visible_distance": 0.40}},
    {"label": "Circle\n(inner=0.15, outer=0.15)",
     "kwargs": {**BASE_KWARGS, "inner_radius": 0.15, "outer_radius": 0.15},
     "cambium": {**BASE_CAMBIUM, "inner_distance": 0.17, "visible_distance": 0.17}},
]


def main(show=True):
    roots = []
    for s in scenarios:
        print(f"\n=== {s['label'].replace(chr(10), ' | ')} ===")
        t0 = time.time()
        root = make_dicot_root(cambium_kwargs=s.get("cambium"), **s["kwargs"])
        root.generate_cells()
        print(f"  Time: {time.time() - t0:.2f} s")
        counts = cell_type_counts(root)
        for t, n in sorted(counts.items()):
            print(f"    {t:22s}: {n}")
        roots.append(root)

    n = len(scenarios)
    n_cols = 3
    n_rows = (n + n_cols - 1) // n_cols
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axs_flat = axs.flatten()

    for i, (root, s) in enumerate(zip(roots, scenarios)):
        root.plot_cells(show=False, ax=axs_flat[i], title=s["label"])
        legend = axs_flat[i].get_legend()
        if legend:
            legend.remove()

    for j in range(n, len(axs_flat)):
        axs_flat[j].set_visible(False)

    plt.suptitle("Dicot root — star-shaped xylem with Apollonian packing", fontsize=14)
    plt.tight_layout()
    if show:
        plt.show()


if __name__ == "__main__":
    main()
