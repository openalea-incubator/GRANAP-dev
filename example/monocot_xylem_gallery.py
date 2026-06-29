"""Gallery: monocot star-shaped xylem mode across several parameterisations."""

import sys
import os
import time
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('..'))

from openalea.granap.root_class import RootAnatomy
from openalea.granap.input_data import OrganInputData


SEED = 0


def make_star_root(**xylem_overrides) -> RootAnatomy:
    data = OrganInputData.for_root()
    data.set_value("xylem", "xylem_shape", "star")
    for field, value in xylem_overrides.items():
        data.set_value("xylem", field, value)
    root = RootAnatomy(data, seed=SEED)
    root.generate_cells()
    return root


def cell_type_counts(root: RootAnatomy) -> dict:
    counts = {}
    for c in root.all_cells.cells:
        counts[c.type] = counts.get(c.type, 0) + 1
    return counts


scenarios = [
    {"label": "Star — no pith",         "kwargs": {}},
    {"label": "Star — pith_radius=0.04", "kwargs": {"pith_radius": 0.04}},
    {"label": "Star — 3 arms",          "kwargs": {"n_vascular_peak": 3, "outer_radius": 0.12}},
    {"label": "Star — 7 arms, pith_radius=0.035, inner_radius=0.035",
     "kwargs": {"n_vascular_peak": 7, "outer_radius": 0.18, "inner_radius": 0.035,
                "pith_radius": 0.035, "arc_bottom": 0.02, "arc_top": 0.012}},
]


def main(show=True):
    roots = []
    for s in scenarios:
        print(f"\n=== {s['label']} ===")
        t0 = time.time()
        root = make_star_root(**s["kwargs"])
        print(f"  Time: {time.time() - t0:.2f}s")
        print("  Cell types:", cell_type_counts(root))
        roots.append(root)

    n_cols = 2
    n_rows = (len(scenarios) + n_cols - 1) // n_cols
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
    axs_flat = axs.flatten()

    for i, (root, s) in enumerate(zip(roots, scenarios)):
        root.plot_cells(show=False, ax=axs_flat[i], title=s["label"])
        leg = axs_flat[i].get_legend()
        if leg:
            leg.remove()

    for j in range(len(scenarios), len(axs_flat)):
        axs_flat[j].set_visible(False)

    plt.suptitle("Monocot — star xylem mode", fontsize=14)
    plt.tight_layout()
    if show:
        plt.show()


if __name__ == "__main__":
    main()
