"""Smoke test and visualisation for secondary xylem / secondary growth."""

import sys
import os
import time
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('..'))

from openalea.granap.root_class import RootAnatomy
from openalea.granap.input_data import OrganInputData, InterCellularSpacesParams


SEED = 0


def make_secondary_root(**overrides) -> RootAnatomy:
    data = OrganInputData.for_dicot_root()
    data.set_value("secondary_growth", "value", True)
    data.set_value("stele", "thickness", 1.0)
    for field, value in overrides.items():
        data.set_value("secondary_xylem", field, value)
    # Intercellular spaces between axial parenchyma cells in the secondary xylem zone
    data.params.append(InterCellularSpacesParams(
        tissue=["stele"],
        inter_cellular_space_proportion=0.1,
        smoothness=0.05,
    ))
    return RootAnatomy(data, seed=SEED)


def cell_type_counts(root: RootAnatomy) -> dict:
    counts = {}
    for c in root.all_cells.cells:
        counts[c.type] = counts.get(c.type, 0) + 1
    return counts


scenarios = [
    {"label": "Defaults\n",      "kwargs": {}},
    {"label": "prop_stele=0.3\n(narrow zones)",    "kwargs": {"prop_stele": 0.3}},
    {"label": "prop_stele=1.0\n(full ring)",       "kwargs": {"prop_stele": 1.0}},
    {"label": "must_be_adjacent=True",             "kwargs": {"must_be_adjacent": True}},
]


def test_secondary_xylem(show=False):
    roots = []
    for s in scenarios:
        print(f"\n=== {s['label'].replace(chr(10), ' | ')} ===")
        t0 = time.time()
        root = make_secondary_root(**s["kwargs"])
        root.generate_cells()
        elapsed = time.time() - t0
        print(f"  Time: {elapsed:.2f}s")

        counts = cell_type_counts(root)
        print("  Cell types:")
        for t, n in sorted(counts.items()):
            print(f"    {t:25s}: {n}")

        roots.append(root)

    # ── Visualisation ─────────────────────────────────────────────────────────

    n = len(scenarios)
    n_cols = 4
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

    plt.suptitle("Dicot root — secondary xylem with Apollonian packing", fontsize=14)
    plt.tight_layout()
    if show:
        plt.show()


if __name__ == "__main__":
    test_secondary_xylem(show=True)
