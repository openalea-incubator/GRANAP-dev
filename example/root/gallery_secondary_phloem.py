"""Gallery: dicot root — secondary phloem generation."""

import sys
import os
import time
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('..'))

from openalea.granap.root_class import RootAnatomy
from openalea.granap.input_data import OrganInputData


SEED = 0


def make_root(**phloem_overrides) -> RootAnatomy:
    data = OrganInputData.for_dicot_root()
    data.set_value("secondary_growth", "value", True)
    data.set_value("stele", "thickness", 1.2)
    for field, value in phloem_overrides.items():
        data.set_value("secondary_phloem", field, value)
    return RootAnatomy(data, seed=SEED)


def cell_type_counts(root: RootAnatomy) -> dict:
    counts = {}
    for c in list(root.all_cells.cells) + list(root.vascular_cells.cells):
        counts[c.type] = counts.get(c.type, 0) + 1
    return counts


scenarios = [
    {"label": "Defaults\n", "kwargs": {}},
]


def main(show=True):
    roots = []
    for s in scenarios:
        print(f"\n=== {s['label'].replace(chr(10), ' | ')} ===")
        t0 = time.time()
        root = make_root(**s["kwargs"])
        root.generate_cells()
        print(f"  Time: {time.time() - t0:.2f}s")
        for t, n in sorted(cell_type_counts(root).items()):
            print(f"    {t:25s}: {n}")
        roots.append(root)

    n = len(scenarios)
    n_cols = 3
    n_rows = (n + n_cols - 1) // n_cols
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows), squeeze=False)
    axs_flat = axs.flatten()

    for i, (root, s) in enumerate(zip(roots, scenarios)):
        root.plot_cells(show=False, ax=axs_flat[i], title=s["label"])
        legend = axs_flat[i].get_legend()
        if legend:
            legend.remove()

    for j in range(n, len(axs_flat)):
        axs_flat[j].set_visible(False)

    plt.suptitle("Dicot root — secondary phloem", fontsize=14)
    plt.tight_layout()
    if show:
        plt.show()


if __name__ == "__main__":
    main()
