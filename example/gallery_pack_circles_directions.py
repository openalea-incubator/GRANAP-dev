import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(".."))

from openalea.granap.root_class import RootAnatomy
from openalea.granap.input_data import OrganInputData

scenarios = [
    {"direction": "center", "label": "center\n(large vessels near centre)"},
    {"direction": "edge",   "label": "edge\n(large vessels near boundary)"},
    {"direction": "middle", "label": "middle\n(large vessels at mid-radius)"},
    {"direction": None,     "label": "None\n(random, normal distribution)"},
]


SEED = 0


def main(show=False):
    fig, axs = plt.subplots(1, len(scenarios), figsize=(5 * len(scenarios), 5))

    for ax, s in zip(axs, scenarios):
        data = OrganInputData.for_dicot_root()
        data.set_value("xylem", "direction", s["direction"])
        root = RootAnatomy(data, seed=SEED)
        root.generate_cells()

        types = {c.type for c in root.all_cells.cells}
        n_xylem = sum(1 for c in root.all_cells.cells if c.type == "xylem")
        print(f"direction={s['direction']!r:8}  cell types: {types}  xylem seeds: {n_xylem}")
        assert n_xylem > 0, f"No xylem cells generated for direction={s['direction']!r}"

        root.plot_cells(show=False, ax=ax, title=s["label"])
        legend = ax.get_legend()
        if legend:
            legend.remove()

    plt.suptitle("Dicot xylem — pack_circles direction comparison", fontsize=13)
    plt.tight_layout()
    if show:
        plt.show()


if __name__ == "__main__":
    main(show=True)
