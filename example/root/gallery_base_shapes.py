"""Build a default monocot root in each base shape; plot tissues and cells."""

import os
import sys

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(".."))

from openalea.granap.root_class import RootAnatomy
from openalea.granap.input_data import OrganInputData

SEED = 0

# One base_shape param dict per outline. width/height are in the same units as
# the organ radius; the star reuses the xylem-star parameters.
SHAPES = {
    "circle":    {"shape": "circle"},
    "triangle":  {"shape": "triangle", "width": 1.6, "height": 1.6},
    "square":    {"shape": "square",   "width": 0.8},
    "ellipse":   {"shape": "ellipse",  "width": 1.0, "height": 1.6},
    "star":      {"shape": "star", "n_peaks": 5, "radius_valley_side": 0.4,
                  "radius_peak_side": 0.6, "arc_peak_side": 0.05, "arc_valley_side": 0.10},
}


def build(shape_params):
    """Default monocot root with the given base_shape param."""
    data = OrganInputData.for_root()          # default monocot (planttype=1)
    data.params.append({"name": "base_shape", **shape_params})
    return RootAnatomy(data, seed=SEED)


def main():
    n = len(SHAPES)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8.5), squeeze=False)
    for col, (name, params) in enumerate(SHAPES.items()):
        root = build(params)

        # Top row: tissue zones (dry run, before any cell is placed).
        root.plot_tissues(ax=axes[0, col], show=False, labels=False, fuse=True)
        axes[0, col].set_title(f"{name} - tissues")

        # Bottom row: generated cells.
        root.plot_cells(ax=axes[1, col], show=False, title=f"{name} - cells")
        legend = axes[1, col].get_legend()
        if legend is not None:
            legend.remove()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
