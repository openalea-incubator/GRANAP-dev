"""Demo: fascicular eustele vs the continuous (non-fascicular) vascular cylinder.

Two ways a *primary* dicot stem can organise its vascular tissue around the pith:

1. **fascicular** (``for_dicot_stem``) — the textbook eustele: discrete collateral
   bundles separated by interfascicular parenchyma (Helianthus, most herbaceous
   dicots).
2. **continuous** (``for_dicot_stem_continuous``) — a non-fascicular cylinder: an
   uninterrupted ring of xylem / cambium / phloem laid down from the start (Linum,
   Ricinus, rapidly-woody dicots). Setting ``vascular_cylinder.xylem_layout`` to
   ``"files"`` forces the xylem vessels into radial files, separated by thin
   parenchyma strips inside the xylem, while the cambium ring and phloem stay
   continuous.

Three panels: the fascicular eustele, the seamless (packed) cylinder, and the
file-organised cylinder — same pith / cortex / epidermis and seed throughout.

Run as a script to see them side by side.
"""

import sys
import os
import time

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(".."))

from openalea.granap.stem_class import StemAnatomy
from openalea.granap.input_data import OrganInputData

SEED = 0


def _continuous(xylem_layout, n_xylem_files=0):
    """Continuous-cylinder preset with the xylem layout set.

    ``n_xylem_files=0`` (auto) fills one radial file per vessel of circumference, so
    every xylem pole reads as its own radial line."""
    data = OrganInputData.for_dicot_stem_continuous()
    data.set_value("vascular_cylinder", "xylem_layout", xylem_layout)
    data.set_value("vascular_cylinder", "n_xylem_files", n_xylem_files)
    return data


SCENARIOS = [
    ("fascicular eustele — discrete bundles", OrganInputData.for_dicot_stem()),
    ("continuous cylinder — packed (seamless)", _continuous("packed")),
    ("continuous cylinder — files (radial poles)", _continuous("files")),
]


def main(show=True):
    fig, axs = plt.subplots(1, 3, figsize=(21, 7))
    for ax, (label, data) in zip(axs.ravel(), SCENARIOS):
        print(f"\n=== {label} ===")
        t0 = time.time()
        stem = StemAnatomy(data, seed=SEED)
        stem.generate_cells()
        n_cambium = sum(1 for c in stem.all_cells.cells if c.type == "cambium")
        print(f"  Time: {time.time() - t0:.2f}s   cambium cells: {n_cambium}")
        stem.plot_cells(show=False, ax=ax, title=label)
        leg = ax.get_legend()
        if leg is not None:
            leg.set_title("tissue")
            for txt in leg.get_texts():
                txt.set_fontsize(6)

    plt.suptitle("Dicot stem — fascicular bundles vs a continuous vascular "
                 "cylinder", fontsize=15)
    plt.tight_layout()
    if show:
        plt.show()


if __name__ == "__main__":
    main()
