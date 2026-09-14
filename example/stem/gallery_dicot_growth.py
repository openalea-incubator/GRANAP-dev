"""Demo: dicot stem growth — primary bundles vs the closed cambium ring.

The *same* eustele — one ring of collateral bundles (xylem inner / phloem outer /
fascicular cambium between) around a central pith — drawn at two growth stages:

1. **primary growth** (``secondary_growth = False``) — the cambium is visible
   only *inside* each bundle (the fascicular cambium strip); the bundles stay
   discrete, separated by pith/cortex parenchyma.
2. **secondary growth** (``secondary_growth = True``) — the interfascicular
   cambium fills the gaps between the bundles, so the fascicular cambia join
   into one continuous meristematic **cambium ring** running the whole eustele.

Only the ``secondary_growth`` flag differs between the two panels; the bundle
spec, count and seed are identical, so the second panel is literally the first
with the cambium ring closed.

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


def _dicot(secondary_growth):
    """Dicot stem preset with the secondary-growth flag set."""
    data = OrganInputData.for_dicot_stem()
    data.set_value("secondary_growth", "value", secondary_growth)
    return data


SCENARIOS = [
    ("primary growth — fascicular cambium only", _dicot(False)),
    ("secondary growth — continuous cambium ring", _dicot(True)),
]


def main(show=True):
    fig, axs = plt.subplots(1, 2, figsize=(15, 8))
    for ax, (label, data) in zip(axs.ravel(), SCENARIOS):
        print(f"\n=== {label} ===")
        t0 = time.time()
        stem = StemAnatomy(data, seed=SEED)
        stem.generate_cells()
        n_cambium = sum(1 for c in stem.all_cells.cells if c.type == "cambium")
        print(f"  Time: {time.time() - t0:.2f}s   cambium cells: {n_cambium}")
        stem.plot_cells(show=False, ax=ax, title=label)
        # Per-panel legend (geopandas colours tab20 from the tissues present in
        # this panel, so the same tissue can differ across panels).
        leg = ax.get_legend()
        if leg is not None:
            leg.set_title("tissue")
            for txt in leg.get_texts():
                txt.set_fontsize(6)

    plt.suptitle("Dicot stem growth — the same eustele before and after the "
                 "cambium ring closes", fontsize=15)
    plt.tight_layout()
    if show:
        plt.show()


if __name__ == "__main__":
    main()
