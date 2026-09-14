"""Monocot leaf cross-section demo (Phases 1-2).

Builds a wide, thin lamina whose concentric peeling reads top-to-bottom as
epidermis / spongy / palisade / spongy / epidermis — the monocot mesophyll
stack — with an even row of transverse veins along the mid-plane (xylem adaxial,
phloem abaxial) and stomata on the abaxial epidermis.  Per-vein thickness bumps
and the dicot leaf come in later phases (see LEAF_PLAN.md).
"""

import os
import sys
import time

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(".."))

from openalea.granap.input_data import OrganInputData
from openalea.granap.leaf_class import LeafAnatomy

SEED = 0


def main(show=True):
    print("=== monocot leaf ===")
    t0 = time.time()
    leaf = LeafAnatomy(OrganInputData.for_monocot_leaf(), seed=SEED)
    leaf.generate_cells()
    counts = {}
    for c in leaf.all_cells.cells:
        counts[c.type] = counts.get(c.type, 0) + 1
    print(f"  Time: {time.time() - t0:.2f}s   cells: {len(leaf.all_cells.cells)}")
    for t in sorted(counts):
        print(f"    {t:12s} {counts[t]}")

    fig, ax = plt.subplots(figsize=(13, 3.5))
    leaf.plot_cells(show=False, ax=ax,
                    title="Monocot leaf — veins (xylem adaxial) + abaxial stomata")
    leg = ax.get_legend()
    if leg is not None:
        leg.set_title("tissue")
        for txt in leg.get_texts():
            txt.set_fontsize(7)
    ax.set_aspect("equal")
    plt.tight_layout()
    if show:
        plt.show()


if __name__ == "__main__":
    main()
