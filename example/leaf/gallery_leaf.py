"""Gallery: monocot vs dicot leaf cross-sections, side by side.

* **monocot** — spongy / palisade / spongy mesophyll, an even row of transverse
  'face' veins (xylem adaxial), amphistomatous;
* **dicot** — dorsiventral palisade (adaxial) / spongy (abaxial), a central midrib
  + scattered minor collateral veins, stomata denser on the abaxial face.

Both built from the ``OrganInputData.for_*_leaf()`` presets.
"""

import os
import sys
import time

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(".."))

from openalea.granap.input_data import OrganInputData
from openalea.granap.leaf_class import LeafAnatomy

SEED = 0


def _twisted_girdered_monocot():
    """A grass-like monocot leaf showing the shape/vein options: a twisted lamina
    with sclerenchyma girders bridging each vein to both epidermes."""
    d = OrganInputData.for_monocot_leaf()
    d.set_values("planttype", twist_amplitude=0.14, twist_waves=1.5)   # not perfectly straight
    d.set_values("vascular_bundle", girder_adaxial=True, girder_abaxial=True,   # grass girders
                 girder_base_width=0.09)
    return d


SCENARIOS = [
    ("monocot leaf — uniform mesophyll + inter-bundle aerenchyma",
     OrganInputData.for_monocot_leaf()),
    ("dicot leaf — palisade (adaxial) / spongy (abaxial), midrib + minor veins",
     OrganInputData.for_dicot_leaf()),
    ("monocot leaf — twisted lamina + sclerenchyma girders to both epidermes",
     _twisted_girdered_monocot()),
]


def main(show=True):
    fig, axs = plt.subplots(len(SCENARIOS), 1, figsize=(14, 10))
    for ax, (label, data) in zip(axs, SCENARIOS):
        print(f"\n=== {label} ===")
        t0 = time.time()
        leaf = LeafAnatomy(data, seed=SEED)
        leaf.generate_cells()
        n_v = sum(1 for c in leaf.all_cells.cells
                  if c.type in ("xylem", "phloem", "sieve element", "cambium"))
        print(f"  Time: {time.time() - t0:.2f}s   cells: {len(leaf.all_cells.cells)}   vascular: {n_v}")
        leaf.plot_cells(show=False, ax=ax, title=label)
        leg = ax.get_legend()
        if leg is not None:
            leg.set_title("tissue")
            for txt in leg.get_texts():
                txt.set_fontsize(6)
        ax.set_aspect("equal")

    plt.suptitle("Leaf cross-sections — monocot / dicot", fontsize=14)
    plt.tight_layout()
    if show:
        plt.show()


if __name__ == "__main__":
    main()
