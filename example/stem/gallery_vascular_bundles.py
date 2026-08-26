"""Demo: stem vascular bundle types and the hollow (fistular) pith.

Renders the canonical stem anatomies GRANAP can build:

* dicot eustele — a ring of open **collateral** bundles (xylem in / phloem out /
  cambium between) around a solid pith;
* dicot **bicollateral** — phloem on both sides of the xylem;
* dicot **concentric** — amphivasal (xylem rings a phloem core) and amphicribral
  (phloem rings a xylem core);
* monocot atactostele — scattered **'face'** bundles (metaxylem at the middle,
  a protoxylem bundle + a lacuna toward the centre, phloem toward the surface)
  wrapped in a sclerenchyma sheath, on a solid pith and on a hollow culm.

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


def _dicot(bundle_overrides=None):
    data = OrganInputData.for_dicot_stem()
    for k, v in (bundle_overrides or {}).items():
        data.set_value("vascular_bundle", k, v)
    return data


def _monocot(cavity_radius=0.0):
    data = OrganInputData.for_monocot_stem()
    if cavity_radius:
        data.set_value("pith", "cavity_radius", cavity_radius)
    return data


SCENARIOS = [
    ("dicot — collateral",  _dicot()),
    ("dicot — bicollateral",        _dicot({"bundle_type": "bicollateral", "inner_phloem_fraction": 0.2})),
    ("dicot — amphivasal",          _dicot({"bundle_type": "concentric", "concentric_type": "amphivasal",
                                            "shape": "circle", "width": 0.16, "height": 0.16})),
    ("dicot — amphicribral",        _dicot({"bundle_type": "concentric", "concentric_type": "amphicribral",
                                            "shape": "circle", "width": 0.16, "height": 0.16})),
    ("monocot — face (solid pith)", _monocot()),
    ("monocot — face (hollow culm)", _monocot(cavity_radius=0.14)),
]


def main(show=True):
    fig, axs = plt.subplots(2, 3, figsize=(19, 13))
    for ax, (label, data) in zip(axs.ravel(), SCENARIOS):
        print(f"\n=== {label} ===")
        t0 = time.time()
        stem = StemAnatomy(data, seed=SEED)
        stem.generate_cells()
        n_v = sum(1 for c in stem.all_cells.cells
                  if c.type in ("xylem", "phloem", "sieve element", "cambium"))
        print(f"  Time: {time.time() - t0:.2f}s   vascular cells: {n_v}")
        stem.plot_cells(show=False, ax=ax, title=label)
        # Keep a per-panel legend: geopandas colours the `tab20` categories from
        # the set of tissues present in *this* panel, so the same tissue can map
        # to different colours across panels — the per-panel legend is the key.
        leg = ax.get_legend()
        if leg is not None:
            leg.set_title("tissue")
            for txt in leg.get_texts():
                txt.set_fontsize(6)

    plt.suptitle("Stem vascular bundles — collateral / bicollateral / concentric / monocot face",
                 fontsize=15)
    plt.tight_layout()
    if show:
        plt.show()


if __name__ == "__main__":
    main()
