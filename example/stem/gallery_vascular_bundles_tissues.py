"""Demo: stem vascular bundle types — TISSUE view (no cells).

Same six scenarios as ``stem_bundles.py`` but drawn with ``plot_tissues``: the
tissue *zones* before any cell is placed.  Layer rings are labelled and
colour-coded; each bundle shows its internal arrangement — xylem
(light-steel-blue, with the vessel outlines in steel-blue on top), phloem
(goldenrod), cambium (green), sclerenchyma sheath (silver) — and cavities
(hollow pith / protoxylem lacuna) read as white voids.

A shared legend (bottom) keys the vascular-zone colours; the layer rings carry
their own on-figure labels.  Run as a script to see the six panels.
"""

import sys
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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

# Keys the vascular-zone colours used by plot_tissues (layer rings are labelled
# on the figure by plot_tissues itself).
_LEGEND = [
    ("xylem zone", "lightsteelblue"),
    ("xylem vessels", "steelblue"),
    ("phloem", "goldenrod"),
    ("cambium", "limegreen"),
    ("sclerenchyma sheath", "silver"),
    ("bundle envelope", "khaki"),
    ("cavity / lacuna (void)", "white"),
]


def main(show=True):
    fig, axs = plt.subplots(2, 3, figsize=(19, 13))
    for ax, (label, data) in zip(axs.ravel(), SCENARIOS):
        print(f"\n=== {label} ===")
        stem = StemAnatomy(data, seed=SEED)
        # plot_tissues does its own dry-run allocation; no generate_cells needed.
        # fuse=True merges the many pith/cortex sub-rings into one zone each.
        stem.plot_tissues(ax=ax, show=False, labels=True, fuse=True)
        ax.set_title(label)

    handles = [Patch(facecolor=c, edgecolor="gray", label=name) for name, c in _LEGEND]
    fig.legend(handles=handles, loc="lower center", ncol=len(_LEGEND), fontsize=8,
               frameon=True, title="vascular-zone colours")
    plt.suptitle("Stem vascular bundles — tissue view (zones before cells)", fontsize=15)
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    if show:
        plt.show()


if __name__ == "__main__":
    main()
