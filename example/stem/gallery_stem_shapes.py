"""Demo: shaping a dicot stem — stem outline, cambium ring, and bundle pattern.

Four cross-sections showing how independently the three "shape" knobs can be set,
plus the non-fascicular alternative:

1. **circle · uniform** — the textbook eustele: a circular stem, a circular cambium
   ring, and one repeated bundle kind (``for_dicot_stem``).
2. **ellipse · mixed [big, small] x6** — an elliptical stem + elliptical cambium
   ring carrying two bundle *kinds* alternating around the ring (a ``bundle_pattern``).
3. **star · [big, small, small] x4 · secondary growth** — a star (lobed) secondary
   cambium with one big + two small bundles per arm, grown into a woody stem
   (secondary xylem behind each bundle, a closed cambium ring linking them).
4. **continuous cambium** — the non-fascicular vascular cylinder: an uninterrupted
   xylem / cambium / phloem ring with the xylem organised into radial files
   (``for_dicot_stem_continuous``).

The first three are the fascicular eustele driven by ``base_shape`` (stem outline),
``vascular_bundle.ring_shape`` / ``secondary_cambium.shape`` (cambium ring), and a
``bundle_pattern`` of ``vascular_bundle`` *kinds*; the last is the continuous
cylinder.  Run as a script to see them side by side.
"""

import sys
import os
import time

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(".."))

from openalea.granap.stem_class import StemAnatomy
from openalea.granap.input_data import (
    OrganInputData, BaseShapeParams, VascularBundleParams, BundlePatternParams,
)

SEED = 0


def _drop(data, name):
    """Remove every param entry with the given name (returns the data)."""
    data.params = [p for p in data.params if getattr(p, "name", None) != name]
    return data


# 1. Circular stem, circular cambium, one uniform bundle kind (classic eustele).
def _circle_uniform():
    return OrganInputData.for_dicot_stem()


# 2. Elliptical stem + elliptical cambium ring, two bundle kinds alternating.
def _ellipse_mixed():
    data = OrganInputData.for_dicot_stem()
    data.params.append(BaseShapeParams(shape="ellipse", width=1.35, height=1.0))
    _drop(data, "vascular_bundle")
    data.params += [
        VascularBundleParams(kind="big",   ring_shape="ellipse", ring_ellipse_ratio=0.75,
                             width=0.16, height=0.22),
        VascularBundleParams(kind="small", ring_shape="ellipse", ring_ellipse_ratio=0.75,
                             width=0.09, height=0.14),
        BundlePatternParams(sequence=["big", "small"], repeats=6),
    ]
    return data


# 3. Star (lobed) secondary cambium, one big + two small bundles per arm, secondary
#    growth grows it into a woody stem.  The primary bundle ring is a 4-arm star and
#    the secondary cambium is a 4-arm star *aligned with it* — its lobes bulge over
#    the big bundles (peak radius outer), which sit on the arms (align_to_arms).
def _star_secondary():
    data = OrganInputData.for_dicot_stem()
    data.set_value("secondary_growth", "value", True)
    # Lobed secondary cambium, peaks (outer, 0.82) over the arms — aligned with the
    # primary star so each lobe bulges over a big bundle.
    data.set_values("secondary_cambium", shape="star", n_peaks=4,
                    radius_peak_side=0.82, radius_valley_side=0.66,
                    arc_peak_side=0.20, arc_valley_side=0.30)
    _drop(data, "vascular_bundle")
    data.params += [
        # Star bundle ring: absolute peak/valley radii (same params as the root),
        # a pronounced 4-arm star around the pith edge (r_prim = 0.4 here).
        VascularBundleParams(kind="big",   ring_shape="star", n_peaks=4,
                             radius_peak_side=0.42, radius_valley_side=0.28,
                             arc_peak_side=0.12, arc_valley_side=0.10,
                             width=0.18, height=0.26),
        VascularBundleParams(kind="small", ring_shape="star", n_peaks=4,
                             radius_peak_side=0.42, radius_valley_side=0.28,
                             arc_peak_side=0.12, arc_valley_side=0.10,
                             width=0.09, height=0.14),
        # big bundles on the arms (align_to_arms); equal-angle spacing so the two
        # smalls sit at even angular steps between the arms; angle rotates the pattern.
        BundlePatternParams(sequence=["big", "small", "small"], repeats=4,
                            spacing="angle", align_to_arms=True, angle=0.0),
    ]
    return data


# 4. Continuous (non-fascicular) vascular cylinder — the continuous cambium ring.
def _continuous():
    data = OrganInputData.for_dicot_stem_continuous()
    data.set_value("vascular_cylinder", "xylem_layout", "files")
    return data


SCENARIOS = [
    ("circle stem · circle cambium · uniform bundles", _circle_uniform()),
    ("ellipse stem · ellipse cambium · [big, small] x6", _ellipse_mixed()),
    ("star cambium · [big, small, small] x4 · secondary growth", _star_secondary()),
    ("continuous cambium — cylinder (radial files)", _continuous()),
]


def main(show=True):
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    for ax, (label, data) in zip(axs.ravel(), SCENARIOS):
        print(f"\n=== {label} ===")
        t0 = time.time()
        stem = StemAnatomy(data, seed=SEED)
        stem.generate_cells()
        print(f"  Time: {time.time() - t0:.2f}s   cells: {len(stem.all_cells.cells)}")
        stem.plot_cells(show=False, ax=ax, title=label)
        leg = ax.get_legend()
        if leg is not None:
            leg.set_title("tissue")
            for txt in leg.get_texts():
                txt.set_fontsize(6)

    plt.suptitle("Dicot stem — stem / cambium / bundle shapes, and a continuous "
                 "cylinder", fontsize=15)
    plt.tight_layout()
    if show:
        plt.show()


if __name__ == "__main__":
    main()
