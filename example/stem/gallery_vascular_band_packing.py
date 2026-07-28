"""Demo: radius-dependent vascular bundle bands in the monocot stem.

A monocot stem (atactostele) can carry more than one ``vascular_bundle`` spec,
each owning a radial band (``radius_min`` .. ``radius_max`` mm from the centre)
with its own ``n_bundles`` and placement.  Within its band a spec is laid out
either:

* ``random`` — bundles scattered through the annulus (kept non-overlapping and
  min-distance spaced, so they never clump); or
* ``even``   — bundles equally spaced on a ring at the band midpoint radius
  (circumference / ``n_bundles`` apart), rotated by the spec's ``angle`` phase.

The ``angle`` phase lets two close-radius ``even`` bands interleave: give the
outer band a half-step offset (``180 / n_bundles``) and one of its bundles falls
between each of the inner band's.

Three panels, left to right:

1. **1 band** — a single random band filling the whole ground tissue (the
   default atactostele).
2. **2 bands** — a random band near the centre + an ``even`` ring further out.
3. **3 bands** — a random band at the centre + two ``even`` bands at nearly the
   same radius, half-step interleaved so their bundles alternate around the ring.

Run as a script to see them side by side.
"""

import sys
import os
import time

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(".."))

from openalea.granap.stem_class import StemAnatomy
from openalea.granap.input_data import OrganInputData, VascularBundleParams

SEED = 0
PITH_THICKNESS = 1.8          # -> pith radius ~0.9 mm, room for several rings

# Shared 'face' bundle geometry (the canonical monocot bundle); each band adds
# its own radius / placement / angle on top of this.
BASE = dict(
    bundle_type="collateral", has_cambium=False,
    xylem_layout="face", lacuna=True, sheath="both",
    width=0.11, height=0.17,
)


def _monocot_bands(bands):
    """Monocot stem data whose single preset bundle spec is replaced by ``bands``.

    ``bands`` is a list of override dicts, one per ``vascular_bundle`` spec.
    """
    data = OrganInputData.for_monocot_stem()
    data.set_value("pith", "thickness", PITH_THICKNESS)
    # Drop the preset's single bundle spec, then add our banded specs.
    data.params = [p for p in data.params if getattr(p, "name", None) != "vascular_bundle"]
    for band in bands:
        data.params.append(VascularBundleParams(**{**BASE, **band}))
    return data


SCENARIOS = [
    ("1 band — random (whole pith)", _monocot_bands([
        dict(placement="random", n_bundles=16),          # radius_min/max = 0 -> full span
    ])),
    ("2 bands — inner random + outer even ring", _monocot_bands([
        dict(placement="random", radius_min=0.0, radius_max=0.42, n_bundles=6),
        dict(placement="even",   radius=0.70, n_bundles=11),
    ])),
    ("3 bands — random centre + two interleaved even rings", _monocot_bands([
        dict(placement="random", radius_min=0.0, radius_max=0.40, n_bundles=5),
        # Two 'even' bands at nearly the same radius; the second is offset half a
        # step (180 / n_bundles = 20 deg) so its bundles sit between the first's.
        dict(placement="even", radius=0.66, n_bundles=9, shape="ellipse"),
        dict(placement="even", radius=0.74, n_bundles=9, angle=20.0, shape="egg"),
    ])),
]


def main(show=True):
    fig, axs = plt.subplots(1, 3, figsize=(21, 8))
    for ax, (label, data) in zip(axs.ravel(), SCENARIOS):
        print(f"\n=== {label} ===")
        t0 = time.time()
        stem = StemAnatomy(data, seed=SEED)
        stem.generate_cells()
        n_v = sum(1 for c in stem.all_cells.cells
                  if c.type in ("xylem", "metaxylem", "protoxylem", "phloem",
                                "sieve element", "companion cell"))
        print(f"  Time: {time.time() - t0:.2f}s   vascular cells: {n_v}")
        stem.plot_cells(show=False, ax=ax, title=label)
        # Per-panel legend (geopandas colours tab20 from the tissues present in
        # this panel, so the same tissue can differ across panels).
        leg = ax.get_legend()
        if leg is not None:
            leg.set_title("tissue")
            for txt in leg.get_texts():
                txt.set_fontsize(6)

    plt.suptitle("Monocot stem — radius-dependent vascular bundle bands "
                 "(random / even / interleaved)", fontsize=15)
    plt.tight_layout()
    if show:
        plt.show()


if __name__ == "__main__":
    main()
