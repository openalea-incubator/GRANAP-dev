"""Demo: dicot stem growth — primary bundles, the ring closing, then wood.

The *same* eustele — one ring of collateral bundles (xylem inner / phloem outer /
fascicular cambium between) around a central pith — drawn at three growth stages:

1. **primary growth** (``secondary_growth = False``) — the cambium is visible
   only *inside* each bundle (the fascicular cambium strip); the bundles stay
   discrete, separated by pith/cortex parenchyma.
2. **interfascicular cambium** — still ``secondary_growth = False`` (same
   primary-sized stem, same discrete bundle layout), but built with
   ``DicotStemAnatomy._build_cambium`` forced onto its ``secondary=True``
   branch — the exact pass that closes the ring under real secondary growth.
   That branch only materialises the interfascicular cambium band; it never
   touches xylem/phloem or the pith radius, so this is the ring closing in
   isolation, with no wood and no growth in size at all — not an
   approximation of an early secondary-growth moment, but the same
   ring-closing code secondary growth itself calls, run on its own.
3. **secondary growth** (``secondary_growth = True``) — the same closed ring
   (now via the library's own path, unforced), grown well outward, with a full
   annulus of secondary xylem (graded vessels + medullar rays) behind it.

Stage 2 is produced by monkeypatching ``DicotStemAnatomy._build_cambium`` for
the duration of that one ``generate_cells()`` call only (see
``_closed_ring_stem`` below) — no library source is modified, and the
patched-out original is restored immediately after. The bundle spec, count and
seed are identical throughout.

Run as a script to see them side by side.
"""

import sys
import os
import time

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(".."))

from openalea.granap.stem_class import StemAnatomy
from openalea.granap.stem_dicot_class import DicotStemAnatomy
from openalea.granap.input_data import OrganInputData

SEED = 0


def _dicot(secondary_growth):
    """Dicot stem preset with the secondary-growth flag set."""
    data = OrganInputData.for_dicot_stem()
    data.set_value("secondary_growth", "value", secondary_growth)
    return data


def _closed_ring_stem(data, seed):
    """Build a primary-growth stem, but with the interfascicular cambium
    closed — ``DicotStemAnatomy._build_cambium``'s own ``secondary=True``
    branch, forced on for this one build via monkeypatch, not by turning on
    ``secondary_growth`` (which would also grow the wood)."""
    original = DicotStemAnatomy._build_cambium

    def forced(self, contour, fascicular, conducting, cambium, secondary, sheath=None):
        return original(self, contour, fascicular, conducting, cambium, True, sheath=sheath)

    DicotStemAnatomy._build_cambium = forced
    try:
        stem = StemAnatomy(data, seed=seed)
        stem.generate_cells()
    finally:
        DicotStemAnatomy._build_cambium = original
    return stem


SCENARIOS = [
    ("primary growth", _dicot(False), False),
    ("interfascicular cambium", _dicot(False), True),
    ("secondary growth", _dicot(True), False),
]


def main(show=True):
    fig, axs = plt.subplots(1, 3, figsize=(21, 8))
    for ax, (label, data, force_close) in zip(axs.ravel(), SCENARIOS):
        print(f"\n=== {label} ===")
        t0 = time.time()
        if force_close:
            stem = _closed_ring_stem(data, SEED)
        else:
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

    plt.suptitle("Dicot stem growth — the same eustele as the cambium ring "
                 "closes, then grows wood", fontsize=15)
    plt.tight_layout()
    if show:
        plt.show()


if __name__ == "__main__":
    main()
