"""Wheat monocot root — a developmental *series* (apex -> collet) with tracked xylem.

Walks up a wheat (Triticum aestivum) root, sampling the anatomy at physical lengths.
Biologically the metaxylem **fuse** going collet -> apex, so there are many small
vessels at the collet and few big ones at the apex.  Here: 3 (fused) central metaxylem
for the lower ~60 mm, then rising toward the collet as it splits into the polyarch ring.
Identity is a fusion group: each xylem id keeps its colour, so you can follow a vessel
(and see which others it fuses with) up the root.

Built with ``RootSeries`` (collet-anchored fusion model; see ROOT_SERIES_PLAN).  Phase 2:
xylem only — no phloem yet.  Termination (a vessel that just stops) is the next addition.
"""

import os
import sys

import numpy as np

sys.path.append(os.path.abspath(".."))

from openalea.granap.input_data import OrganInputData
from openalea.granap.root_series import RootSeries

SEED = 0
N_LEVELS = 12                                 # physical samples apex .. collet
LENGTH_MM = 150.0                             # apex (0) .. collet (LENGTH_MM)
N_COLS = 4                                    # grid layout: 4 per row -> 3 rows


def build_wheat_base() -> OrganInputData:
    """A wheat 'Salmone' monocot-root template (the tissue that refits around the
    tracked vessels).  The xylem/phloem tuning is left in for fidelity but is not used
    under vessel prescription — the series drives the xylem directly."""
    w = OrganInputData.for_root()                     # monocot preset (planttype=1)

    # Stele parenchyma (its thickness is overridden per level by the series).
    w.set_value("stele", "cell_diameter",        0.013)
    w.set_value("stele", "cell_diameter_center", 0.018)

    # Endodermis / pericycle (stele boundary).
    w.set_value("endodermis", "cell_diameter", 0.0162)
    w.set_value("endodermis", "cell_width",    0.0281)
    w.set_value("pericycle", "cell_diameter",  0.0183)
    w.set_value("pericycle", "cell_width",     0.013)

    # Cortex layers (inner / main / outer).
    w.params.append({"name": "inner_cortex", "cell_diameter": 0.018, "cell_width": 0.036,
                     "n_layers": 2, "shift": 0.5, "order": 3.5})
    w.set_value("cortex", "cell_diameter", 0.049)
    w.set_value("cortex", "cell_width",    0.056)
    w.set_value("cortex", "n_layers",      1)
    w.params.append({"name": "outer_cortex", "cell_diameter": 0.053, "cell_width": 0.071,
                     "n_layers": 2, "shift": 0.5, "order": 4.5})

    # Exodermis / epidermis.
    w.set_value("exodermis", "cell_diameter", 0.028)
    w.set_value("exodermis", "cell_width",    0.033)
    w.set_value("epidermis", "cell_diameter", 0.017)
    w.set_value("epidermis", "cell_width",    0.034)

    w.set_value("inter_cellular_spaces", "smoothness", 0.05)
    w.set_value("inter_cellular_spaces", "tissue", ["inner_cortex", "cortex", "outer_cortex"])
    return w


N_TERMINATE  = 2        # how many metaxylem terminate (do NOT fuse — they just stop)
TERMINATE_AT = 90.0     # ...and the length (mm) at which they stop (present above, gone below)
APEX_META = 3           # metaxylem at the apex (the fusion bottoms out here, not at 1)
COLLET_FUSED = 4        # fused metaxylem at the collet (+ the terminators added on top)


def n_fused(length_mm: float) -> int:
    """Number of FUSED metaxylem along the root: ``APEX_META`` at the apex (the fusion
    stops there), rising to ``COLLET_FUSED`` toward the collet.  This schedule is *how you
    say* how many metaxylem a given height has — set its apex value to change it."""
    frac = max(0.0, (length_mm - 60.0) / (LENGTH_MM - 60.0))
    return int(np.clip(round(APEX_META + (COLLET_FUSED - APEX_META) * frac), APEX_META, COLLET_FUSED))


def build_series() -> RootSeries:
    return RootSeries(
        build_wheat_base(),
        start=0.0, end=LENGTH_MM, samples=N_LEVELS,   # sample 0 mm .. 150 mm along the root
        n_fused=n_fused,                              # callable: 3 up to 60 mm, then -> 4
        # one stop-length per terminating vessel -> N_TERMINATE of them, each stopping at
        # TERMINATE_AT (present for length >= 90 mm, gone below).
        terminations=[TERMINATE_AT] * N_TERMINATE,
        # simple linear ramps as (value at start, value at end) = (0 mm, 150 mm) here:
        vessel_radius=(0.035, 0.020),                # single-vessel radius (mm); fused bigger
        stele_radius=(0.09, 0.18),                   # stele widens along the root (mm)
        area_retention=0.4,                          # fused = "slightly bigger"
        migration_length=40.0,                       # migrate toward class positions over ~40 mm
        param_schedules={"stele.cell_diameter_center": (0.014, 0.020)},  # parenchyma coarsens
        seed=SEED,
    )


def main(show=True):
    res = build_series().generate()
    # generic rendering lives in the library now; the example only supplies the
    # wheat-specific cortex retag + a title.
    res.plot(cols=N_COLS, retag=[("inner_cortex", "cortex"), ("outer_cortex", "cortex")],
             suptitle="Wheat root fusion series (collet → apex) — vessel label = primordial ids it contains",
             show=show)
    return res


if __name__ == "__main__":
    main()
