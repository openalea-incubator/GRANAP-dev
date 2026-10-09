"""Dicot root — a primary-growth *series* (apex -> collet) with tracked xylem.

Dicot primary xylem vessels stay put (no fusion/migration); what changes along the root is
the **pith** — a central front that recedes from apex to collet.  A vessel appears once the
pith has receded past it (outer protoxylem first, inner metaxylem last) and grows to its 5PL
target.  Positions + targets are captured once from the collet (smallest pith); the phloem,
cambium and cortex regenerate around the tracked vessels in each section.

Built with ``DicotRootSeries`` (see ROOT_SERIES_PLAN).
"""

import os
import sys

sys.path.append(os.path.abspath(".."))

from openalea.granap.input_data import OrganInputData
from openalea.granap.root_series import DicotRootSeries

SEED = 0
N_LEVELS = 8                                 # physical samples apex .. collet
LENGTH_MM = 120.0                            # apex (0) .. collet (LENGTH_MM)
N_COLS = 4                                   # grid layout


def build_dicot_base() -> OrganInputData:
    """A dicot-root template (the tissue that refits around the tracked vessels).  Primary
    growth only — secondary growth stays off."""
    d = OrganInputData.for_dicot_root()          # dicot preset (planttype=2)
    d.set_value("xylem", "n_vascular_peak", 4)   # tetrarch star
    d.set_value("xylem", "radius_peak_side", 0.24)
    d.set_value("xylem", "radius_valley_side", 0.05)
    d.set_value("stele", "cell_diameter", 0.02)
    return d


def build_series() -> DicotRootSeries:
    return DicotRootSeries(
        build_dicot_base(),
        start=0.0, end=LENGTH_MM, samples=N_LEVELS,   # sample 0 mm .. 120 mm along the root
        # simple linear ramps as (value at start, value at end):
        stele_radius=(0.30, 0.30),      # stele ~constant in primary growth (fixed diameter)
        pith_radius=(0.26, 0.0),        # pith recedes: fills the apex, gone at the collet
        seed=SEED,
    )


def main(show=True):
    res = build_series().generate()
    res.plot(cols=N_COLS,
             suptitle="Dicot root primary-growth series (apex -> collet): pith recedes, "
                      "protoxylem then metaxylem differentiate",
             show=show)
    return res


if __name__ == "__main__":
    main()
