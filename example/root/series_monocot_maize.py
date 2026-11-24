"""Maize monocot root — a developmental *series* (apex -> collar) with tracked xylem.

Walks a maize (Zea mays, B73-like) root from the apex (0) to the collar, sampling
the anatomy at physical lengths.  Unlike the wheat series, **nothing fuses and
nothing terminates**: the metaxylem count is held constant along the whole root
(``META`` vessels, the B73 polyarch ring).  What changes with length is size —
the vessels, the stele and the cortex grow toward the collar — and the tissue
refits around the vessels each section.

Identity is still tracked: each metaxylem keeps its colour, so you can follow one
vessel from the apex up to the collar.  Protoxylem + phloem are regenerated
(untracked) per section.

Built with ``RootSeries`` (see ROOT_SERIES_PLAN); the tissue template is the B73
block from ``monocot_maize.py`` (without aerenchyma, so the stack renders cleanly).
"""

import os
import sys

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(".."))

from openalea.granap.input_data import OrganInputData
from openalea.granap.root_series import RootSeries

SEED = 0
N_LEVELS = 12                                 # physical samples apex .. collar
LENGTH_MM = 1500.0                            # apex (0) .. collar (150 cm)
N_COLS = 4                                    # grid layout: 4 per row -> 3 rows

META = 6                                      # metaxylem, constant apex .. collar
                                             # (the B73 polyarch ring — no fusion)


def build_maize_base() -> OrganInputData:
    """A maize 'B73' monocot-root template (the tissue that refits around the tracked
    vessels).  The series drives the metaxylem count directly; the phloem / protoxylem
    tuning here still applies — those are regenerated per section around them.

    Same numbers as the B73 panel of ``monocot_maize.py``, minus the aerenchyma:
    the series keeps a solid cortex so the semi-3D stack reads cleanly."""
    m = OrganInputData.for_root()                     # monocot preset (planttype=1)

    # Stele parenchyma (its thickness is overridden per level by the series).
    m.set_value("stele", "cell_diameter",        0.006)
    m.set_value("stele", "cell_diameter_center", 0.014)

    # Endodermis / pericycle (stele boundary).
    m.set_value("endodermis", "cell_diameter", 0.016)
    m.set_value("endodermis", "cell_width",    0.028)
    m.set_value("pericycle",  "cell_diameter", 0.0139)
    m.set_value("pericycle",  "cell_width",    0.0127)

    # Cortex layers (inner / main / outer).
    m.params.append({"name": "inner_cortex", "cell_diameter": 0.027, "cell_width": 0.026,
                     "n_layers": 1, "shift": 0.5, "order": 3.5})
    m.set_value("cortex", "cell_diameter", 0.039)
    m.set_value("cortex", "cell_width",    0.042)
    m.set_value("cortex", "n_layers",      3)
    m.params.append({"name": "outer_cortex", "cell_diameter": 0.037, "cell_width": 0.042,
                     "n_layers": 1, "shift": 0.5, "order": 4.5})

    # Exodermis / epidermis.
    m.set_value("exodermis", "cell_diameter", 0.029)
    m.set_value("exodermis", "cell_width",    0.027)
    m.set_value("epidermis", "cell_diameter", 0.018)
    m.set_value("epidermis", "cell_width",    0.031)

    # Phloem bundles (between the xylem poles).
    m.set_value("phloem", "sieve_diameter", 0.014)
    m.set_value("phloem", "cluster_width",  0.016)
    m.set_value("phloem", "cluster_height", 0.030)

    m.set_value("inter_cellular_spaces", "smoothness", 0.05)
    m.set_value("inter_cellular_spaces", "tissue", ["inner_cortex", "cortex", "outer_cortex"])
    return m


def n_fused(length_mm: float) -> int:
    """Number of metaxylem along the root: ``META`` everywhere.  Constant — this is
    the maize series' whole point: the vessels neither fuse nor terminate, they
    only grow toward the collar."""
    return META


def build_series() -> RootSeries:
    return RootSeries(
        build_maize_base(),
        start=0.0, end=LENGTH_MM, samples=N_LEVELS,   # sample 0 mm .. 1500 mm along the root
        n_fused=n_fused,                              # constant META
        terminations=[],                             # nothing stops
        # simple linear ramps as (value at start, value at end) = (0 mm, 1500 mm):
        vessel_radius=(0.026, 0.038),                # metaxylem widen toward the collar
        stele_radius=(0.13, 0.17),                   # stele widens along the root (mm);
                                                    # the apex floor keeps the six
                                                    # vessels from crowding

        area_retention=0.0,                          # no fusion, so this is unused
        migration_length=0.0,                        # no events to migrate away from
        fusion_length=0.0,
        param_schedules={"stele.cell_diameter_center": (0.012, 0.016)},  # parenchyma coarsens
        seed=SEED,
    )


def main(show=True):
    res = build_series().generate()
    fig = res.plot(
        cols=N_COLS,
        retag=[("inner_cortex", "cortex"), ("outer_cortex", "cortex")],
        suptitle="Maize root series (collar → apex) — colour follows one metaxylem identity",
        show=False,
    )
    # RootSeriesResult.plot titles sections in mm; this series is 150 cm long, so
    # relabel in cm to match how the deck presents every root series.
    order = sorted(res.sections, key=lambda s: s["length"], reverse=True)
    for ax, sec in zip(fig.axes, order):
        n = len(sec["vessels"])
        ax.set_title(f"{sec['length'] / 10:.0f} cm   ({n} metaxylem)", fontsize=10)
    if show:
        plt.show()
    return res


if __name__ == "__main__":
    main()
