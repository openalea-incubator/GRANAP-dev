"""Gallery: one monocot root, four metaxylem diameter distributions.

The same root (same seed, same mean and SD of the metaxylem diameter) is built
four times; only ``xylem.vessel_diameter_distribution`` changes:

  normal     the default (None gives exactly the same root)
  lognormal  right-skewed: many vessels a bit below the mean, a few large ones
  uniform    flat between mean +/- SD*sqrt(3)
  empirical  resampled from measured diameters - here a made-up bimodal sample,
             so the root gets a small and a large vessel class

Every family is moment-matched to the same ``vessel_diameter`` (mean) and
``vessel_diameter_sd``, so the four roots differ in the *shape* of the vessel
size spread, not in its mean or width.  Top row: the sections.  Bottom row: the
distribution each vessel is drawn from (histogram of many draws) with the
realised metaxylem diameters of the root above as ticks.
"""

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(".."))

from openalea.granap.distribution import draw
from openalea.granap.input_data import OrganInputData
from openalea.granap.root_class import RootAnatomy

SEED = 0

# A wide stele with many metaxylem and a large spread, so the shape shows.
STELE = {"thickness": 0.6, "cell_diameter": 0.009, "cell_diameter_center": 0.022}
XYLEM = {"n_vascular_bundles": 12, "n_protoxylem": 12,
         "vessel_diameter": 0.05, "vessel_diameter_sd": 0.015, "vessel_diameter_min": 0.01}

# Hypothetical measured diameters (mm): two vessel classes.  Only their shape is
# used; the mean and SD still come from XYLEM.
MEASURED = [0.030, 0.032, 0.034, 0.035, 0.037, 0.064, 0.067, 0.069, 0.071, 0.074]

SCENARIOS = [
    ("Normal (default)", {"family": "normal"}),
    ("Log-normal", {"family": "lognormal"}),
    ("Uniform", {"family": "uniform"}),
    ("Empirical (bimodal sample)", {"family": "empirical", "values": MEASURED}),
]


def make_root(distribution) -> RootAnatomy:
    """The shared monocot root with the given metaxylem diameter distribution."""
    data = OrganInputData.for_root()
    for field, value in STELE.items():
        data.set_value("stele", field, value)
    for field, value in XYLEM.items():
        data.set_value("xylem", field, value)
    data.set_value("xylem", "vessel_diameter_distribution", distribution)
    root = RootAnatomy(data, seed=SEED)
    root.generate_cells()
    return root


def metaxylem_diameters(root: RootAnatomy) -> np.ndarray:
    """Equivalent-circle diameter of each metaxylem vessel (mm)."""
    gdf = root.generate_cells()
    areas = np.array([g.area for t, g in zip(gdf["type"], gdf.geometry) if t == "metaxylem"])
    return np.sqrt(4.0 * areas / np.pi)


def main(show=True):
    mean, sd, lo = XYLEM["vessel_diameter"], XYLEM["vessel_diameter_sd"], XYLEM["vessel_diameter_min"]
    fig, axes = plt.subplots(2, len(SCENARIOS), figsize=(5 * len(SCENARIOS), 9.5),
                             gridspec_kw={"height_ratios": [3, 1.3]})
    bins = np.linspace(0.0, mean + 4.5 * sd, 60)
    for col, (label, dist) in enumerate(SCENARIOS):
        t0 = time.time()
        root = make_root(dist)
        d = metaxylem_diameters(root)
        print(f"{label:28s} {time.time() - t0:5.1f}s  metaxylem diameters (mm): "
              f"mean {d.mean():.4f}  sd {d.std():.4f}  n {d.size}")

        ax = axes[0, col]
        root.plot_cells(ax=ax, show=False, title=label)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        ax.set_axis_off()

        # the distribution the vessels are drawn from: many draws, same mean / SD / floor
        rng = np.random.default_rng(1)
        samples = [draw(rng, mean, sd, lo, None, dist) for _ in range(20_000)]
        ax = axes[1, col]
        ax.hist(samples, bins=bins, density=True, color="#9fbfd9", edgecolor="white")
        ax.plot(d, np.zeros_like(d), "|", color="#1c4f7c", markersize=22, mew=2,
                label="this root's metaxylem")
        ax.axvline(mean, color="#555555", lw=1, ls="--")
        ax.set_xlabel("metaxylem diameter (mm)")
        ax.set_yticks([])
        if col == 0:
            ax.legend(loc="upper right", fontsize=8, frameon=False)

    fig.suptitle(f"Same monocot root, same mean ({mean} mm) and SD ({sd} mm) of the metaxylem "
                 f"diameter — four distribution shapes", fontsize=13)
    plt.tight_layout()
    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    main()
