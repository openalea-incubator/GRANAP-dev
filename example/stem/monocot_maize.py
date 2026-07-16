"""Build a maize-like monocot stem (Zea mays) and plot it.

Maize (Zea mays) stem anatomy — an *atactostele*: collateral 'face' vascular
bundles scattered through a large parenchymatous pith, crowding toward the rind
(smaller + denser near the surface, larger + sparser toward the centre), each
wrapped in a sclerenchyma sheath.

This is modelled with three ``vascular_bundle`` bands (radial annuli, mm from the
stem centre), each its own kind:

* **core** (random)  — a few large bundles scattered through the inner pith;
* **mid**  (even)    — a ring of medium bundles;
* **rind** (even)    — a outer ring of small bundles just under the epidermis.

Configuration model: an organ is configured through its ``OrganInputData`` and
then **built once**.  All tuning is applied to ``data`` *before* building —
``data.set_value(...)`` for existing entries, ``data.params.append(...)`` for
extra ones (here, the three bundle bands replacing the preset's single spec).
"""

import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(".."))

from openalea.granap.stem_class import StemAnatomy
from openalea.granap.input_data import OrganInputData, VascularBundleParams

SEED = 0


# ---------------------------------------------------------------------------
# Bundle bands — most VascularBundleParams fields are spelled out here so the
# file doubles as a reference.  BASE holds the shared 'face' bundle recipe; each
# band overrides only its size, sheath, placement and radial position.
# ---------------------------------------------------------------------------

BASE = dict(
    # -- type: a monocot bundle is a closed collateral 'face' bundle ---------
    bundle_type="collateral", has_cambium=False,
    xylem_layout="face", phloem_outward=True,
    # -- metaxylem (the two big "eyes" at the radial middle) -----------------
    n_metaxylem=2,
    # -- ground parenchyma filling the bundle --------------------------------
    parenchyma_diameter=0.01, parenchyma_width=0.01,
)

BANDS = [
    # core — a few large bundles scattered at random through the inner pith.
    {**BASE, **dict(
        shape="ellipse",
        # -- metaxylem (the two big "eyes" at the radial middle) -----------------
        metaxylem_diameter=0.05, metaxylem_diameter_sd=0.004,
        metaxylem_diameter_min=0.03, metaxylem_gap=0.03,
        # -- protoxylem bundle + tear lacuna toward the centre -------------------
        n_protoxylem=1, protoxylem_diameter=0.014, protoxylem_diameter_min=0.008,
        protoxylem_width=0.035, protoxylem_height=0.035, protoxylem_relative_distance=0.7,
        lacuna=True, lacuna_width=0.028, lacuna_height=0.02,
        # -- phloem cluster (sieve elements + companion cells) toward the surface -
        prop_vessel=0.55, prop_sieve=0.5,
        phloem_width=0.05, phloem_height=0.04, phloem_relative_distance=0.55,
        companion_cell_diameter=0.005, companion_cell_width=0.005,
        # -- sclerenchyma sheath (fibre caps + ring) -----------------------------
        sheath="both", sheath_thickness=0.007,
        sclerenchyma_cell_diameter=0.006, sclerenchyma_cell_width=0.006,
        radius_min=0.0, radius_max=0.55, placement="random", n_bundles=5,
        width=0.15, height=0.24, n_metaxylem=2
    )},
    # mid — an evenly-spaced ring of medium bundles.
    {**BASE, **dict(
        shape="ellipse",
        # -- metaxylem (the two big "eyes" at the radial middle) -----------------
        n_metaxylem=2, metaxylem_diameter=0.05, metaxylem_diameter_sd=0.004,
        metaxylem_diameter_min=0.03, metaxylem_gap=0.03,
        # -- protoxylem bundle + tear lacuna toward the centre -------------------
        n_protoxylem=1, protoxylem_diameter=0.014, protoxylem_diameter_min=0.008,
        protoxylem_width=0.035, protoxylem_height=0.035, protoxylem_relative_distance=0.7,
        lacuna=True, lacuna_width=0.028, lacuna_height=0.02,
        # -- phloem cluster (sieve elements + companion cells) toward the surface -
        prop_vessel=0.55, prop_sieve=0.5,
        phloem_width=0.05, phloem_height=0.04, phloem_relative_distance=0.55,
        companion_cell_diameter=0.005, companion_cell_width=0.005,
        # -- sclerenchyma sheath (fibre caps + ring) -----------------------------
        sheath="both", sheath_thickness=0.007,
        sclerenchyma_cell_diameter=0.006, sclerenchyma_cell_width=0.006,
        radius_min=0.65, radius_max=0.95, placement="even", angle=0.0, n_bundles=12,
        width=0.11, height=0.17,
    )},
    # rind — a dense outer ring of small bundles just under the cortex; half-step
    # offset from the mid ring so they stagger against it.
    {**BASE, **dict(
        shape="ellipse",
        # -- metaxylem (the two big "eyes" at the radial middle) -----------------
        n_metaxylem=2, metaxylem_diameter=0.05, metaxylem_diameter_sd=0.004,
        metaxylem_diameter_min=0.03, metaxylem_gap=0.03,
        # -- protoxylem bundle + tear lacuna toward the centre -------------------
        n_protoxylem=1, protoxylem_diameter=0.014, protoxylem_diameter_min=0.008,
        protoxylem_width=0.035, protoxylem_height=0.035, protoxylem_relative_distance=0.7,
        lacuna=True, lacuna_width=0.028, lacuna_height=0.02,
        # -- phloem cluster (sieve elements + companion cells) toward the surface -
        prop_vessel=0.55, prop_sieve=0.5,
        phloem_width=0.05, phloem_height=0.04, phloem_relative_distance=0.55,
        companion_cell_diameter=0.005, companion_cell_width=0.005,
        # -- sclerenchyma sheath (fibre caps + ring) -----------------------------
        sheath="both", sheath_thickness=0.007,
        sclerenchyma_cell_diameter=0.006, sclerenchyma_cell_width=0.006,
        radius_min=1.0, radius_max=1.24, placement="even", angle=15.0, n_bundles=18,
        width=0.08, height=0.12,
        protoxylem_width=0.022, protoxylem_height=0.022, lacuna_width=0.018,
        phloem_width=0.035, phloem_height=0.028,
    )},
]


def build_maize() -> OrganInputData:
    """Assemble the maize-stem ``OrganInputData`` (monocot preset + 3 bundle bands)."""
    data = OrganInputData.for_monocot_stem() 

    # ── Pith: a large, soft parenchymatous ground tissue, cells growing bigger
    #    toward the centre ──────────────────────────────────────────────────
    data.set_value("pith", "thickness",            30)
    data.set_value("pith", "cell_diameter",        0.4)
    data.set_value("pith", "cell_diameter_center", 0.7)
    data.set_value("pith", "cavity_radius",        0.0)     # solid 

    # ── Cortex layers (inner / main / outer) ───────────────────────────────
    data.params.append({
        "name": "inner_cortex",
        "cell_diameter": 0.4,
        "cell_width": 0.5,
        "n_layers": 3,
        "shift": 0.5,
        "order": 3.5,
    })

    data.set_value("cortex", "cell_diameter", 0.28)
    data.set_value("cortex", "cell_width",    0.29)
    data.set_value("cortex", "n_layers",      3)

    data.params.append({
        "name": "outer_cortex",
        "cell_diameter": 0.15,
        "cell_width": 0.17,
        "n_layers": 1,
        "shift": 0.5,
        "order": 4.5,
    })

    data.params.append({
        "name": "outer_outer_cortex",
        "cell_diameter": 0.012,
        "cell_width": 0.015,
        "n_layers": 1,
        "shift": 0.5,
        "order": 4.8,
    })

    data.params.append({
        "name": "hypodermis",
        "cell_diameter": 0.024,
        "cell_width": 0.03,
        "n_layers": 2,
        "shift": 0.5,
        "order": 5,
    })

    data.set_values("epidermis", 
                    cell_diameter =  0.05,
                    cell_width = 0.06,
                    order = 6)


    # ── Vasculature: ────
    # drop the preset's single bundle spec
    data.params = [p for p in data.params if getattr(p, "name", None) != "vascular_bundle"]
    # add the 3 bands
    for band in BANDS:
        data.params.append(VascularBundleParams(**band))

    return data


def main(show=True):
    data = build_maize()

    print("=== maize monocot stem ===")
    t0 = time.time()
    stem = StemAnatomy(data, seed=SEED)
    stem.generate_cells()

    # Merge the inner/outer cortex tags into a single "cortex" tag.
    stem.retag_cells("inner_cortex", "cortex")
    stem.retag_cells("outer_cortex", "cortex")
    stem.retag_cells("outer_outer_cortex", "cortex")

    dt = time.time() - t0

    counts = {}
    for c in stem.all_cells.cells:
        counts[c.type] = counts.get(c.type, 0) + 1
    n_bundles = sum(b["n_bundles"] for b in BANDS)
    print(f"  Time: {dt:.2f}s   bundles requested: {n_bundles}   cells: {len(stem.all_cells.cells)}")
    for t in sorted(counts):
        print(f"    {t:16s} {counts[t]}")

    fig, ax = plt.subplots(figsize=(10, 10))
    stem.plot_cells(show=False, ax=ax, title="Maize monocot stem — atactostele")
    leg = ax.get_legend()
    if leg is not None:
        leg.set_title("tissue")
        for txt in leg.get_texts():
            txt.set_fontsize(7)
    plt.tight_layout()
    if show:
        plt.show()


if __name__ == "__main__":
    main()
