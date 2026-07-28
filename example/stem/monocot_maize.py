"""Build a maize-like monocot stem (*Zea mays*) and plot it.

Maize stem anatomy — an *atactostele*: collateral 'face' bundles scattered through
a large parenchymatous pith, wrapped in a graded rind.  Here the vasculature is
**three radial bundle bands** (``vascular_bundle`` specs): the inner ``spaced`` band
fills an annulus (``radius_min .. radius_max``), the two ``even`` rings sit at a
single ``radius`` mm from the stem centre, pith -> rind:

* **inner** (``placement="spaced"``) — larger bundles, *taller than wide*, spread
  through the inner pith; each with protoxylem + a tear lacuna;
* **mid** (``placement="even"``) — the biggest bundles, *wider than tall*, with
  protoxylem + lacuna, half-step offset on the peripheral ring so one sits *between*
  each rind bundle;
* **rind** (``placement="even"``) — small bundles, *wider than tall*, on the same
  peripheral ring **just under the hypodermis** (a ``radius`` that reaches past the
  pith into the rind), with **no protoxylem and no lacuna**.

A bundle band may be placed in *any* tissue: an ``even`` ring whose ``radius`` (or a
``random``/``spaced`` band whose ``radius_max``) exceeds the pith radius carries its
bundles out into the cortex / rind (clamped only to the epidermis), so the small
peripheral bundles sit embedded in the rind as they do in a real maize stem.

The rind itself is a graded stack of cortex layers (inner -> outer-outer cortex),
a hypodermis and the epidermis, with a sclerenchyma fibre ring.

Configuration model: an organ is configured through its ``OrganInputData`` and then
built once — all tuning is applied to ``data`` before building.
"""

import os
import sys
import time

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(".."))

from openalea.granap.stem_class import StemAnatomy
from openalea.granap.input_data import OrganInputData, VascularBundleParams

SEED = 0


# ---------------------------------------------------------------------------
# Bundle bands.  BASE is the shared 'face' (monocot, closed collateral) recipe;
# each band overrides only its size, xylem detail, placement and radial position.
# Sizes are in mm; the pith radius is ~1.4 mm.
# ---------------------------------------------------------------------------

BASE = dict(
    bundle_type="collateral", has_cambium=False,     # monocot: closed, no cambium
    xylem_layout="face", phloem_outward=True, shape="ellipse",
    n_metaxylem=2,               # the two big "eyes"
    prop_vessel=0.55, prop_sieve=0.5,
    companion_cell_diameter=0.005, companion_cell_width=0.005,
    parenchyma_diameter=0.01, parenchyma_width=0.01,
    sheath="both", sheath_thickness=0.006,           # sclerenchyma fibre caps + ring
    sclerenchyma_cell_diameter=0.006, sclerenchyma_cell_width=0.006,
)

BANDS = [
    # inner — larger bundles, TALLER than wide, spread through the inner pith
    # (best-candidate 'spaced' placement); protoxylem + tear lacuna.
    {**BASE, **dict(
        radius_min=0.0, radius_max=1.4, placement="spaced", n_bundles=20,
        width=0.13, height=0.20, metaxylem_gap=0.015,
        metaxylem_diameter=0.045, metaxylem_diameter_sd=0.004, metaxylem_diameter_min=0.03,
        n_protoxylem=1, protoxylem_diameter=0.03, protoxylem_diameter_min=0.025,
        protoxylem_width=0.03, protoxylem_height=0.03, protoxylem_relative_distance=0.3,
        lacuna=True, lacuna_width=0.026, lacuna_height=0.018,
        phloem_width=0.1, phloem_height=0.06, phloem_relative_distance=0.5,
    )},
    # mid — the biggest bundles, WIDER than tall, sharing the peripheral ring
    # (single ``radius``) with the rind bundles but half-step offset
    # (angle = 180 / n_bundles) so one sits *between* each rind bundle, just under
    # the hypodermis; protoxylem + lacuna.  ``radius`` reaches past the pith into the
    # rind (a band may be placed in any tissue).
    {**BASE, **dict(
        radius=1.68, placement="even", angle=180.0 / 20, n_bundles=20,
        width=0.20, height=0.13, metaxylem_gap=0.05,
        metaxylem_diameter=0.05, metaxylem_diameter_sd=0.004, metaxylem_diameter_min=0.03,
        n_protoxylem=1, protoxylem_diameter=0.03, protoxylem_diameter_min=0.025,
        protoxylem_width=0.03, protoxylem_height=0.03, protoxylem_relative_distance=0.3,
        lacuna=True, lacuna_width=0.026, lacuna_height=0.018,
        phloem_width=0.1, phloem_height=0.06, phloem_relative_distance=0.5,
    )},
    # rind — small bundles, WIDER than tall, on the same peripheral ring **just under
    # the hypodermis** (embedded in the cortex/rind, not the pith); NO protoxylem and
    # NO lacuna (just the two metaxylem + a phloem cluster).  Tall enough that the
    # phloem sits clear of the metaxylem eyes.
    {**BASE, **dict(
        radius=1.78, placement="even", angle=0.0, n_bundles=20,
        width=0.11, height=0.09, 
        metaxylem_diameter=0.03, metaxylem_diameter_sd=0.002, metaxylem_diameter_min=0.013,
        metaxylem_gap=0.015, 
        n_protoxylem=0, lacuna=False,
        phloem_width=0.032, phloem_height=0.022, phloem_relative_distance=0.3,
    )},
]


def build_maize() -> OrganInputData:
    """Assemble the maize-stem ``OrganInputData`` (monocot preset + 3 bundle bands)."""
    data = OrganInputData.for_monocot_stem()

    # -- Pith: a soft parenchymatous ground tissue (radius ~1.4 mm), cells growing
    #    a little bigger toward the centre; solid (no medullary cavity) -------
    data.set_value("pith", "thickness",            2.8)
    data.set_value("pith", "cell_diameter",        0.05)
    data.set_value("pith", "cell_diameter_center", 0.09)
    data.set_value("pith", "cavity_radius",        0.0)

    # -- Rind: a graded stack of cortex layers (inner -> outer), a hypodermis and
    #    the epidermis; cells shrink toward the surface --------------------------
    data.params.append({
        "name": "inner_cortex",
        "cell_diameter": 0.06, "cell_width": 0.065, "n_layers": 3, "shift": 0.5, "order": 3.5,
    })
    data.set_value("cortex", "cell_diameter", 0.05)
    data.set_value("cortex", "cell_width",    0.05)
    data.set_value("cortex", "n_layers",      3)
    data.params.append({
        "name": "outer_cortex",
        "cell_diameter": 0.035, "cell_width": 0.04, "n_layers": 1, "shift": 0.5, "order": 4.5,
    })
    data.params.append({
        "name": "outer_outer_cortex",
        "cell_diameter": 0.022, "cell_width": 0.025, "n_layers": 1, "shift": 0.5, "order": 4.8,
    })
    data.params.append({
        "name": "hypodermis",
        "cell_diameter": 0.024, "cell_width": 0.028, "n_layers": 2, "shift": 0.5, "order": 5,
    })
    # A thin sclerenchyma fibre ring under the epidermis (maize's tough rind).
    data.set_values("sclerenchyma", cell_diameter=0.018, cell_width=0.018, n_layers=2)
    data.set_values("epidermis", cell_diameter=0.025, cell_width=0.03, order=6)
    data.set_value("inter_cellular_spaces", "smoothness", 0.05)
    data.set_value("inter_cellular_spaces", "tissue",
                         ["inner_cortex", "cortex", "outer_cortex", "outer_outer_cortex"])
    
    # -- Vasculature: drop the preset's single bundle spec, add the three bands ---
    data.params = [p for p in data.params if getattr(p, "name", None) != "vascular_bundle"]
    for band in BANDS:
        data.params.append(VascularBundleParams(**band))

    return data


def main(show=True):
    data = build_maize()
    print("=== maize monocot stem ===")
    t0 = time.time()
    stem = StemAnatomy(data, seed=SEED)
    stem.generate_cells()

    # Merge the graded cortex tags into a single "cortex" tag for the legend.
    for extra in ("inner_cortex", "outer_cortex", "outer_outer_cortex"):
        stem.retag_cells(extra, "cortex")

    dt = time.time() - t0
    counts = {}
    for c in stem.all_cells.cells:
        counts[c.type] = counts.get(c.type, 0) + 1
    n_req = sum(b["n_bundles"] for b in BANDS)
    n_got = len(stem.vascular_tissue_polygons.get("bundle", []))
    print(f"  Time: {dt:.2f}s   bundles requested: {n_req}   placed: {n_got}   "
          f"cells: {len(stem.all_cells.cells)}")
    for t in sorted(counts):
        print(f"    {t:16s} {counts[t]}")

    fig, ax = plt.subplots(figsize=(10, 10))
    stem.plot_cells(show=False, ax=ax, title="Maize monocot stem — atactostele (3 bundle bands)")
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
