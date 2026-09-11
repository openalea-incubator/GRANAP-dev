"""Build a wheat-like monocot root from the ``for_root`` preset and plot it.

Configuration model: an organ is configured through its ``OrganInputData`` and
then **built once** — construction snapshots the params and parses the vascular
geometry. So all tuning is applied to ``data`` *before* the organ is built, using
``data.set_value(...)`` for existing entries and ``data.params.append(...)`` for
new tissue layers.

Wheat (Triticum aestivum) root anatomy
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(".."))

from openalea.granap.root_class import RootAnatomy
from openalea.granap.input_data import OrganInputData

SEED = 0


def anatomy_metrics(root: RootAnatomy) -> dict:
    """Geometry summary of a generated root: root/stele diameters and total
    xylem (metaxylem + protoxylem) cross-sectional area.

    Diameters are area-equivalent (``2 * sqrt(area / pi)``), robust to
    non-circular outlines.

    Stele note: ``generate_layer_polygons`` stores each ring at its cell
    *centre-line*, and the stele is a stack of nested rings. So neither the first
    "stele" ring (inset by ``stele_cell/2``) nor the enclosing layer's centre-line
    (out by ``pericycle_cell/2``) is the true boundary. The stele<->pericycle
    interface is the enclosing layer's *inner* edge = its centre-line radius minus
    half its cell diameter, which recovers the nominal ``stele.thickness``.
    """
    root_area = root.generate_base_shape().area

    polys = root.generate_layer_polygons()
    i0 = next((i for i, p in enumerate(polys) if p["name"] == "stele"), None)
    if i0 is not None and i0 > 0:
        enclosing = polys[i0 - 1]
        stele_radius = np.sqrt(enclosing["polygon"].area / np.pi) - enclosing["cell_diameter"] / 2
    else:
        stele_radius = float("nan")

    xylem_area = sum(c.area for c in root.all_cells.cells if "xylem" in c.type)
    return {
        "root_diameter":  np.sqrt(root_area / np.pi) * 2,
        "stele_diameter": stele_radius * 2,
        "xylem_area":   xylem_area,
    }



def main(show=True):
    wheat = OrganInputData.for_root()  # monocot preset (planttype=1)

    # -- Stele size + radial parenchyma (pith) gradient ---------------------
    wheat.set_value("stele", "thickness",            0.283)
    wheat.set_value("stele", "cell_diameter",        0.013)
    wheat.set_value("stele", "cell_diameter_center", 0.018)

    # -- Vasculature: ring of metaxylem vessels + protoxylem (no star) ------
    wheat.set_value("xylem", "xylem_shape",              "default")
    wheat.set_value("xylem", "n_vascular_bundles",       3)      # polyarch metaxylem ring
    wheat.set_value("xylem", "vessel_diameter",          0.0945)
    wheat.set_value("xylem", "vessel_diameter_sd",       0.0065)
    wheat.set_value("xylem", "ratio_proto_meta",         3.3)
    wheat.set_value("xylem", "protoxylem_diameter",      0.022)
    wheat.set_value("xylem", "protoxylem_cluster_width", 0.023)
    wheat.set_value("xylem", "protoxylem_cluster_height", 0.025)

    # -- Phloem bundles (between the xylem poles) ---------------------------
    wheat.set_value("phloem", "sieve_diameter", 0.014)
    wheat.set_value("phloem", "cluster_width",  0.014)
    wheat.set_value("phloem", "cluster_height", 0.0249)

    # -- Endodermis / pericycle (stele boundary) ----------------------------
    wheat.set_value("endodermis", "cell_diameter", 0.0162)
    wheat.set_value("endodermis", "cell_width",    0.0281)
    wheat.set_value("pericycle", "cell_diameter",  0.0183)
    wheat.set_value("pericycle", "cell_width",     0.013)

    # -- Cortex layers (inner / main / outer) -------------------------------
    wheat.params.append({
        "name": "inner_cortex",
        "cell_diameter": 0.018,
        "cell_width": 0.036,
        "n_layers": 2,
        "shift": 0.5,
        "order": 3.5,
    })

    wheat.set_value("cortex", "cell_diameter", 0.049)
    wheat.set_value("cortex", "cell_width",    0.056)
    wheat.set_value("cortex", "n_layers",      1)

    wheat.params.append({
        "name": "outer_cortex",
        "cell_diameter": 0.053,
        "cell_width": 0.071,
        "n_layers": 2,
        "shift": 0.5,
        "order": 4.5,
    })

    # -- Exodermis / epidermis ----------------------------------------------
    wheat.set_value("exodermis", "cell_diameter", 0.028)
    wheat.set_value("exodermis", "cell_width",    0.033)
    wheat.set_value("epidermis", "cell_diameter", 0.017)
    wheat.set_value("epidermis", "cell_width",    0.034)

    # -- Intercellular spaces across the cortex tissues ---------------------
    wheat.set_value("inter_cellular_spaces", "smoothness", 0.05)
    wheat.set_value("inter_cellular_spaces", "tissue",
                   ["inner_cortex", "cortex", "outer_cortex"])

    # Build once, after all configuration is in place.
    wheat_anatomy = RootAnatomy(wheat, seed=SEED)
    wheat_anatomy.generate_cells()

    # Merge the inner/outer cortex tags into a single "cortex" tag.
    wheat_anatomy.retag_cells("inner_cortex", "cortex")
    wheat_anatomy.retag_cells("outer_cortex", "cortex")

    # Other anatomy of wheat
    wheat_watde = OrganInputData.for_root()  # monocot preset (planttype=1)

    # -- Stele size + radial parenchyma (pith) gradient ---------------------
    wheat_watde.set_value("stele", "thickness",            0.35)
    wheat_watde.set_value("stele", "cell_diameter",        0.010)
    wheat_watde.set_value("stele", "cell_diameter_center", 0.016)

    # -- Vasculature: ring of metaxylem vessels + protoxylem (no star) ------
    wheat_watde.set_value("xylem", "xylem_shape",              "default")
    wheat_watde.set_value("xylem", "n_vascular_bundles",       7)      # polyarch metaxylem ring
    wheat_watde.set_value("xylem", "vessel_diameter",          0.0675)
    wheat_watde.set_value("xylem", "vessel_diameter_sd",       0.00625)
    wheat_watde.set_value("xylem", "ratio_proto_meta",         2)
    wheat_watde.set_value("xylem", "protoxylem_diameter",      0.020)
    wheat_watde.set_value("xylem", "protoxylem_cluster_width", 0.021)
    wheat_watde.set_value("xylem", "protoxylem_cluster_height", 0.021)

    # -- Phloem bundles (between the xylem poles) ---------------------------
    wheat_watde.set_value("phloem", "sieve_diameter", 0.013)
    wheat_watde.set_value("phloem", "cluster_width",  0.025)
    wheat_watde.set_value("phloem", "cluster_height", 0.020)

    # -- Endodermis / pericycle (stele boundary) ----------------------------
    wheat_watde.set_value("endodermis", "cell_diameter", 0.0175)
    wheat_watde.set_value("endodermis", "cell_width",    0.027)
    wheat_watde.set_value("pericycle", "cell_diameter",  0.022)
    wheat_watde.set_value("pericycle", "cell_width",     0.013)

    # -- Cortex layers (inner / main / outer) -------------------------------
    wheat_watde.params.append({
        "name": "inner_cortex",
        "cell_diameter": 0.018,
        "cell_width": 0.024,
        "n_layers": 3,
        "shift": 0.5,
        "order": 3.5,
    })

    wheat_watde.set_value("cortex", "cell_diameter", 0.033)
    wheat_watde.set_value("cortex", "cell_width",    0.038)
    wheat_watde.set_value("cortex", "n_layers",      2)

    wheat_watde.params.append({
        "name": "outer_cortex",
        "cell_diameter": 0.027,
        "cell_width": 0.0269,
        "n_layers": 2,
        "shift": 0.5,
        "order": 4.5,
    })

    # -- Exodermis / epidermis ----------------------------------------------
    wheat_watde.set_value("exodermis", "cell_diameter", 0.029)
    wheat_watde.set_value("exodermis", "cell_width",    0.031)
    wheat_watde.set_value("epidermis", "cell_diameter", 0.029)
    wheat_watde.set_value("epidermis", "cell_width",    0.021)

    # -- Intercellular spaces across the cortex tissues ---------------------
    wheat_watde.set_value("inter_cellular_spaces", "smoothness", 0.05)
    wheat_watde.set_value("inter_cellular_spaces", "tissue",
                   ["inner_cortex", "cortex", "outer_cortex"])

    # Build once, after all configuration is in place.
    wheat_watde_anatomy = RootAnatomy(wheat_watde, seed=SEED)
    wheat_watde_anatomy.generate_cells()

    # Merge the inner/outer cortex tags into a single "cortex" tag.
    wheat_watde_anatomy.retag_cells("inner_cortex", "cortex")
    wheat_watde_anatomy.retag_cells("outer_cortex", "cortex")

    wheat_roots = []
    wheat_roots.append(wheat_anatomy)
    wheat_roots.append(wheat_watde_anatomy)

    n_cols = 2
    n_rows = 1
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axs_flat = axs.flatten()

    labels = ["Wheat Salmone", "Wheat WATDE0230"]
    for i, root in enumerate(wheat_roots):
        m = anatomy_metrics(root)
        title = (
            f"{labels[i]}\n"
            f"root d = {m['root_diameter']:.3f}   "
            f"stele d = {m['stele_diameter']:.3f}   "
            f"xylem area = {m['xylem_area']:.4f}"
        )
        root.plot_cells(show=False, ax=axs_flat[i], title=title)
        legend = axs_flat[i].get_legend()
        if legend:
            legend.remove()

    for j in range(2, len(axs_flat)):
        axs_flat[j].set_visible(False)

    plt.suptitle("wheat anatomy - Cross section", fontsize=14)
    plt.tight_layout()
    if show:
        plt.show()


if __name__ == "__main__":
    main()
