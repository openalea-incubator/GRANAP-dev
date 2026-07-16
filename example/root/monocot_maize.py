"""Build a maize-like monocot root from the ``for_root`` preset and plot it.

Configuration model: an organ is configured through its ``OrganInputData`` and
then **built once** — construction snapshots the params and parses the vascular
geometry. So all tuning is applied to ``data`` *before* the organ is built, using
``data.set_value(...)`` for existing entries and ``data.params.append(...)`` for
new tissue layers.

Maize (Zea mays) root anatomy
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

    # Realized aerenchyma proportion, measured from the *generated* geometry
    # (i.e. after Voronoi + intercellular carving + aerenchyma merge). This is
    # the air-space fraction of the cortex band (air / (air + cortex)); compare
    # it against the requested ``aerenchyma_proportion`` input.
    air_area = sum(
        c.area for c in root.all_cells.cells
        if c.type in ("air space", "aerenchyma")
    )
    cortex_area = sum(c.area for c in root.all_cells.cells if c.type == "cortex")
    denom = air_area + cortex_area
    aerenchyma_proportion = air_area / denom if denom > 0 else 0.0

    return {
        "root_diameter":  np.sqrt(root_area / np.pi) * 2,
        "stele_diameter": stele_radius * 2,
        "xylem_area":   xylem_area,
        "aerenchyma_proportion": aerenchyma_proportion,
    }


def main(show=True):
    maize = OrganInputData.for_root()  # monocot preset (planttype=1)

    # ── Stele size + radial parenchyma (pith) gradient ─────────────────────
    maize.set_value("stele", "thickness",            0.245)
    maize.set_value("stele", "cell_diameter",        0.006)
    maize.set_value("stele", "cell_diameter_center", 0.014)

    # ── Vasculature: ring of metaxylem vessels + protoxylem (no star) ──────
    maize.set_value("xylem", "xylem_shape",              "default")
    maize.set_value("xylem", "n_vascular_bundles",       4)      # polyarch metaxylem ring
    maize.set_value("xylem", "vessel_diameter",          0.056)
    maize.set_value("xylem", "vessel_diameter_sd",       0.006)
    maize.set_value("xylem", "ratio_proto_meta",         2.5)
    maize.set_value("xylem", "protoxylem_diameter",      0.016)
    maize.set_value("xylem", "protoxylem_cluster_width", 0.021)
    maize.set_value("xylem", "protoxylem_cluster_height", 0.021)

    # ── Phloem bundles (between the xylem poles) ───────────────────────────
    maize.set_value("phloem", "sieve_diameter", 0.010)
    maize.set_value("phloem", "cluster_width",  0.012)
    maize.set_value("phloem", "cluster_height", 0.012)

    # ── Endodermis / pericycle (stele boundary) ────────────────────────────
    maize.set_value("endodermis", "cell_diameter", 0.0096)
    maize.set_value("endodermis", "cell_width",    0.016)
    maize.set_value("pericycle", "cell_diameter",  0.0127)
    maize.set_value("pericycle", "cell_width",     0.0087)

    # ── Cortex layers (inner / main / outer) ───────────────────────────────
    maize.params.append({
        "name": "inner_cortex",
        "cell_diameter": 0.022,
        "cell_width": 0.026,
        "n_layers": 1,
        "shift": 0.5,
        "order": 3.5,
    })

    maize.set_value("cortex", "cell_diameter", 0.047)
    maize.set_value("cortex", "cell_width",    0.046)
    maize.set_value("cortex", "n_layers",      2)

    maize.params.append({
        "name": "outer_cortex",
        "cell_diameter": 0.034,
        "cell_width": 0.037,
        "n_layers": 1,
        "shift": 0.5,
        "order": 4.5,
    })

    # ── Exodermis / epidermis ──────────────────────────────────────────────
    maize.set_value("exodermis", "cell_diameter", 0.025)
    maize.set_value("exodermis", "cell_width",    0.036)
    maize.set_value("epidermis", "cell_diameter", 0.024)
    maize.set_value("epidermis", "cell_width",    0.028)

    # ── Intercellular spaces across the cortex tissues ─────────────────────
    maize.set_value("inter_cellular_spaces", "smoothness", 0.05)
    maize.set_value("inter_cellular_spaces", "tissue",
                   ["inner_cortex", "cortex", "outer_cortex"])

    # Build once, after all configuration is in place.
    maize_anatomy = RootAnatomy(maize, seed=SEED)
    maize_anatomy.generate_cells()

    # Merge the inner/outer cortex tags into a single "cortex" tag.
    maize_anatomy.retag_cells("inner_cortex", "cortex")
    maize_anatomy.retag_cells("outer_cortex", "cortex")

    # Other anatomy of maize
    maize_b73 = OrganInputData.for_root()  # monocot preset (planttype=1)

    # ── Stele size + radial parenchyma (pith) gradient ─────────────────────
    maize_b73.set_value("stele", "thickness",            0.293)
    maize_b73.set_value("stele", "cell_diameter",        0.006)
    maize_b73.set_value("stele", "cell_diameter_center", 0.014)

    # ── Vasculature: ring of metaxylem vessels + protoxylem (no star) ──────
    maize_b73.set_value("xylem", "xylem_shape",              "default")
    maize_b73.set_value("xylem", "n_vascular_bundles",       6)      # polyarch metaxylem ring
    maize_b73.set_value("xylem", "vessel_diameter",          0.074)
    maize_b73.set_value("xylem", "vessel_diameter_sd",       0.0075)
    maize_b73.set_value("xylem", "ratio_proto_meta",         1.9)
    maize_b73.set_value("xylem", "protoxylem_diameter",      0.016)
    maize_b73.set_value("xylem", "protoxylem_cluster_width", 0.021)
    maize_b73.set_value("xylem", "protoxylem_cluster_height", 0.021)

    # ── Phloem bundles (between the xylem poles) ───────────────────────────
    maize_b73.set_value("phloem", "sieve_diameter", 0.014)
    maize_b73.set_value("phloem", "cluster_width",  0.016)
    maize_b73.set_value("phloem", "cluster_height", 0.03)

    # ── Endodermis / pericycle (stele boundary) ────────────────────────────
    maize_b73.set_value("endodermis", "cell_diameter", 0.016)
    maize_b73.set_value("endodermis", "cell_width",    0.028)
    maize_b73.set_value("pericycle", "cell_diameter",  0.0139)
    maize_b73.set_value("pericycle", "cell_width",     0.0127)

    # ── Cortex layers (inner / main / outer) ───────────────────────────────
    maize_b73.params.append({
        "name": "inner_cortex",
        "cell_diameter": 0.027,
        "cell_width": 0.026,
        "n_layers": 1,
        "shift": 0.5,
        "order": 3.5,
    })

    maize_b73.set_value("cortex", "cell_diameter", 0.039)
    maize_b73.set_value("cortex", "cell_width",    0.042)
    maize_b73.set_value("cortex", "n_layers",      3)

    maize_b73.params.append({
        "name": "outer_cortex",
        "cell_diameter": 0.037,
        "cell_width": 0.042,
        "n_layers": 1,
        "shift": 0.5,
        "order": 4.5,
    })

    # ── Exodermis / epidermis ──────────────────────────────────────────────
    maize_b73.set_value("exodermis", "cell_diameter", 0.029)
    maize_b73.set_value("exodermis", "cell_width",    0.027)
    maize_b73.set_value("epidermis", "cell_diameter", 0.018)
    maize_b73.set_value("epidermis", "cell_width",    0.031)

    # ── Intercellular spaces across the cortex tissues ─────────────────────
    maize_b73.set_value("inter_cellular_spaces", "smoothness", 0.05)
    maize_b73.set_value("inter_cellular_spaces", "tissue",
                   ["inner_cortex", "cortex", "outer_cortex"])

    # ── Aerenchyma — lysigenous air spaces across the cortex (B73 only) ────
    # `tissue` accepts a list: the cortex sub-layers are treated as one
    # contiguous band (only its innermost ring is preserved). Runs at
    # generation time, before the inner/outer cortex are retagged to "cortex".
    maize_b73.set_value("aerenchyma", "tissue",
                   ["inner_cortex", "cortex", "outer_cortex"])
    maize_b73.set_value("aerenchyma", "aerenchyma_proportion", 0.31)
    maize_b73.set_value("aerenchyma", "aerenchyma_type", 1)
    maize_b73.set_value("aerenchyma", "n_files", 15)

    # Build once, after all configuration is in place.
    maize_b73_anatomy = RootAnatomy(maize_b73, seed=SEED)
    maize_b73_anatomy.generate_cells()

    # Merge the inner/outer cortex tags into a single "cortex" tag.
    maize_b73_anatomy.retag_cells("inner_cortex", "cortex")
    maize_b73_anatomy.retag_cells("outer_cortex", "cortex")

    maize_roots = []
    maize_roots.append(maize_anatomy)
    maize_roots.append(maize_b73_anatomy)

    n_cols = 2
    n_rows = 1
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axs_flat = axs.flatten()

    labels = ["Maize", "Maize B73"]
    for i, root in enumerate(maize_roots):
        m = anatomy_metrics(root)
        title = (
            f"{labels[i]}\n"
            f"root d = {m['root_diameter']:.3f}   "
            f"stele d = {m['stele_diameter']:.3f}   "
            f"xylem area = {m['xylem_area']:.4f}\n"
            f"aerenchyma prop = {m['aerenchyma_proportion']:.3f}"
        )
        root.plot_cells(show=False, ax=axs_flat[i], title=title)
        legend = axs_flat[i].get_legend()
        if legend:
            legend.remove()

    for j in range(2, len(axs_flat)):
        axs_flat[j].set_visible(False)

    plt.suptitle("Maize anatomy - Cross section", fontsize=14)
    plt.tight_layout()
    if show:
        plt.show()


if __name__ == "__main__":
    main()
