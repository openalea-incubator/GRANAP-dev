"""
Importing fspm-TF files: the sections export_frames.py wrote (run first when missing), read two ways.

    python import_frames.py [--show]          figure in ./outputs/tissues.png

1. As any platform reads them, with openalea.fspm_tf alone: the Compartment table (one row per protoplast, wall or
   junction) gives each tissue's number of cells and its area, in the unit its declaration gives; the Connection
   relation, the section's network.
2. Back into metafspm, with openalea.metafspm.io.fspm_tf.import_frame: each frame rebuilds the section's
   DataStructure (its MPG, its variables, bit for bit), with the same cells.
"""
import argparse
import os

import matplotlib
import numpy as np
import pandas as pd

from openalea import fspm_tf
from openalea.metafspm.io.fspm_tf import import_frame

from export_frames import OUTPUTS, PRESETS, export

PROTOPLAST = 0                                                       # GRANAP's kind of a protoplast Compartment


def tissues(path: str) -> pd.DataFrame:
    """Each tissue's protoplasts: their number and their total area (mm2), read with openalea.fspm_tf."""
    section = fspm_tf.read(path).data_structures[0]
    compartments = section.entity_set("Compartment")
    assert compartments.variables["cell_area"].unit == "mm2", compartments.variables["cell_area"].unit
    protoplasts = compartments.columns["kind"] == PROTOPLAST
    table = pd.DataFrame({"tissue": compartments.columns["tissue"][protoplasts],
                          "area": compartments.columns["cell_area"][protoplasts]})
    summary = table.groupby("tissue")["area"].agg(cells="count", area="sum").sort_values("area", ascending=False)
    connections = section.relations[0]
    print(f"{os.path.basename(path)}: {compartments.count} compartments, {int(protoplasts.sum())} protoplasts, "
          f"{connections.count} connections ({connections.source_set} → {connections.target_set})")
    return summary


def main():
    parser = argparse.ArgumentParser(description="GRANAP's fspm-TF sections read back")
    parser.add_argument("--show", action="store_true")
    show = parser.parse_args().show
    paths = [os.path.join(OUTPUTS, preset, "frames", "frame_000000.glb") for preset in PRESETS]
    if not all(os.path.exists(path) for path in paths):
        export()
    summaries = {}
    for preset, path in zip(PRESETS, paths):
        summaries[preset] = tissues(path)
        print(summaries[preset].round(4).to_string())
        identifier, ds = next(iter(import_frame(path).items()))       # {id: DataStructure}, as metafspm wrote them
        cells = int(np.sum(np.asarray(ds.get("kind")) == PROTOPLAST))
        print(f"Rebuilt by metafspm: '{identifier}', a {type(ds).__name__} of {ds.n_nodes()} nodes, {cells} protoplasts\n")
        assert cells == int(summaries[preset]["cells"].sum())

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, len(PRESETS), figsize=(5 * len(PRESETS), 4))
    for axis, (preset, summary) in zip(np.atleast_1d(axes), summaries.items()):
        summary["area"].sort_values().plot.barh(ax=axis)
        axis.set_xlabel("protoplasts' area (mm²)")
        axis.set_ylabel("")
        axis.set_title(preset)
    figure.suptitle("GRANAP's sections, read back from their fspm-TF frames")
    figure.tight_layout()
    path = os.path.join(OUTPUTS, "tissues.png")
    figure.savefig(path, dpi=120)
    print(f"figure: {path}")
    if show:
        plt.show()


if __name__ == "__main__":
    main()
