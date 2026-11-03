"""
Exporting root cross-sections as fspm-TF files: one frame per anatomy, the format the other FSPM platforms (CPlantBox,
GroIMP, VirtualPlantLab) and ParaView read.

    python export_frames.py [--presets P ...]          frames in ./outputs/<preset>/frames: frame_000000.glb

Each section is the one openalea.granap.simulation.simulate builds from a preset (for_root: a monocot root,
for_dicot_root: a dicot one), recorded by metafspm's SceneRecorder (frames only, no summaries) with run.json and
scenario.json beside them. A frame holds the section's whole MPG: the segment, its layers, cells and compartments,
the walls and connections between them, with their variables (declared units, kinds), and the cells' polygons as a
display mesh (GranapAnatomy.display_triangles). Each frame is then checked with fspm-TF's validator.

A second, filtered export shows logging: the segment and its compartments only, with their tissue, kind and area
(and the cells' polygons), in ./outputs/log/<preset>/frames (outputs only: such frames are not reimported).

Open a frame in ParaView with fspm-TF's reader plugin (fspm-TF/paraview/fspm_tf_reader.py), colour by
Compartment.tissue_code (or cell_area); import_frames.py reads the files back.
"""
import argparse
import os

from openalea.fspm_tf.validate import validate_file
from openalea.metafspm.scene.recorder import SceneRecorder

from openalea.granap.simulation import simulate

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = os.path.join(HERE, "outputs")
PRESETS = ("for_root", "for_dicot_root")
# a log of the section: its segment and compartments, their tissue, kind and area
LOG = dict(scales=["SubOrgan", "Compartment"], variables=["tissue", "kind", "cell_area"])


def export(presets=PRESETS, outputs: str = OUTPUTS, options: dict = None) -> list:
    """Build each preset's section and record it as a frame (filtered by *options*, e.g. LOG) in *outputs*/<preset>;
    the frames' paths."""
    paths = []
    for preset in presets:
        folder = os.path.join(outputs, preset)
        simulate(preset=preset, recorder=SceneRecorder(folder, summaries=False, frame_period=1, display_period=1,
                                                       frame_options=options))       # the built section, its frame
        paths.append(os.path.join(folder, "frames", "frame_000000.glb"))
    return paths


def main():
    parser = argparse.ArgumentParser(description="GRANAP's sections as fspm-TF frames, one per preset")
    parser.add_argument("--presets", nargs="+", default=list(PRESETS))
    presets = parser.parse_args().presets
    for path in export(presets):
        report = validate_file(path)
        print(f"{path} ({os.path.getsize(path) / 1e6:.1f} MB): fspm-TF validator {len(report.errors)} error(s), "
              f"{len(report.warnings)} warning(s)")
    for preset, path in zip(presets, export(presets, os.path.join(OUTPUTS, "log"), options=LOG)):
        report = validate_file(path)
        print(f"logged {preset} (scales {LOG['scales']}, {len(LOG['variables'])} variables): "
              f"{os.path.getsize(path) / 1e6:.1f} MB, {len(report.errors)} error(s)")


if __name__ == "__main__":
    main()
