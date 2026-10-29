"""
Exporting root cross-sections as fspm-TF files: one frame per anatomy, the format the other FSPM platforms (CPlantBox,
GroIMP, VirtualPlantLab) and ParaView read.

    python export_frames.py [--presets P ...]          frames in ./outputs/frames: <preset>.glb

Each section is the one openalea.granap.simulation.simulate builds from a preset (for_root: a monocot root,
for_dicot_root: a dicot one). A frame holds the section's whole MPG: the segment, its layers, cells and compartments,
the walls and connections between them, with their variables (declared units, kinds), and the cells' polygons as a
display mesh (GranapAnatomy.display_triangles). Each frame is then checked with fspm-TF's validator.

Open a frame in ParaView with fspm-TF's reader plugin (fspm-TF/paraview/fspm_tf_reader.py), colour by
Compartment.tissue_code (or cell_area); import_frames.py reads the files back.
"""
import argparse
import os

from openalea.fspm_tf.validate import validate_file
from openalea.metafspm.io.fspm_tf import export_scene

from openalea.granap.simulation import simulate

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = os.path.join(HERE, "outputs")
PRESETS = ("for_root", "for_dicot_root")


def export(presets=PRESETS, outputs: str = OUTPUTS) -> list:
    """Build each preset's section and write it as a frame; the frames' paths."""
    folder = os.path.join(outputs, "frames")
    os.makedirs(folder, exist_ok=True)
    paths = []
    for preset in presets:
        section = simulate(preset=preset)
        path = os.path.join(folder, f"{preset}.glb")
        export_scene(path, section.scene)
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description="GRANAP's sections as fspm-TF frames, one per preset")
    parser.add_argument("--presets", nargs="+", default=list(PRESETS))
    for path in export(parser.parse_args().presets):
        report = validate_file(path)
        print(f"{path} ({os.path.getsize(path) / 1e6:.1f} MB): fspm-TF validator {len(report.errors)} error(s), "
              f"{len(report.warnings)} warning(s)")


if __name__ == "__main__":
    main()
