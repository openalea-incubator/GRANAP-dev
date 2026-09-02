"""
Small 3D mesh helpers — no new dependency (pure numpy): just an OBJ exporter
for visual inspection. Cell/vessel geometry itself is built by
generate_cell_3d.extrude_polygon (literal-copy 2D-cell extrusion).
"""

import numpy as np
from typing import Dict, List, Tuple

Mesh = Tuple[np.ndarray, List[List[int]]]  # (vertices (N,3), faces: list of vertex-index lists)


def write_obj(path: str, cells: Dict[str, List[Mesh]]) -> None:
    """Write a set of independent polyhedra (grouped by tissue type) to one OBJ.

    ``cells``: {type_name: [(vertices, faces), ...]}. Each mesh becomes its own
    ``o`` group so a viewer (Blender/MeshLab/pyvista) can toggle tissues.
    """
    with open(path, "w") as f:
        vertex_offset = 0
        for type_name, meshes in cells.items():
            for i, (vertices, faces) in enumerate(meshes):
                f.write(f"o {type_name}_{i}\n")
                for v in vertices:
                    # 4 decimals is ~1e-4 precision -- 3 decimals risked
                    # collapsing the smallest vessels (protoxylem radius down
                    # to ~0.0018) into degenerate geometry, so stayed at 4 and
                    # cut size via point count instead.
                    f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
                for face in faces:
                    idxs = " ".join(str(vertex_offset + idx + 1) for idx in face)
                    f.write(f"f {idxs}\n")
                vertex_offset += len(vertices)
