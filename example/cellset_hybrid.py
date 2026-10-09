"""Mix real CellSet data with a GRANAP stele, and export it for MECHA.

A CellSet-digitized root section carries everything MECHA needs except a
resolved stele: hand-segmenting individual pericycle / phloem / companion cells
is not worth the effort, so the interior of the endodermis is drawn as one
**annular** cell with the visible vessels punched out as holes -- and MECHA
cannot consume a cell with holes.

``CellSetOrgan`` keeps every digitized cell verbatim and regenerates only that
region, by running a small root anatomy whose base shape *is* the real outline.

Run from the ``example/`` directory::

    python cellset_hybrid.py

then feed the XML it writes to MECHA (sibling repo)::

    from openalea.mecha import Mecha, InData, visualize
    data = InData(cellset_file="hybrid.xml")
    data.geometry.set_maturity_stages([1, 3])
    m = Mecha(data)
    m.water_flux()
    visualize(m, visu_type="flow")

GRANAP writes millimetres and MECHA's ``GeometryData.im_scale`` defaults to
1000, so the round trip (CellSet µm -> GRANAP mm -> MECHA µm) is consistent.
"""

import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")           # render headlessly; drop this to use plt.show()
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from openalea.granap.anatomy_writer import AnatomyWriter
from openalea.granap.cellset_organ import CellSetOrgan

# The digitized section lives in the sibling MECHA checkout; a copy is
# vendored under test/inputs so the test suite does not depend on it.
CELLSET = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "test", "inputs", "partial_CellSet_brasicaceae.xml",
)
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
SEED = 0                        # always fix the seed for anything comparable

COLOURS = {
    "epidermis": "#e69f00", "cortex": "#56b4e9", "endodermis": "#009e73",
    "pericycle": "#7b6ba8", "stele": "#f2a7c3", "phloem": "#f0e442",
    "xylem": "#d55e00",
}
DIGITIZED = {"epidermis", "cortex", "endodermis"}


def build() -> CellSetOrgan:
    organ = CellSetOrgan(
        CELLSET,
        seed=SEED,
        # Defaults, spelled out because they are the interesting choices:
        regenerate=("stele",),      # plus any annular cell, whatever its tag
        keep_inner_cells=True,      # the digitized vessels stay, generated around
        donor_xylem_tag="stele",    # the xylem is real data; don't invent vessels
        star_orientation="auto",    # align the plate with the measured vessel axis
    )
    organ.generate_cells()
    return organ


def report(organ: CellSetOrgan) -> None:
    """Per-tissue count *and* area -- cell counts hide size/proportion errors."""
    agg = defaultdict(lambda: [0, 0.0])
    for cell in organ.all_cells.cells:
        agg[cell.type][0] += 1
        agg[cell.type][1] += cell.polygon.area * 1e6      # mm^2 -> um^2
    print(f"{'tissue':12s} {'n':>4s} {'area (um2)':>12s}")
    for tag, (n, area) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        print(f"{tag:12s} {n:4d} {area:12.1f}")

    for graft in organ.graft_report:
        fate = (
            f"retagged '{organ.donor_xylem_tag}'"
            if organ.donor_xylem_tag is not None
            else "deleted"
        )
        print(
            f"\ngraft into '{graft['tag']}': {graft['n_donor_cells']} generated cells, "
            f"{graft['n_inner_kept']} digitized cells kept, "
            f"{graft['n_donor_xylem_removed']} donor vessels {fate},\n"
            f"  plate rotated to {graft['angle']:.1f} deg, "
            f"region tiled to {graft['coverage'] * 100:.2f}%, "
            f"{graft['n_vertices_welded']} interface vertices welded"
        )

    organ.export_to_adjencymatrix()
    topo = organ.topology_report()
    print(
        f"\nnetwork: {topo['n_cells']} cells, {topo['n_walls']} walls, "
        f"{topo['n_junctions']} junctions"
    )
    print(f"  border walls (real outer surface): {topo['border_walls']}")
    # The acceptance test for the graft. A wall with one flanking cell inside the
    # tissue is not just a missing link -- MECHA files it under
    # border_aerenchyma, i.e. it becomes a gas-space boundary in the stele.
    print(f"  border walls inside the tissue:    {topo['interior_border_walls']}"
          f"   <- must be empty")
    print(f"  walls with >2 flanking cells:      {topo['multi_cell_walls']}"
          f"   <- must be 0")
    print("  plasmodesmata across the graft:    "
          + ", ".join(
              f"{a}/{b}={n}"
              for (a, b), n in sorted(topo["plasmodesmata_pairs"].items())
              if {a, b} & {"pericycle", "stele", "phloem"} and a != b
          ))


def render(organ: CellSetOrgan, path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.2))
    for ax, zoom in zip(axes, (False, True)):
        for cell in organ.all_cells.cells:
            xs, ys = cell.polygon.exterior.xy
            generated = not (
                cell.type in DIGITIZED or (cell.type == "xylem" and cell.cgroup == 13)
            )
            ax.fill([x * 1e3 for x in xs], [y * 1e3 for y in ys],
                    facecolor=COLOURS.get(cell.type, "#cccccc"),
                    edgecolor="#22222c", linewidth=0.45,
                    hatch="///" if generated else None)
        ax.set_aspect("equal")
        ax.set_xlabel("µm")
        if zoom:
            pts = [c.polygon.centroid for c in organ.all_cells.cells
                   if c.type in ("stele", "pericycle", "phloem", "xylem")]
            cx = sum(p.x for p in pts) / len(pts) * 1e3
            cy = sum(p.y for p in pts) / len(pts) * 1e3
            ax.set_xlim(cx - 26, cx + 26)
            ax.set_ylim(cy - 26, cy + 26)
            ax.set_title("stele detail — hatched = GRANAP-generated,\n"
                         "solid = digitized (the two real vessels)", fontsize=10)
        else:
            ax.set_ylabel("µm")
            ax.set_title("hybrid: CellSet data + GRANAP stele", fontsize=11)

    handles = [Patch(facecolor=c, edgecolor="#22222c", label=t)
               for t, c in COLOURS.items()]
    handles += [
        Patch(facecolor="white", edgecolor="#22222c", label="digitized (CellSet)"),
        Patch(facecolor="white", edgecolor="#22222c", hatch="///",
              label="generated (GRANAP)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=9, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nrender saved to {path}")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    organ = build()
    report(organ)
    render(organ, os.path.join(OUT_DIR, "hybrid.png"))

    xml_path = os.path.join(OUT_DIR, "hybrid.xml")
    AnatomyWriter(organ).write_to_xml(xml_path)
    print("feed that file to MECHA as InData(cellset_file=...) — see the module docstring")


if __name__ == "__main__":
    main()
