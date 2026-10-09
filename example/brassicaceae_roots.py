"""Template: a batch of CellSet-digitized Brassicaceae roots -> MECHA-ready XML.

This is the *many roots* version of ``cellset_hybrid.py``.  That script walks
through one section in detail; this one is the shape a comparative study wants:
a registry of species, one pipeline applied uniformly to all of them, a hard
correctness gate per section, and a single comparison table at the end.

Only *Arabidopsis thaliana* is digitized so far — every other species is a
placeholder.  That is deliberate: the script runs today, reports which sections
are still missing, and needs no code change when a new file lands.  To add a
species you drop its CellSet XML next to the others and fill in one
:class:`Species` row.

**Per-species tuning.**  Almost nothing should need tuning: region position,
size and shape all come from the digitized outline, and the donor stele is sized
from the region.  The one genuinely species-specific parameter is the number of
xylem poles — Brassicaceae roots are typically diarch (*Arabidopsis*), but the
larger crop roots are often tri- or tetrarch.  That goes in ``donor_overrides``
as ``{"xylem": {"n_vascular_peak": n}}``.  Reach for anything else only when a
section demonstrably needs it, and write down why in ``notes``.

.. warning::

   **Only ``n_vascular_peak=2`` currently produces a MECHA-sound graft.**  Three
   or more poles packs the donor's vessels tighter than the small stele can
   hold; the tessellation comes out with degenerate slivers, and ~30-40 walls
   end up with a single flanking cell.  Measured on the *Arabidopsis* section at
   ``seed=0``: 2 poles → 0 interior border walls, 3/4/5 poles → 30-40.  This is
   upstream of the graft (it is the donor's own vessel packing, not the
   interface welding — changing the weld barely moves the number), so it needs
   fixing in the donor configuration or the root recipe's packing, not here.

   The pole counts below therefore state **the biology, not a working setting**.
   Every section is gated by :func:`check_topology`, so a species that trips
   this is reported as ``TOPOLOGY`` rather than quietly folded into the
   comparison — and the moment the packing is fixed, this registry is already
   correct.  To get a usable network for a polyarch species today, set its
   override to 2 and accept a diarch stele.

**The correctness gate.**  ``check_topology()`` below is not optional
bookkeeping.  A graft can look perfect and still hand MECHA a broken network: a
wall with a single flanking cell inside the tissue is filed by MECHA under
``border_aerenchyma``, i.e. it becomes a gas-space boundary in the middle of the
stele.  Every section must come out with border walls only on its real outer
surface.  A section that fails is reported and its XML is still written, but it
is flagged loudly rather than quietly averaged into a comparison.

Run from the ``example/`` directory::

    python brassicaceae_roots.py

Outputs land in ``example/outputs/brassicaceae/``: one XML and one PNG per
species, a contact-sheet figure, and ``summary.csv``.  Feed an XML to MECHA
exactly as in ``cellset_hybrid.py``, or run the companion
``brassicaceae_kr_mecha.py`` (in the ``mecha`` environment) to get radial
hydraulic conductivity for every section at once.
"""

from __future__ import annotations

import csv
import os
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from openalea.granap.anatomy_writer import AnatomyWriter
from openalea.granap.cellset_organ import CellSetOrgan

HERE = os.path.dirname(os.path.abspath(__file__))
# Where the digitized sections live.  Arabidopsis is vendored into test/inputs
# so the test suite does not depend on a sibling checkout; put new species
# alongside it, or point INPUT_DIRS at wherever your CellSet exports collect.
INPUT_DIRS = [
    os.path.join(HERE, "..", "test", "inputs"),
    os.path.join(HERE, "inputs", "brassicaceae"),
]
OUT_DIR = os.path.join(HERE, "outputs", "brassicaceae")
SEED = 0                      # fixed: every census/figure here is meant to be comparable

# Shared with the single-section walkthrough so both render the same way.
from cellset_hybrid import COLOURS, DIGITIZED       # noqa: E402


@dataclass
class Species:
    """One digitized section, plus the little that is species-specific."""

    name: str                                   # binomial, for figures and the table
    code: str                                   # short slug, for filenames
    cellset_file: str                           # basename; resolved against INPUT_DIRS
    scale: float = 0.001                        # CellSet units -> mm (0.001 = from um)
    donor_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    kwargs: Dict[str, Any] = field(default_factory=dict)   # rare per-section escapes
    notes: str = ""

    def resolve(self) -> Optional[str]:
        for folder in INPUT_DIRS:
            path = os.path.join(folder, self.cellset_file)
            if os.path.isfile(path):
                return path
        return None


# --------------------------------------------------------------------------
# The registry.  `n_vascular_peak` values for the undigitized species are the
# usual textbook arch number for that root and are *expectations, not data* --
# check each against its own section when the digitization arrives.  Anything
# other than 2 currently fails the topology gate (see the module warning); the
# value records what the plant does, and check_topology() records what the code
# can currently deliver.
# --------------------------------------------------------------------------
SOUND_POLE_COUNTS = (2,)
SPECIES: List[Species] = [
    Species(
        "Arabidopsis thaliana", "at",
        "partial_CellSet_brasicaceae.xml",
        donor_overrides={"xylem": {"n_vascular_peak": 2}},
        notes="diarch; the reference section, 40 digitized cells",
    ),
    Species(
        "Arabidopsis lyrata", "al", "al_root_xsec.xml",
        donor_overrides={"xylem": {"n_vascular_peak": 2}},
        notes="diarch, close relative of A. thaliana",
    ),
    Species(
        "Brassica napus", "bn", "bn_root_xsec.xml",
        donor_overrides={"xylem": {"n_vascular_peak": 4}},
        notes="oilseed rape; larger root, typically tetrarch",
    ),
    Species(
        "Brassica oleracea", "bo", "bo_root_xsec.xml",
        donor_overrides={"xylem": {"n_vascular_peak": 4}},
    ),
    Species(
        "Brassica rapa", "br", "br_root_xsec.xml",
        donor_overrides={"xylem": {"n_vascular_peak": 4}},
    ),
    Species(
        "Camelina sativa", "cs", "cs_root_xsec.xml",
        donor_overrides={"xylem": {"n_vascular_peak": 2}},
    ),
    Species(
        "Capsella bursa-pastoris", "cb", "cb_root_xsec.xml",
        donor_overrides={"xylem": {"n_vascular_peak": 2}},
    ),
    Species(
        "Eutrema salsugineum", "es", "es_root_xsec.xml",
        donor_overrides={"xylem": {"n_vascular_peak": 2}},
        notes="halophyte; expect a thicker apoplastic barrier in the MECHA run",
    ),
    Species(
        "Raphanus sativus", "rs", "rs_root_xsec.xml",
        donor_overrides={"xylem": {"n_vascular_peak": 4}},
    ),
    Species(
        "Thlaspi arvense", "ta", "ta_root_xsec.xml",
        donor_overrides={"xylem": {"n_vascular_peak": 2}},
    ),
]


# --------------------------------------------------------------------------


def build(species: Species, path: str) -> CellSetOrgan:
    organ = CellSetOrgan(
        path,
        seed=SEED,
        scale=species.scale,
        donor_overrides=species.donor_overrides,
        **species.kwargs,
    )
    organ.generate_cells()
    organ.export_to_adjencymatrix()
    return organ


def check_topology(organ: CellSetOrgan) -> List[str]:
    """Return the reasons this section is not MECHA-sound; empty means it is."""
    report = organ.topology_report()
    problems = []
    if report["interior_border_walls"]:
        problems.append(
            f"{sum(report['interior_border_walls'].values())} one-cell wall(s) inside "
            f"the tissue {report['interior_border_walls']} — MECHA reads these as "
            f"gas-space boundaries"
        )
    if report["multi_cell_walls"]:
        problems.append(f"{report['multi_cell_walls']} wall(s) with >2 flanking cells")
    for graft in organ.graft_report:
        if graft["coverage"] < 0.9999:
            problems.append(
                f"region '{graft['tag']}' only tiled to {graft['coverage'] * 100:.2f}%"
            )
    return problems


def census(organ: CellSetOrgan) -> Dict[str, Dict[str, float]]:
    """``{tag: {"n": count, "area_um2": total}}`` — always report area, not just count."""
    out: Dict[str, Dict[str, float]] = {}
    for cell in organ.all_cells.cells:
        row = out.setdefault(cell.type, {"n": 0, "area_um2": 0.0})
        row["n"] += 1
        row["area_um2"] += cell.polygon.area * 1e6
    return out


def draw(organ: CellSetOrgan, ax, title: str) -> None:
    for cell in organ.all_cells.cells:
        xs, ys = cell.polygon.exterior.xy
        generated = not (
            cell.type in DIGITIZED or (cell.type == "xylem" and cell.cgroup == 13)
        )
        ax.fill([x * 1e3 for x in xs], [y * 1e3 for y in ys],
                facecolor=COLOURS.get(cell.type, "#cccccc"),
                edgecolor="#22222c", linewidth=0.3,
                hatch="///" if generated else None)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9, style="italic")
    ax.set_xticks([])
    ax.set_yticks([])


def contact_sheet(results: List[Dict[str, Any]], path: str) -> None:
    """One panel per species, all on a shared scale so sizes are comparable."""
    done = [r for r in results if r["organ"] is not None]
    if not done:
        return
    cols = min(5, len(done))
    rows = (len(done) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 3.4 * rows), squeeze=False)

    span = 0.0
    for record in done:
        minx, miny, maxx, maxy = record["organ"]._create_base_shape().bounds
        span = max(span, (maxx - minx) * 1e3, (maxy - miny) * 1e3)
    half = span / 2 * 1.06

    for i, record in enumerate(done):
        ax = axes[i // cols][i % cols]
        flag = "" if not record["problems"] else "  ⚠"
        draw(record["organ"], ax, record["species"].name + flag)
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
    for j in range(len(done), rows * cols):
        axes[j // cols][j % cols].axis("off")

    handles = [Patch(facecolor=c, edgecolor="#22222c", label=t) for t, c in COLOURS.items()]
    handles += [
        Patch(facecolor="white", edgecolor="#22222c", label="digitized (CellSet)"),
        Patch(facecolor="white", edgecolor="#22222c", hatch="///", label="generated (GRANAP)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=9, frameon=False, fontsize=9)
    fig.suptitle("Brassicaceae root cross-sections — CellSet data with a GRANAP stele",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\ncontact sheet -> {path}")


def write_summary(results: List[Dict[str, Any]], path: str) -> List[str]:
    tags = sorted({t for r in results if r["census"] for t in r["census"]})
    header = (["species", "code", "status", "n_cells", "section_area_um2",
               "stele_area_um2", "coverage", "xylem_poles"]
              + [f"n_{t}" for t in tags] + [f"area_{t}" for t in tags] + ["notes"])
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for r in results:
            c = r["census"] or {}
            graft = r["organ"].graft_report[0] if r["organ"] and r["organ"].graft_report else {}
            writer.writerow(
                [r["species"].name, r["species"].code, r["status"],
                 sum(int(v["n"]) for v in c.values()) if c else "",
                 f"{sum(v['area_um2'] for v in c.values()):.1f}" if c else "",
                 f"{graft.get('region_area', 0) * 1e6:.1f}" if graft else "",
                 f"{graft.get('coverage', 0):.4f}" if graft else "",
                 r["species"].donor_overrides.get("xylem", {}).get("n_vascular_peak", ""),
                 ]
                + [int(c[t]["n"]) if t in c else 0 for t in tags]
                + [f"{c[t]['area_um2']:.1f}" if t in c else "0.0" for t in tags]
                + [r["species"].notes]
            )
    print(f"summary   -> {path}")
    return tags


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    results: List[Dict[str, Any]] = []

    for species in SPECIES:
        path = species.resolve()
        record: Dict[str, Any] = {
            "species": species, "organ": None, "census": None,
            "status": "", "problems": [],
        }

        if path is None:
            record["status"] = "no data yet"
            print(f"[skip] {species.name:26s} {species.cellset_file} not found")
            results.append(record)
            continue

        poles = species.donor_overrides.get("xylem", {}).get("n_vascular_peak")
        if poles is not None and poles not in SOUND_POLE_COUNTS:
            print(f"[warn] {species.name:26s} n_vascular_peak={poles} is not yet "
                  f"supported (see the module warning); expect a TOPOLOGY result")

        try:
            organ = build(species, path)
        except Exception as exc:                       # one bad section must not stop the batch
            record["status"] = f"FAILED: {type(exc).__name__}"
            print(f"[FAIL] {species.name:26s} {type(exc).__name__}: {exc}")
            traceback.print_exc()
            results.append(record)
            continue

        problems = check_topology(organ)
        record.update(
            organ=organ, census=census(organ), problems=problems,
            status="ok" if not problems else "TOPOLOGY",
        )

        AnatomyWriter(organ).write_to_xml(os.path.join(OUT_DIR, f"{species.code}_hybrid.xml"))
        fig, ax = plt.subplots(figsize=(5.2, 5.2))
        draw(organ, ax, species.name)
        fig.savefig(os.path.join(OUT_DIR, f"{species.code}_hybrid.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

        report = organ.topology_report()
        print(f"[ok]   {species.name:26s} {report['n_cells']:4d} cells, "
              f"{report['n_walls']:4d} walls, "
              f"border {report['border_walls']}")
        for problem in problems:
            print(f"       !! {problem}")

        results.append(record)

    # ---------------------------------------------------------------- table
    tags = write_summary(results, os.path.join(OUT_DIR, "summary.csv"))
    contact_sheet(results, os.path.join(OUT_DIR, "contact_sheet.png"))

    print(f"\n{'species':26s} {'status':9s} {'cells':>6s} "
          + " ".join(f"{t[:9]:>9s}" for t in tags))
    for r in results:
        c = r["census"] or {}
        print(f"{r['species'].name:26s} {r['status']:9s} "
              f"{sum(int(v['n']) for v in c.values()) if c else 0:6d} "
              + " ".join(f"{int(c[t]['n']) if t in c else 0:9d}" for t in tags))

    ok = sum(1 for r in results if r["status"] == "ok")
    pending = sum(1 for r in results if r["status"] == "no data yet")
    print(f"\n{ok}/{len(results)} sections MECHA-sound, {pending} awaiting digitization")
    bad = [r["species"].name for r in results if r["status"] not in ("ok", "no data yet")]
    if bad:
        print(f"needs attention: {', '.join(bad)}")


if __name__ == "__main__":
    main()
