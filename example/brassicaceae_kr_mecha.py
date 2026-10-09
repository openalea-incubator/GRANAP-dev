"""Estimate radial hydraulic conductivity (kr) for every hybrid section, in MECHA.

Companion to ``brassicaceae_roots.py``: that script turns CellSet digitizations
into MECHA-ready XML, this one runs MECHA over all of them with its **default**
hydraulic parameterisation and reports kr per apoplastic-barrier stage.

.. important::

   **Run this in the ``mecha`` environment, not ``granap``.**  It imports
   ``openalea.mecha`` only — it never touches GRANAP — but it reads the XML that
   ``brassicaceae_roots.py`` (a ``granap`` script) writes.  The two halves of the
   workflow live in different environments on purpose::

       conda activate granap && cd example && python brassicaceae_roots.py
       conda activate mecha  && cd example && python brassicaceae_kr_mecha.py

Default hydraulics means exactly that: ``InData(cellset_file=...)`` loads
MECHA's own defaults for wall / plasmodesmata / aquaporin conductivities and
boundary conditions, and nothing here overrides them.  So the numbers compare
*anatomies*, not parameterisations — which is the point of a cross-species
comparison, and also why they are not predictions of any real root's kr.

What kr is here (``Mecha.standard_water_flow``)::

    kr = Q_soil / (psi_soil - psi_xylem) / perimeter / height / 1e-4

i.e. total flow across the root surface per unit driving pressure per unit
surface area, in **cm/hPa/d**.  It is reported once per maturity stage, because
the apoplastic barrier is what dominates it.

.. warning::

   kr is only as trustworthy as the anatomy's topology.  MECHA files any wall
   with a single flanking cell whose owner is not epidermis under
   ``border_aerenchyma`` and treats it as a gas-space boundary — so a section
   that failed ``brassicaceae_roots.check_topology()`` will still produce a kr,
   just a meaningless one.  This script prints the border-wall counts next to
   every result so that shows up instead of hiding.  For a clean section the
   aerenchyma count should be the number of genuine interior air spaces in the
   digitization (0 for a root that has none, 1 for the reference *Arabidopsis*
   section, which has one real cortex edge wall).
"""

from __future__ import annotations

import contextlib
import csv
import glob
import io
import os
import traceback
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")           # save figures, never block on a window

from openalea.mecha import InData, Mecha, visualize          # noqa: E402
from openalea.mecha.utils.data_loader import BARRIER_KEY     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
IN_DIR = os.path.join(HERE, "outputs", "brassicaceae")
OUT_CSV = os.path.join(IN_DIR, "kr_summary.csv")

# Apoplastic barrier stages to solve, as a maturation series (see BARRIER_KEY):
#   1 = En_cs            endodermal Casparian strip only
#   3 = En_sub           suberized endodermis
#   4 = En_sub + Ex_cs   ... plus an exodermal Casparian strip
#   8 = En_sub + Ex_sub  ... plus a suberized exodermis
# Stages 4 and 8 only differ from 3 in a root that *has* an exodermis; a section
# without one is reported below so an identical kr is read as "no exodermis to
# suberize" rather than as a modelling result.
BARRIERS: List[int] = [1, 3, 4, 8]
HEIGHT_UM: float = 200.0        # modelled segment height; MECHA's own default

# Cell types that carry an exodermal barrier, as MECHA ranks them.
EXODERMIS_RANK = 1

# Barrier stage to draw a flow map for, or None to skip the figures.  One map
# per section is written next to its XML.  Stage 1 (En_cs) is the informative
# one: with only a Casparian strip the apoplastic path is still open through the
# cortex, so the arrows actually show where water crosses.
FLOW_FIGURE_BARRIER: int | None = 1

VERBOSE = False                 # True to let MECHA's solver chatter through


def solve(path: str, flow_figure: str | None = None) -> Dict[str, Any]:
    """Run MECHA on one section and pull out kr per maturity stage."""
    data = InData(cellset_file=path)
    # Pass both lists: set_maturity_stages warns and silently reuses height[0]
    # when the lengths disagree.
    data.geometry.set_maturity_stages(BARRIERS, [HEIGHT_UM] * len(BARRIERS))

    model = Mecha(data)
    model.water_flux(verbose=VERBOSE)

    net = model.network
    stages = []
    for i, stage in enumerate(data.geometry.maturity_stages):
        stages.append({
            "barrier": int(stage["barrier"]),
            "barrier_name": BARRIER_KEY.get(int(stage["barrier"]), str(stage["barrier"])),
            "kr": float(model.kr_tot[i][0]),
            "total_flow": float(model.total_flow[i][0]),
        })

    if flow_figure is not None and FLOW_FIGURE_BARRIER in BARRIERS:
        # maturity_idx indexes BARRIERS, not the barrier number itself.
        # scenario_idx is passed explicitly so MECHA restores that stage's state:
        # the default 'current' would draw whichever stage happened to be solved
        # last, which after this loop is the *final* barrier, not the one asked for.
        visualize(
            model, visu_type="flow",
            maturity_idx=BARRIERS.index(FLOW_FIGURE_BARRIER),
            scenario_idx=0,
            save_path=flow_figure,
        )

    # network.perimeter is 2*pi*distance_avg_epi*1e-4, i.e. **cm** -- and it is
    # the divisor in kr, so it is worth reporting rather than assuming.
    ranks = getattr(net, "cell_ranks", None)
    has_exodermis = bool(ranks is not None and (ranks == EXODERMIS_RANK).any())

    return {
        "n_cells": int(net.n_cells),
        "n_walls": int(net.n_walls),
        "perimeter_cm": float(net.perimeter),
        "has_exodermis": has_exodermis,
        "n_border_walls": len(getattr(net, "border_walls", []) or []),
        "n_border_aerenchyma": len(getattr(net, "border_aerenchyma", []) or []),
        "stages": stages,
    }


def main() -> None:
    files = sorted(glob.glob(os.path.join(IN_DIR, "*.xml")))
    if not files:
        raise SystemExit(
            f"no .xml in {IN_DIR}\n"
            "run `python brassicaceae_roots.py` in the granap env first"
        )

    print(f"{len(files)} section(s) in {IN_DIR}")
    print(f"barriers {BARRIERS} = {[BARRIER_KEY.get(b) for b in BARRIERS]}, "
          f"height {HEIGHT_UM} um, MECHA default hydraulics\n")

    rows: List[Dict[str, Any]] = []
    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        figure = None
        if FLOW_FIGURE_BARRIER is not None:
            label = BARRIER_KEY.get(FLOW_FIGURE_BARRIER, str(FLOW_FIGURE_BARRIER))
            figure = os.path.join(
                IN_DIR, f"{name}_flow_{label.replace(' ', '').replace('+', '_')}.png"
            )
        buffer = io.StringIO()
        try:
            # MECHA prints a lot per solve; keep it unless something goes wrong.
            if VERBOSE:
                result = solve(path, flow_figure=figure)
            else:
                with contextlib.redirect_stdout(buffer):
                    result = solve(path, flow_figure=figure)
        except Exception as exc:
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
            if not VERBOSE and buffer.getvalue():
                print("       --- MECHA output before the failure ---")
                for line in buffer.getvalue().strip().splitlines()[-12:]:
                    print(f"       {line}")
            traceback.print_exc()
            continue

        flag = "" if result["n_border_aerenchyma"] <= 1 else "   <- check topology"
        print(f"[{name}] {result['n_cells']} cells, perimeter "
              f"{result['perimeter_cm'] * 1e4:.1f} um "
              f"({result['perimeter_cm']:.5f} cm), border walls "
              f"{result['n_border_walls']} + {result['n_border_aerenchyma']} "
              f"aerenchyma{flag}")
        if not result["has_exodermis"]:
            print("       no exodermis in this section — barriers 4 and 8 can only "
                  "act on the endodermis, so expect them to match barrier 3")
        for stage in result["stages"]:
            print(f"    barrier {stage['barrier']} ({stage['barrier_name']:<16s}) "
                  f"kr = {stage['kr']:.6g} cm/hPa/d   "
                  f"Q = {stage['total_flow']:.6g}")
            rows.append({
                "section": name,
                "barrier": stage["barrier"],
                "barrier_name": stage["barrier_name"],
                "kr_cm_hPa_d": stage["kr"],
                "total_flow": stage["total_flow"],
                "n_cells": result["n_cells"],
                "n_walls": result["n_walls"],
                "perimeter_cm": result["perimeter_cm"],
                "has_exodermis": result["has_exodermis"],
                "n_border_walls": result["n_border_walls"],
                "n_border_aerenchyma": result["n_border_aerenchyma"],
            })
        if figure and os.path.isfile(figure):
            print(f"    flow map -> {os.path.basename(figure)}")
        print()

    if not rows:
        raise SystemExit("no section solved")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"kr summary -> {OUT_CSV}")

    # Compact cross-section table: one row per section, one column per barrier.
    sections = sorted({r["section"] for r in rows})
    print(f"\n{'section':22s}" + "".join(
        f"{BARRIER_KEY.get(b, b):>18s}" for b in BARRIERS))
    print(f"{'':22s}" + "".join(f"{'kr (cm/hPa/d)':>18s}" for _ in BARRIERS))
    for section in sections:
        cells = []
        for b in BARRIERS:
            match = next((r for r in rows
                          if r["section"] == section and r["barrier"] == b), None)
            cells.append(f"{match['kr_cm_hPa_d']:18.6g}" if match else f"{'-':>18s}")
        print(f"{section:22s}" + "".join(cells))


if __name__ == "__main__":
    main()
