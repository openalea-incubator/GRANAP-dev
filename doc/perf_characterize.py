"""Byte-identical safety net for perf work (see performance_proposals.md).

Pins the seed=0 cell-type census + a coarse geometry hash for four canonical
root configs. Save a baseline on a known-good tree, then re-check after each
optimisation; wins ①/② in the perf doc must keep every hash unchanged.

Usage (from repo root, env `granap`):
    python doc/perf_characterize.py save     # write baseline next to this file
    python doc/perf_characterize.py          # check current tree vs baseline
"""
import sys, os, json, hashlib

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
import matplotlib
matplotlib.use("Agg")

from openalea.granap.root_class import RootAnatomy
from openalea.granap.input_data import OrganInputData

BASELINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "perf_baseline.json")


def _census(root):
    c = {}
    for cell in root.all_cells.cells:
        c[cell.type] = c.get(cell.type, 0) + 1
    return dict(sorted(c.items()))


def _geom_hash(root):
    h = hashlib.sha256()
    for cell in sorted(root.all_cells.cells, key=lambda c: (round(c.x, 4), round(c.y, 4))):
        if cell.polygon is None or cell.polygon.is_empty:
            h.update(b"none")
        else:
            p = cell.polygon
            h.update(f"{cell.type}:{p.area:.6f}:{p.centroid.x:.5f}:{p.centroid.y:.5f};".encode())
    return h.hexdigest()[:16]


def _monocot_default():
    return RootAnatomy(OrganInputData.for_root(), seed=0)


def _monocot_arch():
    d = OrganInputData.for_root()
    d.set_value("xylem", "xylem_shape", "arch")
    return RootAnatomy(d, seed=0)


def _dicot_primary():
    d = OrganInputData.for_dicot_root()
    d.set_value("stele", "thickness", 1.0)
    return RootAnatomy(d, seed=0)


def _dicot_secondary():
    d = OrganInputData.for_dicot_root()
    d.set_value("stele", "thickness", 1.0)
    d.set_value("secondary_growth", "value", True)
    return RootAnatomy(d, seed=0)


VARIANTS = {
    "monocot_default": _monocot_default,
    "monocot_arch": _monocot_arch,
    "dicot_primary": _dicot_primary,
    "dicot_secondary": _dicot_secondary,
}


def run():
    out = {}
    for name, fn in VARIANTS.items():
        r = fn()
        r.generate_cells()
        out[name] = {"census": _census(r), "hash": _geom_hash(r)}
        print(f"{name:18s} {out[name]['hash']}  {out[name]['census']}")
    return out


if __name__ == "__main__":
    result = run()
    if len(sys.argv) > 1 and sys.argv[1] == "save":
        json.dump(result, open(BASELINE, "w"), indent=2)
        print("\nSAVED baseline ->", BASELINE)
    else:
        base = json.load(open(BASELINE))
        ok = True
        for name in VARIANTS:
            if result[name] != base.get(name):
                ok = False
                print(f"\nDRIFT in {name}:\n  base={base.get(name)}\n  now ={result[name]}")
        print("\n" + ("ALL MATCH OK" if ok else "MISMATCH FAIL"))
        sys.exit(0 if ok else 1)
