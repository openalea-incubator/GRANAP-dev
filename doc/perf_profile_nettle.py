"""Profiling harness for the heavy secondary-growth path (see performance_proposals.md).

Runs example/dicot_nettle with the pipeline's per-phase timers enabled and a
cProfile pass, printing the per-phase breakdown + top hotspots.

Usage (from repo root, env `granap`):
    python doc/perf_profile_nettle.py
"""
import sys, os, logging, cProfile, pstats, io, time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "example"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: None                       # never block on a window

# The per-phase timers are log.info(...) calls in Organ.generate_cells.
logging.basicConfig(level=logging.INFO, format="STEP  %(message)s", stream=sys.stdout)

import dicot_nettle

print("\n=== per-phase timers (Organ.generate_cells) ===")
t0 = time.time()
pr = cProfile.Profile()
pr.enable()
dicot_nettle.main(show=False)
pr.disable()
print(f"\nTOTAL main(): {time.time() - t0:.2f}s")

for sort in ("tottime", "cumulative"):
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats(sort).print_stats(25)
    print(f"\n================= sorted by {sort} =================")
    print(s.getvalue())
