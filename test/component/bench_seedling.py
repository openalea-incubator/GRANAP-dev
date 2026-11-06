"""
Benchmark of a seedling-sized root system populated with GRANAP anatomies: a main root
and laterals in three diameter classes, 48 segments; the time of each phase and the sizes.

    python test/mpg_capabilities/bench_seedling.py [n_laterals]      (from GRANAP/, default 8: 48 segments)
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openalea.metafspm.coupling.component import StructuralComponent

from openalea.metafspm.data_structure.mpg import MPG
from openalea.granap.metafspm.component import GranapAnatomy
from openalea.granap.metafspm.library import AnatomyLibrary

from roots import _axis, initiated, planting

N_LATERALS = int(sys.argv[1]) if len(sys.argv) > 1 else 8


class Seedling(StructuralComponent):
    """A main root of 16 segments (0.55 mm), N_LATERALS laterals of 4 segments alternating 0.35 and 0.25 mm."""

    @classmethod
    def initiate_plant(cls, g, plant, parameters):
        main = _axis(g, plant, 16, 0.55e-3)
        for k in range(N_LATERALS):
            _axis(g, plant, 4, (0.35e-3 if k % 2 == 0 else 0.25e-3), bearer=main[1 + k])


class Anatomy(GranapAnatomy):
    library = AnatomyLibrary()


def main():
    timings = {}
    start = time.perf_counter()
    scratch = MPG()                     # the templates are each population's: generated here on their own
    for key in ((1.1, 0., 0.), (0.7, 0., 0.), (0.5, 0., 0.)):
        Anatomy.library.template(scratch, key)
    timings["templates (3 classes)"] = time.perf_counter() - start
    start = time.perf_counter()
    population = initiated(planting(), Seedling, Anatomy)
    g = population.data_structure.structure
    timings["architecture + instantiation"] = time.perf_counter() - start
    start = time.perf_counter()
    timings["arraydict"] = time.perf_counter() - start
    start = time.perf_counter()
    ds = population.data_structure
    timings["DataStructure + junctions"] = time.perf_counter() - start
    start = time.perf_counter()
    view = ds.to_graph_view()
    timings["graph view"] = time.perf_counter() - start
    segments = len(g.property("anatomy_class"))
    print(f"{segments} segments, {g.nb_vertices()} vertices: {ds.n_nodes()} nodes, {ds.n_edges()} edges "
          f"({len(g.junction_vids())} junctions), incidence {view.incidence.shape}")
    for phase, seconds in timings.items():
        print(f"  {phase:30s} {seconds:7.2f} s")


if __name__ == "__main__":
    main()
