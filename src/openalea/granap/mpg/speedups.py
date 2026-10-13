"""
Exact speed-ups of GRANAP's pipeline for the MPG-backed organs (devplan_metafspm.md, P4; your answers: exact only,
in mpg/). The outputs are identical, bit for bit; the comparison folder checks it.

    E1  topology cache: CellGenerator._build_topology answered from memory when called again on identical input
        (the same polygons, coordinate for coordinate, the same ids and protected ids). The aerenchyma calibration
        simplifies the same cells at every attempt (9 of the 11 calls on a dicot root).

cached_topology(organ, ...)       the cache, held by the organ (MPGBacked.section_topology calls it); the topology
                                  computed by the converted snap (mpg/pipeline/snap.py, B5) on the converted pipeline
with exact_speedups(organ): ...   for the block, the class method answers through the organ (GRANAP's own modules)
"""
import hashlib
from contextlib import contextmanager

import shapely

from openalea.granap.generate_cell import CellGenerator

_DESCRIPTOR = CellGenerator.__dict__["_build_topology"]


def _key(polys, cell_ids, protect_ids) -> tuple:
    digest = hashlib.blake2b(digest_size=16)
    for wkb in shapely.to_wkb(list(polys), hex=False):      # None for missing polygons
        digest.update(wkb if wkb is not None else b"-")
    return tuple(cell_ids), frozenset(protect_ids or ()), digest.digest()


def cached_topology(organ, polys, cell_ids, protect_ids=None, cache_size: int = 4):
    """
    CellGenerator._build_topology for *organ*, answered from the organ's cache when called again on identical input
    (E1); computed by the converted snap (B5) on the converted pipeline, else by GRANAP's.
    """
    cache = organ.__dict__.setdefault("_topology_cache", {})
    stats = organ.__dict__.setdefault("speedup_stats", {"topology_hits": 0, "topology_calls": 0})
    stats["topology_calls"] += 1
    key = _key(polys, cell_ids, protect_ids)
    if key in cache:
        stats["topology_hits"] += 1
        return cache[key]
    if getattr(organ, "converted_pipeline", False):
        from openalea.granap.mpg.pipeline.snap import build_topology as compute
    else:
        compute = _DESCRIPTOR.__func__
    result = compute(polys, cell_ids, protect_ids=protect_ids)
    if len(cache) >= cache_size:
        cache.pop(next(iter(cache)))
    cache[key] = result
    return result


@contextmanager
def exact_speedups(organ):
    """
    Within the block, CellGenerator._build_topology answers through the organ (MPGBacked.section_topology: E1, B5).
    Only GRANAP's own modules still call the class method (the organ-specific stages, until P5.6); the converted code
    calls the organ directly (C2).
    """
    current = CellGenerator.__dict__["_build_topology"]
    CellGenerator._build_topology = staticmethod(organ.section_topology)
    try:
        yield
    finally:
        CellGenerator._build_topology = current
        organ.__dict__.pop("_topology_cache", None)
