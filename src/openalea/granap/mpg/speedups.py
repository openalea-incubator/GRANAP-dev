"""
Exact speed-ups of GRANAP's pipeline for the MPG-backed organs (devplan_metafspm.md, P4; your answers: exact only,
in mpg/). The outputs are identical, bit for bit; the comparison folder checks it.

    E1  topology cache: CellGenerator._build_topology answered from memory when called again on identical input
        (the same polygons, coordinate for coordinate, the same ids and protected ids). The aerenchyma calibration
        simplifies the same cells at every attempt (9 of the 11 calls on a dicot root).

with exact_speedups(organ): ...   active for the block (MPGBacked.generate_cells, when organ.exact_speedups)
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


@contextmanager
def exact_speedups(organ, cache_size: int = 4):
    """Within the block, repeated _build_topology calls on identical input return the first call's result."""
    current = CellGenerator.__dict__["_build_topology"]
    compute = current.__func__ if isinstance(current, staticmethod) else current
    cache = {}
    stats = organ.__dict__.setdefault("speedup_stats", {"topology_hits": 0, "topology_calls": 0})

    def build(polys, cell_ids, protect_ids=None):
        stats["topology_calls"] += 1
        key = _key(polys, cell_ids, protect_ids)
        if key in cache:
            stats["topology_hits"] += 1
            return cache[key]
        result = compute(polys, cell_ids, protect_ids=protect_ids)
        if len(cache) >= cache_size:
            cache.pop(next(iter(cache)))
        cache[key] = result
        return result

    CellGenerator._build_topology = staticmethod(build)
    try:
        yield
    finally:
        CellGenerator._build_topology = current
