"""
The aerenchyma calibration, converted (devplan_metafspm.md, P5.3; Q-P5e c): GRANAP's Organ._add_aerenchyma_to_
proportion with the pre-aerenchyma snapshot and each attempt's restore made of shallow copies of the cells instead
of deep copies. Exact: a Cell's attributes are numbers, strings, None or shapely geometries, all immutable, and the
attempts set attributes rather than mutating them, so a shallow copy is as independent as a deep one. The best
attempt's cells are kept and restored at the end, instead of running that attempt again (an attempt depends only on
the snapshot and its target). Every other line is GRANAP's (organ_class.py:652).
"""
import copy
import logging

import numpy as np

log = logging.getLogger("openalea.granap.organ_class")


def _copies(cells):
    """Cells copied one level deep: new Cell objects sharing their (immutable) attribute values."""
    return [copy.copy(c) for c in cells]


def add_aerenchyma_to_proportion(organ, tol: float = 5e-3, max_iter: int = 20):
    """GRANAP's calibration (fill, fuse and smooth a trial target, measure, bisect), on shallow copies."""
    requested = organ.aerenchyma_params.get("aerenchyma_proportion", 0)
    if not requested:
        organ.merge_intercellular_aerenchyma()
        return

    tissue = organ.aerenchyma_params.get("tissue")
    tissues = list(tissue) if isinstance(tissue, (list, tuple)) else [tissue]

    snapshot = _copies(organ.all_cells.cells)
    organ._aerenchyma_start_angle = float(organ.rng.uniform(0, 2 * np.pi))

    def attempt(internal_target: float) -> float:
        organ.all_cells.cells = _copies(snapshot)
        organ.add_aerenchyma(prop_override=internal_target)
        organ.merge_intercellular_aerenchyma()
        return organ._measure_air_proportion(tissues)

    best_target, best_realized = requested, attempt(requested)
    best_err = abs(best_realized - requested)
    best_cells = _copies(organ.all_cells.cells)          # the best attempt's cells, restored at the end
    lo, hi = 0.0, requested
    if best_realized < requested:
        lo, hi = requested, 1.0

    for _ in range(max_iter):
        if best_err <= tol or (hi - lo) < 1e-4:
            break
        mid = 0.5 * (lo + hi)
        realized = attempt(mid)
        err = abs(realized - requested)
        if err < best_err:
            best_target, best_realized, best_err = mid, realized, err
            best_cells = _copies(organ.all_cells.cells)
        if realized > requested:
            hi = mid
        else:
            lo = mid

    organ.all_cells.cells = best_cells           # was attempt(best_target): the same cells, computed again
    log.debug("Calibrated aerenchyma: requested %.3f -> realized %.3f (internal target %.3f)",
              requested, best_realized, best_target)
