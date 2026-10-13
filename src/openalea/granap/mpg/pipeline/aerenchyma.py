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

from openalea.granap.organ_class import Organ

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

    plan = [None]                    # B2b: the fill order, computed at the first attempt, the same for all

    def attempt(internal_target: float) -> float:
        organ.all_cells.cells = _copies(snapshot)
        if _uses_organs(organ, "add_aerenchyma"):
            plan[0] = add_aerenchyma(organ, prop_override=internal_target, plan=plan[0])
        else:
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


def fill_plan(organ):
    """
    What GRANAP's add_aerenchyma computes before filling, which depends only on the cells it starts from: the
    candidates and the areas, and each file's fill order (quadrant buckets, ordered as a connected radial channel or by
    distance), as positions in organ.all_cells.cells. None where add_aerenchyma would return without filling.
    """
    tissue = organ.aerenchyma_params.get("tissue")
    tissues = list(tissue) if isinstance(tissue, (list, tuple)) else [tissue]
    n_files = int(organ.aerenchyma_params.get("n_files", 1))
    aerenchyma_type = int(organ.aerenchyma_params.get("aerenchyma_type", 1))
    start_angle = organ._aerenchyma_start_angle

    def cell_quadrant(cell):
        cell_angle = np.arctan2(cell.y, cell.x) % (2 * np.pi)
        rel = (cell_angle - start_angle) % (2 * np.pi)
        return int(rel / (2 * np.pi / n_files)) % n_files

    tissue_cells = [c for t in tissues for c in organ.all_cells.get_cells_by_type(t)]
    if not tissue_cells:
        return None

    # id_layer grows radially inward, so the single global max is the
    # innermost ring of the combined region: listed tissues act as one
    # contiguous band and only that inner boundary ring is preserved.
    max_tissue_layer = max(c.id_layer for c in tissue_cells)
    candidates = [c for c in tissue_cells if c.id_layer < max_tissue_layer]
    candidates.extend(organ.all_cells.get_cells_by_type("air space"))

    if not candidates:
        return None

    total_tissue_area = sum(c.polygon.area for c in tissue_cells if c.polygon is not None)
    total_air_area = sum(c.polygon.area for c in organ.all_cells.get_cells_by_type("air space") if c.polygon is not None)
    max_possible_area = sum(c.polygon.area for c in candidates if c.polygon is not None)

    quadrant_buckets = [[] for _ in range(n_files)]
    for c in candidates:
        quadrant_buckets[cell_quadrant(c)].append(c)

    if aerenchyma_type == 1:
        # Order each file as a *connected* radial channel rather than by raw angular distance.
        for q, bucket in enumerate(quadrant_buckets):
            central_angle = (start_angle + (q + 0.5) * 2 * np.pi / n_files) % (2 * np.pi)
            quadrant_buckets[q] = organ._connected_radial_order(bucket, central_angle)
    elif aerenchyma_type == 2:
        for q, bucket in enumerate(quadrant_buckets):
            if not bucket:
                continue
            central_angle = (start_angle + (q + 0.5) * 2 * np.pi / n_files) % (2 * np.pi)
            def _ang_dist_seed(cell, ca=central_angle):
                a = np.arctan2(cell.y, cell.x) % (2 * np.pi)
                d = abs(a - ca)
                return min(d, 2 * np.pi - d)
            seed = min(bucket, key=_ang_dist_seed)
            bucket.sort(key=lambda c, s=seed: np.hypot(c.x - s.x, c.y - s.y))

    position = {id(c): k for k, c in enumerate(organ.all_cells.cells)}
    return dict(tissues=tissues, n_files=n_files, total_tissue_area=total_tissue_area,
                total_air_area=total_air_area, max_possible_area=max_possible_area,
                buckets=[[position[id(c)] for c in bucket] for bucket in quadrant_buckets])


def add_aerenchyma(organ, prop_override=None, plan=None):
    """
    GRANAP's Organ.add_aerenchyma (organ_class.py:935), its preparation (fill_plan) given or computed. Returns the
    plan, for the next attempt of a calibration starting from the same cells.
    """
    aerenchyma_prop = (
        prop_override if prop_override is not None
        else organ.aerenchyma_params.get("aerenchyma_proportion", 0)
    )
    if not aerenchyma_prop:
        return plan

    n_files = int(organ.aerenchyma_params.get("n_files", 1))
    organ._aerenchyma_n_files = n_files
    if getattr(organ, "_aerenchyma_start_angle", None) is None:
        organ._aerenchyma_start_angle = float(organ.rng.uniform(0, 2 * np.pi))

    if aerenchyma_prop > 1:
        print("Aerenchyma proportion is greater than 1, setting it to 1")
        aerenchyma_prop = 1

    if plan is None:
        plan = fill_plan(organ)
    if plan is None:
        return plan
    total_tissue_area, total_air_area = plan["total_tissue_area"], plan["total_air_area"]
    max_possible_area = plan["max_possible_area"]

    target_aerenchyma_area = (total_tissue_area + total_air_area) * aerenchyma_prop

    if target_aerenchyma_area > max_possible_area:
        log.warning(
            "Requested aerenchyma proportion %.2f requires %.4f area but only %.4f is available; clamping.",
            aerenchyma_prop, target_aerenchyma_area, max_possible_area,
        )
        aerenchyma_prop = max_possible_area / (total_tissue_area + total_air_area)
        target_aerenchyma_area = max_possible_area

    log.debug("Targeted aerenchyma proportion: %.3f", target_aerenchyma_area / (total_tissue_area + total_air_area))

    target_per_quadrant = (target_aerenchyma_area - total_air_area) / organ._aerenchyma_target_denominator(n_files)

    cells = organ.all_cells.cells
    quadrant_buckets = [[cells[k] for k in bucket] for bucket in plan["buckets"]]
    quadrant_area = [0.0] * n_files
    quadrant_idx = [0] * n_files

    changed = True
    while changed:
        changed = False
        for q in range(n_files):
            if quadrant_area[q] >= target_per_quadrant:
                continue
            bucket = quadrant_buckets[q]
            while quadrant_idx[q] < len(bucket):
                cell = bucket[quadrant_idx[q]]
                quadrant_idx[q] += 1
                if cell.type != "air space" and cell.polygon is not None:
                    cell.type = "air space"
                    quadrant_area[q] += cell.polygon.area
                    changed = True
                    break

    if log.isEnabledFor(logging.DEBUG):
        tissues = plan["tissues"]
        total_tissue_area = sum(
            c.polygon.area
            for t in tissues
            for c in organ.all_cells.get_cells_by_type(t)
            if c.polygon is not None
        )
        total_air_area = sum(c.polygon.area for c in organ.all_cells.get_cells_by_type("air space") if c.polygon is not None)
        log.debug("Actual aerenchyma proportion: %.3f", total_air_area / (total_tissue_area + total_air_area))
    return plan


def _uses_organs(organ, name) -> bool:
    """Whether *organ*'s method *name* is Organ's (not an organ class's own)."""
    return next(c.__dict__[name] for c in type(organ).__mro__ if name in c.__dict__
                and c.__name__ != "MPGBacked") is Organ.__dict__[name]
