"""
GRANAP's numba kernels, in one place (metafspm's conventions.md, "Performance rules"): the loops over a scene's
sections that numpy does not express, each @njit(cache=True, nogil=True), sequential (their work is small: a step's
sections). warm_up() runs every one on tiny inputs, which fills numba's on-disk cache; metafspm's warm_up runs it
with every port's (python -m openalea.metafspm warm_up).
"""
import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def python_sums(values, offsets):
    """
    The sum of each segment values[offsets[k]:offsets[k + 1]] as Python's sum() of its floats gives it (CPython
    3.12+: from 0, Neumaier's compensated summation, the compensation added at the end when finite and non-zero):
    the same bits as the per-segment sum it replaces. 0 for an empty segment.
    """
    out = np.zeros(offsets.size - 1)
    for k in range(offsets.size - 1):
        start, stop = offsets[k], offsets[k + 1]
        if start == stop:
            continue
        total = 0. + values[start]
        compensation = 0.
        for i in range(start + 1, stop):
            value = values[i]
            added = total + value
            if abs(total) >= abs(value):
                compensation += (total - added) + value
            else:
                compensation += (value - added) + total
            total = added
        if compensation != 0. and np.isfinite(compensation):
            total += compensation
        out[k] = total
    return out


def warm_up() -> None:
    """Every kernel compiled (or loaded from numba's cache) on tiny inputs of the types its callers pass."""
    python_sums(np.array([1., 2., 3.]), np.array([0, 2, 2, 3], dtype=np.int64))
