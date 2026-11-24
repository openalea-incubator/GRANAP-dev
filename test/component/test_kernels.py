"""
GRANAP's numba kernels (openalea.granap.metafspm.kernels) give the bits of the Python they replace.
"""
import numpy as np

from openalea.granap.metafspm import kernels


def test_python_sums_are_pythons_sum_of_each_segment():
    random = np.random.default_rng(3)
    lengths = random.integers(0, 6, size=400)
    values = random.lognormal(-6., 3., size=int(lengths.sum())) * random.choice([-1., 1.], size=int(lengths.sum()))
    values[::17] = 1e16                                    # large and small terms: the compensation matters
    offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
    expected = [sum(values[offsets[k]:offsets[k + 1]].tolist()) for k in range(lengths.size)]
    assert kernels.python_sums(values, offsets).tolist() == expected


def test_warm_up_runs_every_kernel():
    kernels.warm_up()
