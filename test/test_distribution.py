"""Size distributions (openalea.granap.distribution) and their wiring into the params.

The default (no distribution, or 'normal') must reproduce the plain
``rng.normal`` draw exactly — same values, same RNG consumption — because every
existing preset and golden census depends on it.  Every other family must be
moment-matched to the (mean, sd) it is given.
"""

import os
import sys

import numpy as np
import pytest
from pydantic import ValidationError

sys.path.append(os.path.abspath(".."))

from openalea.granap.distribution import Distribution, as_distribution, draw
from openalea.granap.geometry_collection import GeometryProcessor
from openalea.granap.input_data import OrganInputData, VascularBundleParams
from openalea.granap.root_class import RootAnatomy

MEAN, SD = 0.035, 0.010
N = 100_000


def _samples(dist, n=N, seed=0):
    rng = np.random.default_rng(seed)
    return np.array([draw(rng, MEAN, SD, dist=dist) for _ in range(n)])


# -- the default is the old draw, exactly ---------------------------------------

@pytest.mark.parametrize("dist", [None, {"family": "normal"}, Distribution()])
def test_normal_reproduces_the_plain_draw_and_rng_state(dist):
    a, b = np.random.default_rng(7), np.random.default_rng(7)
    lo = 0.02
    got = [draw(a, MEAN, SD, lo, None, dist) for _ in range(50)]
    ref = [float(np.clip(b.normal(MEAN, SD), lo, np.inf)) for _ in range(50)]
    assert got == ref
    assert a.random() == b.random(), "the normal path must consume the RNG identically"


# -- moment matching ------------------------------------------------------------

MEASURED = [0.021, 0.024, 0.026, 0.027, 0.029, 0.031, 0.033, 0.036, 0.041, 0.052]


@pytest.mark.parametrize("family", ["normal", "lognormal", "gamma", "logistic", "uniform", "empirical"])
def test_every_family_matches_the_requested_mean_and_sd(family):
    spec = {"family": family, "values": MEASURED} if family == "empirical" else {"family": family}
    x = _samples(spec)
    assert x.mean() == pytest.approx(MEAN, rel=0.01)
    assert x.std() == pytest.approx(SD, rel=0.02)


def test_shapes_differ_where_they_should():
    def skew(x):
        return float(((x - x.mean()) ** 3).mean() / x.std() ** 3)

    def excess_kurtosis(x):
        return float(((x - x.mean()) ** 4).mean() / x.std() ** 4 - 3.0)

    normal = _samples(None)
    assert abs(skew(normal)) < 0.05
    for family in ("lognormal", "gamma"):
        assert skew(_samples({"family": family})) > 0.4, f"{family} should be right-skewed"
    logistic = _samples({"family": "logistic"})
    assert abs(skew(logistic)) < 0.05 and excess_kurtosis(logistic) > 0.8   # 1.2 in theory
    uniform = _samples({"family": "uniform"})
    half = SD * np.sqrt(3.0)
    assert uniform.min() >= MEAN - half and uniform.max() <= MEAN + half


def test_clipping_and_zero_sd():
    rng = np.random.default_rng(0)
    for family in ("lognormal", "gamma", "logistic", "uniform"):
        assert draw(rng, MEAN, 0.0, dist={"family": family}) == MEAN
        x = [draw(rng, MEAN, SD, 0.03, 0.04, {"family": family}) for _ in range(500)]
        assert min(x) >= 0.03 and max(x) <= 0.04


def test_empirical_keeps_the_measured_shape_not_its_scale():
    # the measured values are in other units (x1000): only their shape is used
    x = _samples({"family": "empirical", "values": [v * 1000 for v in MEASURED]})
    assert x.mean() == pytest.approx(MEAN, rel=0.01)
    assert set(np.round((x - MEAN) / SD, 9)) <= set(
        np.round((np.array(MEASURED) - np.mean(MEASURED)) / np.std(MEASURED), 9))


# -- validation -----------------------------------------------------------------

def test_validation():
    with pytest.raises(ValidationError):
        Distribution(family="weibull")
    with pytest.raises(ValidationError):
        Distribution(family="empirical", values=[0.03])
    with pytest.raises(ValidationError):
        Distribution(family="empirical", values=[0.03, 0.03])
    with pytest.raises(ValueError):
        draw(np.random.default_rng(0), -1.0, SD, dist={"family": "lognormal"})
    assert as_distribution(None) is None
    assert as_distribution({"family": "gamma"}) == Distribution(family="gamma")


# -- wiring into the params and the generators ----------------------------------

def test_params_carry_the_distribution_as_a_dict():
    bp = VascularBundleParams(metaxylem_diameter_distribution={"family": "lognormal"})
    assert isinstance(bp.metaxylem_diameter_distribution, Distribution)
    assert bp.model_dump()["metaxylem_diameter_distribution"] == {"family": "lognormal", "values": []}
    # every preset defaults to None (= normal), so the emitted dicts stay plain
    for name in ("for_root", "for_dicot_root", "for_dicot_stem", "for_monocot_stem", "for_dicot_leaf"):
        dists = [v for block in getattr(OrganInputData, name)().to_dict_list()
                 for k, v in block.items() if k.endswith("_distribution")]
        assert dists and all(v is None for v in dists), name


def test_pack_circles_uses_the_distribution():
    from shapely.geometry import Point
    zone = Point(0, 0).buffer(1.0)
    kw = dict(proportion=0.5, direction=None, diameter_max=0.12, diameter_min=0.02,
              diameter_sd=0.03, gradient_function="normal")
    ref = GeometryProcessor.pack_circles(zone, rng=np.random.default_rng(0), **kw)
    same = GeometryProcessor.pack_circles(zone, rng=np.random.default_rng(0),
                                          distribution={"family": "normal"}, **kw)
    other = GeometryProcessor.pack_circles(zone, rng=np.random.default_rng(0),
                                           distribution={"family": "uniform"}, **kw)
    assert same == ref
    assert other != ref


def test_root_vessels_follow_the_distribution():
    def vessel_areas(dist):
        data = OrganInputData.for_root()
        if dist:
            data.set_value("xylem", "vessel_diameter_distribution", dist)
        root = RootAnatomy(data, seed=0)
        gdf = root.generate_cells()
        return sorted(round(g.area, 10) for t, g in zip(gdf["type"], gdf.geometry) if t == "metaxylem")

    default = vessel_areas(None)
    assert vessel_areas({"family": "normal"}) == default
    assert vessel_areas({"family": "uniform"}) != default
