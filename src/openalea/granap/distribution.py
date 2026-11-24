"""Shape of a randomly drawn size (vessel, protoxylem, sieve-element diameter...).

Every size parameter keeps its own scalar fields — ``<x>_diameter`` (mean or
gradient target), ``<x>_diameter_sd``, ``<x>_diameter_min`` — and gains one
optional ``<x>_diameter_distribution``: a :class:`Distribution` that says only
**which shape** the per-element jitter follows.  Every family is moment-matched
to the (mean, sd) the caller passes, in linear space, so ``mean`` and ``sd``
mean the same thing whatever the shape, and nothing downstream needs to know
which family produced a value.

Left unset (``None``) or ``family="normal"``, :func:`draw` makes exactly the
``rng.normal(mean, sd)`` call the code made before, so existing outputs are
unchanged bit for bit.

    vessel_diameter=0.035, vessel_diameter_sd=0.01,
    vessel_diameter_distribution={"family": "lognormal"}

``empirical`` resamples measured values: they are standardised (their own mean
and sd removed) and rescaled to the requested (mean, sd), so a measured sample
gives the *shape* while the parameters keep giving the size.
"""

from typing import Any, List, Literal, Optional

import numpy as np
from pydantic import BaseModel, Field, model_validator

FAMILIES = ("normal", "lognormal", "gamma", "logistic", "uniform", "empirical")


class Distribution(BaseModel):
    """Shape of the per-element size jitter (see the module docstring)."""
    model_config = {"validate_assignment": True}

    family: Literal["normal", "lognormal", "gamma", "logistic", "uniform", "empirical"] = Field(
        default="normal", title="Distribution Family",
        description="Shape of the per-element size jitter, moment-matched to the size's mean "
                    "and sd: 'normal' (default); 'lognormal' / 'gamma' (right-skewed, positive); "
                    "'logistic' (symmetric, heavier tails); 'uniform' (flat, mean +/- sd*sqrt(3)); "
                    "'empirical' (resample the standardised measured `values`).")
    values: List[float] = Field(
        default_factory=list, title="Measured Values",
        description="family='empirical' only: measured sizes (any unit) whose standardised "
                    "shape is resampled; at least 2 distinct values.")

    @model_validator(mode="after")
    def _check_values(self) -> "Distribution":
        if self.family == "empirical":
            v = np.asarray(self.values, dtype=float)
            if v.size < 2 or float(np.std(v)) == 0.0:
                raise ValueError("family='empirical' needs at least 2 distinct values")
        return self

    def sample(self, rng, mean: float, sd: float,
               lo: Optional[float] = None, hi: Optional[float] = None) -> float:
        """One value with this shape, mean ``mean`` and standard deviation ``sd``,
        clipped to [lo, hi]."""
        return _sample(self.family, self.values, rng, mean, sd, lo, hi)


def as_distribution(spec: Any) -> Optional[Distribution]:
    """``None`` | dict | :class:`Distribution` -> :class:`Distribution` or ``None``.

    Params reach the generators as plain dicts (``model_dump``), so a distribution
    arrives as a dict there."""
    if spec is None or isinstance(spec, Distribution):
        return spec
    if isinstance(spec, dict):
        return Distribution(**spec)
    raise TypeError(f"not a distribution spec: {spec!r}")


def draw(rng, mean: float, sd: float, lo: Optional[float] = None,
         hi: Optional[float] = None, dist: Any = None) -> float:
    """One size, ``float(np.clip(x, lo, hi))``, with ``x`` from ``dist``'s family.

    ``dist`` None or normal -> ``rng.normal(mean, sd)``: the exact call (and RNG
    consumption) of the code this replaces."""
    d = as_distribution(dist)
    family = "normal" if d is None else d.family
    values = [] if d is None else d.values
    return _sample(family, values, rng, mean, sd, lo, hi)


def _sample(family, values, rng, mean, sd, lo, hi) -> float:
    if family == "normal":
        x = rng.normal(mean, sd)
    elif sd <= 0.0:
        x = mean
    elif family == "lognormal":
        if mean <= 0.0:
            raise ValueError("lognormal needs a positive mean")
        sigma2 = np.log1p((sd / mean) ** 2)
        x = rng.lognormal(np.log(mean) - sigma2 / 2.0, np.sqrt(sigma2))
    elif family == "gamma":
        if mean <= 0.0:
            raise ValueError("gamma needs a positive mean")
        theta = sd * sd / mean
        x = rng.gamma(mean / theta, theta)
    elif family == "logistic":
        x = rng.logistic(mean, sd * np.sqrt(3.0) / np.pi)
    elif family == "uniform":
        half = sd * np.sqrt(3.0)
        x = rng.uniform(mean - half, mean + half)
    elif family == "empirical":
        v = np.asarray(values, dtype=float)
        z = (v - v.mean()) / v.std()
        x = mean + sd * float(z[rng.integers(z.size)])
    else:
        raise ValueError(f"unknown distribution family: {family!r}")
    return float(np.clip(x, -np.inf if lo is None else lo, np.inf if hi is None else hi))
