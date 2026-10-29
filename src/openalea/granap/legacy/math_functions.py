"""
Normalized shape functions and a rescale wrapper for GRANAP gradients.

Convention: every shape function maps t in [0, inf) -> (0, 1] with f(0) = 1.
Use `rescale` to stretch any shape function to a physical range [lo, hi]:

    scaled(0) = hi  (centre)
    scaled(inf) = lo  (edge)
"""

from __future__ import annotations

from typing import Callable


def five_pl(t: float, c: float = 0.5, b: float = 3.0, m: float = 1.0) -> float:
    """Normalized 5-parameter logistic: f(0) = 1, f(inf) -> 0.

    Parameters
    ----------
    t : float
        Normalized position >= 0 (e.g. radial distance, 0 = centre).
    c : float
        Inflection point — position of the steepest descent.
    b : float
        Hill coefficient — sharpness of the transition.
    m : float
        Asymmetry exponent.
    """
    if t <= 0.0:
        return 1.0
    return 1.0 / (1.0 + (t / c) ** b) ** m


def linear(t: float, **_) -> float:
    """Normalized linear decrease: f(0) = 1, f(1) = 0, clamped outside [0, 1].

    Parameters
    ----------
    t : float
        Normalized position in [0, 1].
    """
    return float(max(0.0, 1.0 - t))


GRADIENT_FUNCTIONS: dict[str, Callable[..., float]] = {
    "five_pl": five_pl,
    "linear":  linear,
}


def rescale(
    func: Callable[..., float],
    lo: float,
    hi: float,
    **shape_kwargs,
) -> Callable[[float], float]:
    """Wrap a normalized shape function so its output spans [lo, hi].

    Parameters
    ----------
    func :
        Normalized shape function with signature ``func(t, **shape_kwargs) -> float``.
    lo :
        Output value at the outer boundary (where func -> 0).
    hi :
        Output value at the centre (where func = 1, t = 0).
    **shape_kwargs :
        Extra keyword arguments forwarded to *func* unchanged.
    """
    if lo == hi:
        def _flat(t: float) -> float:
            return float(lo)
        return _flat

    def _scaled(t: float) -> float:
        return float(lo + (hi - lo) * func(t, **shape_kwargs))

    return _scaled
