"""Monocot root series defined entirely by smooth functions of length.

One cross-section is written by hand — the collar — and the other eleven are
produced by evolving it along the root. Nothing below is a per-section value:
every quantity that changes is a ``length -> value`` schedule, collected in
:data:`RELATIONS` so the same functions that drive the model also draw the
figure that explains it.

Orientation follows RootSeries' own convention: the **collar is the high
length** (1000 mm) and the **apex is 0 mm**, because ``terminations`` are
"present for length >= stop", i.e. present at the collar and gone toward the
apex. ``d(L)`` below converts to the more natural "distance travelled from the
collar", which is what the 30 cm and 40 cm landmarks refer to.

What the schedules say
    metaxylem   8 at the collar, decaying exponentially to 2 by 50 cm and 1 by
                ~59 cm, and under half a vessel by the apex.  The drop is 5
                fusions plus 2 terminations = 7; the terminations are done inside
                the first 30 cm, the fusions run on past it.
    vessel size grows as the count falls — the survivors carry the same water
    stele/root  barely change for the first 40 cm, then taper
    protoxylem  12 poles at the collar falling to 4 at the apex, on the stele's
                own taper rather than the metaxylem count — so the poles thin out
                with the stele that carries them (a third, not a tenth) and are
                still there once the last metaxylem has gone

The apex carries **no metaxylem at all** but keeps its protoxylem and phloem.
That is expressible because ``xylem.n_protoxylem`` is a count in its own right:
poles are specified early at the apex while metaxylem differentiate later, so
there is nothing to scale them off — see the note in
``root_monocot_class.fit_phloem_protoxylem_elements``.
"""

import os
import sys

import numpy as np

sys.path.append(os.path.abspath(".."))

from openalea.granap.input_data import OrganInputData
from openalea.granap.root_series import RootSeries

SEED = 0
N_LEVELS = 30
COLLAR = 1000.0            # mm — the hand-written section
APEX = 0.0
N_COLS = 4

META_COLLAR = 8            # metaxylem in the base section
N_TERMINATE = 2            # vessels that stop rather than fuse
TERMINATIONS = [900.0, 750.0]   # ...at 10 cm and 25 cm from the collar
TERMINATION_ZONE = 300.0   # both terminations are done within the first 30 cm
META_DECAY = 0.00235       # 1/mm — exponential decay rate of the fused count
TIP_MID = 960.0            # mm — where the last vessel goes, and how sharply
TIP_WIDTH = 40.0
POLES_COLLAR = 12          # protoxylem poles in the base section
POLES_APEX = 4             # ...and at the apex, where no metaxylem are left
TAPER_MID = 650.0          # mm — where the root/stele taper is steepest
TAPER_WIDTH = 130.0        # mm — how sharp that transition is


def d(length: float) -> float:
    """Distance travelled from the collar (mm) — what the landmarks refer to.

    Note this is *not* the axis the figures use.  Length itself is measured from
    the apex (0) to the collar (1000), which is how the series is presented; the
    schedules are simply easier to state as "so far from the collar", since that
    is where the root's history starts.
    """
    return COLLAR - float(length)


def _decay(x: float, y0: float, y_inf: float, rate: float) -> float:
    """Exponential relaxation from y0 toward the limit y_inf.

    Smooth everywhere and monotone, with no corner to explain — it approaches
    its limit rather than arriving at one and stopping.
    """
    return y_inf + (y0 - y_inf) * np.exp(-rate * x)


def _logistic(x: float, x0: float, k: float) -> float:
    """1 well before x0, 0 well after, turning over a width of about k.

    The right shape for anything that holds, then gives way: nearly flat at
    first, steepest at x0, flattening again toward its lower limit.
    """
    return 1.0 / (1.0 + np.exp((x - x0) / k))


# --- the schedules -------------------------------------------------------
FUSED_COLLAR = META_COLLAR - N_TERMINATE      # 6 fusing classes + 2 terminators = 8


def fused_groups(length: float) -> float:
    """Fused groups as a CONTINUOUS quantity — this is the actual schedule.

    An exponential decay from 6, times a logistic that closes the last vessel off
    at the tip. Both are smooth, so their product is too — no segments, no corner.

    The decay alone cannot do it: a rate that puts the 2 -> 1 step at ~59 cm from
    the collar still leaves 0.57 of a vessel at the apex, which rounds to one.
    Steepening it until the apex reaches zero drags that step back to 53 cm. The
    tip logistic separates the two, so the count holds at 1 down the length of the
    root and then genuinely ends.
    """
    x = d(length)
    return (_decay(x, FUSED_COLLAR, 0.0, META_DECAY)
            * _logistic(x, TIP_MID, TIP_WIDTH))


def n_fused(length: float) -> int:
    """...and the whole vessels a section actually gets. Rounding happens here,
    at the point of use, and nowhere upstream: anything derived from an already
    rounded count inherits its staircase."""
    return int(round(fused_groups(length)))


def n_active_terminators(length: float) -> int:
    return sum(1 for stop in TERMINATIONS if length >= stop)


def metaxylem(length: float) -> float:
    """Metaxylem as a continuous quantity: the smooth fused-group curve plus the
    terminating vessels still running. The two terminations are events, so they
    are the only discontinuities in it — the 8 -> 1 fall itself is a curve."""
    return fused_groups(length) + n_active_terminators(length)


def n_metaxylem(length: float) -> int:
    """What you would count in the section."""
    return n_fused(length) + n_active_terminators(length)


def vessel_radius(length: float) -> float:
    """Base single-vessel radius, relaxing toward its limit on the *same* rate as
    the count decays: the vessels widen exactly as fast as they are lost. Fused
    vessels grow further still via ``area_retention``."""
    return _decay(d(length), 0.013, 0.026, META_DECAY)


def _taper(length: float) -> float:
    """1 near the collar, 0 near the apex, steepest at TAPER_MID — the shared
    shape for the root and the stele. Barely moves over the first 40 cm, which
    is the whole point of using a sigmoid here rather than a straight line."""
    return _logistic(d(length), TAPER_MID, TAPER_WIDTH)


def _taper01(length: float) -> float:
    """``_taper`` renormalised to hit exactly 1 at the collar and 0 at the apex.

    The raw sigmoid only approaches its asymptotes (0.993 at the collar, 0.063
    at the apex), which is fine where it is used as a *shape* to scale a
    dimension by, but not where both endpoints are values the model states
    outright — there, the section you asked for is the one you should get.
    """
    lo, hi = _taper(APEX), _taper(COLLAR)
    return (_taper(length) - lo) / (hi - lo)


def stele_radius(length: float) -> float:
    """The stele follows the root, because the root is built out from it."""
    return 0.125 + 0.075 * _taper(length)


def cortex_cell_diameter(length: float) -> float:
    return 0.038 + 0.014 * _taper(length)


def cortex_layers(length: float) -> float:
    """Cortex layers as a continuous quantity — the taper sigmoid, scaled."""
    return 4.0 + 1.0 * _taper(length)


def cortex_n_layers(length: float) -> int:
    """...read to the nearest whole layer, so the one step it takes lands where
    the taper is steepest rather than at an arbitrary length."""
    return int(round(cortex_layers(length)))


def protoxylem_poles(length: float) -> float:
    """Protoxylem poles as a continuous quantity.

    Follows the *stele's* taper, not the metaxylem count: poles are laid down
    early at the apex and are a property of the stele, while metaxylem
    differentiate later and centrally. Scaling them off the metaxylem — as a
    proto:meta ratio does — makes the count collapse toward the apex and, worse,
    move non-monotonically, since it multiplies a rising ratio by a falling
    staircase. Taking them off the same sigmoid as ``stele_radius`` instead
    keeps poles roughly proportional to the stele circumference they sit on.
    """
    return POLES_APEX + (POLES_COLLAR - POLES_APEX) * _taper01(length)


def n_protoxylem(length: float) -> int:
    """...and the whole poles a section actually gets."""
    return int(round(protoxylem_poles(length)))


# label -> (function, y-axis label).  The relations figure walks this, so the
# picture cannot drift from the model.
# label -> (continuous curve, y-axis label, the discrete value a section gets or
# None).  The relations figure draws the curve as a line and the discrete value
# as the dots, so you can see the rule and what each section rounded to.
RELATIONS = {
    "metaxylem count":        (metaxylem,            "vessels", n_metaxylem),
    "fused groups":           (fused_groups,         "groups",  n_fused),
    "terminating vessels":    (n_active_terminators, "vessels", None),
    "vessel radius":          (vessel_radius,        "mm",      None),
    "stele radius":           (stele_radius,         "mm",      None),
    "cortex cell diameter":   (cortex_cell_diameter, "mm",      None),
    "cortex layers":          (cortex_layers,        "layers",  cortex_n_layers),
    "protoxylem poles":       (protoxylem_poles,     "poles",   n_protoxylem),
}


def build_collar() -> OrganInputData:
    """The one section written by hand. Everything else is this, evolved."""
    c = OrganInputData.for_root()

    c.set_value("stele", "cell_diameter",        0.012)
    c.set_value("stele", "cell_diameter_center", 0.018)

    c.set_value("xylem", "xylem_shape",               "default")
    c.set_value("xylem", "n_vascular_bundles",        META_COLLAR)
    c.set_value("xylem", "vessel_diameter",           0.036)
    c.set_value("xylem", "vessel_diameter_sd",        0.004)
    c.set_value("xylem", "n_protoxylem",              POLES_COLLAR)
    c.set_value("xylem", "protoxylem_diameter",       0.014)
    c.set_value("xylem", "protoxylem_cluster_width",  0.020)
    c.set_value("xylem", "protoxylem_cluster_height", 0.020)

    c.set_value("phloem", "sieve_diameter", 0.011)
    c.set_value("phloem", "cluster_width",  0.014)
    c.set_value("phloem", "cluster_height", 0.014)

    c.set_value("endodermis", "cell_diameter", 0.015)
    c.set_value("endodermis", "cell_width",    0.026)
    c.set_value("pericycle",  "cell_diameter", 0.014)
    c.set_value("pericycle",  "cell_width",    0.012)

    c.set_value("cortex", "cell_diameter", cortex_cell_diameter(COLLAR))
    c.set_value("cortex", "cell_width",    0.050)
    c.set_value("cortex", "n_layers",      cortex_n_layers(COLLAR))

    c.set_value("exodermis", "cell_diameter", 0.026)
    c.set_value("exodermis", "cell_width",    0.032)
    c.set_value("epidermis", "cell_diameter", 0.018)
    c.set_value("epidermis", "cell_width",    0.030)

    c.set_value("inter_cellular_spaces", "smoothness", 0.05)
    c.set_value("inter_cellular_spaces", "tissue", ["cortex"])
    return c


def build_series() -> RootSeries:
    return RootSeries(
        build_collar(),
        start=APEX, end=COLLAR, samples=N_LEVELS,
        n_fused=n_fused,
        terminations=TERMINATIONS,
        vessel_radius=vessel_radius,
        stele_radius=stele_radius,
        area_retention=0.30,
        # fusion_length=120 pulls a fusing pair together well before the merge,
        # but the metaxylem count decays fast enough near the collar that the
        # very first fusion event falls within 120mm of it, so that pre-fusion
        # "approach" is already tugging two vessels off the collar's own even
        # ring (squeezing them to ~22 deg apart instead of 45). 30 is the
        # largest value (of 120/60/30/10/0, checked directly against the
        # collar's own vessel angles) that still leaves the collar exactly
        # evenly spaced.
        #
        # migration_length is a separate knob - it paces a vessel's glide to
        # its *new* slot after a count change, not the pre-fusion approach -
        # and does not need to shrink with fusion_length: at 30 (matched to
        # fusion_length) a migration completes within one ~34.5mm sample gap
        # (1000mm / 29 samples), so the 30-sample render only ever catches its
        # before/after endpoints - a vessel appears to teleport instead of
        # glide (confirmed on tracked primordial 2: a 0.185 jump, ~11x its own
        # radius). Keeping migration_length at 120 spreads that same glide
        # over several samples (the same jump drops to 0.062, ~3.7x the
        # radius) while the collar - governed by fusion_length alone - stays
        # perfectly even.
        migration_length=120.0,
        fusion_length=30.0,
        param_schedules={
            "cortex.cell_diameter":    cortex_cell_diameter,
            "cortex.n_layers":         cortex_n_layers,
            "xylem.n_protoxylem":      n_protoxylem,
        },
        seed=SEED,
    )


def main(show=True):
    res = build_series().generate()
    res.plot(cols=N_COLS,
             suptitle="Monocot root series — every section a function of length "
                      "(collar 100 cm → apex 0 cm)",
             show=show)
    return res


if __name__ == "__main__":
    main()
