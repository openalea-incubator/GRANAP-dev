"""Root developmental series with identity-tracked xylem vessels (ROOT_SERIES_PLAN).

Sample a root's anatomy along its axis, from one physical position to another (lengths
in mm).  A tracked xylem vessel can, along the way:

* **fuse** with its nearest neighbour (by distance) — the count drops and the survivor
  gets bigger;
* **terminate** — a vessel that just stops: a singleton present only for
  ``length >= its stop``, then gone (its id retired, never reused);
* be **created** — a vessel that appears: a singleton present only for
  ``length <= its appear`` point.

Vessel positions come from the **monocot class's own arrangement** for the current count
(``MonocotRootAnatomy.metaxylem_positions``); when the count changes, vessels *migrate*
toward the new class-optimal positions rather than snapping.  Which vessels fuse is
decided by the real distance between their positions, not by id.  Only xylem vessels are
tracked; the rest of the tissue is regenerated per section (the "refit") via
``RootAnatomy.prescribe_vessels``.

Schedules (count, vessel/stele radius, any evolving param) may be given either as a
``(value_at_low_length, value_at_high_length)`` tuple for a simple linear ramp, or as a
``length -> value`` callable for anything else.

Fused size is set by ``area_retention``: a group's area = ``base_area × (1 +
area_retention × (members - 1))`` — 0 = no growth on fusion, 1 = area-conserving (a pair
is √2 wider), in between = a partial gain ("slightly bigger").
"""

import copy
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Point

from openalea.granap.input_data import OrganInputData
from openalea.granap.root_class import RootAnatomy
from openalea.granap.root_monocot_class import MonocotRootAnatomy

# A schedule is a (low, high) linear ramp tuple, or a length->value callable.
Schedule = Union[Tuple[float, float], Callable[[float], float]]


class RootSeriesResult:
    """Per-section results + the vessel track table (with fusion membership)."""

    def __init__(self, sections, track_rows):
        self.sections = sections            # [{length, root, gdf, vessels}]
        self.track_rows = track_rows        # [{track_id, length, x, y, radius, members}]

    @property
    def lengths(self) -> List[float]:
        return [s["length"] for s in self.sections]

    def track(self, track_id: int) -> List[dict]:
        return [r for r in self.track_rows if r["track_id"] == track_id]

    def follow_primordial(self, p: int) -> List[dict]:
        """Every (length, track_id, members) the primordial ``p`` belongs to — i.e. which
        (possibly fused) vessel it is part of at each position."""
        return [r for r in self.track_rows if p in r["members"]]

    def track_table(self):
        import pandas as pd
        return pd.DataFrame(self.track_rows)

    def plot(self, *, cols: int = 4, cmap_name: str = "tab10", retag=None,
             ax_size: float = 4.0, suptitle: Optional[str] = None, show: bool = True,
             high_to_low: bool = True):
        """Render the whole series as a grid, each tracked vessel in a fixed colour by its
        ``track_id`` and labelled with the primordial ids it contains (``0+1+2`` = fused).
        ``high_to_low`` orders panels from the highest length to the lowest; ``retag`` = a
        list of ``(old, new)`` tissue tags applied per section.  Returns the figure."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        cmap = plt.get_cmap(cmap_name)
        max_id = max((int(r["track_id"]) for r in self.track_rows), default=0)
        secs = sorted(self.sections, key=lambda s: s["length"], reverse=high_to_low)
        # One tissue -> colour mapping for the whole sheet.  Sections do not all
        # carry the same tissues (an apex may have run out of metaxylem), and a
        # per-section mapping would recolour everything in the panels that differ.
        # The names must go through `retag` first: plot_cells draws the retagged
        # cells, so categories taken from the raw frame would not line up.
        _remap = dict(retag or [])
        categories = sorted({_remap.get(str(t), str(t))
                             for s in self.sections for t in s["gdf"]["type"]})
        rows = int(np.ceil(len(secs) / cols))
        fig, axs = plt.subplots(rows, cols, figsize=(ax_size * cols, ax_size * rows))
        axs = np.atleast_1d(axs).ravel()
        for ax, sec in zip(axs, secs):
            root = sec["root"]
            for old, new in (retag or []):
                root.retag_cells(old, new)
            root.plot_cells(show=False, ax=ax, categories=categories,
                            title=f"{sec['length']:.0f} mm  ({len(sec['vessels'])} metaxylem)")
            lg = ax.get_legend()
            if lg:
                lg.remove()
            members_by_id = {v[3]: v[4] for v in sec["vessels"]}
            gdf = sec["gdf"]
            for _, row in gdf[gdf["track_id"].notna()].iterrows():
                tid = int(row["track_id"])
                _fill_poly(ax, row.geometry, cmap(tid % 10))
                c = row.geometry.centroid
                ax.annotate(_fmt_members(members_by_id.get(tid, (tid,))), (c.x, c.y),
                            ha="center", va="center", fontsize=6.5, fontweight="bold",
                            color="white", zorder=6)
            ax.set_aspect("equal")
        for ax in axs[len(secs):]:
            ax.set_visible(False)
        handles = [Patch(facecolor=cmap(i % 10), edgecolor="black", label=f"xylem {i}")
                   for i in range(max_id + 1)]
        fig.legend(handles=handles, loc="lower center", ncol=max_id + 1, fontsize=8,
                   frameon=False, bbox_to_anchor=(0.5, -0.02))
        if suptitle:
            plt.suptitle(suptitle, fontsize=13)
        plt.tight_layout()
        if show:
            plt.show()
        return fig


def _fmt_members(members) -> str:
    """Label of the primordial ids inside a (possibly fused) vessel — the exact ids joined
    by '+', so '0+1+2+3' (four fused) is never confused with '0+3' (two fused)."""
    return "+".join(str(m) for m in sorted(members))


def _read_field(base: OrganInputData, name: str, field: str, default: float) -> float:
    """Read one numeric field off a config param entry (model or raw dict), or ``default``."""
    try:
        entry = base._require(name)
    except Exception:
        return float(default)
    val = getattr(entry, field, None) if not isinstance(entry, dict) else entry.get(field)
    return float(val) if val is not None else float(default)


def _fill_poly(ax, poly, color) -> None:
    """Fill a (possibly Multi)Polygon with a solid colour, on top of the tissue."""
    for p in (poly.geoms if poly.geom_type == "MultiPolygon" else [poly]):
        xs, ys = p.exterior.xy
        ax.fill(xs, ys, color=color, ec="black", lw=0.5, zorder=5)


class RootSeries:
    """Longitudinal series of monocot-root sections with identity-tracked metaxylem that
    fuse / terminate / are created along the root.

    Positions to sample (choose one):
        lengths: explicit list of physical positions (mm) — may be sparse / irregular.
        start, end, samples: sample ``samples`` evenly from ``start`` mm to ``end`` mm.

    Vessel schedules (each a ``(start, end)`` linear-ramp tuple **or** a ``length -> value``
    callable; ``start`` = the value at the first sampled position, ``end`` at the last —
    following the direction you give, not min/max):
        n_fused: number of FUSED metaxylem vessels at a length (terminators / creators
            are extra, added on top).  The most a length can have is the maximum of this
            over the series.
        vessel_radius: base radius of a single vessel (mm); fused vessels are larger via
            ``area_retention``.
        stele_radius: stele radius (mm) at a length.

    Events (optional) — a list with **one entry per vessel**, so the list length is how
    many, and each entry is *that vessel's* length (list different numbers for different
    heights):
        terminations: one stop-length per terminating vessel — present for
            ``length >= stop``, then gone (id retired).  e.g. ``[90, 90]`` = two vessels
            stopping at 90 mm; ``[90, 120]`` = two stopping at different heights.
        creations: one appear-length per created vessel — present for ``length <= appear``.
            e.g. ``[40, 70]`` = two vessels appearing at <= 40 mm and <= 70 mm.

    Other:
        area_retention: fused-vessel area growth (see module docstring); 0.4 by default.
        migration_length: after a fusion / termination / creation reshuffles the class
            slots, vessels migrate toward their new slot over this distance (mm): a vessel
            has closed ``min(1, travelled_since_the_event / migration_length)`` of the gap,
            so it arrives exactly ``migration_length`` mm later.  Measured from the event,
            not from the previous sample, so adding samples interpolates the same series
            rather than changing it.  0 = snap straight to the slot.
        fusion_length: length (mm) over which two vessels destined to fuse *approach each
            other* before merging.  Without it a fusing pair sits on its class slots —
            which are maximally spread — until the count drops, so the merge is a jump
            across a gap (in the wheat example the parents were 1.15x further apart than
            contact).  Over the last ``fusion_length`` mm the pair is pulled off its slots
            onto the touching configuration, so it is exactly in contact at the section
            before it merges and the survivor appears continuously.  The approach is
            held back if it would drive a partner into a vessel it is *not* fusing with,
            so a crowded stele converges partially rather than overlapping.  0 (default)
            = the old abrupt behaviour.
        param_schedules: ``{"name.field": schedule}`` to evolve any other config field.
        ring_fraction: advanced — only used for the even-ring fallback when the class
            arrangement is degenerate.
        seed: shared RNG seed (stable tissue refit across sections).
    """

    def __init__(self, base: OrganInputData, lengths=None, *,
                 start: Optional[float] = None, end: Optional[float] = None,
                 samples: Optional[int] = None,
                 n_fused: Schedule,
                 vessel_radius: Schedule,
                 stele_radius: Schedule,
                 terminations: Optional[List[float]] = None,
                 creations: Optional[List[float]] = None,
                 area_retention: float = 0.4,
                 migration_length: float = 0.0,
                 fusion_length: float = 0.0,
                 param_schedules: Optional[Dict[str, Schedule]] = None,
                 ring_fraction: float = 0.55,
                 seed: int = 0):
        self.base = base
        if lengths is not None:
            self.lengths = [float(x) for x in lengths]
        elif None not in (start, end, samples):
            self.lengths = list(np.linspace(float(start), float(end), int(samples)))
        else:
            raise ValueError("give lengths=[...] or start=, end= and samples=")
        self._span = (self.lengths[0], self.lengths[-1])   # (start, end) = first, last sampled
        self.n_fused = self._as_schedule(n_fused)
        self.vessel_radius = self._as_schedule(vessel_radius)
        self.stele_radius = self._as_schedule(stele_radius)
        self.terminations = [float(s) for s in (terminations or [])]
        self.creations = [float(a) for a in (creations or [])]
        self.area_retention = float(area_retention)
        self.migration_length = float(migration_length)
        self.fusion_length = float(fusion_length)
        self.ring_fraction = float(ring_fraction)
        self.param_schedules = {k: self._as_schedule(v) for k, v in (param_schedules or {}).items()}
        self.seed = seed

        self.N_fuse = max(1, max(int(self.n_fused(L)) for L in self.lengths))
        self.N_term = len(self.terminations)
        self.N_creat = len(self.creations)
        self.N = self.N_fuse + self.N_term + self.N_creat
        self._term_ids = list(range(self.N_fuse, self.N_fuse + self.N_term))
        self._creat_ids = list(range(self.N_fuse + self.N_term, self.N))
        # Lay all N primordials on ONE ring (the class's N-vessel arrangement) with the
        # non-fusing vessels spread among the fusers, so the fusers stay spread (full
        # fusion lands central) and there are no colliding angles.
        self._ring_ids = self._interleaved_ids()
        unit = self._class_slots(self.N, 1.0, 0.3)
        unit.sort(key=lambda p: np.arctan2(p[1], p[0]) % (2 * np.pi))
        self._ppos = {rid: unit[k] for k, rid in enumerate(self._ring_ids)}
        # merge tree fuses by DISTANCE between fuser positions, not by id.
        self._parts = self._all_partitions({f: self._ppos[f] for f in range(self.N_fuse)})
        # which groups fuse where — needed by the pre-fusion approach (empty when
        # fusion_length == 0, so that path costs nothing)
        self._approach = self._fusion_approach_schedule()

    # -- schedules ----------------------------------------------------------
    def _as_schedule(self, spec: Schedule) -> Callable[[float], float]:
        """A ``(start, end)`` tuple -> linear ramp from the first to the last sampled
        position (``start`` at ``lengths[0]``, ``end`` at ``lengths[-1]``); a callable ->
        itself."""
        if callable(spec):
            return spec
        a, b = spec
        start, end = self._span
        if end == start:
            return lambda L, a=a: float(a)
        return lambda L, a=a, b=b, s=start, e=end: a + (b - a) * (L - s) / (e - s)

    # -- primordial ring + distance-based merge tree ------------------------
    def _interleaved_ids(self) -> List[int]:
        """The N primordial ids in ring (angular) order, the non-fusers (terminators +
        creators) spread evenly among the fusers."""
        nonf = self._term_ids + self._creat_ids
        slots = set(int(k * self.N / max(1, len(nonf))) for k in range(len(nonf)))
        ring, ni, fi = [], 0, 0
        for k in range(self.N):
            if k in slots and ni < len(nonf):
                ring.append(nonf[ni]); ni += 1
            else:
                ring.append(fi); fi += 1
        return ring

    def _all_partitions(self, pos) -> Dict[int, List[List[int]]]:
        """Distance-based merge tree over the fuser positions ``pos`` ({id: (x, y)}): for
        every k, the partition of the fuser ids into k clusters, merging the two whose
        centroids are NEAREST each step (ties -> smaller combined size, so a symmetric ring
        still gives balanced pairs).  Which vessels fuse is decided by real distance."""
        clusters = [[i] for i in pos]
        cents = [pos[i] for i in pos]
        parts = {len(clusters): [list(c) for c in clusters]}
        while len(clusters) > 1:
            best, ba, bb = None, 0, 1
            for a in range(len(clusters)):
                for b in range(a + 1, len(clusters)):
                    d = np.hypot(cents[a][0] - cents[b][0], cents[a][1] - cents[b][1])
                    key = (round(d, 9), len(clusters[a]) + len(clusters[b]))
                    if best is None or key < best:
                        best, ba, bb = key, a, b
            merged = sorted(clusters[ba] + clusters[bb])
            mc = (float(np.mean([pos[p][0] for p in merged])),
                  float(np.mean([pos[p][1] for p in merged])))
            clusters = [c for k, c in enumerate(clusters) if k not in (ba, bb)] + [merged]
            cents = [c for k, c in enumerate(cents) if k not in (ba, bb)] + [mc]
            parts[len(clusters)] = [list(c) for c in clusters]
        # k = 0: no metaxylem left at all.  The apex of a root can run out of
        # them entirely while protoxylem and phloem continue (see
        # xylem.n_protoxylem), so the tree has to have a rung below one.
        parts[0] = []
        return parts

    def _fused_radius(self, m: int, length: float) -> float:
        base = float(self.vessel_radius(length))
        return base * np.sqrt(max(1e-9, 1.0 + self.area_retention * (m - 1)))

    def _class_slots(self, k: int, R: float, vd: float):
        """The k metaxylem centres the monocot class would place in a stele of radius R
        (its pizza-slice arrangement), for vessels up to ``vd`` across.  Falls back to an
        even ring if the class geometry is degenerate (e.g. its known n=2 quirk), returns
        too few, or spaces them closer than ``vd`` — a slot arrangement the vessels do not
        actually fit in is no use, since they now migrate all the way onto it."""
        slots = MonocotRootAnatomy.metaxylem_positions(Point(0, 0).buffer(R), k, vd)
        ok = len(slots) == k
        if ok and k > 1:
            for i in range(k):
                for j in range(i + 1, k):
                    if np.hypot(slots[i][0] - slots[j][0], slots[i][1] - slots[j][1]) < vd:
                        ok = False
        if ok:
            return slots
        ring = self.ring_fraction * R
        if k == 1:
            return [(0.0, 0.0)]
        return [(ring * np.cos(2 * np.pi * i / k), ring * np.sin(2 * np.pi * i / k)) for i in range(k)]

    # -- per-length vessel identity + placement -----------------------------
    def _active_vessels(self, length: float):
        """Identity of the vessels at ``length`` (no final position yet): the fused groups
        + the active terminators (length >= stop) + the active creators (length <= appear).
        Each carries members, radius, track_id, and a ``rep`` (its class-N centroid, unit)."""
        out = []
        k = self._n_groups(length)
        for members in self._parts[k]:
            rep = (float(np.mean([self._ppos[m][0] for m in members])),
                   float(np.mean([self._ppos[m][1] for m in members])))
            out.append({"members": tuple(members), "tid": int(min(members)),
                        "r": self._fused_radius(len(members), length), "rep": rep})
        for i, stop in enumerate(self.terminations):
            if length >= stop:
                tid = self._term_ids[i]
                out.append({"members": (tid,), "tid": tid,
                            "r": self._fused_radius(1, length), "rep": self._ppos[tid]})
        for i, appear in enumerate(self.creations):
            if length <= appear:
                tid = self._creat_ids[i]
                out.append({"members": (tid,), "tid": tid,
                            "r": self._fused_radius(1, length), "rep": self._ppos[tid]})
        return out

    @staticmethod
    def _optimal_assign(reps, slots) -> List[int]:
        """Assign each vessel (by its current position ``reps[i]``) to a distinct slot,
        minimizing the TOTAL squared displacement summed over every vessel (the
        assignment problem, solved exactly by the Hungarian algorithm) - not nearest
        pairs first.

        Greedy nearest-pairs-first can strand one vessel with whatever slot is left
        once the closer pairs have claimed theirs, sending it all the way across the
        ring while everyone else barely moves. Minimizing the total (squared) distance
        instead makes one vessel's long haul cost as much as several vessels' small
        shifts, so the solver only accepts it when it is genuinely cheaper overall -
        in practice spreading the adjustment thinly across every vessel rather than
        loading it onto one."""
        reps_a = np.asarray(reps, dtype=float)
        slots_a = np.asarray(slots, dtype=float)
        cost = ((reps_a[:, None, :] - slots_a[None, :, :]) ** 2).sum(axis=2)
        row_ind, col_ind = linear_sum_assignment(cost)
        res = [0] * len(reps)
        for i, j in zip(row_ind, col_ind):
            res[i] = int(j)
        return res

    def _config_at(self, length: float) -> OrganInputData:
        cfg = copy.deepcopy(self.base)
        cfg.set_value("stele", "thickness", 2.0 * float(self.stele_radius(length)))
        for key, fn in self.param_schedules.items():
            name, field = key.split(".", 1)
            cfg.set_value(name, field, fn(length))
        return cfg

    def _event_lengths(self) -> List[float]:
        """The exact lengths at which the vessel set jumps — a fusion (``n_fused`` crossing
        an integer), a termination, or a creation.

        These are properties of the *schedules*, not of how you sampled them, and they are
        what the migration is timed from.  Each boundary is emitted with its two float
        neighbours so the walk always holds a point on either side of it: whichever side
        the old vessel set falls on, the walk lands on the boundary to within one ulp.
        """
        lo, hi = min(self.lengths), max(self.lengths)
        bounds = {e for e in (self.terminations + self.creations) if lo < e < hi}
        # n_fused is stepwise; find every step by scanning, then bisecting the change.
        # The scan is deliberately a FIXED grid over the span: sizing it from the number of
        # requested lengths would make the bracket — and so the crossing, to the last bit —
        # depend on the sampling, which is exactly what this is here to avoid.
        n = 4096
        grid = np.linspace(lo, hi, n + 1)
        ks = [self._n_groups(g) for g in grid]
        for i in range(n):
            if ks[i] == ks[i + 1]:
                continue
            a_, b_ = float(grid[i]), float(grid[i + 1])
            for _ in range(60):
                m = 0.5 * (a_ + b_)
                if self._n_groups(m) == ks[i]:
                    a_ = m
                else:
                    b_ = m
            bounds.add(b_)                       # smallest length still on the high side
        out = set()
        for e in bounds:
            out |= {e, float(np.nextafter(e, hi)), float(np.nextafter(e, lo))}
        return sorted(x for x in out if lo <= x <= hi)

    def _walk_lengths(self) -> List[float]:
        """The lengths the series is *stepped* through, collar -> apex: what the user asked
        for plus the exact event boundaries.  Only the user's lengths are ever rendered."""
        return sorted(set(self.lengths) | set(self._event_lengths()), reverse=True)

    def _n_groups(self, length: float) -> int:
        """Number of fused groups at ``length`` — the schedule, clipped to the tree.

        The floor is 0, not 1: a series may end with no metaxylem at all.
        """
        return int(np.clip(int(self.n_fused(length)), 0, self.N_fuse))

    def _fusion_approach_schedule(self) -> Dict[tuple, Tuple[float, List[tuple]]]:
        """``{parent members: (last_length, [sibling members])}`` — for every group that
        is about to fuse, the LAST sampled length at which it is still separate and the
        other groups it merges with there.

        Walked collar -> apex like :meth:`_migrated_vessels`: whenever the partition for
        the current count absorbs two or more of the previous length's groups into one,
        those groups are the parents of that fusion and the previous length is where they
        must already be touching.
        """
        sched: Dict[tuple, Tuple[float, List[tuple]]] = {}
        if self.fusion_length <= 0:
            return sched
        prev_groups, prev_L = None, None
        for L in self._walk_lengths():
            groups = [tuple(g) for g in self._parts[self._n_groups(L)]]
            if prev_groups is not None:
                for child in groups:
                    kids = set(child)
                    parents = [p for p in prev_groups if set(p) < kids]
                    if len(parents) > 1:                    # a fusion happens at L
                        for p in parents:
                            sched[p] = (prev_L, [q for q in parents if q != p])
            prev_groups, prev_L = groups, L
        return sched

    def _approach_targets(self, L: float, vessels: list, reps: list
                          ) -> Dict[int, Tuple[float, Tuple[float, float], frozenset]]:
        """``{vessel index: (beta, (x, y), partners)}`` — the touching position each
        about-to-fuse vessel should be blended toward at ``L``, how strongly, and the
        indices of the vessels it is fusing with (the only ones it may touch).

        ``beta`` ramps 0 -> 1 over the last ``fusion_length`` mm before the fusion, so a
        pair converges gradually and is exactly touching at the final section before it
        merges.  The contact configuration shrinks the parents toward their common
        centroid by the tightest factor that makes some pair touch
        (``s = min (r_i + r_j) / d_ij``), which reduces to "centres exactly r1+r2 apart"
        for the usual two-parent merge and still behaves for a 3+ way merge (possible when
        ``n_fused`` drops by more than one between samples).
        """
        out: Dict[int, Tuple[float, Tuple[float, float]]] = {}
        if self.fusion_length <= 0 or not self._approach:
            return out
        by_members = {tuple(v["members"]): vi for vi, v in enumerate(vessels)}
        for vi, v in enumerate(vessels):
            info = self._approach.get(tuple(v["members"]))
            if info is None:
                continue
            last_L, siblings = info
            if L < last_L:                                  # already fused below here
                continue
            beta = 1.0 - (L - last_L) / self.fusion_length
            if beta <= 0.0:                                 # still outside the window
                continue
            beta = min(1.0, beta)
            idxs = [vi] + [by_members[s] for s in siblings if s in by_members]
            if len(idxs) < 2:
                continue
            pts = [reps[i] for i in idxs]
            rad = [vessels[i]["r"] for i in idxs]
            cx = float(np.mean([p[0] for p in pts]))
            cy = float(np.mean([p[1] for p in pts]))
            s = 1.0
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    d = np.hypot(pts[a][0] - pts[b][0], pts[a][1] - pts[b][1])
                    if d > 0:
                        s = min(s, (rad[a] + rad[b]) / d)   # never push apart: s <= 1
            px, py = reps[vi]
            out[vi] = (beta, (cx + s * (px - cx), cy + s * (py - cy)), frozenset(idxs))
        return out

    @staticmethod
    def _approach_is_clear(pos, vessels: list, contact: dict) -> bool:
        """No approaching vessel has been driven into one it is *not* fusing with.

        Only the approach can cause this: the class slots are collision-free by
        construction, so the check is limited to the vessels being pulled off them."""
        for vi, (_beta, _target, partners) in contact.items():
            for vj in range(len(vessels)):
                if vj == vi or vj in partners:
                    continue
                d = np.hypot(pos[vi][0] - pos[vj][0], pos[vi][1] - pos[vj][1])
                if d < vessels[vi]["r"] + vessels[vj]["r"] - 1e-12:
                    return False
        return True

    def _migrated_vessels(self) -> Dict[float, list]:
        """Actual vessel positions per length, walking from the highest length to the
        lowest.  When the vessel set changes (a fusion, a termination, a creation) the
        class's slots for the new count are assigned to the vessels by whichever
        pairing minimizes the TOTAL displacement across all of them (see
        :meth:`_optimal_assign`) - not each to its own nearest slot - and each vessel
        then migrates from where it stood at that event toward its new slot; a freshly
        fused vessel starts at the mean of its members' last positions, and a
        newly-appearing vessel at its class-N position.

        Placement is a function of the length alone.  Migration is timed from the event
        (``anchor_L``), not from the previous sample, and the walk steps through the exact
        event boundaries whatever the user asked to render — so adding samples interpolates
        the same series instead of producing a different one.
        """
        want = set(self.lengths)
        prev_pos: Dict[int, Tuple[float, float]] = {}
        anchor: Dict[int, Tuple[float, float]] = {}   # where a vessel began its migration
        anchor_L: Dict[int, float] = {}               # ...and the length at which it began
        prev_key, prev_L = None, None
        out: Dict[float, list] = {}
        for L in self._walk_lengths():
            R = float(self.stele_radius(L))
            vessels = self._active_vessels(L)
            if not vessels:
                # A section past the last metaxylem.  Nothing to place or migrate,
                # but the length still has to appear in the output so the series
                # renders it (protoxylem and phloem are rebuilt per section and do
                # not depend on this).
                out[L] = []
                prev_key, prev_L = (), L
                continue
            # Space the slots for the BIGGEST vessel present: a fused one is wider than
            # ``vessel_radius`` by ``area_retention``, and it has to fit too.
            vd = 2.0 * max(float(v["r"]) for v in vessels)
            slots = self._class_slots(len(vessels), R, vd)

            key = tuple(sorted(tuple(v["members"]) for v in vessels))
            if key != prev_key:
                # The vessel set jumped somewhere in (L, prev_L].  Every exact boundary is
                # walked, so prev_L *is* that jump and ``prev_pos`` is where the vessels
                # stood when it happened: re-anchor them all there and restart the clock.
                seed = []
                for v in vessels:
                    if v["tid"] in prev_pos:
                        seed.append(prev_pos[v["tid"]])
                    else:
                        parents = [prev_pos[m] for m in v["members"] if m in prev_pos]
                        if parents:                    # a fusion: start at their mean
                            seed.append((float(np.mean([p[0] for p in parents])),
                                         float(np.mean([p[1] for p in parents]))))
                        else:                          # brand new: its class-N position
                            seed.append((v["rep"][0] * R, v["rep"][1] * R))
                first = self._optimal_assign(seed, slots)
                for vi, v in enumerate(vessels):
                    # Nothing precedes the first section, so it simply starts on its slot.
                    anchor[v["tid"]] = seed[vi] if prev_L is not None else slots[first[vi]]
                    anchor_L[v["tid"]] = L if prev_L is None else prev_L
            prev_key, prev_L = key, L

            # Assign from the anchors, not from the previous section: between two events
            # the anchors are fixed, so the assignment is stable and the whole placement
            # depends only on L.
            assign = self._optimal_assign([anchor[v["tid"]] for v in vessels], slots)
            base = []
            for vi, v in enumerate(vessels):
                ax, ay = anchor[v["tid"]]
                tx, ty = slots[assign[vi]]
                alpha = 1.0 if self.migration_length <= 0 else \
                    min(1.0, abs(anchor_L[v["tid"]] - L) / self.migration_length)
                base.append((ax + alpha * (tx - ax), ay + alpha * (ty - ay)))

            contact = self._approach_targets(L, vessels, base)

            def place(scale, _b=base, _c=contact):
                """Positions with the pre-fusion approach damped by ``scale``.

                Pre-fusion approach: pull the partners off their (maximally spread) class
                slots and onto the touching configuration, so the merge one section later
                is continuous instead of a jump across a gap.  Applied to the migrated
                position, so beta=1 lands exactly on contact however far the migration
                has got."""
                pos = list(_b)
                for i, (beta, (cx, cy), _p) in _c.items():
                    x, y = _b[i]
                    beta *= scale
                    pos[i] = (x + beta * (cx - x), y + beta * (cy - y))
                return pos

            # A pair converging on each other can sweep across a third vessel it is NOT
            # fusing with (a terminator, or the next group up the merge tree), which would
            # be an invalid anatomy.  Damp the whole approach by the largest factor that
            # keeps every bystander clear: full contact when there is room, a partial
            # approach when there is not.
            pos = place(1.0)
            if contact and not self._approach_is_clear(pos, vessels, contact):
                lo, hi = 0.0, 1.0                      # lo always clear, hi never
                for _ in range(24):
                    mid = 0.5 * (lo + hi)
                    if self._approach_is_clear(place(mid), vessels, contact):
                        lo = mid
                    else:
                        hi = mid
                pos = place(lo)

            actual, newprev = [], {}
            for vi, v in enumerate(vessels):
                ax, ay = pos[vi]
                actual.append((ax, ay, v["r"], v["tid"], v["members"]))
                newprev[v["tid"]] = (ax, ay)
                for m in v["members"]:
                    newprev[m] = (ax, ay)
            prev_pos = newprev
            if L in want:
                out[L] = actual
        return out

    # -- run ----------------------------------------------------------------
    def generate(self) -> RootSeriesResult:
        migrated = self._migrated_vessels()
        sections, track_rows = [], []
        for length in self.lengths:
            vset = migrated[length]
            prescribe = [(x, y, r, tid) for (x, y, r, tid, _m) in vset]
            cfg = self._config_at(length)
            # Tell the section how many metaxylem it actually has. The protoxylem
            # and phloem are regenerated per section (untracked) but are NOT scaled
            # off this count -- poles are set directly via "xylem.n_protoxylem", so
            # a section can keep them after the last metaxylem has gone. Their
            # count/size stays tunable via param_schedules (e.g.
            # "xylem.n_protoxylem", "phloem.sieve_diameter").
            cfg.set_value("xylem", "n_vascular_bundles", len(prescribe))
            root = RootAnatomy(cfg, seed=self.seed).prescribe_vessels(prescribe)
            gdf = root.generate_cells()
            sections.append({"length": length, "root": root, "gdf": gdf, "vessels": vset})
            for (x, y, r, tid, members) in vset:
                track_rows.append({"track_id": tid, "length": length,
                                   "x": x, "y": y, "radius": r, "members": members})
        return RootSeriesResult(sections, track_rows)


class DicotRootSeries:
    """Longitudinal series of **dicot**-root sections with identity-tracked primary xylem.

    Unlike the monocot series, dicot primary xylem vessels do **not** fuse or migrate:
    each protoxylem / metaxylem keeps its place.  What changes along the root is the
    **pith** — a central front that *recedes* (from the apex toward the collar).  Three
    things together define the space a xylem vessel may occupy (per the model): the
    **star shape**, the **5PL diameter function** and the **pith**.  The first two are
    baked in once by extracting a reference section (the smallest-pith end, where the star
    is fully packed by the class): every vessel's fixed radial position and its 5PL target
    size come from there.  The pith is then the only per-section front:

    * a vessel is **present** once the pith has receded past it (there is room outside the
      pith at its position) — so the outer protoxylem appear first and the inner metaxylem
      only near the collar;
    * its size is ``min(5PL target, room the pith leaves)`` — small when it has just
      cleared the pith, growing to its full 5PL target as the pith recedes further.

    Positions are stored as a fraction of the stele radius, so if ``stele_radius`` grows
    along the series the vessels ride out with it ("move a little, not much"); keep
    ``stele_radius`` flat for pure primary growth at a fixed stele.

    Sample positions (choose one): ``lengths=[...]`` or ``start=, end=, samples=``.

    Schedules (each a ``(value_at_start, value_at_end)`` linear-ramp tuple **or** a
    ``length -> value`` callable):
        stele_radius: stele radius (mm) at a length.
        pith_radius: pith radius (mm) at a length — the receding central front (typically
            large at the apex, small/zero at the collar).
        param_schedules: ``{"name.field": schedule}`` to evolve any other config field
            (e.g. ``"xylem.vessel_diameter"`` to grow the 5PL target itself).

    Other:
        present_min_radius: a vessel counts as present once the pith leaves it at least this
            much radius (mm); default = the base xylem ``vessel_diameter_min`` / 2.
        seed: shared RNG seed (stable tissue refit across sections).
    """

    def __init__(self, base: OrganInputData, lengths=None, *,
                 start: Optional[float] = None, end: Optional[float] = None,
                 samples: Optional[int] = None,
                 stele_radius: Schedule,
                 pith_radius: Schedule,
                 present_min_radius: Optional[float] = None,
                 param_schedules: Optional[Dict[str, Schedule]] = None,
                 seed: int = 0):
        self.base = base
        if lengths is not None:
            self.lengths = [float(x) for x in lengths]
        elif None not in (start, end, samples):
            self.lengths = list(np.linspace(float(start), float(end), int(samples)))
        else:
            raise ValueError("give lengths=[...] or start=, end= and samples=")
        self._span = (self.lengths[0], self.lengths[-1])
        self.stele_radius = self._as_schedule(stele_radius)
        self.pith_radius = self._as_schedule(pith_radius)
        self.param_schedules = {k: self._as_schedule(v) for k, v in (param_schedules or {}).items()}
        if present_min_radius is not None:
            self.present_min_radius = float(present_min_radius)
        else:
            self.present_min_radius = 0.5 * _read_field(base, "xylem", "vessel_diameter_min", 0.01)
        self.seed = seed
        self._primordials = None                       # filled lazily by _extract()

    # -- schedules ----------------------------------------------------------
    def _as_schedule(self, spec: Schedule) -> Callable[[float], float]:
        if callable(spec):
            return spec
        a, b = spec
        start, end = self._span
        if end == start:
            return lambda L, a=a: float(a)
        return lambda L, a=a, b=b, s=start, e=end: a + (b - a) * (L - s) / (e - s)

    def _pith_at(self, length: float) -> float:
        """Pith radius at ``length``, snapped to exactly 0 below a tiny epsilon — a
        sub-epsilon pith (e.g. 1e-17 from a ramp) perturbs the packing, making the
        extracted primordial set non-reproducible."""
        pith = float(self.pith_radius(length))
        return 0.0 if pith < 1e-9 else pith

    def _config_at(self, length: float) -> OrganInputData:
        cfg = copy.deepcopy(self.base)
        cfg.set_value("stele", "thickness", 2.0 * float(self.stele_radius(length)))
        cfg.set_value("xylem", "pith_radius", self._pith_at(length))
        for key, fn in self.param_schedules.items():
            name, field = key.split(".", 1)
            cfg.set_value(name, field, fn(length))
        return cfg

    # -- primordial extraction (star + 5PL baked in) ------------------------
    def _ref_length(self) -> float:
        """The sample where the pith is smallest — the star is fully packed there, so it is
        the reference for every vessel's position + 5PL target size."""
        return min(self.lengths, key=self._pith_at)

    def _extract(self):
        """Generate the reference (smallest-pith) section with the normal dicot packer and
        read back each primary-xylem vessel as a primordial: its position as a *fraction* of
        the stele radius, its 5PL target radius, and a track_id.  The star shape and the 5PL
        gradient are captured here once; only the pith varies per section afterwards."""
        if self._primordials is not None:
            return self._primordials
        Lref = self._ref_length()
        Rref = float(self.stele_radius(Lref))
        cfg = self._config_at(Lref)
        root = RootAnatomy(cfg, seed=self.seed)
        gdf = root.generate_cells()
        vessels = self._vessel_circles(root, gdf)          # [(x, y, r)]
        vessels.sort(key=lambda v: np.hypot(v[0], v[1]))   # id 0 = innermost, rising outward
        prim = []
        for tid, (x, y, r) in enumerate(vessels):
            d = np.hypot(x, y)
            prim.append({"tid": tid, "fx": x / Rref, "fy": y / Rref,
                         "fd": d / Rref, "target": r})
        self._primordials = prim
        return prim

    @staticmethod
    def _vessel_circles(root, gdf):
        """The packed primary-xylem vessels of a reference section as ``(x, y, radius)``.

        Prefers ``root.vascular_polygons`` (the placed vessel circles recorded by
        ``_record_xylem_vessels``); falls back to the ``xylem`` cells of the gdf."""
        polys = getattr(root, "vascular_polygons", None)
        if polys:
            out = []
            for p in polys:
                if p.is_empty:
                    continue
                c = p.centroid
                out.append((c.x, c.y, float(np.sqrt(max(p.area, 1e-12) / np.pi))))
            if out:
                return out
        sub = gdf[gdf["type"].isin(["xylem", "metaxylem"])]
        return [(g.centroid.x, g.centroid.y, float(np.sqrt(max(g.area, 1e-12) / np.pi)))
                for g in sub.geometry]

    # -- per-length presence + pith-capped size -----------------------------
    def _active_vessels(self, length: float):
        """The vessels present at ``length`` with their pith-capped positions + sizes.

        A vessel at fractional radius ``fd`` sits at ``fd * stele_radius(L)``; the pith front
        is ``pith_radius(L)``.  ``room = distance - pith`` is how much radius the pith leaves
        it — present once ``room >= present_min_radius``, sized ``min(5PL target, room)``.
        Returns ``[(x, y, r, tid, members)]`` (members = the singleton ``(tid,)``)."""
        R = float(self.stele_radius(length))
        pith = self._pith_at(length)
        out = []
        no_pith = pith < self.present_min_radius   # pith gone -> no constraint, full target
        for p in self._extract():
            d = p["fd"] * R
            if no_pith:
                r = p["target"]
            else:
                room = d - pith
                if room < self.present_min_radius:
                    continue                       # still inside the pith
                r = min(p["target"], room)
            out.append((p["fx"] * R, p["fy"] * R, r, p["tid"], (p["tid"],)))
        return out

    # -- run ----------------------------------------------------------------
    def generate(self) -> RootSeriesResult:
        self._extract()
        sections, track_rows = [], []
        for length in self.lengths:
            vset = self._active_vessels(length)
            prescribe = [(x, y, r, tid) for (x, y, r, tid, _m) in vset]
            cfg = self._config_at(length)
            root = RootAnatomy(cfg, seed=self.seed).prescribe_vessels(prescribe)
            gdf = root.generate_cells()
            sections.append({"length": length, "root": root, "gdf": gdf, "vessels": vset})
            for (x, y, r, tid, members) in vset:
                track_rows.append({"track_id": tid, "length": length,
                                   "x": x, "y": y, "radius": r, "members": members})
        return RootSeriesResult(sections, track_rows)
