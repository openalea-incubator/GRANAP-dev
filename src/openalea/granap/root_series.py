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
        rows = int(np.ceil(len(secs) / cols))
        fig, axs = plt.subplots(rows, cols, figsize=(ax_size * cols, ax_size * rows))
        axs = np.atleast_1d(axs).ravel()
        for ax, sec in zip(axs, secs):
            root = sec["root"]
            for old, new in (retag or []):
                root.retag_cells(old, new)
            root.plot_cells(show=False, ax=ax,
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
        migration_length: vessels migrate toward their class-optimal slot, closing a
            fraction ``min(1, Δlength / migration_length)`` of the gap each step; 0 = snap.
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
        return parts

    def _fused_radius(self, m: int, length: float) -> float:
        base = float(self.vessel_radius(length))
        return base * np.sqrt(max(1e-9, 1.0 + self.area_retention * (m - 1)))

    def _class_slots(self, k: int, R: float, vd: float):
        """The k metaxylem centres the monocot class would place in a stele of radius R
        (its pizza-slice arrangement).  Falls back to an even ring if the class geometry is
        degenerate (e.g. its known n=2 quirk) or returns too few."""
        slots = MonocotRootAnatomy.metaxylem_positions(Point(0, 0).buffer(R), k, vd)
        ok = len(slots) == k
        if ok and k > 1:
            for i in range(k):
                for j in range(i + 1, k):
                    if np.hypot(slots[i][0] - slots[j][0], slots[i][1] - slots[j][1]) < vd * 0.6:
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
        k = int(np.clip(int(self.n_fused(length)), 1, self.N_fuse))
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
    def _greedy_assign(reps, slots) -> List[int]:
        """Assign each vessel (by its current position ``reps[i]``) to a distinct slot,
        nearest pairs first — so a vessel goes to the class slot closest to where it already
        is (no id/angle shortcuts, no crossing over)."""
        pairs = sorted((np.hypot(reps[i][0] - slots[j][0], reps[i][1] - slots[j][1]), i, j)
                       for i in range(len(reps)) for j in range(len(slots)))
        vtaken, staken, res = set(), set(), [0] * len(reps)
        for _d, i, j in pairs:
            if i not in vtaken and j not in staken:
                res[i] = j
                vtaken.add(i); staken.add(j)
        return res

    def _config_at(self, length: float) -> OrganInputData:
        cfg = copy.deepcopy(self.base)
        cfg.set_value("stele", "thickness", 2.0 * float(self.stele_radius(length)))
        for key, fn in self.param_schedules.items():
            name, field = key.split(".", 1)
            cfg.set_value(name, field, fn(length))
        return cfg

    def _migrated_vessels(self) -> Dict[float, list]:
        """Actual vessel positions per length, walking from the highest length to the
        lowest.  At each length the vessels are assigned to the class's slots for the
        current count by NEAREST to where they already are (greedy), then moved a fraction
        of the way there; a freshly fused vessel starts at the mean of its members' last
        positions, and a newly-appearing vessel at its class-N position."""
        order = sorted(self.lengths, reverse=True)
        prev_pos: Dict[int, Tuple[float, float]] = {}
        prev_L = None
        out: Dict[float, list] = {}
        for L in order:
            R = float(self.stele_radius(L))
            vd = 2.0 * float(self.vessel_radius(L))
            vessels = self._active_vessels(L)
            slots = self._class_slots(len(vessels), R, vd)
            reps = []
            for v in vessels:
                if v["tid"] in prev_pos:
                    reps.append(prev_pos[v["tid"]])
                else:
                    seeds = [prev_pos[m] for m in v["members"] if m in prev_pos]
                    if seeds:
                        reps.append((float(np.mean([p[0] for p in seeds])),
                                     float(np.mean([p[1] for p in seeds]))))
                    else:
                        reps.append((v["rep"][0] * R, v["rep"][1] * R))
            assign = self._greedy_assign(reps, slots)
            alpha = 1.0 if (prev_L is None or self.migration_length <= 0) \
                else min(1.0, abs(prev_L - L) / self.migration_length)
            actual, newprev = [], {}
            for vi, v in enumerate(vessels):
                tx, ty = slots[assign[vi]]
                px, py = reps[vi]
                ax, ay = px + alpha * (tx - px), py + alpha * (ty - py)
                actual.append((ax, ay, v["r"], v["tid"], v["members"]))
                newprev[v["tid"]] = (ax, ay)
                for m in v["members"]:
                    newprev[m] = (ax, ay)
            prev_pos, prev_L, out[L] = newprev, L, actual
        return out

    # -- run ----------------------------------------------------------------
    def generate(self) -> RootSeriesResult:
        migrated = self._migrated_vessels()
        sections, track_rows = [], []
        for length in self.lengths:
            vset = migrated[length]
            prescribe = [(x, y, r, tid) for (x, y, r, tid, _m) in vset]
            root = RootAnatomy(self._config_at(length), seed=self.seed).prescribe_vessels(prescribe)
            gdf = root.generate_cells()
            sections.append({"length": length, "root": root, "gdf": gdf, "vessels": vset})
            for (x, y, r, tid, members) in vset:
                track_rows.append({"track_id": tid, "length": length,
                                   "x": x, "y": y, "radius": r, "members": members})
        return RootSeriesResult(sections, track_rows)
