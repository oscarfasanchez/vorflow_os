"""Corner relief: a bend of one barrier line that the construction cannot make exact is changed only there.

The empty-circle construction makes a corner exact down to about 10 degrees on
its own, but a run of sharp corners (a zigzag of 20-60 degrees with short legs)
can need faces shorter than ``pipeline.MIN_H_FRACTION`` x lc. ``pipeline``
then names the corners it could not make (the floor check, or defects left
after the repair rounds), and only those are relieved, before Gmsh runs:

- the two legs end where they are RELIEF_WIDTH x lc apart, at
  t = width / (2 sin(theta / 2)) from the apex (never more than
  RELIEF_LEG_FRACTION of either leg), and a chord joins the two new ends;
- *line* tips (``barrier_tips="line"``, the default) keep one barrier line
  from the middle of the chord to the apex, along the bisector, so the barrier
  still reaches its tip; only the sliver narrower than the width is lost;
- *cut* tips (``barrier_tips="cut"``) stop at the chord; the tip lies inside
  cells that straddle it, as for #73's spike truncation.

A corner named again gets RELIEF_GROWTH x its width. A line tip named more
than LINE_TRIES times keeps only its chord. Two relieved corners that would
leave less than MIN_LEFT x lc of their shared leg get one chord across both
(each line tip then runs from its nearest point on that chord). Every corner
is recorded with its place, angle, width, mode and largest move. A bend with
a leg shorter than MIN_LEG x lc is not relieved: that is digitising noise or
dense digitising, for which the error asks to resample the line instead.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import shapely
from shapely.geometry import LineString

from .lines import BarrierLine

RELIEF_WIDTH = 0.5          # x lc: the legs end where they are this far apart
RELIEF_GROWTH = 1.5         # a corner named again gets this x its last width
RELIEF_LEG_FRACTION = 0.45  # x each leg: the most a relief takes from a leg
MIN_LEFT = 0.5              # x lc: two reliefs that would leave less of their shared leg become one chord
LINE_TRIES = 3              # a line tip named more often than this keeps only its chord
RELIEF_TRIES = 8            # a corner named more often than this is not relieved again
REACH = 3.0                 # x lc: a failing spot is put down to the nearest bend this close
MIN_LEG = 0.5               # x lc: a bend with a shorter leg is digitising detail, not a corner to relieve
TIP_MODES = ("line", "cut")


def bends(lines: list) -> dict:
    """{apex: (line, vertex, inside angle)} of every interior vertex of a line that no other line touches,
    that is not must-keep (a crossing, a boundary or standard-line touch, a stem end) and whose legs are both
    at least MIN_LEG x lc (shorter ones are noise or dense digitising, which the error message is for)."""
    counts = {}
    for b in lines:
        for p in map(tuple, b.xy):
            counts[p] = counts.get(p, 0) + 1
    out = {}
    for k, b in enumerate(lines):
        keep = _keep(b)
        for i in range(1, len(b.xy) - 1):
            p = tuple(b.xy[i])
            u, w = b.xy[i - 1] - b.xy[i], b.xy[i + 1] - b.xy[i]
            if counts[p] != 1 or keep[i] or min(np.hypot(*u), np.hypot(*w)) < MIN_LEG * b.lc:
                continue
            theta = math.acos(float(np.clip(u @ w / (np.hypot(*u) * np.hypot(*w)), -1.0, 1.0)))
            out[p] = (k, i, theta)
    return out


def nearest_bends(lines: list, points: np.ndarray, reach_lc: float = REACH) -> set:
    """The bends nearest to ``points``, each within ``reach_lc`` x the lc of its line."""
    found = bends(lines)
    if not found or len(points) == 0:
        return set()
    apexes = np.array(list(found), dtype=float)
    reach = np.array([reach_lc * lines[k].lc for k, _, _ in found.values()])
    out = set()
    for p in np.asarray(points, dtype=float).reshape(-1, 2):
        d = np.hypot(*(apexes - p).T)
        j = int(np.argmin(d))
        if d[j] <= reach[j]:
            out.add(tuple(apexes[j]))
    return out


def relieve(lines: list, named: dict, mode: str) -> tuple:
    """(lines, records): the bends in ``named`` ({apex: times named}) relieved (module docstring)."""
    if not named:
        return lines, []
    found = bends(lines)
    out, records = [], []
    for k, b in enumerate(lines):
        cuts = {i: named[p] for p, (kk, i, _) in found.items() if kk == k and p in named}
        if not cuts:
            out.append(b)
            continue
        main, spurs, recs = _relieve_line(b, cuts, mode)
        out.append(main)
        out += spurs
        records += recs
    return out, records


def _relieve_line(b: BarrierLine, cuts: dict, mode: str) -> tuple:
    """(main line, tip lines, records) of line ``b`` with the vertices in ``cuts`` ({vertex: times named}) relieved."""
    xy, n, kept = b.xy, len(b.xy), _keep(b)
    t, width, theta = {}, {}, {}
    for i, times in cuts.items():
        u, w = xy[i - 1] - xy[i], xy[i + 1] - xy[i]
        lu, lw = float(np.hypot(*u)), float(np.hypot(*w))
        theta[i] = math.acos(float(np.clip(u @ w / (lu * lw), -1.0, 1.0)))
        width[i] = RELIEF_WIDTH * b.lc * RELIEF_GROWTH ** (min(times, RELIEF_TRIES) - 1)
        t[i] = min(width[i] / (2 * math.sin(theta[i] / 2)), RELIEF_LEG_FRACTION * lu, RELIEF_LEG_FRACTION * lw)
    pts, keep, groups, i = [xy[0]], [kept[0]], [], 1
    while i < n - 1:
        if i not in t:
            pts.append(xy[i])
            keep.append(kept[i])
            i += 1
            continue
        j = i
        while j + 1 in t and np.hypot(*(xy[j + 1] - xy[j])) - t[j] - t[j + 1] < MIN_LEFT * b.lc:
            j += 1
        a = xy[i] + t[i] * _unit(xy[i - 1] - xy[i])
        c = xy[j] + t[j] * _unit(xy[j + 1] - xy[j])
        members = list(range(i, j + 1))
        line_tips = [v for v in members if mode == "line" and cuts[v] <= LINE_TRIES]
        # each tip line starts on the chord at a vertex of its own, so it meets the chord at a junction
        feet = sorted(((_param(xy[v], a, c), v) for v in line_tips))
        chord = [a] + [a + s_v * (c - a) for s_v, _ in feet] + [c]
        chord = [q for k, q in enumerate(chord) if k == 0 or np.hypot(*(q - chord[k - 1])) > 1e-9 * b.lc]
        pts += chord
        keep += [True] * len(chord)
        groups.append((members, line_tips, {v: a + s_v * (c - a) for s_v, v in feet}))
        i = j + 1
    pts.append(xy[-1])
    keep.append(kept[-1])
    main = dataclasses.replace(b, xy=np.array(pts, dtype=float), keep=np.array(keep, dtype=bool))
    spurs, records = [], []
    for members, line_tips, feet in groups:
        for v in members:
            line_tip = v in line_tips
            if line_tip:
                spurs.append(BarrierLine(b.line_id, np.array([feet[v], xy[v]]), b.lc, b.tol, np.array([True, True])))
            records.append({"kind": "corner", "location": xy[v].copy(), "angle": math.degrees(theta[v]),
                            "width": width[v], "tip": "line" if line_tip else "cut", "merged": len(members) > 1,
                            "line_id": b.line_id})
    moved = _moves(xy, main.xy, spurs)
    for r in records:
        r["deviation"] = moved
    return main, spurs, records


def _keep(b: BarrierLine) -> np.ndarray:
    """The must-keep mask of ``b`` (only its ends when it has none)."""
    if b.keep is not None:
        return np.asarray(b.keep, dtype=bool)
    keep = np.zeros(len(b.xy), dtype=bool)
    keep[[0, -1]] = True
    return keep


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.hypot(*v)


def _param(p: np.ndarray, a: np.ndarray, c: np.ndarray) -> float:
    """Where along chord a-c (0 to 1) the point nearest to ``p`` lies; 0.5 when the chord is one point."""
    d = c - a
    dd = float(d @ d)
    if dd == 0.0:
        return 0.5
    return min(max(float((p - a) @ d) / dd, 0.0), 1.0)


def _moves(old: np.ndarray, new: np.ndarray, spurs: list) -> float:
    """Hausdorff distance between the line as drawn and its relieved version (main line and tip lines)."""
    relieved = shapely.union_all([LineString(new)] + [LineString(s.xy) for s in spurs])
    return float(LineString(old).hausdorff_distance(relieved))
