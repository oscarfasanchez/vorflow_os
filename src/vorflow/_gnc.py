"""One-ring ghost-head estimation for built DISU grids.

The geometric method reproduces affine heads. The experimental conductivity
method assumes locally planar scalar-K transmission; it is not a general
heterogeneous-flow reconstruction or an implementation of ModelMuse/Gridgen.
"""

from collections import Counter
from dataclasses import dataclass
from numbers import Real
import warnings

import numpy as np
from scipy.optimize import linprog
from shapely.geometry import Point

from ._gnc_export import GncCorrections


@dataclass
class _Geometry:
    xyz: np.ndarray
    polygons: list
    top: np.ndarray
    bottom: np.ndarray
    volume: np.ndarray
    adjacency: list
    normals: np.ndarray
    tolerance: float
    blocked: set


def _pairs(pairs, nodes, name):
    found = set()
    for pair in pairs:
        if (
            len(pair) != 2
            or any(
                isinstance(n, (bool, np.bool_))
                or not isinstance(n, (int, np.integer))
                or not 0 <= n < nodes
                for n in pair
            )
            or pair[0] == pair[1]
        ):
            raise ValueError(f"{name} must contain distinct active node indices.")
        found.add(tuple(sorted(map(int, pair))))
    return found


def _geometry(grid, blocked_pairs, barriers):
    grid._require_build()
    if grid.options.cl12 != "plan":
        raise ValueError("GNC requires cl12='plan' perpendicular face distances.")
    if not grid.nodes:
        raise ValueError("GNC requires at least one active cell.")
    blocked = _pairs(blocked_pairs, grid.nodes, "blocked_pairs")
    cut = blocked | _pairs(barriers, grid.nodes, "barriers")
    cells = grid.cellmap[grid.cellmap.node >= 0].sort_values("node")
    xyz = np.column_stack((cells.x, cells.y, (cells.top + cells.bot) / 2))
    polygons = [
        grid._grids[int(row.grid)].geometry.iloc[int(row.icpl)]
        for row in cells.itertuples()
    ]
    links = grid.links
    normals = np.column_stack(
        (
            np.cos(np.radians(links["angle"])),
            np.sin(np.radians(links["angle"])),
            np.zeros(len(links["n"])),
        )
    )
    adjacency = [[] for _ in xyz]
    for i, (n, m, ihc, cn, cm) in enumerate(
        zip(links["n"], links["m"], links["ihc"], links["cl_n"], links["cl_m"])
    ):
        n, m = int(n), int(m)
        if ihc == 0:
            normals[i] = [0, 0, np.sign(xyz[m, 2] - xyz[n, 2])]
        if tuple(sorted((n, m))) not in cut:
            adjacency[n].append((m, i, normals[i].copy(), float(cn), float(cm)))
            adjacency[m].append((n, i, -normals[i].copy(), float(cm), float(cn)))
    for neighbours in adjacency:
        neighbours.sort(key=lambda edge: edge[0])
    area = grid.get_gridprops_mf6_disu()["area"]
    return _Geometry(
        xyz,
        polygons,
        cells.top.to_numpy(),
        cells.bot.to_numpy(),
        (cells.top - cells.bot).to_numpy() * area,
        adjacency,
        normals,
        1e-9 * max(float(np.ptp(xyz, axis=0).max()), 1.0),
        blocked,
    )


def estimate_gnc(
    grid,
    k=None,
    *,
    method="geometric",
    connections="3d",
    min_offset=0.0,
    blocked_pairs=(),
    barriers=(),
    strict=False,
    contrast=10.0,
):
    """Estimate one-ring corrections without using solved heads.

    Parameters
    ----------
    grid : DisuGrid
        Built grid with perpendicular ``cl12='plan'`` distances.
    k : float or array_like of shape (nodes,), optional
        Current positive scalar isotropic conductivity in active-node order.
        Required for ``method='k_aware'``; ignored by ``'geometric'``.
    method : {'geometric', 'k_aware'}, default 'geometric'
        Geometry-only affine interpolation, or experimental scalar-K local
        transmission interpolation. Both search immediate connected neighbours
        only. Recompute K-aware corrections when K changes.
    connections : {'3d', 'all'}, default '3d'
        Correct vertical connections and lateral elevation offsets, or also
        in-plane offsets (for example at a splice). Aligned links are skipped.
    min_offset : float, default 0
        Skip a correction when transverse offset divided by perpendicular
        centre distance is no greater than this dimensionless threshold.
    blocked_pairs : iterable of pairs of int, optional
        Zero-based active-node pairs excluded from correction and support.
        This does not create hydraulic barriers in the numerical model.
    barriers : iterable of pairs of int, optional
        Zero-based node pairs on either side of a barrier (for example
        ``DisuGrid.barrier_faces(...).pairs``). No ghost head interpolates
        across them; the barrier connections themselves are still corrected,
        each ghost from its own side.
    strict : bool, default False
        Raise if any selected connection lacks admissible support; otherwise
        warn and return it in ``unsupported`` without a correction record.
    contrast : float, default 10
        For ``k_aware``, the smallest conductivity ratio across a face (either
        direction) treated as a material contact. Smaller ratios are treated
        as the same material and weighted geometrically. ``1`` treats every
        conductivity difference as a contact. Ignored by ``'geometric'``.

    Returns
    -------
    GncCorrections
        One collection of ghost records, unsupported physical connections and
        diagnostics, with MF6/MFUSG export helpers. Nothing is attached to a model.

    Notes
    -----
    Fractions are nonnegative and retain positive host-cell influence. Each
    ghost lies within its host prism. A physical connection receives either a
    valid one-host correction or a complete admissible pair of corrections.
    An unsupported connection retains standard model conductance; this is not
    an accuracy guarantee. ``k_aware`` assumes each face at or above the
    contrast is the material contact. That holds where contacts follow cell
    faces; a stair-stepped dipping contact violates it and gives flow errors
    even when heads match the ideal reference.
    Confined scalar isotropic flow is the tested physical scope; anisotropy
    tensors and head-dependent interpolation weights are not supported.
    """
    if method not in ("geometric", "k_aware"):
        raise ValueError("method must be 'geometric' or 'k_aware'.")
    if connections not in ("3d", "all"):
        raise ValueError("connections must be '3d' or 'all'.")
    if not isinstance(strict, (bool, np.bool_)):
        raise ValueError("strict must be True or False.")
    if isinstance(min_offset, (bool, np.bool_)) or not np.isscalar(min_offset):
        raise ValueError("min_offset must be a finite nonnegative number.")
    try:
        min_offset = float(min_offset)
    except (TypeError, ValueError) as error:
        raise ValueError("min_offset must be a finite nonnegative number.") from error
    if not np.isfinite(min_offset) or min_offset < 0:
        raise ValueError("min_offset must be a finite nonnegative number.")
    if isinstance(contrast, (bool, np.bool_)) or not isinstance(contrast, Real):
        raise ValueError("contrast must be a finite number of at least 1.")
    contrast = float(contrast)
    if not np.isfinite(contrast) or contrast < 1:
        raise ValueError("contrast must be a finite number of at least 1.")
    g = _geometry(grid, blocked_pairs, barriers)
    if method == "k_aware":
        if k is None:
            raise ValueError("k_aware requires current scalar conductivity k.")
        if np.iscomplexobj(k):
            raise ValueError("k must contain real scalar conductivities.")
        try:
            k = np.broadcast_to(np.asarray(k, float), (grid.nodes,))
        except (TypeError, ValueError) as error:
            raise ValueError("k must be scalar or have shape (nodes,).") from error
        if not np.isfinite(k).all() or np.any(k <= 0):
            raise ValueError("k must contain positive finite scalar conductivities.")
    counts, records, unsupported = Counter(), [], []

    def endpoint(n, m, target, link):
        offset = target - g.xyz[n]
        if np.linalg.norm(offset) <= g.tolerance:
            return None, None
        if (
            not g.polygons[n].buffer(g.tolerance).covers(Point(*target[:2]))
            or not g.bottom[n] - g.tolerance <= target[2] <= g.top[n] + g.tolerance
        ):
            return None, "ghost outside host prism"
        neighbours = g.adjacency[n]
        if not neighbours:
            return None, "no interpolation support"
        js = np.array([edge[0] for edge in neighbours], dtype=int)
        vectors = g.xyz[js] - g.xyz[n]
        if method == "k_aware":
            # Only a face with a contrast at or above the threshold is a contact.
            vectors = np.array(
                [
                    vector + (k[n] / k[j] - 1) * cm * normal
                    if max(k[n], k[j]) >= contrast * (1 - 1e-12) * min(k[n], k[j])
                    else vector
                    for vector, (j, _, normal, _, cm) in zip(vectors, neighbours)
                ]
            )
        if not np.isfinite(vectors).all():
            return None, "nonfinite property reconstruction"
        scale = np.maximum(np.max(abs(vectors), axis=0), abs(offset))
        scale = np.maximum(scale, g.tolerance)
        distance = np.linalg.norm(g.xyz[js] - g.xyz[n], axis=1)
        fit = linprog(
            distance / max(distance.max(), g.tolerance) + 1e-6,
            A_eq=vectors.T / scale[:, None],
            b_eq=offset / scale,
            A_ub=np.ones((1, len(js))),
            b_ub=[1 - 1e-8],
            bounds=(0, None),
            method="highs",
        )
        if not fit.success:
            return None, "no bounded interpolation support"
        keep = fit.x > 1e-11
        weights, js = fit.x[keep], js[keep]
        if not 1 <= len(js) <= 5:
            return None, "contributor count outside portable USG limit"
        if (
            not np.isfinite(weights).all()
            or np.any(weights < 0)
            or weights.sum() > 1 - 1e-8 + 1e-12
        ):
            return None, "interpolation fractions violate positive host bound"
        residual = float(np.linalg.norm(weights @ vectors[keep] - offset))
        if residual > g.tolerance:
            return None, "reconstruction residual"
        singular = np.linalg.svd(vectors[keep].T, compute_uv=False)
        positive = singular[singular > 1e-12 * singular.max()]
        if not len(positive):
            return None, "degenerate interpolation support"
        return (
            dict(
                n=int(n),
                m=int(m),
                link=int(link),
                target=target.tolist(),
                contributors=js.tolist(),
                weights=weights.tolist(),
                paths=[[int(j)] for j in js],
                residual=residual,
                effective_rank=len(positive),
                active_condition=float(positive.max() / positive.min()),
                kind="vertical" if grid.links["ihc"][link] == 0 else "horizontal",
            ),
            None,
        )

    links = grid.links
    for link, (a, b, cn, cb) in enumerate(
        zip(links["n"], links["m"], links["cl_n"], links["cl_m"])
    ):
        a, b = int(a), int(b)
        normal, distance = g.normals[link], float(cn + cb)
        delta = g.xyz[b] - g.xyz[a] - distance * normal
        if np.linalg.norm(delta) <= max(g.tolerance, min_offset * distance):
            counts["aligned_or_below_threshold"] += 1
            continue
        if (
            connections == "3d"
            and links["ihc"][link] != 0
            and abs(g.xyz[b, 2] - g.xyz[a, 2]) <= g.tolerance
        ):
            counts["outside_3d_scope"] += 1
            continue
        if tuple(sorted((a, b))) in g.blocked:
            reason = "excluded connection"
        elif abs(delta @ normal) > g.tolerance:
            reason = "normal-distance mismatch or gap"
        else:
            reason = "no bounded interpolation support"
            orientations = [(a, b, delta), (b, a, -delta)]
            if g.volume[b] > g.volume[a]:
                orientations.reverse()
            for n, m, offset in orientations:
                record, reason = endpoint(n, m, g.xyz[n] + offset, link)
                if reason is None:
                    if record is not None:
                        records.append(record)
                    counts["single_host_links"] += 1
                    break
            if reason is None:
                continue
            targets = None
            if links["ihc"][link] == 0:
                contact = g.polygons[a].intersection(g.polygons[b])
                if contact.is_empty or contact.area <= 0:
                    reason = "missing vertical contact footprint"
                else:
                    xy = contact.centroid.coords[0]
                    targets = (np.r_[xy, g.xyz[a, 2]], np.r_[xy, g.xyz[b, 2]])
            else:
                contact = g.polygons[a].boundary.intersection(g.polygons[b].boundary)
                lower, upper = max(g.bottom[a], g.bottom[b]), min(g.top[a], g.top[b])
                if contact.is_empty or contact.length <= g.tolerance or upper <= lower:
                    reason = "missing horizontal contact"
                else:
                    face = np.r_[contact.centroid.coords[0], (lower + upper) / 2]
                    targets = (face - cn * normal, face + cb * normal)
            if targets is not None:
                ra, ea = endpoint(a, b, targets[0], link)
                rb, eb = endpoint(b, a, targets[1], link)
                reason = ea or eb
                if reason is None:
                    added = [r for r in (ra, rb) if r is not None]
                    records.extend(added)
                    counts["paired_target_links"] += 1
                    counts[
                        "two_ghost_links" if len(added) == 2 else "one_ghost_links"
                    ] += 1
                    continue
        unsupported.append(dict(link=link, n=a, m=b, reason=reason))
        counts[reason] += 1
    if strict and unsupported:
        raise ValueError(
            f"{len(unsupported)} selected GNC connections lack support: {dict(counts)}"
        )
    if unsupported:
        warnings.warn(
            f"{len(unsupported)} selected connections left without GNC; "
            "inspect result.unsupported.",
            RuntimeWarning,
            stacklevel=2,
        )
    return GncCorrections(
        records=records,
        unsupported=unsupported,
        diagnostics=dict(counts),
        nodes=grid.nodes,
        method=method,
        connections=connections,
        min_offset=min_offset,
        contrast=contrast if method == "k_aware" else None,
    )
