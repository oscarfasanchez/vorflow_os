"""Inspectable ghost-node corrections and engine-specific package data.

This module returns FloPy constructor arguments without importing FloPy or
attaching a package to a model. Node numbers in the collection are zero-based.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from math import fsum
from operator import index
from pathlib import Path

import numpy as np


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer.")
    try:
        return index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer.") from error


@dataclass
class GncCorrections:
    """Correction records, unsupported connections and construction diagnostics.

    Each record supplies zero-based ``n``, ``m``, ``contributors`` and
    ``weights``. Its ghost head is ``h[n] + sum(w * (h[j] - h[n]))``.
    Geometry and diagnostic fields such as ``link``, ``target``, ``residual``,
    ``kind``, ``paths`` and conditioning measures remain available in records.
    Two records can correct opposite endpoints of the same physical connection.

    Inputs are copied so serialization cannot alter the caller's records.
    Fractions are nonnegative, with a strictly positive remaining host fraction.
    At most five contributors per record are supported by both engines.
    ``contrast`` only records the K-aware threshold the records were built
    with; ``estimate_gnc`` leaves it None for ``method="geometric"``, where it
    has no effect.
    """

    records: list[dict] = field(default_factory=list)
    unsupported: list[dict] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    nodes: int = 0
    method: str = "geometric"
    connections: str = "3d"
    min_offset: float = 0.0
    contrast: float | None = None

    def __post_init__(self):
        self.nodes = _integer(self.nodes, "nodes")
        if self.nodes < 0:
            raise ValueError("nodes must be nonnegative.")
        try:
            self.min_offset = float(self.min_offset)
        except (TypeError, ValueError) as error:
            raise ValueError("min_offset must be finite and nonnegative.") from error
        if not np.isfinite(self.min_offset) or self.min_offset < 0:
            raise ValueError("min_offset must be finite and nonnegative.")
        if self.contrast is not None:
            try:
                self.contrast = float(self.contrast)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "contrast must be None or a finite number of at least 1."
                ) from error
            if not np.isfinite(self.contrast) or self.contrast < 1:
                raise ValueError(
                    "contrast must be None or a finite number of at least 1."
                )
        if not isinstance(self.records, list):
            raise ValueError("records must be a list of dictionaries.")
        self.records = deepcopy(self.records)
        self.unsupported = deepcopy(self.unsupported)
        self.diagnostics = deepcopy(self.diagnostics)
        self._validated_rows()

    def _node(self, value):
        node = _integer(value, "GNC node index")
        if not 0 <= node < self.nodes:
            raise ValueError(f"GNC node index {node} is outside [0, {self.nodes}).")
        return node

    def _validated_rows(self):
        rows = []
        for record in self.records:
            if not isinstance(record, dict) or not {
                "n",
                "m",
                "contributors",
                "weights",
            }.issubset(record):
                raise ValueError(
                    "Each GNC record needs n, m, contributors and weights."
                )
            n, m = self._node(record["n"]), self._node(record["m"])
            if n == m:
                raise ValueError("GNC endpoints n and m must be different nodes.")
            try:
                contributors = np.asarray(record["contributors"])
                raw_weights = np.asarray(record["weights"])
                if np.iscomplexobj(raw_weights):
                    raise ValueError("Complex fractions are not supported.")
                weights = raw_weights.astype(float)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "GNC contributors and weights must be one-dimensional numeric arrays."
                ) from error
            if contributors.ndim != 1 or weights.ndim != 1:
                raise ValueError(
                    "GNC contributors and weights must be one-dimensional."
                )
            if not len(contributors) or len(contributors) != len(weights):
                raise ValueError(
                    "GNC contributors and weights must have equal nonzero length."
                )
            if len(contributors) > 5:
                raise ValueError("GNC supports at most five contributors per record.")
            contributors = [self._node(j) for j in contributors]
            if not np.isfinite(weights).all() or np.any(weights < 0):
                raise ValueError("GNC weights must be finite and nonnegative.")
            if np.any(weights >= 1) or fsum(weights) >= 1:
                raise ValueError(
                    "GNC weights must leave a positive host fraction (sum < 1)."
                )
            rows.append((n, m, contributors, weights.tolist()))
        return rows

    def _padded_rows(self):
        rows = self._validated_rows()
        width = max((len(weights) for _, _, _, weights in rows), default=0)
        padded = []
        for n, m, contributors, weights in rows:
            repeat = width - len(weights) + 1
            padded.append(
                (
                    n,
                    m,
                    [contributors[0]] * repeat + contributors[1:],
                    [weights[0] / repeat] * repeat + weights[1:],
                )
            )
        return width, padded

    @property
    def corrected_connections(self):
        """Number of unique physical endpoint pairs, counting both hosts once."""
        return len({tuple(sorted((n, m))) for n, m, _, _ in self._validated_rows()})

    @property
    def record_count(self):
        """Number of ghost records; this is the engine's NUMGNC."""
        return len(self.records)

    @property
    def experimental(self):
        """Whether the collection uses the experimental K-aware construction."""
        return self.method == "k_aware"

    @property
    def rings(self):
        """Largest recorded support-path length, or one for an empty collection."""
        return max(
            (len(path) for record in self.records for path in record.get("paths", [])),
            default=1,
        )

    def get_gridprops_mf6_gnc(self, explicit=False):
        """Return ``ModflowGwfgnc`` kwargs, or None when no package is needed.

        Node IDs use FloPy's zero-based one-element tuples. Do not combine GNC
        with XT3D in the same MODFLOW 6 model. No model or package is modified.
        """
        width, rows = self._padded_rows()
        if not rows:
            return None
        return {
            "explicit": bool(explicit),
            "numgnc": len(rows),
            "numalphaj": width,
            "gncdata": [
                ((n,), (m,), *[(j,) for j in contributors], *weights)
                for n, m, contributors, weights in rows
            ],
        }

    def get_gridprops_mfusg_gnc(self, explicit=True):
        """Return ``MfUsgGnc`` kwargs, or None when no package is needed.

        Node IDs are zero-based integers. FloPy stores USG fractions as float32;
        after ``model.write_input()``, call ``write_mfusg_gnc(package.fn_path)``
        to preserve the collection's double-precision fractions in native input.
        """
        width, rows = self._padded_rows()
        if not rows:
            return None
        return {
            "numgnc": len(rows),
            "numalphaj": width,
            "isymgncn": int(explicit),
            "gncdata": [
                (n, m, *contributors, *weights) for n, m, contributors, weights in rows
            ],
        }

    def write_mfusg_gnc(self, filepath, explicit=True):
        """Write native one-based USG input with 17-digit fractions.

        Call after FloPy writes its registered GNC package, using the same
        ``explicit`` choice. An empty collection is a no-op: an existing file
        is left untouched. Returns the written Path, or None for no correction.
        """
        width, rows = self._padded_rows()
        if not rows:
            return None
        lines = [
            "# GNC: double-precision fractions, one-based nodes",
            f"0 0 {len(rows)} {width} 0 {int(explicit)} 0",
        ]
        lines.extend(
            " ".join(
                [
                    str(n + 1),
                    str(m + 1),
                    *[str(j + 1) for j in contributors],
                    *[f"{weight:.17g}" for weight in weights],
                ]
            )
            for n, m, contributors, weights in rows
        )
        path = Path(filepath)
        path.write_text("\n".join(lines) + "\n", encoding="ascii")
        return path
