from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd

from .fields import (
    AutoExponentialField,
    AutoLinearField,
    ConstantField,
    ExponentialField,
    MeshField,
    ThresholdField,
)


def _extract_int_tags(entry_list: Iterable[Any]) -> List[int]:
    """Extract integer tags from gmsh dimtags or plain tag lists."""
    out: List[int] = []
    for item in entry_list or []:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            out.append(int(item[1]))
        else:
            out.append(int(item))
    return out


def _as_bool(val: Any, default: bool = True) -> bool:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return default
    return bool(val)


def _is_auto_field(field: MeshField) -> bool:
    return isinstance(field, (AutoLinearField, AutoExponentialField))


@dataclass(frozen=True)
class _GroupKey:
    field: MeshField
    feature_lc: Optional[float]
    tag_kinds: Tuple[str, ...]
    restricted: bool


@dataclass
class _FieldGroup:
    field: MeshField
    feature_lc: Optional[float]
    points: Set[int]
    lines: Set[int]
    surfaces: Set[int]
    restricted: bool
    restrict_surfaces: Set[int]


def build_background_field(
    gmsh_api,
    gmsh_map: Dict[str, Dict[int, List[Any]]],
    polygons_gdf,
    lines_gdf,
    points_gdf,
    background_lc: float,
    *,
    verbosity: int = 0,
) -> Optional[int]:
    """Build and set a single Gmsh background field.

    Behavior:
    - If a feature row has a non-empty `fields` list, those recipes are used.
    - Otherwise, legacy defaults (Threshold-based) are generated.
    - All resulting fields are combined using a global Min.

    Returns:
        The id of the Min background field (or None if no fields created).
    """
    if background_lc is None:
        raise ValueError("background_lc must be set to a positive float")

    bg = float(background_lc)
    if bg <= 0:
        raise ValueError(f"background_lc must be positive. Got {background_lc}.")

    groups: Dict[_GroupKey, _FieldGroup] = {}

    def add_request(
        field: MeshField,
        *,
        points: Sequence[int] = (),
        lines: Sequence[int] = (),
        surfaces: Sequence[int] = (),
        restrict_to_surfaces: Optional[Sequence[int]] = None,
        feature_lc: Optional[float] = None,
    ) -> None:
        tag_kinds: List[str] = []
        if points:
            tag_kinds.append("points")
        if lines:
            tag_kinds.append("lines")
        if surfaces:
            tag_kinds.append("surfaces")

        restricted = bool(restrict_to_surfaces)
        lc_key = float(feature_lc) if (_is_auto_field(field) and feature_lc is not None) else None

        key = _GroupKey(field=field, feature_lc=lc_key, tag_kinds=tuple(tag_kinds), restricted=restricted)
        if key not in groups:
            groups[key] = _FieldGroup(
                field=field,
                feature_lc=lc_key,
                points=set(),
                lines=set(),
                surfaces=set(),
                restricted=restricted,
                restrict_surfaces=set(),
            )

        grp = groups[key]
        grp.points.update(int(t) for t in points)
        grp.lines.update(int(t) for t in lines)
        grp.surfaces.update(int(t) for t in surfaces)
        if restrict_to_surfaces:
            grp.restrict_surfaces.update(int(t) for t in restrict_to_surfaces)

    def get_row_param(row, key: str, default: float) -> float:
        if key in row and not pd.isna(row[key]):
            return float(row[key])
        return float(default)

    # --- Points ---
    for idx, row in points_gdf.iterrows():
        if idx not in gmsh_map.get("points", {}):
            continue

        pt_tags = _extract_int_tags(gmsh_map["points"][idx])
        lc = max(get_row_param(row, "lc", 5.0), 0.001)
        d_min = get_row_param(row, "dist_min", lc * 2.0)
        d_max = get_row_param(row, "dist_max", bg * 1.5)

        recipes = row.get("fields")
        if recipes:
            for f in recipes:
                if isinstance(f, MeshField):
                    add_request(f, points=pt_tags, feature_lc=lc)
        else:
            add_request(ThresholdField(lc, d_min, d_max, bg), points=pt_tags)

    # --- Lines (including virtual straddle points) ---
    for idx, row in lines_gdf.iterrows():
        lc = max(get_row_param(row, "lc", 10.0), 0.001)
        d_min = get_row_param(row, "dist_min", lc * 1.0)
        d_max = get_row_param(row, "dist_max", bg * 1.5)

        recipes = row.get("fields")

        if idx in gmsh_map.get("lines", {}):
            curve_tags = _extract_int_tags(gmsh_map["lines"][idx])
            if recipes:
                for f in recipes:
                    if isinstance(f, MeshField):
                        add_request(f, lines=curve_tags, feature_lc=lc)
            else:
                add_request(ThresholdField(lc, d_min, d_max, bg), lines=curve_tags)

        elif idx in gmsh_map.get("points", {}):
            # Barrier/straddle lines are represented by point pairs.
            is_barrier = row.get("is_barrier", False)
            straddle = row.get("straddle_width", 0)
            if is_barrier or (straddle and float(straddle) > 0):
                pt_tags = _extract_int_tags(gmsh_map["points"][idx])
                d_min2 = get_row_param(row, "dist_min", lc * 2.0)
                d_max2 = get_row_param(row, "dist_max", bg * 1.5)

                if recipes:
                    for f in recipes:
                        if isinstance(f, MeshField):
                            add_request(f, points=pt_tags, feature_lc=lc)
                else:
                    add_request(ThresholdField(lc, d_min2, d_max2, bg), points=pt_tags)

    # --- Polygons ---
    for idx, row in polygons_gdf.iterrows():
        target_lc = get_row_param(row, "lc", bg)

        densify_val = row.get("densify", None)
        if isinstance(densify_val, (int, float)) and not isinstance(densify_val, bool) and densify_val > 0:
            boundary_lc = min(target_lc, float(densify_val))
        else:
            boundary_lc = target_lc

        recipes = row.get("fields")

        # Embedded polygon surfaces.
        if idx in gmsh_map.get("surfaces", {}):
            surf_tags = _extract_int_tags(gmsh_map["surfaces"][idx])

            dim_tags = [(2, int(t)) for t in surf_tags]
            boundaries = gmsh_api.model.getBoundary(dim_tags, combined=True, oriented=False, recursive=False)
            curve_tags = [int(b[1]) for b in boundaries if b[0] == 1]

            if recipes:
                # For polygons, interpret user recipes as interior (restricted) fields.
                for f in recipes:
                    if isinstance(f, MeshField):
                        add_request(
                            f,
                            lines=curve_tags,
                            restrict_to_surfaces=surf_tags,
                            feature_lc=target_lc,
                        )
                continue

            # Legacy interior behavior.
            if boundary_lc < bg or target_lc < bg:
                if curve_tags and boundary_lc < target_lc:
                    d_min = get_row_param(row, "dist_min", 0.0)
                    d_max_in = get_row_param(row, "dist_max_in", -1.0)
                    if d_max_in > d_min:
                        d_max_inner = d_max_in
                    else:
                        d_max_inner = d_min + (boundary_lc * 5.0)
                        d_max_inner = max(d_max_inner, d_min + (target_lc - boundary_lc) * 0.2)

                    add_request(
                        ThresholdField(boundary_lc, d_min, d_max_inner, target_lc),
                        lines=curve_tags,
                        restrict_to_surfaces=surf_tags,
                    )
                else:
                    add_request(ConstantField(target_lc), restrict_to_surfaces=surf_tags)

            # Legacy exterior behavior.
            d_max_out = get_row_param(row, "dist_max_out", 0.0)
            if d_max_out > 0 and curve_tags and boundary_lc < bg:
                d_min = get_row_param(row, "dist_min", 0.0)
                add_request(ThresholdField(boundary_lc, d_min, d_max_out, bg), lines=curve_tags)

        # Field-only polygon curves.
        elif idx in gmsh_map.get("poly_curves", {}):
            curve_tags = _extract_int_tags(gmsh_map["poly_curves"][idx])

            if recipes:
                for f in recipes:
                    if isinstance(f, MeshField):
                        add_request(f, lines=curve_tags, feature_lc=target_lc)
                continue

            d_max_out = get_row_param(row, "dist_max_out", 0.0)
            if curve_tags and d_max_out > 0 and boundary_lc < bg:
                d_min = get_row_param(row, "dist_min", 0.0)
                add_request(ThresholdField(boundary_lc, d_min, d_max_out, bg), lines=curve_tags)

    # --- Materialize fields in gmsh ---
    field_ids: List[int] = []

    for grp_key, grp in groups.items():
        tags_dict = {
            "points": sorted(grp.points),
            "lines": sorted(grp.lines),
            "surfaces": sorted(grp.surfaces),
        }

        feature_lc = grp.feature_lc
        if feature_lc is None and _is_auto_field(grp.field):
            # If an auto-field got grouped without lc (shouldn't happen), skip.
            continue

        base_id = grp.field.create(gmsh_api, tags_dict, bg, feature_lc=feature_lc)
        if base_id is None:
            continue

        if grp.restricted:
            if not grp.restrict_surfaces:
                continue
            f_rest = gmsh_api.model.mesh.field.add("Restrict")
            gmsh_api.model.mesh.field.setNumber(f_rest, "IField", base_id)
            gmsh_api.model.mesh.field.setNumbers(
                f_rest, "SurfacesList", [int(t) for t in sorted(grp.restrict_surfaces)]
            )
            field_ids.append(int(f_rest))
        else:
            field_ids.append(int(base_id))

    # Always include a background constant.
    field_ids.append(int(ConstantField(bg).create(gmsh_api, {}, bg)))

    if not field_ids:
        return None

    f_min = gmsh_api.model.mesh.field.add("Min")
    gmsh_api.model.mesh.field.setNumbers(f_min, "FieldsList", [float(i) for i in field_ids])
    gmsh_api.model.mesh.field.setAsBackgroundMesh(f_min)

    if verbosity > 1:
        print(f"FieldGraph: {len(groups)} groups -> {len(field_ids)} fields (plus Min).")

    return int(f_min)
