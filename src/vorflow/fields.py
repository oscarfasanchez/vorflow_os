import math


def _has_any_entities(tags_dict: dict) -> bool:
    return bool(tags_dict.get("points") or tags_dict.get("lines") or tags_dict.get("surfaces"))


def _normalize_tags(tags_dict: dict) -> dict:
    """Return a copy with int tags and only known keys."""
    out = {"points": [], "lines": [], "surfaces": []}
    if not tags_dict:
        return out
    for key in ("points", "lines", "surfaces"):
        vals = tags_dict.get(key) or []
        out[key] = [int(v) for v in vals]
    return out

class MeshField:
    """
    Base class for all mesh size fields.
    """
    def create(self, gmsh_api, tags_dict, background_lc, feature_lc=None):
        """
        Creates the Gmsh field(s) and returns the field ID.
        
        Args:
            gmsh_api: The gmsh module.
            tags_dict (dict): Dictionary of tags {'points': [], 'lines': [], 'surfaces': []}.
            background_lc (float): Global background mesh size.
            feature_lc (float, optional): The target resolution of the specific feature group.
        """
        raise NotImplementedError("Subclasses must implement create()")

    def __eq__(self, other):
        """Equality check for grouping."""
        return isinstance(other, self.__class__) and self.__dict__ == other.__dict__

    def __hash__(self):
        """Hash for dictionary keys."""
        # Create a tuple of sorted item pairs to ensure consistent hashing.
        # If a value is unhashable (e.g. list/dict), fall back to repr().
        items = []
        for k, v in sorted(self.__dict__.items()):
            try:
                hash(v)
                items.append((k, v))
            except TypeError:
                items.append((k, repr(v)))
        return hash((self.__class__.__name__, tuple(items)))

# --- Manual Fields ---

class ConstantField(MeshField):
    def __init__(self, size):
        self.size = float(size)

    def create(self, gmsh_api, tags_dict, background_lc, feature_lc=None):
        tags_dict = _normalize_tags(tags_dict)
        f_const = gmsh_api.model.mesh.field.add("MathEval")
        gmsh_api.model.mesh.field.setString(f_const, "F", str(float(self.size)))

        # If entity lists are provided, apply this constant only to those entities.
        # This is primarily useful for polygon interior sizing (via SurfacesList).
        if _has_any_entities(tags_dict):
            f_rest = gmsh_api.model.mesh.field.add("Restrict")
            gmsh_api.model.mesh.field.setNumber(f_rest, "IField", f_const)
            if tags_dict.get("points"):
                gmsh_api.model.mesh.field.setNumbers(f_rest, "PointsList", tags_dict["points"])
            if tags_dict.get("lines"):
                gmsh_api.model.mesh.field.setNumbers(f_rest, "CurvesList", tags_dict["lines"])
            if tags_dict.get("surfaces"):
                gmsh_api.model.mesh.field.setNumbers(f_rest, "SurfacesList", tags_dict["surfaces"])
            return f_rest

        return f_const


class ThresholdField(MeshField):
    def __init__(self, size_min, dist_min, dist_max, size_max=None):
        self.size_min = float(size_min)
        self.dist_min = float(dist_min)
        self.dist_max = float(dist_max)
        self.size_max = float(size_max) if size_max is not None else None

    def create(self, gmsh_api, tags_dict, background_lc, feature_lc=None):
        tags_dict = _normalize_tags(tags_dict)
        # 1. Distance Field (can combine points, curves, surfaces)
        f_dist = gmsh_api.model.mesh.field.add("Distance")
        
        has_entities = False
        if tags_dict.get('points'):
            gmsh_api.model.mesh.field.setNumbers(f_dist, "PointsList", tags_dict['points'])
            has_entities = True
        if tags_dict.get('lines'):
            gmsh_api.model.mesh.field.setNumbers(f_dist, "CurvesList", tags_dict['lines'])
            has_entities = True
        if tags_dict.get('surfaces'):
            gmsh_api.model.mesh.field.setNumbers(f_dist, "SurfacesList", tags_dict['surfaces'])
            has_entities = True
            
        if not has_entities:
            gmsh_api.model.mesh.field.remove(f_dist)
            return None

        # 2. Threshold Field
        f_thresh = gmsh_api.model.mesh.field.add("Threshold")
        gmsh_api.model.mesh.field.setNumber(f_thresh, "InField", f_dist)
        gmsh_api.model.mesh.field.setNumber(f_thresh, "SizeMin", self.size_min)
        gmsh_api.model.mesh.field.setNumber(f_thresh, "SizeMax", self.size_max if self.size_max else background_lc)
        gmsh_api.model.mesh.field.setNumber(f_thresh, "DistMin", self.dist_min)
        gmsh_api.model.mesh.field.setNumber(f_thresh, "DistMax", self.dist_max)
        
        return f_thresh

class ExponentialField(MeshField):
    def __init__(self, size_min, decay_length, size_max=None):
        self.size_min = float(size_min)
        self.decay_length = float(decay_length)
        self.size_max = float(size_max) if size_max is not None else None

    def create(self, gmsh_api, tags_dict, background_lc, feature_lc=None):
        tags_dict = _normalize_tags(tags_dict)
        f_dist = gmsh_api.model.mesh.field.add("Distance")
        
        has_entities = False
        if tags_dict.get('points'):
            gmsh_api.model.mesh.field.setNumbers(f_dist, "PointsList", tags_dict['points'])
            has_entities = True
        if tags_dict.get('lines'):
            gmsh_api.model.mesh.field.setNumbers(f_dist, "CurvesList", tags_dict['lines'])
            has_entities = True
        if tags_dict.get('surfaces'):
            gmsh_api.model.mesh.field.setNumbers(f_dist, "SurfacesList", tags_dict['surfaces'])
            has_entities = True
            
        if not has_entities:
            gmsh_api.model.mesh.field.remove(f_dist)
            return None

        s_max = self.size_max if self.size_max else background_lc
        
        f_math = gmsh_api.model.mesh.field.add("MathEval")
        expr = f"{s_max} - ({s_max} - {self.size_min}) * Exp(-F{f_dist} / {self.decay_length})"
        gmsh_api.model.mesh.field.setString(f_math, "F", expr)
        return f_math

# --- Auto Fields ---

class AutoLinearField(MeshField):
    def __init__(self, growth_factor=1.2):
        self.fac = float(growth_factor)

    def create(self, gmsh_api, tags_dict, background_lc, feature_lc=None):
        if feature_lc is None: return None
        
        cs = float(feature_lc)
        cs_dom = float(background_lc)
        fac = self.fac

        if fac <= 1.0: raise ValueError("Growth factor must be > 1.0")
        if cs >= cs_dom: return None

        # Calculate transition
        min_trans_cells = math.log(cs_dom / cs) / math.log(fac)
        min_trans_dist = cs * ((fac ** min_trans_cells) - 1) / math.log(fac)
        
        dist_min = cs / 2.0
        dist_max = min_trans_dist / 2.0

        # Delegate to ThresholdField logic
        # We create a temporary ThresholdField to reuse its create logic
        temp_field = ThresholdField(cs, dist_min, dist_max, cs_dom)
        return temp_field.create(gmsh_api, tags_dict, background_lc)

class AutoExponentialField(MeshField):
    def __init__(self, growth_factor=1.1):
        self.fac = float(growth_factor)

    def create(self, gmsh_api, tags_dict, background_lc, feature_lc=None):
        if feature_lc is None: return None
        
        cs = float(feature_lc)
        fac = self.fac
        
        if fac <= 1.0: raise ValueError("Growth factor must be > 1.0")

        tags_dict = _normalize_tags(tags_dict)

        f_dist = gmsh_api.model.mesh.field.add("Distance")
        has_entities = False
        if tags_dict.get('points'):
            gmsh_api.model.mesh.field.setNumbers(f_dist, "PointsList", tags_dict['points'])
            has_entities = True
        if tags_dict.get('lines'):
            gmsh_api.model.mesh.field.setNumbers(f_dist, "CurvesList", tags_dict['lines'])
            has_entities = True
        if tags_dict.get('surfaces'):
            gmsh_api.model.mesh.field.setNumbers(f_dist, "SurfacesList", tags_dict['surfaces'])
            has_entities = True
            
        if not has_entities:
            gmsh_api.model.mesh.field.remove(f_dist)
            return None

        f_math = gmsh_api.model.mesh.field.add("MathEval")
        log_fac = math.log(fac)
        expr = f"{cs} * {fac}^(Log(1 + F{f_dist} * 2 * {log_fac} / {cs}) / {log_fac})"
        
        gmsh_api.model.mesh.field.setString(f_math, "F", expr)
        return f_math


class RestrictField(MeshField):
    """Restrict another field to a set of entities."""

    def __init__(self, inner_field: MeshField, points=None, lines=None, surfaces=None):
        self.inner_field = inner_field
        self.points = tuple(int(p) for p in (points or ()))
        self.lines = tuple(int(l) for l in (lines or ()))
        self.surfaces = tuple(int(s) for s in (surfaces or ()))

    def create(self, gmsh_api, tags_dict, background_lc, feature_lc=None):
        inner_id = self.inner_field.create(gmsh_api, tags_dict, background_lc, feature_lc=feature_lc)
        if inner_id is None:
            return None

        f_rest = gmsh_api.model.mesh.field.add("Restrict")
        gmsh_api.model.mesh.field.setNumber(f_rest, "IField", inner_id)
        if self.points:
            gmsh_api.model.mesh.field.setNumbers(f_rest, "PointsList", list(self.points))
        if self.lines:
            gmsh_api.model.mesh.field.setNumbers(f_rest, "CurvesList", list(self.lines))
        if self.surfaces:
            gmsh_api.model.mesh.field.setNumbers(f_rest, "SurfacesList", list(self.surfaces))
        return f_rest


class MinField(MeshField):
    """Take the minimum of multiple fields."""

    def __init__(self, fields):
        self.fields = tuple(fields)

    def create(self, gmsh_api, tags_dict, background_lc, feature_lc=None):
        ids = []
        for f in self.fields:
            fid = f.create(gmsh_api, tags_dict, background_lc, feature_lc=feature_lc)
            if fid is not None:
                ids.append(float(fid))

        if not ids:
            return None

        f_min = gmsh_api.model.mesh.field.add("Min")
        gmsh_api.model.mesh.field.setNumbers(f_min, "FieldsList", ids)
        return f_min