import math

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
        # Create a tuple of sorted item pairs to ensure consistent hashing
        return hash((self.__class__.__name__, tuple(sorted(self.__dict__.items()))))


class DistanceField(MeshField):
    """Creates a Gmsh Distance field from points/lines/surfaces tags."""

    def __init__(self, include_surfaces=True, sampling=20):
        self.include_surfaces = bool(include_surfaces)
        self.sampling = int(sampling)

    def create(self, gmsh_api, tags_dict):
        f_dist = gmsh_api.model.mesh.field.add("Distance")

        has_entities = False
        if tags_dict.get('points'):
            gmsh_api.model.mesh.field.setNumbers(f_dist, "PointsList", tags_dict['points'])
            has_entities = True
        if tags_dict.get('lines'):
            gmsh_api.model.mesh.field.setNumbers(f_dist, "CurvesList", tags_dict['lines'])
            gmsh_api.model.mesh.field.setNumber(f_dist, 'Sampling', self.sampling)
            has_entities = True
        if self.include_surfaces and tags_dict.get('surfaces'):#TODO check if loops needed
            gmsh_api.model.mesh.field.setNumbers(f_dist, "SurfacesList", tags_dict['surfaces'])
            gmsh_api.model.mesh.field.setNumber(f_dist, 'Sampling', self.sampling)
            has_entities = True

        if not has_entities:
            gmsh_api.model.mesh.field.remove(f_dist)
            return None

        return f_dist

# --- Manual Fields ---

class ConstantField(MeshField):
    def __init__(self, size):
        self.size = float(size)

    def create(self, gmsh_api, tags_dict, background_lc, feature_lc=None):
        const = gmsh_api.model.mesh.field.add("Constant")
        gmsh_api.model.mesh.field.setNumber(const, "VIn", self.size)
        gmsh_api.model.mesh.field.setNumber(const, "VOut", background_lc)
        return const


class ThresholdField(MeshField):
    def __init__(self, size_min, dist_min, dist_max, size_max=None, sampling=20):
        self.size_min = float(size_min)
        self.dist_min = float(dist_min)
        self.dist_max = float(dist_max)
        self.size_max = float(size_max) if size_max is not None else None
        self.sampling = int(sampling)
    def create(self, gmsh_api, tags_dict, background_lc, feature_lc=None, constant_in =False):
        # 1. Distance Field (can combine points, curves, surfaces)
        f_dist = DistanceField(include_surfaces=True, sampling=self.sampling).create(
            gmsh_api, tags_dict
        )
        if f_dist is None:
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
    def __init__(self, size_min, decay_length, size_max=None, sampling=20):
        self.size_min = float(size_min)
        self.decay_length = float(decay_length)
        self.size_max = float(size_max) if size_max is not None else None
        self.sampling = int(sampling)

    def create(self, gmsh_api, tags_dict, background_lc, feature_lc=None):
        f_dist = DistanceField(include_surfaces=False, sampling=self.sampling).create(
            gmsh_api, tags_dict
        )
        if f_dist is None:
            return None

        s_max = self.size_max if self.size_max else background_lc
        
        f_math = gmsh_api.model.mesh.field.add("MathEval")
        expr = f"{s_max} - ({s_max} - {self.size_min}) * Exp(-F{f_dist} / {self.decay_length})"
        gmsh_api.model.mesh.field.setString(f_math, "F", expr)
        return f_math

# --- Auto Fields ---

class AutoLinearField(MeshField):
    def __init__(self, growth_factor=1.2, sampling=20):
        self.fac = float(growth_factor)
        self.sampling = int(sampling)

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
        temp_field = ThresholdField(cs, dist_min, dist_max, cs_dom, sampling=self.sampling)
        return temp_field.create(gmsh_api, tags_dict, background_lc)

class AutoExponentialField(MeshField):
    def __init__(self, growth_factor=1.1):
        self.fac = float(growth_factor)

    def create(self, gmsh_api, tags_dict, background_lc, feature_lc=None, sampling=10):
        if feature_lc is None: return None
        
        cs = float(feature_lc)
        fac = self.fac
        
        if fac <= 1.0: raise ValueError("Growth factor must be > 1.0")

        f_dist = DistanceField(include_surfaces=False, sampling=int(sampling)).create(
            gmsh_api, tags_dict
        )
        if f_dist is None:
            return None

        f_math = gmsh_api.model.mesh.field.add("MathEval")
        log_fac = math.log(fac)
        expr = f"{cs} * {fac}^(Log(1 + F{f_dist} * 2 * {log_fac} / {cs}) / {log_fac})"

        gmsh_api.model.mesh.field.setString(f_math, "F", expr)
        return f_math