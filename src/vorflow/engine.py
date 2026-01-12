import gmsh
import sys
import math
import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid
from .fields import MeshField, ThresholdField, ExponentialField, AutoLinearField, AutoExponentialField, ConstantField

class MeshGenerator:
    def __init__(self, background_lc=None,verbosity=0, mesh_algorithm=6, smoothing_steps=10, optimization_cycles=2):
        """
        Initializes the Gmsh-based mesh generator.

        This class is responsible for taking clean geometric inputs and using Gmsh
        to produce a high-quality triangular mesh.

        Args:
            background_lc (float, optional): The default target mesh size for areas
                not controlled by a specific refinement field.
            verbosity (int): Gmsh verbosity level (0=silent, 1=basic, 2=debug).
            mesh_algorithm (int): The 2D mesh algorithm to use. Common choices are
                5 (Delaunay) for speed or 6 (Frontal-Delaunay) for quality.
            smoothing_steps (int): Number of internal Lloyd smoothing iterations
                performed by Gmsh during mesh generation.
            optimization_cycles (int): Number of explicit optimization passes
                (e.g., Relocate2D, Laplace2D) to run after the initial mesh is generated.
        """
        self.background_lc = background_lc
        self.verbosity = verbosity
        self.mesh_algorithm = mesh_algorithm
        self.smoothing_steps = smoothing_steps
        self.optimization_cycles = optimization_cycles

        self.initialized = False
        self.nodes = None
        self.node_tags = None
        self.zones_gdf = None
    
    def _force_close_polygon(self, poly):
        """Ensure a polygon's exterior and interior rings are closed."""
        if not isinstance(poly, Polygon):
            return poly

        # Close exterior ring
        if poly.exterior.coords[0] != poly.exterior.coords[-1]:
            exterior_coords = list(poly.exterior.coords)
            exterior_coords.append(exterior_coords[0])
            poly = Polygon(exterior_coords, [list(i.coords) for i in poly.interiors])

        # Close interior rings
        new_interiors = []
        for interior in poly.interiors:
            if interior.coords[0] != interior.coords[-1]:
                interior_coords = list(interior.coords)
                interior_coords.append(interior_coords[0])
                new_interiors.append(interior_coords)
            else:
                new_interiors.append(list(interior.coords))
        
        return Polygon(poly.exterior, new_interiors)

    def _initialize_gmsh(self):
        if not gmsh.is_initialized():
            gmsh.initialize()
            gmsh.option.setNumber("General.Verbosity", self.verbosity)
            gmsh.option.setNumber("Geometry.Tolerance", 1e-6)
            gmsh.option.setNumber("Geometry.OCCBooleanPreserveNumbering", 1)
            gmsh.model.add("mesh_model")
            self.initialized = True

    def _finalize_gmsh(self):
        if gmsh.is_initialized():
            gmsh.finalize()
            self.initialized = False

    def _add_geometry(self, polygons_gdf, lines_gdf, points_gdf):
        """
        Transfers Shapely geometries from GeoDataFrames into the Gmsh model.

        This method adds points, lines, and polygons to Gmsh's internal CAD
        kernel (OCC). It also handles special cases like "straddle" lines and
        pre-processes barrier features before fragmenting all geometries to
        create a consistent topological model.
        """
        input_tag_info = {}

        # Non-embedded geometry does not participate in fragmentation.
        # We still track it so mesh-size fields can be applied later.
        nonembedded_point_tags = {}
        nonembedded_line_tags = {}
        nonembedded_surface_tags = {}
        # For non-embedded polygons, we track their boundary curves so size
        # fields can be applied without forcing the polygon to cut/fragment the domain.
        nonembedded_poly_curve_tags = {}

        # Embedded geometry DOES participate in fragmentation.
        embedded_point_tags = []
        embedded_line_tags = []
        embedded_surface_tags = []
        
        def to_key(dim, tag):
            return (int(dim), int(tag))

        def is_embedded(row) -> bool:
            val = row.get('embed', True)
            if pd.isna(val):
                return True
            return bool(val)
        
        # Add all point features to the Gmsh model first.
        for idx, row in points_gdf.iterrows():
            tag = gmsh.model.occ.addPoint(row.geometry.x, row.geometry.y, 0)
            key = to_key(0, tag)
            if is_embedded(row):
                input_tag_info[key] = {'type': 'point', 'id': idx}
                embedded_point_tags.append(key)
            else:
                nonembedded_point_tags.setdefault(int(idx), []).append(key)
            
        # Create a buffer zone around barrier lines. This is used to trim back
        # other lines, preventing their endpoints from interfering with the
        # meshing of the barrier features.
        barrier_buffers = []
        for idx, row in lines_gdf.iterrows():
            val = row.get('is_barrier', False)
            is_barrier = (val is True) or (str(val).lower() in ['true', '1', 'yes'])
            straddle = row.get('straddle_width')
            
            if is_barrier: 
                lc = max(row.get('lc', 10.0), 0.001)
                if straddle and straddle > 0:
                    eps = straddle / 2.0
                else:
                    eps = lc * 0.20
                
                # The trim buffer is made slightly larger than the feature's half-width
                # to ensure a clean separation between standard lines and the
                # sensitive node pairs used for straddle barriers.
                trim_eps = eps * 1.20
                
                buf = row.geometry.buffer(trim_eps, cap_style=2)
                barrier_buffers.append(buf)
        
        barrier_zone = None
        if barrier_buffers:
            barrier_zone = unary_union(barrier_buffers)
            barrier_zone = make_valid(barrier_zone)
            if self.verbosity > 0:
                print(f"Constructed Barrier Zone from {len(barrier_buffers)} barriers.")

        # Add line features to the model, handling barriers and standard lines differently.
        for idx, row in lines_gdf.iterrows():
            val = row.get('is_barrier', False)
            is_barrier = (val is True) or (str(val).lower() in ['true', '1', 'yes'])
            straddle = row.get('straddle_width')
            lc = max(row.get('lc', 10.0), 0.001)
            
            use_virtual_straddle = is_barrier or (straddle is not None and straddle > 0)

            embedded = is_embedded(row)
            
            if use_virtual_straddle:
                # For barriers or "straddle" lines, we don't add the line itself.
                # Instead, we place pairs of points along the line's path. These
                # points will become nodes in the triangular mesh, forcing the
                # subsequent Voronoi cell edges to align with the original line.
                line = row.geometry
                length = line.length
                num_segments = int(max(1, np.ceil(length / lc)))
                distances = np.linspace(0, length, num_segments + 1)
                
                if straddle and straddle > 0:
                    epsilon = straddle / 2.0
                else:
                    epsilon = lc * 0.20
                
                for d in distances:
                    p = line.interpolate(d)
                    t_val = d
                    p_near = line.interpolate(min(t_val + 0.01, length))
                    if t_val >= length - 0.001:
                         p_near = line.interpolate(max(t_val - 0.01, 0))
                         dx, dy = p.x - p_near.x, p.y - p_near.y
                    else:
                         dx, dy = p_near.x - p.x, p_near.y - p.y
                    
                    mag = np.sqrt(dx*dx + dy*dy)
                    if mag == 0: mag = 1
                    dx, dy = dx/mag, dy/mag
                    nx, ny = -dy, dx
                    
                    # Create two points, offset from the original line by the normal.
                    lx, ly = p.x + nx*epsilon, p.y + ny*epsilon
                    lt = gmsh.model.occ.addPoint(lx, ly, 0)
                    k_l = to_key(0, lt)
                    if embedded:#TODO probably this always true for barriers
                        input_tag_info[k_l] = {'type': 'point', 'id': idx}
                        embedded_point_tags.append(k_l)
                    else:
                        nonembedded_point_tags.setdefault(int(idx), []).append(k_l)
                    
                    rx, ry = p.x - nx*epsilon, p.y - ny*epsilon
                    rt = gmsh.model.occ.addPoint(rx, ry, 0)
                    k_r = to_key(0, rt)
                    if embedded:
                        input_tag_info[k_r] = {'type': 'point', 'id': idx}
                        embedded_point_tags.append(k_r)
                    else:
                        nonembedded_point_tags.setdefault(int(idx), []).append(k_r)

            else:
                # This is a standard line feature that will act as a constraint
                # in the mesh, but not a hard barrier.
                geom = row.geometry
                
                # Trim the line against the barrier zone to avoid intersections.
                if barrier_zone:
                    if geom.intersects(barrier_zone):
                        try:
                            original_len = geom.length
                            geom = geom.difference(barrier_zone)
                            
                            if self.verbosity > 1:
                                print(f"  Line {idx} trimmed by barrier (Len: {original_len:.2f} -> {geom.length:.2f})")
                                
                        except Exception as e:
                            print(f"Warning: Failed to trim line {idx}: {e}")
                
                if geom.is_empty:
                    continue
                
                # A line might be split into multiple parts after being trimmed.
                if geom.geom_type == 'LineString':
                    parts = [geom]
                elif geom.geom_type == 'MultiLineString':
                    parts = geom.geoms
                else:
                    parts = []
                
                for part in parts:
                    # Filter out tiny fragments that might remain after trimming.
                    if part.length < 1e-6: continue
                    
                    coords = list(part.coords)
                    if len(coords) < 2: continue
                    
                    # Add each segment of the line to Gmsh.
                    pt_tags = [gmsh.model.occ.addPoint(x, y, 0) for x, y in coords]
                    for i in range(len(pt_tags) - 1):
                        l = gmsh.model.occ.addLine(pt_tags[i], pt_tags[i+1])
                        
                        key = to_key(1, l)
                        if embedded:
                            embedded_line_tags.append(key)
                            input_tag_info[key] = {'type': 'line', 'id': idx}
                        else:
                            nonembedded_line_tags.setdefault(int(idx), []).append(key)

        # Add polygon features to the model.
        if not polygons_gdf.empty:
            print(f"Adding {len(polygons_gdf)} polygons to Gmsh...")
            for idx, row in polygons_gdf.iterrows():
                embedded = is_embedded(row)
                geom = row['geometry']
                if geom.geom_type == 'Polygon':
                    polys = [geom]
                elif geom.geom_type == 'MultiPolygon':
                    polys = geom.geoms
                else:
                    continue
                
                for poly in polys:
                    if poly.is_empty:
                        continue
                        
                    # Ensure the polygon is valid and closed before processing.
                    poly = self._force_close_polygon(poly)

                    def create_loop(coords):
                        # Remove consecutive duplicates and points that are too close
                        clean_coords = []
                        for pt in coords:
                            if not clean_coords:
                                clean_coords.append(pt)
                                continue
                            
                            # Check distance to last point
                            dist = math.sqrt((pt[0]-clean_coords[-1][0])**2 + (pt[1]-clean_coords[-1][1])**2)
                            if dist > 1e-5: # Slightly larger than Gmsh tolerance to be safe
                                clean_coords.append(pt)
                        
                        # Check closure with first point
                        if len(clean_coords) > 1:
                             dist = math.sqrt((clean_coords[0][0]-clean_coords[-1][0])**2 + (clean_coords[0][1]-clean_coords[-1][1])**2)
                             if dist < 1e-5:
                                 clean_coords.pop()
                            
                        if len(clean_coords) < 3:
                            # A polygon must have at least 3 points (triangle)
                            return None

                        p_tags = [gmsh.model.occ.addPoint(x, y, 0) for x, y in clean_coords]
                        l_tags = []
                        for i in range(len(p_tags)):
                            p1 = p_tags[i]
                            p2 = p_tags[(i + 1) % len(p_tags)]
                            try:
                                l_tags.append(gmsh.model.occ.addLine(p1, p2))
                            except Exception as e:
                                print(f"Error adding line {p1}-{p2}: {e}")
                                return None
                        
                        try:
                            loop_tag = gmsh.model.occ.addCurveLoop(l_tags)
                            return loop_tag, l_tags
                        except Exception as e:
                            print(f"Error adding curve loop: {e}")
                            return None, []

                    # 1. Exterior Boundary
                    ext_coords = list(poly.exterior.coords)
                    exterior_loop_tag, exterior_lines = create_loop(ext_coords)
                    
                    if exterior_loop_tag is None:
                        print(f"Warning: Skipping degenerate polygon {idx}")
                        continue

                    # 2. Interior Boundaries (Holes)
                    loops = [exterior_loop_tag]
                    boundary_curve_tags = list(exterior_lines)
                    for interior in poly.interiors:
                        int_coords = list(interior.coords)
                        interior_loop_tag, interior_lines = create_loop(int_coords)
                        if interior_loop_tag is not None:
                            loops.append(interior_loop_tag)
                            boundary_curve_tags.extend(interior_lines)

                    # Create plane surface with holes (embedded polygons participate in fragment)
                    try:
                        s_tag = gmsh.model.occ.addPlaneSurface(loops)
                    except Exception as e:
                        print(f"Error creating surface for polygon {idx}: {e}")
                        continue

                    key = to_key(2, s_tag)
                    if embedded:  # TODO check if I should add not embedded for fields
                        input_tag_info[key] = {'type': 'surface', 'id': idx}
                        embedded_surface_tags.append(key)
                    else:
                        # These surfaces/curves are NOT included in fragment, so they won't cut the domain.
                        nonembedded_surface_tags.setdefault(int(idx), []).append(key)
                        nonembedded_poly_curve_tags.setdefault(int(idx), []).extend(
                            [(1, int(t)) for t in boundary_curve_tags]
                        )
        #call the gui before fragmentation for debugging
        if self.verbosity > 1:
            gmsh.model.occ.synchronize()
            gmsh.fltk.run()

        # "Fragment" combines all the individual geometries into a single,
        # topologically consistent model. This is where intersections are
        # calculated and new, smaller entities are created at overlaps.
        # Only embedded geometry participates in fragmentation.
        object_tags = embedded_surface_tags + embedded_line_tags + embedded_point_tags
        
        if not object_tags:
            print("Warning: No geometry to mesh.")
            return {
                'points': nonembedded_point_tags,
                'lines': nonembedded_line_tags,
                'surfaces': nonembedded_surface_tags,
                'straddle_surfs': {},
                'poly_curves': nonembedded_poly_curve_tags,
            }

        print(f"Fragmenting {len(object_tags)} objects...")
        out_dt, out_map = gmsh.model.occ.fragment(object_tags, [])
        gmsh.model.occ.synchronize()
        
        # After fragmentation, we need to rebuild our map of which original
        # feature corresponds to which new Gmsh tags.
        final_map = {
            'points': dict(nonembedded_point_tags),
            'lines': dict(nonembedded_line_tags),
            'surfaces': dict(nonembedded_surface_tags),
            'straddle_surfs': {},
            'poly_curves': dict(nonembedded_poly_curve_tags),
        }
        
        print(f"Reconstructing Map (Input Tags: {len(object_tags)}, Out Map Len: {len(out_map)})...")
        
        for i, input_dimtag in enumerate(object_tags):
            if i < len(out_map):
                res_tags = out_map[i]
            else:
                res_tags = [input_dimtag]

            # Look up the original feature ID using the pre-fragmentation tag.
            key = to_key(input_dimtag[0], input_dimtag[1])
            
            if key in input_tag_info:
                info = input_tag_info[key]
                kind = info['type']
                feat_id = int(info['id'])
                
                if kind == 'point':
                    if feat_id not in final_map['points']:
                        final_map['points'][feat_id] = []
                    final_map['points'][feat_id].extend(res_tags)
                    
                elif kind == 'line':
                    if feat_id not in final_map['lines']:
                        final_map['lines'][feat_id] = []
                    final_map['lines'][feat_id].extend(res_tags)
                    
                elif kind == 'surface':
                    if feat_id not in final_map['surfaces']:
                        final_map['surfaces'][feat_id] = []
                    final_map['surfaces'][feat_id].extend(res_tags)
                    
                elif kind == 'straddle_surf':
                    if feat_id not in final_map['straddle_surfs']:
                        final_map['straddle_surfs'][feat_id] = []
                    final_map['straddle_surfs'][feat_id].extend(res_tags)
            else:
                print(f"Warning: Tag {key} lost during fragmentation mapping.")

        return final_map
    
    def _setup_fields(self, gmsh_map, polygons_gdf, lines_gdf, points_gdf):
        """
        Configures Gmsh mesh size fields based on the input features.

        This method creates and combines various fields (`Distance`, `Threshold`,
        `MathEval`) to control the mesh element size across the domain. It uses
        the parameters (e.g., `lc`, `dist_min`, `dist_max`) from the
        original conceptual model features to define how the mesh should be
        refined near points, along lines, and within polygons.
        """
        if self.verbosity > 0:
            print(f"--- Setup Fields Debug ---")
            print(f"Polygons GDF: {len(polygons_gdf)} rows")
            print(f"Gmsh Surface Map: {len(gmsh_map.get('surfaces', {}))} entries")
            if not polygons_gdf.empty:
                first_idx = polygons_gdf.index[0]
                print(f"First Poly Index: {first_idx} (Type: {type(first_idx)})")
                if gmsh_map['surfaces']:
                    first_key = list(gmsh_map['surfaces'].keys())[0]
                    print(f"First Map Key: {first_key} (Type: {type(first_key)})")
                    print(f"Match? {first_idx in gmsh_map['surfaces']}")
                else:
                    print("Gmsh Surface Map is EMPTY.")

        # Collect all created Gmsh field ids so we can combine them at the end.
        field_list = []
        
        # The global background mesh size is always required.
        # We create a Constant field for it and always set a background mesh.
        if self.background_lc is None:
            raise ValueError(
                "MeshGenerator.background_lc must be provided. "
                "If you don't want to constrain the mesh, pass a very large value."
            )
        global_max_lc = float(self.background_lc)

        def extract_tags(entry_list):
            """Return a clean list of integer tags from Gmsh's dimtag-ish output.

            Gmsh commonly returns lists of (dim, tag) tuples; some maps in this
            code also store raw tag ints. We normalize both to an int tag list.
            """
            clean_tags = []
            for item in entry_list:
                if isinstance(item, (tuple, list)) and len(item) >= 2:
                    clean_tags.append(item[1])
                else:
                    clean_tags.append(item)
            return clean_tags

        def get_row_param(row, key, default):
            if key in row and not pd.isna(row[key]):
                return float(row[key])
            return float(default)

        def _normalize_fields(value):
            """Normalize feature field specifications to a list[MeshField].

            Supported inputs:
            - None / NaN -> []
            - MeshField  -> [field]
            - list/tuple/set of mixed values -> only MeshField entries are kept
            """
            if value is None:
                return []
            if isinstance(value, float) and pd.isna(value):
                return []
            if isinstance(value, MeshField):
                return [value]
            if isinstance(value, (list, tuple, set)):
                return [v for v in value if isinstance(v, MeshField)]
            return []

        def _auto_threshold_from_row(row, background_lc):
            """Create a default ThresholdField from dist_min/dist_max + lc.

            This centralizes the legacy behavior that used to live in
            ConceptualMesh.add_*: if a feature specifies dist_min/dist_max it
            implies a distance-based size transition around that feature.

                        Design notes:
                        - We only auto-create this field when at least one of dist_min/dist_max
                            is provided AND the feature has a valid lc.
                        - This is a *distance-based* transition around the feature:
                            - SizeMin = lc (feature resolution)
                            - SizeMax = background_lc (global resolution)
                            - DistMin >= 0.5 * lc (avoid near-zero gradients)
                            - DistMax defaults to ~5 * background_lc (transition length scale)
                            - We enforce (DistMax - DistMin) >= 3 * SizeMax for a gentle gradient.
            """
            dist_min = row.get('dist_min', None)
            dist_max = row.get('dist_max', None)

            if (dist_min is None or (isinstance(dist_min, float) and pd.isna(dist_min))) and (
                dist_max is None or (isinstance(dist_max, float) and pd.isna(dist_max))
            ):
                return None

            # We need both the feature resolution and a global background resolution
            # to define an implicit ThresholdField.
            if background_lc is None or (isinstance(background_lc, float) and pd.isna(background_lc)):
                return None

            feature_lc = row.get('lc', None)
            if feature_lc is None or (isinstance(feature_lc, float) and pd.isna(feature_lc)):
                return None

            feature_lc = float(feature_lc)
            dist_min = None if (dist_min is None or (isinstance(dist_min, float) and pd.isna(dist_min))) else float(dist_min)
            dist_max = None if (dist_max is None or (isinstance(dist_max, float) and pd.isna(dist_max))) else float(dist_max)

            # Default distances if one of the values is omitted.
            # - DistMin: at least one local element size.
            # - DistMax: a broader transition scale based on global mesh size.
            if dist_min is None:
                dist_min = feature_lc
            if dist_max is None:
                dist_max = float(background_lc) * 5.0

            # Force a reasonable relationship with the target resolution.
            # DistMin should not be smaller than the local size.
            dist_min = max(dist_min, feature_lc*0.5)

            # Enforce a minimum transition span relative to SizeMax.
            # This avoids very steep growth that can make the mesher struggle.
            min_span = 3.0 * float(background_lc)
            if (dist_max - dist_min) < min_span:
                dist_max = dist_min + min_span

            # Final safety.
            if dist_max <= dist_min:
                dist_max = dist_min + max(float(background_lc), feature_lc, 1e-3)

            return ThresholdField(size_min=feature_lc, dist_min=dist_min, dist_max=dist_max, size_max=background_lc)
        
        # Configure mesh size fields using MeshField objects attached to features.
        #
        # Data model expectations:
        # - ConceptualMesh stores the user's desired behavior in GeoDataFrame rows.
        # - Fields are created here (engine side) because the engine has the Gmsh
        #   tags and is responsible for mapping features -> CAD entities.
        #
        # How fields can be specified per feature:
        # - `fields`: list[MeshField] (the only supported explicit mechanism)
        # - `dist_min/dist_max` (+ lc): shorthand for an automatic ThresholdField
        #
        # Grouping:
        # - We build ONE gmsh field per unique (field parameters + lc).
        # - We intentionally do NOT split by geometry type because a single Gmsh
        #   Distance/Threshold field can target points/curves/surfaces at once.
        # - `feature_lc` is part of grouping because Auto* fields compute their
        #   transition based on the local target size.
        #
        # Each group accumulates feature ids per geometry type so we can later
        # gather all relevant gmsh tags into a single tags_dict.
        field_objects = {}
        for gdf, geom_type in [(points_gdf, 'points'), (lines_gdf, 'lines'), (polygons_gdf, 'surfaces')]:
            for idx, row in gdf.iterrows():
                # 1) Collect explicitly specified fields.
                row_fields = []
                row_fields.extend(_normalize_fields(row.get('fields', None)))

                # 2) Optionally add an implicit ThresholdField based on dist_min/dist_max.
                auto_field = _auto_threshold_from_row(row, global_max_lc)
                if auto_field is not None:
                    row_fields.append(auto_field)

                if not row_fields:
                    continue

                # Cache the feature lc for Auto* fields.
                feature_lc = row.get('lc', None)
                if feature_lc is None or (isinstance(feature_lc, float) and pd.isna(feature_lc)):
                    feature_lc = None
                else:
                    feature_lc = float(feature_lc)

                for field in row_fields:
                    if field is None or not isinstance(field, MeshField):
                        continue
                    key = (hash(field), feature_lc)
                    if key not in field_objects:
                        field_objects[key] = {
                            'field': field,
                            'feature_lc': feature_lc,
                            'feature_ids_by_geom': {'points': [], 'lines': [], 'surfaces': []},
                        }
                    field_objects[key]['feature_ids_by_geom'][geom_type].append(int(idx))
        
        # Now create and apply each unique field to the corresponding features.
        # We gather all Gmsh entity tags for the features and let the MeshField
        # implementation create the appropriate Distance/Threshold/etc field.
        for key, info in field_objects.items():
            field = info['field']
            feature_lc = info.get('feature_lc', None)
            feature_ids_by_geom = info.get('feature_ids_by_geom', {'points': [], 'lines': [], 'surfaces': []})
            
            # Gather all Gmsh tags for the features using this field.
            # Note: embedded geometry is mapped under gmsh_map[geom_type].
            # For embed=False polygons, we keep their boundary curves under
            # gmsh_map['poly_curves'] so fields can still be applied without
            # cutting/fragmenting the domain.
            tags_dict = { 'points': [], 'lines': [], 'surfaces': [] }

            # Points
            for fid in feature_ids_by_geom.get('points', []):
                if fid in gmsh_map.get('points', {}):
                    tags_dict['points'].extend(extract_tags(gmsh_map['points'][fid]))

            # Lines
            for fid in feature_ids_by_geom.get('lines', []):
                if fid in gmsh_map.get('lines', {}):
                    tags_dict['lines'].extend(extract_tags(gmsh_map['lines'][fid]))
                # Straddle/barrier lines may have been converted into points.
                elif fid in gmsh_map.get('points', {}):
                    tags_dict['points'].extend(extract_tags(gmsh_map['points'][fid]))

            # Surfaces
            for fid in feature_ids_by_geom.get('surfaces', []):
                if fid in gmsh_map.get('surfaces', {}):
                    tags_dict['surfaces'].extend(extract_tags(gmsh_map['surfaces'][fid]))
                # Field-only polygons (embed=False): apply distance-based fields to boundary curves.
                elif fid in gmsh_map.get('poly_curves', {}):
                    curve_dimtags = gmsh_map['poly_curves'][fid]
                    tags_dict['lines'].extend(extract_tags(curve_dimtags))

            if not any(tags_dict.values()):
                continue
            
            # Create the Gmsh field using the provided MeshField object.
            f_id = field.create(
                gmsh_api=gmsh,
                tags_dict=tags_dict,
                background_lc=global_max_lc,
                feature_lc=feature_lc
            )
            
            if f_id is not None:
                field_list.append(f_id)

        #now lets add the background constant field if specified
        if self.background_lc is not None:
            const_field = ConstantField(size=self.background_lc)
            f_id = const_field.create(
                gmsh_api=gmsh,
                tags_dict={},
                background_lc=self.background_lc,
                feature_lc=None
            )
            if f_id is not None:
                field_list.append(f_id)

        # Combine all active fields using a Min field and set it as the background mesh.
        # At any (x,y), Gmsh will take the smallest requested element size.
        if field_list:
            min_field = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [float(f) for f in field_list])
            gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

        # Disable Gmsh's default sizing mechanisms so fields fully control mesh size.
        # Otherwise, mesh sizing from points/curvature/boundary can compete with fields.
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)


    def _embed_features(self, gmsh_map, polygons_gdf, lines_gdf, points_gdf):
        """
        Explicitly embeds features into domain surfaces to ensure mesh conformity.
        
        This handles cases where fragmentation splits surfaces, requiring
        geometric discovery to find the correct surface for points/lines.
        """
        if self.verbosity > 0:
            print("Explicitly embedding features into domain surfaces...")

        def is_embedded(row):
            val = row.get('embed', True)
            if pd.isna(val): return True
            return bool(val)

        # 1. Collect Domain Surfaces (Candidate Pool)
        domain_surface_tags = set()
        if not polygons_gdf.empty:
            for idx, row in polygons_gdf.iterrows():
                if is_embedded(row) and idx in gmsh_map.get('surfaces', {}):
                    for dt in gmsh_map['surfaces'][idx]:
                        # Ensure we are tracking actual surfaces (dim=2)
                        if isinstance(dt, (tuple, list)) and len(dt) >= 2 and dt[0] == 2:
                            domain_surface_tags.add(dt[1])

        if not domain_surface_tags:
            return 

        # Helper for geometric embedding search and application
        def embed_entity(dim, tag):
            # 1. Get Bounding Box
            try:
                bbox = gmsh.model.getBoundingBox(dim, tag)
            except Exception:
                return # Entity might not exist or be invalid
            
            xmin, ymin, zmin, xmax, ymax, zmax = bbox
            
            # Expand slightly to find touching surfaces
            eps = 1e-4 
            
            # 2. Find Candidate Surfaces
            candidates = gmsh.model.getEntitiesInBoundingBox(
                xmin - eps, ymin - eps, zmin - eps,
                xmax + eps, ymax + eps, zmax + eps,
                dim=2
            )
            
            # 3. Filter candidates to only include our domain surfaces
            valid_candidates = [
                c[1] for c in candidates 
                if c[0] == 2 and c[1] in domain_surface_tags
            ]
            
            if not valid_candidates:
                return

            # 4. Correctness Check: Verify entity is actually on the surface
            target_matches = []
            
            check_x, check_y, check_z = 0.0, 0.0, 0.0
            
            if dim == 0: # Point
                # For a point, the bbox center is the point
                check_x = (xmin + xmax) / 2.0
                check_y = (ymin + ymax) / 2.0
                check_z = (zmin + zmax) / 2.0
            elif dim == 1: # Line
                # Use Midpoint for valid check (avoid endpoints that might touch multiple surfaces)
                pmin, pmax = gmsh.model.getParametrizationBounds(1, tag)
                pmid = (pmin + pmax) / 2.0
                val = gmsh.model.getValue(1, tag, [pmid])
                check_x, check_y, check_z = val[0], val[1], val[2]

            for surf_tag in valid_candidates:
                # Project testing point to surface
                cp_coords, _ = gmsh.model.getClosestPoint(2, surf_tag, [check_x, check_y, check_z])
                
                # Calculate Euclidean distance
                dist = math.sqrt(
                    (check_x - cp_coords[0])**2 + 
                    (check_y - cp_coords[1])**2 + 
                    (check_z - cp_coords[2])**2
                )
                
                if dist < 1e-6:
                    target_matches.append(surf_tag)

            # 5. Embed the entity into the verified surfaces
            if target_matches:
                # Deduplicate tags
                target_matches = list(set(target_matches))
                for st in target_matches:
                    gmsh.model.mesh.embed(dim, [tag], 2, st)

        # Iterate and Embed Points
        if points_gdf is not None and not points_gdf.empty:
            for idx, row in points_gdf.iterrows():
                if is_embedded(row) and idx in gmsh_map.get('points', {}):
                    for dt in gmsh_map['points'][idx]:
                        if dt[0] == 0:
                            embed_entity(0, dt[1])

        # Iterate and Embed Lines
        if lines_gdf is not None and not lines_gdf.empty:
            for idx, row in lines_gdf.iterrows():
                if is_embedded(row):
                    # Standard Lines
                    if idx in gmsh_map.get('lines', {}):
                        for dt in gmsh_map['lines'][idx]:
                            if dt[0] == 1:
                                embed_entity(1, dt[1])
                    # Barrier/Straddle Points (these are points derived from lines)
                    if idx in gmsh_map.get('points', {}): 
                        for dt in gmsh_map['points'][idx]:
                            if dt[0] == 0:
                                embed_entity(0, dt[1])


    def generate(self, clean_polys, clean_lines, clean_points, output_file=None, launch_gmsh_gui=False):
        """
        Executes the full mesh generation workflow.

        This method orchestrates the entire process:
        1. Initializes Gmsh.
        2. Transfers geometries into the Gmsh model.
        3. Sets up mesh size fields.
        4. Generates the 2D triangular mesh.
        5. Performs optional post-generation optimization.
        6. Extracts the resulting nodes and their tags.

        Args:
            clean_polys (GeoDataFrame): Non-overlapping polygons.
            clean_lines (GeoDataFrame): Snapped and cleaned lines.
            clean_points (GeoDataFrame): Snapped and cleaned points.
            output_file (str, optional): If provided, saves the mesh to this path.
            launch_gmsh_gui (Boolean, optional): This allow to see triangular mesh results
                using the GMSH GUI, and allow to review visually the fields and the triangular
                mesh quality

        Returns:
            bool: True if generation was successful.
        
        Raises:
            Exception: If any step in the Gmsh process fails.
        """
        self._initialize_gmsh()
        try:
            print("Transferring Geometry to Gmsh...")
            gmsh_map = self._add_geometry(clean_polys, clean_lines, clean_points)
            
            # Ensure features are correctly embedded in surfaces before meshing
            self._embed_features(gmsh_map, clean_polys, clean_lines, clean_points)
            
            print("Setting up Resolution Fields...")
            self._setup_fields(gmsh_map, clean_polys, clean_lines, clean_points)
            
            # Set the core meshing algorithm.
            gmsh.option.setNumber("Mesh.Algorithm", self.mesh_algorithm) 
            
            # Set the number of internal smoothing steps.
            gmsh.option.setNumber("Mesh.Smoothing", self.smoothing_steps)

            print("Generating Triangular Mesh...")
            gmsh.model.mesh.generate(2)
            
            # Run explicit optimization passes after generation for higher quality.
            if self.optimization_cycles > 0:
                if self.verbosity > 0:
                    print(f"Running {self.optimization_cycles} Optimization Cycles (Relocate2D & Laplace2D)...")
                
                for i in range(self.optimization_cycles):
                    if self.verbosity > 1:
                        print(f"  -> Cycle {i+1}/{self.optimization_cycles}")
                    # Moves nodes to improve element shape (compactness).
                    gmsh.model.mesh.optimize("Relocate2D",niter=1)
                    # Smooths the mesh to relax gradients (reduces drift).
                    gmsh.model.mesh.optimize("Laplace2D",niter=1)

            
            if output_file:
                gmsh.write(output_file)

            # --- Node extraction (domain-only) ---
            # Do NOT use gmsh.model.mesh.getNodes() without args here.
            # That returns nodes from all entities, including standalone 1D meshes
            # on curves (e.g. field-only rivers) and any non-fragmented 2D surfaces.
            # Those extra nodes can unintentionally constrain downstream Voronoi
            # tessellation.

            def _is_embedded_row(row) -> bool:
                val = row.get('embed', True)
                if pd.isna(val):
                    return True
                return bool(val)

            def _accumulate_nodes(dim: int, ent_tag: int, include_boundary: bool, tag_to_xy: dict[int, tuple[float, float]]):
                nt, nc, _ = gmsh.model.mesh.getNodes(dim, int(ent_tag), includeBoundary=bool(include_boundary))
                if len(nt) == 0:
                    return
                pts = np.array(nc, dtype=float).reshape(-1, 3)
                for t, p in zip(nt, pts):
                    tt = int(t)
                    if tt not in tag_to_xy:
                        tag_to_xy[tt] = (float(p[0]), float(p[1]))

            tag_to_xy: dict[int, tuple[float, float]] = {}

            # 1) Domain surfaces: embedded polygons only
            domain_surface_tags: list[int] = []
            if clean_polys is not None and not clean_polys.empty and 'embed' in clean_polys.columns:
                embedded_poly_ids = [int(i) for i, r in clean_polys.iterrows() if _is_embedded_row(r)]
            elif clean_polys is not None and not clean_polys.empty:
                # Historical behavior: polygons were all embedded.
                embedded_poly_ids = [int(i) for i in clean_polys.index]
            else:
                embedded_poly_ids = []

            for fid in embedded_poly_ids:
                if fid in gmsh_map.get('surfaces', {}):
                    for dimtag in gmsh_map['surfaces'][fid]:
                        if isinstance(dimtag, (tuple, list)) and len(dimtag) >= 2 and int(dimtag[0]) == 2:
                            domain_surface_tags.append(int(dimtag[1]))

            # If we cannot determine domain surfaces from the map, fall back to
            # all 2D nodes (still avoids 1D-only nodes).
            if not domain_surface_tags:
                node_tags, coords, _ = gmsh.model.mesh.getNodes(2, -1, includeBoundary=True)
                nodes_3d = np.array(coords, dtype=float).reshape(-1, 3)
                self.nodes = nodes_3d[:, :2]
                self.node_tags = node_tags
            else:
                for s in domain_surface_tags:
                    _accumulate_nodes(2, s, True, tag_to_xy)

                # 2) Embedded constraints (optional safety)
                if clean_points is not None and not clean_points.empty and 'embed' in clean_points.columns:
                    for fid, row in clean_points.iterrows():
                        if not _is_embedded_row(row):
                            continue
                        if int(fid) in gmsh_map.get('points', {}):
                            for dimtag in gmsh_map['points'][int(fid)]:
                                if isinstance(dimtag, (tuple, list)) and len(dimtag) >= 2 and int(dimtag[0]) == 0:
                                    _accumulate_nodes(0, int(dimtag[1]), True, tag_to_xy)

                if clean_lines is not None and not clean_lines.empty and 'embed' in clean_lines.columns:
                    for fid, row in clean_lines.iterrows():
                        if not _is_embedded_row(row):
                            continue

                        if int(fid) in gmsh_map.get('lines', {}):
                            for dimtag in gmsh_map['lines'][int(fid)]:
                                if isinstance(dimtag, (tuple, list)) and len(dimtag) >= 2 and int(dimtag[0]) == 1:
                                    _accumulate_nodes(1, int(dimtag[1]), True, tag_to_xy)
                        # Straddle/barrier lines may have been converted into points.
                        elif int(fid) in gmsh_map.get('points', {}):
                            for dimtag in gmsh_map['points'][int(fid)]:
                                if isinstance(dimtag, (tuple, list)) and len(dimtag) >= 2 and int(dimtag[0]) == 0:
                                    _accumulate_nodes(0, int(dimtag[1]), True, tag_to_xy)

                # Finalize de-duplicated node arrays
                node_tags = np.array(list(tag_to_xy.keys()), dtype=np.uint64)
                nodes_xy = np.array([tag_to_xy[int(t)] for t in node_tags], dtype=float)
                self.nodes = nodes_xy
                self.node_tags = node_tags

            self.zones_gdf = clean_polys
            if launch_gmsh_gui:
                gmsh.fltk.run()
            self._finalize_gmsh()
            return True

        except Exception as e:
            print(f"Mesh Generation Failed: {e}")
            self._finalize_gmsh()
            raise e