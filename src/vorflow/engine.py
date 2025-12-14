import gmsh
import sys
import math
import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid
from .field_graph import build_background_field

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

        field_only_points = {}
        field_only_lines = {}
        field_only_poly_curves = {}
        
        def to_key(dim, tag):
            return (int(dim), int(tag))

        def is_embedded(row) -> bool:
            val = row.get('embed', True)
            if pd.isna(val):
                return True
            return bool(val)
        
        # Add all point features to the Gmsh model first.
        embedded_point_tags = []
        for idx, row in points_gdf.iterrows():
            tag = gmsh.model.occ.addPoint(row.geometry.x, row.geometry.y, 0)
            key = to_key(0, tag)
            if is_embedded(row):
                input_tag_info[key] = {'type': 'point', 'id': idx}
                embedded_point_tags.append(key)
            else:
                field_only_points.setdefault(int(idx), []).append(key)
            
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
        embedded_line_tags = []
        embedded_surface_tags = []
        
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
                    if embedded:
                        input_tag_info[k_l] = {'type': 'point', 'id': idx}
                        embedded_point_tags.append(k_l)
                    else:
                        field_only_points.setdefault(int(idx), []).append(k_l)
                    
                    rx, ry = p.x - nx*epsilon, p.y - ny*epsilon
                    rt = gmsh.model.occ.addPoint(rx, ry, 0)
                    k_r = to_key(0, rt)
                    if embedded:
                        input_tag_info[k_r] = {'type': 'point', 'id': idx}
                        embedded_point_tags.append(k_r)
                    else:
                        field_only_points.setdefault(int(idx), []).append(k_r)

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
                            field_only_lines.setdefault(int(idx), []).append(key)

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

                    if embedded:
                        # Create plane surface with holes (embedded polygons participate in fragment)
                        try:
                            s_tag = gmsh.model.occ.addPlaneSurface(loops)
                        except Exception as e:
                            print(f"Error creating surface for polygon {idx}: {e}")
                            continue

                        key = to_key(2, s_tag)
                        input_tag_info[key] = {'type': 'surface', 'id': idx}
                        embedded_surface_tags.append(key)
                    else:
                        # Field-only polygons are represented by their boundary curves only.
                        # These curves are NOT included in fragment, so they won't cut the domain.
                        if boundary_curve_tags:
                            field_only_poly_curves.setdefault(int(idx), []).extend(
                                [to_key(1, t) for t in boundary_curve_tags]
                            )

        # "Fragment" combines all the individual geometries into a single,
        # topologically consistent model. This is where intersections are
        # calculated and new, smaller entities are created at overlaps.
        # Only embedded geometry participates in fragmentation.
        object_tags = embedded_surface_tags + embedded_line_tags + embedded_point_tags
        
        if not object_tags:
            print("Warning: No geometry to mesh.")
            return {
                'points': field_only_points,
                'lines': field_only_lines,
                'surfaces': {},
                'straddle_surfs': {},
                'poly_curves': field_only_poly_curves,
            }

        print(f"Fragmenting {len(object_tags)} objects...")
        out_dt, out_map = gmsh.model.occ.fragment(object_tags, [])
        gmsh.model.occ.synchronize()
        
        # After fragmentation, we need to rebuild our map of which original
        # feature corresponds to which new Gmsh tags.
        final_map = {
            'points': dict(field_only_points),
            'lines': dict(field_only_lines),
            'surfaces': {},
            'straddle_surfs': {},
            'poly_curves': dict(field_only_poly_curves),
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

        if self.background_lc is None:
            raise ValueError("MeshGenerator.background_lc must be provided to build mesh-size fields")

        build_background_field(
            gmsh,
            gmsh_map,
            polygons_gdf,
            lines_gdf,
            points_gdf,
            self.background_lc,
            verbosity=self.verbosity,
        )
        
        # Disable Gmsh's default size-setting mechanisms. We want our fields
        # to have complete control over the mesh size.
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    def generate(self, clean_polys, clean_lines, clean_points, output_file=None):
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

        Returns:
            bool: True if generation was successful.
        
        Raises:
            Exception: If any step in the Gmsh process fails.
        """
        self._initialize_gmsh()
        try:
            print("Transferring Geometry to Gmsh...")
            gmsh_map = self._add_geometry(clean_polys, clean_lines, clean_points)
            
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
                
            node_tags, coords, _ = gmsh.model.mesh.getNodes()
            nodes_3d = np.array(coords).reshape(-1, 3)
            self.nodes = nodes_3d[:, :2]
            self.node_tags = node_tags
            self.zones_gdf = clean_polys
            
            self._finalize_gmsh()
            return True

        except Exception as e:
            print(f"Mesh Generation Failed: {e}")
            self._finalize_gmsh()
            raise e