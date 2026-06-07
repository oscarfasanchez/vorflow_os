import numpy as np
import geopandas as gpd
import pandas as pd
import gmsh
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, Point, LineString, MultiPolygon
from shapely.ops import unary_union, split
from shapely.validation import make_valid

class VoronoiTessellator:
    def __init__(
        self,
        mesh_generator,
        conceptual_mesh,
        clip_to_boundary=True,
        boundary_centering="clip",
        boundary_inset_fraction=0.5,
        boundary_corner_angle=135.0,
        boundary_tolerance=None,
    ):
        """
        Initializes the Voronoi tessellator.

        This class takes a triangular mesh (typically from `MeshGenerator`) and
        computes its dual: a Voronoi diagram. The resulting grid of polygonal
        cells is suitable for use in cell-centered finite volume models.

        Args:
            mesh_generator (MeshGenerator): An instance of the mesh generator that
                contains the generated triangular mesh nodes.
            conceptual_mesh (ConceptualMesh): The conceptual model, used for CRS,
                domain boundaries, and feature information.
            clip_to_boundary (bool): If True, the final Voronoi grid will be
                clipped to the domain boundary defined in the conceptual model.
            boundary_centering (str): ``"clip"`` keeps the historical behavior.
                ``"inset_mirror"`` shifts boundary generators inward and adds
                mirrored outside ghosts so boundary-cell centers move off the
                clipped face.
            boundary_inset_fraction (float): Fraction of local boundary-node
                spacing used for the inward shift in ``"inset_mirror"`` mode.
            boundary_corner_angle (float): Boundary vertices with a local angle
                below this value are treated as sharp corners and left unchanged.
            boundary_tolerance (float, optional): Distance tolerance used to
                classify generator nodes as boundary nodes.
        """
        if boundary_centering not in {"clip", "inset_mirror"}:
            raise ValueError("boundary_centering must be either 'clip' or 'inset_mirror'.")
        if boundary_inset_fraction <= 0:
            raise ValueError("boundary_inset_fraction must be positive.")
        if boundary_corner_angle <= 0 or boundary_corner_angle >= 180:
            raise ValueError("boundary_corner_angle must be between 0 and 180 degrees.")
        if boundary_tolerance is not None and boundary_tolerance < 0:
            raise ValueError("boundary_tolerance must be non-negative when provided.")

        self.mg = mesh_generator
        self.cm = conceptual_mesh
        self.voronoi_gdf = None
        self.final_grid = None
        self.nodes = mesh_generator.nodes
        self.node_tags = mesh_generator.node_tags
        self.zones_gdf = mesh_generator.zones_gdf
        self.clip_to_boundary = clip_to_boundary
        self.boundary_centering = boundary_centering
        self.boundary_inset_fraction = float(boundary_inset_fraction)
        self.boundary_corner_angle = float(boundary_corner_angle)
        self.boundary_tolerance = boundary_tolerance

    def _domain_geometry(self):
        """Return the current meshing domain geometry."""
        if not self.cm.clean_polygons.empty:
            domain_geom = unary_union(self.cm.clean_polygons.geometry)
            if not domain_geom.is_valid:
                domain_geom = make_valid(domain_geom)
            return domain_geom
        if hasattr(self.cm, 'domain_boundary') and self.cm.domain_boundary:
            domain_geom = self.cm.domain_boundary
            if not domain_geom.is_valid:
                domain_geom = make_valid(domain_geom)
            return domain_geom
        return None

    def _boundary_tolerance(self, nodes, domain_geom):
        if self.boundary_tolerance is not None:
            return float(self.boundary_tolerance)
        minx, miny, maxx, maxy = domain_geom.bounds
        domain_scale = max(maxx - minx, maxy - miny, 1.0)
        node_scale = 1.0
        if len(nodes) > 0:
            node_scale = max(np.ptp(nodes[:, 0]), np.ptp(nodes[:, 1]), 1.0)
        return max(domain_scale, node_scale) * 1e-8

    def _ring_angle_at_point(self, point, domain_geom, tolerance):
        """Return the local ring angle for a boundary vertex, if matched."""
        polygons = []
        if isinstance(domain_geom, Polygon):
            polygons = [domain_geom]
        elif isinstance(domain_geom, MultiPolygon):
            polygons = list(domain_geom.geoms)

        for poly in polygons:
            rings = [poly.exterior, *poly.interiors]
            for ring in rings:
                coords = list(ring.coords)
                if len(coords) < 4:
                    continue
                open_coords = coords[:-1]
                for i, coord in enumerate(open_coords):
                    if Point(coord).distance(point) > tolerance:
                        continue
                    prev_coord = np.asarray(open_coords[i - 1], dtype=float)
                    current = np.asarray(coord, dtype=float)
                    next_coord = np.asarray(open_coords[(i + 1) % len(open_coords)], dtype=float)
                    v1 = prev_coord - current
                    v2 = next_coord - current
                    mag1 = np.linalg.norm(v1)
                    mag2 = np.linalg.norm(v2)
                    if mag1 == 0 or mag2 == 0:
                        return None
                    cos_theta = np.dot(v1, v2) / (mag1 * mag2)
                    cos_theta = min(1.0, max(-1.0, cos_theta))
                    return float(np.degrees(np.arccos(cos_theta)))
        return None

    def _is_sharp_boundary_corner(self, point, domain_geom, tolerance):
        angle = self._ring_angle_at_point(point, domain_geom, tolerance)
        return angle is not None and angle < self.boundary_corner_angle

    def _local_boundary_tangent(self, boundary, point, spacing):
        distance = boundary.project(point)
        eps = max(spacing * 0.25, boundary.length * 1e-9, 1e-9)
        before = max(0.0, distance - eps)
        after = min(boundary.length, distance + eps)
        if before == after:
            before = max(0.0, distance - 1e-9)
            after = min(boundary.length, distance + 1e-9)
        p1 = boundary.interpolate(before)
        p2 = boundary.interpolate(after)
        tangent = np.array([p2.x - p1.x, p2.y - p1.y], dtype=float)
        norm = np.linalg.norm(tangent)
        if norm == 0:
            return None
        return tangent / norm

    def _inward_normal(self, domain_geom, point, tangent, offset):
        normals = [
            np.array([-tangent[1], tangent[0]], dtype=float),
            np.array([tangent[1], -tangent[0]], dtype=float),
        ]
        probe_distance = max(offset * 0.5, 1e-9)
        for normal in normals:
            probe = Point(point.x + normal[0] * probe_distance, point.y + normal[1] * probe_distance)
            if domain_geom.covers(probe):
                return normal
        return None

    def _prepare_boundary_centered_nodes(self, nodes, node_tags):
        """
        Shift non-corner boundary nodes inward and add mirrored outside ghosts.

        Returns prepared nodes, prepared tags, and a metadata frame keyed by
        node_id. Tags only cover real nodes; appended ghosts receive -1 in
        _build_raw_voronoi.
        """
        if self.boundary_centering != "inset_mirror":
            metadata = pd.DataFrame(
                {
                    "node_id": node_tags,
                    "source_x": nodes[:, 0],
                    "source_y": nodes[:, 1],
                    "boundary_centering": "clip",
                    "boundary_inset": 0.0,
                    "boundary_centered": False,
                }
            )
            return nodes, node_tags, np.empty((0, 2)), metadata

        domain_geom = self._domain_geometry()
        if domain_geom is None or domain_geom.is_empty:
            raise RuntimeError(
                "boundary_centering='inset_mirror' requires a generated domain geometry."
            )

        boundary = domain_geom.boundary
        tolerance = self._boundary_tolerance(nodes, domain_geom)
        points = [Point(float(x), float(y)) for x, y in nodes]
        boundary_mask = np.array([boundary.distance(point) <= tolerance for point in points])
        boundary_indices = np.flatnonzero(boundary_mask)

        prepared = nodes.astype(float, copy=True)
        ghost_nodes = []
        records = []

        if len(boundary_indices) > 1:
            boundary_xy = nodes[boundary_indices]
            distances = np.linalg.norm(
                boundary_xy[:, None, :] - boundary_xy[None, :, :],
                axis=2,
            )
            distances[distances == 0.0] = np.nan
            nearest_spacing = np.nanmin(distances, axis=1)
            spacing_by_index = {
                int(idx): float(spacing)
                for idx, spacing in zip(boundary_indices, nearest_spacing)
                if not np.isnan(spacing) and spacing > 0
            }
        else:
            spacing_by_index = {}

        for i, (node, tag, point) in enumerate(zip(nodes, node_tags, points)):
            centered = False
            inset = 0.0
            if i in spacing_by_index and not self._is_sharp_boundary_corner(point, domain_geom, tolerance):
                spacing = spacing_by_index[i]
                inset = spacing * self.boundary_inset_fraction
                tangent = self._local_boundary_tangent(boundary, point, spacing)
                normal = None if tangent is None else self._inward_normal(domain_geom, point, tangent, inset)
                if normal is not None:
                    prepared[i] = node + normal * inset
                    ghost_nodes.append(node - normal * inset)
                    centered = True

            records.append(
                {
                    "node_id": tag,
                    "source_x": float(node[0]),
                    "source_y": float(node[1]),
                    "boundary_centering": "inset_mirror" if centered else "clip",
                    "boundary_inset": float(inset if centered else 0.0),
                    "boundary_centered": bool(centered),
                }
            )

        ghosts = np.asarray(ghost_nodes, dtype=float) if ghost_nodes else np.empty((0, 2))
        return prepared, node_tags, ghosts, pd.DataFrame(records)

    def _build_raw_voronoi(self, nodes, node_tags):
        """
        Computes the mathematical Voronoi diagram from a set of generator points.

        This method uses `scipy.spatial.Voronoi` to calculate the unbounded
        Voronoi diagram. It filters out invalid or infinite regions and returns
        the finite polygons as a GeoDataFrame.

        Args:
            nodes (np.ndarray): An array of (x, y) coordinates for the generator points.
            node_tags (np.ndarray): An array of IDs corresponding to each node.

        Returns:
            gpd.GeoDataFrame: A GeoDataFrame containing the Voronoi polygons, with
                columns for the generator's node_id, x, and y coordinates.
        """
        if len(nodes) < 3:
            print("Error: Not enough nodes to generate Voronoi.")
            return gpd.GeoDataFrame()

        vor = Voronoi(nodes)
        polygons = []
        ids = []
        gen_x = []
        gen_y = []
        
        for i, region_index in enumerate(vor.point_region):
            region = vor.regions[region_index]
            # Skip infinite regions (those containing -1).
            if not region or -1 in region:
                continue
            
            verts = vor.vertices[region]
            poly = Polygon(verts)
            
            if poly.is_valid:
                polygons.append(poly)
                # Store the coordinates of the generator point for this cell.
                gen_x.append(nodes[i][0])
                gen_y.append(nodes[i][1])
                
                # Assign the node tag (ID) to the cell. Ghost nodes (used to
                # bound the diagram) will not have a tag.
                if i < len(node_tags):
                    ids.append(node_tags[i])
                else:
                    ids.append(-1) 
        
        gdf = gpd.GeoDataFrame(
            {'node_id': ids, 'x': gen_x, 'y': gen_y}, 
            geometry=polygons, 
            crs=self.cm.crs
        )
        return gdf

    def _enforce_barriers(self, grid_gdf):
        """
        Splits Voronoi cells that are crossed by barrier lines.

        This method iterates through all line features marked as `is_barrier=True`
        (and that do not use the `straddle_width` method) and cuts any Voronoi
        cell they intersect. The largest resulting piece of a split cell retains
        the original cell's ID, while smaller pieces are assigned new, unique IDs.

        Args:
            grid_gdf (gpd.GeoDataFrame): The current Voronoi grid.

        Returns:
            gpd.GeoDataFrame: An updated grid with cells split along barrier lines.
        """
        if self.cm.clean_lines.empty:
            return grid_gdf
            
        # We only need to cut barriers that were NOT handled by the "straddle"
        # method in the mesh generator. Straddled barriers are already aligned.
        mask_barrier = self.cm.clean_lines['is_barrier'] == True
        
        if 'straddle_width' in self.cm.clean_lines.columns:
            mask_no_straddle = (self.cm.clean_lines['straddle_width'].isna()) | (self.cm.clean_lines['straddle_width'] <= 0)
        else:
            mask_no_straddle = True
            
        barriers_to_cut = self.cm.clean_lines[mask_barrier & mask_no_straddle]
        
        if barriers_to_cut.empty:
            return grid_gdf
            
        print(f"Enforcing Barrier Cuts on {len(barriers_to_cut)} lines (Straddle lines skipped)...")
        
        current_grid = grid_gdf

        # Keep track of the highest node ID to assign to new cell fragments.
        max_id = grid_gdf['node_id'].max()

        for idx, row in barriers_to_cut.iterrows():
            line = row.geometry
            
            # Use a spatial index to quickly find cells that might intersect the line.
            possible_matches_index = list(current_grid.sindex.query(line, predicate='intersects'))
            candidate_cells = current_grid.iloc[possible_matches_index]
            
            cells_to_keep = []
            cells_to_remove_indices = []
            
            for cell_idx, cell_row in candidate_cells.iterrows():
                cell_poly = cell_row.geometry
                
                if not cell_poly.intersects(line):
                    continue
                    
                try:
                    # Split the cell polygon by the barrier line.
                    split_result = split(cell_poly, line)
                    
                    if len(split_result.geoms) > 1:
                        valid_pieces = []
                        
                        # Sort the pieces by area. The largest piece will keep the original ID.
                        sorted_pieces = sorted(list(split_result.geoms), key=lambda p: p.area, reverse=True)
                        
                        for i, piece in enumerate(sorted_pieces):
                            if isinstance(piece, (Polygon, MultiPolygon)):
                                new_row = cell_row.copy()
                                new_row.geometry = piece
                                
                                if i == 0:
                                    # The largest piece keeps the original ID and attributes.
                                    pass 
                                else:
                                    # Smaller pieces get a new ID and their own centroid.
                                    max_id += 1
                                    new_row['node_id'] = max_id
                                    new_row['x'] = piece.centroid.x
                                    new_row['y'] = piece.centroid.y
                                    
                                valid_pieces.append(new_row)
                        
                        # If the split was successful, mark the original cell for removal.
                        if valid_pieces:
                            cells_to_remove_indices.append(cell_idx)
                            cells_to_keep.extend(valid_pieces)
                            
                except Exception as e:
                    print(f"Warning: Failed to split cell {cell_row['node_id']}: {e}")
            
            # Rebuild the grid with the split cells.
            if cells_to_remove_indices:
                current_grid = current_grid.drop(cells_to_remove_indices)
                new_df = gpd.GeoDataFrame(cells_to_keep, crs=current_grid.crs)
                current_grid = pd.concat([current_grid, new_df], ignore_index=True)
        
        return current_grid

    def generate(self):
        """
        Executes the full Voronoi tessellation workflow.

        This method orchestrates the process of:
        1. Adding "ghost" nodes to create a bounded Voronoi diagram.
        2. Computing the raw Voronoi polygons.
        3. Clipping the grid to the model domain.
        4. Assigning zone IDs to cells based on their generator point location.
        5. Enforcing barrier lines by splitting cells.
        6. Calculating final cell properties.

        Returns:
            gpd.GeoDataFrame: The final, clean Voronoi grid.
        """
        if self.nodes is None or len(self.nodes) == 0:
            print("Error: No nodes found in MeshGenerator.")
            return gpd.GeoDataFrame()
        
        print(f"Extracting {len(self.nodes)} Nodes from Gmsh...")
        nodes, tags = self.nodes, self.node_tags
        nodes = np.asarray(nodes, dtype=float)
        tags = np.asarray(tags)
        if len(tags) != len(nodes):
            raise ValueError("MeshGenerator nodes and node_tags must have the same length.")

        nodes, tags, boundary_ghost_nodes, node_metadata = self._prepare_boundary_centered_nodes(nodes, tags)
        
        # To create a bounded Voronoi diagram from a finite set of points, a common
        # technique is to add "ghost" nodes far outside the area of interest. The
        # large, unwanted cells generated by these ghosts can then be clipped away.
        minx, miny = np.min(nodes, axis=0)
        maxx, maxy = np.max(nodes, axis=0)
        w, h = maxx - minx, maxy - miny
        buffer = max(w, h) * 10
        
        ghost_nodes = np.array([
            [minx - buffer, miny - buffer],
            [maxx + buffer, miny - buffer],
            [maxx + buffer, maxy + buffer],
            [minx - buffer, maxy + buffer]
        ])
        
        combined_nodes = np.vstack([nodes, boundary_ghost_nodes, ghost_nodes])
        
        print("Computing Mathematical Voronoi...")
        raw_gdf = self._build_raw_voronoi(combined_nodes, tags)
        print(f"  -> Raw Polygons: {len(raw_gdf)}")
        
        # Remove the cells generated by the ghost nodes.
        raw_gdf = raw_gdf[raw_gdf['node_id'] != -1]
        print(f"  -> After Ghost Filter: {len(raw_gdf)}")
        
        if raw_gdf.crs is None and self.cm.crs:
            raw_gdf.set_crs(self.cm.crs, inplace=True)

        
        if self.clip_to_boundary:
            print("Clipping to Domain Boundary...")
            domain_geom = self._domain_geometry()
            if domain_geom is None:
                print("Error: No domain geometry found (no polygons).")
                return gpd.GeoDataFrame()

            domain_gdf = gpd.GeoDataFrame(
                geometry=[domain_geom], 
                crs=self.cm.crs
            )
            
            bounded_voronoi = gpd.clip(raw_gdf, domain_gdf)
            print(f"  -> After Domain Clip: {len(bounded_voronoi)}")
        
            if len(bounded_voronoi) == 0:
                print("Warning: Clipping resulted in 0 cells. Check CRS or Domain Box.")
                return bounded_voronoi
        else:
            bounded_voronoi = raw_gdf

        print("Enforcing Hydrogeological Zones (Optimization: Point Sampling)...")
        zones = self.cm.clean_polygons[['geometry', 'zone_id', 'z_order']]
        
        # To assign a zone ID to each Voronoi cell, we perform a spatial join
        # between the cell's generator point and the zone polygons. This is much
        # faster than doing a polygon-on-polygon overlay.
        
        # 1. Create a temporary GeoDataFrame of the generator points.
        pts_gdf = gpd.GeoDataFrame(
            {'node_id': bounded_voronoi['node_id']},
            geometry=gpd.points_from_xy(bounded_voronoi.x, bounded_voronoi.y),
            crs=bounded_voronoi.crs
        )
        
        # 2. Spatially join the points to the zones.
        joined = gpd.sjoin(pts_gdf, zones, how='left', predicate='intersects')
        
        # 3. If a point falls on a boundary between zones, it may have multiple
        # matches. We use the `z_order` from the conceptual model to pick the
        # highest-priority zone.
        if 'z_order' in joined.columns:
            joined = joined.sort_values('z_order', ascending=False)
        
        joined = joined.drop_duplicates(subset='node_id')
        
        # 4. Merge the zone information back into the main grid.
        zoned_grid = bounded_voronoi.merge(
            joined[['node_id', 'zone_id', 'z_order']],
            on='node_id',
            how='left'
        )
        if self.boundary_centering == "inset_mirror":
            zoned_grid = zoned_grid.merge(
                node_metadata,
                on="node_id",
                how="left",
            )
        
        print(f"  -> Zones Assigned: {len(zoned_grid)}")
        
        # Clipping can sometimes create MultiPolygons; explode them into single parts.
        zoned_grid = zoned_grid.explode(index_parts=True).reset_index(drop=True)
        
        # Enforce barriers by splitting cells.
        self.final_grid = self._enforce_barriers(zoned_grid)
        print(f"  -> After Barrier Cuts: {len(self.final_grid)}")
        
        # Final cleanup after potential splits.
        self.final_grid = self.final_grid.explode(index_parts=True).reset_index(drop=True)

        # The 'x' and 'y' columns should always refer to the generator point
        # coordinates, which are essential for quality analysis. We add separate
        # columns for the geometric centroid of the final cell.
        self.final_grid['centroid_x'] = self.final_grid.geometry.centroid.x
        self.final_grid['centroid_y'] = self.final_grid.geometry.centroid.y
        
        print(f"Final Voronoi Grid Generated: {len(self.final_grid)} cells.")
        return self.final_grid

    def export_to_shapefile(self, filepath):
        if self.final_grid is not None and not self.final_grid.empty:
            self.final_grid.to_file(filepath)
            print(f"Saved to {filepath}")
        else:
            print("No grid to export.")
