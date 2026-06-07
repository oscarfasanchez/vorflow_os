#%%
"""Structured quad-buffer example.

This example shows the opt-in structured buffer path for a line feature and a
polygon boundary. The mixed gmsh element grid is useful for checking where quads
were created before looking at the Voronoi dual.
"""

import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon, box

from vorflow import ConceptualMesh, MeshGenerator, VoronoiTessellator
from vorflow.utils import build_connectivity, calculate_mesh_quality


#%%
domain = box(0, 0, 12, 7)
fault = LineString([(0, 3.5), (12, 3.5)])
inner_zone = Polygon([(4, 2), (8, 2), (8, 5), (4, 5)])

mesh = ConceptualMesh(crs=None)
mesh.add_polygon(domain, zone_id="domain", resolution=1.5, densify=True)
mesh.add_polygon(
    inner_zone,
    zone_id="inner-zone",
    resolution=1.0,
    z_order=1,
    densify=True,
    quad_buffer=True,
    quad_buffer_thickness=1,
)
mesh.add_line(
    fault,
    line_id="fault-buffer",
    resolution=1.0,
    is_barrier=True,
    quad_buffer=True,
    quad_buffer_thickness=1,
    dist_min=0.5,
    dist_max=2.0,
)

clean_polys, clean_lines, clean_points = mesh.generate()

mesher = MeshGenerator(
    background_lc=1.5,
    verbosity=0,
    smoothing_steps=0,
    optimization_cycles=0,
)
mesher.generate(clean_polys, clean_lines, clean_points)

element_grid = mesher.get_element_grid()
quads = mesher.get_element_grid("quads")
triangles = mesher.get_element_grid("triangles")

print(f"2D elements: {len(element_grid)}")
print(f"  triangles: {len(triangles)}")
print(f"  quads:     {len(quads)}")


#%%
tessellator = VoronoiTessellator(mesher, mesh, clip_to_boundary=True)
voronoi_grid = tessellator.generate()
connectivity = build_connectivity(voronoi_grid, center="centroid")
quality = calculate_mesh_quality(
    voronoi_grid,
    calc_ortho=True,
    calc_skewness=True,
    connectivity=connectivity,
)

print("\nCentroid-connectivity quality:")
print(quality[["compactness", "drift_ratio", "ortho_error", "skewness"]].describe().T)


#%%
fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

triangles.plot(ax=axes[0], facecolor="#d7e3f4", edgecolor="#8aa0bd", linewidth=0.35)
quads.plot(ax=axes[0], facecolor="#f3c97a", edgecolor="#9b6b1f", linewidth=0.55)
axes[0].plot(*fault.xy, color="black", linewidth=1.4)
axes[0].plot(*inner_zone.exterior.xy, color="black", linewidth=1.0)
axes[0].set_title("Gmsh element grid: triangles and structured quads")
axes[0].set_aspect("equal")
axes[0].axis("off")

quality.plot(
    column="ortho_error",
    ax=axes[1],
    cmap="magma_r",
    edgecolor="#555555",
    linewidth=0.25,
    legend=True,
)
axes[1].plot(*fault.xy, color="cyan", linewidth=1.2)
axes[1].plot(*inner_zone.exterior.xy, color="cyan", linewidth=1.0)
axes[1].set_title("Voronoi cells colored by centroid orthogonality error")
axes[1].set_aspect("equal")
axes[1].axis("off")

plt.show()
