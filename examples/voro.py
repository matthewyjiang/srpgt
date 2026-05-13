import pyvoro2.planar as pyvoro2_planar

# Define points
points = [[5.0, 7.0], [1.7, 3.2], [8.1, 6.3]]

# Define bounding box limits
domain = pyvoro2_planar.RectangularCell(((0.0, 10.0), (0.0, 10.0)), periodic=(False, False))

# Compute Voronoi cells
cells = pyvoro2_planar.compute(
    points,
    domain=domain,
)

# Extract vertices for the first point
point_index = 0
cell = cells[point_index]
vertices = cell['vertices']

print(f"Vertices of the Voronoi cell around point {points[point_index]}:")
for vertex in vertices:
    print(vertex)
