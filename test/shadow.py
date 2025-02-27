import numpy as np

# Light direction (0, 1, 0) -> pointing straight down
light_dir = np.array([0, -1, 0])  # Notice the negative y-direction

# Define an orthographic projection for shadow mapping
def orthographic_projection_matrix(left, right, bottom, top, near, far):
    return np.array([
        [2 / (right - left), 0, 0, -(right + left) / (right - left)],
        [0, 2 / (top - bottom), 0, -(top + bottom) / (top - bottom)],
        [0, 0, -2 / (far - near), -(far + near) / (far - near)],
        [0, 0, 0, 1]
    ])

# Example scene bounds (adjust as needed)
ortho_proj = orthographic_projection_matrix(-10, 10, -10, 10, -20, 20)

# Shadow map resolution
SHADOW_MAP_SIZE = 512
shadow_map = np.full((SHADOW_MAP_SIZE, SHADOW_MAP_SIZE), np.inf)

# Function to transform a vertex to light space
def project_to_light_space(vertex, light_matrix):
    homogeneous_vertex = np.append(vertex, 1)
    projected = np.dot(light_matrix, homogeneous_vertex)
    return projected[:3] / projected[3]  # Normalize

# Generate shadow map from light's perspective
def generate_shadow_map(vertices):
    global shadow_map
    for v in vertices:
        proj = project_to_light_space(v, ortho_proj)
        x, y = int((proj[0] + 1) * 0.5 * SHADOW_MAP_SIZE), int((proj[1] + 1) * 0.5 * SHADOW_MAP_SIZE)
        if 0 <= x < SHADOW_MAP_SIZE and 0 <= y < SHADOW_MAP_SIZE:
            shadow_map[x, y] = min(shadow_map[x, y], proj[2])  # Store closest depth

# Shadow test function
def is_in_shadow(fragment):
    proj = project_to_light_space(fragment, ortho_proj)
    x, y = int((proj[0] + 1) * 0.5 * SHADOW_MAP_SIZE), int((proj[1] + 1) * 0.5 * SHADOW_MAP_SIZE)
    if 0 <= x < SHADOW_MAP_SIZE and 0 <= y < SHADOW_MAP_SIZE:
        return proj[2] > shadow_map[x, y]  # If further than shadow map depth, it's in shadow
    return False

# Example usage
vertices = np.array([
    [-5, 2, 0],  # Vertex 1
    [3, 3, 5],   # Vertex 2
    [7, 8, 2]    # Vertex 3
])
generate_shadow_map(vertices)

fragment = np.array([1, 2, 3])  # Test fragment
print("In shadow:", is_in_shadow(fragment))
