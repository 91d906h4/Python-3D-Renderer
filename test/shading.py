import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Function to normalize a vector
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
        return v
    return v / norm

# Function to create the vertices of an icosahedron
def icosahedron():
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    vertices = np.array([[-1,  phi,  0],
                         [ 1,  phi,  0],
                         [-1, -phi,  0],
                         [ 1, -phi,  0],
                         [ 0, -1,  phi],
                         [ 0,  1,  phi],
                         [ 0, -1, -phi],
                         [ 0,  1, -phi],
                         [ phi,  0, -1],
                         [ phi,  0,  1],
                         [-phi,  0, -1],
                         [-phi,  0,  1]])

    # Normalize all vertices
    vertices = np.array([normalize(v) for v in vertices])
    
    return vertices

# Function to create triangles from the icosahedron faces
def icosahedron_faces(vertices):
    faces = [[vertices[0], vertices[11], vertices[5]],
             [vertices[0], vertices[5], vertices[1]],
             [vertices[0], vertices[1], vertices[7]],
             [vertices[0], vertices[7], vertices[10]],
             [vertices[0], vertices[10], vertices[11]],
             [vertices[1], vertices[5], vertices[9]],
             [vertices[5], vertices[11], vertices[4]],
             [vertices[11], vertices[10], vertices[2]],
             [vertices[10], vertices[7], vertices[6]],
             [vertices[7], vertices[1], vertices[8]],
             [vertices[3], vertices[9], vertices[4]],
             [vertices[3], vertices[4], vertices[2]],
             [vertices[3], vertices[2], vertices[6]],
             [vertices[3], vertices[6], vertices[8]],
             [vertices[3], vertices[8], vertices[9]],
             [vertices[4], vertices[9], vertices[5]],
             [vertices[2], vertices[4], vertices[11]],
             [vertices[6], vertices[2], vertices[10]],
             [vertices[8], vertices[6], vertices[7]],
             [vertices[9], vertices[8], vertices[1]]]
    return faces

# Function to calculate the normal of a face
def calculate_normal(face):
    v1 = face[1] - face[0]
    v2 = face[2] - face[0]
    normal = np.cross(v1, v2)
    return normal / np.linalg.norm(normal)  # Return the normalized normal vector

# Function to compute lighting intensity using the dot product
def compute_lighting(normal, light_dir):
    return np.clip(np.dot(normal, light_dir), 0, 1)  # Return value between 0 and 1

# Get the vertices and faces for the icosahedron
vertices = icosahedron()
faces = icosahedron_faces(vertices)

# Define the light direction (a vector pointing in the direction of the light)
light_dir = np.array([0, 0, 1])  # Direction of light (arbitrary)

# Normalize the light direction
light_dir = light_dir / np.linalg.norm(light_dir)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Add the sphere faces to the plot with flat shading
for face in faces:
    normal = calculate_normal(np.array(face))  # Calculate the normal for the face
    lighting = compute_lighting(normal, light_dir)  # Calculate the lighting intensity
    color = (lighting, lighting, lighting)  # Flat shading: grayscale color based on light intensity
    ax.add_collection3d(Poly3DCollection([face], facecolors=color, linewidths=0, alpha=1))

# Set the limits of the axes
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])
ax.set_zlim([-1.5, 1.5])

# Set labels for clarity
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Equal aspect ratio for all axes
ax.set_box_aspect([1, 1, 1])

# Show the plot
plt.show()
