import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import cv2  # OpenCV for image handling

# Load texture image
texture = cv2.imread("ground.png")
texture = cv2.resize(texture, (512, 512))
texture = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

# Function to sample color from texture using UV coordinates
def sample_texture(uv, texture):
    h, w, _ = texture.shape
    u, v = uv
    x = int(u * (w - 1))
    y = int((1 - v) * (h - 1))  # Flip V to match image coordinate system
    return texture[y, x] / 255.0  # Normalize to [0,1]

# Function to generate a sphere mesh
def generate_sphere(radius, latitude_divisions, longitude_divisions):
    vertices = []
    faces = []
    
    for i in range(latitude_divisions + 1):
        lat = math.pi * i / latitude_divisions  # Latitude angle
        for j in range(longitude_divisions):
            lon = 2 * math.pi * j / longitude_divisions  # Longitude angle
            
            x = radius * math.sin(lat) * math.cos(lon)
            y = radius * math.sin(lat) * math.sin(lon)
            z = radius * math.cos(lat)
            vertices.append([x, y, z])
    
    for i in range(latitude_divisions):
        for j in range(longitude_divisions):
            top_left = i * longitude_divisions + j
            top_right = top_left + 1
            bottom_left = (i + 1) * longitude_divisions + j
            bottom_right = bottom_left + 1
            
            if j == longitude_divisions - 1:
                top_right -= longitude_divisions
                bottom_right -= longitude_divisions
            
            faces.append([top_left, bottom_left, top_right])
            faces.append([top_right, bottom_left, bottom_right])
    
    return vertices, faces

# Projection and UV Mapping
def project_points_to_2d(points):
    return [(x, y) for x, y, z in points]

def normalize_uv_coordinates(points_2d):
    min_x, max_x = min(x for x, y in points_2d), max(x for x, y in points_2d)
    min_y, max_y = min(y for x, y in points_2d), max(y for x, y in points_2d)
    
    return [((x - min_x) / (max_x - min_x), (y - min_y) / (max_y - min_y)) for x, y in points_2d]

def assign_uv_coordinates(points):
    projected_points = project_points_to_2d(points)
    return normalize_uv_coordinates(projected_points)

# Function to visualize UV mapping on 2D texture
def visualize_uv_mapping(uv_coordinates, faces, texture):
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Show texture image as background
    ax.imshow(texture, extent=[0, 1, 0, 1], origin="upper")
    
    uv_coords = np.array(uv_coordinates)

    for face in faces:
        uv_face = [uv_coords[idx] for idx in face]
        poly = plt.Polygon(uv_face, edgecolor='red', fill=False, linewidth=1)
        ax.add_patch(poly)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("U Coordinate")
    ax.set_ylabel("V Coordinate")
    ax.set_title("UV Mapping on Texture")
    plt.grid(True)
    plt.show()

# Function to visualize the 3D textured sphere
def visualize_textured_sphere(points, faces, uv_coordinates, texture):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    vertices = np.array(points)
    uv_coords = np.array(uv_coordinates)

    for face in faces:
        face_vertices = [vertices[idx] for idx in face]
        face_uvs = [uv_coords[idx] for idx in face]
        
        # Sample texture color for the face
        face_color = np.mean([sample_texture(uv, texture) for uv in face_uvs], axis=0)
        
        ax.add_collection3d(Poly3DCollection([face_vertices], facecolor=face_color))

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    plt.title("Textured 3D Sphere")
    plt.show()

# Generate sphere
points, planes = generate_sphere(radius=1, latitude_divisions=16, longitude_divisions=16)

# Assign UV coordinates
uv_coordinates = assign_uv_coordinates(points)

# Visualize 2D UV Mapping
visualize_uv_mapping(uv_coordinates, planes, texture)

# Visualize 3D textured sphere
visualize_textured_sphere(points, planes, uv_coordinates, texture)


import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import cv2  # OpenCV for image handling

# Load texture image
texture = cv2.imread("uvgrid.png")
texture = cv2.resize(texture, (512, 512))
texture = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

# Function to sample color from texture using UV coordinates
def sample_texture(uv, texture):
    h, w, _ = texture.shape
    u, v = uv
    x = int(u * (w - 1))
    y = int((1 - v) * (h - 1))  # Flip V to match image coordinate system
    return texture[y, x] / 255.0  # Normalize to [0,1]

# Function to generate a sphere mesh with proper spherical UV mapping
def generate_sphere(radius, latitude_divisions, longitude_divisions):
    vertices = []
    uv_coords = []
    faces = []
    
    for i in range(latitude_divisions + 1):
        lat = math.pi * i / latitude_divisions  # Latitude angle (0 to π)
        v = i / latitude_divisions  # Normalize V (0 to 1)
        for j in range(longitude_divisions):
            lon = 2 * math.pi * j / longitude_divisions  # Longitude angle (0 to 2π)
            u = j / longitude_divisions  # Normalize U (0 to 1)
            
            x = radius * math.sin(lat) * math.cos(lon)
            y = radius * math.sin(lat) * math.sin(lon)
            z = radius * math.cos(lat)
            
            vertices.append([x, y, z])
            uv_coords.append([u, v])  # Store UV coordinates

    for i in range(latitude_divisions):
        for j in range(longitude_divisions):
            top_left = i * longitude_divisions + j
            top_right = top_left + 1
            bottom_left = (i + 1) * longitude_divisions + j
            bottom_right = bottom_left + 1
            
            if j == longitude_divisions - 1:  # Wrap texture around
                top_right -= longitude_divisions
                bottom_right -= longitude_divisions
            
            faces.append([top_left, bottom_left, top_right])
            faces.append([top_right, bottom_left, bottom_right])
    
    return vertices, faces, uv_coords

# Function to visualize UV mapping on 2D texture
def visualize_uv_mapping(uv_coordinates, faces, texture):
    fig, ax = plt.subplots(figsize=(6, 6))
    
    ax.imshow(texture, extent=[0, 1, 0, 1], origin="upper")
    
    uv_coords = np.array(uv_coordinates)

    for face in faces:
        uv_face = [uv_coords[idx] for idx in face]
        poly = plt.Polygon(uv_face, edgecolor='red', fill=False, linewidth=1)
        ax.add_patch(poly)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("U Coordinate")
    ax.set_ylabel("V Coordinate")
    ax.set_title("UV Mapping on Texture")
    plt.grid(True)
    plt.show()

# Function to visualize the 3D textured sphere
def visualize_textured_sphere(points, faces, uv_coordinates, texture):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    vertices = np.array(points)
    uv_coords = np.array(uv_coordinates)

    for face in faces:
        face_vertices = [vertices[idx] for idx in face]
        face_uvs = [uv_coords[idx] for idx in face]
        
        face_color = np.mean([sample_texture(uv, texture) for uv in face_uvs], axis=0)
        
        ax.add_collection3d(Poly3DCollection([face_vertices], facecolor=face_color, edgecolor='k', alpha=0.9))

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    plt.title("Textured 3D Sphere")
    plt.show()

# Generate sphere with proper spherical UV mapping
points, planes, uv_coordinates = generate_sphere(radius=1, latitude_divisions=16, longitude_divisions=16)

# Visualize 2D UV Mapping
visualize_uv_mapping(uv_coordinates, planes, texture)

# Visualize 3D textured sphere
visualize_textured_sphere(points, planes, uv_coordinates, texture)
