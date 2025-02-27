# Import libraries
import math

def generate_sphere(radius, latitude_divisions, longitude_divisions):
    vertices = []
    faces = []
    
    for i in range(latitude_divisions + 1):
        lat = math.pi * i / latitude_divisions  # Latitude angle
        for j in range(longitude_divisions):
            lon = 2 * math.pi * j / longitude_divisions  # Longitude angle
            
            # Convert spherical coordinates to Cartesian coordinates
            x = radius * math.sin(lat) * math.cos(lon)
            y = radius * math.sin(lat) * math.sin(lon)
            z = radius * math.cos(lat)
            vertices.append([x, y, z])
    
    # Generate faces (triangles)
    for i in range(latitude_divisions):
        for j in range(longitude_divisions):
            # Vertices of the current face
            top_left = i * longitude_divisions + j
            top_right = top_left + 1
            bottom_left = (i + 1) * longitude_divisions + j
            bottom_right = bottom_left + 1
            
            # Wrap around the faces at the edges
            if j == longitude_divisions - 1:
                top_right -= longitude_divisions
                bottom_right -= longitude_divisions
            
            # Add two triangles per face
            faces.append([top_left, bottom_left, top_right])
            faces.append([top_right, bottom_left, bottom_right])
    
    return vertices, faces