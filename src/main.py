# Import libraries
from camera import Camera
from object import Object
from operation import Operation
from screen import Camera, Screen


def main() -> None:
    object = Object()
    object.set_origin([0, 0, -3])
    object.add_vertices(vertices=[
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
        [0, 0, 5],
    ])
    object.add_planes(planes=[
        [0, 1, 2], [2, 3, 0],
        [4, 5, 6], [6, 7, 4],
        [0, 1, 5], [5, 4, 0],
    ], color=[0, 255, 0])
    object.add_planes(planes=[
        [1, 2, 6], [6, 5, 1],
        [2, 3, 6], [6, 7, 3],
        [0, 3, 7], [7, 4, 0],
    ], color=[255, 0, 0])
    object.add_lines(lines=[
        [0, 8],
    ], color=[0, 255, 0])

    object1 = Object()
    object1.set_origin([-5, 0, -10])
    object1.add_vertices(vertices=[
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
        [0, 0, 2.5],
    ])
    object1.add_planes(planes=[
        [0, 1, 2], [2, 3, 0],
        [4, 5, 6], [6, 7, 4],
        [0, 1, 5], [5, 4, 0],
    ], color=[0, 255, 255])
    object1.add_planes(planes=[
        [1, 2, 6], [6, 5, 1],
        [2, 3, 6], [6, 7, 3],
        [0, 3, 7], [7, 4, 0],
    ], color=[255, 0, 255])
    object1.add_lines(lines=[
        [0, 8],
    ], color=[0, 255, 0])

    object2 = Object()
    object2.set_origin([0, 0, 0])
    object2.add_vertices(vertices=[
        [-10, 0, 10],
        [10, 0, 10],
        [-10, 0, -10],
        [10, 0, -10],
    ])
    object2.add_planes(planes=[
        [0, 1, 2], [3, 2, 1],
    ], color=[255, 255, 255])

    object4 = Object()
    object4.set_origin([0, 0, 0])
    object4.add_vertices(vertices=[
        [0, 0, 0],
        [1, 0, 0],
        [2, 0, 0],
        [0, 0, 1],
        [1, 0, 1],
        [2, 0, 1],
        [0, 0, 2],
        [1, 0, 2],
        [2, 0, 2],
    ])
    object4.add_planes(planes=[
        [3, 1, 0], [1, 3, 4],
        [7, 5, 4], [5, 7, 8],
    ], color=[255, 255, 255])
    object4.add_planes(planes=[
        [4, 2, 1], [2, 4, 5],
        [6, 4, 3], [4, 6, 7],
    ], color=[128, 128, 128])


    #####
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

    # Example usage
    object3 = Object()
    object3.set_origin([0, 0, -5])
    vertices, faces = generate_sphere(radius=1, latitude_divisions=8, longitude_divisions=8)
    object3.add_vertices(vertices=vertices)
    object3.add_planes(planes=faces, color=[255, 255, 0])
    #####

    camera = Camera(origin=[0, 1, 0], direction=[0, 0, -1], up=[0, 1, 0])
    screen = Screen(camera, distance=1.0, width=300, height=225, world_width=2.0, light=[0, 1, 0])

    while True:
        Operation.rotate(object, {"z": 3, "y": 2, "x": 1}, in_place=True)
        Operation.rotate(object1, {"z": -1, "y": -2, "x": -3}, in_place=True)
        Operation.rotate(object3, {"z": -1, "y": -2, "x": -3}, in_place=True)

        screen.project([object, object1, object3, object2, object4])
        screen.render()


if __name__ == "__main__":
    main()
