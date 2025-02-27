# Import libraries
from camera import Camera
from object import Object
from screen import Screen
from operation import Operation
from utils.sphere import generate_sphere


def game() -> None:
    floor = Object()
    floor.set_origin([0, 0, 0])
    floor.add_vertices(vertices=[
        [0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0],
        [0, 0, 1], [1, 0, 1], [2, 0, 1], [3, 0, 1],
        [0, 0, 2], [1, 0, 2], [2, 0, 2], [3, 0, 2],
        [0, 0, 3], [1, 0, 3], [2, 0, 3], [3, 0, 3],
        [0, 0, 4], [1, 0, 4], [2, 0, 4], [3, 0, 4],
        [0, 0, 5], [1, 0, 5], [2, 0, 5], [3, 0, 5],
    ])
    floor.add_planes(planes=[
        [0, 4, 1], [1, 4, 5], [1, 5, 2], [2, 5, 6],
        [2, 6, 3], [3, 6, 7], [6, 10, 7], [7, 10, 11],
        [10, 14, 11], [11, 14, 15], [9, 13, 10], [10, 13, 14],
        [8, 12, 9], [9, 12, 13], [12, 16, 13], [13, 16, 17],
        [16, 20, 17], [17, 20, 21], [17, 21, 18], [18, 21, 22],
        [18, 22, 19], [19, 22, 23],
    ], color=[255, 255, 255])


    object1 = Object()
    vertices, faces = generate_sphere(radius=0.3, latitude_divisions=8, longitude_divisions=8)
    object1.add_vertices(vertices=vertices)
    object1.add_planes(planes=faces, color=[255, 255, 0])
    object1.set_origin([0.5, 0.5, 0.5])
    
    object2 = Object()
    object2.add_vertices(vertices=[
        [0, 0, 0],
        [0.5, 0, 0],
        [0.5, 0.5, 0],
        [0, 0.5, 0],
        [0, 0, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0.5],
        [0, 0.5, 0.5],
    ])
    object2.add_planes(planes=[[0, 1, 2], [2, 3, 0]], color=[0, 255, 255])
    object2.add_planes(planes=[[4, 5, 6], [6, 7, 4]], color=[0, 255, 255])
    object2.add_planes(planes=[[0, 1, 5], [5, 4, 0]], color=[0, 255, 255])
    object2.add_planes(planes=[[1, 2, 6], [6, 5, 1]], color=[0, 255, 255])
    object2.add_planes(planes=[[2, 3, 6], [6, 7, 3]], color=[0, 255, 255])
    object2.add_planes(planes=[[0, 3, 7], [7, 4, 0]], color=[0, 255, 255])
    object2.set_origin(object2.center + [2.0, 0, 4.0])

    camera = Camera(origin=[0, 1, 0], direction=[0, 0, 1], up=[0, 1, 0])
    screen = Screen(camera, distance=1.0, width=300, height=225, world_width=2.0, light=[0, 1, 0])

    while True:
        Operation.rotate(object1, {"z": -1, "y": -2, "x": -3}, in_place=True)
        Operation.rotate(object2, {"z": 1, "y": 2, "x": 3}, in_place=True)

        screen.project([floor, object1, object2])
        screen.render()

        if camera.origin[0] > 2 and camera.origin[2] > 4:
            print("You win!")
            break


if __name__ == "__main__":
    game()
