# Python 3D Renderer

This is a simple 3D renderer build with Python.

## Overview

This project is a real-time 3D renderer implemented in Python. It features a custom rasterization pipeline with shadow mapping, a movable camera, and interactive rendering using Matplotlib. The renderer supports near-plane clipping, Z-buffering, and flat shading to achieve realistic depth and lighting effects.

## Features

- **Z-Buffering**: Ensures correct depth sorting for overlapping objects.
- **Flat Shading**: Computes lighting per polygon for a classic low-poly aesthetic.
- **Near-Plane Clipping**: Avoids rendering artifacts by clipping geometry behind the camera.
- **Interactive Camera Controls**:
  - `WASD` for movement
  - Arrow keys for rotation
  - `E/Q` to move up/down

## Dependencies

```sh
pip install numpy matplotlib numba
```

## Usage

1. Clone the repository

```
git clone https://github.com/your-repo/3d-renderer.git
```

2. Import `Camera`, `Screen`, `Object`, and `Operation`

```py
# Import libraries
from camera import Camera
from object import Object
from screen import Screen
from operation import Operation
```

3. Instantiate `Camera`, `Screen`, and `Object`

```py
# Instantiate cameta
camera = Camera(
    origin=[0, 0, 0],
    direction=[0, 0, 1],
    up=[0, 1, 0],
)

# Instantiate screen
screen = Screen(
    camera,
    distance=1.0,
    width=800,
    height=640,
    world_width=2.0,
    light=[0, 1, 0], # Environment light source
)

# Instantiate objects
object = Object() # This is a plane
object.add_vertices(vertices=[
    [0, 0, 0], [1, 0, 0],
    [1, 1, 0], [0, 1, 0],
])
object.add_planes(planes=[
    [0, 1, 2], [2, 3, 0]
], color=[0, 255, 255])
```

4. Render the scene

```py
while True:
    # Project to screen
    screen.project([object]) # The input is a list of objects

    # Rendering
    screen.render()
```

## Demo

Single Cube Rotation

![](./assets/cube.gif)

A simple game rendered by Python 3D Renderer.

![](./assets/game.png)
