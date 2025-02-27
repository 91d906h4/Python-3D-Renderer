# Import libraries
import time

import numpy as np
import matplotlib.pyplot as plt

from camera import Camera
from object import Object
from numba import njit, prange
from matplotlib.backend_bases import KeyEvent


class Screen:
    def __init__(self, camera: Camera, distance: float, width: int, height: int, world_width: float, light: np.ndarray, near: float = 0.1, parallel: bool = False) -> None:
        """Screen constructor.

        Args:
            camera (Camera): The camera object.
            distance (float): The distance from the camera to the screen.
            width (int): The width of the screen.
            height (int): The height of the screen.
            world_width (float): The width of the world.
            light (np.ndarray): The light direction.
            near (float, optional): The distance to the near plane. Defaults to 0.1.
            parallel (bool, optional): If True, the screen is parallel to the camera. Defaults to False.
        """

        self.camera = camera
        self.distance = distance
        self.width = width
        self.height = height
        self.world_width = world_width
        self.world_height = (world_width * height) / width
        self.light = light / np.linalg.norm(light) # Normalize
        self.near = near
        self.parallel = parallel

        # Compute screen center position
        self.center = self.camera.origin + self.camera.direction * self.distance

        # Compute the screen corners
        half_w = self.camera.right * (self.world_width / 2)
        half_h = self.camera.up * (self.world_height / 2)
        self.bottom_left = self.center - half_w - half_h

        # Initialize the frame
        self.__init_frame()

    def __init_frame(self) -> None:
        """Initialize the frame."""

        # Initialize blank frame
        self.frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        self.figure, self.ax = plt.subplots()
        self.im = self.ax.imshow(self.frame)

        # Connect event handlers
        self.figure.canvas.mpl_connect('key_press_event', self.__key_press_event)

        # Disable default key press handler
        self.figure.canvas.mpl_disconnect(self.figure.canvas.manager.key_press_handler_id)

        plt.ion() # Enable interactive mode
        plt.show()

    def __key_press_event(self, event: KeyEvent) -> None:
        """Handle key press events to update camera position and orientation.

        Args:
            event (KeyEvent): The key press event.
        """

        if event.key == 'w': # Move camera forward
            # Fix Y-axis to preserve the camera height
            self.camera.move(np.array([self.camera.direction[0], 0, self.camera.direction[2]]) * 0.1)
        elif event.key == 's': # Move camera backward
            # Fix Y-axis to preserve the camera height
            self.camera.move(np.array([self.camera.direction[0], 0, self.camera.direction[2]]) * -0.1)
        elif event.key == 'd': # Move camera right
            self.camera.move(self.camera.right * 0.1)
        elif event.key == 'a': # Move camera left
            self.camera.move(self.camera.right * -0.1)
        elif event.key == 'e': # Move camera up
            self.camera.move(self.camera.up * 0.1)
        elif event.key == 'q': # Move camera down
            self.camera.move(self.camera.up * -0.1)
        elif event.key == 'up': # Rotate camera up
            self.camera.rotate(self.camera.right, np.pi / 100)
        elif event.key == 'down': # Rotate camera down
            self.camera.rotate(self.camera.right, -np.pi / 100)
        elif event.key == 'left': # Rotate camera left
            # Force the camera to rotate left around the y-axis
            self.camera.rotate(np.array([0, 1, 0]), np.pi / 100)
        elif event.key == 'right': # Rotate camera right
            # Force the camera to rotate right around the y-axis
            self.camera.rotate(np.array([0, 1, 0]), -np.pi / 100)

    def __world_to_camera(self, vertex: np.ndarray) -> np.ndarray:
        """Convert a world coordinate vertex to camera space."""

        relative = vertex - self.camera.origin
        X_cam = np.dot(relative, self.camera.right)
        Y_cam = np.dot(relative, self.camera.up)
        Z_cam = np.dot(relative, self.camera.direction)

        return np.array([X_cam, Y_cam, Z_cam])

    def __project_vertex(self, cam_vertex: np.ndarray) -> tuple:
        """Project a camera-space vertex onto screen pixel coordinates.

        Args:
            cam_vertex (np.ndarray): A 3D vertex in camera space.

        Returns:
            tuple: A tuple (pixel_x, pixel_y, Z_cam)
        """

        X_cam, Y_cam, Z_cam = cam_vertex

        # Perspective projection
        x_screen = (X_cam / Z_cam) * self.distance
        y_screen = (Y_cam / Z_cam) * self.distance

        # Map to pixel coordinates (with y flipped so (0,0) is top-left)
        pixel_x = int(((x_screen + self.world_width / 2) / self.world_width) * self.width)
        pixel_y = self.height - int(((y_screen + self.world_height / 2) / self.world_height) * self.height) - 1

        return (pixel_x, pixel_y, Z_cam)

    def __clip_polygon_to_near_plane(self, vertices: list[np.ndarray]) -> list[np.ndarray]:
        """Clips a polygon against the near plane.

        Each vertex is a np.array([X, Y, Z]). Returns a new list of vertices.

        Args:
            vertices (list): A list of camera-space vertices.

        Returns:
            list: A list of camera-space vertices.
        """

        clipped = []
        num_vertices = len(vertices)

        for i in range(num_vertices):
            curr = vertices[i]
            nxt = vertices[(i + 1) % num_vertices]

            curr_inside = curr[2] > self.near
            nxt_inside = nxt[2] > self.near

            if curr_inside:
                clipped.append(curr)

            # If the edge crosses the near plane, compute intersection
            if curr_inside != nxt_inside:
                # t gives the interpolation factor where the edge intersects the plane Z == near
                t = (self.near - curr[2]) / (nxt[2] - curr[2])

                intersection = curr + t * (nxt - curr)
                clipped.append(intersection)

        return clipped

    def __clip_line_to_near_plane(self, v1: np.ndarray, v2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Clips a line segment defined by two camera-space vertices to the near plane.

        Returns a tuple (new_v1, new_v2) if the line is partially or fully in front of the near plane,
        or None if the entire line is behind.

        Args:
            v1 (np.ndarray): A 3D vertex in camera space.
            v2 (np.ndarray): A 3D vertex in camera space.

        Returns:
            tuple: A tuple (new_v1, new_v2)
        """

        inside1 = v1[2] > self.near
        inside2 = v2[2] > self.near

        # Both vertices are inside
        if inside1 and inside2:
            return v1, v2

        # Both vertices are outside
        if not inside1 and not inside2:
            return None

        # Vertex 2 is outside
        if inside1:
            t = (self.near - v1[2]) / (v2[2] - v1[2])
            new_v2 = v1 + t * (v2 - v1)
            return v1, new_v2

        # Vertex 1 is outside
        else:
            t = (self.near - v2[2]) / (v1[2] - v2[2])
            new_v1 = v2 + t * (v1 - v2)
            return new_v1, v2

    def __draw_plane(self, v1: tuple, v2: tuple, v3: tuple, color: list, frame: np.ndarray, z_buffer: np.ndarray) -> None:
        """Draw a triangle on the screen.

        Args:
            v1 (tuple): First vertex of the triangle.
            v2 (tuple): Second vertex of the triangle.
            v3 (tuple): Third vertex of the triangle.
            color (list): Color of the triangle.
            frame (np.ndarray): The frame to draw on.
            z_buffer (np.ndarray): The Z-buffer.
        """

        x1, y1, z1 = v1
        x2, y2, z2 = v2
        x3, y3, z3 = v3

        # Compute bounding box
        min_x = max(min(x1, x2, x3), 0)
        max_x = min(max(x1, x2, x3), self.width - 1)
        min_y = max(min(y1, y2, y3), 0)
        max_y = min(max(y1, y2, y3), self.height - 1)

        denom = float((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))
        if denom == 0:
            return

        # Rasterize the triangle using barycentric coordinates
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                w1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denom
                w2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denom
                w3 = 1 - w1 - w2

                if w1 >= 0 and w2 >= 0 and w3 >= 0:
                    # Interpolate depth
                    z = w1 * z1 + w2 * z2 + w3 * z3

                    if z < z_buffer[y, x]:
                        z_buffer[y, x] = z
                        frame[y, x] = color

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def __draw_plane_parallel(v1: tuple, v2: tuple, v3: tuple, color: list, frame: np.ndarray, z_buffer: np.ndarray) -> None:
        """Draw a triangle using Numba JIT for high performance.

        Args:
            v1 (tuple): First vertex of the triangle.
            v2 (tuple): Second vertex of the triangle.
            v3 (tuple): Third vertex of the triangle.
            color (list): Color of the triangle.
            frame (np.ndarray): The frame to draw on.
            z_buffer (np.ndarray): The Z-buffer.
        """

        x1, y1, z1 = v1
        x2, y2, z2 = v2
        x3, y3, z3 = v3

        height, width, _ = frame.shape

        # Compute bounding box
        min_x = max(min(x1, x2, x3), 0)
        max_x = min(max(x1, x2, x3), width - 1)
        min_y = max(min(y1, y2, y3), 0)
        max_y = min(max(y1, y2, y3), height - 1)

        # Compute denominator for barycentric coordinates
        denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)

        # Degenerate triangle
        if denom == 0: return

        inv_denom = 1.0 / denom

        # Precompute step values for incremental updates
        dw1dx = (y2 - y3) * inv_denom
        dw2dx = (y3 - y1) * inv_denom
        dw3dx = -dw1dx - dw2dx

        # Parallelized loop for each row
        for y in prange(min_y, max_y + 1):  # Parallel loop using prange
            # Compute initial barycentric coordinates for the row
            w1 = ((y2 - y3) * (min_x - x3) + (x3 - x2) * (y - y3)) * inv_denom
            w2 = ((y3 - y1) * (min_x - x3) + (x1 - x3) * (y - y3)) * inv_denom
            w3 = 1 - w1 - w2

            for x in range(min_x, max_x + 1):
                if w1 >= 0 and w2 >= 0 and w3 >= 0:
                    # Interpolate depth
                    z = w1 * z1 + w2 * z2 + w3 * z3
                    if z < z_buffer[y, x]:
                        z_buffer[y, x] = z
                        frame[y, x, 0] = color[0]  # R
                        frame[y, x, 1] = color[1]  # G
                        frame[y, x, 2] = color[2]  # B

                # Move to the next pixel in x direction
                w1 += dw1dx
                w2 += dw2dx
                w3 += dw3dx

    def __draw_line(self, v1: tuple, v2: tuple, color: list, frame: np.ndarray, z_buffer: np.ndarray) -> None:
        """Draw a line on the screen.

        Args:
            v1 (tuple): First vertex of the line.
            v2 (tuple): Second vertex of the line.
            color (list): Color of the line.
            frame (np.ndarray): The frame to draw on.
            z_buffer (np.ndarray): The Z-buffer.
        """

        x1, y1, z1 = v1
        x2, y2, z2 = v2

        steps = int(max(abs(x2 - x1), abs(y2 - y1))) + 1
        xs = np.linspace(x1, x2, steps)
        ys = np.linspace(y1, y2, steps)
        zs = np.linspace(z1, z2, steps)

        for x, y, z in zip(xs, ys, zs):
            ix = int(round(x))
            iy = int(round(y))

            if 0 <= ix < self.width and 0 <= iy < self.height:
                if z < z_buffer[iy, ix]:
                    z_buffer[iy, ix] = z
                    frame[iy, ix] = color

    def __flat_shading(self, plane: np.ndarray, center: np.ndarray, color: list[int]) -> list:
        """Applies flat shading to a plane.

        Args:
            plane (np.ndarray): The plane to apply flat shading to.
            center (np.ndarray): The center of the object.
            color (list[int]): The color of the plane.

        Returns:
            list: The shaded color of the plane.
        """

        v1, v2, v3 = plane

        normal = np.cross(v2 - v1, v3 - v1)
        norm = np.linalg.norm(normal)

        # Avoid division by zero
        if norm < 1e-6:
            return color

        # Adjust normal direction based on the relation between the centers of the plane and object
        face_center = np.mean(plane, axis=0)
        if np.dot(normal / norm, face_center - center) < 0:
            normal *= -1

        factor = np.clip(np.dot(normal / norm, self.light), 0, 1) * 0.5 + 0.5 # Normalize

        return [int(c * factor) for c in color]

    def project(self, objects: list[Object]) -> None:
        """
        Projects a 3D object onto the 2D screen using near-plane clipping and Z-buffered triangle rasterization.

        Args:
            object (Object): The 3D object to project.

        Returns:
            np.ndarray: A 2D image (frame) of the projected object.
        """

        # Initialize a black frame and a Z-buffer
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        z_buffer = np.full((self.height, self.width), np.inf)

        for object in objects:
            # First, apply the object’s origin offset to its vertices
            world_vertices = object.vertices + object.origin
            world_center = object.center + object.origin

            # Process each plane (triangle) of the object
            for vertex_indices, color in object.planes:
                # Convert the triangle's vertices to camera space
                cam_vertices = [self.__world_to_camera(world_vertices[v]) for v in vertex_indices]

                # Falt shading
                color = self.__flat_shading(world_vertices[vertex_indices], world_center, color)

                # Clip the triangle against the near plane
                clipped_cam_vertices = self.__clip_polygon_to_near_plane(cam_vertices)

                # Not enough vertices to form a visible polygon
                if len(clipped_cam_vertices) < 3:
                    continue

                # Triangulate the (possibly clipped) polygon using fan triangulation
                for i in range(1, len(clipped_cam_vertices) - 1):
                    triangle = [clipped_cam_vertices[0], clipped_cam_vertices[i], clipped_cam_vertices[i + 1]]

                    # Project each vertex from camera space to pixel coordinates
                    projected = [self.__project_vertex(v) for v in triangle]

                    if self.parallel: # Use CPU parallelization
                        self.__draw_plane_parallel(projected[0], projected[1], projected[2], color, frame, z_buffer)
                    else:
                        self.__draw_plane(projected[0], projected[1], projected[2], color, frame, z_buffer)

            # Process each line of the object
            for vertex_indices, color in object.lines:
                v1_world = world_vertices[vertex_indices[0]]
                v2_world = world_vertices[vertex_indices[1]]

                v1_cam = self.__world_to_camera(v1_world)
                v2_cam = self.__world_to_camera(v2_world)
                clipped_line = self.__clip_line_to_near_plane(v1_cam, v2_cam)

                if clipped_line is None:
                    continue

                v1_clip, v2_clip = clipped_line
                p1 = self.__project_vertex(v1_clip)
                p2 = self.__project_vertex(v2_clip)

                self.__draw_line(p1, p2, color, frame, z_buffer)

        # Update the frame
        self.frame = frame

    def render(self, delay: float = 0.0) -> None:
        """Renders a frame on the screen.

        Args:
            frame (np.ndarray): The frame to render.
            delay (float, optional): Delay in seconds. Defaults to 0.0.
        """

        self.im.set_data(self.frame)
        self.ax.draw_artist(self.ax.patch)
        self.ax.draw_artist(self.im)
        self.figure.suptitle(f"Camera: {self.camera.origin}")
        self.figure.canvas.blit(self.ax.bbox)
        self.figure.canvas.flush_events()

        # Set delay
        time.sleep(delay)
