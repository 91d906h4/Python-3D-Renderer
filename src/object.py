# Import libraries
import numpy as np


class Object:
    def __init__(self) -> None:
        """Object class constructor."""

        self.center = np.zeros(3, dtype=np.float32)
        self.origin = np.zeros(3, dtype=np.float32)
        self.vertices = np.empty((0, 3), dtype=np.float32)
        self.lines = []
        self.planes = []

    def __update_center(self) -> None:
        """Update the center of the object."""

        self.center = np.mean(self.vertices, axis=0)

    def set_origin(self, origin: list) -> None:
        """Set the origin of the object.

        Args:
            origin (list): Origin of the object.
        """

        self.origin[:] = origin

    def add_vertices(self, vertices: list) -> None:
        """Add vertices to the object.

        Args:
            vertices (list): Vertices to add.
        """

        self.vertices = np.append(self.vertices, vertices, axis=0)

        self.__update_center() # Update the center

    def add_lines(self, lines: list, color: list) -> None:
        """Add lines to the object.

        Args:
            planes (list): Lines to add (list of vertex indices).
            color (list): Color for the lines ([R, G, B] values).
        """

        self.lines.extend(zip(lines, [color] * len(lines)))

    def add_planes(self, planes: list, color: list) -> None:
        """Add planes to the object.

        Args:
            planes (list): Planes to add (list of vertex indices).
            color (list): Color for the planes ([R, G, B] values).
        """

        self.planes.extend(zip(planes, [color] * len(planes)))
