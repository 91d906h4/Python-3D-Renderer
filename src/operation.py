# Import libraries
import numpy as np

from object import Object


class Operation:
    @staticmethod
    def rotate(object: Object, rotation: dict, in_place: bool = False) -> np.ndarray | None:
        """Object rotation function.

        Rotate the vertices based on the rotation matrix or angles provided.
        Notice that the order of rotation matters since the matrix multiplication is not commutative.

        Args:
            object (Object): Object to rotate.
            rotation (dict): Dictionary containing rotation information (Notice that the key is x, y, or z; the value is angle).
            in_place (bool, optional): If True, the object will be rotated in place. Defaults to False.

        Returns:
            np.ndarray: Rotated vertices.
        """

        # Define rotation matrix
        rotation_matrix = np.eye(3)

        for (direction, angle) in rotation.items():
            # Convert angle to radians
            radians = np.radians(angle)

            if direction == "x":
                rotation_matrix = rotation_matrix @ np.array([
                    [1, 0,               0               ],
                    [0, np.cos(radians), -np.sin(radians)],
                    [0, np.sin(radians), np.cos(radians) ],
                ])

            elif direction == "y":
                rotation_matrix = rotation_matrix @ np.array([
                    [np.cos(radians),  0, np.sin(radians)],
                    [0,                1,               0],
                    [-np.sin(radians), 0, np.cos(radians)],
                ])

            elif direction == "z":
                rotation_matrix = rotation_matrix @ np.array([
                    [np.cos(radians), -np.sin(radians), 0],
                    [np.sin(radians), np.cos(radians),  0],
                    [0,               0,                1],
                ])

        # Perform roration
        rotated_vertices = object.vertices @ rotation_matrix

        if in_place:
            object.vertices = rotated_vertices
        else:
            return rotated_vertices
