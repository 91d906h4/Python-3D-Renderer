# Import libraries
import numpy as np


class Camera:
    def __init__(self, origin: np.ndarray, direction: np.ndarray, up: np.ndarray) -> None:
        """Camera constructor.

        Args:
            origin (np.ndarray): The camera's origin.
            direction (np.ndarray): The camera's direction.
            up (np.ndarray): The camera's up vector.
        """

        self.origin = origin
        self.direction = direction / np.linalg.norm(direction) # Normalize
        up = up / np.linalg.norm(up) # Normalize
        self.right = np.cross(self.direction, up) # Compute right vector
        self.up = np.cross(self.right, self.direction) # Recompute up to ensure orthogonality

    def move(self, delta: np.ndarray) -> None:
        """Move the camera by a given delta.

        Args:
            delta (np.ndarray): The delta to move the camera by.
        """

        self.origin += delta

    def rotate(self, axis: np.ndarray, angle: float) -> None:
        """Rotates the camera around a given axis by a certain angle.

        Args:
            axis (np.ndarray): The axis to rotate around.
            angle (float): The angle to rotate by.
        """

        # Normalize the rotation axis
        axis = axis / np.linalg.norm(axis)

        # Compute rotation matrix using Rodrigues' rotation formula
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        one_minus_cos = 1 - cos_angle

        # Rotation matrix around arbitrary axis
        rot_matrix = np.array([
            [cos_angle + axis[0]**2 * one_minus_cos, axis[0] * axis[1] * one_minus_cos - axis[2] * sin_angle, axis[0] * axis[2] * one_minus_cos + axis[1] * sin_angle],
            [axis[1] * axis[0] * one_minus_cos + axis[2] * sin_angle, cos_angle + axis[1]**2 * one_minus_cos, axis[1] * axis[2] * one_minus_cos - axis[0] * sin_angle],
            [axis[2] * axis[0] * one_minus_cos - axis[1] * sin_angle, axis[2] * axis[1] * one_minus_cos + axis[0] * sin_angle, cos_angle + axis[2]**2 * one_minus_cos]
        ])

        # Rotate direction and right vectors
        self.direction = np.dot(rot_matrix, self.direction)
        self.right = np.dot(rot_matrix, self.right)

        # Recompute up vector to maintain orthogonality
        self.up = np.cross(self.right, self.direction)
        self.up /= np.linalg.norm(self.up)
