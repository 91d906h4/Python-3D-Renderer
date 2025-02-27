import numpy as np
import pygame
import math

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# 3D Plane (Triangle) defined by 3 points
plane_3d = np.array([
    [-1, -1, 5],  # Point A
    [1, -1, 5],   # Point B
    [0, 1, 5]     # Point C
])

# Rotation matrix for Y-axis
def rotation_matrix_y(theta):
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    return np.array([
        [cos_t, 0, sin_t],
        [0, 1, 0],
        [-sin_t, 0, cos_t]
    ])

# Simple perspective projection
def project_3d_to_2d(point3d):
    """ Projects 3D point to 2D screen using simple perspective projection. """
    focal_length = 500  # Adjust for zoom effect
    x, y, z = point3d
    x2d = int(WIDTH / 2 + (x * focal_length) / z)
    y2d = int(HEIGHT / 2 - (y * focal_length) / z)
    return x2d, y2d

# Scanline triangle fill
def draw_filled_triangle(p1, p2, p3, color):
    """ Draws a filled triangle using a simple scanline algorithm. """
    pygame.draw.polygon(screen, color, [p1, p2, p3])

def main():
    angle = 0
    running = True

    while running:
        screen.fill((0, 0, 0))  # Clear screen

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Compute rotation
        angle += 0.01  # Rotate over time
        rot_matrix = rotation_matrix_y(angle)

        # Rotate and project each point
        rotated_points = [rot_matrix @ p for p in plane_3d]
        projected_points = [project_3d_to_2d(p) for p in rotated_points]

        # Draw filled triangle
        draw_filled_triangle(*projected_points, (0, 255, 0))

        # Update screen
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
