import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

# Cube vertex positions
vertices = [
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1]
]

# Cube edges (not needed for textures, but useful for reference)
edges = [
    (0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), 
    (0,4), (1,5), (2,6), (3,7)
]

# Cube faces
faces = [
    (0,1,2,3), (3,2,6,7), (7,6,5,4), (4,5,1,0), (1,5,6,2), (4,0,3,7)
]

# Texture coordinates (corresponding to cube faces)
tex_coords = [
    (0, 0), (1, 0), (1, 1), (0, 1)
]

def load_texture(image_path):
    texture_surface = pygame.image.load(image_path)
    texture_data = pygame.image.tostring(texture_surface, "RGB", 1)
    width, height = texture_surface.get_width(), texture_surface.get_height()
    
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, texture_data)
    
    return texture_id

def draw_cube():
    glBindTexture(GL_TEXTURE_2D, texture)
    glBegin(GL_QUADS)
    for i, face in enumerate(faces):
        for j, vertex in enumerate(face):
            glTexCoord2fv(tex_coords[j])  # Map texture coordinate
            glVertex3fv(vertices[vertex])  # Map vertex position
    glEnd()

def main():
    global texture
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)
    
    texture = load_texture("texture.png")  # Load your texture image

    glEnable(GL_TEXTURE_2D)  # Enable textures
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                running = False

        glRotatef(1, 3, 1, 1)  # Rotate cube
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        draw_cube()
        pygame.display.flip()
        pygame.time.wait(10)

    pygame.quit()

if __name__ == "__main__":
    main()
