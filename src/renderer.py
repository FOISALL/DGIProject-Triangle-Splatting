import pygame as pygame
import numpy as np
import math # For sin, cos, abs, max
from render_utils import *
from dataclasses import dataclass

class Globals:
    PI = 3.14159265358979323846
    epsilon = 0.000001
    
    SCREEN_WIDTH = 500
    SCREEN_HEIGHT = 500
    t = 0  # timer
    
    triangles = []
    focalLength = 500
    cameraPosition = np.array([0, 0, -3.001])
    currentColor = None
    depthBuffer = None  # Will initialize later
    
    # Movement
    delta = 0.1  # movement speed modifier
    yaw = 0.05
    R = np.eye(3)  # 3x3 identity matrix

    # Initialize depth buffer
    depthBuffer = np.full((SCREEN_HEIGHT, SCREEN_WIDTH), float('inf'))

def vertex_shader(v, p):
    """
    Args:
        v: Input 3D point (numpy array [x,y,z] or similar)
        p: Pixel object to be modified
    """
    p_camera = np.array(v) - Globals.cameraPosition
    
    # Apply rotation matrix
    x, y, z = p_camera@ Globals.R
  
    
    # Perspective projection
    p.zinv = 1.0 / z if z != 0 else float('inf')

    if z > 0:
        # Perspective divide and viewport transform
        p.x = int(Globals.focalLength * x / z) + (Globals.SCREEN_WIDTH // 2)
        p.y = int(Globals.focalLength * y / z) + (Globals.SCREEN_HEIGHT // 2)
    else:
        # Point is behind camera
        p.x = p.y = -1


def interpolate_pvector(a: np.ndarray, b: np.ndarray, num_results: int) -> list[np.ndarray]:
    if num_results <= 0:
        return []

    if num_results == 1:
        return [a.copy()]
    
    step = (b - a) / (num_results - 1)
    results = []
    for i in range(num_results):
        results.append(a + step * i)

    return results

def interpolate_pixel(a: Pixel, b: Pixel, num_results: int) -> list[Pixel]:

    if num_results <= 0:
        return []
    
    if num_results == 1:
        return [Pixel(a.x, a.y, a.zinv)]
    
    # Calculate steps for each component
    x_step = (b.x - a.x) / (num_results - 1)
    y_step = (b.y - a.y) / (num_results - 1)
    z_step = (b.zinv - a.zinv) / (num_results - 1)
    
    results = []
    for i in range(num_results):
        x = int(round(a.x + x_step * i))
        y = int(round(a.y + y_step * i))
        zinv = a.zinv + z_step * i
        results.append(Pixel(x, y, zinv))
    
    return results

def update_rotation(yaw: float):

    # Create temporary yaw rotation matrix (Y-axis rotation)
    temp = np.array([
        [math.cos(yaw), 0, -math.sin(yaw)],  # X column
        [0,        1,  0],         # Y column 
        [math.sin(yaw), 0,  math.cos(yaw)]   # Z column
    ])
    
    # Matrix multiplication: R = R @ temp (accumulate rotation)
    Globals.R = Globals.R @ temp

def draw_line(a: Pixel, b: Pixel, col: np.ndarray, surface: pygame.Surface):
    """
    Draws a line between two pixels with the specified color
    
    Args:
        a: Start Pixel
        b: End Pixel
        col: Color as numpy array [r, g, b] (0-1 range)
        surface: Pygame surface to draw on
    """
    # Calculate number of interpolation points based on edge length
    delta = np.array([abs(a.x - b.x), abs(a.y - b.y)])
    npixels = int(max(delta[0], delta[1])) + 1
    
    # Get interpolated pixels
    line_pixels = interpolate_pixel(a, b, npixels)
    
    # Convert color from 0-1 range to 0-255
    color_rgb = (col * 255).astype(int)
    pygame_color = (color_rgb[0], color_rgb[1], color_rgb[2])
    
    # Draw each pixel
    for p in line_pixels:
        if (0 <= p.x < Globals.SCREEN_WIDTH and 
            0 <= p.y < Globals.SCREEN_HEIGHT):
            surface.set_at((p.x, p.y), pygame_color)

def draw_polygon_edges(vertices: list[Pixel], surface: pygame.Surface, color: np.ndarray = None):
    """
    Draws the edges of a polygon given its vertices
    
    Args:
        vertices: List of Pixel objects representing polygon vertices
        surface: Pygame surface to draw on
        color: Optional color as numpy array [r, g, b] (0-1 range). Defaults to white.
    """
    if len(vertices) < 2:
        return  # Need at least 2 points
    
    # Default to white if no color specified
    if color is None:
        color = np.array([1.0, 1.0, 1.0])  # White
    
    # Draw edges between consecutive vertices
    for i in range(len(vertices)):
        j = (i + 1) % len(vertices)  # Next vertex with wrap-around
        draw_line(vertices[i], vertices[j], color, surface)

# 1. Initialize Pygame
pygame.init()

# 2. Set up the display window
# Define window dimensions
WIDTH, HEIGHT = 600, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Set the window title
pygame.display.set_caption("Green Pixel Example")

# Define colors (RGB tuples)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

# 3. Game loop
running = True
while running:
    # Event handling
    for event in pygame.event.get():
        # If the user clicks the close button
        if event.type == pygame.QUIT:
            running = False

    # 4. Drawing operations
    # Fill the background with black (or any other color)
    screen.fill(BLACK)

    # Draw a single green pixel at coordinates (100, 150)
    # The draw.pixel() method expects the surface, color, and position
    # The position is an (x, y) tuple, where (0,0) is the top-left corner.
    pygame.draw.circle(screen, GREEN, (100, 150), 1) # A circle with radius 1 is a pixel

    # Alternative way to draw a pixel using set_at (less common for drawing directly, more for manipulating existing pixels)
    # screen.set_at((101, 150), GREEN)

    # In your game loop:
    a = Pixel(100, 100, 0)  # Example start pixel
    b = Pixel(150, 120, 0)  # Example end pixel
    color = np.array([1.0, 0.5, 0.2])  # Orange color (RGB 0-1)

    draw_line(a, b, color, screen)

        # Create some polygon vertices
    polygon = [
        Pixel(90, 140, 0),
        Pixel(160, 140, 0),
        Pixel(200, 120, 0),
        Pixel(180, 160, 0)
    ]

    # Draw with default white color
    draw_polygon_edges(polygon, screen)

    # Or with custom color (red in this case)
    draw_polygon_edges(polygon, screen, np.array([1.0, 0.0, 0.0]))


    # 5. Update the display
    # This makes everything you've drawn visible on the screen
    pygame.display.flip() # or pygame.display.update()

# 6. Quit Pygame
pygame.quit()
print("Pygame window closed.")