import pygame as pygame
import numpy as np
import math # For sin, cos, abs, max
from render_utils import *
from dataclasses import dataclass
from scipy.spatial import KDTree

class Globals:
    PI = 3.14159265358979323846
    epsilon = 0.000001
    
    SCREEN_WIDTH = 1200
    SCREEN_HEIGHT = 800
    t = 0  # timer
    
    triangles = []
    focalLength = 500
    cameraPosition = np.array([0, 0, -3.001])
    currentColor = None
    depthBuffer = None  # Will initialize later
    pointSize = 2

    # Store loaded points
    pointcloudData: list[Point3D] = []
    
    # Movement
    delta = 0.1  # movement speed modifier
    yaw = 0.05
    pitch = 0.05
    R = np.eye(3)  # 3x3 identity matrix

    # Initialize depth buffer
    depthBuffer = np.full((SCREEN_HEIGHT, SCREEN_WIDTH), float(0))

    # Visualization controls
    scale_factor = 0.2  # Scale down large coordinates
    show_debug = True

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
    p.zinv = 1.0 / z if z != 0 else float(0)

    epsilon = 1e-6
    if z-epsilon > 0:
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

def update_rotation(yaw: float, pitch: float):

    # Create temporary yaw rotation matrix (Y-axis rotation)
    yawMat = np.array([
        [math.cos(yaw), 0, -math.sin(yaw)],  # X column
        [0,             1,              0],         # Y column 
        [math.sin(yaw), 0,  math.cos(yaw)]   # Z column
    ])
    
    # Matrix multiplication: R = R @ temp (accumulate rotation)
    Globals.R = Globals.R @ yawMat

    if pitch != 0:
        cos_p = math.cos(pitch)
        sin_p = math.sin(pitch)
        pitchMat = np.array([
            [1,     0,      0],
            [0, cos_p, -sin_p],
            [0, sin_p, cos_p]
        ])
        Globals.R = Globals.R @ pitchMat

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

def ComputePolygonRows(vertexPixels: list[Pixel]):
    # 1. Find min and max y-values
    minY = float('inf')
    maxY = -float('inf')

    for p in vertexPixels:
        if p.y < minY:
            minY = p.y
        if (p.y > maxY):
            maxY = p.y
    
    
    # 2. Calculate number of rows needed
    rows = int(maxY - minY) + 1
    
    # 3. Initialize left and right boundaries
    leftPixels = []
    rightPixels = []
    for i in range(rows):
        # Initialize with extreme values
        leftPixels.append(Pixel(float('inf'), int(minY) + i, float('inf')))
        rightPixels.append(Pixel(-float('inf'), int(minY) + i, -float('inf')))
    
    # 4. Process each edge
    for i in range(len(vertexPixels)):
        j = (i + 1) % len(vertexPixels)  # Next vertex (wraps around)
        start = vertexPixels[i]
        end = vertexPixels[j]
        
        # Skip horizontal edges
        if start.y == end.y: 
            continue
        
        
        # Determine top and bottom points
        if start.y < end.y:
            top = start
            bottom = end
        else:
            top = end
            bottom = start
        
        
        # Calculate steps needed
        steps = int(bottom.y - top.y) + 1
        
        # Interpolate along edge
        edgePixels = interpolate_pixel(top, bottom, steps)

        # Update left and right boundaries
        for p in edgePixels:
            row = int(p.y - minY)
            if 0 <= row < rows:
                # Update left boundary
                if (p.x < leftPixels[row].x or 
                   (p.x == leftPixels[row].x and p.zinv > leftPixels[row].zinv)):
                    leftPixels[row].x = p.x
                    leftPixels[row].zinv = p.zinv
                # Update right boundary
                if (p.x > rightPixels[row].x or 
                   (p.x == rightPixels[row].x and p.zinv > rightPixels[row].zinv)):
                    rightPixels[row].x = p.x
                    rightPixels[row].zinv = p.zinv
    return leftPixels, rightPixels


def DrawRows(leftPixels: list[Pixel], rightPixels: list[Pixel], surface: pygame.Surface):
    # Get the current drawing color (converted from 0-1 range to 0-255)
    if Globals.currentColor is None:
        return
    
    color_rgb = (Globals.currentColor * 255).astype(int)
    fillColor = (color_rgb[0], color_rgb[1], color_rgb[2])
    
    # Loop through each row (scanline)
    for row in range(len(leftPixels)):
        left = leftPixels[row]
        right = rightPixels[row]
        
        # Skip if left boundary is right of right boundary (invalid row)
        if left.x > right.x: 
            continue
        
        # Interpolate between left and right boundaries
        rowPixels = interpolate_pixel(left, right, int(right.x - left.x) + 1)

        # Draw each pixel in the row
        for pixel in rowPixels:
            # Only draw if within screen bounds
            if (pixel.x >= 0 and pixel.x < Globals.SCREEN_WIDTH and 
                pixel.y >= 0 and pixel.y < Globals.SCREEN_HEIGHT):
                if pixel.zinv >= Globals.depthBuffer[pixel.y][pixel.x]:  # changing > to >= fixed the weird black lines
                    surface.set_at((pixel.x, pixel.y), fillColor)
                    Globals.depthBuffer[pixel.y][pixel.x] = pixel.zinv

def DrawPolygon(vertices: list[np.ndarray], surface: pygame.Surface):
    V = len(vertices)
    vertexPixels = []
    valid_vertices = len(vertices)

    for i in range(V):
        projected = Pixel(0, 0, 0)
        vertex_shader(vertices[i], projected)  # Project into it
        if projected.x == -1 or \
           projected.x < 0 or projected.x >= Globals.SCREEN_WIDTH or \
           projected.y < 0 or projected.y >= Globals.SCREEN_HEIGHT:
            valid_vertices -= 1
            
        
        vertexPixels.append(projected)  # Add to list
    
    if valid_vertices == 0:
        return
    
    for p in vertexPixels:
        p.x = np.clip(p.x, 0, Globals.SCREEN_WIDTH-1)
        p.y = np.clip(p.y, 0, Globals.SCREEN_HEIGHT-1)
    leftPixels, rightPixels = ComputePolygonRows(vertexPixels)
    DrawRows(leftPixels, rightPixels, surface)




# pointcloud specifics:


def render_point_cloud(

    surface: pygame.Surface

):
    """
    Renders a point cloud onto the Pygame surface with depth testing.

    """
    # Reset depth buffer for this frame
    Globals.depthBuffer.fill(float(0))

    for point in Globals.pointcloudData:
        # Create a temporary Pixel object to use the vertex_shader
        p_screen = Pixel(0, 0, 0.0)
        
        # Convert Point3D's (x,y,z) to a NumPy array for the shader
        v_3d = np.array([point.x, point.y, point.z])

        # Apply the vertex shader to get screen coordinates and zinv

        vertex_shader(v_3d, p_screen)


        # Check if the point is in front of the camera and on screen
        if p_screen.x != -1 and \
           0 <= p_screen.x < Globals.SCREEN_WIDTH and \
           0 <= p_screen.y < Globals.SCREEN_HEIGHT:
            
            # Perform depth test
            if p_screen.zinv > Globals.depthBuffer[p_screen.y, p_screen.x]:
                # This point is behind something already drawn at this pixel, skip it
                continue
            
            # If it's closer, update the depth buffer
            Globals.depthBuffer[p_screen.y, p_screen.x] = p_screen.zinv
            
            # Convert color from 0-255 to Pygame format
            point_color = (point.r, point.g, point.b)
            
            # Draw the point as a single pixel
            # You could also draw a small circle for a "splat" effect, similar to the paper
            # For a single pixel:
            surface.set_at((p_screen.x, p_screen.y), point_color)
            
            # For a small splat (e.g., radius 1 for a 3x3 pixel area):
            # pygame.draw.circle(surface, point_color, (p_screen.x, p_screen.y), 1)

# Debug code by AI
def draw_debug_info(surface: pygame.Surface):
    """Display camera position and controls."""
    if not Globals.show_debug:
        return
    
    font = pygame.font.SysFont('Arial', 20)
    lines = [
        f"Camera Position: {Globals.cameraPosition}",
        f"Controls: WASD - Move, SPACE/SHIFT - Up/Down, Arrows - Rotate",
        f"Points: {len(Globals.pointcloudData)}",
        # f"Scale: {Globals.scale_factor}",
        f"F1 - Toggle debug info"
    ]
    
    for i, line in enumerate(lines):
        text = font.render(line, True, (255, 255, 255))
        surface.blit(text, (10, 10 + i * 25))

# Triangle functions

def initializeTriangle(point: Point3D, neighbours):
    q = [point.x,point.y,point.z]
    U = []
    for i in range(0,3):
        u = np.random.uniform(-1.0,1.0,3)
        U.append(u/ np.linalg.norm(u))


    # finally triangle is scaled according to surrounding point density d and k which is just some random value
    
    d = (neighbours[0][1] +neighbours[1][1] +neighbours[2][1])/3
    k = 2.2 # same initialization as paper

    V = []

    # vi = q + k*d*ui
    V = []
    for u in U:
        v = q + k * d * u  # Proper vector addition
        V.append(v)

    return Triangle(
    vertices=V,
    color=np.array([point.r, point.g, point.b]),
    opacity=0.28,
    sigma=1.16
    )





def initialize_triangles(points: list[Point3D]) -> list[Triangle]:
    print("Start triangle initialization")
    
    # Build KD-tree for fast nearest neighbor queries
    coords = np.array([[p.x, p.y, p.z] for p in points])
    kdtree = KDTree(coords)
    
    triangles = []
    for i, point in enumerate(points):
        # Query for 4 nearest neighbors (including self)
        distances, indices = kdtree.query([point.x, point.y, point.z], k=4)
        
        # Get 3 closest neighbors (skip first which is the point itself)
        neighbours = [(points[idx], dist) for idx, dist in zip(indices[1:], distances[1:])]
        
        triangles.append(initializeTriangle(point, neighbours))
    
    print("Triangles initialized")
    return triangles

# Signed Distance Field
def SDF(triangle: Triangle):
    vertices = triangle.vertices
    edges = []
    for i in range(len(vertices)):
        # first we define all the edges of the triangle
        edge = vertices[i] - vertices[i-1]
        pass
    


# get point cloud data

Globals.pointcloudData = load_points3D("south-building/sparse/points3D.txt")

# get triangle initialization from pointcloud

Globals.triangles = initialize_triangles(Globals.pointcloudData)

counter = 0

# 1. Initialize Pygame
# pygame.init()
# Add this at startup
pygame.init()
pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)
screen = pygame.display.set_mode((Globals.SCREEN_WIDTH, Globals.SCREEN_HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)
clock = pygame.time.Clock()

# 2. Set up the display window

screen = pygame.display.set_mode((Globals.SCREEN_WIDTH, Globals.SCREEN_HEIGHT))

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
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_F1:
                Globals.show_debug = not Globals.show_debug
        
    # Handle continuous key presses
    keys = pygame.key.get_pressed()
    
    # Movement
    forward = Globals.R @ np.array([0, 0, 1])
    right = Globals.R @ np.array([1, 0, 0])
    up = np.array([0, 1, 0])
    
    if keys[pygame.K_w]: Globals.cameraPosition += forward * Globals.delta
    if keys[pygame.K_s]: Globals.cameraPosition -= forward * Globals.delta
    if keys[pygame.K_a]: Globals.cameraPosition -= right * Globals.delta
    if keys[pygame.K_d]: Globals.cameraPosition += right * Globals.delta
    if keys[pygame.K_LSHIFT]: Globals.cameraPosition += up * Globals.delta
    if keys[pygame.K_SPACE]: Globals.cameraPosition -= up * Globals.delta
    
    # Rotation
    dyaw, dpitch = 0, 0
    if keys[pygame.K_LEFT]: dyaw = Globals.yaw
    if keys[pygame.K_RIGHT]: dyaw = -Globals.yaw
    if keys[pygame.K_UP]: dpitch = Globals.pitch
    if keys[pygame.K_DOWN]: dpitch = -Globals.pitch
    update_rotation(dyaw, dpitch)

    # 4. Drawing operations
    # Fill the background with black (or any other color)
    screen.fill(BLACK)

    Globals.depthBuffer.fill(float(0))  # Clear depth buffer

        # --- Render the point cloud ---
    # render_point_cloud(screen)

    # Check first triangle's vertices
    # if Globals.triangles:
    #     print("First triangle vertices:")
    #     for i, v in enumerate(Globals.triangles[0].vertices):
    #         print(f"Vertex {i}: {v}, shape: {np.shape(v)}")


    i = 0
    for triangle in Globals.triangles:
        i += 1
        if i % 100 == 0:
            #print("good trinagle")
            Globals.currentColor = triangle.color  # Already in 0-1 range
            DrawPolygon(triangle.vertices, screen)
            # Optional: Draw wireframe for debugging
            # vertices_pixels = []
            # for v in triangle.vertices:
            #     p = Pixel(0, 0, 0)
            #     vertex_shader(v, p)
            #     vertices_pixels.append(p)
            # draw_polygon_edges(vertices_pixels, screen, np.array([1.0, 1.0, 1.0]))

        # --- Test triangle rendering ---
    # Globals.depthBuffer.fill(float(0))  # Clear depth buffer
    
    # Create a test triangle right in front of the camera
    # test_triangle = [
    #     np.array([-0.5, -0.5, 1.0]),  # Bottom left
    #     np.array([0.5, -0.5, 1.0]),   # Bottom right
    #     np.array([0.0, 0.5, 1.0])     # Top center
    # ]
    
    # # Set a bright red color for the test triangle
    # Globals.currentColor = np.array([1.0, 0.0, 0.0])  # Red
    
    # # Draw just this one triangle
    # DrawPolygon(test_triangle, screen)
    
        
    
    draw_debug_info(screen)



    # Alternative way to draw a pixel using set_at (less common for drawing directly, more for manipulating existing pixels)
    # screen.set_at((101, 150), GREEN)



    # 5. Update the display
    # This makes everything you've drawn visible on the screen
    pygame.display.flip() # or pygame.display.update()
    clock.tick(60)
    #print(counter)
    counter += 1

# 6. Quit Pygame
pygame.quit()
print("Pygame window closed.")