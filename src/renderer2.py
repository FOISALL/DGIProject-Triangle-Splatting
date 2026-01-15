import pygame
import numpy as np
import math
import os
import sysvenv
from dataclasses import dataclass

# --- Constants and Data Structures ---
@dataclass
class Point3D:
    """Represents a 3D point from the points3D.txt file."""
    id: int
    x: float
    y: float
    z: float
    r: int
    g: int
    b: int
    error: float

class Pixel:
    def __init__(self, x: int, y: int, zinv: float):
        self.x = x
        self.y = y
        self.zinv = zinv
    
    def __str__(self):
        return f"Pixel(x={self.x}, y={self.y}, zinv={self.zinv})"

class Globals:
    SCREEN_WIDTH = 1200  # Increased for better viewing
    SCREEN_HEIGHT = 800
    focalLength = 500.0  # Increased focal length
    point_size = 2  # Size of rendered points
    
    # Camera settings
    cameraPosition = np.array([0.0, 0.0, 5.0])  # Start in front of points
    R = np.eye(3)  # Rotation matrix
    movement_speed = 0.1
    rotation_speed = 0.02
    
    # Point cloud data
    point_cloud_data: list[Point3D] = []
    depthBuffer = np.full((SCREEN_HEIGHT, SCREEN_WIDTH), float('inf'))
    
    # Visualization controls
    scale_factor = 0.2  # Scale down large coordinates
    show_debug = True

# --- Core Functions ---
def load_points3D(filepath: str) -> list[Point3D]:
    """Loads 3D points from a COLMAP points3D.txt file."""
    points = []
    print(f"Loading points from: {filepath}")
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split()
                if len(parts) < 8:
                    continue

                try:
                    point_id = int(parts[0])
                    x = float(parts[1]) * Globals.scale_factor
                    y = float(parts[2]) * Globals.scale_factor
                    z = float(parts[3]) * Globals.scale_factor
                    r = int(parts[4])
                    g = int(parts[5])
                    b = int(parts[6])
                    error = float(parts[7])
                    
                    points.append(Point3D(point_id, x, y, z, r, g, b, error))
                except ValueError:
                    continue
    except Exception as e:
        print(f"Error loading points: {e}")
        return []
    
    print(f"Loaded {len(points)} points")
    return points

def vertex_shader(v: np.ndarray, p: Pixel):
    """Projects 3D point to 2D screen coordinates."""
    # Transform to camera space
    p_camera = v - Globals.cameraPosition
    x, y, z = p_camera @ Globals.R
    
    p.zinv = 1.0 / z if z > 0.01 else 0.0  # Avoid division by zero
    
    if z > 0.01:  # Only render points in front of camera
        p.x = int(Globals.focalLength * x / z) + (Globals.SCREEN_WIDTH // 2)
        p.y = int(Globals.focalLength * y / z) + (Globals.SCREEN_HEIGHT // 2)
    else:
        p.x = p.y = -1  # Mark as off-screen

def update_rotation(yaw: float, pitch: float = 0):
    """Update camera rotation with yaw and pitch."""
    if yaw != 0:
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        yaw_mat = np.array([
            [cos_y, 0, sin_y],
            [0, 1, 0],
            [-sin_y, 0, cos_y]
        ])
        Globals.R = Globals.R @ yaw_mat
    
    if pitch != 0:
        cos_p = math.cos(pitch)
        sin_p = math.sin(pitch)
        pitch_mat = np.array([
            [1, 0, 0],
            [0, cos_p, -sin_p],
            [0, sin_p, cos_p]
        ])
        Globals.R = Globals.R @ pitch_mat

def render_point_cloud(surface: pygame.Surface):
    """Render the point cloud with depth testing."""
    Globals.depthBuffer.fill(float('inf'))  # Reset depth buffer
    
    for point in Globals.point_cloud_data:
        p_screen = Pixel(0, 0, 0.0)
        v_3d = np.array([point.x, point.y, point.z])
        
        vertex_shader(v_3d, p_screen)
        
        # Check if point is visible
        if (p_screen.x != -1 and 
            0 <= p_screen.x < Globals.SCREEN_WIDTH and 
            0 <= p_screen.y < Globals.SCREEN_HEIGHT):
            
            # Depth test (use < for proper depth comparison)
            if p_screen.zinv < Globals.depthBuffer[p_screen.y, p_screen.x]:
                Globals.depthBuffer[p_screen.y, p_screen.x] = p_screen.zinv
                pygame.draw.circle(
                    surface, 
                    (point.r, point.g, point.b), 
                    (p_screen.x, p_screen.y), 
                    Globals.point_size
                )

def draw_debug_info(surface: pygame.Surface):
    """Display camera position and controls."""
    if not Globals.show_debug:
        return
    
    font = pygame.font.SysFont('Arial', 20)
    lines = [
        f"Camera Position: {Globals.cameraPosition}",
        f"Controls: WASD - Move, QE - Up/Down, Arrows - Rotate",
        f"Points: {len(Globals.point_cloud_data)}",
        f"Scale: {Globals.scale_factor}",
        f"F1 - Toggle debug info"
    ]
    
    for i, line in enumerate(lines):
        text = font.render(line, True, (255, 255, 255))
        surface.blit(text, (10, 10 + i * 25))

# --- Main Program ---
def main():
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((Globals.SCREEN_WIDTH, Globals.SCREEN_HEIGHT))
    pygame.display.set_caption("Point Cloud Viewer")
    clock = pygame.time.Clock()
    
    # Load point cloud data
    data_dir = "south-building/sparse"
    points_file = os.path.join(data_dir, "points3D.txt")
    Globals.point_cloud_data = load_points3D(points_file)
    
    if not Globals.point_cloud_data:
        print("No points loaded. Exiting.")
        return
    
    # Main loop
    running = True
    while running:
        # Handle events
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
        
        if keys[pygame.K_w]: Globals.cameraPosition += forward * Globals.movement_speed
        if keys[pygame.K_s]: Globals.cameraPosition -= forward * Globals.movement_speed
        if keys[pygame.K_a]: Globals.cameraPosition -= right * Globals.movement_speed
        if keys[pygame.K_d]: Globals.cameraPosition += right * Globals.movement_speed
        if keys[pygame.K_q]: Globals.cameraPosition += up * Globals.movement_speed
        if keys[pygame.K_e]: Globals.cameraPosition -= up * Globals.movement_speed
        
        # Rotation
        yaw, pitch = 0, 0
        if keys[pygame.K_LEFT]: yaw = Globals.rotation_speed
        if keys[pygame.K_RIGHT]: yaw = -Globals.rotation_speed
        if keys[pygame.K_UP]: pitch = Globals.rotation_speed
        if keys[pygame.K_DOWN]: pitch = -Globals.rotation_speed
        update_rotation(yaw, pitch)
        
        # Rendering
        screen.fill((0, 0, 0))
        render_point_cloud(screen)
        draw_debug_info(screen)
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()

if __name__ == "__main__":
    main()