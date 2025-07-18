import numpy as np
from dataclasses import dataclass
import math
import os

class Pixel:
    def __init__(self, x: int, y: int, zinv: float):
        self.x = x
        self.y = y
        self.zinv = zinv
    
    # Convert to a tuple for compatibility with existing code
    # (similar to toPVector() in Java)
    def to_vector(self):
        return (self.x, self.y, self.zinv)
    
    # Alternatively, if you need numpy array specifically
    def toNumpy(self):
        
        return np.array([self.x, self.y, self.zinv])
    
    # For better string representation when printing
    def toStrong(self):
        return f"Pixel(x={self.x}, y={self.y}, zinv={self.zinv})"
    
class Intersection:
    def __init__(self):
        self.position = np.zeros(3)  # Equivalent to new PVector()
        self.distance = math.inf    # Equivalent to Float.MAX_VALUE
        self.triangle_index = -1    # Python convention for index fields
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
    # add TRACK[] later

class Triangle:
    # I wanted to use a regular class instead of data class since mutability might become a problem
    # AI informed me that using this slots command would save memory 
    # I have not used it before but upon reading on the internet it seems true
    __slots__ = ['vertices', 'color', 'opacity', 'sigma'] 

    def __init__(self, vertices, color, opacity = 0.8, sigma = 1.0):
        self.vertices = np.array(vertices, dtype=np.float32)
        self.color = np.array(color, dtype=np.uint8)
        self.opacity = float(opacity)
        self.sigma = float(sigma)


def load_points3D(filepath: str) -> list[Point3D]:
    """
    Loads 3D points from a COLMAP points3D.txt file.
    
    Args:
        filepath: The path to the points3D.txt file.
        
    Returns:
        A list of Point3D objects.
    """
    points = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue # Skip empty lines and comments

                parts = line.split()
                try:
                    point_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    z = float(parts[3])
                    r = int(parts[4])
                    g = int(parts[5])
                    b = int(parts[6])
                    error = float(parts[7])
                    
                    # We are intentionally ignoring the TRACK[] data for now
                    
                    points.append(Point3D(point_id, x, y, z, r, g, b, error))
                except ValueError as e:
                    print(f"Skipping malformed line: {line} - Error: {e}")
                    continue
    except FileNotFoundError:
        print(f"Error: points3D file not found at {filepath}")
    return points

# Example usage (add this to your main script, outside the game loop initially)
# Assuming 'sparse' is a folder in the same directory as your script
# and points3D.txt is inside 'sparse/0' (or wherever your specific path leads)
# You provided a screenshot of 'sparse/0/points3D.txt'
COLMAP_DATA_DIR = "south-building/sparse" # Or adjust if 'sparse' is nested further
POINTS_FILE_PATH = os.path.join(COLMAP_DATA_DIR, "points3D.txt") # Adjust "0" if needed

all_points_3d: list[Point3D] = []
try:
    all_points_3d = load_points3D(POINTS_FILE_PATH)
    print(f"Loaded {len(all_points_3d)} 3D points.")
except Exception as e:
    print(f"Failed to load points: {e}")

