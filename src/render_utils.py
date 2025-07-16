import numpy as np
from dataclasses import dataclass
import math

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


