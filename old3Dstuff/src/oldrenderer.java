// GLOBAL VARIABLES
// Here, I put all globals variables into the 'Globals' class just to keep them tidy. You don't have to do this.
public static class Globals{
  static final float PI = 3.14159265358979323846f;
  static final float epsilon = 0.000001f;

  static int SCREEN_WIDTH = 500;
  static int SCREEN_HEIGHT = 500;
  static int t;    //timer

  static ArrayList<Triangle> triangles;
  static float focalLength = 500;
  static PVector cameraPosition = new PVector(0, 0, -3.001);
  //static PVector lightPos = 
  //static PVector lightColor = 
  //static PVector indirectLight =
  static PVector currentColor;
  static float[][] depthBuffer = new float[SCREEN_HEIGHT][SCREEN_WIDTH];

  // Movement
  static final float delta = 0.1;  //movement speed modifier for lights and camera
  static final float yaw = 0.05;
  static Matrix3x3 R = new Matrix3x3();
}

void VertexShader(PVector v, Pixel p) {
    PVector pCamera = PVector.sub(v, Globals.cameraPosition);
    
    float x = pCamera.x * Globals.R.matrix[0][0] + 
              pCamera.y * Globals.R.matrix[1][0] + 
              pCamera.z * Globals.R.matrix[2][0];
    float y = pCamera.x * Globals.R.matrix[0][1] + 
              pCamera.y * Globals.R.matrix[1][1] + 
              pCamera.z * Globals.R.matrix[2][1];
    float z = pCamera.x * Globals.R.matrix[0][2] + 
              pCamera.y * Globals.R.matrix[1][2] + 
              pCamera.z * Globals.R.matrix[2][2];
    
    p.zinv = 1.0f / z;
    
    // Change this to use the rotated z value consistently
    if (z > 0) {  // Changed from pCamera.z > 0 to z > 0
        p.x = (int)(Globals.focalLength * x / z) + (Globals.SCREEN_WIDTH/2);
        p.y = (int)(Globals.focalLength * y / z) + (Globals.SCREEN_HEIGHT/2);
    } else {
        p.x = p.y = -1;
    }
}

//void Interpolate(PVector a, PVector b, int N, ArrayList<PVector> r){
//  if (N == 1) {
//    PVector midpoint = PVector.lerp(a, b, 0.5);
//    r.add(midpoint);
//    return;
//  }
//  for (int i = 0; i < N; i++) {
//    float t = float(i) / (N-1);
//    PVector interpolated = PVector.lerp(a, b, t);
//    r.add(interpolated);
//  }
//}
void Interpolate(PVector a, PVector b, int numResults, ArrayList<PVector> r) {
  r.clear(); // Clear any existing values
  
  if (numResults <= 0) return;
  
  if (numResults == 1) {
    r.add(a.copy()); // When only one result is requested, return the start value
    return;
  }
  
  PVector step = PVector.sub(b, a).div(numResults - 1);
  
  
  for (int i = 0; i < numResults; i++) {
    PVector interpolated = PVector.add(a, PVector.mult(step, i));
    r.add(interpolated);
  }
}

void Interpolate(Pixel a, Pixel b, int numResults, ArrayList<Pixel> results) {
    results.clear();
    
    if (numResults <= 0) return;
    
    if (numResults == 1) {
        results.add(new Pixel(a.x, a.y, a.zinv));
        return;
    }
    
    // Calculate steps for each component
    float xStep = (float)(b.x - a.x) / (numResults - 1);
    float yStep = (float)(b.y - a.y) / (numResults - 1);
    float zStep = (b.zinv - a.zinv) / (numResults - 1);
    
    for (int i = 0; i < numResults; i++) {
        int x = (int)Math.round(a.x + xStep * i);
        int y = (int)Math.round(a.y + yStep * i);
        float zinv = a.zinv + zStep * i;
        results.add(new Pixel(x, y, zinv));
    }
}

void updateRotation(Matrix3x3 r, float yaw) {
    // Create temporary rotation matrix
    Matrix3x3 temp = new Matrix3x3();
    temp.setColumn(0, cos(yaw), 0, sin(yaw));
    temp.setColumn(1, 0, 1, 0);
    temp.setColumn(2, -sin(yaw), 0, cos(yaw));
    
    // Multiply with existing rotation to accumulate
    r.multiply(temp);
}

// Represents an intersection with a triangle - see lab instructions
public static class Intersection {
    public PVector position;
    public float distance;
    public int triangleIndex;

    public Intersection() {
        position = new PVector();
        distance = Float.MAX_VALUE;
        triangleIndex = -1;
    }
}

// Main function to find the closest intersection - see lab instructions
public static boolean ClosestIntersection(PVector start, PVector dir, ArrayList<Triangle> triangles, Intersection closestIntersection) {
    boolean intersected = false;

    //your code here...

    return intersected;
}

//see lab instructions
PVector DirectLight(Intersection i)
{
  //your code here
  PVector R = new PVector();
  
  return R;
}

//THIS IS THE MAIN ENTRY POINT TO THE PROGRAM
void setup() {
  size(500, 500);
  background(0);
  
  //example of how to initialise a timer (if you want to do that)
  Globals.t = millis();  
  
  //example of how to load a model in
  Globals.triangles = new ArrayList<>();
  CornellBox cb = new CornellBox();
  cb.loadTestModel(Globals.triangles);
  System.out.println("Loaded " + Globals.triangles.size() + " triangles.");
  
  // Initialize rotation matrix to identity
  Globals.R.setColumn(0, 1, 0, 0);
  Globals.R.setColumn(1, 0, 1, 0);
  Globals.R.setColumn(2, 0, 0, 1);
}

void draw() {
    println("Starting draw() - Triangle count: " + Globals.triangles.size());
    if (Globals.triangles.size() == 0) {
        println("ERROR: No triangles loaded!");
        return;
    }
    
    // Clear depth buffer (initialize to 0 representing infinite depth)
    for (int y = 0; y < Globals.SCREEN_HEIGHT; y++) {
        for (int x = 0; x < Globals.SCREEN_WIDTH; x++) {
            Globals.depthBuffer[y][x] = 0;
        }
    }
    
    background(0);
    
    for (Triangle triangle : Globals.triangles) {
        // Prepare vertex lists - one for filling, one for edges
        ArrayList<PVector> vertices = new ArrayList<PVector>();
        ArrayList<Pixel> projPos = new ArrayList<Pixel>();
        
        vertices.add(triangle.v0);
        vertices.add(triangle.v1);
        vertices.add(triangle.v2);
        
        // Project vertices once
        boolean allVisible = true;
        for (PVector vertex : vertices) {
            Pixel projected = new Pixel(0, 0, 0);
            VertexShader(vertex, projected);
            projPos.add(projected);
            if (projected.x < 0 || projected.y < 0) {
                allVisible = false;
            }
        }
        
        // Draw filled polygon if all vertices are visible
        if (allVisible) {
            Globals.currentColor = triangle.col;
            DrawPolygon(vertices);
        }
        
        // Draw wireframe edges if all vertices are visible
        if (projPos.get(0).x >= 0 && projPos.get(1).x >= 0 && projPos.get(2).x >= 0) {
            stroke(255); // White lines
            noFill();    // Don't fill for wireframe
            DrawPolygonEdges(projPos);
            
            // Optional: Draw vertices as dots
            for (Pixel p : projPos) {
                ellipse(p.x, p.y, 3, 3);
            }
        }
    }
}

void DrawLine(Pixel a, Pixel b, PVector col) {
    // Calculate number of interpolation points based on edge length
    PVector delta = PVector.sub(a.toPVector(), b.toPVector());
    
    delta.x = abs(delta.x);
    delta.y = abs(delta.y);
    int npixels = int(max(delta.x, delta.y)) + 1;
    //int steps = max(2, (int)(edgeLength * 0.5)); // Adjust multiplier for density
    
    //ArrayList<PVector> points = new ArrayList<PVector>();
    //Interpolate(a, b, steps, points);
    
    ArrayList<Pixel> line = new ArrayList<Pixel>();
    Interpolate(a, b, npixels, line);
    
    // Draw each interpolated point
    int colorRGB = color(col.x * 255, col.y * 255, col.z * 255);
    for (Pixel p : line) {
      set((int)p.x, (int)p.y, colorRGB); // Use set() as specified
  }
    
}

//void DrawPolygonEdges(ArrayList<PVector> vertices) {
//  int V = vertices.size();
//  ArrayList<PVector> projectedVertices = new ArrayList<PVector>();
  
//  if (V < 2) return;  // Need at least 2 points
//  PVector edgeColor = new PVector(1, 1, 1);
//  // Project vertices to 2D
//    for (int i = 0; i < vertices.size(); i++) {
//        int j = (i + 1) % vertices.size(); // Next vertex (wraps around)
//        DrawLine(vertices.get(i), vertices.get(j), edgeColor);
//    }
  
//  // Draw edges with color

//}
void DrawPolygonEdges(ArrayList<Pixel> vertices) {
  int V = vertices.size();
  ArrayList<PVector> projectedVertices = new ArrayList<PVector>();
  
  if (V < 2) return;  // Need at least 2 points
  PVector edgeColor = new PVector(1, 1, 1);
  // Project vertices to 2D
    for (int i = 0; i < vertices.size(); i++) {
        int j = (i + 1) % vertices.size(); // Next vertex (wraps around)
        DrawLine(vertices.get(i), vertices.get(j), edgeColor);
    }
  
  // Draw edges with color

}

//void ComputePolygonRows(ArrayList<PVector> vertexPixels, ArrayList<PVector>
//leftPixels, ArrayList<PVector> rightPixels)
//{
////1. Find max and min y−value of the polygon and compute the
////number of rows it occupies.
//    float minY = Float.MAX_VALUE;
//    float maxY = Float.MIN_VALUE;
//    for (PVector v : vertexPixels) {
//        if (v.y < minY) minY = v.y;
//        if (v.y > maxY) maxY = v.y;
//    }
    
//    // Calculate number of rows needed
//    int rows = (int)(maxY - minY) + 1;
////2. Resize leftPixels and rightPixels so that they have an
////element for each row.
//leftPixels.clear();
//    rightPixels.clear();
//    for (int i = 0; i < rows; i++) {
//        // Create new PVectors for each row
//        leftPixels.add(new PVector(Float.MAX_VALUE, minY + i));
//        rightPixels.add(new PVector(Float.MIN_VALUE, minY + i));
//    }
//        for (int i = 0; i < vertexPixels.size(); i++) {
//        int j = (i + 1) % vertexPixels.size(); // Next vertex (wraps around)
//        PVector start = vertexPixels.get(i);
//        PVector end = vertexPixels.get(j);
        
//        // Skip horizontal edges
//        if (start.y == end.y) continue;
        
//        // Determine which point is top and which is bottom
//        PVector top, bottom;
//        if (start.y < end.y) {
//            top = start;
//            bottom = end;
//        } else {
//            top = end;
//            bottom = start;
//        }
        
//        // Calculate number of steps needed for this edge
//        int steps = (int)(bottom.y - top.y) + 1;
        
//        // Interpolate along the edge
//        ArrayList<PVector> edgePixels = new ArrayList<PVector>();
//        Interpolate(top, bottom, steps, edgePixels);
        
//        // 4. Update left and right boundaries
//        for (PVector p : edgePixels) {
//            int row = (int)(p.y - minY);
//            if (row >= 0 && row < rows) {
//                // Update left boundary
//                if (p.x < leftPixels.get(row).x) {
//                    leftPixels.get(row).x = p.x;
//                }
//                // Update right boundary
//                if (p.x > rightPixels.get(row).x) {
//                    rightPixels.get(row).x = p.x;
//                }
//            }
//        }
//    }
////3. Initialize the x−coordinates in leftPixels to some really large
////value and the x−coordinates in rightPixels to some really small value.
////4. Loop through all edges of the polygon and use linear interpolation
////to find the x−coordinate for each row it occupies. Update the corresponding
////values in rightPixels and leftPixels.
//}

void ComputePolygonRows(ArrayList<Pixel> vertexPixels, ArrayList<Pixel> leftPixels, ArrayList<Pixel> rightPixels) {
    // 1. Find min and max y-values
    float minY = Float.MAX_VALUE;
    float maxY = Float.MIN_VALUE;
    for (Pixel p : vertexPixels) {
        if (p.y < minY) minY = p.y;
        if (p.y > maxY) maxY = p.y;
    }
    
    // 2. Calculate number of rows needed
    int rows = (int)(maxY - minY) + 1;
    
    // 3. Initialize left and right boundaries
    leftPixels.clear();
    rightPixels.clear();
    for (int i = 0; i < rows; i++) {
        // Initialize with extreme values
        leftPixels.add(new Pixel(Integer.MAX_VALUE, (int)minY + i, Float.MAX_VALUE));
        rightPixels.add(new Pixel(Integer.MIN_VALUE, (int)minY + i, -Float.MAX_VALUE));
    }
    
    // 4. Process each edge
    for (int i = 0; i < vertexPixels.size(); i++) {
        int j = (i + 1) % vertexPixels.size(); // Next vertex (wraps around)
        Pixel start = vertexPixels.get(i);
        Pixel end = vertexPixels.get(j);
        
        // Skip horizontal edges
        if (start.y == end.y) continue;
        
        // Determine top and bottom points
        Pixel top, bottom;
        if (start.y < end.y) {
            top = start;
            bottom = end;
        } else {
            top = end;
            bottom = start;
        }
        
        // Calculate steps needed
        int steps = (int)(bottom.y - top.y) + 1;
        
        // Interpolate along edge
        ArrayList<Pixel> edgePixels = new ArrayList<Pixel>();
        Interpolate(top, bottom, steps, edgePixels);
        
        // Update left and right boundaries
        for (Pixel p : edgePixels) {
            int row = (int)(p.y - minY);
            if (row >= 0 && row < rows) {
                // Update left boundary
                if (p.x < leftPixels.get(row).x || 
                   (p.x == leftPixels.get(row).x && p.zinv > leftPixels.get(row).zinv)) {
                    leftPixels.get(row).x = p.x;
                    leftPixels.get(row).zinv = p.zinv;
                }
                // Update right boundary
                if (p.x > rightPixels.get(row).x || 
                   (p.x == rightPixels.get(row).x && p.zinv > rightPixels.get(row).zinv)) {
                    rightPixels.get(row).x = p.x;
                    rightPixels.get(row).zinv = p.zinv;
                }
            }
        }
    }
}

void DrawRows(ArrayList<Pixel> leftPixels, ArrayList<Pixel> rightPixels) {
    // Get the current drawing color (converted from 0-1 range to 0-255)
    int fillColor = color(Globals.currentColor.x * 255, 
                         Globals.currentColor.y * 255, 
                         Globals.currentColor.z * 255);
    
    // Loop through each row (scanline)
    for (int row = 0; row < leftPixels.size(); row++) {
        Pixel left = leftPixels.get(row);
        Pixel right = rightPixels.get(row);
        
        // Skip if left boundary is right of right boundary (invalid row)
        if (left.x > right.x) continue;
        
        
        
        // Calculate how many pixels we need to draw in this row
        // int pixelsInRow = (int)(right.x - left.x) + 1;
        
        // Create a temporary list to store all pixels in this row
        //ArrayList<PVector> rowPixels = new ArrayList<PVector>();
        ArrayList<Pixel> rowPixels = new ArrayList<Pixel>();
        
        // Interpolate between left and right boundaries
        //Interpolate(left, right, pixelsInRow, rowPixels);
        Interpolate(left, right, (int)(right.x - left.x) + 1, rowPixels);
        
        // Draw each pixel in the row
        for (Pixel pixel : rowPixels) {
            // Round to integer coordinates and draw
            
            // Only draw if within screen bounds
            
              // set()
              if (pixel.x >= 0 && pixel.x < width && pixel.y >= 0 && pixel.y < height) {
                  if (pixel.zinv >= Globals.depthBuffer[pixel.y][pixel.x]) {// changing > to >= fxed the weird black lines
                    set(pixel.x, pixel.y, fillColor);
                    Globals.depthBuffer[pixel.y][pixel.x] = pixel.zinv;
                  }
              }
          }
        
    }
}

void DrawPolygon(ArrayList<PVector> vertices) {
    int V = vertices.size();
    ArrayList<Pixel> vertexPixels = new ArrayList<Pixel>();
    
    for (int i = 0; i < V; ++i) {
        //PVector projected = new PVector(); // Create new PVector
        Pixel projected = new Pixel(0, 0, 0);
        VertexShader(vertices.get(i), projected); // Project into it
        vertexPixels.add(projected); // Add to list
    }
    
    ArrayList<Pixel> leftPixels = new ArrayList<Pixel>();
    ArrayList<Pixel> rightPixels = new ArrayList<Pixel>();
    ComputePolygonRows(vertexPixels, leftPixels, rightPixels);
    DrawRows(leftPixels, rightPixels);
}


void update() {
  // Compute frame time:
  int t2 = millis();
  float dt = float(t2-Globals.t);
  Globals.t = t2;
  println("Render time: " + dt + " ms.");
}

void keyPressed() {
    // Camera movement (translation)
    if (keyCode == UP) {
        // Move forward along camera's view direction (negative Z in camera space)
        Globals.cameraPosition.z += Globals.delta;
    } else if (keyCode == DOWN) {
        // Move backward
        Globals.cameraPosition.z -= Globals.delta;
    } else if (keyCode == LEFT) {
        // Move left (negative X in world space)
        Globals.cameraPosition.x -= Globals.delta;
    } else if (keyCode == RIGHT) {
        // Move right (positive X)
        Globals.cameraPosition.x += Globals.delta;
    }
    
    // Camera rotation (yaw)
    if (key == 'a') {
        // Rotate left (positive yaw)
        updateRotation(Globals.R, Globals.yaw);
    } else if (key == 'd') {
        // Rotate right (negative yaw)
        updateRotation(Globals.R, -Globals.yaw);
    }
    
    // Camera elevation
    if (key == 'w') {
        // Move up (negative Y in world space)
        Globals.cameraPosition.y -= Globals.delta;
    } else if (key == 's') {
        // Move down (positive Y)
        Globals.cameraPosition.y += Globals.delta;
    }
}