import java.util.ArrayList;

class Triangle {
    public PVector v0, v1, v2;
    public PVector normal;
    public PVector col;

    public Triangle(PVector v0, PVector v1, PVector v2, PVector col) {
        //System.out.printf("Create triangle: (%f,%f, %f) (%f,%f, %f) (%f,%f, %f)\n", v0.x, v0.y, v0.z,v1.x, v1.y, v1.z,v2.x, v2.y, v2.z);
        this.v0 = new PVector(v0.x, v0.y, v0.z);
        this.v1 = new PVector(v1.x, v1.y, v1.z);
        this.v2 = new PVector(v2.x, v2.y, v2.z);
        this.col = new PVector(col.x, col.y, col.z);
        this.computeNormal(); //problem in this function
    }

    private void computeNormal() {
        PVector e1 = PVector.sub(this.v1,this.v0);
        PVector e2 = PVector.sub(this.v2,this.v0);
        this.normal = e2.cross(e1).normalize();
    }
}


class Pixel {
  int x;
  int y;
  float zinv;
  
      public Pixel(int x, int y, float zinv) {
        this.x = x;
        this.y = y;
        this.zinv = zinv;
    }
    
       // Convert to PVector for compatibility with existing code
    public PVector toPVector() {
        return new PVector(x, y, zinv);
    }
}


public static class Matrix3x3 {
    float[][] matrix;

    // Constructor
    public Matrix3x3(float[][] values) {
        if (values.length != 3 || values[0].length != 3) {
            throw new IllegalArgumentException("Matrix must be 3x3.");
        }
        this.matrix = values;
    }
    
    public Matrix3x3() {
      float[][] values = {
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
       };
       this.matrix = values;
    }

    public void setColumn(int col, float a, float b, float c) {
        if ((col > 2) || (col < 0)) return;
        this.matrix[col][0] = a;
        this.matrix[col][1] = b;
        this.matrix[col][2] = c;
    }

    // Get determinant of the matrix
    public float determinant() {
        return matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
             - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
             + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);
    }

    // Get the inverse of the matrix
    public Matrix3x3 inverse() {
        float det = determinant();
        if (abs(det) < 1e-6) { // Prevent division by zero
            throw new ArithmeticException("Matrix is singular and cannot be inverted.");
        }

        float[][] inv = new float[3][3];

        // Compute the adjugate matrix (cofactor matrix transposed)
        inv[0][0] = (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) / det;
        inv[0][1] = (matrix[0][2] * matrix[2][1] - matrix[0][1] * matrix[2][2]) / det;
        inv[0][2] = (matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1]) / det;

        inv[1][0] = (matrix[1][2] * matrix[2][0] - matrix[1][0] * matrix[2][2]) / det;
        inv[1][1] = (matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0]) / det;
        inv[1][2] = (matrix[0][2] * matrix[1][0] - matrix[0][0] * matrix[1][2]) / det;

        inv[2][0] = (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]) / det;
        inv[2][1] = (matrix[0][1] * matrix[2][0] - matrix[0][0] * matrix[2][1]) / det;
        inv[2][2] = (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]) / det;

        return new Matrix3x3(inv);
    }

    // Matrix multiplication
    public Matrix3x3 multiply(Matrix3x3 other) {
        float[][] result = new float[3][3];

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                result[i][j] = 0;
                for (int k = 0; k < 3; k++) {
                    result[i][j] += this.matrix[i][k] * other.matrix[k][j];
                }
            }
        }
        return new Matrix3x3(result);
    }

    // Multiply the matrix by a vector
    public PVector multiply(PVector vector) {
        float x = matrix[0][0] * vector.x + matrix[0][1] * vector.y + matrix[0][2] * vector.z;
        float y = matrix[1][0] * vector.x + matrix[1][1] * vector.y + matrix[1][2] * vector.z;
        float z = matrix[2][0] * vector.x + matrix[2][1] * vector.y + matrix[2][2] * vector.z;
        return new PVector(x, y, z);
    }

    // Print matrix for debugging
    public void print() {
      System.out.printf("\nMatrix:\n");
      System.out.printf("[ %.4f  %.4f  %.4f ]\n", matrix[0][0], matrix[1][0], matrix[2][0]);
      System.out.printf("[ %.4f  %.4f  %.4f ]\n", matrix[0][1], matrix[1][1], matrix[2][1]);
      System.out.printf("[ %.4f  %.4f  %.4f ]\n", matrix[0][2], matrix[1][2], matrix[2][2]);
    }

    // Get matrix data
    public float[][] getMatrix() {
        return matrix;
    }
}

public class CornellBox {
    public void loadTestModel(ArrayList<Triangle> triangles) {
        // Define colors
        PVector red = new PVector(0.75f, 0.15f, 0.15f);
        PVector yellow = new PVector(0.75f, 0.75f, 0.15f);
        PVector green = new PVector(0.15f, 0.75f, 0.15f);
        PVector cyan = new PVector(0.15f, 0.75f, 0.75f);
        PVector blue = new PVector(0.15f, 0.15f, 0.75f);
        PVector purple = new PVector(0.75f, 0.15f, 0.75f);
        PVector white = new PVector(0.75f, 0.75f, 0.75f);

        triangles.clear();
        triangles.ensureCapacity(5 * 2 * 3);

        // Room dimensions
        float L = 555.0f;

        // Define room corners
        PVector A = new PVector(L, 0.0f, 0.0f);
        PVector B = new PVector(0.0f, 0.0f, 0.0f);
        PVector C = new PVector(L, 0.0f, L);
        PVector D = new PVector(0.0f, 0.0f, L);
        PVector E = new PVector(L, L, 0.0f);
        PVector F = new PVector(0.0f, L, 0.0f);
        PVector G = new PVector(L, L, L);
        PVector H = new PVector(0.0f, L, L);

        //Floor
        triangles.add(new Triangle(C, B, A, green));      
        triangles.add(new Triangle(C, D, B, green));

        // Left wall
        triangles.add(new Triangle(A, E, C, purple));
        triangles.add(new Triangle(C, E, G, purple));

        // Right wall
        triangles.add(new Triangle(F, B, D, yellow));
        triangles.add(new Triangle(H, F, D, yellow));

        // Ceiling
        triangles.add(new Triangle(E, F, G, cyan));
        triangles.add(new Triangle(F, H, G, cyan));

        // Back wall
        triangles.add(new Triangle(G, D, C, white));
        triangles.add(new Triangle(G, H, D, white));

        // Short block
        A = new PVector(290, 0, 114);
        B = new PVector(130, 0, 65);
        C = new PVector(240, 0, 272);
        D = new PVector(82, 0, 225);
        E = new PVector(290, 165, 114);
        F = new PVector(130, 165, 65);
        G = new PVector(240, 165, 272);
        H = new PVector(82, 165, 225);

        // Front
        triangles.add(new Triangle(E, B, A, red));
        triangles.add(new Triangle(E, F, B, red));

        // Back
        triangles.add(new Triangle(F, D, B, red));
        triangles.add(new Triangle(F, H, D, red));

        // Back
        triangles.add(new Triangle(H, C, D, red));
        triangles.add(new Triangle(H, G, C, red));

        // Left
        triangles.add(new Triangle(G, E, C, red));
        triangles.add(new Triangle(E, A, C, red));

        // Top
        triangles.add(new Triangle(G, F, E, red));
        triangles.add(new Triangle(G, H, F, red));

        // Tall block
        A = new PVector(423, 0, 247);
        B = new PVector(265, 0, 296);
        C = new PVector(472, 0, 406);
        D = new PVector(314, 0, 456);
        E = new PVector(423, 330, 247);
        F = new PVector(265, 330, 296);
        G = new PVector(472, 330, 406);
        H = new PVector(314, 330, 456);

        // Front
        triangles.add(new Triangle(E, B, A, blue));
        triangles.add(new Triangle(E, F, B, blue));

        // Back
        triangles.add(new Triangle(F, D, B, blue));
        triangles.add(new Triangle(F, H, D, blue));

        // Back
        triangles.add(new Triangle(H, C, D, blue));
        triangles.add(new Triangle(H, G, C, blue));

        // Left
        triangles.add(new Triangle(G, E, C, blue));
        triangles.add(new Triangle(E, A, C, blue));

        // Top
        triangles.add(new Triangle(G, F, E, blue));
        triangles.add(new Triangle(G, H, F, blue));    
        
        PVector oneoneone = new PVector(1, 1, 1);
        // Scale to the volume [-1,1]^3
        for (int i = 0; i < triangles.size(); i++) {
            Triangle tri = triangles.get(i);
            tri.v0.mult(2/L);
            tri.v1.mult(2/L);    
            tri.v2.mult(2/L);
             
            tri.v0.sub(oneoneone);
            tri.v1.sub(oneoneone);
            tri.v2.sub(oneoneone);

            tri.v0.x *= -1;
            tri.v1.x *= -1;
            tri.v2.x *= -1;
            tri.v0.y *= -1;
            tri.v1.y *= -1;
            tri.v2.y *= -1;

            triangles.get(i).computeNormal();
            
            //System.out.printf("Tri %d [ (%f,%f,%f) (%f,%f,%f) (%f,%f,%f)]\n", i, triangles.get(i).v0.x, triangles.get(i).v0.y, triangles.get(i).v0.z, triangles.get(i).v1.x, triangles.get(i).v1.y, triangles.get(i).v1.z, triangles.get(i).v2.x, triangles.get(i).v2.y, triangles.get(i).v2.z);
        }
    }
}