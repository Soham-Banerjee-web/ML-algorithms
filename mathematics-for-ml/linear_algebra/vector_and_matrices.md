# 📘 Vectors and Matrices  

This document covers the **fundamentals of vectors and matrices**, which form the foundation of Machine Learning mathematics.  

---

## 🔹 1. Vectors  

### Definition  
A **vector** is an ordered list of numbers that represents:  
- A **point** in space  
- A **direction and magnitude**  
- Data in **n-dimensional space**  

\[
\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ v_3 \\ \vdots \\ v_n \end{bmatrix}
\]

Example (3D vector):  

\[
\mathbf{v} = \begin{bmatrix} 2 \\ -1 \\ 3 \end{bmatrix}
\]

---

### Vector Operations  

1. **Addition**  
\[
\mathbf{u} + \mathbf{v} = \begin{bmatrix} u_1 + v_1 \\ u_2 + v_2 \\ \cdots \\ u_n + v_n \end{bmatrix}
\]

2. **Scalar Multiplication**  
\[
c \cdot \mathbf{v} = \begin{bmatrix} c v_1 \\ c v_2 \\ \cdots \\ c v_n \end{bmatrix}
\]

3. **Dot Product (Inner Product)**  
\[
\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^n u_i v_i
\]

Geometric interpretation:  
\[
\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\|\|\mathbf{v}\|\cos\theta
\]

4. **Norm (Length of a Vector)**  
\[
\|\mathbf{v}\| = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}
\]

5. **Cross Product** (only in 3D)  
\[
\mathbf{u} \times \mathbf{v} = \begin{bmatrix} u_2v_3 - u_3v_2 \\ u_3v_1 - u_1v_3 \\ u_1v_2 - u_2v_1 \end{bmatrix}
\]

---

### Special Vectors  

- **Zero vector:**  
\[
\mathbf{0} = \begin{bmatrix} 0 \\ 0 \\ \cdots \\ 0 \end{bmatrix}
\]  

- **Unit vector:** \( \|\mathbf{v}\| = 1 \)  

- **Orthonormal vectors:** vectors that are orthogonal and of unit length  

---

## 🔹 2. Matrices  

### Definition  
A **matrix** is a rectangular array of numbers arranged in rows and columns.  

\[
A = \begin{bmatrix} 
a_{11} & a_{12} & \cdots & a_{1n} \\ 
a_{21} & a_{22} & \cdots & a_{2n} \\ 
\vdots & \vdots & \ddots & \vdots \\ 
a_{m1} & a_{m2} & \cdots & a_{mn} 
\end{bmatrix}
\]

- Dimension: \( m \times n \) (rows × columns)  
- **Square matrix:** \( m = n \)  

---

### Matrix Operations  

1. **Addition**  
\[
A + B = [a_{ij} + b_{ij}]
\]

2. **Scalar Multiplication**  
\[
cA = [c \cdot a_{ij}]
\]

3. **Transpose**  
\[
A^T = [a_{ji}]
\]

4. **Matrix Multiplication**  
If \( A \) is \( m \times n \) and \( B \) is \( n \times p \):  

\[
C = AB, \quad C_{ij} = \sum_{k=1}^n a_{ik} b_{kj}
\]

5. **Identity Matrix**  
\[
I_n = \begin{bmatrix} 
1 & 0 & \cdots & 0 \\ 
0 & 1 & \cdots & 0 \\ 
\vdots & \vdots & \ddots & \vdots \\ 
0 & 0 & \cdots & 1 
\end{bmatrix}
\]

6. **Inverse Matrix**  
\[
A^{-1}A = AA^{-1} = I
\]  
Exists only if \( A \) is **square and non-singular** (\(\det A \neq 0\)).  

7. **Determinant** (for \(2 \times 2\) matrix)  
\[
\det\begin{bmatrix} a & b \\ c & d \end{bmatrix} = ad - bc
\]

---

### Special Matrices  

- **Diagonal matrix:** all off-diagonal elements = 0  
- **Symmetric matrix:** \( A = A^T \)  
- **Orthogonal matrix:** \( A^T A = I \)  
- **Sparse matrix:** most entries = 0  

---

## 🔹 3. Relationship Between Vectors and Matrices  

- A **vector** is a special case of a **matrix** (\(n \times 1\) or \(1 \times n\))  
- A matrix can be seen as a **collection of vectors** (rows or columns)  
- Multiplying a matrix by a vector transforms the vector  

\[
A\mathbf{v} = \text{linear transformation of } \mathbf{v}
\]

---

## 🔹 4. Eigenvalues and Eigenvectors  

For a square matrix \( A \):  

\[
A\mathbf{v} = \lambda \mathbf{v}
\]

- \( \mathbf{v} \): eigenvector (direction unchanged)  
- \( \lambda \): eigenvalue (scaling factor)  

Characteristic equation:  

\[
\det(A - \lambda I) = 0
\]

---

## 🔹 5. Applications in Machine Learning  

- **Linear Regression**:  
\[
\mathbf{y} = X\mathbf{w} + \mathbf{\epsilon}
\]  

- **Principal Component Analysis (PCA)**: uses eigenvectors & eigenvalues of covariance matrices  

- **Neural Networks**: forward pass = repeated **matrix-vector multiplications**  

- **Optimization**: gradient descent uses vector/matrix calculus  

---

## 🔹 6. Python Examples  

```python
import numpy as np

# Vectors
v = np.array([2, -1, 3])
u = np.array([1, 4, -2])

# Dot product
dot = np.dot(u, v)

# Norm
norm_v = np.linalg.norm(v)

# Matrix
A = np.array([[2, 1],
              [1, 2]])

# Determinant
det_A = np.linalg.det(A)

# Eigenvalues & Eigenvectors
eigvals, eigvecs = np.linalg.eig(A)

print("Dot Product:", dot)
print("Norm of v:", norm_v)
print("Determinant of A:", det_A)
print("Eigenvalues:", eigvals)
print("Eigenvectors:\n", eigvecs)
