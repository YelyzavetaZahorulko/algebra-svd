import numpy as np


def svd(matrix):
    a_t_a = np.dot(matrix.T, matrix)  # A^T * A
    a_a_t = np.dot(matrix, matrix.T)  # A * A^T

    eigenvalues_v, v = np.linalg.eig(a_t_a)
    eigenvalues_u, u = np.linalg.eigh(a_a_t)

    sorted_v = np.argsort(eigenvalues_v)[::-1]
    sorted_u = np.argsort(eigenvalues_u)[::-1]

    eigenvalues_v = eigenvalues_v[sorted_v]
    eigenvalues_u = eigenvalues_u[sorted_u]

    v = v[:, sorted_v]
    u = u[:, sorted_u]

    sigma_values = np.sqrt(np.maximum(eigenvalues_v, 0))
    sigma = np.zeros((matrix.shape[0], matrix.shape[1]))
    min_dim = min(matrix.shape)
    sigma[:min_dim, :min_dim] = np.diag(sigma_values[:min_dim])

    u = u[:, :matrix.shape[0]]
    v = v[:, :matrix.shape[1]]

    for i in range(min_dim):
        if sigma_values[i] != 0:
            u[:, i] = np.dot(matrix, v[:, i]) / sigma_values[i]
        else:
            u[:, i] = np.zeros(matrix.shape[0])

    for i in range(min_dim):
        if np.sign(u[0, i]) != np.sign(v[0, i]):
            u[:, i] = -u[:, i]
            v[:, i] = -v[:, i]

    matrix_reconstructed = np.dot(u, np.dot(sigma, v.T))

    return u, sigma, v.T, matrix_reconstructed


A = np.array([[3, 4], [4, 7], [5, 7]])
U, sigma, Vt, matrix_reconstructed = svd(A)

print("Original matrix A:\n", A)
print("Reconstructed matrix A_reconstructed:\n", matrix_reconstructed)
print("U:\n", U)
print("Sigma:\n", sigma)
print("V^T:\n", Vt)

B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
U, sigma, Vt, matrix_reconstructed = svd(B)

print("\nOriginal matrix B:\n", B)
print("Reconstructed matrix B_reconstructed:\n", matrix_reconstructed)

C = np.array([[1, 2, 3], [4, 5, 6]])
U, sigma, Vt, matrix_reconstructed = svd(C)

print("\nOriginal matrix C:\n", C)
print("Reconstructed matrix C_reconstructed:\n", matrix_reconstructed)

D = np.array([[1, 2], [3, 4], [5, 6]])
U, sigma, Vt, matrix_reconstructed = svd(D)

print("\nOriginal matrix D:\n", D)
print("Reconstructed matrix D_reconstructed:\n", matrix_reconstructed)
