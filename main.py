
import numpy as np

def echelon(A):
    """ Return Row Echelon Form of matrix A """

    # if matrix A has no columns or rows,
    # it is already in REF, so we return itself
    r, c = A.shape
    if r == 0 or c == 0:
        return A

    # we search for non-zero element in the first column
    for i in range(len(A)):
        if A[i,0] != 0:
            break
    else:
        # if all elements in the first column is zero,
        # we perform REF on matrix from second column
        B = row_echelon(A[:,1:])
        # and then add the first zero-column back
        return np.hstack([A[:,:1], B])

    # if non-zero element happens not in the first row,
    # we switch rows
    if i > 0:
        ith_row = A[i].copy()
        A[i] = A[0]
        A[0] = ith_row

    # we divide first row by first element in it
    A[0] = A[0] / A[0,0]
    # we subtract all subsequent rows with first row (it has 1 now as first element)
    # multiplied by the corresponding element in the first column
    A[1:] -= A[0] * A[1:,0:1]

    # we perform REF on matrix from second row, from second column
    B = echelon(A[1:,1:])

    # we add first row and first (zero) column, and return
    return np.vstack([A[:1], np.hstack([A[1:,:1], B]) ])
    
def find_solution(coordinates, answers):
    return np.linalg.solve(coordinates, answers)
    
def scalar_multiplication(v1, v2):
    return np.dot(v1, v2)
    
def multiplication_by_scalar(vector, scalar):
    return np.multiply(vector, scalar)
    
def vector_substraction(v1, v2):
    return np.subtract(v1, v2)
    
def Gram_Shmidt_alg(v_list):
    orto_proj_list = []
    for i in v_list:
        proj = np.array(i)
        for j in orto_proj_list:
            proj = vector_substraction(proj, multiplication_by_scalar(j, (scalar_multiplication(i, j))/(scalar_multiplication(j, j))))
        orto_proj_list.append(proj)
    return orto_proj_list

def find_basis_sum(basisV, basisU):
    return echelon(basisV + basisU)
    
def find_basis_intersection(basisV, basisU):
    subspace_V_ort = Gram_Shmidt_alg(np.array(basisV))
    subspace_U_ort = Gram_Shmidt_alg(np.array(basisU))
    echelonized_basis = echelon(np.array(subspace_U_ort + subspace_V_ort))
    echelonized_basis = echelonized_basis[~np.all(echelonized_basis == 0, axis=1)]
    return find_solution(echelonized_basis, [0] * len(echelonized_basis))

print(find_basis_intersection([[1, 2, 1], [1, 1, -1], [1, 3, 3]], [[2, 3, -1], [1, 2, 2], [1, 1, -3]]))
