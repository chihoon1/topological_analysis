'''
Compute Homology by using boundary map with mod2 scalar field
Homology is computed on Vietoris-Rips Complex
'''
import numpy as np

'''
Below Two functions under this step are auxiliary functions that help with computing the k-th homology
Boundary map is based on mod2 scalar field.
function named mod2_row_reduction was needed to compute dimension of image and kernel of a boundary map.
'''
def get_boundary_map_matrix(higher_dim_simplex, lower_dim_simplex):
    # param: higher_dim_simplex is a list of sets with k simplex
    # param: lower_dim_simplex is a list of sets with k-1 simplex
    # return a numpy 2d array representing matrix representation of bounday map dk
    boundary_map = np.zeros((len(lower_dim_simplex),len(higher_dim_simplex)))
    for j in range(len(higher_dim_simplex)):  # column of matrix
        for i in range(len(lower_dim_simplex)):  # row of matrix
            if lower_dim_simplex[i].issubset(higher_dim_simplex[j]):
                boundary_map[i, j] = 1
    return boundary_map


def mod2_row_reduction(mod2_matrix):
    # param: mod2_matrix(np.2d array) is a matrix with its entry 0 or 1
    # return row reduced mod2_matrix
    row_reduced_mat = mod2_matrix.copy()
    stable_rows = set()  # contains the row index of rows where no further row reduction required
    for j in range(mod2_matrix.shape[1]):  # j represents column index
        summand_row_idx = None
        for i in range(mod2_matrix.shape[0]):  # this loop is for finding a summand row for row reduction
            if row_reduced_mat[i,j] == 1 and i not in stable_rows:
                summand_row_idx = i
                stable_rows.add(summand_row_idx)
                break
        else:  # this else block will be executed if there is no leading one in j-th column. so skip
            continue
        for i in range(mod2_matrix.shape[0]):  # this loop is for row reduction operations
            if row_reduced_mat[i,j] == 1:
                if i != summand_row_idx:
                    # do mod2 summation between summand row and current row
                    row_reduced_mat[i] = np.absolute(row_reduced_mat[i] - row_reduced_mat[summand_row_idx])
    return row_reduced_mat


'''
First, get Vietoris Rips simplicial complex calling the functions from step 1.
Then, use the functions from step 4 to compute the dimension of k-th homology of VR(X[r]).
'''
def compute_Homology_k(vrc, k):
    # param: vrc(a set of tuples) is a Vietoris Rips Complex of X at r. VR(X[r])
    # # param: r(integer)>=0 and criteria for forming a simplex
    # param: k(integer) represents k in k-th homology: ker(dk)/ img(dk+1)
    k_plus_one_simplex = []
    k_simplex = []
    k_minus_one_simplex = []
    # get a list of k+1 simplices, k simplices, and k-1 simplices
    for simplex_tup in vrc:
        if len(simplex_tup) == k + 2:  # k+2 vertices, so k+1 simplex
            k_plus_one_simplex.append(set(simplex_tup))
        if len(simplex_tup) == k + 1:  # k+1 vertices, so k simplex
            k_simplex.append(set(simplex_tup))
        if len(simplex_tup) == k:  # k vertices, so k-1 simplex
            k_minus_one_simplex.append(set(simplex_tup))
    # get matrix representation of boundary maps dk and dk+1
    boundary_k_plus_one = get_boundary_map_matrix(k_plus_one_simplex, k_simplex)
    boundary_k = get_boundary_map_matrix(k_simplex, k_minus_one_simplex)
    # perform row reduction on matrix form of dk and dk+1
    boundary_k_plus_one = mod2_row_reduction(boundary_k_plus_one)
    boundary_k = mod2_row_reduction(boundary_k)
    # compute the dimension of ker and image of matrices
    if boundary_k_plus_one.shape[0] == 0 or boundary_k_plus_one.shape[1] == 0:
        dim_img_k_plus_one = 0
    else:
        dim_img_k_plus_one = np.linalg.matrix_rank(boundary_k_plus_one)
    if boundary_k.shape[0] == 0 or boundary_k.shape[1] == 0:
        dim_img_k = 0
    else:
        dim_img_k = np.linalg.matrix_rank(boundary_k)
    # dim(ker) is derived from rank-nullity theorem. dim Ck = dim ker dk + dim img dk
    dim_ker_k = len(k_simplex) - dim_img_k
    return dim_ker_k - dim_img_k_plus_one
